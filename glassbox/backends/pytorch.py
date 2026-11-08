"""PyTorch / HuggingFace engine adapter (spec §BACKENDS-pytorch).

Preserves single-GPU and native HuggingFace execution using :class:`DeepTap`
and :class:`SteeringController` with zero behavioral regressions.
"""

from __future__ import annotations

from typing import Any, ClassVar

from glassbox.backends.base import BaseEngineAdapter
from glassbox.steering import SteeringController
from glassbox.tap.hooks import DeepTap

_SERVING_ENGINE_NAMES = {
    "LLM",
    "LLMEngine",
    "AsyncLLMEngine",
    "Engine",
    "SRTServer",
    "ModelRunner",
    "AsyncEngine",
}


class PyTorchEngineAdapter(BaseEngineAdapter):
    """Adapter for PyTorch and HuggingFace models."""

    backend_name: ClassVar[str] = "pytorch"

    def __init__(self, tp_rank: int = 0, tp_size: int = 1, **kwargs: Any) -> None:
        super().__init__(tp_rank=tp_rank, tp_size=tp_size, **kwargs)
        self.engine: Any = None
        self.tap: DeepTap | None = None
        self.steering: SteeringController | None = None
        self.prompt_text: str = ""
        self.output_text: str = ""
        self._orig_generate: Any = None
        self._orig_forward: Any = None

    @classmethod
    def validate_engine(cls, engine: Any) -> bool:
        """Return True if engine is a PyTorch module or duck-types forward hooks."""
        if engine is None:
            return False
        cls_name = type(engine).__name__
        mod_name = getattr(type(engine), "__module__", "")
        if cls_name in _SERVING_ENGINE_NAMES:
            return False
        if mod_name.startswith(("vllm", "sglang")):
            return False
        if (
            hasattr(engine, "model_executor")
            or hasattr(engine, "tokenizer_manager")
            or hasattr(engine, "server_args")
        ):
            return False

        try:
            import torch.nn as nn

            if isinstance(engine, nn.Module):
                return True
        except ImportError:
            pass

        has_hook = hasattr(engine, "register_forward_hook") and callable(
            getattr(engine, "register_forward_hook", None)
        )
        has_forward = hasattr(engine, "forward") and callable(getattr(engine, "forward", None))
        return has_hook or has_forward

    def attach(self, engine: Any, watchlist: dict[str, Any], **kwargs: Any) -> DeepTap:
        """Attach DeepTap and steering hooks to the PyTorch engine."""
        self.engine = engine
        self.prompt_text = str(kwargs.get("prompt", self.prompt_text))

        tap = kwargs.get("tap")
        if tap is not None and isinstance(tap, DeepTap):
            self.tap = tap
        else:
            addresses = list(watchlist.get("addresses", []))
            window = kwargs.get("window", watchlist.get("merged_window", {}))
            trigger = kwargs.get("trigger", watchlist.get("effective_trigger", "always"))
            dtype = kwargs.get(
                "precision",
                kwargs.get("dtype", watchlist.get("effective_precision", "fp16")),
            )
            num_heads = kwargs.get("num_heads")
            head_reduction = kwargs.get("head_reduction", "l2_norm")
            capture_mode = kwargs.get("capture_mode", "all")
            batch_index = int(kwargs.get("batch_index", 0))

            self.tap = DeepTap(
                addresses,
                window=window,
                trigger=trigger,
                dtype=dtype,
                num_heads=num_heads,
                head_reduction=head_reduction,
                capture_mode=capture_mode,
                batch_index=batch_index,
            )

        self.tap.attach(engine)

        steer = kwargs.get("steer")
        ablate = kwargs.get("ablate")
        if steer is not None or ablate is not None:
            self.steering = SteeringController(engine, steer=steer, ablate=ablate)
            self.steering.attach()

        if hasattr(engine, "generate") and callable(getattr(engine, "generate", None)):
            orig_gen = engine.generate
            self._orig_generate = orig_gen

            def _wrapped_generate(*args: Any, **g_kwargs: Any) -> Any:
                if args and isinstance(args[0], str) and not self.prompt_text:
                    self.prompt_text = args[0]
                elif isinstance(g_kwargs.get("prompt"), str) and not self.prompt_text:
                    self.prompt_text = str(g_kwargs["prompt"])
                res = orig_gen(*args, **g_kwargs)
                if self.tap is not None and getattr(self.tap, "_staging_seq", None):
                    self.tap.flush_sequence()
                if isinstance(res, str):
                    self.output_text = res
                elif hasattr(res, "text") and isinstance(res.text, str):
                    self.output_text = res.text
                return res

            try:
                engine.generate = _wrapped_generate
            except Exception:
                pass

        if hasattr(engine, "forward") and callable(getattr(engine, "forward", None)):
            orig_fwd = engine.forward
            self._orig_forward = orig_fwd

            def _wrapped_forward(*args: Any, **f_kwargs: Any) -> Any:
                res = orig_fwd(*args, **f_kwargs)
                if self.tap is not None and getattr(self.tap, "_staging_seq", None):
                    self.tap.flush_sequence()
                return res

            try:
                engine.forward = _wrapped_forward
            except Exception:
                pass

        self._is_attached = True
        return self.tap

    def detach(self) -> None:
        """Detach all hooks and restore engine methods (idempotent)."""
        if self.steering is not None:
            try:
                self.steering.detach()
            except Exception:
                pass
            self.steering = None

        if self.tap is not None:
            try:
                self.tap.detach()
            except Exception:
                pass

        if self.engine is not None:
            if self._orig_generate is not None:
                try:
                    self.engine.generate = self._orig_generate
                except Exception:
                    pass
            if self._orig_forward is not None:
                try:
                    self.engine.forward = self._orig_forward
                except Exception:
                    pass

        self._orig_generate = None
        self._orig_forward = None
        self.engine = None
        self._is_attached = False

    def flush(self) -> None:
        """Flush pending staging tokens from DeepTap."""
        if self.tap is not None and getattr(self.tap, "_staging_seq", None):
            self.tap.flush_sequence()
