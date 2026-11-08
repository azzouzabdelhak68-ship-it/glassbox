"""SGLang engine adapter with RadixAttention support (spec §BACKENDS-sglang).

Instruments SGLang serving engines, extracting extend_start_loc and
req_pool_indices for continuous batching step manifests under RadixAttention.
"""

from __future__ import annotations

from typing import Any, ClassVar

from glassbox.backends.base import BaseEngineAdapter, StepManifest


class SGLangEngineAdapter(BaseEngineAdapter):
    """Engine adapter for SGLang runtime with RadixAttention continuous batching."""

    backend_name: ClassVar[str] = "sglang"

    def __init__(self, tp_rank: int = 0, tp_size: int = 1, **kwargs: Any) -> None:
        super().__init__(tp_rank=tp_rank, tp_size=tp_size, **kwargs)
        self.engine: Any = None
        self.watchlist: dict[str, Any] = {}
        self._handles: list[Any] = []
        self._orig_step: Any = None
        self._step_counter: int = 0
        self.captured_records: list[Any] = []

    @classmethod
    def validate_engine(cls, engine: Any) -> bool:
        """Return True if engine matches SGLang engine signatures."""
        if engine is None:
            return False
        cls_name = type(engine).__name__
        mod_name = getattr(type(engine), "__module__", "")

        if cls_name in ("Engine", "SRTServer", "ModelRunner", "ServerArgs", "AsyncEngine"):
            return True
        if mod_name.startswith("sglang"):
            return True
        if (
            hasattr(engine, "tokenizer_manager")
            or hasattr(engine, "server_args")
            or hasattr(engine, "model_runner")
        ):
            return True
        return False

    def extract_step_manifest(self, batch_or_meta: Any, step_id: int = 0) -> StepManifest:
        """Extract StepManifest from RadixAttention batch metadata."""
        extend_start_loc: list[int] = []
        req_pool_indices: list[int | str] = []
        logical_positions: list[int] | None = None
        actual_num_tokens: int = 0

        if isinstance(batch_or_meta, dict):
            ext_loc = batch_or_meta.get(
                "extend_start_loc", batch_or_meta.get("extend_seq_lens")
            )
            r_pool = batch_or_meta.get(
                "req_pool_indices", batch_or_meta.get("req_ids")
            )
            pos = batch_or_meta.get("logical_positions", batch_or_meta.get("positions"))
            num_tokens = batch_or_meta.get(
                "actual_num_tokens", batch_or_meta.get("num_tokens")
            )
        else:
            ext_loc = getattr(
                batch_or_meta,
                "extend_start_loc",
                getattr(batch_or_meta, "extend_seq_lens", None),
            )
            r_pool = getattr(
                batch_or_meta,
                "req_pool_indices",
                getattr(batch_or_meta, "req_ids", None),
            )
            pos = getattr(
                batch_or_meta,
                "logical_positions",
                getattr(batch_or_meta, "positions", None),
            )
            num_tokens = getattr(
                batch_or_meta,
                "actual_num_tokens",
                getattr(batch_or_meta, "num_tokens", None),
            )

        if ext_loc is not None:
            if hasattr(ext_loc, "tolist"):
                extend_start_loc = ext_loc.tolist()
            else:
                extend_start_loc = list(ext_loc)

        if r_pool is not None:
            if hasattr(r_pool, "tolist"):
                req_pool_indices = r_pool.tolist()
            else:
                req_pool_indices = list(r_pool)

        if pos is not None:
            if hasattr(pos, "tolist"):
                logical_positions = pos.tolist()
            else:
                logical_positions = list(pos)

        if num_tokens is not None:
            actual_num_tokens = int(num_tokens)
        elif extend_start_loc and len(extend_start_loc) > 1:
            actual_num_tokens = int(extend_start_loc[-1])
        elif logical_positions is not None:
            actual_num_tokens = len(logical_positions)
        elif req_pool_indices:
            actual_num_tokens = len(req_pool_indices)
            if not extend_start_loc:
                extend_start_loc = list(range(len(req_pool_indices) + 1))

        flags = {
            "engine": "sglang",
            "radix_attention": True,
            "req_pool_indices": req_pool_indices,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
        }

        manifest = StepManifest(
            step_id=step_id,
            req_ids=req_pool_indices,
            query_start_loc=extend_start_loc,
            logical_positions=logical_positions,
            actual_num_tokens=actual_num_tokens,
            flags=flags,
        )
        self.record_step(manifest)
        return manifest

    def attach(
        self, engine: Any, watchlist: dict[str, Any], **kwargs: Any
    ) -> SGLangEngineAdapter:
        """Attach instrumentation to SGLang engine."""
        self.engine = engine
        self.watchlist = dict(watchlist)

        # Hook model runner if accessible
        model_runner = getattr(engine, "model_runner", None)
        model = getattr(engine, "model", getattr(model_runner, "model", None))

        if model is not None and hasattr(model, "register_forward_hook"):
            def _generic_hook(module: Any, args: Any, output: Any) -> Any:
                self.captured_records.append(output)
                return output

            self._handles.append(model.register_forward_hook(_generic_hook))

        # Wrap step method if present to record manifests
        if hasattr(engine, "step") and callable(getattr(engine, "step", None)):
            orig_step = engine.step
            self._orig_step = orig_step

            def _wrapped_step(*s_args: Any, **s_kwargs: Any) -> Any:
                res = orig_step(*s_args, **s_kwargs)
                self._step_counter += 1
                batch_meta = s_args[0] if s_args else s_kwargs.get("batch", res)
                self.extract_step_manifest(batch_meta, step_id=self._step_counter)
                return res

            try:
                engine.step = _wrapped_step
            except Exception:
                pass

        self._is_attached = True
        return self

    def detach(self) -> None:
        """Detach all hooks and restore engine methods (idempotent)."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()

        if self.engine is not None and self._orig_step is not None:
            try:
                self.engine.step = self._orig_step
            except Exception:
                pass

        self._orig_step = None
        self.engine = None
        self._is_attached = False

    def flush(self) -> None:
        """Flush internal records."""
        self.captured_records.clear()
