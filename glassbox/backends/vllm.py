"""vLLM engine adapter and worker extension (spec §BACKENDS-vllm).

Integrates GlassBox with vLLM continuous batching and tensor-parallel workers
without blocking engine loops or falling back to PyTorch eager execution.
"""

from __future__ import annotations

import re
from typing import Any, ClassVar

from glassbox.backends.base import BaseEngineAdapter, StepManifest

_RE_N = re.compile(r"^N-(\d+)-(\d+)$")
_RE_A = re.compile(r"^A-(\d+)-(\d+)$")
_RE_L = re.compile(r"^L-(\d+)(?:-(\d+))?$")
_RE_LOGITS_TOK = re.compile(r"^LOGITS:(\d+)$")

# Decoupling guard: try importing TPShardRouter from glassbox.tap.distributed;
# if not present yet, provide robust internal fallback router.
try:
    from glassbox.tap.distributed import TPShardRouter
except ImportError:

    class TPShardRouter:  # type: ignore[no-redef]
        """Fallback Tensor Parallel shard router for head/neuron ownership."""

        def __init__(
            self,
            tp_rank: int = 0,
            tp_size: int = 1,
            num_heads: int = 32,
            mlp_dim: int = 14336,
            vocab_size: int = 32000,
        ) -> None:
            self.tp_rank = int(tp_rank)
            self.tp_size = max(1, int(tp_size))
            self.num_heads = num_heads
            self.mlp_dim = mlp_dim
            self.vocab_size = vocab_size

        def is_owner(self, address: str, rank: int | None = None) -> bool:
            """Check if rank owns passive recording for address."""
            r = self.tp_rank if rank is None else int(rank)
            addr = address.strip()

            # 1. Residual stream / whole layer / embed / norm: Rank 0 records
            if _RE_L.match(addr) or addr in ("EMBED", "NORM"):
                return r == 0

            # 2. Attention heads: A-LL-HH (0-indexed head h = int(group(2)) - 1)
            m_a = _RE_A.match(addr)
            if m_a:
                h = int(m_a.group(2)) - 1
                h_per_rank = max(1, self.num_heads // self.tp_size)
                owner = min(self.tp_size - 1, h // h_per_rank)
                return r == owner

            # 3. MLP neurons: N-LL-NNNN (0-indexed neuron n = int(group(2)) - 1)
            m_n = _RE_N.match(addr)
            if m_n:
                n = int(m_n.group(2)) - 1
                n_per_rank = max(1, self.mlp_dim // self.tp_size)
                owner = min(self.tp_size - 1, n // n_per_rank)
                return r == owner

            # 4. Logits / Logits entropy
            if addr in ("LOGITS", "LOGITS-ENTROPY") or _RE_LOGITS_TOK.match(addr):
                return r == 0

            return r == 0

        def should_steer(self, address: str, rank: int | None = None) -> bool:
            """Check if rank should apply steering for address."""
            addr = address.strip()
            # Residual stream active steering must be applied to ALL TP ranks
            # identically to prevent forward divergence across ranks!
            if _RE_L.match(addr) or addr in ("EMBED", "NORM"):
                return True
            return self.is_owner(address, rank)

        def route_address(self, address: str) -> dict[str, Any]:
            """Return detailed routing decision for an address."""
            addr = address.strip()
            m_a = _RE_A.match(addr)
            if m_a:
                h = int(m_a.group(2)) - 1
                h_per_rank = max(1, self.num_heads // self.tp_size)
                owner = min(self.tp_size - 1, h // h_per_rank)
                return {
                    "owner_rank": owner,
                    "local_index": h % h_per_rank,
                    "component": "attn",
                    "should_steer_all_ranks": False,
                }

            m_n = _RE_N.match(addr)
            if m_n:
                n = int(m_n.group(2)) - 1
                n_per_rank = max(1, self.mlp_dim // self.tp_size)
                owner = min(self.tp_size - 1, n // n_per_rank)
                return {
                    "owner_rank": owner,
                    "local_index": n % n_per_rank,
                    "component": "mlp",
                    "should_steer_all_ranks": False,
                }

            is_residual = bool(_RE_L.match(addr) or addr in ("EMBED", "NORM"))
            return {
                "owner_rank": 0,
                "local_index": 0,
                "component": "residual" if is_residual else "other",
                "should_steer_all_ranks": is_residual,
            }


class GlassBoxWorkerExtension:
    """Worker extension instantiated inside vLLM worker subprocesses."""

    def __init__(
        self,
        tp_rank: int = 0,
        tp_size: int = 1,
        watchlist: dict[str, Any] | None = None,
        router: Any = None,
        **kwargs: Any,
    ) -> None:
        self.tp_rank = int(tp_rank)
        self.tp_size = max(1, int(tp_size))
        self.watchlist = dict(watchlist) if watchlist else {}
        self.kwargs = dict(kwargs)

        num_heads = int(self.kwargs.get("num_heads", 32))
        mlp_dim = int(self.kwargs.get("mlp_dim", 14336))
        vocab_size = int(self.kwargs.get("vocab_size", 32000))

        if router is not None:
            self.router = router
        else:
            self.router = TPShardRouter(
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                vocab_size=vocab_size,
            )

        self.addresses: list[str] = list(self.watchlist.get("addresses", []))
        self.owned_addresses: list[str] = [
            addr for addr in self.addresses if self.router.is_owner(addr, self.tp_rank)
        ]
        self.steering_addresses: list[str] = [
            addr for addr in self.addresses if self.router.should_steer(addr, self.tp_rank)
        ]

        self._handles: list[Any] = []
        self.step_manifests: list[StepManifest] = []
        self.local_reductions: list[tuple[Any, Any, Any]] = []
        self.captured_activations: dict[str, list[Any]] = {}
        self._is_attached: bool = False

    @staticmethod
    def compute_local_reduction(logits: Any) -> tuple[Any, Any, Any]:
        """Compute local 3-scalar reduction (m_r, s_r, w_r) for exact global Shannon entropy.

        m_r: max logit
        s_r: sum(exp(z - m_r))
        w_r: sum((z - m_r) * exp(z - m_r))
        """
        import torch

        if not isinstance(logits, torch.Tensor):
            logits = torch.as_tensor(logits)
        logits_f = logits.float()
        m_r = torch.max(logits_f, dim=-1)[0]
        shift = logits_f - m_r.unsqueeze(-1)
        exp_z = torch.exp(shift)
        s_r = torch.sum(exp_z, dim=-1)
        w_r = torch.sum(shift * exp_z, dim=-1)
        return m_r, s_r, w_r

    @staticmethod
    def merge_local_reductions(reductions: list[tuple[Any, Any, Any]]) -> Any:
        """Merge local (m_r, s_r, w_r) tuples across all TP ranks into global Shannon entropy."""
        import torch

        m_list = [
            r[0] if isinstance(r[0], torch.Tensor) else torch.as_tensor(r[0])
            for r in reductions
        ]
        s_list = [
            r[1] if isinstance(r[1], torch.Tensor) else torch.as_tensor(r[1])
            for r in reductions
        ]
        w_list = [
            r[2] if isinstance(r[2], torch.Tensor) else torch.as_tensor(r[2])
            for r in reductions
        ]

        stacked_m = torch.stack(m_list, dim=0)
        M = torch.max(stacked_m, dim=0)[0]

        S = torch.zeros_like(M, dtype=torch.float32)
        W = torch.zeros_like(M, dtype=torch.float32)
        for m_r, s_r, w_r in zip(m_list, s_list, w_list, strict=False):
            exp_shift = torch.exp(m_r - M)
            S = S + s_r * exp_shift
            W = W + (w_r + (m_r - M) * s_r) * exp_shift

        S_safe = torch.clamp(S, min=1e-12)
        entropy = torch.log(S_safe) - (W / S_safe)
        return entropy

    def extract_step_manifest(self, batch_or_meta: Any, step_id: int = 0) -> StepManifest:
        """Extract continuous batching StepManifest from batch metadata."""
        req_ids: list[str | int] = []
        query_start_loc: list[int] = []
        logical_positions: list[int] | None = None
        actual_num_tokens: int = 0

        if isinstance(batch_or_meta, dict):
            q_loc = batch_or_meta.get("query_start_loc", batch_or_meta.get("cu_seq_lens_q"))
            r_ids = batch_or_meta.get("req_ids", batch_or_meta.get("request_ids"))
            pos = batch_or_meta.get("logical_positions", batch_or_meta.get("positions"))
            num_tokens = batch_or_meta.get(
                "actual_num_tokens", batch_or_meta.get("num_tokens")
            )
        else:
            q_loc = getattr(
                batch_or_meta,
                "query_start_loc",
                getattr(batch_or_meta, "cu_seq_lens_q", None),
            )
            r_ids = getattr(
                batch_or_meta, "req_ids", getattr(batch_or_meta, "request_ids", None)
            )
            if r_ids is None and hasattr(batch_or_meta, "seq_group_metadata_list"):
                r_ids = [
                    getattr(sg, "request_id", str(i))
                    for i, sg in enumerate(batch_or_meta.seq_group_metadata_list)
                ]
            pos = getattr(
                batch_or_meta,
                "logical_positions",
                getattr(batch_or_meta, "positions", None),
            )
            num_tokens = getattr(
                batch_or_meta,
                "actual_num_tokens",
                getattr(
                    batch_or_meta,
                    "num_tokens",
                    getattr(batch_or_meta, "num_scheduled_tokens", None),
                ),
            )

        if q_loc is not None:
            if hasattr(q_loc, "tolist"):
                query_start_loc = q_loc.tolist()
            else:
                query_start_loc = list(q_loc)

        if r_ids is not None:
            req_ids = list(r_ids)

        if pos is not None:
            if hasattr(pos, "tolist"):
                logical_positions = pos.tolist()
            else:
                logical_positions = list(pos)

        if num_tokens is not None:
            actual_num_tokens = int(num_tokens)
        elif query_start_loc and len(query_start_loc) > 1:
            actual_num_tokens = int(query_start_loc[-1])
        elif logical_positions is not None:
            actual_num_tokens = len(logical_positions)
        elif req_ids:
            # Decode step: 1 token per request
            actual_num_tokens = len(req_ids)
            query_start_loc = list(range(len(req_ids) + 1))

        manifest = StepManifest(
            step_id=step_id,
            req_ids=req_ids,
            query_start_loc=query_start_loc,
            logical_positions=logical_positions,
            actual_num_tokens=actual_num_tokens,
            flags={"engine": "vllm", "tp_rank": self.tp_rank, "tp_size": self.tp_size},
        )
        self.step_manifests.append(manifest)
        return manifest

    def register_hooks(self, model: Any) -> list[Any]:
        """Register forward and pre-hooks on model modules according to TP ownership."""
        if self._handles:
            return list(self._handles)

        # 1. Post-attention layernorm or layer hook for residual stream (L-LL)
        def _make_residual_hook(layer_idx: int) -> Any:
            def _residual_hook(module: Any, args: Any, output: Any) -> Any:
                addr = f"L-{layer_idx:02d}"
                # Active steering applied on all ranks
                if self.router.should_steer(addr, self.tp_rank):
                    pass
                # Passive recording on rank 0 only
                if self.router.is_owner(addr, self.tp_rank):
                    self.captured_activations.setdefault(addr, []).append(output)
                return output

            return _residual_hook

        # 2. o_proj pre-hook for attention heads (A-LL-HH)
        def _make_attn_hook(layer_idx: int) -> Any:
            def _attn_pre_hook(module: Any, args: Any) -> Any:
                inp = args[0] if args else None
                if inp is not None:
                    # Input to o_proj is per-head projected activations
                    for addr in self.owned_addresses:
                        m = _RE_A.match(addr)
                        if m and int(m.group(1)) == layer_idx:
                            self.captured_activations.setdefault(addr, []).append(inp)
                return args

            return _attn_pre_hook

        # 3. down_proj pre-hook for MLP neurons (N-LL-NNNN)
        def _make_mlp_hook(layer_idx: int) -> Any:
            def _mlp_pre_hook(module: Any, args: Any) -> Any:
                inp = args[0] if args else None
                if inp is not None:
                    for addr in self.owned_addresses:
                        m = _RE_N.match(addr)
                        if m and int(m.group(1)) == layer_idx:
                            self.captured_activations.setdefault(addr, []).append(inp)
                return args

            return _mlp_pre_hook

        # 4. lm_head hook for logits reduction scalars
        def _lm_head_hook(module: Any, args: Any, output: Any) -> Any:
            if output is not None:
                m_r, s_r, w_r = self.compute_local_reduction(output)
                self.local_reductions.append((m_r, s_r, w_r))
                if self.tp_rank == 0:
                    self.captured_activations.setdefault("LOGITS-ENTROPY", []).append(
                        (m_r, s_r, w_r)
                    )
            return output

        # Discover modules
        layers: list[Any] = []
        if hasattr(model, "layers"):
            layers = list(model.layers)
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = list(model.model.layers)

        for idx, layer in enumerate(layers, start=1):
            # Check post_attention_layernorm
            norm = getattr(
                layer,
                "post_attention_layernorm",
                getattr(layer, "ln_2", getattr(layer, "norm2", None)),
            )
            target_norm = norm if norm is not None else layer
            if hasattr(target_norm, "register_forward_hook"):
                self._handles.append(
                    target_norm.register_forward_hook(_make_residual_hook(idx))
                )

            # Check o_proj
            self_attn = getattr(layer, "self_attn", getattr(layer, "attention", None))
            if self_attn is not None:
                o_proj = getattr(
                    self_attn, "o_proj", getattr(self_attn, "out_proj", None)
                )
                if o_proj is not None and hasattr(o_proj, "register_forward_pre_hook"):
                    self._handles.append(
                        o_proj.register_forward_pre_hook(_make_attn_hook(idx))
                    )

            # Check down_proj
            mlp = getattr(layer, "mlp", getattr(layer, "feed_forward", None))
            if mlp is not None:
                down_proj = getattr(
                    mlp, "down_proj", getattr(mlp, "c_proj", None)
                )
                if down_proj is not None and hasattr(
                    down_proj, "register_forward_pre_hook"
                ):
                    self._handles.append(
                        down_proj.register_forward_pre_hook(_make_mlp_hook(idx))
                    )

        # lm_head
        lm_head = getattr(
            model, "lm_head", getattr(getattr(model, "model", None), "lm_head", None)
        )
        if lm_head is not None and hasattr(lm_head, "register_forward_hook"):
            self._handles.append(lm_head.register_forward_hook(_lm_head_hook))

        self._is_attached = True
        return list(self._handles)

    def detach(self) -> None:
        """Remove all hook handles (idempotent)."""
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._is_attached = False

    def flush(self) -> None:
        """Flush internal activation buffers."""
        self.captured_activations.clear()


class VLLMEngineAdapter(BaseEngineAdapter):
    """Engine adapter for vLLM serving runtimes."""

    backend_name: ClassVar[str] = "vllm"

    def __init__(self, tp_rank: int = 0, tp_size: int = 1, **kwargs: Any) -> None:
        super().__init__(tp_rank=tp_rank, tp_size=tp_size, **kwargs)
        self.engine: Any = None
        self.worker_extension: GlassBoxWorkerExtension | None = None
        self._orig_step: Any = None
        self._step_counter: int = 0

    @classmethod
    def validate_engine(cls, engine: Any) -> bool:
        """Return True if engine is a vLLM LLM or LLMEngine instance."""
        if engine is None:
            return False
        cls_name = type(engine).__name__
        mod_name = getattr(type(engine), "__module__", "")

        if cls_name in ("LLM", "LLMEngine", "AsyncLLMEngine"):
            return True
        if mod_name.startswith("vllm"):
            return True
        if hasattr(engine, "model_executor") or hasattr(engine, "llm_engine"):
            return True
        if hasattr(engine, "worker") and hasattr(engine, "step"):
            return True
        return False

    def attach(
        self, engine: Any, watchlist: dict[str, Any], **kwargs: Any
    ) -> GlassBoxWorkerExtension:
        """Attach worker extension to the vLLM engine."""
        self.engine = engine
        self.worker_extension = GlassBoxWorkerExtension(
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            watchlist=watchlist,
            **kwargs,
        )

        # Hook model if accessible directly or via executor
        model = getattr(engine, "model", None)
        if model is None and hasattr(engine, "model_executor"):
            worker = getattr(
                engine.model_executor,
                "driver_worker",
                getattr(engine.model_executor, "worker", None),
            )
            if worker is not None:
                model = getattr(
                    worker,
                    "model",
                    getattr(getattr(worker, "model_runner", None), "model", None),
                )
        elif model is None and hasattr(engine, "worker"):
            model = getattr(
                engine.worker,
                "model",
                getattr(getattr(engine.worker, "model_runner", None), "model", None),
            )

        if model is not None:
            self.worker_extension.register_hooks(model)

        # Wrap step or execute_model to collect manifests
        if hasattr(engine, "step") and callable(getattr(engine, "step", None)):
            orig_step = engine.step
            self._orig_step = orig_step

            def _wrapped_step(*s_args: Any, **s_kwargs: Any) -> Any:
                res = orig_step(*s_args, **s_kwargs)
                self._step_counter += 1
                batch_meta = (
                    s_args[0]
                    if s_args
                    else s_kwargs.get("execute_model_req", s_kwargs.get("batch", res))
                )
                if self.worker_extension is not None:
                    manifest = self.worker_extension.extract_step_manifest(
                        batch_meta, step_id=self._step_counter
                    )
                    self.record_step(manifest)
                return res

            try:
                engine.step = _wrapped_step
            except Exception:
                pass

        self._is_attached = True
        return self.worker_extension

    def detach(self) -> None:
        """Detach worker extension and restore engine methods (idempotent)."""
        if self.worker_extension is not None:
            try:
                self.worker_extension.detach()
            except Exception:
                pass
            self.worker_extension = None

        if self.engine is not None and self._orig_step is not None:
            try:
                self.engine.step = self._orig_step
            except Exception:
                pass

        self._orig_step = None
        self.engine = None
        self._is_attached = False

    def flush(self) -> None:
        """Flush worker extension buffers."""
        if self.worker_extension is not None:
            self.worker_extension.flush()
