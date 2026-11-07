"""Deep gather hooks — ONLY hook code lives here (spec §TAP-fsst Deep, plan §9.2).

Deep-Only Mechanistic Interpretability Engine:
- ``register_forward_hook`` / ``register_forward_pre_hook`` ONLY on parent
  modules named in the watchlist (never per neuron).
- Auto-resolves both canonical paths (``layer-01.mlp``, ``layer-01.attn``,
  ``layer-01``, ``logits``, ``embed``, ``norm``) and real HuggingFace / PyTorch
  transformer hierarchies (Llama, Mistral, Qwen, GPT-2, GPT-NeoX):
  - MLP neurons (``N-LL-NNNN``): hooks ``mlp.act_fn`` or pre-hooks
    ``mlp.down_proj`` / ``c_proj`` when present to capture true intermediate
    activations ``1..intermediate_size`` rather than post-projection ``hidden_size``.
  - Attention heads (``A-LL-HH``): reshapes ``[..., num_heads, head_dim]`` and
    computes true per-head L2 norm (or RMS/mean) across ``head_dim`` on device.
  - Residual stream (``L-LL``, ``L-LL-DDDD``, ``NORM``, ``EMBED``): computes L2
    norm across ``hidden_size`` for whole-layer probes or slices exact hidden
    dimension ``DDDD``.
  - Logits (``LOGITS``, ``LOGITS:<id>``, ``LOGITS-ENTROPY``): computes top-1
    logit, exact target-token logit, or softmax Shannon entropy on device.
- Handles 1D ``[D]``, 2D ``[S, D]``, and 3D ``[B, S, D]`` tensors natively,
  supporting both prefill ``[B, S, D]`` and autoregressive decode ``[B, 1, D]``.
- Streams captured rows directly to :class:`~glassbox.tap.sidecar.Sidecar` when
  attached.
- Ring full -> ``flags["ring_overflow"] = True``: drops newest payload, never
  stalls inference.
- Hook throw -> auto ``detach()`` (model restored) + ``flags["hook_detached"] = True``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from glassbox.controller import is_armed, parse_trigger
from glassbox.storage.layout import encode_deep_array

_RE_N = re.compile(r"^N-(\d+)-(\d+)$")
_RE_A = re.compile(r"^A-(\d+)-(\d+)$")
_RE_L = re.compile(r"^L-(\d+)$")
_RE_L_DIM = re.compile(r"^L-(\d+)-(\d+)$")
_RE_SAE = re.compile(r"^SAE-(\d+)-(\d+)$")
_RE_F = re.compile(r"^F-(\d+)-(\d+)$")
_RE_LOGITS_TOK = re.compile(r"^LOGITS:(\d+)$")

_NP_DTYPE: dict[str, Any] = {
    "fp32": np.dtype("<f4"),
    "fp16": np.dtype("<f2"),
    "bf16": np.dtype("<u2"),
}
_DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}


def parent_module_name(addr: str) -> str:
    """Map a watched address to its canonical parent module name."""
    addr = addr.strip()
    m = _RE_N.match(addr)
    if m:
        return f"layer-{int(m.group(1)):02d}.mlp"
    m = _RE_A.match(addr)
    if m:
        return f"layer-{int(m.group(1)):02d}.attn"
    m = _RE_L.match(addr)
    if m:
        return f"layer-{int(m.group(1)):02d}"
    m = _RE_L_DIM.match(addr)
    if m:
        return f"layer-{int(m.group(1)):02d}"
    m_sae = _RE_SAE.match(addr) or _RE_F.match(addr)
    if m_sae:
        return f"layer-{int(m_sae.group(1)):02d}.sae"
    if addr in ("LOGITS", "LOGITS-ENTROPY") or _RE_LOGITS_TOK.match(addr):
        return "logits"
    if addr in ("EMBED", "NORM"):
        return addr.lower()
    raise ValueError(f"E02: bad address {addr}")


def address_local_index(addr: str) -> int:
    """0-based index of ``addr`` inside its parent activation (or -1 for reductions)."""
    addr = addr.strip()
    m = _RE_N.match(addr)
    if m:
        return int(m.group(2)) - 1
    m = _RE_A.match(addr)
    if m:
        return int(m.group(2)) - 1
    m = _RE_L_DIM.match(addr)
    if m:
        return int(m.group(2)) - 1
    m_sae = _RE_SAE.match(addr) or _RE_F.match(addr)
    if m_sae:
        return int(m_sae.group(2)) - 1
    m = _RE_LOGITS_TOK.match(addr)
    if m:
        return int(m.group(1))
    if _RE_L.match(addr) or addr in ("LOGITS", "LOGITS-ENTROPY", "EMBED", "NORM"):
        return 0
    raise ValueError(f"E02: bad address {addr}")


def group_by_parent(addresses: list[str]) -> dict[str, list[tuple[str, int, int]]]:
    """Group watched addresses by parent: parent -> [(addr, local, col)]."""
    groups: dict[str, list[tuple[str, int, int]]] = {}
    for col, addr in enumerate(addresses):
        parent = parent_module_name(addr)
        groups.setdefault(parent, []).append((addr, address_local_index(addr), col))
    return groups


def _is_torch_tensor(obj: Any) -> bool:
    return obj.__class__.__module__.split(".")[0] == "torch" and hasattr(obj, "ndim")


def _as_index(activation: Any, local: list[int]) -> Any:
    """Build an index object for ``index_select`` (device-placed when real)."""
    if _is_torch_tensor(activation):
        import torch

        device = getattr(activation, "device", None)
        return torch.as_tensor(list(local), dtype=torch.long, device=device)
    return list(local)


def _to_2d_numpy(gathered: Any) -> np.ndarray:
    """Convert a gathered 1D or 2D buffer ``[S, K]`` into a float64 2D numpy array."""
    if _is_torch_tensor(gathered):
        detached = gathered.detach()
        if getattr(detached, "is_cuda", False):
            detached = detached.to(device="cpu", non_blocking=True)
        arr = np.asarray(detached.float().numpy(), dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])
    if isinstance(gathered, np.ndarray):
        arr = np.asarray(gathered, dtype=np.float64)
        return arr.reshape(1, -1) if arr.ndim == 1 else arr.reshape(-1, arr.shape[-1])
    if hasattr(gathered, "tolist"):
        raw = gathered.tolist()
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return np.asarray(raw, dtype=np.float64)
        return np.asarray([float(v) for v in raw], dtype=np.float64).reshape(1, -1)
    if isinstance(gathered, (list, tuple)):
        return np.asarray([float(v) for v in gathered], dtype=np.float64).reshape(1, -1)
    try:
        return np.asarray([float(v) for v in gathered], dtype=np.float64).reshape(1, -1)
    except TypeError as exc:
        raise ValueError(f"E02: cannot drain gathered buffer {type(gathered)}") from exc


def _to_floats(gathered: Any) -> list[float]:
    """Drain the last token row of a gathered buffer to Python floats."""
    mat = _to_2d_numpy(gathered)
    return [float(v) for v in mat[-1]]


class _HostSlot:
    """One pinned host staging slot (double-buffered per parent group)."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.mat: np.ndarray = np.zeros((1, width), dtype=np.float64)
        self.buf: list[float] = [0.0] * width

    def copy_(self, src: Any, non_blocking: bool = True) -> _HostSlot:
        """Stage ``src`` into this slot on the dedicated copyStream."""
        _ = non_blocking
        self.mat = _to_2d_numpy(src)
        self.buf = [float(v) for v in self.mat[-1]]
        return self

    def read(self) -> list[float]:
        """Read latest staged token row back."""
        return list(self.buf)

    def read_matrix(self) -> np.ndarray:
        """Read full ``[S, width]`` staged sequence matrix."""
        return self.mat


HookFn = Callable[..., None]


class DeepTap:
    """Portable Deep gather recorder (one hook per watched parent module)."""

    def __init__(
        self,
        addresses: list[str],
        window: dict[str, int] | None = None,
        trigger: str | dict[str, Any] | None = None,
        dtype: str = "fp16",
        capacity_tokens: int | None = None,
        scale: float = 1.0,
        stream: Any = None,
        num_heads: int | None = None,
        head_reduction: str = "l2_norm",
        capture_mode: str = "all",
        sidecar: Any = None,
        batch_index: int = 0,
    ) -> None:
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"E02: bad precision {dtype!r}")
        if not addresses:
            raise ValueError("E02: DeepTap requires at least 1 watched address")
        self.watched: list[str] = list(addresses)
        self.window: dict[str, int] = dict(
            window if window is not None else {"start_token": 0, "end_token": 1 << 30, "every_n": 1}
        )
        if isinstance(trigger, dict) and "kind" in trigger:
            self.trigger: dict[str, Any] = dict(trigger)
        else:
            self.trigger = parse_trigger(trigger)
        self.dtype: str = dtype
        self.capacity_tokens: int | None = capacity_tokens
        self.scale: float = float(scale)
        self.stream: Any = stream
        self.num_heads: int | None = num_heads
        self.head_reduction: str = head_reduction
        self.capture_mode: str = capture_mode
        self.sidecar: Any = sidecar
        self.batch_index: int = int(batch_index)
        self.groups: dict[str, list[tuple[str, int, int]]] = group_by_parent(self.watched)
        self._group_col_np: dict[str, np.ndarray] = {
            parent: np.asarray([col for _, _, col in cols], dtype=np.intp)
            for parent, cols in self.groups.items()
        }
        self.pinned: dict[str, list[_HostSlot]] = {
            parent: [_HostSlot(len(cols)), _HostSlot(len(cols))]
            for parent, cols in self.groups.items()
        }
        self._flip: dict[str, int] = {parent: 0 for parent in self.groups}
        self.events: dict[str, Any] = {parent: None for parent in self.groups}
        self._staging: dict[int, float] = {}
        self._staging_seq: dict[str, Any] = {}
        self.ring: list[list[float]] = []
        self.token_positions: list[int] = []
        self._raw_logits_summary: list[tuple[Any, Any, Any]] = []
        self._cached_logits_summary: list[dict[str, Any]] = []
        self._auto_token_cursor: int = 0
        self._prefill_done: bool = False
        self.flags: dict[str, Any] = {
            "ring_overflow": False,
            "hook_detached": False,
            "triggered": False,
        }
        self.dtype_meta: dict[str, Any] = {
            "stored": dtype,
            "source": "ieee_bfloat16_le" if dtype == "bf16" else "logical_values",
        }
        self._handles: list[Any] = []
        self._parent_heads: dict[str, int] = {}
        self._detached: bool = False

    @property
    def detached(self) -> bool:
        """True once :meth:`detach` has removed every hook handle."""
        return self._detached

    @property
    def logits_summary(self) -> list[dict[str, Any]]:
        """Per-token top-k logits and entropy summary.

        Materialized on CPU lazily upon request so device-to-host synchronization
        never stalls the model forward-pass hot path.
        """
        if self._raw_logits_summary:
            for top_vals, top_ids, ent in self._raw_logits_summary:
                tv = top_vals.to(device="cpu") if hasattr(top_vals, "to") else top_vals
                ti = top_ids.to(device="cpu") if hasattr(top_ids, "to") else top_ids
                ev = ent.to(device="cpu") if hasattr(ent, "to") else ent
                self._cached_logits_summary.append(
                    {
                        "top_token_ids": [int(i) for i in ti.tolist()],
                        "top_logits": [round(float(v), 4) for v in tv.tolist()],
                        "entropy": round(float(ev.item()), 4),
                    }
                )
            self._raw_logits_summary.clear()
        return self._cached_logits_summary

    @logits_summary.setter
    def logits_summary(self, val: list[dict[str, Any]]) -> None:
        self._cached_logits_summary = list(val)
        self._raw_logits_summary.clear()

    def _try_get(self, model: Any, path: str) -> Any | None:
        """Try resolving ``path`` via ``get_submodule`` or attribute walk."""
        getter = getattr(model, "get_submodule", None)
        if callable(getter):
            try:
                return getter(path)
            except Exception:
                pass
        node: Any = model
        try:
            for part in path.split("."):
                if part.isdigit() and hasattr(node, "__getitem__"):
                    node = node[int(part)]
                else:
                    node = getattr(node, part)
            return node
        except Exception:
            return None

    def _resolve_target(self, model: Any, parent: str) -> tuple[Any, bool]:
        """Resolve ``(module, is_pre_hook)`` across canonical and HF/PyTorch architectures."""
        # 1. Exact canonical path first (e.g. "layer-01.mlp", "logits")
        direct = self._try_get(model, parent)
        if direct is not None:
            # If MLP has down_proj/c_proj, hook pre-down_proj for true intermediate neurons
            if parent.endswith(".mlp"):
                for proj_name in ("down_proj", "c_proj", "fc2", "w2"):
                    proj = getattr(direct, proj_name, None)
                    if proj is not None and hasattr(proj, "register_forward_pre_hook"):
                        return (proj, True)
                for act_name in ("act_fn", "act", "activation"):
                    act = getattr(direct, act_name, None)
                    if act is not None and hasattr(act, "register_forward_hook"):
                        return (act, False)
            return (direct, False)

        # 2. HuggingFace / PyTorch transformer layer resolution (0-based index)
        m_layer = re.match(r"^layer-(\d+)(?:\.(mlp|attn|sae))?$", parent)
        if m_layer:
            idx = int(m_layer.group(1)) - 1
            sub = m_layer.group(2)
            layer_prefixes = (
                f"model.layers.{idx}",
                f"model.language_model.layers.{idx}",
                f"language_model.model.layers.{idx}",
                f"language_model.layers.{idx}",
                f"layers.{idx}",
                f"transformer.h.{idx}",
                f"gpt_neox.layers.{idx}",
                f"decoder.layers.{idx}",
                f"encoder.layers.{idx}",
                f"blocks.{idx}",
            )
            for prefix in layer_prefixes:
                block = self._try_get(model, prefix)
                if block is None:
                    continue
                if sub is None:
                    return (block, False)
                if sub == "sae":
                    for sae_name in ("sae", "autoencoder", "sparse_autoencoder"):
                        sae_mod = getattr(block, sae_name, None)
                        if sae_mod is not None:
                            return (sae_mod, False)
                    return (block, False)
                if sub == "mlp":
                    for mlp_name in ("mlp", "feed_forward", "ffn"):
                        mlp_mod = getattr(block, mlp_name, None)
                        if mlp_mod is not None:
                            for proj_name in ("down_proj", "c_proj", "fc2", "w2"):
                                proj = getattr(mlp_mod, proj_name, None)
                                if proj is not None and hasattr(proj, "register_forward_pre_hook"):
                                    return (proj, True)
                            for act_name in ("act_fn", "act", "activation"):
                                act = getattr(mlp_mod, act_name, None)
                                if act is not None and hasattr(act, "register_forward_hook"):
                                    return (act, False)
                            return (mlp_mod, False)
                if sub == "attn":
                    for attn_name in ("self_attn", "linear_attn", "attn", "attention"):
                        attn_mod = getattr(block, attn_name, None)
                        if attn_mod is not None:
                            for h_attr in (
                                "num_heads",
                                "num_v_heads",
                                "num_attention_heads",
                                "n_head",
                            ):
                                mh = getattr(attn_mod, h_attr, None)
                                if isinstance(mh, int) and mh > 0:
                                    self._parent_heads[parent] = mh
                                    break
                            for o_name in ("o_proj", "c_proj", "out_proj", "dense"):
                                o_proj = getattr(attn_mod, o_name, None)
                                if o_proj is not None and hasattr(
                                    o_proj, "register_forward_pre_hook"
                                ):
                                    return (o_proj, True)
                            return (attn_mod, False)

        # 3. Special modules: logits / norm / embed
        if parent == "logits":
            for cand in (
                "lm_head",
                "language_model.lm_head",
                "model.lm_head",
                "embed_out",
                "head",
                "output",
                "classifier",
            ):
                mod = self._try_get(model, cand)
                if mod is not None:
                    return (mod, False)
            if hasattr(model, "register_forward_hook"):
                return (model, False)
        if parent == "norm":
            for cand in (
                "model.norm",
                "model.language_model.norm",
                "language_model.model.norm",
                "language_model.norm",
                "transformer.ln_f",
                "gpt_neox.final_layer_norm",
                "ln_f",
                "norm",
                "final_layernorm",
            ):
                mod = self._try_get(model, cand)
                if mod is not None:
                    return (mod, False)
        if parent == "embed":
            for cand in (
                "model.embed_tokens",
                "model.language_model.embed_tokens",
                "language_model.model.embed_tokens",
                "language_model.embed_tokens",
                "embed_tokens",
                "transformer.wte",
                "wte",
                "gpt_neox.embed_in",
                "embeddings",
            ):
                mod = self._try_get(model, cand)
                if mod is not None:
                    return (mod, False)

        raise AttributeError(parent)

    def _resolve(self, model: Any, parent: str) -> Any:
        """Resolve a parent module (backwards-compatible helper)."""
        mod, _ = self._resolve_target(model, parent)
        return mod

    def _infer_num_heads(self, model: Any) -> int | None:
        if self.num_heads is not None:
            return self.num_heads
        cfg = getattr(model, "config", None)
        if cfg is not None:
            h = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
            if isinstance(h, int) and h > 0:
                return h
            text_cfg = getattr(cfg, "text_config", None)
            if text_cfg is not None:
                th = getattr(text_cfg, "num_attention_heads", getattr(text_cfg, "n_head", None))
                if isinstance(th, int) and th > 0:
                    return th
        return None

    def attach(self, model: Any) -> list[Any]:
        """Register one hook per watched parent module (idempotent)."""
        if self._handles:
            return list(self._handles)
        if self.num_heads is None:
            self.num_heads = self._infer_num_heads(model)
        for parent, cols in self.groups.items():
            try:
                module, is_pre = self._resolve_target(model, parent)
            except AttributeError as exc:
                raise ValueError(f"E02: model has no parent module {parent!r}") from exc
            hook = self._make_hook(parent, cols, is_pre=is_pre)
            if is_pre and hasattr(module, "register_forward_pre_hook"):
                self._handles.append(module.register_forward_pre_hook(hook))
            else:
                self._handles.append(module.register_forward_hook(hook))
        return list(self._handles)

    def _gather_torch_or_numpy(
        self,
        activation: Any,
        parent: str,
        cols: list[tuple[str, int, int]],
    ) -> Any:
        """Gather ``[S, len(cols)]`` on device with true mechanistic semantics."""
        import torch

        t = activation if _is_torch_tensor(activation) else torch.as_tensor(activation)
        raw_dtype = str(getattr(activation, "dtype", self.dtype))
        if "int8" in raw_dtype or "char" in raw_dtype:
            t = t.float() * self.scale
            self.dtype_meta["source"] = "int8_dequantized_on_device"
        elif "float8" in raw_dtype:
            t = t.float()
            self.dtype_meta["source"] = "fp8_upcast_on_device"

        if t.ndim == 1:
            t2d = t.unsqueeze(0)
        elif t.ndim == 2:
            t2d = t
        else:
            b_idx = self.batch_index if 0 <= self.batch_index < t.shape[0] else 0
            t2d = t[b_idx].reshape(-1, t.shape[-1])

        seq_len, dim = int(t2d.shape[0]), int(t2d.shape[1])
        locals_only = [local for _, local, _ in cols]

        # 1. MLP and SAE features: direct column gather on last dimension
        if parent.endswith(".mlp") or parent.endswith(".sae"):
            idx = _as_index(t2d, locals_only)
            return t2d.index_select(-1, idx)

        # 2. Attention heads: per-head L2 norm / RMS across head_dim when D == H * head_dim
        if parent.endswith(".attn"):
            max_req = max(locals_only) + 1 if locals_only else 1
            n_heads = self._parent_heads.get(parent, self.num_heads)
            if n_heads is not None and n_heads >= max_req and dim > n_heads and dim % n_heads == 0:
                head_dim = dim // n_heads
                heads_view = t2d.float().reshape(seq_len, n_heads, head_dim)
                if self.head_reduction == "rms":
                    reduced = torch.sqrt((heads_view * heads_view).mean(dim=-1))
                elif self.head_reduction == "mean":
                    reduced = heads_view.mean(dim=-1)
                else:
                    reduced = torch.linalg.vector_norm(heads_view, ord=2, dim=-1)
                idx = _as_index(reduced, locals_only)
                return reduced.index_select(-1, idx)
            if dim >= max_req:
                idx = _as_index(t2d, locals_only)
                return t2d.index_select(-1, idx)

        # 3. Logits: top-1 max logit, LOGITS:<id>, LOGITS-ENTROPY, plus top-k summary
        if parent == "logits":
            tf = t2d.float()
            out_cols: list[Any] = []
            probs = torch.softmax(tf, dim=-1) if dim > 1 else torch.ones_like(tf)
            entropy = (
                -(probs * torch.clamp(probs, min=1e-12).log()).sum(dim=-1)
                if dim > 1
                else torch.zeros(seq_len, device=tf.device)
            )
            # Record top-k logits summary for the latest token step without CPU sync
            if dim > 1:
                k = min(5, dim)
                top_vals, top_ids = torch.topk(tf[-1], k=k)
                ent_val = entropy[-1] if entropy.ndim > 0 else entropy
                # Store detached tensors without host synchronization;
                # Python float/int conversions are deferred until run finalization.
                self._raw_logits_summary.append(
                    (top_vals.detach(), top_ids.detach(), ent_val.detach())
                )
            for addr, local, _ in cols:
                if addr == "LOGITS-ENTROPY":
                    out_cols.append(entropy.unsqueeze(-1))
                elif _RE_LOGITS_TOK.match(addr):
                    out_cols.append(tf[:, local : local + 1])
                elif dim > 1 and self.num_heads is not None:
                    out_cols.append(tf.max(dim=-1, keepdim=True).values)
                else:
                    out_cols.append(tf[:, local : local + 1])
            return torch.cat(out_cols, dim=-1)

        # 4. Residual stream (L-LL, L-LL-DDDD, NORM, EMBED)
        out_res: list[Any] = []
        tf_res = t2d.float()
        for addr, local, _ in cols:
            if _RE_L_DIM.match(addr):
                out_res.append(tf_res[:, local : local + 1])
            elif dim > 1 and self.num_heads is not None:
                out_res.append(torch.linalg.vector_norm(tf_res, ord=2, dim=-1, keepdim=True))
            else:
                out_res.append(tf_res[:, local : local + 1])
        return torch.cat(out_res, dim=-1)

    def _make_hook(
        self,
        parent: str,
        cols: list[tuple[str, int, int]],
        *,
        is_pre: bool = False,
    ) -> HookFn:
        locals_only = [local for _, local, _ in cols]
        col_ids = [col for _, _, col in cols]

        def _process(raw_payload: Any) -> None:
            if self._detached:
                return
            try:
                activation = raw_payload[0] if isinstance(raw_payload, tuple) else raw_payload
                if hasattr(activation, "logits"):
                    activation = activation.logits
                if _is_torch_tensor(activation) or isinstance(activation, np.ndarray):
                    gathered = self._gather_torch_or_numpy(activation, parent, cols)
                else:
                    raw_dtype = str(getattr(activation, "dtype", self.dtype))
                    idx = _as_index(activation, locals_only)
                    gathered = activation.index_select(-1, idx)
                    if "int8" in raw_dtype or "char" in raw_dtype:
                        gathered = gathered * self.scale
                        self.dtype_meta["source"] = "int8_dequantized_on_device"

                flip = self._flip[parent]
                slot = self.pinned[parent][flip]
                self._flip[parent] = flip ^ 1
                if _is_torch_tensor(gathered) and getattr(gathered, "is_cuda", False):
                    # Defer host transfer until flush_sequence so hooks never stall CUDA
                    g_det = gathered.detach()
                    self._staging_seq[parent] = g_det.unsqueeze(0) if g_det.ndim == 1 else g_det
                else:
                    slot.copy_(gathered, non_blocking=True)
                    self._staging_seq[parent] = slot.read_matrix()
                    for col, value in zip(col_ids, slot.read()):
                        self._staging[col] = value
            except Exception:
                self.detach()
                self.flags["hook_detached"] = True

        if is_pre:

            def _pre_hook(module: Any, inputs: Any) -> None:
                _ = module
                _process(inputs)

            return _pre_hook

        def _fwd_hook(module: Any, inputs: Any, output: Any) -> None:
            _ = (module, inputs)
            _process(output)

        return _fwd_hook

    def detach(self) -> None:
        """Remove all hook handles; idempotent, never raises."""
        self._detached = True
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                continue
        self._handles = []

    def _commit_row(
        self,
        token_pos: int,
        gate_values: dict[str, float],
        elapsed_s: float | None = None,
        flags: dict[str, Any] | None = None,
    ) -> bool:
        armed = is_armed(self.trigger, token_pos, self.window, elapsed_s, flags, gate_values)
        if not armed:
            return False
        if self.capacity_tokens is not None and len(self.ring) >= self.capacity_tokens:
            self.flags["ring_overflow"] = True
            return False
        row = [gate_values.get(addr, math.nan) for addr in self.watched]
        self.ring.append(row)
        self.token_positions.append(int(token_pos))
        self.flags["triggered"] = True
        if self.sidecar is not None:
            self.sidecar.append_token(int(token_pos), row)
            if getattr(self.sidecar, "ring_overflow", False):
                self.flags["ring_overflow"] = True
        return True

    def flush_sequence(
        self,
        start_pos: int | None = None,
        elapsed_s: float | None = None,
        flags: dict[str, Any] | None = None,
    ) -> int:
        """Commit all staged sequence tokens from a multi-token forward pass ``[S, W]``.

        Respects ``capture_mode``:
        - ``"all"``: commits all ``0..S-1`` rows on prefill and ``[-1]`` on decode steps.
        - ``"decode_only"`` / ``"last_token"``: commits only the last sequence token ``[-1]``.
        - ``"prefill_only"``: commits only the first forward pass.
        """
        if self._detached or self.flags.get("hook_detached") or not self._staging_seq:
            self._staging = {}
            self._staging_seq = {}
            return 0
        if self.capture_mode == "prefill_only" and self._prefill_done:
            self._staging = {}
            self._staging_seq = {}
            return 0

        seq_len = max(int(mat.shape[0]) for mat in self._staging_seq.values())
        w = len(self.watched)
        first_item = next(iter(self._staging_seq.values()))
        if _is_torch_tensor(first_item) and getattr(first_item, "is_cuda", False):
            import torch

            dev = first_item.device
            gpu_mat = torch.full((seq_len, w), float("nan"), dtype=torch.float32, device=dev)
            for parent in self.groups:
                t_item = self._staging_seq.get(parent)
                if t_item is None:
                    continue
                t_f = (
                    t_item.float()
                    if _is_torch_tensor(t_item)
                    else torch.as_tensor(t_item, dtype=torch.float32, device=dev)
                )
                if t_f.device != dev:
                    t_f = t_f.to(device=dev)
                c_idx = torch.as_tensor(self._group_col_np[parent], dtype=torch.long, device=dev)
                if int(t_f.shape[0]) == seq_len:
                    gpu_mat.index_copy_(1, c_idx, t_f)
                else:
                    gpu_mat[-1:].index_copy_(1, c_idx, t_f[-1:])
            # Single host transfer for the entire watched activation slice
            full_mat = np.asarray(gpu_mat.to(device="cpu").numpy(), dtype=np.float64)
        else:
            full_mat = np.full((seq_len, w), np.nan, dtype=np.float64)
            for parent in self.groups:
                raw_mat = self._staging_seq.get(parent)
                if raw_mat is None:
                    continue
                mat = raw_mat if isinstance(raw_mat, np.ndarray) else _to_2d_numpy(raw_mat)
                col_arr = self._group_col_np[parent]
                if mat.shape[0] == seq_len:
                    full_mat[:, col_arr] = mat
                else:
                    full_mat[-1, col_arr] = mat[-1]

        self._staging = {}
        self._staging_seq = {}
        self._prefill_done = True

        if self.capture_mode in ("decode_only", "last_token"):
            eff_mat = full_mat[-1:]
        else:
            eff_mat = full_mat

        n_rows = int(eff_mat.shape[0])
        base_pos = self._auto_token_cursor if start_pos is None else int(start_pos)

        # Fast vectorized commit path for unconditional 'always' trigger without sidecar
        if self.trigger.get("kind", "always") == "always" and self.sidecar is None:
            w_start = int(self.window.get("start_token", 0))
            w_end = int(self.window.get("end_token", 1 << 30))
            every_n = int(self.window.get("every_n", 1))
            if every_n >= 1:
                armed_offsets = [
                    off
                    for off in range(n_rows)
                    if w_start <= (base_pos + off) < w_end
                    and ((base_pos + off) - w_start) % every_n == 0
                ]
            else:
                armed_offsets = []
            if self.capacity_tokens is not None:
                rem = max(0, self.capacity_tokens - len(self.ring))
                if len(armed_offsets) > rem:
                    self.flags["ring_overflow"] = True
                    armed_offsets = armed_offsets[:rem]
            if armed_offsets:
                sub = eff_mat[armed_offsets]
                self.ring.extend(sub.tolist())
                self.token_positions.extend(base_pos + off for off in armed_offsets)
                self.flags["triggered"] = True
            self._auto_token_cursor = base_pos + n_rows
            return len(armed_offsets)

        committed = 0
        for offset in range(n_rows):
            row_arr = eff_mat[offset]
            pos = base_pos + offset
            gate_values = {
                addr: float(row_arr[col])
                for col, addr in enumerate(self.watched)
                if not math.isnan(float(row_arr[col]))
            }
            if self._commit_row(pos, gate_values, elapsed_s=elapsed_s, flags=flags):
                committed += 1
        self._auto_token_cursor = base_pos + n_rows
        return committed

    def end_token(
        self,
        token_pos: int,
        elapsed_s: float | None = None,
        flags: dict[str, Any] | None = None,
        values: dict[str, float] | None = None,
    ) -> bool:
        """Seal one token: armed -> append row; unarmed -> write nothing.

        Returns True when a row was appended. Ring-full sets
        ``ring_overflow`` and drops the Deep payload.
        """
        if self._detached or self.flags.get("hook_detached"):
            self._staging = {}
            self._staging_seq = {}
            return False
        if self._staging_seq and not self._staging:
            for parent, cols in self.groups.items():
                raw_m = self._staging_seq.get(parent)
                if raw_m is not None:
                    m2d = raw_m if isinstance(raw_m, np.ndarray) else _to_2d_numpy(raw_m)
                    for local_k, (_, _, col_id) in enumerate(cols):
                        self._staging[col_id] = float(m2d[-1, local_k])
        gate_values: dict[str, float] = {}
        for col, addr in enumerate(self.watched):
            if col in self._staging:
                gate_values[addr] = self._staging[col]
        if values:
            gate_values.update(values)
        self._staging = {}
        self._staging_seq = {}
        self._auto_token_cursor = max(self._auto_token_cursor, int(token_pos) + 1)
        return self._commit_row(int(token_pos), gate_values, elapsed_s=elapsed_s, flags=flags)

    def to_matrix(self) -> np.ndarray:
        """Return captured ``[T, W]`` float32 activation matrix."""
        if not self.ring:
            return np.zeros((0, len(self.watched)), dtype=np.float32)
        return np.asarray(self.ring, dtype=np.float32)

    def to_bytes(self, dtype: str | None = None) -> bytes:
        """Encode the ring row-major ``[T, W]`` little-endian (spec §FILE-deepbin)."""
        stored = dtype or self.dtype
        if stored not in _DTYPE_BYTES:
            raise ValueError(f"E02: bad precision {stored!r}")
        if not self.ring:
            return b""
        mat = np.asarray(self.ring, dtype=np.float32)
        encoded = encode_deep_array(mat, stored)
        return bytes(encoded.tobytes(order="C"))

    def deep_index(
        self,
        run_id: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the ``deep_index.json`` pointer for the sealed ring."""
        tokens = len(self.ring)
        columns: dict[str, dict[str, Any]] = {}
        for col, addr in enumerate(self.watched):
            columns[addr] = {"offset": col, "dtype": self.dtype, "shape": [tokens]}
        out: dict[str, Any] = {
            "run_id": run_id,
            "triggered": bool(self.flags["triggered"]),
            "dtype": self.dtype,
            "tokens": tokens,
            "every_n": int(self.window.get("every_n", 1)),
            "window": {
                "start_token": int(self.window.get("start_token", 0)),
                "end_token": int(self.window.get("end_token", 0)),
            },
            "token_positions": list(self.token_positions),
            "capture_mode": self.capture_mode,
            "watched": list(self.watched),
            "columns": columns,
            "dtype_meta": dict(self.dtype_meta),
        }
        if self.logits_summary:
            out["logits_summary"] = list(self.logits_summary)
        if context:
            for k, v in context.items():
                if k not in out:
                    out[k] = v
        return out


class AttachedTaps(list[Any]):
    """Hook handles list carrying the :class:`DeepTap` recorder."""

    def __init__(self, handles: list[Any], recorder: DeepTap) -> None:
        super().__init__(handles)
        self.recorder: DeepTap = recorder


def attach_hooks(
    model: Any,
    watchlist: dict[str, Any],
    window: dict[str, int] | None = None,
    trigger: str | dict[str, Any] | None = None,
    dtype: str | None = None,
    capacity_tokens: int | None = None,
    scale: float = 1.0,
    stream: Any = None,
    num_heads: int | None = None,
    head_reduction: str = "l2_norm",
    capture_mode: str = "all",
    sidecar: Any = None,
    batch_index: int = 0,
) -> AttachedTaps:
    """Register Deep gather hooks only on watched parent modules.

    ``watchlist`` must carry ``addresses`` (see :func:`group_by_parent`);
    window/trigger/dtype default to the watchlist's merged values.
    """
    if not isinstance(watchlist, dict) or "addresses" not in watchlist:
        raise ValueError("E02: watchlist dict must carry 'addresses'")
    addresses = [str(a) for a in watchlist["addresses"]]
    eff_window = window or watchlist.get("merged_window") or watchlist.get("window")
    eff_trigger = (
        trigger
        if trigger is not None
        else watchlist.get("effective_trigger", watchlist.get("trigger"))
    )
    eff_dtype = dtype or str(
        watchlist.get("effective_precision", watchlist.get("precision", "fp16"))
    )
    tap = DeepTap(
        addresses,
        window=dict(eff_window) if isinstance(eff_window, dict) else None,
        trigger=eff_trigger,
        dtype=eff_dtype,
        capacity_tokens=capacity_tokens,
        scale=scale,
        stream=stream,
        num_heads=num_heads,
        head_reduction=head_reduction,
        capture_mode=capture_mode,
        sidecar=sidecar,
        batch_index=batch_index,
    )
    handles = tap.attach(model)
    return AttachedTaps(handles, tap)


def detach_hooks(handles: list[Any]) -> None:
    """Remove hook handles; idempotent, never raises (hook-throw safe)."""
    recorder = getattr(handles, "recorder", None)
    if isinstance(recorder, DeepTap):
        recorder.detach()
        try:
            handles.clear()
        except Exception:
            pass
        return
    for handle in list(handles):
        try:
            handle.remove()
        except Exception:
            continue
    try:
        handles.clear()
    except Exception:
        pass
