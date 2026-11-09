r"""Real-time activation steering and causal ablation engine (spec §STEER).

Allows researchers and engineers to move from passive flight recording to
active causal intervention:
- **Additive Steering**: Inject steering vectors :math:`x \leftarrow x + \alpha v`
  at specific layers (residual stream, MLP, or attention) with optional L2 norm
  preservation to avoid perplexity explosions.
- **Directional Subspace Ablation**: Project out refusal, sycophancy, or bias
  directions :math:`x \leftarrow x - \frac{x \cdot v}{\|v\|^2} v`.
- **Neuron Clamping / Zeroing**: Zero out, clamp, or replace individual MLP
  neurons (``N-LL-NNNN``) during forward passes.
- **Attention Head Ablation**: Silence or clamp specific attention heads
  (``A-LL-HH``) to test causal contribution to downstream outputs.
- **Token Conditioning**: Steer only during autoregressive decode (``mode="decode"``),
  prompt prefill (``mode="prefill"``), or specific token index positions.

Seamlessly integrates as a standalone context manager or directly inside
:meth:`glassbox.GlassBox.watch(..., steer=[...], ablate=[...])`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_RE_N = re.compile(r"^N-(\d+)-(\d+)$")
_RE_A = re.compile(r"^A-(\d+)-(\d+)$")
_RE_L = re.compile(r"^L-(\d+)$")


def _is_torch_tensor(obj: Any) -> bool:
    return obj.__class__.__module__.split(".")[0] == "torch" and hasattr(obj, "ndim")


@dataclass
class SteeringVector:
    """Specification for real-time directional activation addition."""

    layer: int
    vector: Any  # torch.Tensor, np.ndarray, or list[float]
    scale: float = 1.0
    component: str = "residual"  # "residual", "mlp", "attn"
    mode: str = "all"  # "all", "decode", "prefill"
    norm_preserve: bool = False
    positions: list[int] | None = None
    # Pulsed steering for autoregressive generation. When ``pulse_steps`` is set the
    # vector is applied for only that many consecutive decode steps (never during
    # prefill) once the pulse is triggered, then disarms. Trigger either at a fixed
    # decode step (``pulse_start_step``, 1-indexed, counted per generation) or from
    # outside via :meth:`SteeringController.fire_pulse` (e.g. a LogitsProcessor that
    # watches generated text).
    pulse_steps: int | None = None
    pulse_start_step: int | None = None

    def __post_init__(self) -> None:
        if self.pulse_steps is not None and self.pulse_steps < 1:
            raise ValueError(f"E02: pulse_steps must be >= 1 (got {self.pulse_steps})")
        if self.pulse_start_step is not None:
            if self.pulse_steps is None:
                raise ValueError("E02: pulse_start_step requires pulse_steps")
            if self.pulse_start_step < 1:
                raise ValueError(f"E02: pulse_start_step must be >= 1 (got {self.pulse_start_step})")
        if self.layer < 1:
            raise ValueError(f"E02: layer index must be >= 1 (got {self.layer})")
        if self.component not in ("residual", "mlp", "attn"):
            raise ValueError(
                f"E02: bad component {self.component!r}; expected 'residual'|'mlp'|'attn'"
            )
        if self.mode not in ("all", "decode", "prefill"):
            raise ValueError(f"E02: bad mode {self.mode!r}; expected 'all'|'decode'|'prefill'")


@dataclass
class AblationTarget:
    """Specification for causal ablation (zeroing, clamping, or orthogonal removal)."""

    target: str  # "N-64-0024", "A-18-03", "L-64", etc.
    mode: str = "zero"  # "zero", "clamp_max", "clamp_min", "clamp", "mean", "orthogonal"
    clamp_min: float | None = None
    clamp_max: float | None = None
    # Scalar, or a per-head / per-feature mean vector (torch.Tensor, np.ndarray, list[float]).
    # For attention heads (A-LL-HH) pass a [head_dim] vector for true mean-ablation.
    mean_value: Any = 0.0
    direction: Any = None  # torch.Tensor, np.ndarray, or list[float] for orthogonal mode
    token_mode: str = "all"  # "all", "decode", "prefill"
    positions: list[int] | None = None

    def __post_init__(self) -> None:
        valid_modes = ("zero", "clamp_max", "clamp_min", "clamp", "mean", "orthogonal")
        if self.mode not in valid_modes:
            raise ValueError(f"E02: bad ablation mode {self.mode!r}; expected one of {valid_modes}")
        if self.token_mode not in ("all", "decode", "prefill"):
            raise ValueError(
                f"E02: bad token_mode {self.token_mode!r}; expected 'all'|'decode'|'prefill'"
            )
        if self.mode == "orthogonal" and self.direction is None:
            raise ValueError("E02: orthogonal ablation requires direction vector")


class SteeringController:
    """Active intervention controller for PyTorch LLMs."""

    def __init__(
        self,
        model: Any | None = None,
        *,
        steer: list[SteeringVector] | SteeringVector | None = None,
        ablate: list[AblationTarget] | AblationTarget | None = None,
        num_heads: int | None = None,
    ) -> None:
        self.model: Any | None = model
        self.num_heads: int | None = num_heads
        self.steering_vectors: list[SteeringVector] = []
        self.ablation_targets: list[AblationTarget] = []
        self._handles: list[Any] = []
        self._attached: bool = False
        self._stats: dict[str, Any] = {
            "invocations": 0,
            "tokens_steered": 0,
            "tokens_ablated": 0,
            "hooks_installed": 0,
        }
        self._parent_heads: dict[str, int] = {}
        self._pulse_state: dict[int, dict[str, int]] = {}

        if steer is not None:
            if isinstance(steer, SteeringVector):
                self.steering_vectors.append(steer)
            else:
                self.steering_vectors.extend(steer)

        if ablate is not None:
            if isinstance(ablate, AblationTarget):
                self.ablation_targets.append(ablate)
            else:
                self.ablation_targets.extend(ablate)

    @property
    def stats(self) -> dict[str, Any]:
        """Telemetry statistics on active steering and ablation invocations."""
        return dict(self._stats)

    def add_steering(
        self,
        layer: int,
        vector: Any,
        scale: float = 1.0,
        component: str = "residual",
        mode: str = "all",
        norm_preserve: bool = False,
        positions: list[int] | None = None,
        pulse_steps: int | None = None,
        pulse_start_step: int | None = None,
    ) -> SteeringController:
        """Add a real-time steering vector intervention."""
        sv = SteeringVector(
            layer=layer,
            vector=vector,
            scale=scale,
            component=component,
            mode=mode,
            norm_preserve=norm_preserve,
            positions=positions,
            pulse_steps=pulse_steps,
            pulse_start_step=pulse_start_step,
        )
        self.steering_vectors.append(sv)
        if self._attached and self.model is not None:
            self.detach()
            self.attach(self.model)
        return self

    def fire_pulse(self, steps: int | None = None) -> int:
        """Arm every pulsed steering vector for its next decode step(s).

        Call this from outside the forward pass (e.g. a ``LogitsProcessor`` that sees
        the generated text) at the moment the pulse should begin. Returns the number
        of pulsed vectors armed.
        """
        armed = 0
        for sv in self.steering_vectors:
            if sv.pulse_steps is None:
                continue
            st = self._pulse_state.setdefault(id(sv), {"decode_calls": 0, "remaining": 0})
            st["remaining"] = int(steps if steps is not None else sv.pulse_steps)
            armed += 1
        return armed

    def reset_pulses(self) -> None:
        """Disarm all pulses and reset per-generation decode counters."""
        self._pulse_state.clear()

    def add_ablation(
        self,
        target: str,
        mode: str = "zero",
        clamp_min: float | None = None,
        clamp_max: float | None = None,
        mean_value: Any = 0.0,
        direction: Any = None,
        token_mode: str = "all",
        positions: list[int] | None = None,
    ) -> SteeringController:
        """Add a causal ablation intervention."""
        at = AblationTarget(
            target=target,
            mode=mode,
            clamp_min=clamp_min,
            clamp_max=clamp_max,
            mean_value=mean_value,
            direction=direction,
            token_mode=token_mode,
            positions=positions,
        )
        self.ablation_targets.append(at)
        if self._attached and self.model is not None:
            self.detach()
            self.attach(self.model)
        return self

    def clear(self) -> SteeringController:
        """Clear all active steering and ablation rules."""
        self.detach()
        self.steering_vectors.clear()
        self.ablation_targets.clear()
        self._pulse_state.clear()
        self._stats = {
            "invocations": 0,
            "tokens_steered": 0,
            "tokens_ablated": 0,
            "hooks_installed": 0,
        }
        return self

    def _infer_num_heads(self, model: Any) -> int | None:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return self.num_heads
        for attr in ("num_attention_heads", "n_head", "num_heads"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        return self.num_heads

    def _try_get(self, root: Any, path: str) -> Any:
        try:
            node = root
            for part in path.split("."):
                if part.isdigit() and isinstance(node, (list, tuple)):
                    node = node[int(part)]
                elif hasattr(node, "__getitem__") and not hasattr(node, part):
                    try:
                        node = node[int(part)] if part.isdigit() else node[part]
                    except (IndexError, KeyError, TypeError):
                        node = getattr(node, part)
                else:
                    node = getattr(node, part)
            return node
        except Exception:
            return None

    def _resolve_target(self, model: Any, parent: str) -> tuple[Any, bool]:
        """Resolve (module, is_pre_hook) across canonical and HF/PyTorch architectures."""
        direct = self._try_get(model, parent)
        if direct is not None:
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

        m_layer = re.match(r"^layer-(\d+)(?:\.(mlp|attn))?$", parent)
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
                            for h_attr in ("num_heads", "num_attention_heads", "n_head"):
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
            return (self._try_get(model, f"model.layers.{idx}"), False)

        raise ValueError(f"E02: cannot resolve parent module {parent!r} in model")

    def attach(self, model: Any | None = None) -> SteeringController:
        """Attach steering and ablation hooks to the model."""
        if self._attached:
            return self
        self.reset_pulses()
        target_model = model or self.model
        if target_model is None:
            raise ValueError("E02: SteeringController.attach() requires a model")
        self.model = target_model
        if self.num_heads is None:
            self.num_heads = self._infer_num_heads(self.model)

        # Group rules by target parent module
        grouped_steer: dict[str, list[SteeringVector]] = {}
        for sv in self.steering_vectors:
            parent = f"layer-{sv.layer:02d}"
            if sv.component != "residual":
                parent = f"{parent}.{sv.component}"
            grouped_steer.setdefault(parent, []).append(sv)

        grouped_ablate: dict[str, list[tuple[AblationTarget, int | None]]] = {}
        for at in self.ablation_targets:
            local_idx: int | None = None
            m_n = _RE_N.match(at.target)
            if m_n:
                parent = f"layer-{int(m_n.group(1)):02d}.mlp"
                local_idx = int(m_n.group(2)) - 1
            else:
                m_a = _RE_A.match(at.target)
                if m_a:
                    parent = f"layer-{int(m_a.group(1)):02d}.attn"
                    local_idx = int(m_a.group(2)) - 1
                else:
                    m_l = _RE_L.match(at.target)
                    if m_l:
                        parent = f"layer-{int(m_l.group(1)):02d}"
                    else:
                        parent = at.target
            grouped_ablate.setdefault(parent, []).append((at, local_idx))

        all_parents = set(grouped_steer.keys()) | set(grouped_ablate.keys())

        for parent in all_parents:
            steers = grouped_steer.get(parent, [])
            ablates = grouped_ablate.get(parent, [])
            try:
                module, is_pre = self._resolve_target(self.model, parent)
            except Exception as exc:
                raise ValueError(f"E02: cannot hook steering parent {parent!r}") from exc

            hook_fn = self._build_intervention_hook(parent, steers, ablates, is_pre=is_pre)
            if is_pre and hasattr(module, "register_forward_pre_hook"):
                self._handles.append(module.register_forward_pre_hook(hook_fn))
            else:
                self._handles.append(module.register_forward_hook(hook_fn))

        self._attached = True
        self._stats["hooks_installed"] = len(self._handles)
        return self

    def detach(self) -> None:
        """Detach all steering and ablation hooks from the model."""
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        self._attached = False

    def __enter__(self) -> SteeringController:
        return self.attach()

    def __exit__(self, *args: Any) -> None:
        self.detach()

    def _apply_steering(
        self,
        hidden: Any,
        steers: list[SteeringVector],
    ) -> Any:
        import torch

        seq_len = hidden.shape[-2] if hidden.ndim >= 2 else 1
        for sv in steers:
            # Pulsed steering: decode-only, fires for N steps, then disarms.
            if sv.pulse_steps is not None:
                st = self._pulse_state.setdefault(id(sv), {"decode_calls": 0, "remaining": 0})
                if seq_len > 1:
                    # New prefill == new generation: reset counters, disarm.
                    st["decode_calls"] = 0
                    st["remaining"] = 0
                    continue
                st["decode_calls"] += 1
                if sv.pulse_start_step is not None and st["decode_calls"] == sv.pulse_start_step:
                    st["remaining"] = sv.pulse_steps
                if st["remaining"] <= 0:
                    continue
                st["remaining"] -= 1
                self._stats["pulses_applied"] = self._stats.get("pulses_applied", 0) + 1

            # Token mode filter
            if sv.mode == "decode" and seq_len > 1:
                continue
            if sv.mode == "prefill" and seq_len == 1:
                continue

            v = sv.vector
            if not _is_torch_tensor(v):
                v = torch.as_tensor(v, dtype=hidden.dtype, device=hidden.device)
            else:
                v = v.to(device=hidden.device, dtype=hidden.dtype)

            if v.ndim == 1:
                v = v.view(*([1] * (hidden.ndim - 1)), -1)

            # Sliced position application if specified
            if sv.positions is not None and hidden.ndim >= 2:
                # Steer only specified positions
                mask = torch.zeros(seq_len, dtype=torch.bool, device=hidden.device)
                for p in sv.positions:
                    if 0 <= p < seq_len:
                        mask[p] = True
                if not mask.any():
                    continue
                steered_slice = hidden[:, mask] + sv.scale * v
                if sv.norm_preserve:
                    o_norm = torch.linalg.vector_norm(hidden[:, mask], dim=-1, keepdim=True)
                    raw_n = torch.linalg.vector_norm(steered_slice, dim=-1, keepdim=True)
                    n_norm = raw_n.clamp_min(1e-8)
                    steered_slice = steered_slice * (o_norm / n_norm)
                hidden[:, mask] = steered_slice
            else:
                steered = hidden + sv.scale * v
                if sv.norm_preserve:
                    o_norm = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True)
                    n_norm = torch.linalg.vector_norm(steered, dim=-1, keepdim=True).clamp_min(1e-8)
                    steered = steered * (o_norm / n_norm)
                hidden = steered

            self._stats["tokens_steered"] += seq_len
        return hidden

    def _apply_ablation(
        self,
        tensor: Any,
        parent: str,
        ablates: list[tuple[AblationTarget, int | None]],
    ) -> Any:
        import torch

        seq_len = tensor.shape[-2] if tensor.ndim >= 2 else 1
        dim = int(tensor.shape[-1])

        for at, local_idx in ablates:
            if at.token_mode == "decode" and seq_len > 1:
                continue
            if at.token_mode == "prefill" and seq_len == 1:
                continue

            # 1. Directional Orthogonal Ablation
            if at.mode == "orthogonal" and at.direction is not None:
                d = at.direction
                if not _is_torch_tensor(d):
                    d = torch.as_tensor(d, dtype=tensor.dtype, device=tensor.device)
                else:
                    d = d.to(device=tensor.device, dtype=tensor.dtype)
                if d.ndim == 1:
                    d = d.view(*([1] * (tensor.ndim - 1)), -1)
                d_sq = torch.sum(d * d, dim=-1, keepdim=True).clamp_min(1e-8)
                proj = (torch.sum(tensor * d, dim=-1, keepdim=True) / d_sq) * d
                tensor = tensor - proj
                self._stats["tokens_ablated"] += seq_len
                continue

            # 2. Attention Head Ablation
            if parent.endswith(".attn") and local_idx is not None:
                n_heads = self._parent_heads.get(parent, self.num_heads)
                if n_heads is not None and dim > n_heads and dim % n_heads == 0:
                    head_dim = dim // n_heads
                    h_start = local_idx * head_dim
                    h_end = h_start + head_dim
                    if at.mode == "zero":
                        tensor[..., h_start:h_end] = 0.0
                    elif at.mode == "clamp_max" and at.clamp_max is not None:
                        tensor[..., h_start:h_end] = torch.clamp_max(
                            tensor[..., h_start:h_end], at.clamp_max
                        )
                    elif at.mode == "clamp_min" and at.clamp_min is not None:
                        tensor[..., h_start:h_end] = torch.clamp_min(
                            tensor[..., h_start:h_end], at.clamp_min
                        )
                    elif at.mode == "mean":
                        tensor[..., h_start:h_end] = self._mean_like(
                            at.mean_value, tensor[..., h_start:h_end]
                        )
                    self._stats["tokens_ablated"] += seq_len
                    continue

            # 3. Neuron or Column Slicing Ablation
            if local_idx is not None and local_idx < dim:
                if at.mode == "zero":
                    tensor[..., local_idx] = 0.0
                elif at.mode == "clamp_max" and at.clamp_max is not None:
                    tensor[..., local_idx] = torch.clamp_max(tensor[..., local_idx], at.clamp_max)
                elif at.mode == "clamp_min" and at.clamp_min is not None:
                    tensor[..., local_idx] = torch.clamp_min(tensor[..., local_idx], at.clamp_min)
                elif at.mode == "clamp":
                    tensor[..., local_idx] = torch.clamp(
                        tensor[..., local_idx], min=at.clamp_min, max=at.clamp_max
                    )
                elif at.mode == "mean":
                    tensor[..., local_idx] = self._mean_like(at.mean_value, tensor[..., local_idx])
                self._stats["tokens_ablated"] += seq_len

        return tensor

    @staticmethod
    def _mean_like(mean_value: Any, like: Any) -> Any:
        """Coerce a scalar or per-feature mean (tensor/ndarray/list) to ``like``'s device/dtype."""
        import torch

        if _is_torch_tensor(mean_value):
            return mean_value.to(device=like.device, dtype=like.dtype)
        if isinstance(mean_value, (int, float)):
            return mean_value
        return torch.as_tensor(mean_value, dtype=like.dtype, device=like.device)

    def _build_intervention_hook(
        self,
        parent: str,
        steers: list[SteeringVector],
        ablates: list[tuple[AblationTarget, int | None]],
        *,
        is_pre: bool = False,
    ) -> Any:
        def _intervene(payload: Any) -> Any:
            self._stats["invocations"] += 1
            is_tuple = isinstance(payload, tuple)
            target = payload[0] if is_tuple else payload

            if not _is_torch_tensor(target):
                return payload

            # Ablations applied first (suppress circuits)
            if ablates:
                target = self._apply_ablation(target, parent, ablates)

            # Steering applied second (add directional shifts)
            if steers:
                target = self._apply_steering(target, steers)

            if is_tuple:
                return (target,) + payload[1:]
            return target

        if is_pre:

            def _pre_hook(module: Any, inputs: Any) -> Any:
                _ = module
                return _intervene(inputs)

            return _pre_hook

        def _fwd_hook(module: Any, inputs: Any, output: Any) -> Any:
            _ = (module, inputs)
            return _intervene(output)

        return _fwd_hook


def steer(
    layer: int,
    vector: Any,
    scale: float = 1.0,
    component: str = "residual",
    mode: str = "all",
    norm_preserve: bool = False,
    positions: list[int] | None = None,
    pulse_steps: int | None = None,
    pulse_start_step: int | None = None,
) -> SteeringVector:
    """Convenience factory for a :class:`SteeringVector`."""
    return SteeringVector(
        layer=layer,
        vector=vector,
        scale=scale,
        component=component,
        mode=mode,
        norm_preserve=norm_preserve,
        positions=positions,
        pulse_steps=pulse_steps,
        pulse_start_step=pulse_start_step,
    )


def ablate(
    target: str,
    mode: str = "zero",
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    mean_value: Any = 0.0,
    direction: Any = None,
    token_mode: str = "all",
    positions: list[int] | None = None,
) -> AblationTarget:
    """Convenience factory for an :class:`AblationTarget`."""
    return AblationTarget(
        target=target,
        mode=mode,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        mean_value=mean_value,
        direction=direction,
        token_mode=token_mode,
        positions=positions,
    )
