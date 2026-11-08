"""Distributed Tensor Parallel sharding and zero-NVLink routing (spec §DIST-tp).

Provides TPShardRouter to coordinate activation probe ownership and active steering
across multi-GPU nodes under Tensor Parallelism (TP) without requiring NVLink communication.

Key invariants:
- Residual Stream (L-*, NORM, EMBED): Replicated across TP ranks post row-parallel all_reduce.
  Passive recording: Rank 0 only.
  Active steering: Applied identically across ALL TP ranks to prevent forward divergence.
- Attention Heads (A-LL-HH): Column-parallel partitioned across ranks.
  Owned by rank r = (HH - 1) // heads_per_rank.
- MLP Neurons (N-LL-NNNN): Column-parallel partitioned across ranks.
  Owned by rank r = (NNNN - 1) // neurons_per_rank.
- Logits Entropy (LOGITS-ENTROPY): Sharded vocabulary across ParallelLMHead.
  Each rank computes 3-scalar reduction (m_r, s_r, w_r).
  Collator / Router reconstructs exact global Shannon entropy offline with 0 inter-GPU traffic.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def _to_numpy(x: Any) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


_RE_A = re.compile(r"^A-(\d+)-(\d+)$")
_RE_N = re.compile(r"^N-(\d+)-(\d+)$")
_RE_L = re.compile(r"^L-(\d+)(?:-(\d+))?$")
_RE_LOGITS_TOK = re.compile(r"^LOGITS:(\d+)$")


class ShardRoute(tuple):
    """Routing decision tuple (owner_rank, local_index) with attribute and mapping access."""

    def __new__(
        cls,
        owner_rank: int,
        local_index: int,
        component: str = "",
        should_steer_all_ranks: bool = False,
    ) -> ShardRoute:
        return super().__new__(cls, (int(owner_rank), int(local_index)))

    def __init__(
        self,
        owner_rank: int,
        local_index: int,
        component: str = "",
        should_steer_all_ranks: bool = False,
    ) -> None:
        self.owner_rank = int(owner_rank)
        self.local_index = int(local_index)
        self.component = component
        self.should_steer_all_ranks = should_steer_all_ranks

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "owner_rank":
                return self.owner_rank
            if key == "local_index":
                return self.local_index
            if key == "component":
                return self.component
            if key == "should_steer_all_ranks":
                return self.should_steer_all_ranks
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


class LocalReduction(tuple):
    """Local logits reduction tuple (m_r, s_r, w_r) supporting unpacking and property access."""

    def __new__(cls, m_r: Any, s_r: Any, w_r: Any) -> LocalReduction:
        return super().__new__(cls, (m_r, s_r, w_r))

    @property
    def m_r(self) -> Any:
        return self[0]

    @property
    def s_r(self) -> Any:
        return self[1]

    @property
    def w_r(self) -> Any:
        return self[2]

    @property
    def m(self) -> Any:
        return self[0]

    @property
    def s(self) -> Any:
        return self[1]

    @property
    def w(self) -> Any:
        return self[2]


class TPShardRouter:
    """Tensor Parallel shard router for probe ownership, steering, and logits entropy reduction.

    Manages column-parallel sharding of attention heads and MLP neurons,
    passive recording vs active steering for residual stream probes, and
    3-scalar local reduction of vocabulary logits for exact zero-communication
    global Shannon entropy reconstruction.
    """

    def __init__(
        self,
        tp_rank: int = 0,
        tp_size: int = 1,
        architecture: dict[str, Any] | None = None,
        num_heads: int | None = None,
        mlp_dim: int | None = None,
        vocab_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize TPShardRouter.

        Args:
            tp_rank: Current process tensor-parallel rank (0 <= tp_rank < tp_size).
            tp_size: Total tensor-parallel world size (>= 1).
            architecture: Optional model architecture config dict containing heads/neurons/vocab.
            num_heads: Explicit attention head count override.
            mlp_dim: Explicit intermediate neuron count override.
            vocab_size: Explicit vocabulary size override.
            **kwargs: Extra arguments for compatibility.
        """
        self.tp_rank = int(tp_rank)
        self.tp_size = max(1, int(tp_size))

        arch: dict[str, Any] = {}
        if architecture is not None and isinstance(architecture, dict):
            if "architecture" in architecture and isinstance(architecture["architecture"], dict):
                arch = dict(architecture["architecture"])
            else:
                arch = dict(architecture)
            if "inventory" in architecture and isinstance(architecture["inventory"], dict):
                inv = architecture["inventory"]
                if "neurons_per_layer" in inv and "mlp_per_layer" not in arch:
                    arch["mlp_per_layer"] = inv["neurons_per_layer"]

        heads_val = (
            num_heads
            if num_heads is not None
            else (
                arch.get("num_attention_heads")
                or arch.get("heads")
                or arch.get("n_head")
                or arch.get("num_heads")
                or kwargs.get("num_heads")
                or 32
            )
        )
        self.num_attention_heads = int(heads_val)
        self.num_heads = self.num_attention_heads

        neurons_val = (
            mlp_dim
            if mlp_dim is not None
            else (
                arch.get("intermediate_size")
                or arch.get("num_neurons")
                or arch.get("mlp_per_layer")
                or arch.get("neurons_per_layer")
                or arch.get("mlp_dim")
                or kwargs.get("num_neurons")
                or kwargs.get("mlp_dim")
                or 8192
            )
        )
        self.num_neurons = int(neurons_val)
        self.intermediate_size = self.num_neurons
        self.mlp_dim = self.num_neurons

        vocab_val = (
            vocab_size
            if vocab_size is not None
            else (
                arch.get("vocab_size")
                or arch.get("vocab")
                or kwargs.get("vocab_size")
                or 32000
            )
        )
        self.vocab_size = int(vocab_val)

        self.heads_per_rank = max(1, self.num_attention_heads // self.tp_size)
        self.neurons_per_rank = max(1, self.num_neurons // self.tp_size)
        self.architecture = architecture

    @staticmethod
    def _shard_index(idx: int, total_items: int, tp_size: int) -> tuple[int, int]:
        """Partition total_items contiguously across tp_size ranks with balanced remainder.

        Distributes remainder items across the first (total_items % tp_size) ranks.
        When total_items is divisible by tp_size, this is identical to:
            owner = idx // (total_items // tp_size)
            local = idx % (total_items // tp_size)

        Args:
            idx: 0-based global item index (0 <= idx < total_items).
            total_items: Total count of items (heads or neurons).
            tp_size: World size (>= 1).

        Returns:
            (owner_rank, local_index) where 0 <= owner_rank < tp_size.
        """
        if tp_size <= 1:
            return 0, idx
        base = total_items // tp_size
        rem = total_items % tp_size
        cutoff = rem * (base + 1)
        if idx < cutoff:
            return idx // (base + 1), idx % (base + 1)
        else:
            if base == 0:
                return min(idx, tp_size - 1), 0
            offset = idx - cutoff
            return rem + offset // base, offset % base

    def is_owner(self, address: str, rank: int | None = None) -> bool:
        """Check if rank owns passive recording for the given probe address.

        Rules:
        - LOGITS-ENTROPY: All ranks participate in local reduction (returns True on all ranks).
        - Residual Stream (L-*, NORM, EMBED): Recorded on Rank 0 only (returns True if rank == 0).
        - Attention Heads (A-LL-HH): Head HH (1-based) is owned by (HH - 1) // heads_per_rank.
        - MLP Neurons (N-LL-NNNN): Neuron NNNN (1-based) is owned by (NNNN - 1) // neurons_per_rank.
        - Other / unrecognized addresses default to True on rank 0.
        """
        r = self.tp_rank if rank is None else int(rank)
        addr = address.strip()

        if addr == "LOGITS-ENTROPY":
            return True

        if addr.startswith("L-") or addr in ("NORM", "EMBED"):
            return r == 0

        if addr.startswith("A-"):
            parts = addr.split("-")
            try:
                head_idx = int(parts[-1])
                if head_idx < 1:
                    return False
                if head_idx > self.num_attention_heads:
                    owner_rank = (head_idx - 1) // self.heads_per_rank
                    return r == owner_rank
                owner_rank, _ = self._shard_index(
                    head_idx - 1, self.num_attention_heads, self.tp_size
                )
                return r == owner_rank
            except (ValueError, IndexError):
                return r == 0

        if addr.startswith("N-"):
            parts = addr.split("-")
            try:
                neuron_idx = int(parts[-1])
                if neuron_idx < 1:
                    return False
                if neuron_idx > self.num_neurons:
                    owner_rank = (neuron_idx - 1) // self.neurons_per_rank
                    return r == owner_rank
                owner_rank, _ = self._shard_index(
                    neuron_idx - 1, self.num_neurons, self.tp_size
                )
                return r == owner_rank
            except (ValueError, IndexError):
                return r == 0

        return r == 0

    def should_steer(self, address: str, rank: int | None = None) -> bool:
        """Check if rank should apply active steering for the given probe address.

        Rules:
        - Residual Stream (L-*, NORM, EMBED): Steering applied to ALL TP ranks identically
          to prevent forward divergence across ranks (returns True for all ranks).
        - Attention Heads (A-LL-HH) and MLP Neurons (N-LL-NNNN): Only owner rank applies steering.
        - LOGITS-ENTROPY: Entropy is a reduction metric, not a steerable activation (returns False).
        - Other / unrecognized addresses default to rank 0.
        """
        r = self.tp_rank if rank is None else int(rank)
        addr = address.strip()

        if addr.startswith("L-") or addr in ("NORM", "EMBED"):
            return True

        if addr == "LOGITS-ENTROPY":
            return False

        if addr.startswith("A-") or addr.startswith("N-"):
            return self.is_owner(addr, r)

        return self.is_owner(addr, r)

    def route_address(self, address: str, rank: int | None = None) -> ShardRoute:
        """Route probe address to owning rank and local shard index.

        Returns:
            ShardRoute tuple (owner_rank, local_index) with attribute access.
            - Attention Heads A-LL-HH: owner_rank = (HH - 1) // heads_per_rank,
              local_index = (HH - 1) % heads_per_rank.
            - MLP Neurons N-LL-NNNN: owner_rank = (NNNN - 1) // neurons_per_rank,
              local_index = (NNNN - 1) % neurons_per_rank.
            - Residual Stream (L-*, NORM, EMBED): owner_rank = 0, local_index = 0.
            - Other addresses: owner_rank = 0, local_index = 0.
        """
        addr = address.strip()

        if addr.startswith("A-"):
            parts = addr.split("-")
            try:
                head_idx = int(parts[-1])
                if head_idx < 1:
                    return ShardRoute(-1, -1, component="attn", should_steer_all_ranks=False)
                if head_idx > self.num_attention_heads:
                    owner = (head_idx - 1) // self.heads_per_rank
                    local = (head_idx - 1) % self.heads_per_rank
                    return ShardRoute(owner, local, component="attn", should_steer_all_ranks=False)
                owner, local = self._shard_index(
                    head_idx - 1, self.num_attention_heads, self.tp_size
                )
                return ShardRoute(owner, local, component="attn", should_steer_all_ranks=False)
            except (ValueError, IndexError):
                return ShardRoute(0, 0, component="attn", should_steer_all_ranks=False)

        if addr.startswith("N-"):
            parts = addr.split("-")
            try:
                neuron_idx = int(parts[-1])
                if neuron_idx < 1:
                    return ShardRoute(-1, -1, component="mlp", should_steer_all_ranks=False)
                if neuron_idx > self.num_neurons:
                    owner = (neuron_idx - 1) // self.neurons_per_rank
                    local = (neuron_idx - 1) % self.neurons_per_rank
                    return ShardRoute(owner, local, component="mlp", should_steer_all_ranks=False)
                owner, local = self._shard_index(
                    neuron_idx - 1, self.num_neurons, self.tp_size
                )
                return ShardRoute(owner, local, component="mlp", should_steer_all_ranks=False)
            except (ValueError, IndexError):
                return ShardRoute(0, 0, component="mlp", should_steer_all_ranks=False)

        is_residual = addr.startswith("L-") or addr in ("NORM", "EMBED")
        return ShardRoute(
            0,
            0,
            component="residual" if is_residual else "other",
            should_steer_all_ranks=is_residual,
        )

    @staticmethod
    def compute_local_logits_reduction(
        local_logits: torch.Tensor | np.ndarray,
    ) -> tuple[float, float, float] | tuple[Any, Any, Any]:
        """Compute 3-scalar reduction (m_r, s_r, w_r) on rank-local vocabulary slice.

        Mathematical specification:
            m_r = max_{j in V_r} x_j
            s_r = sum_{j in V_r} exp(x_j - m_r)
            w_r = sum_{j in V_r} (x_j - m_r) * exp(x_j - m_r)

        Args:
            local_logits: Tensor or ndarray of shape [vocab_slice] (1D),
                [batch, vocab_slice] (2D), or [batch, seq, vocab_slice] (3D).

        Returns:
            (m_r, s_r, w_r) tuple of floats for 1D single token,
            or LocalReduction tuple of arrays/tensors for batched inputs.
        """
        if torch is not None and isinstance(local_logits, torch.Tensor):
            t = local_logits.detach()
            m_r = torch.max(t, dim=-1).values
            is_neginf = torch.isneginf(t)
            safe_m = torch.where(torch.isneginf(m_r), torch.zeros_like(m_r), m_r)
            shift = torch.where(
                is_neginf, torch.full_like(t, -float("inf")), t - safe_m.unsqueeze(-1)
            )
            exp_shift = torch.where(is_neginf, torch.zeros_like(t), torch.exp(shift))
            safe_shift = torch.where(is_neginf, torch.zeros_like(t), shift)
            term = safe_shift * exp_shift
            s_r = torch.sum(exp_shift, dim=-1)
            w_r = torch.sum(term, dim=-1)

            if t.ndim == 1:
                return LocalReduction(
                    float(m_r.item()), float(s_r.item()), float(w_r.item())
                )
            return LocalReduction(m_r, s_r, w_r)

        x = np.asarray(local_logits)
        m_r = np.max(x, axis=-1)
        is_neginf = np.isneginf(x)
        safe_m = np.where(np.isneginf(m_r), 0.0, m_r)
        shift = np.where(is_neginf, -np.inf, x - np.expand_dims(safe_m, axis=-1))
        exp_shift = np.where(is_neginf, 0.0, np.exp(shift))
        safe_shift = np.where(is_neginf, 0.0, shift)
        term = safe_shift * exp_shift
        s_r = np.sum(exp_shift, axis=-1)
        w_r = np.sum(term, axis=-1)

        if x.ndim == 1:
            return LocalReduction(float(m_r), float(s_r), float(w_r))
        return LocalReduction(m_r, s_r, w_r)

    @staticmethod
    def reconstruct_global_entropy(
        scalars: list[tuple[float, float, float]] | np.ndarray | Any,
    ) -> float | np.ndarray:
        """Reconstruct exact global Shannon entropy (in nats) from rank reduction scalars.

        Mathematical reconstruction:
            M = max_r m_r
            S = sum_r s_r * exp(m_r - M)
            W = sum_r (w_r + (m_r - M) * s_r) * exp(m_r - M)
            H = ln(S) - W / S

        This formula is numerically stable via max-shift M and mathematically
        identical to full-vocab Shannon entropy H = -sum p_i ln p_i.

        Args:
            scalars: list of (m_r, s_r, w_r) tuples per rank, or structured
                reduction ndarray / Tensor of shape [num_ranks, ...] or [num_ranks, ..., 3].

        Returns:
            Global Shannon entropy as float (for 1D single token) or ndarray.
        """
        if torch is not None and isinstance(scalars, torch.Tensor):
            scalars = scalars.detach().cpu().numpy()

        if isinstance(scalars, (list, tuple)):
            if len(scalars) == 0:
                raise ValueError("scalars cannot be empty")
            first = scalars[0]
            if isinstance(first, (tuple, list)):
                if len(first) != 3:
                    raise ValueError(
                        f"Expected 3 reduction scalars (m_r, s_r, w_r), got len={len(first)}"
                    )
                m_list = [_to_numpy(x[0]) for x in scalars]
                s_list = [_to_numpy(x[1]) for x in scalars]
                w_list = [_to_numpy(x[2]) for x in scalars]
                m_arr = np.stack(m_list, axis=0)
                s_arr = np.stack(s_list, axis=0)
                w_arr = np.stack(w_list, axis=0)
            else:
                arr = np.asarray(scalars)
                if arr.shape[-1] == 3:
                    m_arr = arr[..., 0]
                    s_arr = arr[..., 1]
                    w_arr = arr[..., 2]
                elif arr.ndim >= 2 and arr.shape[1] == 3:
                    m_arr = arr[:, 0, ...]
                    s_arr = arr[:, 1, ...]
                    w_arr = arr[:, 2, ...]
                else:
                    raise ValueError(f"Unrecognized scalars array shape: {arr.shape}")
        elif isinstance(scalars, np.ndarray):
            if scalars.shape[-1] == 3:
                m_arr = scalars[..., 0]
                s_arr = scalars[..., 1]
                w_arr = scalars[..., 2]
            elif scalars.ndim >= 2 and scalars.shape[1] == 3:
                m_arr = scalars[:, 0, ...]
                s_arr = scalars[:, 1, ...]
                w_arr = scalars[:, 2, ...]
            else:
                raise ValueError(f"Unrecognized scalars array shape: {scalars.shape}")
        else:
            raise TypeError(f"Unsupported scalars type: {type(scalars)}")

        M = np.max(m_arr, axis=0)
        is_neginf_rank = np.isneginf(m_arr)
        safe_m = np.where(is_neginf_rank, 0.0, m_arr)
        safe_M = np.where(np.isneginf(M), 0.0, M)
        shift = np.where(is_neginf_rank, -np.inf, safe_m - safe_M)
        exp_shift = np.where(is_neginf_rank, 0.0, np.exp(shift))
        safe_shift = np.where(is_neginf_rank, 0.0, shift)

        S_rank = np.where(is_neginf_rank, 0.0, s_arr * exp_shift)
        W_rank = np.where(is_neginf_rank, 0.0, (w_arr + safe_shift * s_arr) * exp_shift)

        S = np.sum(S_rank, axis=0)
        W = np.sum(W_rank, axis=0)
        safe_S = np.maximum(S, 1e-30)
        H = np.where(np.isneginf(M), 0.0, np.log(safe_S) - W / safe_S)

        if H.ndim == 0:
            return float(H)
        return H


# Module-level aliases for direct function imports
compute_local_logits_reduction = TPShardRouter.compute_local_logits_reduction
reconstruct_global_entropy = TPShardRouter.reconstruct_global_entropy

__all__ = [
    "TPShardRouter",
    "ShardRoute",
    "LocalReduction",
    "compute_local_logits_reduction",
    "reconstruct_global_entropy",
]
