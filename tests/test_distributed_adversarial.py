"""Adversarial stress-test harness for distributed sharding and logits entropy reduction.

Evaluates glassbox/tap/distributed.py against:
1. Non-power-of-2 TP sizes (3, 5, 7, 11, 13) and asymmetric vocab chunk allocations.
2. Extreme numerical logits (+5000, -5000, mixed infinite-scale logits, all equal logits).
3. Exact double-precision Shannon entropy reconstruction oracle comparison.
4. Invalid address formats, edge cases, out-of-range heads/neurons.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from glassbox.tap.distributed import (
    ShardRoute,
    TPShardRouter,
)

# ==============================================================================
# Helper Oracle: Exact Double-Precision Shannon Entropy Ground Truth
# ==============================================================================


def double_precision_entropy_oracle(logits: np.ndarray | torch.Tensor) -> float:
    """Exact double-precision reference implementation of Shannon entropy (in nats).

    H = - sum_{p > 0} p_i * ln(p_i)
    Uses max-subtraction in float64 for stability.
    Gracefully handles -inf logits (p_i = 0, contributing 0 to entropy).
    """
    if isinstance(logits, torch.Tensor):
        x = logits.detach().cpu().to(torch.float64).numpy()
    else:
        x = np.asarray(logits, dtype=np.float64)

    # Flatten if 1D single token
    if x.ndim > 1:
        x = x.reshape(-1)

    # Mask out -inf values
    valid_mask = ~np.isneginf(x) & ~np.isnan(x)
    if not np.any(valid_mask):
        return 0.0

    valid_x = x[valid_mask]
    m = np.max(valid_x)
    shift = valid_x - m
    exp_shift = np.exp(shift)
    sum_exp = np.sum(exp_shift)

    p = exp_shift / sum_exp
    # Avoid log(0) for p == 0
    p_nonzero = p[p > 0]
    entropy = -np.sum(p_nonzero * np.log(p_nonzero))
    return float(entropy)


# ==============================================================================
# 1. Non-Power-of-2 TP Sizes & Asymmetric Allocations
# ==============================================================================


class TestAdversarialNonPowerOfTwoAndAsymmetric:
    """Adversarial tests for non-power-of-2 tp_sizes and asymmetric vocabulary slices."""

    @pytest.mark.parametrize("tp_size", [3, 5, 7, 11, 13])
    def test_asymmetric_vocab_chunk_entropy_reconstruction(self, tp_size: int):
        """Asymmetric vocab chunk allocations across non-power-of-2 ranks."""
        vocab_size = 32000
        torch.manual_seed(42 + tp_size)
        logits = torch.randn(vocab_size, dtype=torch.float64)

        # Generate asymmetric cut points across tp_size ranks
        # E.g. uneven splits that sum to vocab_size
        base = vocab_size // tp_size
        remainder = vocab_size % tp_size
        slice_sizes = [base + (1 if r < remainder else 0) for r in range(tp_size)]
        assert sum(slice_sizes) == vocab_size

        splits = []
        curr = 0
        for sz in slice_sizes:
            splits.append(logits[curr : curr + sz])
            curr += sz

        # Each rank computes local reduction
        router = TPShardRouter(tp_rank=0, tp_size=tp_size, vocab_size=vocab_size)
        scalars = [router.compute_local_logits_reduction(shard) for shard in splits]

        recon_entropy = router.reconstruct_global_entropy(scalars)
        gt_entropy = double_precision_entropy_oracle(logits)

        diff = abs(gt_entropy - recon_entropy)
        assert diff < 1e-12, (
            f"Asymmetric vocab slices failed on tp_size={tp_size}: diff={diff:.6e}"
        )

    @pytest.mark.parametrize("tp_size", [3, 5, 7])
    def test_non_divisible_num_heads_sharding_behavior(self, tp_size: int):
        """Verify behavior when num_heads is not divisible by tp_size."""
        num_heads = 32  # 32 is not divisible by 3, 5, 7
        routers = [
            TPShardRouter(tp_rank=r, tp_size=tp_size, num_heads=num_heads)
            for r in range(tp_size)
        ]

        out_of_bounds_heads = []
        for h in range(1, num_heads + 1):
            addr = f"A-01-{h:02d}"
            route = routers[0].route_address(addr)
            if route.owner_rank >= tp_size or route.owner_rank < 0:
                out_of_bounds_heads.append((h, route.owner_rank))

        # Balanced sharding partitions all heads into valid ranks 0..tp_size-1
        assert len(out_of_bounds_heads) == 0, (
            f"Unexpected out-of-bounds heads for tp_size={tp_size}: {out_of_bounds_heads}"
        )
        for h in range(1, num_heads + 1):
            addr = f"A-01-{h:02d}"
            owners = [r for r, router in enumerate(routers) if router.is_owner(addr)]
            assert len(owners) == 1, f"Head {addr} should have exactly one owner"

    @pytest.mark.parametrize("tp_size", [3, 5, 7])
    def test_divisible_num_heads_sharding_clean(self, tp_size: int):
        """When num_heads is divisible by tp_size, all heads must be cleanly partitioned."""
        # Choose num_heads divisible by tp_size
        num_heads = tp_size * 8
        routers = [
            TPShardRouter(tp_rank=r, tp_size=tp_size, num_heads=num_heads)
            for r in range(tp_size)
        ]

        for h in range(1, num_heads + 1):
            addr = f"A-01-{h:02d}"
            route = routers[0].route_address(addr)
            assert 0 <= route.owner_rank < tp_size
            assert 0 <= route.local_index < 8
            # Exactly one rank owns it
            owners = [r for r, router in enumerate(routers) if router.is_owner(addr)]
            assert len(owners) == 1
            assert owners[0] == route.owner_rank


# ==============================================================================
# 2. Extreme Numerical Logits
# ==============================================================================


class TestAdversarialExtremeNumericalLogits:
    """Stress tests on extreme logits values (+5000, -5000, equal, infinite)."""

    def test_all_equal_positive_large_logits(self):
        """All logits equal to +5000.0: entropy must exactly equal ln(V)."""
        V = 1000
        router = TPShardRouter(tp_rank=0, tp_size=4)
        logits = torch.full((V,), 5000.0, dtype=torch.float64)

        shards = torch.chunk(logits, 4)
        scalars = [router.compute_local_logits_reduction(s) for s in shards]
        recon = router.reconstruct_global_entropy(scalars)

        expected = math.log(V)
        assert abs(recon - expected) < 1e-12

    def test_all_equal_negative_large_logits(self):
        """All logits equal to -5000.0: entropy must exactly equal ln(V)."""
        V = 1000
        router = TPShardRouter(tp_rank=0, tp_size=4)
        logits = torch.full((V,), -5000.0, dtype=torch.float64)

        shards = torch.chunk(logits, 4)
        scalars = [router.compute_local_logits_reduction(s) for s in shards]
        recon = router.reconstruct_global_entropy(scalars)

        expected = math.log(V)
        assert abs(recon - expected) < 1e-12

    def test_extreme_cross_rank_dynamic_range(self):
        """Rank 0 has logits around +5000, Rank 1 has logits around -5000."""
        router = TPShardRouter(tp_rank=0, tp_size=2)
        s0 = torch.randn(500, dtype=torch.float64) + 5000.0
        s1 = torch.randn(500, dtype=torch.float64) - 5000.0
        full = torch.cat([s0, s1])

        r0 = router.compute_local_logits_reduction(s0)
        r1 = router.compute_local_logits_reduction(s1)

        recon = router.reconstruct_global_entropy([r0, r1])
        gt = double_precision_entropy_oracle(full)

        assert not math.isnan(recon)
        assert not math.isinf(recon)
        assert abs(recon - gt) < 1e-12

    def test_one_hot_extreme_logit(self):
        """One logit is +5000.0, all other logits are -5000.0 (entropy must be 0.0)."""
        router = TPShardRouter(tp_rank=0, tp_size=2)
        logits = torch.full((1000,), -5000.0, dtype=torch.float64)
        logits[42] = 5000.0

        shards = torch.chunk(logits, 2)
        scalars = [router.compute_local_logits_reduction(s) for s in shards]
        recon = router.reconstruct_global_entropy(scalars)

        assert recon == pytest.approx(0.0, abs=1e-12)

    def test_mixed_negative_infinity_logits_nan_failure(self):
        """Check behavior when logits contain -inf (masked tokens)."""
        router = TPShardRouter(tp_rank=0, tp_size=2)
        # Logits with -inf (e.g. padding/masking)
        logits_with_inf = torch.tensor([-float("inf"), 1.0, 2.0, -float("inf"), 3.0])

        r = router.compute_local_logits_reduction(logits_with_inf)
        # Safe masking ensures w_r is not NaN
        is_w_nan = (
            math.isnan(r.w_r)
            if isinstance(r.w_r, float)
            else torch.isnan(r.w_r).any().item()
        )
        assert not is_w_nan, "Reduction w_r must not be NaN"
        recon = router.reconstruct_global_entropy([r])
        gt = double_precision_entropy_oracle(logits_with_inf)
        assert abs(recon - gt) < 1e-6

    def test_all_negative_infinity_shard_failure(self):
        """Check behavior when an entire shard is -inf (unused vocab slots)."""
        router = TPShardRouter(tp_rank=0, tp_size=2)
        shard_valid = torch.tensor([1.0, 2.0, 3.0])
        shard_masked = torch.tensor([-float("inf"), -float("inf"), -float("inf")])

        r0 = router.compute_local_logits_reduction(shard_valid)
        r1 = router.compute_local_logits_reduction(shard_masked)

        recon = router.reconstruct_global_entropy([r0, r1])
        assert not math.isnan(recon), "Entropy must not be NaN when a shard has all -inf"
        expected = double_precision_entropy_oracle(shard_valid)
        assert recon == pytest.approx(expected, abs=1e-6)


# ==============================================================================
# 3. Exact Double-Precision Shannon Entropy Oracle Comparison
# ==============================================================================


class TestAdversarialDoublePrecisionOracle:
    """Rigorously verify reconstruction accuracy against float64 mathematical ground truth."""

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_random_logits_double_precision_parity(self, tp_size: int):
        """Verify parity against float64 oracle across 32,000 vocab tokens."""
        torch.manual_seed(1337 + tp_size)
        vocab_size = 32000
        logits = torch.randn(vocab_size, dtype=torch.float64)

        router = TPShardRouter(tp_rank=0, tp_size=tp_size, vocab_size=vocab_size)
        shards = torch.chunk(logits, tp_size)
        scalars = [router.compute_local_logits_reduction(s) for s in shards]

        recon = router.reconstruct_global_entropy(scalars)
        gt = double_precision_entropy_oracle(logits)

        diff = abs(recon - gt)
        assert diff < 1e-12, f"Discrepancy {diff:.6e} on tp_size={tp_size}"

    @pytest.mark.parametrize("temperature", [0.01, 0.1, 1.0, 10.0, 100.0])
    def test_temperature_scaling_accuracy(self, temperature: float):
        """Verify across different temperature regimes (near one-hot to near-uniform)."""
        torch.manual_seed(999)
        vocab_size = 5000
        raw_logits = torch.randn(vocab_size, dtype=torch.float64)
        scaled_logits = raw_logits / temperature

        router = TPShardRouter(tp_rank=0, tp_size=4, vocab_size=vocab_size)
        shards = torch.chunk(scaled_logits, 4)
        scalars = [router.compute_local_logits_reduction(s) for s in shards]

        recon = router.reconstruct_global_entropy(scalars)
        gt = double_precision_entropy_oracle(scaled_logits)

        diff = abs(recon - gt)
        assert diff < 1e-11, f"Diff {diff:.6e} at temp={temperature}"

    def test_batched_multi_rank_entropy_shape_and_values(self):
        """Verify batched 2D and 3D inputs match oracle for all tokens."""
        batch, seq, vocab = 3, 5, 1000
        tp_size = 4
        torch.manual_seed(2024)
        logits_3d = torch.randn(batch, seq, vocab, dtype=torch.float64)

        router = TPShardRouter(tp_rank=0, tp_size=tp_size, vocab_size=vocab)
        shards = torch.chunk(logits_3d, tp_size, dim=-1)
        scalars = [router.compute_local_logits_reduction(s) for s in shards]

        recon_3d = router.reconstruct_global_entropy(scalars)
        assert recon_3d.shape == (batch, seq)

        # Check each (b, s) token against oracle
        for b in range(batch):
            for s in range(seq):
                gt = double_precision_entropy_oracle(logits_3d[b, s])
                val = float(recon_3d[b, s])
                assert abs(val - gt) < 1e-12


# ==============================================================================
# 4. Invalid Address Formats, Edge Cases, Out-of-Range Bounds
# ==============================================================================


class TestAdversarialAddressValidationAndEdgeCases:
    """Probe address edge cases, malformed strings, and out-of-range bounds."""

    def test_zero_and_negative_head_indices(self):
        """0-based or negative head index (e.g. A-01-00)."""
        router = TPShardRouter(tp_rank=0, tp_size=4, num_heads=32)
        route_zero = router.route_address("A-01-00")
        # In Python: (0 - 1) // 8 = -1 // 8 = -1
        assert route_zero.owner_rank == -1
        assert router.is_owner("A-01-00") is False

    def test_out_of_range_head_index(self):
        """Head index far exceeding num_heads (e.g. A-01-999)."""
        router = TPShardRouter(tp_rank=0, tp_size=4, num_heads=32)
        route_out = router.route_address("A-01-999")
        assert route_out.owner_rank == (999 - 1) // 8
        assert route_out.owner_rank >= 4
        # None of the 4 valid ranks own this head
        for r in range(4):
            assert router.is_owner("A-01-999", rank=r) is False

    def test_zero_and_negative_neuron_indices(self):
        """0-based neuron index (e.g. N-01-0000)."""
        router = TPShardRouter(tp_rank=0, tp_size=4, mlp_dim=8192)
        route_zero = router.route_address("N-01-0000")
        assert route_zero.owner_rank == -1
        assert router.is_owner("N-01-0000") is False

    def test_out_of_range_neuron_index(self):
        """Neuron index far exceeding mlp_dim (e.g. N-01-99999)."""
        router = TPShardRouter(tp_rank=0, tp_size=4, mlp_dim=8192)
        route_out = router.route_address("N-01-99999")
        assert route_out.owner_rank == (99999 - 1) // 2048
        assert route_out.owner_rank >= 4

    @pytest.mark.parametrize(
        "malformed_addr",
        [
            "",
            "   ",
            "A-",
            "N-",
            "L-",
            "A-01",
            "N-01",
            "MALFORMED_PROBE",
            "unknown:layer",
        ],
    )
    def test_malformed_address_fallback_safety(self, malformed_addr: str):
        """Malformed addresses must not crash and fallback to rank 0."""
        router = TPShardRouter(tp_rank=0, tp_size=4)
        route = router.route_address(malformed_addr)
        assert isinstance(route, ShardRoute)
        assert route.owner_rank == 0
        assert route.local_index == 0

        # Should not raise exception
        is_owned = router.is_owner(malformed_addr)
        assert isinstance(is_owned, bool)

        steer = router.should_steer(malformed_addr)
        assert isinstance(steer, bool)
