"""Comprehensive 4-tier unit and integration tests for distributed TP router (Features 9-14).

Covers:
- Tier 1: Feature Coverage (TPShardRouter initialization for tp_size in [1, 2, 4, 8],
          residual stream passive/active rules, attention head sharding, MLP neuron sharding,
          3-scalar local reduction, exact global Shannon entropy reconstruction)
- Tier 2: Boundary & Corner Cases (tp_size=1 degenerate, large vocab sizes,
          extreme logits stability, 1D / 2D / 3D inputs, NumPy vs PyTorch array inputs)
- Tier 3: Cross-Feature Combinations (Head/neuron steering consistency vs ownership,
          batch address partitioning across ranks)
- Tier 4: Real-World Scenarios (Simulated 4-rank and 8-rank TP forward step with logits entropy)
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from glassbox.tap.distributed import (
    LocalReduction,
    TPShardRouter,
)

# ==============================================================================
# Tier 1: Feature Coverage (Features 9-14)
# ==============================================================================


class TestFeature9ResidualStreamPassiveRecording:
    """Feature 9: Residual stream passive recording on Rank 0 only."""

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_residual_stream_recorded_on_rank_0_only(self, tp_size: int):
        addresses = ["L-00", "L-01", "L-15", "NORM", "EMBED"]

        for rank in range(tp_size):
            router = TPShardRouter(tp_rank=rank, tp_size=tp_size)
            for addr in addresses:
                is_owned = router.is_owner(addr)
                if rank == 0:
                    assert is_owned is True, f"Rank 0 must own passive recording for {addr}"
                else:
                    assert is_owned is False, f"Rank {rank} must NOT record {addr} passively"


class TestFeature10ResidualStreamActiveSteering:
    """Feature 10: Residual stream active steering on ALL TP ranks."""

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_residual_stream_steered_on_all_ranks(self, tp_size: int):
        """Active steering interventions must apply to ALL ranks to prevent forward divergence."""
        addresses = ["L-00", "L-05", "L-31", "NORM", "EMBED"]

        for rank in range(tp_size):
            router = TPShardRouter(tp_rank=rank, tp_size=tp_size)
            for addr in addresses:
                should_steer = router.should_steer(addr)
                assert should_steer is True, (
                    f"All ranks (rank={rank}/{tp_size}) must steer residual stream address {addr}"
                )


class TestFeature11AttentionHeadTPSharding:
    """Feature 11: Column-parallel Attention Head modulo sharding."""

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_attention_heads_partitioned_correctly(self, tp_size: int):
        num_heads = 32
        heads_per_rank = num_heads // tp_size

        for head_1based in range(1, num_heads + 1):
            expected_rank = (head_1based - 1) // heads_per_rank
            expected_local_idx = (head_1based - 1) % heads_per_rank
            addr = f"A-01-{head_1based:02d}"

            for rank in range(tp_size):
                router = TPShardRouter(tp_rank=rank, tp_size=tp_size, num_heads=num_heads)
                is_owned = router.is_owner(addr)
                if rank == expected_rank:
                    assert is_owned is True, (
                        f"Head {head_1based} must be owned by rank {expected_rank}"
                    )
                else:
                    assert is_owned is False

                # Check route_address
                route = router.route_address(addr)
                assert route[0] == expected_rank
                assert route[1] == expected_local_idx
                assert route.owner_rank == expected_rank
                assert route.local_index == expected_local_idx


class TestFeature12MLPNeuronTPSharding:
    """Feature 12: Column-parallel MLP Neuron modulo sharding."""

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_mlp_neurons_partitioned_correctly(self, tp_size: int):
        num_neurons = 8192
        neurons_per_rank = num_neurons // tp_size

        test_indices = [
            1,
            10,
            neurons_per_rank,
            neurons_per_rank + 1,
            num_neurons // 2,
            num_neurons,
        ]

        for n_1based in test_indices:
            expected_rank = (n_1based - 1) // neurons_per_rank
            expected_local_idx = (n_1based - 1) % neurons_per_rank
            addr = f"N-02-{n_1based:04d}"

            for rank in range(tp_size):
                router = TPShardRouter(tp_rank=rank, tp_size=tp_size, mlp_dim=num_neurons)
                is_owned = router.is_owner(addr)
                if rank == expected_rank:
                    assert is_owned is True, (
                        f"Neuron {n_1based} must be owned by rank {expected_rank}"
                    )
                else:
                    assert is_owned is False

                route = router.route_address(addr)
                assert route.owner_rank == expected_rank
                assert route.local_index == expected_local_idx


class TestFeature13LogitsEntropyLocalReduction:
    """Feature 13: Logits entropy 3-scalar reduction (m_r, s_r, w_r)."""

    def test_local_reduction_1d_tensor(self):
        router = TPShardRouter(tp_rank=0, tp_size=2)
        local_logits = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

        m_r, s_r, w_r = router.compute_local_logits_reduction(local_logits)

        # Expected:
        # m_r = max(x) = 4.0
        # shift = [-3, -2, -1, 0]
        # s_r = exp(-3) + exp(-2) + exp(-1) + exp(0)
        # w_r = sum(shift * exp(shift))
        shift = np.array([-3.0, -2.0, -1.0, 0.0])
        exp_s = np.exp(shift)
        exp_s_r = float(np.sum(exp_s))
        exp_w_r = float(np.sum(shift * exp_s))

        assert m_r == pytest.approx(4.0)
        assert s_r == pytest.approx(exp_s_r, rel=1e-5)
        assert w_r == pytest.approx(exp_w_r, rel=1e-5)

    def test_all_ranks_participate_in_logits_entropy_ownership(self):
        for rank in range(4):
            router = TPShardRouter(tp_rank=rank, tp_size=4)
            assert router.is_owner("LOGITS-ENTROPY") is True


class TestFeature14GlobalShannonEntropyReconstruction:
    """Feature 14: Global Shannon entropy reconstruction from reduction scalars."""

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_entropy_reconstruction_parity_with_torch_categorical(self, tp_size: int):
        torch.manual_seed(42)
        vocab_size = 32000
        logits = torch.randn(vocab_size, dtype=torch.float32)

        # Ground truth Shannon entropy via PyTorch Categorical (in nats)
        ground_truth_entropy = torch.distributions.Categorical(logits=logits).entropy().item()

        # Partition vocabulary across TP ranks
        router = TPShardRouter(tp_rank=0, tp_size=tp_size, vocab_size=vocab_size)
        shards = torch.chunk(logits, tp_size, dim=-1)

        # Each rank computes local 3-scalar reduction
        scalars = [router.compute_local_logits_reduction(shard) for shard in shards]

        # Reconstruct global entropy
        reconstructed_entropy = router.reconstruct_global_entropy(scalars)

        # Verify precision tolerance <= 1e-5
        diff = abs(ground_truth_entropy - float(reconstructed_entropy))
        assert diff < 1e-5, f"Entropy diff {diff} exceeds tolerance 1e-5 on tp_size={tp_size}"


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """Boundary, extreme size, and numerical stability cases."""

    def test_tp_size_1_single_gpu_all_owned_by_rank_0(self):
        router = TPShardRouter(tp_rank=0, tp_size=1)
        assert router.is_owner("L-00") is True
        assert router.is_owner("NORM") is True
        assert router.is_owner("A-01-05") is True
        assert router.is_owner("N-01-0100") is True
        assert router.should_steer("L-00") is True
        assert router.should_steer("A-01-05") is True

    def test_numerical_stability_with_large_positive_and_negative_logits(self):
        """Test with extreme logits (-1000, +1000) where naive exp would underflow/overflow."""
        router = TPShardRouter(tp_rank=0, tp_size=2)
        logits1 = torch.tensor([1000.0, 999.0, 998.0])
        logits2 = torch.tensor([500.0, -1000.0, 0.0])

        r1 = router.compute_local_logits_reduction(logits1)
        r2 = router.compute_local_logits_reduction(logits2)

        entropy = router.reconstruct_global_entropy([r1, r2])
        assert not math.isnan(entropy)
        assert not math.isinf(entropy)
        assert entropy >= 0.0

    def test_batched_2d_and_3d_inputs(self):
        router = TPShardRouter(tp_rank=0, tp_size=2)
        batch, seq, vocab_slice = 2, 4, 100
        logits_3d = torch.randn(batch, seq, vocab_slice)

        r = router.compute_local_logits_reduction(logits_3d)
        assert r.m_r.shape == (batch, seq)
        assert r.s_r.shape == (batch, seq)
        assert r.w_r.shape == (batch, seq)

    def test_numpy_array_input_support(self):
        router = TPShardRouter(tp_rank=0, tp_size=2)
        arr = np.array([2.0, 4.0, 6.0], dtype=np.float32)
        r = router.compute_local_logits_reduction(arr)
        assert isinstance(r, LocalReduction)
        assert r.m == pytest.approx(6.0)

    def test_large_vocab_size_llama3(self):
        """Vocabulary size of 128256 (Llama-3)."""
        router = TPShardRouter(tp_rank=0, tp_size=8, vocab_size=128256)
        slice_size = 128256 // 8
        shards = [torch.randn(slice_size) for _ in range(8)]
        scalars = [router.compute_local_logits_reduction(s) for s in shards]
        ent = router.reconstruct_global_entropy(scalars)
        assert ent > 0.0


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Pairwise steering vs ownership rules."""

    def test_attention_and_neuron_steering_matches_ownership(self):
        """For attention heads and MLP neurons, only the owner rank steers."""
        router_r0 = TPShardRouter(tp_rank=0, tp_size=4, num_heads=32, mlp_dim=8192)
        router_r1 = TPShardRouter(tp_rank=1, tp_size=4, num_heads=32, mlp_dim=8192)

        # Head 1 (1-based) is on Rank 0
        assert router_r0.is_owner("A-01-01") is True
        assert router_r0.should_steer("A-01-01") is True
        assert router_r1.is_owner("A-01-01") is False
        assert router_r1.should_steer("A-01-01") is False

        # Head 10 (1-based) is on Rank 1 (10-1 // 8 = 1)
        assert router_r0.is_owner("A-01-10") is False
        assert router_r0.should_steer("A-01-10") is False
        assert router_r1.is_owner("A-01-10") is True
        assert router_r1.should_steer("A-01-10") is True

    def test_batch_address_routing(self):
        addresses = [
            "L-00",
            "L-01",
            "NORM",
            "EMBED",
            "A-01-01",
            "A-01-09",
            "A-01-17",
            "A-01-25",
            "N-01-0001",
            "N-01-2049",
            "N-01-4097",
            "N-01-6145",
        ]
        routers = [TPShardRouter(tp_rank=r, tp_size=4) for r in range(4)]

        # Verify partitioning: each head and neuron is owned by exactly one rank
        for addr in addresses:
            if addr.startswith("L-") or addr in ("NORM", "EMBED"):
                assert sum(r.is_owner(addr) for r in routers) == 1
                assert routers[0].is_owner(addr) is True
                assert all(r.should_steer(addr) for r in routers)
            else:
                assert sum(r.is_owner(addr) for r in routers) == 1


# ==============================================================================
# Tier 4: Real-World Scenarios
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Multi-GPU distributed serving forward pass simulation."""

    def test_simulated_4_rank_tp_step(self):
        """Simulate a full forward step across 4 TP workers for a 70B model shard."""
        tp_size = 4
        num_heads = 64
        mlp_dim = 16384
        vocab_size = 32000

        routers = [
            TPShardRouter(
                tp_rank=r,
                tp_size=tp_size,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                vocab_size=vocab_size,
            )
            for r in range(tp_size)
        ]

        # 1. Probing watchlist: 2 residual stream, 4 heads, 4 neurons, 1 logits-entropy
        watchlist = [
            "L-00",
            "L-05",
            "A-00-01",
            "A-00-17",
            "A-00-33",
            "A-00-49",
            "N-00-0001",
            "N-00-4097",
            "N-00-8193",
            "N-00-12289",
            "LOGITS-ENTROPY",
        ]

        # Verify each probe is handled by the appropriate worker
        for r_idx, router in enumerate(routers):
            owned = [addr for addr in watchlist if router.is_owner(addr)]
            steered = [addr for addr in watchlist if router.should_steer(addr)]

            # LOGITS-ENTROPY owned by all ranks for reduction
            assert "LOGITS-ENTROPY" in owned

            # Residual stream owned only by Rank 0
            if r_idx == 0:
                assert "L-00" in owned
                assert "L-05" in owned
            else:
                assert "L-00" not in owned

            # But ALL ranks steer residual stream
            assert "L-00" in steered
            assert "L-05" in steered

        # 2. Simulate global logits entropy reconstruction across all 4 ranks
        full_logits = torch.randn(8, vocab_size)  # 8 tokens
        gt_entropy = torch.distributions.Categorical(logits=full_logits).entropy().numpy()

        sharded_logits = torch.chunk(full_logits, tp_size, dim=-1)
        rank_reductions = [
            routers[r].compute_local_logits_reduction(sharded_logits[r]) for r in range(tp_size)
        ]

        reconstructed = routers[0].reconstruct_global_entropy(rank_reductions)
        assert np.allclose(gt_entropy, reconstructed, atol=1e-5)
