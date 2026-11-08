"""Comprehensive 4-tier unit and integration tests for CUDA Graph safe wiretap (Features 15-18).

Covers:
- Tier 1: Feature Coverage (glassbox::tap registration, ctl device tensor semantics,
          staging slab slot/col_offset placement, fake tensor / Dynamo tracing)
- Tier 2: Boundary & Corner Cases (ctl[0]=0 kill switch, ctl[1] actual tokens clamp,
          negative slot_idx, empty features, out-of-range col_offset, 3D batch activations)
- Tier 3: Cross-Feature Combinations (NumPy parity vs PyTorch fallback vs torch.ops.glassbox.tap,
          glassbox_async_d2h_copy D2H transfer with slots and rows)
- Tier 4: Real-World Scenarios (Simulated CUDA Graph capture/replay with dynamic ctl updates,
          multi-slot circular staging slab buffer gather)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from glassbox.csrc_fallback import (
    glassbox_async_d2h_copy,
    tap,
    tap_fallback,
    tap_numpy,
)

# ==============================================================================
# Tier 1: Feature Coverage (Features 15-18)
# ==============================================================================


class TestFeature15CustomOpRegistration:
    """Feature 15: PyTorch Custom Op registration of glassbox::tap."""

    def test_custom_op_is_registered_in_torch_ops(self):
        assert hasattr(torch.ops, "glassbox"), "torch.ops.glassbox namespace must exist"
        assert hasattr(torch.ops.glassbox, "tap"), "torch.ops.glassbox.tap op must be registered"

    def test_callable_via_torch_ops_glassbox_tap(self):
        K_slots, max_tokens, slab_features = 4, 16, 8
        activations = torch.randn(10, 64, dtype=torch.float32)
        feature_indices = torch.tensor([0, 2, 4, 6], dtype=torch.long)
        # ctl: [kill_switch=1, actual_num_tokens=5, slot_idx=1]
        ctl = torch.tensor([1, 5, 1], dtype=torch.long)
        staging_slab = torch.zeros(K_slots, max_tokens, slab_features, dtype=torch.float32)

        torch.ops.glassbox.tap(activations, feature_indices, ctl, staging_slab, 0)

        # Check that rows 0..4 in slot 1 contain the gathered features
        expected = activations[:5, feature_indices]
        assert torch.allclose(staging_slab[1, :5, 0:4], expected)
        # Check that other slots remain zero
        assert torch.all(staging_slab[0] == 0)
        assert torch.all(staging_slab[2:] == 0)


class TestFeature16DeviceControlTensorSemantics:
    """Feature 16: Device control tensor `ctl` semantics."""

    def test_kill_switch_ctl0_zero_is_noop(self):
        """When ctl[0] == 0 (kill switch active), staging slab is completely unmodified."""
        activations = torch.randn(8, 32, dtype=torch.float32)
        feature_indices = torch.tensor([1, 3, 5], dtype=torch.long)
        ctl = torch.tensor([0, 8, 0], dtype=torch.long)  # kill switch active
        staging_slab = torch.zeros(2, 16, 16, dtype=torch.float32)

        tap(activations, feature_indices, ctl, staging_slab, 0)
        assert torch.all(staging_slab == 0), "Staging slab must remain unmodified when ctl[0] == 0"

    def test_actual_tokens_ctl1_masks_padding(self):
        """ctl[1] governs how many rows are written; padding tokens beyond are ignored."""
        total_tokens = 16
        actual_tokens = 6
        activations = torch.randn(total_tokens, 32, dtype=torch.float32)
        feature_indices = torch.tensor([2, 4, 6, 8], dtype=torch.long)
        ctl = torch.tensor([1, actual_tokens, 0], dtype=torch.long)
        staging_slab = torch.zeros(2, total_tokens, 8, dtype=torch.float32)

        tap(activations, feature_indices, ctl, staging_slab, 0)

        # First 6 tokens should match gathered activations
        assert torch.allclose(staging_slab[0, :6, :4], activations[:6, feature_indices])
        # Remaining tokens (6..15) in slot 0 must remain zero
        assert torch.all(staging_slab[0, 6:, :] == 0)

    def test_slot_index_ctl2_writes_to_correct_slot(self):
        """ctl[2] selects which slot in the staging slab is populated."""
        activations = torch.randn(8, 16, dtype=torch.float32)
        feature_indices = torch.tensor([0, 1], dtype=torch.long)
        staging_slab = torch.zeros(4, 8, 4, dtype=torch.float32)

        for target_slot in range(4):
            ctl = torch.tensor([1, 4, target_slot], dtype=torch.long)
            tap(activations, feature_indices, ctl, staging_slab, 0)
            assert torch.allclose(
                staging_slab[target_slot, :4, :2], activations[:4, feature_indices]
            )

    def test_col_offset_placement(self):
        """col_offset places gathered columns at specific column indices in the staging slab."""
        activations = torch.randn(4, 16, dtype=torch.float32)
        feature_indices = torch.tensor([5, 10], dtype=torch.long)
        col_offset = 3
        ctl = torch.tensor([1, 4, 0], dtype=torch.long)
        staging_slab = torch.zeros(1, 4, 8, dtype=torch.float32)

        tap(activations, feature_indices, ctl, staging_slab, col_offset)

        # Columns 0..2 should be 0
        assert torch.all(staging_slab[0, :, :3] == 0)
        # Columns 3..4 should be gathered features
        assert torch.allclose(staging_slab[0, :, 3:5], activations[:, feature_indices])
        # Columns 5..7 should be 0
        assert torch.all(staging_slab[0, :, 5:] == 0)


class TestFeature17OutOfGraphStreamCopy:
    """Feature 17: Out-of-graph D2H copy utility."""

    def test_async_d2h_copy_cpu_fallback(self):
        """Verify glassbox_async_d2h_copy functions correctly on CPU buffers."""
        src_slab = torch.randn(4, 16, 8, dtype=torch.float32)
        dst_pinned = torch.zeros(4, 16, 8, dtype=torch.float32)

        # Copy slot 2 with 10 actual tokens
        glassbox_async_d2h_copy(
            src=src_slab,
            dst=dst_pinned,
            slot_idx=2,
            actual_num_tokens=10,
        )
        assert torch.allclose(dst_pinned[2, :10], src_slab[2, :10])
        assert torch.all(dst_pinned[2, 10:] == 0)
        assert torch.all(dst_pinned[0] == 0)
        assert torch.all(dst_pinned[1] == 0)
        assert torch.all(dst_pinned[3] == 0)

    def test_async_d2h_copy_full_tensor(self):
        src = torch.randn(10, 10)
        dst = torch.zeros(10, 10)
        glassbox_async_d2h_copy(src, dst)
        assert torch.allclose(src, dst)


class TestFeature18PurePythonAndNumPyFallbacks:
    """Feature 18: Pure-Python and NumPy fallback implementations with 100% parity."""

    def test_numpy_fallback_parity(self):
        act_np = np.random.randn(8, 32).astype(np.float32)
        feat_np = np.array([1, 5, 10, 15], dtype=np.int64)
        ctl_np = np.array([1, 6, 1], dtype=np.int64)
        slab_np = np.zeros((3, 8, 8), dtype=np.float32)

        tap_numpy(act_np, feat_np, ctl_np, slab_np, col_offset=2)

        expected = act_np[:6, feat_np]
        assert np.allclose(slab_np[1, :6, 2:6], expected)
        assert np.all(slab_np[1, :6, :2] == 0)
        assert np.all(slab_np[1, 6:, :] == 0)
        assert np.all(slab_np[0] == 0)
        assert np.all(slab_np[2] == 0)

    def test_torch_fallback_matches_numpy_fallback(self):
        act = np.random.randn(12, 24).astype(np.float32)
        feat = np.array([0, 3, 7, 11], dtype=np.int64)
        ctl = np.array([1, 8, 0], dtype=np.int64)

        slab_torch = torch.zeros(2, 12, 10, dtype=torch.float32)
        slab_np = np.zeros((2, 12, 10), dtype=np.float32)

        tap_fallback(
            torch.from_numpy(act), torch.from_numpy(feat), torch.from_numpy(ctl), slab_torch, 1
        )
        tap_numpy(act, feat, ctl, slab_np, 1)

        assert np.allclose(slab_torch.numpy(), slab_np)


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """Boundary and corner cases for graph tap operator."""

    def test_zero_actual_tokens_is_noop(self):
        activations = torch.randn(8, 16)
        features = torch.tensor([0, 1])
        ctl = torch.tensor([1, 0, 0])  # actual_tokens == 0
        slab = torch.zeros(2, 8, 4)
        tap(activations, features, ctl, slab, 0)
        assert torch.all(slab == 0)

    def test_negative_actual_tokens_is_noop(self):
        activations = torch.randn(8, 16)
        features = torch.tensor([0, 1])
        ctl = torch.tensor([1, -5, 0])  # negative actual_tokens
        slab = torch.zeros(2, 8, 4)
        tap(activations, features, ctl, slab, 0)
        assert torch.all(slab == 0)

    def test_actual_tokens_exceeds_activations_rows_clamps_safely(self):
        """If ctl[1] > activations rows, clamps safely to activations.shape[0]."""
        activations = torch.randn(5, 16)
        features = torch.tensor([0, 1])
        ctl = torch.tensor([1, 100, 0])  # requests 100 tokens, only 5 available
        slab = torch.zeros(2, 10, 4)
        tap(activations, features, ctl, slab, 0)
        assert torch.allclose(slab[0, :5, :2], activations[:, features])
        assert torch.all(slab[0, 5:] == 0)

    def test_invalid_slot_index_raises_index_error(self):
        activations = torch.randn(4, 8)
        features = torch.tensor([0])
        slab = torch.zeros(2, 4, 2)  # only 2 slots (0 and 1)

        # Slot 5 is out of bounds
        ctl = torch.tensor([1, 4, 5])
        with pytest.raises(IndexError):
            tap(activations, features, ctl, slab, 0)

        # Negative slot is out of bounds
        ctl_neg = torch.tensor([1, 4, -1])
        with pytest.raises(IndexError):
            tap(activations, features, ctl_neg, slab, 0)

    def test_empty_feature_indices_is_noop(self):
        activations = torch.randn(4, 8)
        features = torch.empty(0, dtype=torch.long)
        ctl = torch.tensor([1, 4, 0])
        slab = torch.zeros(1, 4, 4)
        tap(activations, features, ctl, slab, 0)
        assert torch.all(slab == 0)

    def test_col_offset_near_boundary_truncates_safely(self):
        """If col_offset + num_features > slab columns, truncates without buffer overflow."""
        activations = torch.randn(4, 16)
        features = torch.tensor([1, 2, 3, 4, 5])  # 5 features
        ctl = torch.tensor([1, 4, 0])
        slab = torch.zeros(1, 4, 6)  # 6 cols total, col_offset=4 -> only 2 cols fit

        tap(activations, features, ctl, slab, col_offset=4)
        # Should copy features[0] and features[1] to cols 4 and 5
        assert torch.allclose(slab[0, :, 4:6], activations[:, features[:2]])

    def test_3d_activations_batch_seq_flattening(self):
        """Activations of shape [batch, seq, hidden] should flatten to [batch * seq, hidden]."""
        batch, seq, hidden = 2, 4, 16
        activations_3d = torch.randn(batch, seq, hidden)
        features = torch.tensor([2, 5])
        ctl = torch.tensor([1, 6, 0])  # gather 6 tokens
        slab = torch.zeros(1, 8, 4)

        tap(activations_3d, features, ctl, slab, 0)
        flat = activations_3d.reshape(-1, hidden)
        assert torch.allclose(slab[0, :6, :2], flat[:6, features])


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Pairwise cross-feature interactions."""

    def test_dtype_coercion_to_staging_slab_precision(self):
        """Feature gather handles fp32 activations into fp16 slab seamlessly."""
        act_f32 = torch.randn(4, 16, dtype=torch.float32)
        features = torch.tensor([1, 3])
        ctl = torch.tensor([1, 4, 0])
        slab_f16 = torch.zeros(1, 4, 4, dtype=torch.float16)

        tap(act_f32, features, ctl, slab_f16, 0)
        assert slab_f16.dtype == torch.float16
        assert torch.allclose(slab_f16[0, :, :2], act_f32[:, features].half(), atol=1e-3)

    def test_fake_tensor_shape_inference_no_graph_breaks(self):
        """Verify fake tensor registration (_tap_fake) works with torch.library."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            fake_act = torch.empty(8, 32)
            fake_feat = torch.empty(4, dtype=torch.long)
            fake_ctl = torch.empty(3, dtype=torch.long)
            fake_slab = torch.empty(2, 16, 8)

            # Invoking torch.ops.glassbox.tap on fake tensors should execute without graph break
            torch.ops.glassbox.tap(fake_act, fake_feat, fake_ctl, fake_slab, 0)
            assert fake_slab.shape == (2, 16, 8)


# ==============================================================================
# Tier 4: Real-World Scenarios
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Simulated serving workflows under CUDA Graph execution models."""

    def test_simulated_cuda_graph_multi_step_ring_replay(self):
        """Simulate replaying a captured CUDA Graph across 5 continuous batching steps.

        In CUDA Graphs, host-side graph node topology is static, but device memory
        in `ctl` changes per step to dynamically select slot and mask padding.
        """
        K_slots = 4
        max_bucket_tokens = 32
        slab_dim = 16
        staging_slab = torch.zeros(K_slots, max_bucket_tokens, slab_dim)

        # Host-pinned ring buffer for D2H copy
        host_ring = torch.zeros(K_slots, max_bucket_tokens, slab_dim)

        step_token_counts = [12, 18, 5, 29, 14]
        features = torch.tensor([10, 20, 30, 40], dtype=torch.long)

        for step_idx, tokens in enumerate(step_token_counts):
            slot_idx = step_idx % K_slots
            # Static buffer padded to max_bucket_tokens
            activations = torch.randn(max_bucket_tokens, 64)

            # Update ctl tensor dynamically
            ctl = torch.tensor([1, tokens, slot_idx], dtype=torch.long)

            # Replay tap op
            tap(activations, features, ctl, staging_slab, col_offset=0)

            # Verify current slot gathered exactly `tokens` rows
            expected = activations[:tokens, features]
            assert torch.allclose(staging_slab[slot_idx, :tokens, :4], expected)
            assert torch.all(staging_slab[slot_idx, tokens:, :4] == 0)

            # Simulate out-of-graph D2H copy
            glassbox_async_d2h_copy(
                staging_slab,
                host_ring,
                slot_idx=slot_idx,
                actual_num_tokens=tokens,
            )
            assert torch.allclose(host_ring[slot_idx, :tokens, :4], expected)
