"""Adversarial stress-test harness for wiretap op, device control tensor ctl,
ring buffer, and storage collator.

Evaluates glassbox/csrc_fallback.py, csrc/ring_buffer.h, and glassbox/storage/collator.py:
1. Extreme ctl inputs: ctl[0]==0 (kill switch), ctl[1]==0 (zero tokens),
   actual_num_tokens > shape[0], out-of-bounds ctl[2] (slot index),
   and negative/out-of-bounds col_offset.
2. Host ring buffer: SPSC contracts, capacity boundaries, DROP_NEWEST fail-open policy under
   massive drop rates, and concurrent multi-threaded stress.
3. Storage collator: Corrupted WAL records (CRC32 mismatches, corrupted headers, bad JSON),
   multiple rank segment interleaving, high drop-rate tombstones, and tombstone precedence.
4. Compaction: Multi-rank WAL compaction under dropped steps, empty frames (0 tokens),
   100% drop-rate tombstones, zero-token active steps, and multi-rank Shannon entropy reduction.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from glassbox.csrc_fallback import (
    glassbox_async_d2h_copy,
    tap,
    tap_fallback,
    tap_numpy,
)
from glassbox.storage.collator import (
    RECORD_FRAME_SIZE,
    SEGMENT_HEADER_SIZE,
    VirtualIndexReader,
    WalSegmentWriter,
)
from glassbox.storage.layout import read_deep_index, read_deep_slice

# ==============================================================================
# 1. Wiretap Op & Device Control Tensor (ctl) Extreme Inputs
# ==============================================================================


class TestWiretapFallbackAdversarial:
    """Adversarial stress tests for glassbox/csrc_fallback.py."""

    def test_ctl0_kill_switch_with_corrupted_inputs_and_nans(self):
        """When ctl[0] == 0, staging slab must remain strictly untouched
        even with extreme inputs."""
        act = torch.full((8, 32), float("nan"), dtype=torch.float32)
        act[0, 0] = float("inf")
        feat = torch.tensor([0, 1, 5, 10], dtype=torch.long)
        slab = torch.zeros(4, 16, 16, dtype=torch.float32)

        # Extreme OOB slot_idx with kill switch = 0
        ctl_oob = torch.tensor([0, 8, 99999], dtype=torch.long)
        tap_fallback(act, feat, ctl_oob, slab, col_offset=0)
        assert torch.all(slab == 0), "Staging slab must remain zero when ctl[0] == 0"

        # Negative slot_idx with kill switch = 0
        ctl_neg = torch.tensor([0, 8, -50], dtype=torch.long)
        tap_fallback(act, feat, ctl_neg, slab, col_offset=0)
        assert torch.all(slab == 0), "Staging slab must remain zero when ctl[0] == 0"

    def test_ctl1_zero_and_negative_actual_tokens(self):
        """When ctl[1] <= 0, no tokens should be gathered; staging slab untouched."""
        act = torch.randn(8, 32, dtype=torch.float32)
        feat = torch.tensor([1, 3], dtype=torch.long)
        slab = torch.zeros(4, 16, 8, dtype=torch.float32)

        # ctl[1] == 0
        ctl_zero = torch.tensor([1, 0, 0], dtype=torch.long)
        tap_fallback(act, feat, ctl_zero, slab, col_offset=0)
        assert torch.all(slab == 0)

        # ctl[1] < 0
        ctl_neg = torch.tensor([1, -10, 0], dtype=torch.long)
        tap_fallback(act, feat, ctl_neg, slab, col_offset=0)
        assert torch.all(slab == 0)

        # ctl[1] <= 0 with out-of-bounds slot index should safely no-op without raising IndexError
        ctl_zero_oob = torch.tensor([1, 0, 999], dtype=torch.long)
        tap_fallback(act, feat, ctl_zero_oob, slab, col_offset=0)
        assert torch.all(slab == 0)

    def test_actual_tokens_exceeds_activations_and_slab_capacity(self):
        """actual_num_tokens > shape[0] must clamp to min(actual, act_rows, slab_rows)."""
        act_rows = 4
        slab_rows = 6
        act = torch.randn(act_rows, 16, dtype=torch.float32)
        feat = torch.tensor([2, 4], dtype=torch.long)
        slab = torch.zeros(2, slab_rows, 8, dtype=torch.float32)

        # actual_tokens = 10 (> act_rows and > slab_rows)
        ctl = torch.tensor([1, 10, 0], dtype=torch.long)
        tap_fallback(act, feat, ctl, slab, col_offset=0)
        # Should write exactly min(10, 4, 6) = 4 rows
        assert torch.allclose(slab[0, :act_rows, :2], act[:act_rows, feat])
        assert torch.all(slab[0, act_rows:, :] == 0)

        # Huge actual_tokens (2^31 - 1)
        slab.zero_()
        ctl_huge = torch.tensor([1, 2**31 - 1, 0], dtype=torch.long)
        tap_fallback(act, feat, ctl_huge, slab, col_offset=0)
        assert torch.allclose(slab[0, :act_rows, :2], act[:act_rows, feat])
        assert torch.all(slab[0, act_rows:, :] == 0)

    def test_slot_index_out_of_bounds_raises_index_error(self):
        """When ctl[0] == 1 and actual_tokens > 0, out-of-bounds ctl[2] must raise IndexError."""
        act = torch.randn(4, 16, dtype=torch.float32)
        feat = torch.tensor([0, 1], dtype=torch.long)
        slab = torch.zeros(2, 8, 4, dtype=torch.float32)

        # Negative slot index
        ctl_neg = torch.tensor([1, 2, -1], dtype=torch.long)
        with pytest.raises(IndexError, match="out of bounds"):
            tap_fallback(act, feat, ctl_neg, slab, col_offset=0)

        # slot index >= slab.shape[0]
        ctl_oob = torch.tensor([1, 2, 2], dtype=torch.long)
        with pytest.raises(IndexError, match="out of bounds"):
            tap_fallback(act, feat, ctl_oob, slab, col_offset=0)

        ctl_huge = torch.tensor([1, 2, 1000000], dtype=torch.long)
        with pytest.raises(IndexError, match="out of bounds"):
            tap_fallback(act, feat, ctl_huge, slab, col_offset=0)

    def test_col_offset_and_boundary_col_offsets(self):
        """col_offset behavior: clamping when exceeding slab_cols, error on negative offset."""
        act = torch.randn(4, 16, dtype=torch.float32)
        feat = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        slab = torch.zeros(2, 4, 8, dtype=torch.float32)
        ctl = torch.tensor([1, 4, 0], dtype=torch.long)

        # Partial fit: col_offset = 6, slab_cols = 8, 4 features -> only 2 features fit
        tap_fallback(act, feat, ctl, slab, col_offset=6)
        assert torch.allclose(slab[0, :, 6:8], act[:, feat[:2]])

        # Complete overflow: col_offset >= slab_cols (e.g. 10 >= 8) -> safely no-ops without error
        slab.zero_()
        tap_fallback(act, feat, ctl, slab, col_offset=10)
        assert torch.all(slab == 0), "Slab must remain zero on col_offset >= slab_cols"

        # Negative col_offset: safely no-ops without memory wrapping or crash
        slab.zero_()
        tap_fallback(act, feat, ctl, slab, col_offset=-1)
        assert torch.all(slab == 0), "Slab must remain zero on negative col_offset (-1)"
        tap_fallback(act, feat, ctl, slab, col_offset=-5)
        assert torch.all(slab == 0), "Slab must remain zero on negative col_offset (-5)"

        # NumPy fallback parity
        slab_np = np.zeros((2, 4, 8), dtype=np.float32)
        tap_numpy(act.numpy(), feat.numpy(), [1, 4, 0], slab_np, col_offset=10)
        assert np.all(slab_np == 0)
        tap_numpy(act.numpy(), feat.numpy(), [1, 4, 0], slab_np, col_offset=-5)
        assert np.all(slab_np == 0)

    def test_numpy_fallback_parity_under_extreme_inputs(self):
        """tap_numpy must exhibit identical behavior and boundary protection to PyTorch fallback."""
        act_np = np.random.randn(4, 16).astype(np.float32)
        feat_np = np.array([1, 3], dtype=np.int64)
        slab_np = np.zeros((2, 8, 4), dtype=np.float32)

        # 1. ctl[0] == 0
        fb_ctl_kill = [0, 4, 0]
        tap_numpy(act_np, feat_np, fb_ctl_kill, slab_np, col_offset=0)
        assert np.all(slab_np == 0)

        # 2. ctl[1] <= 0
        tap_numpy(act_np, feat_np, [1, 0, 0], slab_np, col_offset=0)
        assert np.all(slab_np == 0)

        # 3. actual_tokens > act_rows
        tap_numpy(act_np, feat_np, [1, 100, 0], slab_np, col_offset=0)
        assert np.allclose(slab_np[0, :4, :2], act_np[:4, feat_np])
        assert np.all(slab_np[0, 4:, :] == 0)

        # 4. Out of bounds slot
        with pytest.raises(IndexError, match="out of bounds"):
            tap_numpy(act_np, feat_np, [1, 2, -1], slab_np, col_offset=0)
        with pytest.raises(IndexError, match="out of bounds"):
            tap_numpy(act_np, feat_np, [1, 2, 2], slab_np, col_offset=0)

        # 5. Invalid ctl array size < 3
        with pytest.raises(ValueError, match="at least 3 elements"):
            tap_numpy(act_np, feat_np, [1, 2], slab_np, col_offset=0)

    def test_3d_batch_activations_reshaping(self):
        """3D activations [batch, seq, hidden] must reshape smoothly into 2D [tokens, hidden]."""
        batch, seq, hidden = 2, 3, 16
        act_3d = torch.randn(batch, seq, hidden, dtype=torch.float32)
        feat = torch.tensor([0, 2], dtype=torch.long)
        slab = torch.zeros(1, 10, 4, dtype=torch.float32)
        ctl = torch.tensor([1, 6, 0], dtype=torch.long)

        tap(act_3d, feat, ctl, slab, col_offset=0)
        act_flat = act_3d.reshape(-1, hidden)
        assert torch.allclose(slab[0, :6, :2], act_flat[:6, feat])

    def test_async_d2h_copy_cpu_boundaries(self):
        """glassbox_async_d2h_copy CPU fallback correctly copies
        row-limited and slot-limited slices."""
        src = torch.randn(4, 8, 16, dtype=torch.float32)
        dst = torch.zeros(4, 8, 16, dtype=torch.float32)

        # Copy slot 1, 3 rows only
        glassbox_async_d2h_copy(src, dst, slot_idx=1, actual_num_tokens=3)
        assert torch.allclose(dst[1, :3], src[1, :3])
        assert torch.all(dst[1, 3:] == 0)
        assert torch.all(dst[0] == 0)
        assert torch.all(dst[2:] == 0)


# ==============================================================================
# 2. Host Ring Buffer (csrc/ring_buffer.h) SPSC Contracts & Stress Harness
# ==============================================================================


class TestHostRingBufferAdversarial:
    """Empirical verification of wait-free SPSC ring buffer and DROP_NEWEST policy."""

    def test_cpp_ring_buffer_stress_harness(self):
        """Compile and execute C++ stress harness verifying DROP_NEWEST,
        boundaries, and concurrency."""
        if shutil.which("g++") is None:
            pytest.skip("g++ compiler not found in PATH")
        cpp_code = """
#include "csrc/ring_buffer.h"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

int main() {
    // 1. Capacity boundaries: capacity == 0 defaults to DEFAULT_CAPACITY (16)
    glassbox::HostRingBuffer rb0(64, 0);
    assert(rb0.capacity() == glassbox::HostRingBuffer::DEFAULT_CAPACITY);

    // 2. DROP_NEWEST under massive drop rate
    glassbox::HostRingBuffer rb(64, 4);
    for (uint64_t s = 1; s <= 10000; ++s) {
        auto* slot = rb.try_acquire(s, 4);
        if (slot) {
            rb.commit(slot);
        }
    }
    assert(rb.occupancy() == 4);
    assert(rb.get_dropped_count() == 9996);

    // Drain tombstones (bounded by TOMBSTONE_CAPACITY=256)
    std::vector<glassbox::TombstoneEntry> tombstones(512);
    uint32_t drained = rb.drain_tombstones(tombstones.data(), 512);
    assert(drained == glassbox::HostRingBuffer::TOMBSTONE_CAPACITY);
    assert(rb.drain_tombstones(tombstones.data(), 512) == 0);

    // 3. Multi-threaded SPSC stress
    glassbox::HostRingBuffer rb_mt(128, 16);
    const uint64_t num_msgs = 50000;
    std::atomic<bool> done{false};
    std::atomic<uint64_t> received{0};

    std::thread consumer([&]() {
        while (!done.load(std::memory_order_relaxed) || rb_mt.occupancy() > 0) {
            const auto* slot = rb_mt.peek();
            if (slot) {
                received.fetch_add(1, std::memory_order_relaxed);
                rb_mt.release(slot);
            } else {
                std::this_thread::yield();
            }
        }
    });

    std::thread producer([&]() {
        for (uint64_t s = 1; s <= num_msgs; ++s) {
            while (true) {
                auto* slot = rb_mt.try_acquire(s, 8);
                if (slot) {
                    rb_mt.commit(slot);
                    break;
                }
                std::this_thread::yield();
            }
        }
        done.store(true, std::memory_order_release);
    });

    producer.join();
    consumer.join();
    assert(received.load() == num_msgs);

    std::cout << "SPSC_STRESS_OK" << std::endl;
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "stress.cpp"
            bin_file = Path(tmpdir) / "stress"
            src_file.write_text(cpp_code)

            compile_cmd = [
                "g++",
                "-std=c++17",
                "-pthread",
                "-O2",
                "-I.",
                str(src_file),
                "-o",
                str(bin_file),
            ]
            comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
            assert comp_res.returncode == 0, f"Compilation failed: {comp_res.stderr}"

            run_res = subprocess.run([str(bin_file)], capture_output=True, text=True)
            assert run_res.returncode == 0, f"Execution failed: {run_res.stderr}"
            assert "SPSC_STRESS_OK" in run_res.stdout


# ==============================================================================
# 3. Storage Collator Corruptions, Interleaving & High Drop Rates
# ==============================================================================


class TestStorageCollatorCorruptionsAndInterleaving:
    """Adversarial stress tests for glassbox/storage/collator.py."""

    def test_wal_crc32_payload_corruption_rejection(self, tmp_path):
        """Corrupted payload bytes causing CRC32 mismatch must be excluded
        while preserving valid records."""
        run_dir = tmp_path / "RUN-CRC32"
        rank_dir = run_dir / "ranks" / "rank_0"

        with WalSegmentWriter(rank_dir, rank_id=0) as writer:
            for s in range(1, 6):
                matrix = np.full((4, 2), fill_value=float(s), dtype=np.float32)
                writer.write_activation(
                    step_id=s,
                    num_tokens=4,
                    addresses=["L-00", "A-00-01"],
                    req_ids=["req_1"],
                    query_start_loc=[0, 4],
                    matrix=matrix,
                    dtype="fp32",
                )

        seg_file = list(rank_dir.glob("*.bin"))[0]
        data = bytearray(seg_file.read_bytes())

        # Corrupt step 3's payload
        pos = SEGMENT_HEADER_SIZE
        found = False
        import struct

        while pos + RECORD_FRAME_SIZE <= len(data):
            magic, rec_type, flags, payload_len, step_id, crc32 = struct.unpack(
                "<4sHHIQI", data[pos : pos + RECORD_FRAME_SIZE]
            )
            if step_id == 3:
                found = True
                data[pos + RECORD_FRAME_SIZE + 5] ^= 0xFF  # Flip byte
                break
            pos += RECORD_FRAME_SIZE + payload_len

        assert found
        seg_file.write_bytes(data)

        reader = VirtualIndexReader(run_dir)
        # Step 3 must be excluded due to CRC32 mismatch
        assert reader.active_steps == [1, 2, 4, 5]
        assert reader.query_slice(3, "req_1", "L-00") is None
        # Other steps remain intact
        assert reader.query_slice(1, "req_1", "L-00") is not None
        assert reader.query_slice(5, "req_1", "L-00") is not None

    def test_wal_frame_magic_and_header_corruption(self, tmp_path):
        """Corrupted record magic (e.g. non-WALR) terminates segment scan
        gracefully without crash."""
        run_dir = tmp_path / "RUN-BADMAGIC"
        rank_dir = run_dir / "ranks" / "rank_0"

        with WalSegmentWriter(rank_dir, rank_id=0) as writer:
            for s in range(1, 6):
                matrix = np.full((2, 1), fill_value=float(s), dtype=np.float32)
                writer.write_activation(
                    step_id=s,
                    num_tokens=2,
                    addresses=["L-00"],
                    req_ids=["r1"],
                    query_start_loc=[0, 2],
                    matrix=matrix,
                )

        seg_file = list(rank_dir.glob("*.bin"))[0]
        data = bytearray(seg_file.read_bytes())

        # Corrupt step 4's magic header
        pos = SEGMENT_HEADER_SIZE
        import struct

        while pos + RECORD_FRAME_SIZE <= len(data):
            magic, rec_type, flags, payload_len, step_id, crc32 = struct.unpack(
                "<4sHHIQI", data[pos : pos + RECORD_FRAME_SIZE]
            )
            if step_id == 4:
                data[pos : pos + 4] = b"CORR"
                break
            pos += RECORD_FRAME_SIZE + payload_len

        seg_file.write_bytes(data)
        reader = VirtualIndexReader(run_dir)
        # Scan terminates gracefully at corrupted frame: steps 1, 2, 3 preserved
        assert reader.active_steps == [1, 2, 3]

    def test_wal_segment_header_corruption(self, tmp_path):
        """Corrupted segment header magic causes reader to skip
        corrupted segment file gracefully."""
        run_dir = tmp_path / "RUN-BADSEG"
        rank_dir = run_dir / "ranks" / "rank_0"

        with WalSegmentWriter(rank_dir, rank_id=0) as writer:
            writer.write_activation(
                1, 2, ["L-00"], ["r1"], [0, 2], np.ones((2, 1), dtype=np.float32)
            )

        seg_file = list(rank_dir.glob("*.bin"))[0]
        data = bytearray(seg_file.read_bytes())
        data[0:8] = b"BADMAGIC"  # Corrupt WAL_MAGIC
        seg_file.write_bytes(data)

        reader = VirtualIndexReader(run_dir)
        assert len(reader.active_steps) == 0

    def test_multi_rank_segment_interleaving_and_rotation(self, tmp_path):
        """4 ranks with differing rotation thresholds and out-of-order
        segment chunking join cleanly."""
        run_dir = tmp_path / "RUN-INTERLEAVE-4R"

        # Rank 0 rotates every 2 steps
        r0 = run_dir / "ranks" / "rank_0"
        with WalSegmentWriter(r0, rank_id=0, max_segment_size=600) as w:
            for s in range(1, 6):
                w.write_activation(
                    s,
                    2,
                    ["L-00", "LOGITS-ENTROPY_m", "LOGITS-ENTROPY_s", "LOGITS-ENTROPY_w"],
                    ["req1"],
                    [0, 2],
                    np.ones((2, 4), dtype=np.float32) * s,
                )

        # Rank 1 rotates every step
        r1 = run_dir / "ranks" / "rank_1"
        with WalSegmentWriter(r1, rank_id=1, max_segment_size=300) as w:
            for s in range(1, 6):
                w.write_activation(
                    s,
                    2,
                    ["A-00-01", "LOGITS-ENTROPY_m", "LOGITS-ENTROPY_s", "LOGITS-ENTROPY_w"],
                    ["req1"],
                    [0, 2],
                    np.ones((2, 4), dtype=np.float32) * s,
                )

        # Rank 2 does not rotate (single segment)
        r2 = run_dir / "ranks" / "rank_2"
        with WalSegmentWriter(r2, rank_id=2, max_segment_size=100000) as w:
            for s in range(1, 6):
                w.write_activation(
                    s,
                    2,
                    ["A-00-02", "LOGITS-ENTROPY_m", "LOGITS-ENTROPY_s", "LOGITS-ENTROPY_w"],
                    ["req1"],
                    [0, 2],
                    np.ones((2, 4), dtype=np.float32) * s,
                )

        # Rank 3 drops step 2 and step 4 with tombstones
        r3 = run_dir / "ranks" / "rank_3"
        with WalSegmentWriter(r3, rank_id=3, max_segment_size=400) as w:
            for s in range(1, 6):
                if s in (2, 4):
                    w.write_tombstone(s, reason=f"Rank 3 drop step {s}")
                else:
                    w.write_activation(
                        s,
                        2,
                        ["A-00-03", "LOGITS-ENTROPY_m", "LOGITS-ENTROPY_s", "LOGITS-ENTROPY_w"],
                        ["req1"],
                        [0, 2],
                        np.ones((2, 4), dtype=np.float32) * s,
                    )

        reader = VirtualIndexReader(run_dir)
        assert reader.active_steps == [1, 3, 5]
        assert reader.dropped_steps == [2, 4]

        # Verify query_slice on dropped steps yields None
        assert reader.query_slice(2, "req1", "L-00") is None
        assert reader.query_slice(4, "req1", "A-00-01") is None

        # Verify active steps query across ranks
        for s in [1, 3, 5]:
            assert reader.query_slice(s, "req1", "L-00") is not None
            assert reader.query_slice(s, "req1", "A-00-01") is not None
            assert reader.query_slice(s, "req1", "A-00-02") is not None
            assert reader.query_slice(s, "req1", "A-00-03") is not None
            entropy = reader.query_slice(s, "req1", "LOGITS-ENTROPY")
            assert entropy is not None and len(entropy) == 2

    def test_high_drop_rate_tombstone_precedence(self, tmp_path):
        """When 80% of steps are dropped by one rank, tombstone precedence
        marks them dropped globally."""
        run_dir = tmp_path / "RUN-HIGHDROP"
        r0 = run_dir / "ranks" / "rank_0"
        r1 = run_dir / "ranks" / "rank_1"

        total_steps = 10
        with WalSegmentWriter(r0, rank_id=0) as w0, WalSegmentWriter(r1, rank_id=1) as w1:
            for s in range(1, total_steps + 1):
                # Rank 0 always writes activations
                w0.write_activation(
                    s, 2, ["L-00"], ["r1"], [0, 2], np.ones((2, 1), dtype=np.float32) * s
                )
                # Rank 1 drops 8 out of 10 steps (all except steps 1 and 10)
                if s not in (1, 10):
                    w1.write_tombstone(s, reason="High drop rate stress")
                else:
                    w1.write_activation(
                        s, 2, ["A-00-01"], ["r1"], [0, 2], np.ones((2, 1), dtype=np.float32) * s
                    )

        reader = VirtualIndexReader(run_dir)
        assert reader.active_steps == [1, 10]
        assert reader.dropped_steps == [2, 3, 4, 5, 6, 7, 8, 9]


# ==============================================================================
# 4. Offline Compaction Adversarial Scenarios
# ==============================================================================


class TestCompactionAdversarialScenarios:
    """Stress testing offline compaction under multi-rank dropped steps,
    empty frames, and reductions."""

    def test_compaction_skips_dropped_steps_and_reconstructs_dense_matrix(self, tmp_path):
        """Compaction skips dropped steps seamlessly; output deep.bin matches active steps only."""
        run_dir = tmp_path / "RUN-COMPACT-DROPS"
        out_dir = tmp_path / "OUTPUT-COMPACT-DROPS"

        r0 = run_dir / "ranks" / "rank_0"
        with WalSegmentWriter(r0, rank_id=0) as w:
            w.write_activation(
                1, 4, ["L-00"], ["r1"], [0, 4], np.ones((4, 1), dtype=np.float32) * 10
            )
            w.write_tombstone(2, reason="Stall")
            w.write_activation(
                3, 4, ["L-00"], ["r1"], [0, 4], np.ones((4, 1), dtype=np.float32) * 30
            )

        reader = VirtualIndexReader(run_dir)
        reader.compact_to_deep_bin(out_dir, precision="fp16", chunk_tokens=4)

        idx = read_deep_index(str(out_dir))
        assert idx["tokens"] == 8  # Step 1 (4 tokens) + Step 3 (4 tokens)
        slice_data = read_deep_slice(str(out_dir), 0, 7)
        assert slice_data.shape == (8, 1)
        assert np.allclose(slice_data[:4, 0], 10.0)
        assert np.allclose(slice_data[4:, 0], 30.0)

    def test_compaction_handles_empty_frames_with_zero_tokens(self, tmp_path):
        """Compacting active steps where one step has num_tokens == 0 proceeds
        without corrupting offsets."""
        run_dir = tmp_path / "RUN-COMPACT-EMPTY"
        out_dir = tmp_path / "OUTPUT-COMPACT-EMPTY"

        r0 = run_dir / "ranks" / "rank_0"
        with WalSegmentWriter(r0, rank_id=0) as w:
            w.write_activation(
                1, 4, ["L-00"], ["r1"], [0, 4], np.ones((4, 1), dtype=np.float32) * 15
            )
            # Empty frame
            w.write_activation(2, 0, ["L-00"], [], [0], np.zeros((0, 1), dtype=np.float32))
            w.write_activation(
                3, 4, ["L-00"], ["r1"], [0, 4], np.ones((4, 1), dtype=np.float32) * 45
            )

        reader = VirtualIndexReader(run_dir)
        reader.compact_to_deep_bin(out_dir, precision="fp16", chunk_tokens=4)

        idx = read_deep_index(str(out_dir))
        assert idx["tokens"] == 8
        slice_data = read_deep_slice(str(out_dir), 0, 7)
        assert slice_data.shape == (8, 1)
        assert np.allclose(slice_data[:4, 0], 15.0)
        assert np.allclose(slice_data[4:, 0], 45.0)

    def test_compaction_all_dropped_steps_raises_value_error(self, tmp_path):
        """When 100% of steps are dropped, compaction raises ValueError."""
        run_dir = tmp_path / "RUN-ALL-DROPPED"
        out_dir = tmp_path / "OUTPUT-ALL-DROPPED"

        r0 = run_dir / "ranks" / "rank_0"
        with WalSegmentWriter(r0, rank_id=0) as w:
            w.write_tombstone(1, reason="All dropped")
            w.write_tombstone(2, reason="All dropped")

        reader = VirtualIndexReader(run_dir)
        with pytest.raises(ValueError, match="No active .* steps to compact"):
            reader.compact_to_deep_bin(out_dir)

    def test_compaction_zero_total_tokens_raises_value_error(self, tmp_path):
        """When all active steps have 0 tokens, compaction raises ValueError."""
        run_dir = tmp_path / "RUN-ALL-ZERO"
        out_dir = tmp_path / "OUTPUT-ALL-ZERO"

        r0 = run_dir / "ranks" / "rank_0"
        with WalSegmentWriter(r0, rank_id=0) as w:
            w.write_activation(1, 0, ["L-00"], [], [0], np.zeros((0, 1), dtype=np.float32))

        reader = VirtualIndexReader(run_dir)
        with pytest.raises(ValueError, match="Total tokens across active steps is 0"):
            reader.compact_to_deep_bin(out_dir)

    def test_compaction_multi_rank_entropy_reduction(self, tmp_path):
        """Compaction evaluates and merges LOGITS-ENTROPY across TP ranks into final deep.bin."""
        run_dir = tmp_path / "RUN-COMPACT-ENTROPY"
        out_dir = tmp_path / "OUTPUT-COMPACT-ENTROPY"

        r0 = run_dir / "ranks" / "rank_0"
        r1 = run_dir / "ranks" / "rank_1"

        with WalSegmentWriter(r0, rank_id=0) as w0, WalSegmentWriter(r1, rank_id=1) as w1:
            w0.write_activation(
                1,
                4,
                ["L-00", "LOGITS-ENTROPY_m", "LOGITS-ENTROPY_s", "LOGITS-ENTROPY_w"],
                ["req1"],
                [0, 4],
                np.ones((4, 4), dtype=np.float32) * 5.0,
            )
            w1.write_activation(
                1,
                4,
                ["A-00-01", "LOGITS-ENTROPY_m", "LOGITS-ENTROPY_s", "LOGITS-ENTROPY_w"],
                ["req1"],
                [0, 4],
                np.ones((4, 4), dtype=np.float32) * 5.0,
            )

        reader = VirtualIndexReader(run_dir)
        reader.compact_to_deep_bin(out_dir, precision="fp32", chunk_tokens=4)

        idx = read_deep_index(str(out_dir))
        assert "LOGITS-ENTROPY" in idx["watched"]
        slice_data = read_deep_slice(str(out_dir), 0, 3)

        col_map = {name: i for i, name in enumerate(idx["watched"])}
        ent_vals = slice_data[:, col_map["LOGITS-ENTROPY"]]
        # Ground truth: 2 equal ranks with (m=5, s=5, w=5)
        # S = 5 + 5 = 10, W = 5 + 5 = 10. H = ln(10) - (10/10) = 2.302585 - 1.0 = 1.302585
        expected_h = math.log(10.0) - 1.0
        assert np.allclose(ent_vals, expected_h, atol=1e-4)
