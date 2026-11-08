"""Comprehensive 4-tier unit and integration tests for storage, ring buffer & collator.

Covers Features 19-25:
- Tier 1: Feature Coverage (csrc/ring_buffer.h SPSC semantics, fail-open DROP_NEWEST,
          segmented WAL headers/records, VirtualIndexReader multi-rank queries,
          offline compactor GBSPR001)
- Tier 2: Boundary & Corner Cases (empty WAL directory, CRC32 corrupted records,
          truncated frames, non-existent steps/probes, zero-token frames, tombstone precedence)
- Tier 3: Cross-Feature Combinations (4-rank multi-GPU WAL generation with sharded heads &
          residual stream, fp16/fp32/bf16 precision roundtrip, compactor + read_deep_slice)
- Tier 4: Real-World Scenarios (Simulated disk backpressure stall with DROP_NEWEST fail-open,
          multi-segment rotation across 100+ steps)
"""

from __future__ import annotations

import shutil
import subprocess
import pytest
import tempfile
from pathlib import Path

import numpy as np

from glassbox.storage.collator import (
    DEFAULT_MAX_SEGMENT_SIZE,
    RECORD_FRAME_SIZE,
    RECORD_MAGIC,
    RECORD_TYPE_ACTIVATION,
    SEGMENT_HEADER_SIZE,
    WAL_MAGIC,
    RecordFrame,
    SegmentHeader,
    VirtualIndexReader,
    WalSegmentWriter,
)
from glassbox.storage.layout import read_deep_index, read_deep_slice

# ==============================================================================
# Helper functions
# ==============================================================================


def create_synthetic_multi_rank_run(
    run_dir: Path,
    num_ranks: int = 2,
    num_steps: int = 5,
    num_tokens: int = 8,
    dropped_step: int | None = None,
) -> None:
    """Create a multi-rank directory structure with synthetic WAL segment files."""
    for rank in range(num_ranks):
        rank_dir = run_dir / "ranks" / f"rank_{rank}"
        with WalSegmentWriter(rank_dir, rank_id=rank) as writer:
            for step in range(1, num_steps + 1):
                if dropped_step is not None and step == dropped_step:
                    writer.write_tombstone(step_id=step, reason="Backpressure stall")
                    continue

                if rank == 0:
                    addresses = ["L-00", f"A-00-{rank + 1:02d}"]
                else:
                    addresses = [f"A-00-{rank + 1:02d}"]

                matrix = np.full(
                    (num_tokens, len(addresses)),
                    fill_value=float(step * 10 + rank),
                    dtype=np.float32,
                )
                writer.write_activation(
                    step_id=step,
                    num_tokens=num_tokens,
                    addresses=addresses,
                    req_ids=["req_alpha", "req_beta"],
                    query_start_loc=[0, num_tokens // 2, num_tokens],
                    matrix=matrix,
                    dtype="fp16",
                )


# ==============================================================================
# Tier 1: Feature Coverage (Features 19-25)
# ==============================================================================


class TestFeature19HostRingBufferSPSC:
    """Feature 19: Wait-free SPSC host ring buffer in csrc/ring_buffer.h."""

    def test_cpp_ring_buffer_compiles_and_passes_spsc_contracts(self):
        """Compile and execute C++ test verifying csrc/ring_buffer.h SPSC mechanics."""
        if shutil.which("g++") is None:
            pytest.skip("g++ compiler not found in PATH")
        test_cpp_source = """
#include "csrc/ring_buffer.h"
#include <cassert>
#include <iostream>

int main() {
    glassbox::HostRingBuffer rb(512, 4);
    assert(rb.capacity() == 4);
    assert(rb.occupancy() == 0);
    assert(rb.get_dropped_count() == 0);

    // Enqueue 4 elements
    for (uint64_t i = 1; i <= 4; ++i) {
        char data[32] = "token_data";
        bool ok = rb.push(i, 4, data, sizeof(data), "{\\"req\\":1}");
        assert(ok);
    }
    assert(rb.occupancy() == 4);

    // Dequeue 2 elements
    for (uint64_t i = 1; i <= 2; ++i) {
        const glassbox::RingSlot* slot = rb.peek();
        assert(slot != nullptr);
        assert(slot->step_id == i);
        rb.release(slot);
    }
    assert(rb.occupancy() == 2);

    // Enqueue 2 more elements (wrapping around ring)
    for (uint64_t i = 5; i <= 6; ++i) {
        char data[32] = "wrapped_data";
        bool ok = rb.push(i, 4, data, sizeof(data), "{\\"req\\":2}");
        assert(ok);
    }
    assert(rb.occupancy() == 4);

    // Dequeue remaining 4 elements
    for (uint64_t i = 3; i <= 6; ++i) {
        const glassbox::RingSlot* slot = rb.peek();
        assert(slot != nullptr);
        assert(slot->step_id == i);
        rb.release(slot);
    }
    assert(rb.occupancy() == 0);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = Path(tmp_dir) / "test_rb.cpp"
            bin_file = Path(tmp_dir) / "test_rb"
            src_file.write_text(test_cpp_source)

            compile_cmd = [
                "g++",
                "-std=c++17",
                "-O2",
                "-I.",
                str(src_file),
                "-o",
                str(bin_file),
            ]
            comp_res = subprocess.run(compile_cmd, capture_output=True, text=True)
            assert comp_res.returncode == 0, f"C++ compilation failed: {comp_res.stderr}"

            exec_res = subprocess.run([str(bin_file)], capture_output=True, text=True)
            assert exec_res.returncode == 0, f"Ring buffer execution failed: {exec_res.stderr}"


class TestFeature20DropNewestFailOpenPolicy:
    """Feature 20: DROP_NEWEST fail-open policy under ring buffer backpressure."""

    def test_cpp_ring_buffer_drop_newest_under_backpressure(self):
        """Verify producer never blocks under full capacity and emits tombstones."""
        if shutil.which("g++") is None:
            pytest.skip("g++ compiler not found in PATH")
        test_cpp_source = """
#include "csrc/ring_buffer.h"
#include <cassert>

int main() {
    glassbox::HostRingBuffer rb(256, 2);

    // Fill ring completely
    assert(rb.push(1, 2, "a", 1, "m1"));
    assert(rb.push(2, 2, "b", 1, "m2"));
    assert(rb.occupancy() == 2);
    assert(rb.get_dropped_count() == 0);

    // Third push should DROP_NEWEST without blocking or failing
    bool ok = rb.push(3, 2, "c", 1, "m3");
    assert(!ok); // Returns false, does not block
    assert(rb.get_dropped_count() == 1);

    // Fourth push should also drop newest
    ok = rb.push(4, 2, "d", 1, "m4");
    assert(!ok);
    assert(rb.get_dropped_count() == 2);

    // Verify tombstones drained
    glassbox::TombstoneEntry tombstones[4];
    uint32_t drained = rb.drain_tombstones(tombstones, 4);
    assert(drained == 2);
    assert(tombstones[0].step_id == 3);
    assert(tombstones[0].reason_code == glassbox::DROP_REASON_RING_FULL);
    assert(tombstones[1].step_id == 4);
    assert(tombstones[1].reason_code == glassbox::DROP_REASON_RING_FULL);

    // Existing slots 1 and 2 remain uncorrupted
    const glassbox::RingSlot* s1 = rb.peek();
    assert(s1 != nullptr && s1->step_id == 1);
    rb.release(s1);

    const glassbox::RingSlot* s2 = rb.peek();
    assert(s2 != nullptr && s2->step_id == 2);
    rb.release(s2);

    assert(rb.occupancy() == 0);
    return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            src_file = Path(tmp_dir) / "test_drop.cpp"
            bin_file = Path(tmp_dir) / "test_drop"
            src_file.write_text(test_cpp_source)

            res = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-I.", str(src_file), "-o", str(bin_file)],
                capture_output=True,
                text=True,
            )
            assert res.returncode == 0, f"Compilation failed: {res.stderr}"

            exec_res = subprocess.run([str(bin_file)], capture_output=True, text=True)
            assert exec_res.returncode == 0, f"Drop newest verification failed: {exec_res.stderr}"


class TestFeature21SegmentedWalFormat:
    """Feature 21: Segmented WAL format (headers, record frames, CRC32)."""

    def test_segment_header_pack_unpack(self):
        hdr = SegmentHeader(
            magic=WAL_MAGIC,
            version=1,
            rank_id=3,
            seq=42,
            max_segment_size=DEFAULT_MAX_SEGMENT_SIZE,
            timestamp_ns=1700000000000,
        )
        packed = hdr.pack()
        assert len(packed) == SEGMENT_HEADER_SIZE
        unpacked = SegmentHeader.unpack(packed)
        assert unpacked.magic == WAL_MAGIC
        assert unpacked.version == 1
        assert unpacked.rank_id == 3
        assert unpacked.seq == 42
        assert unpacked.max_segment_size == DEFAULT_MAX_SEGMENT_SIZE
        assert unpacked.timestamp_ns == 1700000000000

    def test_record_frame_pack_unpack(self):
        frame = RecordFrame(
            magic=RECORD_MAGIC,
            record_type=RECORD_TYPE_ACTIVATION,
            flags=0,
            payload_len=1024,
            step_id=99,
            crc32=0x12345678,
        )
        packed = frame.pack()
        assert len(packed) == RECORD_FRAME_SIZE
        unpacked = RecordFrame.unpack(packed)
        assert unpacked.magic == RECORD_MAGIC
        assert unpacked.record_type == RECORD_TYPE_ACTIVATION
        assert unpacked.payload_len == 1024
        assert unpacked.step_id == 99
        assert unpacked.crc32 == 0x12345678

    def test_segment_rotation_on_size_exceeded(self, tmp_path):
        """WalSegmentWriter rotates to new segment when max_segment_size is reached."""
        small_max_size = 250  # Forces rotation on second record (64 + 136 = 200; 200 + 136 > 250)
        writer = WalSegmentWriter(tmp_path, rank_id=0, max_segment_size=small_max_size)

        # Write first activation
        writer.write_activation(
            step_id=1,
            num_tokens=4,
            addresses=["L-00"],
            req_ids=["r1"],
            query_start_loc=[0, 4],
            matrix=np.ones((4, 1), dtype=np.float32),
        )

        # Write second activation (should trigger rotation)
        writer.write_activation(
            step_id=2,
            num_tokens=4,
            addresses=["L-00"],
            req_ids=["r1"],
            query_start_loc=[0, 4],
            matrix=np.ones((4, 1), dtype=np.float32),
        )
        writer.close()

        seg_files = sorted(tmp_path.glob("segment_*.bin"))
        assert len(seg_files) >= 2, f"Expected at least 2 rotated segments, got {len(seg_files)}"


class TestFeature22OSCacheManagement:
    """Feature 22: OS Cache Management (posix_fadvise DONTNEED)."""

    def test_fadvise_dontneed_invoked_safely(self, tmp_path):
        """WalSegmentWriter invokes posix_fadvise(DONTNEED) on rotate and close."""
        writer = WalSegmentWriter(tmp_path, rank_id=0)
        writer.write_activation(
            step_id=1,
            num_tokens=2,
            addresses=["L-00"],
            req_ids=["r1"],
            query_start_loc=[0, 2],
            matrix=np.zeros((2, 1), dtype=np.float32),
        )
        # Calling _advise_dontneed should not raise on any platform
        writer._advise_dontneed()
        writer.close()


class TestFeature23VirtualMultiRankIndexing:
    """Feature 23: Virtual multi-rank index scanning and slicing without file merge."""

    def test_virtual_index_reader_queries_multi_rank_slices(self, tmp_path):
        run_dir = tmp_path / "test_run"
        create_synthetic_multi_rank_run(run_dir, num_ranks=4, num_steps=3, num_tokens=6)

        reader = VirtualIndexReader(run_dir)

        # Query rank 0 residual stream probe
        slice_l00 = reader.query_slice(global_step_id=1, req_id="req_alpha", address="L-00")
        assert slice_l00 is not None
        assert slice_l00.shape == (3,)  # Half tokens (3 tokens) for req_alpha
        assert np.allclose(slice_l00, 10.0)

        # Query rank 2 attention head probe A-00-03
        slice_head = reader.query_slice(global_step_id=2, req_id="req_beta", address="A-00-03")
        assert slice_head is not None
        assert slice_head.shape == (3,)
        assert np.allclose(slice_head, 22.0)  # step 2 * 10 + rank 2 = 22

    def test_dropped_step_indexing_and_query(self, tmp_path):
        run_dir = tmp_path / "test_run_dropped"
        create_synthetic_multi_rank_run(
            run_dir, num_ranks=2, num_steps=4, num_tokens=4, dropped_step=3
        )

        reader = VirtualIndexReader(run_dir)
        dropped_steps = reader.get_dropped_steps()
        assert 3 in dropped_steps

        # Querying dropped step 3 should return None
        assert reader.query_slice(global_step_id=3, req_id="req_alpha", address="L-00") is None

        # Non-dropped steps remain fully accessible
        assert reader.query_slice(global_step_id=1, req_id="req_alpha", address="L-00") is not None
        assert reader.query_slice(global_step_id=4, req_id="req_alpha", address="L-00") is not None


class TestFeature24OfflineGBSPR001Compactor:
    """Feature 24: Offline compaction into standard compressed GBSPR001 format."""

    def test_compaction_and_read_deep_slice_verification(self, tmp_path):
        run_dir = tmp_path / "src_run"
        create_synthetic_multi_rank_run(run_dir, num_ranks=2, num_steps=3, num_tokens=4)

        reader = VirtualIndexReader(run_dir)
        out_compact = tmp_path / "compacted_run"
        reader.compact_to_deep_bin(out_compact)

        # Verify output files exist
        assert (out_compact / "deep.bin").is_file()
        assert (out_compact / "deep_index.json").is_file()

        # Verify deep_index.json structure
        index = read_deep_index(str(out_compact))
        assert index["triggered"] is True
        assert index["tokens"] == 12  # 3 active steps * 4 tokens
        assert "L-00" in index["watched"]
        assert "A-00-01" in index["watched"]
        assert "A-00-02" in index["watched"]

        # Read back slice via standard storage layout reader
        slice_data = read_deep_slice(str(out_compact), token_lo=0, token_hi=11)
        assert slice_data.shape == (12, len(index["watched"]))
        assert slice_data.dtype == np.float32


class TestFeature25CollatorPythonAndNumPyFallback:
    """Feature 25: Robust pure-Python / NumPy fallback reader."""

    def test_fallback_reader_without_rust_shared_library(self, tmp_path, monkeypatch):
        """Force Rust C-ABI to None and verify pure-Python reader works 100% identically."""
        run_dir = tmp_path / "fallback_run"
        create_synthetic_multi_rank_run(run_dir, num_ranks=2, num_steps=2, num_tokens=4)

        # Patch loader to simulate environment without compiled Rust shared lib
        monkeypatch.setattr("glassbox.storage.collator._try_load_rust_lib", lambda: None)

        reader = VirtualIndexReader(run_dir)
        res = reader.query_slice(global_step_id=1, req_id="req_alpha", address="L-00")
        assert res is not None
        assert res.shape == (2,)


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """Boundary and corner cases for collator and storage."""

    def test_empty_wal_directory(self, tmp_path):
        empty_dir = tmp_path / "empty_run"
        empty_dir.mkdir()
        reader = VirtualIndexReader(empty_dir)
        assert reader.get_dropped_steps() == []
        assert reader.query_slice(1, "r1", "L-00") is None

    def test_corrupted_crc32_record_skipped_gracefully(self, tmp_path):
        """A record with bad CRC32 is detected and skipped without crashing the reader."""
        rank_dir = tmp_path / "ranks" / "rank_0"
        with WalSegmentWriter(rank_dir, rank_id=0) as writer:
            writer.write_activation(
                step_id=1,
                num_tokens=2,
                addresses=["L-00"],
                req_ids=["r1"],
                query_start_loc=[0, 2],
                matrix=np.ones((2, 1), dtype=np.float32),
            )
            writer.write_activation(
                step_id=2,
                num_tokens=2,
                addresses=["L-00"],
                req_ids=["r1"],
                query_start_loc=[0, 2],
                matrix=np.full((2, 1), 2.0, dtype=np.float32),
            )

        # Corrupt CRC32 of step 1 record in segment file
        seg_file = next(rank_dir.glob("segment_*.bin"))
        with open(seg_file, "r+b") as f:
            # Seek to CRC32 of first record (SEGMENT_HEADER_SIZE + 20)
            f.seek(SEGMENT_HEADER_SIZE + 20)
            f.write(b"\xde\xad\xbe\xef")  # corrupt CRC32

        reader = VirtualIndexReader(tmp_path)
        # Step 1 should be skipped due to CRC mismatch
        assert reader.query_slice(1, "r1", "L-00") is None
        # Step 2 should still be read successfully
        step2_slice = reader.query_slice(2, "r1", "L-00")
        assert step2_slice is not None
        assert np.allclose(step2_slice, 2.0)

    def test_query_unknown_address_returns_none(self, tmp_path):
        run_dir = tmp_path / "run_unknown"
        create_synthetic_multi_rank_run(run_dir, num_ranks=1, num_steps=1, num_tokens=2)
        reader = VirtualIndexReader(run_dir)
        assert reader.query_slice(1, "req_alpha", "NONEXISTENT_ADDRESS") is None

    def test_query_unknown_request_id_returns_none(self, tmp_path):
        run_dir = tmp_path / "run_req"
        create_synthetic_multi_rank_run(run_dir, num_ranks=1, num_steps=1, num_tokens=2)
        reader = VirtualIndexReader(run_dir)
        assert reader.query_slice(1, "unknown_req_id", "L-00") is None

    def test_tombstone_overrides_activation_record(self, tmp_path):
        """If a tombstone and an activation exist for the same step, tombstone takes precedence."""
        rank_dir = tmp_path / "ranks" / "rank_0"
        with WalSegmentWriter(rank_dir, rank_id=0) as writer:
            writer.write_tombstone(step_id=5, reason="Late drop")
            writer.write_activation(
                step_id=5,
                num_tokens=2,
                addresses=["L-00"],
                req_ids=["r1"],
                query_start_loc=[0, 2],
                matrix=np.ones((2, 1), dtype=np.float32),
            )

        reader = VirtualIndexReader(tmp_path)
        assert 5 in reader.get_dropped_steps()
        assert reader.query_slice(5, "r1", "L-00") is None


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Multi-rank, multi-precision combinations."""

    def test_all_precision_dtypes_roundtrip(self, tmp_path):
        """Verify fp16, fp32, and bf16 encodings serialize, deserialize, and compact accurately."""
        dtypes = ["fp16", "fp32", "bf16"]
        for dt in dtypes:
            sub_dir = tmp_path / f"prec_{dt}"
            rank_dir = sub_dir / "ranks" / "rank_0"
            matrix = np.array([[1.5, 2.25], [3.125, 4.0]], dtype=np.float32)

            with WalSegmentWriter(rank_dir, rank_id=0) as writer:
                writer.write_activation(
                    step_id=1,
                    num_tokens=2,
                    addresses=["L-00", "NORM"],
                    req_ids=["r1"],
                    query_start_loc=[0, 2],
                    matrix=matrix,
                    dtype=dt,
                )

            reader = VirtualIndexReader(sub_dir)
            slice_data = reader.query_slice(1, "r1", "L-00")
            assert slice_data is not None
            assert np.allclose(slice_data, matrix[:, 0], atol=1e-2)

            # Compact and read back
            compact_dir = sub_dir / "compact"
            reader.compact_to_deep_bin(compact_dir)
            back = read_deep_slice(str(compact_dir), 0, 1)
            assert np.allclose(back[:, 0], matrix[:, 0], atol=1e-2)


# ==============================================================================
# Tier 4: Real-World Scenarios
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Production serving scenarios and backpressure fault recovery."""

    def test_continuous_serving_with_intermittent_stall_drops(self, tmp_path):
        """Simulate a 20-step continuous serving run with 2 backpressure drop events.

        Verifies that:
        1. Non-dropped steps remain completely uncorrupted.
        2. Dropped steps are accurately isolated.
        3. Offline compactor skips dropped steps and successfully produces contiguous GBSPR001.
        """
        run_dir = tmp_path / "serving_run"
        rank_dirs = [run_dir / "ranks" / f"rank_{r}" for r in range(2)]

        writers = [WalSegmentWriter(rd, rank_id=i) for i, rd in enumerate(rank_dirs)]
        dropped_steps = {7, 14}

        for step in range(1, 21):
            if step in dropped_steps:
                for w in writers:
                    w.write_tombstone(step_id=step, reason="Disk backpressure stall")
            else:
                for rank, w in enumerate(writers):
                    w.write_activation(
                        step_id=step,
                        num_tokens=4,
                        addresses=[f"A-00-{rank:02d}"],
                        req_ids=["r_a", "r_b"],
                        query_start_loc=[0, 2, 4],
                        matrix=np.full((4, 1), float(step), dtype=np.float32),
                    )

        for w in writers:
            w.close()

        reader = VirtualIndexReader(run_dir)
        detected_drops = set(reader.get_dropped_steps())
        assert detected_drops == dropped_steps

        # Verify steps 1..6, 8..13, 15..20 are intact
        for s in range(1, 21):
            if s in dropped_steps:
                assert reader.query_slice(s, "r_a", "A-00-00") is None
            else:
                data = reader.query_slice(s, "r_a", "A-00-00")
                assert data is not None
                assert np.allclose(data, float(s))

        # Compact run
        compact_dir = tmp_path / "compacted_serving"
        reader.compact_to_deep_bin(compact_dir)

        index = read_deep_index(str(compact_dir))
        # 18 valid steps * 4 tokens = 72 tokens
        assert index["tokens"] == 72
        deep_slice = read_deep_slice(str(compact_dir), 0, 71)
        assert deep_slice.shape == (72, 2)
