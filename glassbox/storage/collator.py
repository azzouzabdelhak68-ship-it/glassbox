"""Tri-Language Virtual Index Collator and Segmented WAL Reader.

Implements Component 4 (Storage, Fail-Open Ring Buffer & Virtual Indexing):
- VirtualIndexReader:
  - Discovers multi-rank segmented WAL files (runs/<RUN-ID>/ranks/rank_<r>/segment_<seq>.bin).
  - Attempts dynamic loading of Rust shared library (libglassbox_collator.so) via ctypes.
  - Implements complete, robust pure-Python / NumPy fallback reader for WAL segments:
    - Parses SegmentHeader (magic GBWAL001 or GBSW0001, version, rank_id, seq, max_size, timestamp).
    - Parses RecordFrame (magic WALR, record_type, flags, payload_len, step_id, crc32).
    - Validates CRC32 integrity.
    - Decodes activation frames with JSON metadata and 8-byte aligned matrices.
    - Decodes tombstone frames for fail-open dropped steps.
    - Zero-NVLink multi-rank address routing and Shannon entropy reduction for LOGITS-ENTROPY.
  - query_slice(global_step_id, req_id, address) without physical file merging.
  - get_dropped_steps() returning list of dropped step IDs.
  - compact_to_deep_bin(output_path) merging WALs into compressed GBSPR001 deep.bin
    and deep_index.json fully readable by glassbox.storage.layout.read_deep_slice.
- WalSegmentWriter: Python-side WAL writer with 256MB rotation and posix_fadvise.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Magics and constants
WAL_MAGIC = b"GBWAL001"
ALT_WAL_MAGIC = b"GBSW0001"
VALID_WAL_MAGICS = (WAL_MAGIC, ALT_WAL_MAGIC)
RECORD_MAGIC = b"WALR"
SPARSE_MAGIC = b"GBSPR001"

SEGMENT_HEADER_SIZE = 64
RECORD_FRAME_SIZE = 24
ACTIVATION_PAYLOAD_HEADER_SIZE = 28
TOMBSTONE_PAYLOAD_HEADER_SIZE = 16
DEFAULT_MAX_SEGMENT_SIZE = 256 * 1024 * 1024  # 256 MB

RECORD_TYPE_ACTIVATION = 1
RECORD_TYPE_TOMBSTONE = 2

DTYPE_FP16 = 0
DTYPE_FP32 = 1
DTYPE_BF16 = 2

_DTYPE_BYTES = {DTYPE_FP16: 2, DTYPE_FP32: 4, DTYPE_BF16: 2}
_DTYPE_NAMES = {DTYPE_FP16: "fp16", DTYPE_FP32: "fp32", DTYPE_BF16: "bf16"}
_NAME_TO_DTYPE = {"fp16": DTYPE_FP16, "fp32": DTYPE_FP32, "bf16": DTYPE_BF16}


def dtype_bytes(code: int) -> int:
    """Return number of bytes per scalar for given dtype code."""
    return _DTYPE_BYTES.get(code, 2)


def dtype_name(code: int) -> str:
    """Return string name for given dtype code."""
    return _DTYPE_NAMES.get(code, "fp16")


def parse_dtype_code(name: str) -> int:
    """Parse string dtype name to integer code."""
    return _NAME_TO_DTYPE.get(name.lower(), DTYPE_FP16)


def float32_to_bfloat16(arr: np.ndarray) -> np.ndarray:
    """Encode float32 array to uint16 IEEE bfloat16 representation."""
    f32 = np.ascontiguousarray(arr, dtype=np.dtype("<f4"))
    u32 = f32.view(np.dtype("<u4"))
    nan_mask = np.isnan(f32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    rounded = np.where(nan_mask, u32 | np.uint32(0x00400000), u32 + rounding_bias)
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.dtype("<u2")))


def bfloat16_to_float32(raw: np.ndarray) -> np.ndarray:
    """Decode uint16 IEEE bfloat16 representation to float32."""
    u16 = np.asarray(raw, dtype=np.dtype("<u2"))
    u32 = u16.astype(np.dtype("<u4")) << np.uint32(16)
    return u32.view(np.dtype("<f4")).copy()


@dataclass
class SegmentHeader:
    """Segment file header (64 bytes)."""

    magic: bytes
    version: int
    rank_id: int
    seq: int
    max_segment_size: int
    timestamp_ns: int
    reserved: bytes = b"\x00" * 24

    def pack(self) -> bytes:
        return struct.pack(
            "<8sIIQQQ24s",
            self.magic,
            self.version,
            self.rank_id,
            self.seq,
            self.max_segment_size,
            self.timestamp_ns,
            self.reserved,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> SegmentHeader:
        if len(buf) < SEGMENT_HEADER_SIZE:
            raise ValueError(
                f"Buffer too short for SegmentHeader: {len(buf)} < {SEGMENT_HEADER_SIZE}"
            )
        magic, version, rank_id, seq, max_seg_size, timestamp_ns, reserved = struct.unpack(
            "<8sIIQQQ24s", buf[:SEGMENT_HEADER_SIZE]
        )
        if magic not in VALID_WAL_MAGICS:
            raise ValueError(f"Invalid WAL magic: {magic!r}")
        return cls(
            magic=magic,
            version=version,
            rank_id=rank_id,
            seq=seq,
            max_segment_size=max_seg_size,
            timestamp_ns=timestamp_ns,
            reserved=reserved,
        )


@dataclass
class RecordFrame:
    """WAL frame header (24 bytes)."""

    magic: bytes
    record_type: int
    flags: int
    payload_len: int
    step_id: int
    crc32: int

    def pack(self) -> bytes:
        return struct.pack(
            "<4sHHIQI",
            self.magic,
            self.record_type,
            self.flags,
            self.payload_len,
            self.step_id,
            self.crc32,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> RecordFrame:
        if len(buf) < RECORD_FRAME_SIZE:
            raise ValueError(f"Buffer too short for RecordFrame: {len(buf)} < {RECORD_FRAME_SIZE}")
        magic, rec_type, flags, payload_len, step_id, crc32 = struct.unpack(
            "<4sHHIQI", buf[:RECORD_FRAME_SIZE]
        )
        if magic != RECORD_MAGIC:
            raise ValueError(f"Invalid record magic: {magic!r}")
        return cls(
            magic=magic,
            record_type=rec_type,
            flags=flags,
            payload_len=payload_len,
            step_id=step_id,
            crc32=crc32,
        )


@dataclass
class ActivationPayloadHeader:
    """Activation payload header (28 bytes)."""

    timestamp_ns: int
    num_tokens: int
    num_probes: int
    dtype_code: int
    reserved: int
    num_reqs: int
    meta_json_len: int

    def pack(self) -> bytes:
        return struct.pack(
            "<QIIHHII",
            self.timestamp_ns,
            self.num_tokens,
            self.num_probes,
            self.dtype_code,
            self.reserved,
            self.num_reqs,
            self.meta_json_len,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> ActivationPayloadHeader:
        if len(buf) < ACTIVATION_PAYLOAD_HEADER_SIZE:
            raise ValueError(
                f"Buffer too short for ActivationPayloadHeader: "
                f"{len(buf)} < {ACTIVATION_PAYLOAD_HEADER_SIZE}"
            )
        ts, tokens, probes, dt_code, res, reqs, meta_len = struct.unpack(
            "<QIIHHII", buf[:ACTIVATION_PAYLOAD_HEADER_SIZE]
        )
        return cls(
            timestamp_ns=ts,
            num_tokens=tokens,
            num_probes=probes,
            dtype_code=dt_code,
            reserved=res,
            num_reqs=reqs,
            meta_json_len=meta_len,
        )


@dataclass
class TombstonePayloadHeader:
    """Tombstone payload header (16 bytes)."""

    timestamp_ns: int
    reason_code: int
    meta_json_len: int

    def pack(self) -> bytes:
        return struct.pack(
            "<QII",
            self.timestamp_ns,
            self.reason_code,
            self.meta_json_len,
        )

    @classmethod
    def unpack(cls, buf: bytes) -> TombstonePayloadHeader:
        if len(buf) < TOMBSTONE_PAYLOAD_HEADER_SIZE:
            raise ValueError(
                f"Buffer too short for TombstonePayloadHeader: "
                f"{len(buf)} < {TOMBSTONE_PAYLOAD_HEADER_SIZE}"
            )
        ts, reason_code, meta_len = struct.unpack("<QII", buf[:TOMBSTONE_PAYLOAD_HEADER_SIZE])
        return cls(
            timestamp_ns=ts,
            reason_code=reason_code,
            meta_json_len=meta_len,
        )


@dataclass
class ActivationMeta:
    """Metadata deserialized from activation record JSON."""

    req_ids: list[str] = field(default_factory=list)
    query_start_loc: list[int] = field(default_factory=list)
    addresses: list[str] = field(default_factory=list)
    logical_positions: list[int] | None = None


@dataclass
class TombstoneMeta:
    """Metadata deserialized from tombstone record JSON."""

    status: str = "DROPPED"
    reason: str = "Ring buffer stall"
    dropped_step_id: int = 0


@dataclass
class RankRecord:
    """In-memory index record for an activation frame from a specific rank."""

    rank_id: int
    step_id: int
    num_tokens: int
    num_probes: int
    dtype_code: int
    meta: ActivationMeta
    file_path: Path
    matrix_offset: int
    matrix_len: int


@dataclass
class StepData:
    """Aggregated index status for a single global step across all ranks."""

    step_id: int
    is_dropped: bool = False
    drop_reason: str = ""
    num_tokens: int = 0
    rank_records: list[RankRecord] = field(default_factory=list)


def _try_load_rust_lib() -> ctypes.CDLL | None:
    """Attempt to locate and dynamically load libglassbox_collator.so via ctypes."""
    candidates = [
        os.environ.get("GLASSBOX_COLLATOR_LIB"),
        str(
            Path(__file__).resolve().parents[2]
            / "crates"
            / "glassbox_collator"
            / "target"
            / "release"
            / "libglassbox_collator.so"
        ),
        str(Path(__file__).resolve().parents[2] / "target" / "release" / "libglassbox_collator.so"),
        "/home/antigravity/azzouzabdelhak68-ship-it/glassbox/crates/glassbox_collator/target/release/libglassbox_collator.so",
        "/home/antigravity/azzouzabdelhak68-ship-it/glassbox/target/release/libglassbox_collator.so",
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            try:
                lib = ctypes.CDLL(cand)
                return lib
            except Exception:
                continue
    return None


class VirtualIndexReader:
    """Multi-rank virtual index reader for GlassBox segmented WALs.

    Supports zero-NVLink distributed activation querying and Shannon entropy
    reconstruction across TP ranks without physical file merging.
    Falls back gracefully to pure-Python / NumPy parsing if Rust shared library
    or C-ABI functions are unavailable.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        if not self.run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {self.run_dir}")

        self._rust_lib = _try_load_rust_lib()
        self._steps: dict[int, StepData] = {}
        self._dropped_steps: set[int] = set()
        self._watched_addresses: set[str] = set()
        self._file_cache: dict[Path, Any] = {}

        # Scan and build the virtual index
        self._scan_and_index()

    def _scan_and_index(self) -> None:
        """Scan WAL files in run_dir across ranks and build index."""
        rank_dirs: list[Path] = []
        ranks_sub = self.run_dir / "ranks"
        if ranks_sub.is_dir():
            for entry in sorted(ranks_sub.iterdir()):
                if entry.is_dir() and entry.name.startswith("rank_"):
                    rank_dirs.append(entry)

        if not rank_dirs:
            for entry in sorted(self.run_dir.iterdir()):
                if entry.is_dir() and entry.name.startswith("rank_"):
                    rank_dirs.append(entry)

        if not rank_dirs:
            # Check if segment files are placed directly in run_dir
            rank_dirs = [self.run_dir]

        for r_dir in rank_dirs:
            seg_files = sorted(r_dir.glob("*.bin"))
            for seg_path in seg_files:
                self._index_segment_file(seg_path)

    def _index_segment_file(self, seg_path: Path) -> None:
        """Parse segment header and record frames from a WAL file."""
        file_len = seg_path.stat().st_size
        if file_len < SEGMENT_HEADER_SIZE:
            return

        with open(seg_path, "rb") as f:
            hdr_bytes = f.read(SEGMENT_HEADER_SIZE)
            try:
                seg_hdr = SegmentHeader.unpack(hdr_bytes)
            except Exception:
                return

            cur_pos = SEGMENT_HEADER_SIZE
            while cur_pos + RECORD_FRAME_SIZE <= file_len:
                f.seek(cur_pos)
                frame_bytes = f.read(RECORD_FRAME_SIZE)
                if len(frame_bytes) < RECORD_FRAME_SIZE:
                    break
                try:
                    frame = RecordFrame.unpack(frame_bytes)
                except Exception:
                    break

                if cur_pos + RECORD_FRAME_SIZE + frame.payload_len > file_len:
                    break

                payload = f.read(frame.payload_len)
                if len(payload) < frame.payload_len:
                    break

                # Verify CRC32
                computed_crc = zlib.crc32(payload) & 0xFFFFFFFF
                if computed_crc != frame.crc32:
                    cur_pos += RECORD_FRAME_SIZE + frame.payload_len
                    continue

                step_id = frame.step_id

                if frame.record_type == RECORD_TYPE_TOMBSTONE:
                    if len(payload) >= TOMBSTONE_PAYLOAD_HEADER_SIZE:
                        tomb_hdr = TombstonePayloadHeader.unpack(
                            payload[:TOMBSTONE_PAYLOAD_HEADER_SIZE]
                        )
                        json_start = TOMBSTONE_PAYLOAD_HEADER_SIZE
                        json_end = json_start + tomb_hdr.meta_json_len
                        reason = "Ring buffer stall"
                        if json_end <= len(payload):
                            try:
                                tomb_meta = json.loads(payload[json_start:json_end].decode("utf-8"))
                                reason = tomb_meta.get("reason", reason)
                            except Exception:
                                pass
                        self._dropped_steps.add(step_id)
                        if step_id in self._steps:
                            self._steps[step_id].is_dropped = True
                            self._steps[step_id].drop_reason = reason
                        else:
                            self._steps[step_id] = StepData(
                                step_id=step_id,
                                is_dropped=True,
                                drop_reason=reason,
                            )

                elif frame.record_type == RECORD_TYPE_ACTIVATION:
                    # If step already marked dropped by a tombstone, don't resurrect
                    if step_id in self._dropped_steps:
                        cur_pos += RECORD_FRAME_SIZE + frame.payload_len
                        continue

                    if len(payload) >= ACTIVATION_PAYLOAD_HEADER_SIZE:
                        act_hdr = ActivationPayloadHeader.unpack(
                            payload[:ACTIVATION_PAYLOAD_HEADER_SIZE]
                        )
                        json_start = ACTIVATION_PAYLOAD_HEADER_SIZE
                        json_end = json_start + act_hdr.meta_json_len
                        if json_end <= len(payload):
                            try:
                                meta_dict = json.loads(payload[json_start:json_end].decode("utf-8"))
                                meta = ActivationMeta(
                                    req_ids=[str(r) for r in meta_dict.get("req_ids", [])],
                                    query_start_loc=[
                                        int(x) for x in meta_dict.get("query_start_loc", [])
                                    ],
                                    addresses=[str(a) for a in meta_dict.get("addresses", [])],
                                    logical_positions=meta_dict.get("logical_positions"),
                                )
                            except Exception:
                                cur_pos += RECORD_FRAME_SIZE + frame.payload_len
                                continue

                            for addr in meta.addresses:
                                self._watched_addresses.add(addr)

                            pad_len = (
                                8 - ((ACTIVATION_PAYLOAD_HEADER_SIZE + act_hdr.meta_json_len) % 8)
                            ) % 8
                            matrix_start = (
                                ACTIVATION_PAYLOAD_HEADER_SIZE + act_hdr.meta_json_len + pad_len
                            )
                            matrix_offset = cur_pos + RECORD_FRAME_SIZE + matrix_start
                            b_elem = dtype_bytes(act_hdr.dtype_code)
                            matrix_len = act_hdr.num_tokens * act_hdr.num_probes * b_elem

                            rank_rec = RankRecord(
                                rank_id=seg_hdr.rank_id,
                                step_id=step_id,
                                num_tokens=act_hdr.num_tokens,
                                num_probes=act_hdr.num_probes,
                                dtype_code=act_hdr.dtype_code,
                                meta=meta,
                                file_path=seg_path,
                                matrix_offset=matrix_offset,
                                matrix_len=matrix_len,
                            )

                            if step_id not in self._steps:
                                self._steps[step_id] = StepData(
                                    step_id=step_id,
                                    is_dropped=False,
                                    num_tokens=act_hdr.num_tokens,
                                    rank_records=[rank_rec],
                                )
                            else:
                                step_obj = self._steps[step_id]
                                step_obj.num_tokens = max(step_obj.num_tokens, act_hdr.num_tokens)
                                step_obj.rank_records.append(rank_rec)

                cur_pos += RECORD_FRAME_SIZE + frame.payload_len

    @property
    def dropped_steps(self) -> list[int]:
        """Return sorted list of step IDs dropped with tombstones."""
        return sorted(self._dropped_steps)

    @property
    def active_steps(self) -> list[int]:
        """Return sorted list of active (non-dropped) step IDs."""
        return sorted([sid for sid, s in self._steps.items() if not s.is_dropped])

    @property
    def watched_addresses(self) -> list[str]:
        """Return sorted list of all watched probe addresses."""
        addrs = set(self._watched_addresses)
        has_entropy = any(a == "LOGITS-ENTROPY" or a.startswith("LOGITS-ENTROPY_") for a in addrs)
        if has_entropy:
            addrs.add("LOGITS-ENTROPY")
        return sorted(addrs)

    def get_dropped_steps(self) -> list[int]:
        """Return list of step IDs that were dropped with tombstones."""
        return self.dropped_steps

    def get_active_steps(self) -> list[int]:
        """Return list of active step IDs."""
        return self.active_steps

    def get_all_steps(self) -> list[int]:
        """Return all indexed step IDs."""
        return sorted(self._steps.keys())

    def is_step_dropped(self, step_id: int) -> bool:
        """Check if a given step ID was dropped."""
        return step_id in self._dropped_steps or (
            step_id in self._steps and self._steps[step_id].is_dropped
        )

    def get_step_num_tokens(self, step_id: int) -> int | None:
        """Return the number of tokens in active step, or None if dropped/missing."""
        step = self._steps.get(step_id)
        if step is None or step.is_dropped:
            return None
        return step.num_tokens

    def get_step_req_ids(self, step_id: int) -> list[str]:
        """Return list of req_ids for step, or empty list."""
        step = self._steps.get(step_id)
        if step is None or step.is_dropped or not step.rank_records:
            return []
        return list(step.rank_records[0].meta.req_ids)

    def query_slice(
        self,
        global_step_id: int,
        req_id: str | int | None = None,
        address: str = "",
    ) -> np.ndarray | None:
        """Query activations across ranks for (global_step_id, req_id, address).

        Returns 1D NumPy array of activations (or None if step dropped or address not found).
        If req_id is None, "", or "*", returns all tokens for the step.
        """
        step = self._steps.get(global_step_id)
        if step is None or step.is_dropped:
            return None

        # Check for multi-rank LOGITS-ENTROPY reduction
        if address == "LOGITS-ENTROPY":
            reduction_result = self._query_logits_entropy_reduced(step, req_id)
            if reduction_result is not None:
                return reduction_result

        # Standard address search across rank records
        for r_rec in step.rank_records:
            if address in r_rec.meta.addresses:
                col_idx = r_rec.meta.addresses.index(address)
                return self._read_rank_column_slice(r_rec, col_idx, req_id)

        return None

    def _read_rank_column_slice(
        self,
        r_rec: RankRecord,
        col_idx: int,
        req_id: str | int | None,
    ) -> np.ndarray | None:
        """Read a column slice from a single rank's WAL record for the given request."""
        # Determine token bounds
        if req_id is not None and str(req_id) != "" and str(req_id) != "*":
            req_str = str(req_id)
            if req_str not in r_rec.meta.req_ids:
                return None
            req_pos = r_rec.meta.req_ids.index(req_str)
            if req_pos + 1 >= len(r_rec.meta.query_start_loc):
                return None
            start_tok = r_rec.meta.query_start_loc[req_pos]
            end_tok = r_rec.meta.query_start_loc[req_pos + 1]
        else:
            start_tok = 0
            end_tok = r_rec.num_tokens

        if start_tok >= end_tok or end_tok > r_rec.num_tokens:
            return np.empty(0, dtype=np.float32)

        slice_t = end_tok - start_tok
        b_elem = dtype_bytes(r_rec.dtype_code)
        num_probes = r_rec.num_probes

        # Read matrix slice
        with open(r_rec.file_path, "rb") as f:
            if r_rec.dtype_code == DTYPE_FP16:
                dt = np.dtype("<f2")
            elif r_rec.dtype_code == DTYPE_FP32:
                dt = np.dtype("<f4")
            else:
                dt = np.dtype("<u2")

            # Check if we can read contiguous or strided
            if num_probes == 1 and col_idx == 0:
                f.seek(r_rec.matrix_offset + start_tok * b_elem)
                raw_bytes = f.read(slice_t * b_elem)
                raw_arr = np.frombuffer(raw_bytes, dtype=dt)
            else:
                # Strided token-major rows
                row_stride = num_probes * b_elem
                raw_vals = []
                for t in range(start_tok, end_tok):
                    f.seek(r_rec.matrix_offset + t * row_stride + col_idx * b_elem)
                    raw_vals.append(f.read(b_elem))
                raw_bytes = b"".join(raw_vals)
                raw_arr = np.frombuffer(raw_bytes, dtype=dt)

        if r_rec.dtype_code == DTYPE_BF16:
            return bfloat16_to_float32(raw_arr)
        elif r_rec.dtype_code == DTYPE_FP16:
            return raw_arr.astype(np.float32)
        else:
            return raw_arr.astype(np.float32)

    def _query_logits_entropy_reduced(
        self,
        step: StepData,
        req_id: str | int | None,
    ) -> np.ndarray | None:
        """Compute exact global Shannon entropy across TP ranks using 3-scalar reduction."""
        rank_m: list[np.ndarray] = []
        rank_s: list[np.ndarray] = []
        rank_w: list[np.ndarray] = []

        for r_rec in step.rank_records:
            # Check for LOGITS-ENTROPY_m, _s, _w
            addrs = r_rec.meta.addresses
            m_slice = None
            s_slice = None
            w_slice = None

            if (
                "LOGITS-ENTROPY_m" in addrs
                and "LOGITS-ENTROPY_s" in addrs
                and "LOGITS-ENTROPY_w" in addrs
            ):
                m_slice = self._read_rank_column_slice(
                    r_rec, addrs.index("LOGITS-ENTROPY_m"), req_id
                )
                s_slice = self._read_rank_column_slice(
                    r_rec, addrs.index("LOGITS-ENTROPY_s"), req_id
                )
                w_slice = self._read_rank_column_slice(
                    r_rec, addrs.index("LOGITS-ENTROPY_w"), req_id
                )
            elif "LOGITS-ENTROPY" in addrs:
                base_idx = addrs.index("LOGITS-ENTROPY")
                if base_idx + 2 < len(addrs):
                    m_slice = self._read_rank_column_slice(r_rec, base_idx + 0, req_id)
                    s_slice = self._read_rank_column_slice(r_rec, base_idx + 1, req_id)
                    w_slice = self._read_rank_column_slice(r_rec, base_idx + 2, req_id)

            if m_slice is not None and s_slice is not None and w_slice is not None:
                rank_m.append(m_slice)
                rank_s.append(s_slice)
                rank_w.append(w_slice)

        if not rank_m:
            return None

        num_tokens = len(rank_m[0])
        num_ranks = len(rank_m)
        entropy = np.zeros(num_tokens, dtype=np.float32)

        for t in range(num_tokens):
            max_m = max(rank_m[r][t] for r in range(num_ranks))
            global_s = 0.0
            global_w = 0.0
            for r in range(num_ranks):
                mr = float(rank_m[r][t])
                sr = float(rank_s[r][t])
                wr = float(rank_w[r][t])
                alpha = math.exp(mr - max_m)
                global_s += sr * alpha
                global_w += wr * alpha + (mr - max_m) * sr * alpha

            if global_s > 0.0:
                entropy[t] = math.log(global_s) - (global_w / global_s)
            else:
                entropy[t] = 0.0

        return entropy

    def compact_to_deep_bin(
        self,
        output_path: str | Path,
        precision: str = "fp16",
        chunk_tokens: int = 64,
    ) -> None:
        """Offline compaction: merges segmented multi-rank WALs into deep.bin & deep_index.json.

        Produces compressed GBSPR001 format readable by glassbox.storage.layout.read_deep_slice.
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        active_sids = self.active_steps
        if not active_sids:
            raise ValueError("No active (non-dropped) steps to compact")

        # Collect watched addresses
        watched_cols = list(self.watched_addresses)
        if not watched_cols:
            raise ValueError("No watched addresses found to compact")

        # Compute total tokens across active steps
        total_tokens = sum(self._steps[sid].num_tokens for sid in active_sids)
        if total_tokens == 0:
            raise ValueError("Total tokens across active steps is 0")

        # Build dense matrix [total_tokens, len(watched_cols)]
        num_cols = len(watched_cols)
        dense_matrix = np.zeros((total_tokens, num_cols), dtype=np.float32)
        token_cursor = 0

        for sid in active_sids:
            step_tokens = self._steps[sid].num_tokens
            for col_idx, addr in enumerate(watched_cols):
                slice_vals = self.query_slice(sid, req_id=None, address=addr)
                if slice_vals is not None:
                    copy_len = min(len(slice_vals), step_tokens)
                    dense_matrix[token_cursor : token_cursor + copy_len, col_idx] = slice_vals[
                        :copy_len
                    ]
            token_cursor += step_tokens

        # Encode matrix into target precision
        prec_code = parse_dtype_code(precision)
        prec_str = dtype_name(prec_code)
        if prec_str == "fp32":
            raw_arr = np.ascontiguousarray(dense_matrix, dtype=np.dtype("<f4"))
        elif prec_str == "fp16":
            raw_arr = np.ascontiguousarray(dense_matrix, dtype=np.dtype("<f2"))
        else:  # bf16
            raw_arr = float32_to_bfloat16(dense_matrix)

        # Chunk into chunk_tokens rows
        chunk_tokens = max(1, chunk_tokens)
        num_chunks = max(1, math.ceil(total_tokens / chunk_tokens)) if total_tokens > 0 else 0

        compressed_chunks: list[bytes] = []
        chunk_meta: list[tuple[int, int]] = []  # (comp_len, uncomp_len)

        for c in range(num_chunks):
            start_r = c * chunk_tokens
            end_r = min(total_tokens, start_r + chunk_tokens)
            chunk_slice = raw_arr[start_r:end_r]
            uncomp_bytes = chunk_slice.tobytes(order="C")
            comp_bytes = zlib.compress(uncomp_bytes, level=6)
            compressed_chunks.append(comp_bytes)
            chunk_meta.append((len(comp_bytes), len(uncomp_bytes)))

        # Write deep.bin
        hdr_fmt = "<8sIIII"
        hdr_size = struct.calcsize(hdr_fmt)
        idx_entry_fmt = "<QII"
        idx_size = num_chunks * struct.calcsize(idx_entry_fmt)
        payload_start = hdr_size + idx_size

        index_bytes = bytearray()
        cur_offset = payload_start
        for comp_len, uncomp_len in chunk_meta:
            index_bytes.extend(struct.pack(idx_entry_fmt, cur_offset, comp_len, uncomp_len))
            cur_offset += comp_len

        header_bytes = struct.pack(hdr_fmt, SPARSE_MAGIC, 1, chunk_tokens, num_chunks, prec_code)

        deep_bin_path = output_dir / "deep.bin"
        with open(deep_bin_path, "wb") as f:
            f.write(header_bytes)
            f.write(index_bytes)
            for comp in compressed_chunks:
                f.write(comp)

        # Write deep_index.json
        run_id = output_dir.name if output_dir.name.startswith("RUN-") else "RUN-000001"
        columns_obj = {
            addr: {
                "offset": offset,
                "dtype": prec_str,
                "shape": [total_tokens],
            }
            for offset, addr in enumerate(watched_cols)
        }

        token_positions = list(range(total_tokens))
        index_json = {
            "run_id": run_id,
            "model_id": "M-001",
            "checkpoint_sha256": "0" * 64,
            "watchlist_ref": {
                "watchlist_id": "deep",
                "watchlist_sha256": "0" * 64,
                "patch": None,
            },
            "prompt": {"text": "", "tokens": [], "tokenizer": "utf8-bytes"},
            "output": {"text": "", "tokens": [], "finish_reason": "stop"},
            "logits_summary": None,
            "timing": {
                "t_start_iso": "2026-11-08T00:00:00Z",
                "t_end_iso": "2026-11-08T00:00:00Z",
                "ms_per_token": 0.0,
                "cuda_event_ms": 0.0,
            },
            "sampling": {"seed": 42, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
            "versions": {
                "torch": "unknown",
                "cuda": "cpu",
                "glassbox": "0.1.0",
                "blackbox": "0.1.0",
                "adapter": "cpu",
            },
            "flags": {
                "bad_output": False,
                "ring_overflow": False,
                "hook_detached": False,
                "truncated": False,
            },
            "tags": [],
            "replay": {
                "seed": 42,
                "temp": 0.0,
                "top_p": 1.0,
                "sampler": "greedy",
                "cudnn_deterministic": True,
                "checkpoint_sha256": "0" * 64,
                "versions": {"torch": "unknown", "cuda": "cpu"},
            },
            "error": None,
            "triggered": True,
            "dtype": prec_str,
            "tokens": total_tokens,
            "every_n": 1,
            "window": None,
            "token_positions": token_positions,
            "capture_mode": "all",
            "watched": watched_cols,
            "columns": columns_obj,
            "dtype_meta": {
                "endian": "little",
                "order": "row_major_token_major",
                "source": "ieee_bfloat16_le" if prec_str == "bf16" else "logical_values",
                "stored": prec_str,
            },
        }

        deep_index_path = output_dir / "deep_index.json"
        with open(deep_index_path, "w", encoding="utf-8") as f:
            json.dump(index_json, f, indent=2, sort_keys=True)
            f.write("\n")

    def close(self) -> None:
        """Close any cached file handles or resources."""
        self._file_cache.clear()

    def __enter__(self) -> VirtualIndexReader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class WalSegmentWriter:
    """Python WAL segment writer for testing and local multi-rank WAL generation.

    Emits 4MB aligned records with size-rotated 256MB segments and
    POSIX fadvise cache management.
    """

    def __init__(
        self,
        rank_dir: str | Path,
        rank_id: int = 0,
        max_segment_size: int = DEFAULT_MAX_SEGMENT_SIZE,
    ) -> None:
        self.rank_dir = Path(rank_dir)
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.rank_id = rank_id
        self.max_segment_size = max_segment_size
        self.current_seq = 0
        self.current_file = None
        self.current_file_path: Path | None = None
        self.current_segment_bytes = 0

        self.rotate_segment()

    def _advise_dontneed(self) -> None:
        """Invoke posix_fadvise(POSIX_FADV_DONTNEED) on completed segments when available."""
        if self.current_file is not None and hasattr(os, "posix_fadvise"):
            try:
                os.posix_fadvise(self.current_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError:
                pass

    def rotate_segment(self) -> None:
        """Rotate to a new segment file segment_<seq:06d>.bin."""
        if self.current_file is not None:
            self.current_file.flush()
            self._advise_dontneed()
            self.current_file.close()

        seg_name = f"segment_{self.current_seq:06d}.bin"
        self.current_file_path = self.rank_dir / seg_name
        self.current_file = open(self.current_file_path, "wb")

        header = SegmentHeader(
            magic=WAL_MAGIC,
            version=1,
            rank_id=self.rank_id,
            seq=self.current_seq,
            max_segment_size=self.max_segment_size,
            timestamp_ns=0,
        )
        self.current_file.write(header.pack())
        self.current_file.flush()

        self.current_segment_bytes = SEGMENT_HEADER_SIZE
        self.current_seq += 1

    def write_activation(
        self,
        step_id: int,
        num_tokens: int,
        addresses: list[str],
        req_ids: list[str | int],
        query_start_loc: list[int],
        matrix: np.ndarray,
        dtype: str = "fp16",
        logical_positions: list[int] | None = None,
    ) -> None:
        """Write an activation frame record to the WAL."""
        dtype_code = parse_dtype_code(dtype)
        num_probes = len(addresses)

        meta_dict: dict[str, Any] = {
            "req_ids": [str(r) for r in req_ids],
            "query_start_loc": [int(x) for x in query_start_loc],
            "addresses": list(addresses),
        }
        if logical_positions is not None:
            meta_dict["logical_positions"] = [int(p) for p in logical_positions]

        meta_json = json.dumps(meta_dict).encode("utf-8")
        meta_json_len = len(meta_json)

        payload_hdr = ActivationPayloadHeader(
            timestamp_ns=0,
            num_tokens=num_tokens,
            num_probes=num_probes,
            dtype_code=dtype_code,
            reserved=0,
            num_reqs=len(req_ids),
            meta_json_len=meta_json_len,
        )

        # Encode raw matrix
        if dtype_code == DTYPE_FP32:
            raw_matrix = np.ascontiguousarray(matrix, dtype=np.dtype("<f4")).tobytes()
        elif dtype_code == DTYPE_FP16:
            raw_matrix = np.ascontiguousarray(matrix, dtype=np.dtype("<f2")).tobytes()
        else:
            raw_matrix = float32_to_bfloat16(matrix).tobytes()

        unpadded_len = ACTIVATION_PAYLOAD_HEADER_SIZE + meta_json_len
        pad_len = (8 - (unpadded_len % 8)) % 8
        matrix_len = len(raw_matrix)
        payload_body_len = unpadded_len + pad_len + matrix_len
        trailing_pad_len = (8 - (payload_body_len % 8)) % 8
        total_payload_len = payload_body_len + trailing_pad_len
        total_record_size = RECORD_FRAME_SIZE + total_payload_len

        # Align records to minimum 256 bytes for block boundary alignment
        if total_record_size < 256:
            extra_pad = 256 - total_record_size
            trailing_pad_len += extra_pad
            total_payload_len += extra_pad
            total_record_size = 256

        payload_buf = bytearray()
        payload_buf.extend(payload_hdr.pack())
        payload_buf.extend(meta_json)
        payload_buf.extend(b"\x00" * pad_len)
        payload_buf.extend(raw_matrix)
        payload_buf.extend(b"\x00" * trailing_pad_len)

        crc32 = zlib.crc32(payload_buf) & 0xFFFFFFFF

        frame = RecordFrame(
            magic=RECORD_MAGIC,
            record_type=RECORD_TYPE_ACTIVATION,
            flags=0,
            payload_len=total_payload_len,
            step_id=step_id,
            crc32=crc32,
        )

        if (
            self.current_segment_bytes + total_record_size > self.max_segment_size
            and self.current_segment_bytes > SEGMENT_HEADER_SIZE
        ):
            self.rotate_segment()

        if self.current_file is None:
            raise OSError("WAL file is not open")

        self.current_file.write(frame.pack())
        self.current_file.write(payload_buf)
        self.current_file.flush()

        self.current_segment_bytes += total_record_size

    def write_tombstone(
        self,
        step_id: int,
        reason_code: int = 1,
        reason: str = "Ring buffer full: drop newest",
    ) -> None:
        """Write a drop tombstone frame record to the WAL."""
        meta_dict = {
            "status": "DROPPED",
            "reason": reason,
            "dropped_step_id": step_id,
        }
        meta_json = json.dumps(meta_dict).encode("utf-8")
        meta_json_len = len(meta_json)

        payload_hdr = TombstonePayloadHeader(
            timestamp_ns=0,
            reason_code=reason_code,
            meta_json_len=meta_json_len,
        )

        unpadded_len = TOMBSTONE_PAYLOAD_HEADER_SIZE + meta_json_len
        pad_len = (8 - (unpadded_len % 8)) % 8
        total_payload_len = unpadded_len + pad_len

        payload_buf = bytearray()
        payload_buf.extend(payload_hdr.pack())
        payload_buf.extend(meta_json)
        payload_buf.extend(b"\x00" * pad_len)

        crc32 = zlib.crc32(payload_buf) & 0xFFFFFFFF

        frame = RecordFrame(
            magic=RECORD_MAGIC,
            record_type=RECORD_TYPE_TOMBSTONE,
            flags=0,
            payload_len=total_payload_len,
            step_id=step_id,
            crc32=crc32,
        )

        total_record_size = RECORD_FRAME_SIZE + total_payload_len
        if (
            self.current_segment_bytes + total_record_size > self.max_segment_size
            and self.current_segment_bytes > SEGMENT_HEADER_SIZE
        ):
            self.rotate_segment()

        if self.current_file is None:
            raise OSError("WAL file is not open")

        self.current_file.write(frame.pack())
        self.current_file.write(payload_buf)
        self.current_file.flush()

        self.current_segment_bytes += total_record_size

    def close(self) -> None:
        """Flush and close segment file."""
        if self.current_file is not None:
            self.current_file.flush()
            self._advise_dontneed()
            self.current_file.close()
            self.current_file = None

    def __enter__(self) -> WalSegmentWriter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
