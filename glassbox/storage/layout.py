"""Store layout: manifests + runs/{deep.bin,deep_index.json,seal.json} + chain (spec §FILE-*).

Deep-Only Architecture:
Every run is a Deep run consisting of:
- ``deep.bin``: COLD raw little-endian activation matrix, row-major ``float[T][W]``
  (token-major: token ``t``, watched index ``j`` at byte ``(t*W + j) * B``).
  ``T`` = actual captured tokens in window (after ``every_n`` subsample),
  ``W`` = ``len(watched)`` in ``deep_index.json`` order.
  ``fp32`` B=4 (IEEE 754 binary32), ``fp16`` B=2 (IEEE 754 binary16),
  ``bf16`` B=2 (true IEEE bfloat16 high-16-bit encoding via uint32>>16).
- ``deep_index.json``: Unified HOT run manifest containing the activation index
  (``triggered, dtype, tokens, every_n, token_positions, capture_mode, watched,
  columns{addr: {offset, stride_bytes, dtype, shape}}, dtype_meta``) AND full
  execution context (``model_id, checkpoint_sha256, watchlist_ref, prompt,
  output, logits_summary, timing, sampling, versions, flags, tags, replay, error``).
- ``seal.json``: Cryptographic hash-chain seal over ``deep_index.json``,
  ``deep.bin``, and canonical ``replay``.

Exit mapping: E02 ``ValueError`` (bad RUN-ID / bad payload / bad slice).
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import zlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

RUN_RE = re.compile(r"^RUN-(\d{6})$")

DTYPE_BYTES: dict[str, int] = {"fp32": 4, "fp16": 2, "bf16": 2}
_LE_DTYPE: dict[str, str] = {"fp32": "<f4", "fp16": "<f2", "bf16": "<u2"}

SPARSE_MAGIC = b"GBSPR001"
LEGACY_SPARSE_MAGIC = b"BBSPR001"
SPARSE_MAGICS = (b"GBSPR001", b"BBSPR001")
_PREC_CODE = {"fp16": 0, "fp32": 1, "bf16": 2}
_CODE_PREC = {0: "fp16", 1: "fp32", 2: "bf16"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def get_store_dir(store: str | None = None) -> str:
    """Resolve store root: arg > ``$GLASSBOX_DIR`` > ``$BLACKBOX_DIR`` > ``./glassbox_store``."""
    if store:
        return store
    if "GLASSBOX_DIR" in os.environ:
        return os.environ["GLASSBOX_DIR"]
    if "BLACKBOX_DIR" in os.environ:
        return os.environ["BLACKBOX_DIR"]
    if os.path.exists("./blackbox_store") and not os.path.exists("./glassbox_store"):
        return "./blackbox_store"
    return "./glassbox_store"


def run_dir(store: str, run_id: str) -> str:
    """Resolve ``$GLASSBOX_DIR/runs/RUN-xxx`` path (validates RUN-ID)."""
    if not RUN_RE.match(run_id):
        raise ValueError(f"E02: bad run id {run_id!r}")
    return os.path.join(store, "runs", run_id)


def _max_run_number(store: str) -> int:
    """Max sealed/allocated run number from chain + run dirs (0 if none)."""
    best = 0
    chain = os.path.join(store, "chain.jsonl")
    if os.path.isfile(chain):
        with open(chain, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    m = RUN_RE.match(str(obj.get("run_id", "")))
                    if m:
                        best = max(best, int(m.group(1)))
                except (ValueError, AttributeError):
                    continue
    runs_root = os.path.join(store, "runs")
    if os.path.isdir(runs_root):
        for name in os.listdir(runs_root):
            m = RUN_RE.match(name)
            if m:
                best = max(best, int(m.group(1)))
    return best


def next_run_id(store: str | None = None) -> str:
    """Next RUN-ID without allocating (``000001`` when store empty)."""
    root = get_store_dir(store)
    return f"RUN-{_max_run_number(root) + 1:06d}"


def allocate_run(store: str | None = None) -> tuple[str, str]:
    """Allocate next RUN-ID and ensure its directory; return ``(id, path)``."""
    root = get_store_dir(store)
    rid = next_run_id(root)
    path = run_dir(root, rid)
    os.makedirs(path, exist_ok=True)
    return (rid, path)


def precision_bytes(precision: str) -> int:
    """Bytes per value for a logical precision (E02 on unknown)."""
    try:
        return DTYPE_BYTES[precision]
    except KeyError as exc:
        raise ValueError(f"E02: bad precision {precision!r}; expected fp32|fp16|bf16") from exc


def expected_deep_bytes(w: int, t: int, precision: str) -> int:
    """Per-run ``deep.bin`` bytes ``W*T*B`` (estimator adds ``R`` + 1.15)."""
    return int(w) * int(t) * precision_bytes(precision)


def encode_deep_array(matrix: np.ndarray, precision: str = "fp16") -> np.ndarray:
    """Encode a 2D ``[T, W]`` matrix into little-endian storage array for ``precision``."""
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        raise ValueError(f"E02: deep matrix must be 2D [T, W], got ndim={arr.ndim}")
    precision_bytes(precision)
    if precision == "fp32":
        return np.ascontiguousarray(arr, dtype=np.dtype("<f4"))
    if precision == "fp16":
        return np.ascontiguousarray(arr, dtype=np.dtype("<f2"))
    # True IEEE bfloat16: upper 16 bits of IEEE 754 binary32 (sign + 8-bit exp + 7-bit mantissa)
    f32 = np.ascontiguousarray(arr, dtype=np.dtype("<f4"))
    u32 = f32.view(np.dtype("<u4"))
    # Round-to-nearest-even for finite values, preserve NaN payload
    nan_mask = np.isnan(f32)
    rounding_bias = np.uint32(0x7FFF) + ((u32 >> np.uint32(16)) & np.uint32(1))
    rounded = np.where(nan_mask, u32 | np.uint32(0x00400000), u32 + rounding_bias)
    return np.ascontiguousarray((rounded >> np.uint32(16)).astype(np.dtype("<u2")))


def decode_deep_array(raw: np.ndarray, precision: str = "fp16") -> np.ndarray:
    """Decode a little-endian storage array back to float32 logical values."""
    precision_bytes(precision)
    if precision == "bf16":
        u16 = np.asarray(raw, dtype=np.dtype("<u2"))
        u32 = u16.astype(np.dtype("<u4")) << np.uint32(16)
        return u32.view(np.dtype("<f4")).copy()
    return np.asarray(raw, dtype=np.float32)


def write_sparse_deep_bin(
    run_path: str,
    matrix: np.ndarray,
    precision: str = "fp16",
    chunk_tokens: int = 64,
) -> str:
    """Write chunked zlib-compressed sparse row-major LE array to ``deep.bin``."""
    base = os.path.basename(os.path.normpath(run_path))
    if not RUN_RE.match(base):
        raise ValueError(f"E02: bad run path {run_path!r}")
    arr = encode_deep_array(matrix, precision)
    tokens = int(arr.shape[0])
    prec_code = _PREC_CODE[precision]

    num_chunks = max(1, math.ceil(tokens / chunk_tokens)) if tokens > 0 else 0
    chunks_data: list[bytes] = []
    chunk_meta: list[tuple[int, int]] = []

    for c in range(num_chunks):
        c_start = c * chunk_tokens
        c_end = min(tokens, c_start + chunk_tokens)
        chunk_arr = arr[c_start:c_end]
        raw_bytes = chunk_arr.tobytes(order="C")
        comp = zlib.compress(raw_bytes, level=6)
        chunks_data.append(comp)
        chunk_meta.append((len(comp), len(raw_bytes)))

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

    os.makedirs(run_path, exist_ok=True)
    out = os.path.join(run_path, "deep.bin")
    with open(out, "wb") as fh:
        fh.write(header_bytes)
        fh.write(index_bytes)
        for comp in chunks_data:
            fh.write(comp)
    return out


def write_deep_bin(
    run_path: str,
    matrix: np.ndarray,
    precision: str = "fp16",
    compression: str | None = None,
    chunk_tokens: int = 64,
) -> str:
    """Write row-major ``[T, W]`` LE ``deep.bin`` via numpy; return its path.

    When ``compression`` is ``"sparse"``, ``"sparse_rle"``, or ``"zlib"``, writes
    chunked compressed blocks allowing 5-10x disk savings while preserving fast
    sub-millisecond slice random access.

    Raises:
        ValueError: E02 bad RUN-ID dir, bad precision, or non-2D matrix.
    """
    if compression in ("sparse", "sparse_rle", "zlib", "compressed"):
        return write_sparse_deep_bin(
            run_path, matrix, precision=precision, chunk_tokens=chunk_tokens
        )
    base = os.path.basename(os.path.normpath(run_path))
    if not RUN_RE.match(base):
        raise ValueError(f"E02: bad run path {run_path!r}")
    arr = encode_deep_array(matrix, precision)
    os.makedirs(run_path, exist_ok=True)
    out = os.path.join(run_path, "deep.bin")
    with open(out, "wb") as fh:
        fh.write(arr.tobytes(order="C"))
    return out


def default_run_context(
    run_id: str,
    model_id: str = "M-001",
    checkpoint_sha256: str = "0" * 64,
    **overrides: Any,
) -> dict[str, Any]:
    """Build standard execution context fields stored inside ``deep_index.json``."""
    now = _utc_now_iso()
    ctx: dict[str, Any] = {
        "run_id": run_id,
        "model_id": model_id,
        "checkpoint_sha256": checkpoint_sha256,
        "watchlist_ref": {
            "watchlist_id": "deep",
            "watchlist_sha256": "0" * 64,
            "patch": None,
        },
        "prompt": {"text": "", "tokens": [], "tokenizer": "utf8-bytes"},
        "output": {"text": "", "tokens": [], "finish_reason": "stop"},
        "logits_summary": None,
        "timing": {
            "t_start_iso": now,
            "t_end_iso": now,
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
            "checkpoint_sha256": checkpoint_sha256,
            "versions": {"torch": "unknown", "cuda": "cpu"},
        },
        "error": None,
    }
    for k, v in overrides.items():
        if v is not None or k in ("error", "logits_summary"):
            ctx[k] = v
    return ctx


def write_deep_index(
    run_path: str,
    run_id: str,
    *,
    triggered: bool,
    dtype: str = "fp16",
    tokens: int = 0,
    every_n: int = 1,
    watched: list[str] | None = None,
    window: dict[str, Any] | None = None,
    dtype_meta: dict[str, Any] | None = None,
    token_positions: list[int] | None = None,
    capture_mode: str = "all",
    context: dict[str, Any] | None = None,
) -> str:
    """Write unified ``deep_index.json`` HOT manifest & activation pointer; return its path.

    Merges any existing or supplied execution ``context`` (prompt, output,
    timing, replay, model_id, checkpoint_sha256, flags, tags) directly into
    ``deep_index.json`` so every Deep run is self-contained without ``outer.json``.
    """
    base = os.path.basename(os.path.normpath(run_path))
    if not RUN_RE.match(base):
        raise ValueError(f"E02: bad run path {run_path!r}")
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"E02: bad dtype {dtype!r}; expected fp32|fp16|bf16")
    if every_n < 1:
        raise ValueError(f"E02: every_n must be >= 1 (got {every_n})")

    out = os.path.join(run_path, "deep_index.json")
    existing: dict[str, Any] = {}
    if os.path.isfile(out):
        try:
            with open(out, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                existing = loaded
        except (OSError, ValueError):
            existing = {}

    merged_ctx = default_run_context(run_id)
    for k, v in existing.items():
        if k in merged_ctx:
            merged_ctx[k] = v
    if context:
        for k, v in context.items():
            merged_ctx[k] = v
    if not isinstance(merged_ctx.get("replay"), dict):
        merged_ctx["replay"] = default_run_context(
            run_id,
            str(merged_ctx.get("model_id", "M-001")),
            str(merged_ctx.get("checkpoint_sha256", "0" * 64)),
        )["replay"]

    names = list(watched) if watched else []
    columns: dict[str, dict[str, Any]] = {}
    if triggered:
        for offset, addr in enumerate(names):
            columns[addr] = {
                "offset": offset,
                "dtype": dtype,
                "shape": [int(tokens)],
            }
    meta: dict[str, Any] = {
        "stored": dtype,
        "source": "ieee_bfloat16_le" if dtype == "bf16" else "logical_values",
        "endian": "little",
        "order": "row_major_token_major",
    }
    if dtype_meta:
        meta.update(dtype_meta)

    positions = (
        [int(p) for p in token_positions]
        if token_positions is not None
        else [i * int(every_n) for i in range(int(tokens))]
    )

    index: dict[str, Any] = {
        **merged_ctx,
        "run_id": run_id,
        "triggered": bool(triggered),
        "dtype": dtype,
        "tokens": int(tokens),
        "every_n": int(every_n),
        "window": dict(window) if window else None,
        "token_positions": positions,
        "capture_mode": capture_mode,
        "watched": names,
        "columns": columns,
        "dtype_meta": meta,
    }
    os.makedirs(run_path, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def write_absent_deep_index(
    run_path: str,
    run_id: str,
    *,
    dtype: str = "fp16",
    every_n: int = 1,
    watched: list[str] | None = None,
    context: dict[str, Any] | None = None,
) -> str:
    """Write an untriggered Deep index (``triggered=false``, no ``deep.bin``)."""
    return write_deep_index(
        run_path,
        run_id,
        triggered=False,
        dtype=dtype,
        tokens=0,
        every_n=every_n,
        watched=watched or [],
        token_positions=[],
        context=context,
    )


def read_deep_index(run_path: str) -> dict[str, Any]:
    """Load ``deep_index.json`` (E02 when missing/not a dict)."""
    path = os.path.join(run_path, "deep_index.json")
    if not os.path.isfile(path):
        raise ValueError(f"E02: no deep_index.json in {run_path}")
    with open(path, encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict):
        raise ValueError(f"E02: bad deep_index.json in {run_path}")
    return index


def _read_sparse_deep_slice(
    deep_path: str,
    token_lo: int,
    token_hi: int,
    w: int,
    dtype: str,
    cols: list[int],
) -> np.ndarray:
    """Read a random-access slice from a chunked sparse ``deep.bin``."""
    hdr_fmt = "<8sIIII"
    hdr_size = struct.calcsize(hdr_fmt)
    idx_entry_fmt = "<QII"
    idx_entry_size = struct.calcsize(idx_entry_fmt)

    with open(deep_path, "rb") as fh:
        hdr_raw = fh.read(hdr_size)
        magic, ver, chunk_tokens, num_chunks, prec_code = struct.unpack(hdr_fmt, hdr_raw)
        if magic not in SPARSE_MAGICS:
            raise ValueError("E02: bad sparse header")

        c_lo = token_lo // chunk_tokens
        c_hi = token_hi // chunk_tokens

        chunks_parts: list[np.ndarray] = []
        raw_dt = np.dtype(_LE_DTYPE[dtype])

        for c in range(c_lo, min(c_hi + 1, num_chunks)):
            fh.seek(hdr_size + c * idx_entry_size)
            offset, comp_len, uncomp_len = struct.unpack(idx_entry_fmt, fh.read(idx_entry_size))
            fh.seek(offset)
            comp_data = fh.read(comp_len)
            decomp = zlib.decompress(comp_data)
            chunk_arr = np.frombuffer(decomp, dtype=raw_dt).reshape(-1, w)

            c_start = c * chunk_tokens
            row_start = max(0, token_lo - c_start)
            row_end = min(chunk_arr.shape[0], token_hi - c_start + 1)
            chunks_parts.append(chunk_arr[row_start:row_end, cols])

    if not chunks_parts:
        return np.empty((0, len(cols)), dtype=np.float32)
    merged = np.concatenate(chunks_parts, axis=0)
    return decode_deep_array(merged, dtype)


def read_deep_slice(
    run_path: str,
    token_lo: int,
    token_hi: int,
    watched_idx: list[int] | None = None,
) -> np.ndarray:
    """Mmap or chunk-read the exact slice ``deep.bin[token_lo..token_hi, watched_idx]``.

    Returns float32 logical values with shape ``[slice_T, len(watched_idx)]``
    (all columns when ``watched_idx`` is None). Only the requested token rows
    are mapped/decompressed via byte offset so multi-GB files are never read in full.

    Raises:
        ValueError: E02 missing Deep, bad range, or bad column index.
    """
    index = read_deep_index(run_path)
    if not index.get("triggered"):
        raise ValueError(f"E02: no Deep for run {index.get('run_id')}: triggered=false")
    watched = list(index.get("watched", []))
    tokens = int(index.get("tokens", 0))
    dtype = str(index.get("dtype", "fp16"))
    if token_lo < 0 or token_hi < token_lo or token_hi >= tokens:
        raise ValueError(f"E02: bad token slice {token_lo}..{token_hi} for T={tokens}")
    cols = list(range(len(watched))) if watched_idx is None else list(watched_idx)
    for j in cols:
        if j < 0 or j >= len(watched):
            raise ValueError(f"E02: bad watched index {j} for W={len(watched)}")
    deep_path = os.path.join(run_path, "deep.bin")
    if not os.path.isfile(deep_path):
        raise ValueError(f"E02: deep_index triggered but no deep.bin in {run_path}")
    b = precision_bytes(dtype)
    w = len(watched)
    actual_bytes = os.path.getsize(deep_path)

    with open(deep_path, "rb") as fh:
        head_magic = fh.read(8)
    if head_magic in SPARSE_MAGICS:
        return _read_sparse_deep_slice(deep_path, token_lo, token_hi, w, dtype, cols)

    expect_bytes = tokens * w * b
    if actual_bytes < expect_bytes:
        raise ValueError(f"E02: deep.bin short: {actual_bytes}B < T*W*B={expect_bytes}B")
    raw_dt = np.dtype(_LE_DTYPE[dtype])
    slice_t = token_hi - token_lo + 1
    byte_offset = token_lo * w * b
    mapped = np.memmap(
        deep_path,
        dtype=raw_dt,
        mode="r",
        offset=byte_offset,
        shape=(slice_t, w),
    )
    part = np.asarray(mapped[:, cols])
    return decode_deep_array(part, dtype)
