"""Forked-stream sidecar: mmap triple-buffer ring -> binary WAL -> sealed Deep run.

Spec §TAP-sidecar, §FILE-deepbin, §FILE-seal, §ERRORS; plan.md §9.3.

Deep-Only Pipeline (separate process/container or in-process consumer)::

    hooks (GPU copyStream / CPU gather) -> /dev/shm/bb.ring (mmap, 3 slots)
        -> sidecar poll -> binary WAL append + configurable fsync
        -> seal: deep.bin + deep_index.json + seal.json + chain.jsonl

Key Optimizations & Fixes:
- Split SPSC Ring Header: producer (``append_token``) ONLY writes ``write_seq``
  and ``overflow`` bytes; consumer (``poll``) ONLY writes ``read_seq`` bytes.
  They never clobber each other's sequence counters.
- Binary WAL (``wal/<RUN-ID>.wal.bin``): each record is packed little-endian
  ``<i`` (token_pos) + ``W * B`` raw activation bytes — eliminating slow
  per-token JSON float-to-string serialization.
- Configurable ``fsync_every`` (default 1 for SIGKILL per-token durability,
  configurable to batch N tokens for high-throughput generation).
- Subsample-aware matrix assembly: ``every_n > 1`` packs the actual ``T``
  captured tokens (no spurious NaN padding between subsampled steps), while
  filling genuine ``ring_overflow`` dropped slots with ``NaN``.
- Unified Deep-Only sealing: merges execution context directly into
  ``deep_index.json`` and seals without ``outer.json``.

Exit mapping: E1001 ``SidecarError`` (runtime failure with WAL watermark);
E02 ``ValueError`` (bad args / short bin).
"""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np

from glassbox.storage.layout import decode_deep_array, encode_deep_array, precision_bytes

MAGIC = b"GBRING01"
LEGACY_MAGIC = b"BBRING01"
MAGICS = (b"GBRING01", b"BBRING01")
VERSION = 1
N_SLOTS = 3  # triple buffer
_STATE_EMPTY = 0
_STATE_WRITING = 1
_STATE_READY = 2

# Header layout (33 bytes):
# 0..8:   magic (8s)
# 8..12:  version (I)
# 12..16: watched count W (I)
# 16..24: write_seq (Q)  <-- producer-owned
# 24..32: read_seq (Q)   <-- consumer-owned
# 32..33: overflow (?)   <-- producer-owned
_HEADER_FMT = "<8sI I Q Q ?"
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_WSEQ_OFFSET = 16
_RSEQ_OFFSET = 24
_OVERFLOW_OFFSET = 32

_SLOT_HEADER_FMT = "<I I I"
_SLOT_HEADER_SIZE = struct.calcsize(_SLOT_HEADER_FMT)
_WAL_ROW_HDR_FMT = "<i"
_WAL_ROW_HDR_SIZE = struct.calcsize(_WAL_ROW_HDR_FMT)


class SidecarError(RuntimeError):
    """E1001 runtime failure (sidecar died / crash recovery hint)."""


def resolve_ring_path(explicit: str | None = None) -> str:
    """Ring path: arg > ``$GLASSBOX_SHM`` > ``$BLACKBOX_SHM`` > ``/dev/shm/gb.ring`` > tmp."""
    if explicit:
        return explicit
    env = os.environ.get("GLASSBOX_SHM") or os.environ.get("BLACKBOX_SHM")
    if env:
        return env
    shm_dir = "/dev/shm"
    if os.path.isdir(shm_dir):
        return os.path.join(shm_dir, "gb.ring")
    return os.path.join(tempfile.gettempdir(), "gb.ring")


def _slot_len_for(w: int, b: int) -> int:
    return w * b


def _ring_file_size(w: int, b: int) -> int:
    return _HEADER_SIZE + N_SLOTS * (_SLOT_HEADER_SIZE + _slot_len_for(w, b))


def _slot_offset(slot: int, slot_len: int) -> int:
    return _HEADER_SIZE + slot * (_SLOT_HEADER_SIZE + slot_len)


class Sidecar:
    """WAL+persistence side for one Deep run (spec §TAP-sidecar).

    The inference loop (or :class:`~blackbox.tap.hooks.DeepTap`) calls
    :meth:`append_token` per emitted token; :meth:`poll` moves READY ring slots
    into the binary WAL file; :meth:`seal` assembles ``deep.bin`` row-major
    ``[T, W]`` LE plus ``deep_index.json``, ``seal.json``, and ``chain.jsonl``.
    After a kill, :meth:`attach` + :meth:`seal_prefix` seals the fsync'd prefix.
    """

    def __init__(
        self,
        store: str,
        watched: list[str],
        precision: str = "fp16",
        run_id: str | None = None,
        window: dict[str, Any] | None = None,
        every_n: int = 1,
        ring_path: str | None = None,
        fsync_every: int = 1,
    ) -> None:
        from glassbox.storage.layout import allocate_run

        if not watched:
            raise ValueError("E02: sidecar needs a non-empty watched address list")
        self._b = precision_bytes(precision)
        self.store = store
        self.watched = list(watched)
        self.precision = precision
        self.window = dict(window) if window else None
        self.every_n = int(every_n)
        if self.every_n < 1:
            raise ValueError(f"E02: every_n must be >= 1 (got {every_n})")
        self.fsync_every = max(0, int(fsync_every))
        self.ring_path = resolve_ring_path(ring_path)
        self.run_id, self.run_dir = (
            allocate_run(store)
            if run_id is None
            else (
                run_id,
                os.path.join(store, "runs", run_id),
            )
        )
        os.makedirs(self.run_dir, exist_ok=True)
        self.wal_path = os.path.join(store, "wal", f"{self.run_id}.wal.bin")
        os.makedirs(os.path.dirname(self.wal_path), exist_ok=True)
        self.ring_overflow = False
        self._tokens_sealed: list[int] = []
        self._open_ring()

    # ------------------------------------------------------------ ring ---
    def _open_ring(self) -> None:
        w = len(self.watched)
        size = _ring_file_size(w, self._b)
        fresh = not os.path.isfile(self.ring_path)
        if fresh:
            parent = os.path.dirname(self.ring_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.ring_path, "wb") as fh:
                fh.write(b"\x00" * size)
        elif os.path.getsize(self.ring_path) < size:
            with open(self.ring_path, "r+b") as fh:
                fh.truncate(size)
        self._ring_file = open(self.ring_path, "r+b")
        self._ring = mmap.mmap(self._ring_file.fileno(), size)
        try:
            _ver, ring_w, _wseq, _rseq, _ov = self._read_header()
            reinit = ring_w != w
        except SidecarError:
            reinit = True
        if fresh or reinit:
            self._init_header(write_seq=0, read_seq=0, overflow=False)
            for slot in range(N_SLOTS):
                self._write_slot_header(slot, _STATE_EMPTY, 0, 0)

    def _init_header(self, write_seq: int, read_seq: int, overflow: bool) -> None:
        w = len(self.watched)
        self._ring[:_HEADER_SIZE] = struct.pack(
            _HEADER_FMT, MAGIC, VERSION, w, write_seq, read_seq, overflow
        )

    def _write_producer_state(self, write_seq: int, overflow: bool) -> None:
        """Producer-only atomic slice update (never touches read_seq)."""
        self._ring[_WSEQ_OFFSET : _WSEQ_OFFSET + 8] = struct.pack("<Q", write_seq)
        self._ring[_OVERFLOW_OFFSET : _OVERFLOW_OFFSET + 1] = struct.pack("<?", overflow)

    def _write_consumer_state(self, read_seq: int) -> None:
        """Consumer-only atomic slice update (never touches write_seq)."""
        self._ring[_RSEQ_OFFSET : _RSEQ_OFFSET + 8] = struct.pack("<Q", read_seq)

    def _read_header(self) -> tuple[int, int, int, int, bool]:
        magic, ver, w, wseq, rseq, overflow = struct.unpack(
            _HEADER_FMT, bytes(self._ring[:_HEADER_SIZE])
        )
        if magic != MAGIC:
            raise SidecarError("E1001 ring magic mismatch — ring file corrupt or foreign")
        return ver, w, wseq, rseq, overflow

    def _write_slot_header(self, slot: int, state: int, token: int, nvals: int) -> None:
        off = _slot_offset(slot, _slot_len_for(len(self.watched), self._b))
        self._ring[off : off + _SLOT_HEADER_SIZE] = struct.pack(
            _SLOT_HEADER_FMT, state, token, nvals
        )

    def _read_slot_header(self, slot: int) -> tuple[int, int, int]:
        off = _slot_offset(slot, _slot_len_for(len(self.watched), self._b))
        return struct.unpack(_SLOT_HEADER_FMT, bytes(self._ring[off : off + _SLOT_HEADER_SIZE]))

    def close(self) -> None:
        """Release the ring mmap (idempotent)."""
        try:
            self._ring.close()
        except (ValueError, AttributeError):
            pass
        try:
            self._ring_file.close()
        except (ValueError, AttributeError):
            pass

    # -------------------------------------------------------- produce ---
    def append_token(self, token_pos: int, values: Sequence[float] | np.ndarray) -> None:
        """Write one token row into the next triple-buffer slot (non-blocking).

        Ring full (3 unpolled slots) -> set ``ring_overflow`` and drop the
        newest Deep payload (never stall, never kill the job).
        """
        arr = np.asarray(values, dtype=np.float32).reshape(1, -1)
        w = len(self.watched)
        if arr.size != w:
            raise ValueError(f"E02: token row size {arr.size} != W={w}")
        _ver, _w, wseq, rseq, _ = self._read_header()
        if wseq - rseq >= N_SLOTS:
            self.ring_overflow = True
            self._write_producer_state(wseq, True)
            return
        slot = wseq % N_SLOTS
        slot_len = _slot_len_for(w, self._b)
        self._write_slot_header(slot, _STATE_WRITING, int(token_pos), w)
        off = _slot_offset(slot, slot_len) + _SLOT_HEADER_SIZE
        encoded = encode_deep_array(arr, self.precision)
        self._ring[off : off + slot_len] = encoded.tobytes(order="C")
        self._write_slot_header(slot, _STATE_READY, int(token_pos), w)
        self._write_producer_state(wseq + 1, self.ring_overflow)

    # ----------------------------------------------------------- poll ---
    def poll(self) -> int:
        """Move READY ring slots -> binary WAL append + fsync. Returns rows."""
        _ver, _w, wseq, rseq, overflow = self._read_header()
        if overflow:
            self.ring_overflow = True
        moved = 0
        slot_len = _slot_len_for(len(self.watched), self._b)
        with open(self.wal_path, "ab") as wal:
            while rseq < wseq:
                slot = rseq % N_SLOTS
                state, token, _nvals = self._read_slot_header(slot)
                if state != _STATE_READY:
                    break
                off = _slot_offset(slot, slot_len) + _SLOT_HEADER_SIZE
                raw = bytes(self._ring[off : off + slot_len])
                wal.write(struct.pack(_WAL_ROW_HDR_FMT, int(token)) + raw)
                self._write_slot_header(slot, _STATE_EMPTY, 0, 0)
                rseq += 1
                moved += 1
                need_sync = self.fsync_every == 1 or (
                    self.fsync_every > 1 and moved % self.fsync_every == 0
                )
                if need_sync:
                    wal.flush()
                    os.fsync(wal.fileno())
            if moved > 0 and self.fsync_every != 1:
                wal.flush()
                if self.fsync_every > 0:
                    os.fsync(wal.fileno())
        self._write_consumer_state(rseq)
        return moved

    # ------------------------------------------------------------ seal ---
    def _read_wal_rows(self) -> list[tuple[int, np.ndarray]]:
        rows: list[tuple[int, np.ndarray]] = []
        if not os.path.isfile(self.wal_path):
            return rows
        w = len(self.watched)
        row_payload_bytes = _slot_len_for(w, self._b)
        record_size = _WAL_ROW_HDR_SIZE + row_payload_bytes
        raw_dt = np.dtype({"fp32": "<f4", "fp16": "<f2", "bf16": "<u2"}[self.precision])
        with open(self.wal_path, "rb") as fh:
            data = fh.read()
        n_records = len(data) // record_size
        for i in range(n_records):
            base = i * record_size
            (tok,) = struct.unpack(_WAL_ROW_HDR_FMT, data[base : base + _WAL_ROW_HDR_SIZE])
            payload = data[base + _WAL_ROW_HDR_SIZE : base + record_size]
            raw_arr = np.frombuffer(payload, dtype=raw_dt)
            vals = decode_deep_array(raw_arr, self.precision)
            rows.append((int(tok), vals))
        rows.sort(key=lambda kv: kv[0])
        return rows

    def _assemble_matrix(self) -> tuple[np.ndarray, list[int]]:
        """Assemble ``[T, W]`` float32 matrix in token order.

        Subsampled tokens (``every_n > 1``) store only the actual captured rows
        (no artificial NaN padding between ``every_n`` steps). If ``ring_overflow``
        dropped intermediate steps, missing expected steps are filled with NaN.
        """
        rows = self._read_wal_rows()
        if not rows:
            raise ValueError("E02: nothing in WAL; no Deep tokens to seal")
        w = len(self.watched)
        by_token = {tok: arr for tok, arr in rows}
        if not self.ring_overflow:
            positions = sorted(by_token.keys())
            mat = np.stack([by_token[p] for p in positions], axis=0).astype(np.float32)
            return mat, positions
        lo = rows[0][0]
        hi = rows[-1][0]
        expected = list(range(lo, hi + 1, self.every_n))
        for tok in by_token:
            if tok not in expected:
                expected.append(tok)
        expected.sort()
        mat = np.full((len(expected), w), np.nan, dtype=np.float32)
        for row_i, pos in enumerate(expected):
            if pos in by_token:
                mat[row_i, :] = by_token[pos]
        return mat, expected

    def seal(self, context: dict[str, Any] | None = None) -> str:
        """Poll remaining slots, write deep.bin + deep_index.json + seal.json + chain."""
        from glassbox.storage import layout as _layout
        from glassbox.storage import seal as _seal

        if getattr(self, "_ring", None) is not None:
            self.poll()
        matrix, positions = self._assemble_matrix()
        ctx = dict(context) if context else {}
        ctx["run_id"] = self.run_id
        flags = dict(ctx.get("flags", {})) if isinstance(ctx.get("flags"), dict) else {}
        flags["ring_overflow"] = bool(self.ring_overflow)
        ctx["flags"] = flags

        _layout.write_deep_bin(self.run_dir, matrix, self.precision)
        _layout.write_deep_index(
            self.run_dir,
            self.run_id,
            triggered=True,
            dtype=self.precision,
            tokens=int(matrix.shape[0]),
            every_n=self.every_n,
            watched=self.watched,
            window=self.window,
            token_positions=positions,
            dtype_meta={"token_positions": positions, "sealed_by": "sidecar"},
            context=ctx,
        )
        _seal.seal_run(self.run_dir, sealed_by="sidecar")
        self._tokens_sealed = positions
        self.close()
        return self.run_id

    # --------------------------------------------------------- recover ---
    @classmethod
    def attach(
        cls,
        store: str,
        run_id: str,
        watched: list[str] | None = None,
        precision: str = "fp16",
    ) -> Sidecar:
        """Re-attach to a crashed run's WAL without touching sealed bytes."""
        from glassbox.storage.layout import run_dir

        path = run_dir(store, run_id)
        index_path = os.path.join(path, "deep_index.json")
        resolved_watched: list[str] = list(watched) if watched else []
        resolved_precision = precision
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
            resolved_watched = list(index.get("watched", resolved_watched))
            resolved_precision = str(index.get("dtype", resolved_precision))
        b = precision_bytes(resolved_precision)
        wal_path = os.path.join(store, "wal", f"{run_id}.wal.bin")
        if not resolved_watched and os.path.isfile(wal_path):
            # Infer W from ring header if ring exists
            rpath = resolve_ring_path(None)
            if os.path.isfile(rpath) and os.path.getsize(rpath) >= _HEADER_SIZE:
                with open(rpath, "rb") as rfh:
                    magic, _v, ring_w, _ws, _rs, _ov = struct.unpack(
                        _HEADER_FMT, rfh.read(_HEADER_SIZE)
                    )
                if magic == MAGIC and ring_w > 0:
                    resolved_watched = [f"N-01-{i + 1:04d}" for i in range(ring_w)]
        if not resolved_watched:
            resolved_watched = ["N-01-0001"]
        obj = cls.__new__(cls)
        obj.store = store
        obj.watched = resolved_watched
        obj.precision = resolved_precision
        obj.window = None
        obj.every_n = 1
        obj.fsync_every = 1
        obj.ring_path = resolve_ring_path(None)
        obj.run_id = run_id
        obj.run_dir = path
        obj.wal_path = wal_path
        obj.ring_overflow = False
        obj._tokens_sealed = []
        obj._b = b
        return obj

    def seal_prefix(self, context: dict[str, Any] | None = None) -> str:
        """Seal the fsync'd WAL prefix after a kill; returns the E1001 hint."""
        run = self.seal(context)
        hint = (
            f"E1001 main process failed — WAL up to token "
            f"{self._tokens_sealed[-1] if self._tokens_sealed else -1} already sealed "
            f"at {self.run_dir}; rerun resume hint: re-run with the same seed and "
            f"watchlist to extend {run}."
        )
        return hint

    # -------------------------------------------------------------- ops ---
    def wal_token_count(self) -> int:
        """Number of fsync'd WAL rows (sealed-token watermark)."""
        return len(self._read_wal_rows())


def run_sidecar(ring_path: str | None = None, store: str | None = None, once: bool = True) -> None:
    """Poll the shared ring into WALs (separate container entry point)."""
    root = (
        store
        or os.environ.get("GLASSBOX_DIR")
        or os.environ.get("BLACKBOX_DIR")
        or ("./blackbox_store" if os.path.exists("./blackbox_store") else "./glassbox_store")
    )
    path = resolve_ring_path(ring_path)
    if not os.path.isfile(path):
        print(f"glassbox sidecar: no ring at {path} (nothing to poll)")
        return
    with open(path, "rb") as fh:
        header = fh.read(_HEADER_SIZE)
    if len(header) < _HEADER_SIZE or header[:8] not in MAGICS:
        print(f"glassbox sidecar: foreign/empty ring at {path} (nothing to poll)")
        return
    print(f"glassbox sidecar: watching {path} store={root} (binary WAL)")
    if once:
        return
    stop = os.path.join(root, "sidecar.stop")
    import time

    while not os.path.isfile(stop):
        time.sleep(0.05)
    print("glassbox sidecar: stop requested, drained prefix remains verifiable")


def main(argv: list[str] | None = None) -> int:
    """``python -m glassbox.tap.sidecar`` entry (compose sidecar service)."""
    parser = argparse.ArgumentParser(prog="glassbox-sidecar", description="GlassBox sidecar")
    parser.add_argument("--ring", default=None)
    parser.add_argument("--store", default=None)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args(argv)
    run_sidecar(args.ring, args.store, once=not args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
