"""Deep persistence integration (spec §FILE-deepbin/deepindex/seal, §TAP-sidecar).

CPU-only: DeepTap ring + storage + seal + Sidecar WAL, no GPU, no /dev/shm
(ring_path + store live under tmp_path). Run with
``BLACKBOX_GPU=cpu pytest tests/test_deep_integration.py -q``.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import pytest

from glassbox.controller import run_with_watchlist
from glassbox.exceptions import SizeFlagError
from glassbox.storage import layout as _layout
from glassbox.storage import seal as _seal
from glassbox.tap.hooks import DeepTap
from glassbox.tap.sidecar import Sidecar

SHA = "ab12cd34" * 8
MANIFEST: dict[str, Any] = {
    "model_id": "M-001",
    "checkpoint_sha256": SHA,
    "architecture": {"layers": 4, "hidden": 512, "heads": 8, "mlp_per_layer": 1024},
}


def _outer(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": SHA,
        "replay": {"seed": 42, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
    }


def _watchlist_yaml(path: str, wid: str, n_addr: int = 12) -> str:
    addrs = [f"N-02-{i:04d}" for i in range(1, n_addr + 1)]
    doc = {
        "version": 1,
        "model_id": "M-001",
        "checkpoint_sha256": SHA,
        "watchlist_id": wid,
        "trigger": "always",
        "window": {"start_token": 0, "end_token": 511, "every_n": 1},
        "precision": "fp16",
        "entries": [{"addresses": addrs}],
    }
    import yaml

    wl_path = os.path.join(path, wid + ".yaml")
    with open(wl_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh)
    man_path = os.path.join(path, "manifest.json")
    with open(man_path, "w", encoding="utf-8") as fh:
        json.dump(MANIFEST, fh)
    return wl_path


def test_deep_bytes_exact_1k_x10() -> None:
    """1k watched x 10 tokens fp16 -> exactly 20000 bytes, reshape (10, 1000)."""
    watched = [f"N-02-{i:04d}" for i in range(1, 1001)]
    tap = DeepTap(watched, window={"start_token": 0, "end_token": 10, "every_n": 1})
    rng = random.Random(7)
    for pos in range(10):
        tap._staging = {c: rng.uniform(-1.0, 1.0) for c in range(1000)}
        assert tap.end_token(pos) is True
    assert len(tap.ring) == 10
    blob = tap.to_bytes()
    assert len(blob) == 1000 * 10 * 2
    arr = np.frombuffer(blob, dtype=np.dtype("<f2")).reshape(10, 1000)
    assert arr.shape == (10, 1000)
    assert np.isfinite(arr.astype(np.float32)).all()


def test_deep_bin_storage_roundtrip_and_tamper(tmp_path: object) -> None:
    """write_deep_bin + index + seal verify OK; 1-byte tamper -> FAIL."""
    store = os.path.join(str(tmp_path), "store")
    run_id, run_dir = _layout.allocate_run(store)
    rng = np.random.default_rng(3)
    matrix = rng.uniform(-1.0, 1.0, size=(10, 8)).astype(np.float32)
    watched = [f"N-02-{i:04d}" for i in range(1, 9)]
    _layout.write_deep_bin(run_dir, matrix, "fp16")
    _layout.write_deep_index(
        run_dir,
        run_id,
        triggered=True,
        dtype="fp16",
        tokens=10,
        watched=watched,
        context=_outer(run_id),
    )
    _seal.seal_run(run_dir)
    assert _seal.verify_run(run_id, store) is True
    part = _layout.read_deep_slice(run_dir, 2, 5, [0, 3])
    assert part.shape == (4, 2)
    assert part.dtype == np.float32
    with open(os.path.join(run_dir, "deep.bin"), "r+b") as fh:
        fh.seek(0)
        first = fh.read(1)
        fh.seek(0)
        fh.write(bytes([first[0] ^ 0xFF]))
    assert _seal.verify_run(run_id, store) is False


def test_sidecar_kill_prefix_still_verifies(tmp_path: object, monkeypatch: object) -> None:
    """Simulated kill -9: WAL fsync'd prefix seals + verifies via attach."""
    store = os.path.join(str(tmp_path), "store")
    ring = os.path.join(str(tmp_path), "bb.ring")
    monkeypatch.setenv("BLACKBOX_SHM", ring)
    watched = [f"N-02-{i:04d}" for i in range(1, 5)]
    sc = Sidecar(store, watched, precision="fp16", ring_path=ring)
    run_id = sc.run_id
    rng = random.Random(11)
    for pos in range(6):
        sc.append_token(pos, [rng.uniform(-1.0, 1.0) for _ in watched])
        if pos % 2 == 1:
            sc.poll()  # drain incrementally: triple buffer holds 3 unpolled
    assert sc.poll() == 0
    assert sc.wal_token_count() == 6
    assert sc.ring_overflow is False
    # Overflow path: 5 rapid appends without polling drop to 3 + flag.
    sc2 = Sidecar(store, watched, precision="fp16", ring_path=ring)
    for pos in range(100, 105):
        sc2.append_token(pos, [0.5 for _ in watched])
    assert sc2.poll() == 3
    assert sc2.ring_overflow is True
    sc2.close()
    # main dies here without sealing; recovery re-attaches and seals prefix.
    recovered = Sidecar.attach(store, run_id)
    hint = recovered.seal_prefix(_outer(run_id))
    assert "E1001" in hint and run_id in hint
    assert _seal.verify_run(run_id, store) is True
    index = _layout.read_deep_index(os.path.join(store, "runs", run_id))
    assert index["tokens"] == 6 and index["triggered"] is True


def test_red_estimate_needs_force(tmp_path: object) -> None:
    """RED without --force raises SizeFlagError and seals nothing."""
    wl_path = _watchlist_yaml(str(tmp_path), "cohort-red")
    man_path = os.path.join(str(tmp_path), "manifest.json")
    # W=12 T=511 R=1e8 B=2 -> ~1.4e12 bytes -> RED (>1000GB).
    with pytest.raises(SizeFlagError, match="E05"):
        run_with_watchlist(wl_path, manifest=man_path, runs=100_000_000)
    ok = run_with_watchlist(wl_path, manifest=man_path, runs=100_000_000, force=True)
    assert ok["flag"] == "RED" and ok["force"] is True


def test_every_n_subsample_and_absent_deep(tmp_path: object) -> None:
    """every_n:10 keeps 0,10,20 of 25 tokens; never-armed seals triggered=false."""
    watched = ["N-02-0001", "A-02-04"]
    tap = DeepTap(watched, window={"start_token": 0, "end_token": 25, "every_n": 10})
    for pos in range(25):
        tap._staging = {0: 0.1 * pos, 1: 0.2 * pos}
        tap.end_token(pos)
    assert len(tap.ring) == 3  # tokens 0, 10, 20

    store = os.path.join(str(tmp_path), "store")
    run_id, run_dir = _layout.allocate_run(store)
    _layout.write_absent_deep_index(run_dir, run_id, watched=watched, context=_outer(run_id))
    _seal.seal_run(run_dir)
    index = _layout.read_deep_index(run_dir)
    assert index["triggered"] is False
    assert not os.path.isfile(os.path.join(run_dir, "deep.bin"))
    with pytest.raises(ValueError, match="triggered=false"):
        _layout.read_deep_slice(run_dir, 0, 1)
    assert _seal.verify_run(run_id, store) is True


def test_sparse_chunked_compression_exactness(tmp_path: object) -> None:
    """Chunked sparse deep.bin compresses zero-heavy activations and preserves random access."""
    store = os.path.join(str(tmp_path), "sparse_store")
    run_id, run_dir = _layout.allocate_run(store)
    watched = [f"N-01-{i:04d}" for i in range(1, 51)]

    # 128 tokens x 50 watched, ~85% zeros (standard sparse MLP activations)
    rng = np.random.default_rng(42)
    dense_mat = np.zeros((128, 50), dtype=np.float32)
    mask = rng.random((128, 50)) > 0.85
    dense_mat[mask] = rng.uniform(0.1, 5.0, size=np.count_nonzero(mask)).astype(np.float32)

    # Write compressed with 32 tokens per chunk
    bin_path = _layout.write_deep_bin(
        run_dir, dense_mat, precision="fp16", compression="sparse", chunk_tokens=32
    )
    uncomp_expected = 128 * 50 * 2  # 12,800 bytes
    comp_size = os.path.getsize(bin_path)
    assert comp_size < uncomp_expected // 2, (
        f"Expected >2x compression, got {comp_size}/{uncomp_expected}"
    )

    _layout.write_deep_index(
        run_dir,
        run_id,
        triggered=True,
        dtype="fp16",
        tokens=128,
        every_n=1,
        watched=watched,
        context=_outer(run_id),
        dtype_meta={"compression": "sparse_zlib"},
    )
    _seal.seal_run(run_dir)
    assert _seal.verify_run(run_id, store) is True

    # Sub-slice across chunk boundaries (tokens 25..75 crosses chunk 0, 1, 2)
    sub = _layout.read_deep_slice(run_dir, 25, 75, watched_idx=[0, 10, 25, 49])
    assert sub.shape == (51, 4)
    # Check numerical fidelity (fp16 round-trip)
    expected_fp16 = np.asarray(dense_mat[25:76, [0, 10, 25, 49]], dtype=np.float16).astype(
        np.float32
    )
    np.testing.assert_allclose(sub, expected_fp16, rtol=1e-3, atol=1e-3)
