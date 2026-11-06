"""Phase 1 seal tests: write deep_index/deep.bin, seal, verify OK, tamper -> FAIL."""

from __future__ import annotations

import json
import os

import numpy as np

from glassbox.storage.layout import allocate_run, write_deep_bin, write_deep_index
from glassbox.storage.seal import seal_run, verify_run


def _create_deep_run(run_dir: str, run_id: str, sha: str) -> None:
    watched = ["N-01-0001", "N-01-0002"]
    mat = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    write_deep_bin(run_dir, mat, "fp16")
    ctx = {
        "model_id": "M-001",
        "checkpoint_sha256": sha,
        "watchlist_ref": {
            "watchlist_id": "pilot-v1",
            "watchlist_sha256": "00" * 32,
            "patch": None,
        },
        "prompt": {"text": "Hello", "tokens": [1, 2], "tokenizer": "test"},
        "output": {"text": "Hi", "tokens": [3], "finish_reason": "stop"},
        "timing": {
            "t_start_iso": "2026-09-15T10:00:00Z",
            "t_end_iso": "2026-09-15T10:00:01Z",
            "ms_per_token": 1.0,
            "cuda_event_ms": 0.0,
        },
        "sampling": {"seed": 42, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
        "versions": {"torch": "2.4.0", "cuda": "none", "blackbox": "0.1.0", "adapter": "cpu"},
        "flags": {"bad_output": False, "ring_overflow": False, "truncated": False},
        "tags": ["good"],
        "replay": {
            "seed": 42,
            "temp": 0.0,
            "top_p": 1.0,
            "sampler": "greedy",
            "cudnn_deterministic": True,
            "checkpoint_sha256": sha,
            "versions": {"torch": "2.4.0", "cuda": "none"},
        },
    }
    write_deep_index(
        run_dir,
        run_id,
        triggered=True,
        dtype="fp16",
        tokens=2,
        watched=watched,
        context=ctx,
    )


def test_seal_verify_ok_tamper_fail(tmp_path: object, monkeypatch: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    rid, rpath = allocate_run(store)
    assert rid == "RUN-000001"
    sha = "ab" * 32
    _create_deep_run(rpath, rid, sha)
    digest = seal_run(rpath)
    assert len(digest) == 64
    assert verify_run(rid, store) is True
    with open(os.path.join(rpath, "deep_index.json"), encoding="utf-8") as fh:
        obj = json.load(fh)
    obj["prompt"]["text"] = "Hello!"  # 1-byte-class tamper
    with open(os.path.join(rpath, "deep_index.json"), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
    assert verify_run(rid, store) is False


def test_seal_verify_tamper_deep_bin_fail(tmp_path: object, monkeypatch: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    rid, rpath = allocate_run(store)
    sha = "ab" * 32
    _create_deep_run(rpath, rid, sha)
    seal_run(rpath)
    assert verify_run(rid, store) is True
    bin_file = os.path.join(rpath, "deep.bin")
    with open(bin_file, "r+b") as fh:
        b = fh.read(1)
        fh.seek(0)
        fh.write(bytes([b[0] ^ 0xFF]))
    assert verify_run(rid, store) is False


def test_chain_continuity(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    sha = "cd" * 32
    rid1, rp1 = allocate_run(store)
    _create_deep_run(rp1, rid1, sha)
    seal_run(rp1)
    rid2, rp2 = allocate_run(store)
    assert rid2 == "RUN-000002"
    _create_deep_run(rp2, rid2, sha)
    seal_run(rp2)
    assert verify_run(rid1, store) is True
    assert verify_run(rid2, store) is True
    with open(os.path.join(rp2, "seal.json"), encoding="utf-8") as fh:
        seal2 = json.load(fh)
    with open(os.path.join(rp1, "seal.json"), encoding="utf-8") as fh:
        seal1 = json.load(fh)
    assert seal2["prev_hash"] == seal1["hash"]
