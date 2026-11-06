"""Two-phase search over a synthetic good/bad cohort (spec §SEARCH-tiers/grammar).

CPU-only: 3 good + 3 bad runs x 4 watched addresses x 20 tokens, sealed in a
tmp store. Bad runs fire N-02-0001 at tokens 15..19; one bad run is tampered
after sealing (seal FAIL rows are still returned, flagged).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np

from glassbox.search import index as _index
from glassbox.search.query import compare_runs, find_runs
from glassbox.storage import layout as _layout
from glassbox.storage import seal as _seal

SHA = "ab12cd34" * 8
WATCHED = ["N-02-0001", "N-02-0002", "A-02-01", "LOGITS"]
T = 20


def _outer(run_id: str, prompt: str, tag: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": SHA,
        "watchlist_ref": {"watchlist_id": "cohort-A-v3", "watchlist_sha256": "0" * 64},
        "prompt": {"text": prompt, "tokens": [1, 2], "tokenizer": "test-bpe"},
        "output": {"text": "out", "tokens": [3, 4], "finish_reason": "stop"},
        "timing": {
            "t_start_iso": "2026-09-18T00:00:00Z",
            "t_end_iso": "2026-09-18T00:00:01Z",
            "ms_per_token": 1.0,
        },
        "sampling": {"seed": 42, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
        "versions": {"torch": "test", "cuda": "cpu", "blackbox": "0.3.0", "adapter": "cpu"},
        "flags": {"bad_output": False, "ring_overflow": False, "truncated": False},
        "tags": [tag],
        "replay": {
            "seed": 42,
            "temp": 0.0,
            "top_p": 1.0,
            "sampler": "greedy",
            "checkpoint_sha256": SHA,
        },
        "error": None,
    }


def _seal_cohort_run(store: str, prompt: str, tag: str, bad: bool, seed: int) -> str:
    run_id, run_dir = _layout.allocate_run(store)
    rng = random.Random(seed)
    mat = np.asarray(
        [[rng.uniform(-0.2, 0.4) for _ in WATCHED] for _ in range(T)], dtype=np.float32
    )
    if bad:
        mat[15:20, 0] = np.asarray([rng.uniform(0.85, 0.99) for _ in range(5)], dtype=np.float32)
    _layout.write_deep_bin(run_dir, mat, "fp16")
    meta = _outer(run_id, prompt, tag)
    _layout.write_deep_index(
        run_dir,
        run_id,
        triggered=True,
        dtype="fp16",
        tokens=T,
        watched=WATCHED,
        context=meta,
    )
    _seal.seal_run(run_dir)
    return run_id


def _build_cohort(store: str) -> dict[str, list[str]]:
    good = [_seal_cohort_run(store, "hello world", "cohort=good", False, 100 + i) for i in range(3)]
    bad = [
        _seal_cohort_run(store, "instructions for testing", "cohort=bad", True, 200 + i)
        for i in range(3)
    ]
    return {"good": good, "bad": bad}


def test_hot_entry_aggregates_and_reindex(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    cohort = _build_cohort(store)
    assert _index.reindex(store) == 6
    entry = _index.ensure_hot(cohort["bad"][0], store)
    assert entry["has_deep"] is True
    stats = entry["addresses"]["N-02-0001"]
    assert stats["max"] is not None and stats["max"] > 0.8
    assert stats["firing_rate_0.8"] > 0.0
    assert stats["n_win10"] == 2  # T=20 -> two 10-token chunks
    assert entry["tags"] == ["cohort=bad"]


def test_find_neuron_hot_then_cold(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    cohort = _build_cohort(store)
    rows, meta = find_runs("N-02-0001 AT token=17 > 0.8", store=store, limit=20)
    assert {r["run"] for r in rows} == set(cohort["bad"])
    assert all(r["token"] == 17 and r["value"] > 0.8 for r in rows)
    assert all(r["hash_seal_verified"] is True for r in rows)
    assert meta["cold_slices"] >= 3  # Hot narrowed before Cold opened


def test_find_prompt_and_range_max(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    _build_cohort(store)
    rows, _ = find_runs('prompt CONTAINS "instructions for"', store=store)
    assert len(rows) == 3
    rows, _ = find_runs("N-02-0001 AT tokens 15..19 MAX > 0.8", store=store)
    assert len(rows) == 3
    rows, _ = find_runs("N-02-0002 AT token=17 > 0.8", store=store)
    assert rows == []  # no false positives from Hot either


def test_tampered_run_still_returned_flagged(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    cohort = _build_cohort(store)
    victim = os.path.join(store, "runs", cohort["bad"][0], "deep_index.json")
    with open(victim, encoding="utf-8") as fh:
        idx = json.load(fh)
    idx["prompt"]["text"] = "edited after seal"
    with open(victim, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2, sort_keys=True)
        fh.write("\n")
    assert _seal.verify_run(cohort["bad"][0], store) is False
    rows, _ = find_runs("N-02-0001 AT token=17 > 0.8", store=store)
    flagged = [r for r in rows if r["run"] == cohort["bad"][0]]
    assert len(flagged) == 1 and flagged[0]["hash_seal_verified"] is False


def test_compare_good_vs_bad(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    _build_cohort(store)
    results, meta = compare_runs(
        "tag:cohort=good",
        "tag:cohort=bad",
        ["N-02-0001", "A-02-01"],
        stats="mean,p95,firing_rate",
        tokens=(15, 19),
        limit=10000,
        store=store,
    )
    assert meta["cold_slices"] > 0
    by_addr = {r["address"]: r for r in results}
    n1 = by_addr["N-02-0001"]
    assert n1["good"]["n"] == 3 and n1["bad"]["n"] == 3
    assert n1["bad"]["mean"] > n1["good"]["mean"]
    assert n1["bad"]["firing_rate"] > 0.9
    assert n1["good"]["firing_rate"] < 0.2
    assert n1["tokens"] == "15..19"


def test_compare_empty_group_e06_shape(tmp_path: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    _build_cohort(store)
    results, _ = compare_runs(
        "tag:cohort=nope", "tag:cohort=bad", ["N-02-0001"], tokens=(0, 5), store=store
    )
    assert results[0]["good"].get("empty") is True
