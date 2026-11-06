"""Dataframe contract (spec §A.12): columns, dtypes, NaN, seal warnings."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import pytest

from glassbox import to_dataframe
from glassbox.storage import layout as _layout
from glassbox.storage import seal as _seal
from glassbox.to_dataframe import DEEP_COLUMNS, FULL_COLUMNS, SealBrokenWarning

SHA = "ab12cd34" * 8
WATCHED = [f"N-02-{i:04d}" for i in range(1, 5)]
T = 10


def _outer_meta(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": SHA,
        "prompt": {"text": "hi", "tokens": [1, 2]},
        "output": {"text": "yo", "tokens": [3, 4, 5]},
        "replay": {"seed": 1, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
    }


@pytest.fixture()
def dstore(tmp_path: object, monkeypatch: object) -> dict[str, str]:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    run_id, run_dir = _layout.allocate_run(store)
    rng = random.Random(4)
    mat = np.asarray(
        [[rng.uniform(-1.0, 1.0) for _ in WATCHED] for _ in range(T)], dtype=np.float32
    )
    _layout.write_deep_bin(run_dir, mat, "fp16")
    meta = _outer_meta(run_id)
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

    untrig_id, untrig_dir = _layout.allocate_run(store)
    untrig_meta = _outer_meta(untrig_id)
    _layout.write_absent_deep_index(
        untrig_dir,
        untrig_id,
        dtype="fp16",
        watched=[],
        context=untrig_meta,
    )
    _seal.seal_run(untrig_dir)
    return {"store": store, "deep": run_id, "untriggered": untrig_id}


def test_deep_columns_dtypes(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], what="deep")
    assert list(df.columns) == DEEP_COLUMNS
    assert len(df) == T * len(WATCHED)
    assert str(df.dtypes["token_pos"]) == "int32"
    assert str(df.dtypes["value"]) == "float32"
    assert str(df.dtypes["watched"]) == "bool"
    assert bool(df.iloc[0]["seal_ok"]) is True


def test_full_columns_broadcast(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], what="full", tokens=(0, 1))
    assert list(df.columns) == FULL_COLUMNS
    assert len(df) == 2 * len(WATCHED)
    assert df.iloc[0]["model_id"] == "M-001"
    assert df.iloc[0]["prompt_text"] == "hi"


def test_deep_long_form_values(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], tokens=(2, 5), addresses=["N-02-0001", "N-02-0003"])
    assert len(df) == 4 * 2  # 4 tokens x 2 addresses
    assert set(df["address"].tolist()) == {"N-02-0001", "N-02-0003"}
    assert df["token_pos"].tolist() == [2, 2, 3, 3, 4, 4, 5, 5]
    assert bool(df["watched"].all())
    assert str(df.dtypes["token_pos"]) == "int32"


def test_unwatched_nan_not_interpolated(dstore: dict[str, str]) -> None:
    df = to_dataframe(dstore["deep"], tokens=(0, 0), addresses=["N-02-0009"])
    assert len(df) == 1
    assert bool(df.iloc[0]["watched"]) is False
    assert float(df.iloc[0]["value"]) != float(df.iloc[0]["value"])  # NaN


def test_broken_seal_warns_not_drops(dstore: dict[str, str]) -> None:
    victim = os.path.join(dstore["store"], "runs", dstore["deep"], "deep_index.json")
    with open(victim, encoding="utf-8") as fh:
        idx = json.load(fh)
    idx["output"]["text"] = "edited"
    with open(victim, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2, sort_keys=True)
    with pytest.warns(SealBrokenWarning):
        df = to_dataframe(dstore["deep"], what="deep")
    assert bool(df.iloc[0]["seal_ok"]) is False


def test_missing_run_bad_slice_outer_only(dstore: dict[str, str]) -> None:
    with pytest.raises(Exception, match="not found"):
        to_dataframe("RUN-999999", what="deep")
    with pytest.raises(ValueError, match="E02"):
        to_dataframe(dstore["deep"], tokens=(8, 3))
    with pytest.raises(ValueError, match="triggered=false"):
        to_dataframe(dstore["untriggered"], what="deep")
    with pytest.raises(ValueError, match="unknown what"):
        to_dataframe(dstore["deep"], what="sideways")
