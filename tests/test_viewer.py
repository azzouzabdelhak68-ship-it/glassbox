"""Viewer CLI + HTML contract (spec §VIEW-cli/html/partial, plan §10.3-10.4).

CPU-only: one Deep run (8 watched x 10 tokens) + one untriggered Deep run sealed
in a tmp store; CLI exercised via ``cli.main`` with ``BLACKBOX_DIR`` pinned.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np
import pytest

from glassbox import viewer
from glassbox.cli import main
from glassbox.storage import layout as _layout
from glassbox.storage import seal as _seal

SHA = "ab12cd34" * 8
WATCHED = [f"N-02-{i:04d}" for i in range(1, 9)]
T = 10


def _outer_meta(run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": SHA,
        "prompt": {"text": "Explain why the sky is blue", "tokens": [1, 2]},
        "output": {"text": "because of scattering", "tokens": [3, 4, 5]},
        "replay": {
            "seed": 42,
            "temp": 0.0,
            "top_p": 1.0,
            "sampler": "greedy",
            "checkpoint_sha256": SHA,
        },
    }


def _seal_deep(store: str, seed: int = 9) -> str:
    run_id, run_dir = _layout.allocate_run(store)
    rng = random.Random(seed)
    mat = np.asarray(
        [[rng.uniform(-0.5, 0.5) for _ in WATCHED] for _ in range(T)], dtype=np.float32
    )
    mat[7, 0] = 0.92
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
    return run_id


def _seal_untriggered(store: str) -> str:
    run_id, run_dir = _layout.allocate_run(store)
    meta = _outer_meta(run_id)
    _layout.write_absent_deep_index(
        run_dir,
        run_id,
        watched=[],
        context=meta,
    )
    _seal.seal_run(run_dir)
    return run_id


@pytest.fixture()
def vstore(tmp_path: object, monkeypatch: object) -> dict[str, str]:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    return {"store": store, "deep": _seal_deep(store), "untriggered": _seal_untriggered(store)}


def test_view_table_values_and_unwatched_grey(vstore: dict[str, str], capsys: object) -> None:
    code = main(
        ["view", vstore["deep"], "--tokens", "0..9", "--addrs", "N-02-0001,N-02-0009", "--no-color"]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "0.92" in out and "--" in out  # watched value + grey not_watched
    assert "VERIFIED" in out and "N-02-0001" in out


def test_view_json_exact_null_not_watched(vstore: dict[str, str], capsys: object) -> None:
    code = main(
        ["view", vstore["deep"], "--tokens", "7..7", "--addrs", "N-02-0001,N-02-0009", "--json"]
    )
    assert code == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["hash_seal_verified"] is True
    row = doc["tokens"][0]
    assert row["pos"] == 7 and row["values"]["N-02-0001"] == pytest.approx(0.92, abs=0.01)
    assert row["values"]["N-02-0009"] is None
    assert row["values"]["_note"] == "not_watched"


def test_view_untriggered_notice(vstore: dict[str, str], capsys: object) -> None:
    assert main(["view", vstore["untriggered"], "--no-color"]) == 0
    assert "Untriggered Deep run" in capsys.readouterr().out


def test_view_tamper_red_badge_still_renders(vstore: dict[str, str], capsys: object) -> None:
    victim = os.path.join(vstore["store"], "runs", vstore["deep"], "deep_index.json")
    with open(victim, encoding="utf-8") as fh:
        idx = json.load(fh)
    idx["output"]["text"] = "tampered"
    with open(victim, "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2, sort_keys=True)
    assert main(["view", vstore["deep"], "--no-color"]) == 0
    out = capsys.readouterr().out
    assert "FAIL" in out and "0.92" in out  # red badge but still renders


def test_view_errors(vstore: dict[str, str]) -> None:
    assert main(["view", "RUN-999999"]) == 6
    assert main(["view", vstore["deep"], "--tokens", "9..3"]) == 2
    assert main(["view", vstore["deep"], "--port", "80"]) == 2


def test_export_html_self_contained(vstore: dict[str, str], tmp_path: object) -> None:
    target = os.path.join(str(tmp_path), "run.html")
    assert main(["view", vstore["deep"], "--export-html", target]) == 0
    with open(target, encoding="utf-8") as fh:
        html_text = fh.read()
    for marker in (
        "PROMPT (tokenized chips)",
        "#token=",
        "co-occurrence, not causality",
        "max-width:960px",
        "0.92",
        "not_watched",
        "system-ui",
        'type="application/json"',
    ):
        assert marker in html_text, marker
    assert "http" not in html_text.replace("http.server", "")  # no external refs
    assert len(html_text.encode("utf-8")) < 200 * 1024  # slice embedded, never full TBs


def test_viewer_module_defaults(tmp_path: object, monkeypatch: object) -> None:
    store = os.path.join(str(tmp_path), "store")
    monkeypatch.setenv("BLACKBOX_DIR", store)
    run_id = _seal_deep(store)
    assert viewer.default_selected(viewer.load_run(run_id, store)) == 7  # peak |value|
    table = viewer.token_table(run_id, store)  # defaults: 32 tokens x 8 watched
    assert table["lo"] == 0 and len(table["addrs"]) == 8
    doc = viewer.to_json(table)
    assert doc["tokens"][7]["values"]["N-02-0001"] == pytest.approx(0.92, abs=0.01)
