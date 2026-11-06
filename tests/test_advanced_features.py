"""Tests for Advanced Mechanistic Interpretability Features & Optimizations.

Covers:
- Zero-Sync Logits Capture & Batched Inference
- Chunked Sparse Storage
- Logit Lens & Residual Stream Analysis (blackbox.lens)
- Sparse Autoencoder (SAE) Latent Addresses (SAE-LL-FFFF, F-LL-FFFF)
- Direct Logit Attribution (DLA in compare)
- Auto-Profiler (blackbox.profile)
- Static HTML SVG Heatmaps & Circuit Diagrams
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from glassbox import lens, profile
from glassbox.lens import format_lens_table
from glassbox.search.query import compare_runs
from glassbox.storage import layout as _layout
from glassbox.storage import seal as _seal
from glassbox.viewer import render_html
from glassbox.watchlist import expand_address, is_single_address, validate_addresses

MANIFEST: dict[str, Any] = {
    "model_id": "M-001",
    "checkpoint_sha256": "01234567" * 8,
    "architecture": {
        "layers": 8,
        "hidden": 512,
        "heads": 8,
        "mlp_per_layer": 2048,
        "vocab": 32000,
    },
}


def test_sae_address_syntax_and_expansion() -> None:
    """SAE-LL-FFFF and F-LL-FFFF addresses parse, expand in ranges, and validate."""
    assert is_single_address("SAE-04-0042") is True
    assert is_single_address("F-04-0042") is True

    range_expanded = expand_address("SAE-02-0001..0004")
    assert range_expanded == [
        "SAE-02-0001",
        "SAE-02-0002",
        "SAE-02-0003",
        "SAE-02-0004",
    ]

    f_expanded = expand_address("F-03-0010..0012")
    assert f_expanded == ["F-03-0010", "F-03-0011", "F-03-0012"]

    # Validates vs manifest
    validate_addresses(["SAE-04-0001", "F-08-0100", "LOGITS"], MANIFEST)


def test_logit_lens_computation(tmp_path: object) -> None:
    """Logit lens computes layer-by-layer residual delta and peak update."""
    store = os.path.join(str(tmp_path), "store")
    run_id, run_dir = _layout.allocate_run(store)

    # 4 layers of residual stream + LOGITS
    watched = ["L-01", "L-02", "L-03", "L-04", "LOGITS"]
    # 5 tokens, 5 watched
    mat = np.array(
        [
            [1.0, 2.5, 4.0, 4.2, 8.5],
            [1.1, 2.4, 4.1, 4.3, 8.6],
            [0.9, 3.8, 4.5, 4.6, 9.0],
            [1.2, 2.7, 4.2, 4.4, 8.7],
            [1.0, 2.5, 4.0, 4.2, 8.5],
        ],
        dtype=np.float32,
    )
    _layout.write_deep_bin(run_dir, mat, precision="fp16")
    _layout.write_deep_index(
        run_dir,
        run_id,
        triggered=True,
        dtype="fp16",
        tokens=5,
        every_n=1,
        watched=watched,
        context={
            "run_id": run_id,
            "logits_summary": [
                {"top_token_ids": [42, 10], "top_logits": [8.5, 6.2], "entropy": 0.45}
            ]
            * 5,
        },
    )
    _seal.seal_run(run_dir)

    lens_data = lens(run_id, store=store, token_pos=0)
    assert lens_data["run_id"] == run_id
    assert lens_data["layers"] == [1, 2, 3, 4]
    # L-02 had delta 1.5 (2.5 - 1.0), which is the largest information write
    assert lens_data["max_delta_layer"] == 2
    assert lens_data["layer_deltas"][2] == 1.5

    table_text = format_lens_table(lens_data)
    assert "Peak computation update at Layer L-02" in table_text
    assert "Top Token ID 42" in table_text


def test_direct_logit_attribution(tmp_path: object) -> None:
    """compare_runs computes causal attribution between good and bad cohorts."""
    store = os.path.join(str(tmp_path), "store")

    # Seed two runs: one 'good' with high N-01-0001, one 'bad' with low N-01-0001
    for r_idx, (cohort_tag, val) in enumerate([("tag:cohort=good", 5.0), ("tag:cohort=bad", 0.5)]):
        rid, rdir = _layout.allocate_run(store)
        mat = np.full((10, 2), val, dtype=np.float32)
        _layout.write_deep_bin(rdir, mat, precision="fp16")
        _layout.write_deep_index(
            rdir,
            rid,
            triggered=True,
            dtype="fp16",
            tokens=10,
            every_n=1,
            watched=["N-01-0001", "N-01-0002"],
            context={
                "run_id": rid,
                "tags": [cohort_tag.replace("tag:", "")],
                "checkpoint_sha256": MANIFEST["checkpoint_sha256"],
            },
        )
        _seal.seal_run(rdir)

    results, meta = compare_runs(
        "tag:cohort=good",
        "tag:cohort=bad",
        ["N-01-0001"],
        tokens=(0, 5),
        store=store,
    )
    assert len(results) == 1
    res = results[0]
    assert res["delta_mean"] == pytest.approx(4.5, abs=0.1)
    assert res["attribution"] > 0.0  # Positively attributed to good cohort


def test_profiler_generates_watchlist(tmp_path: object) -> None:
    """Auto-profiler selects top-k addresses and exports valid YAML."""
    out_yaml = os.path.join(str(tmp_path), "calibrated_watchlist.yaml")
    selected = profile(
        None,
        MANIFEST,
        top_k=20,
        output_yaml=out_yaml,
    )
    assert len(selected) == 20
    assert "LOGITS" in selected
    assert os.path.isfile(out_yaml)


def test_viewer_html_embeds_svg_heatmap_and_circuit(tmp_path: object) -> None:
    """render_html embeds inline SVG activation heatmap and circuit diagram."""
    store = os.path.join(str(tmp_path), "store")
    rid, rdir = _layout.allocate_run(store)
    watched = ["N-01-0001", "A-01-01", "LOGITS"]
    mat = np.random.uniform(0.1, 2.0, (16, 3)).astype(np.float32)
    _layout.write_deep_bin(rdir, mat, precision="fp16")
    _layout.write_deep_index(
        rdir,
        rid,
        triggered=True,
        dtype="fp16",
        tokens=16,
        every_n=1,
        watched=watched,
        context={"run_id": rid},
    )
    _seal.seal_run(rdir)

    html_content = render_html(rid, store=store)
    assert "Activation Heatmap Matrix" in html_content
    assert "<svg" in html_content
    assert "hm-cell" in html_content
    assert "Co-occurrence Circuit Graph" in html_content
