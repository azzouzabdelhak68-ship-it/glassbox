"""Phase 1 CPU scan tests: fake config + safetensors header, no GPU."""

from __future__ import annotations

import json
import os
import struct

import pytest

from glassbox.exceptions import ManifestExistsError
from glassbox.scan import scan

FAKE_CONFIG = {
    "model_type": "llama3",
    "num_hidden_layers": 4,
    "hidden_size": 512,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "vocab_size": 32000,
}


def _make_weights(tmp_path: object, config: object = None) -> str:
    """Create fake weights dir with config.json + one safetensors file."""
    import pathlib

    root = pathlib.Path(str(tmp_path)) / "weights"
    root.mkdir(parents=True, exist_ok=True)
    cfg = FAKE_CONFIG if config is None else config
    (root / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    header = json.dumps(
        {
            "__metadata__": {"format": "test"},
            "w": {"dtype": "F16", "shape": [8, 8], "data_offsets": [0, 128]},
        }
    ).encode("utf-8")
    with open(root / "model.safetensors", "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(b"\x00" * 128)
    return str(root)


def test_scan_cpu_builds_manifest(tmp_path: object, monkeypatch: object) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    w = _make_weights(tmp_path)
    out = os.path.join(str(tmp_path), "manifests")
    m = scan(w, output=out)
    assert m["architecture"]["layers"] == 4
    assert m["architecture"]["hidden"] == 512
    assert m["architecture"]["heads"] == 8
    assert m["architecture"]["mlp_per_layer"] == 1024
    assert len(m["checkpoint_sha256"]) == 64
    assert m["scan"]["gpu_used"] is False
    import ast as _ast
    import pathlib as _pl

    _tree = _ast.parse(
        (_pl.Path(__file__).parent.parent / "glassbox" / "scan.py").read_text(encoding="utf-8")
    )
    _imports = [
        (n.module or "") if isinstance(n, _ast.ImportFrom) else a.name
        for n in _ast.walk(_tree)
        for a in (n.names if isinstance(n, (_ast.Import, _ast.ImportFrom)) else [])
    ]
    assert not any(str(i).split(".")[0] == "torch" for i in _imports)
    assert os.path.isfile(os.path.join(out, "manifest.json"))


def test_scan_api_model_fails_deep_only() -> None:
    with pytest.raises(FileNotFoundError, match="E02"):
        scan("gpt-4o-2026-03-15")


def test_scan_missing_config_e02(tmp_path: object) -> None:
    import pathlib

    empty = pathlib.Path(str(tmp_path)) / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="E02"):
        scan(str(empty), output=os.path.join(str(tmp_path), "m"))


def test_scan_exists_needs_force_e03(tmp_path: object) -> None:
    w = _make_weights(tmp_path)
    out = os.path.join(str(tmp_path), "m")
    scan(w, output=out)
    with pytest.raises(ManifestExistsError):
        scan(w, output=out)
    m2 = scan(w, output=out, force=True)
    assert m2["architecture"]["layers"] == 4


def test_scan_new_sha_new_dir(tmp_path: object, monkeypatch: object) -> None:
    monkeypatch.chdir(str(tmp_path))
    w = _make_weights(tmp_path)
    m1 = scan(w)  # default manifests/<id>_<sha8>/ under cwd
    cfg2 = dict(FAKE_CONFIG)
    cfg2["vocab_size"] = 32001
    import pathlib

    root = pathlib.Path(w)
    root.joinpath("config.json").write_text(json.dumps(cfg2), encoding="utf-8")
    m2 = scan(w)
    assert m1["checkpoint_sha256"] != m2["checkpoint_sha256"]
