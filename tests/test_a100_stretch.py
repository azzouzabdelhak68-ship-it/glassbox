"""Tests for A100 (40 GB) stretch notebook, VLM/hybrid linear_attn hooks, and Drive cap."""

from __future__ import annotations

import ast
import json
import os
from typing import Any

import pytest
import torch
import torch.nn as nn
import yaml

from glassbox import GlassBox
from glassbox.tap.hooks import DeepTap
from glassbox.watchlist import expand_addresses


def _clean_cell_source(raw_source: str) -> str:
    cleaned_lines: list[str] = []
    in_shell = False
    for line in raw_source.split("\n"):
        stripped = line.strip()
        if stripped.startswith("!") or stripped.startswith("%") or in_shell:
            cleaned_lines.append(f"# {line}")
            in_shell = line.rstrip().endswith("\\")
        else:
            cleaned_lines.append(line)
            in_shell = False
    return "\n".join(cleaned_lines)


def test_a100_stretch_notebook_and_note_validity() -> None:
    note_path = "A100_STRETCH_NOTE.md"
    nb_path = os.path.join("notebooks", "a100_stretch_test.ipynb")

    assert os.path.isfile(note_path), "A100_STRETCH_NOTE.md missing"
    assert os.path.isfile(nb_path), "notebooks/a100_stretch_test.ipynb missing"

    with open(note_path, encoding="utf-8") as fh:
        note_text = fh.read()
    assert "notebooks/a100_stretch_test.ipynb" in note_text
    assert "Qwen/Qwen3.8-27B" in note_text
    assert "1.0 GB" in note_text
    assert "2.0 GB" in note_text

    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    assert nb.get("nbformat") == 4
    assert nb.get("metadata", {}).get("colab", {}).get("gpuType") == "A100"

    cells = nb.get("cells", [])
    assert len(cells) == 9
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    assert len(code_cells) == 7

    all_src = "\n".join("".join(c.get("source", [])) for c in cells)
    for expected in (
        "Qwen/Qwen3-14B",
        "unsloth/Qwen3-32B-bnb-4bit",
        "Qwen/Qwen3.8-27B-FP8",
        "Qwen/Qwen3.8-27B",
        "_has_fast_deltanet_kernels",
        "ONE_GB_BYTES",
        "TWO_GB_HARD_CAP_BYTES",
        "WALLCLOCK_BUDGET_MINUTES",
        "DriveTelemetryStreamer",
        "HardStorageCapReached",
        "ABORTED_AT_2GB_LIMIT.json",
        "SAFETY_BATTERY",
        "EXPANDED_WATCHED",
        "WallClockStopper",
        "_fast_tok_str",
        "KeyboardInterrupt",
        "safety_circuit_neurons.csv",
    ):
        assert expected in all_src, f"Missing {expected!r} in a100_stretch_test.ipynb"

    for idx, cell in enumerate(code_cells):
        raw = "".join(cell.get("source", []))
        ast.parse(_clean_cell_source(raw), filename=f"cell_{idx}.py")


def test_qwen38_vlm_hybrid_linear_attn_and_fp8_hooks(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies VLM language_model path, 3:1 Gated DeltaNet linear_attn, and FP8 upcast."""
    monkeypatch.setenv("GLASSBOX_DIR", str(tmp_path / "gb_store"))

    class MockMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Linear(32, 64)
            self.down_proj = nn.Linear(64, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(torch.relu(self.up_proj(x)))

    class MockGatedDeltaNet(nn.Module):
        """Mimics Qwen3.8 linear_attn with distinct num_v_heads=8 and out_proj."""

        def __init__(self) -> None:
            super().__init__()
            self.num_v_heads = 8
            self.in_proj = nn.Linear(32, 64)
            self.out_proj = nn.Linear(64, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.out_proj(torch.tanh(self.in_proj(x)))

    class MockFullSelfAttn(nn.Module):
        """Mimics Qwen3.8 full self_attn with num_heads=4 and o_proj."""

        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 4
            self.qkv = nn.Linear(32, 32)
            self.o_proj = nn.Linear(32, 32)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.o_proj(self.qkv(x))

    class MockHybridDecoderLayer(nn.Module):
        def __init__(self, is_linear: bool) -> None:
            super().__init__()
            if is_linear:
                self.linear_attn = MockGatedDeltaNet()
            else:
                self.self_attn = MockFullSelfAttn()
            self.mlp = MockMLP()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            attn_out = self.linear_attn(x) if hasattr(self, "linear_attn") else self.self_attn(x)
            return x + attn_out + self.mlp(x)

    class MockLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(128, 32)
            self.layers = nn.ModuleList(
                [
                    MockHybridDecoderLayer(is_linear=True),
                    MockHybridDecoderLayer(is_linear=True),
                    MockHybridDecoderLayer(is_linear=True),
                    MockHybridDecoderLayer(is_linear=False),
                ]
            )
            self.norm = nn.LayerNorm(32)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            h = self.embed_tokens(input_ids)
            for blk in self.layers:
                h = blk(h)
            return self.norm(h)

    class MockInnerWrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = MockLanguageModel()

    class MockQwen38VLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            class TextCfg:
                num_hidden_layers = 4
                hidden_size = 32
                num_attention_heads = 4
                intermediate_size = 64
                vocab_size = 128

            class CompositeVlmCfg:
                model_type = "qwen3_5"
                text_config = TextCfg()

            self.config = CompositeVlmCfg()
            self.model = MockInnerWrapper()
            self.lm_head = nn.Linear(32, 128)

        def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
            _ = kwargs
            h = self.model.language_model(input_ids)
            return self.lm_head(h)

        def generate(
            self, input_ids: torch.Tensor, max_new_tokens: int = 3, **kwargs: Any
        ) -> torch.Tensor:
            _ = kwargs
            curr = input_ids
            # Prefill pass [1, S]
            _ = self(curr)
            for step in range(max_new_tokens):
                next_tok = torch.tensor([[10 + step]])
                # Autoregressive decode pass [1, 1]
                _ = self(next_tok)
                curr = torch.cat([curr, next_tok], dim=1)
            return curr

    vlm = MockQwen38VLM()
    manifest = {
        "model_id": "Qwen_Qwen3.8-27B",
        "checkpoint_sha256": "0" * 64,
        "architecture": {
            "layers": 4,
            "hidden": 32,
            "heads": 4,
            "mlp_per_layer": 64,
            "vocab": 128,
        },
    }
    raw_addrs = [
        "EMBED",
        "N-01-0001..0008",
        "A-01-01..04",
        "L-01",
        "L-01-0001..0004",
        "N-04-0001..0008",
        "A-04-01..04",
        "L-04",
        "NORM",
        "LOGITS",
        "LOGITS-ENTROPY",
    ]
    wl_path = str(tmp_path / "vlm_watchlist.yaml")
    with open(wl_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "version": 1,
                "model_id": manifest["model_id"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "watchlist_id": "test_qwen38_vlm",
                "trigger": "always",
                "window": {"start_token": 0, "end_token": 256, "every_n": 1},
                "precision": "fp16",
                "entries": [{"addresses": raw_addrs}],
            },
            fh,
        )

    in_ids = torch.tensor([[2, 4, 6, 8]])
    with GlassBox.watch(wl_path, model=vlm, manifest=manifest, prompt="test vlm") as ctx:
        out_ids = vlm.generate(in_ids, max_new_tokens=3)
        assert ctx.tap is not None
        assert ctx.tap.flags["hook_detached"] is False
        mat = ctx.tap.to_matrix()

    assert out_ids.shape == (1, 7)
    # 4 prefill tokens + 3 decode tokens = 7 rows across all expanded columns
    expected_cols = expand_addresses(raw_addrs)
    assert mat.shape == (7, len(expected_cols))
    assert not torch.isnan(torch.from_numpy(mat)).any()
    assert ctx.run_id is not None
    assert GlassBox.verify(ctx.run_id) is True

    df = GlassBox.to_dataframe(ctx.run_id)
    assert set(expected_cols).issubset(set(df["address"].unique()))

    # Verify FP8 tensor upcasting path in DeepTap when PyTorch supports float8_e4m3fn
    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is not None:
        tap_fp8 = DeepTap(["N-01-0001", "N-01-0002"])
        fp8_act = torch.tensor([[1.5, -2.0, 0.5, 3.0]], dtype=torch.float32).to(fp8_dtype)
        cols = tap_fp8.groups["layer-01.mlp"]
        gathered = tap_fp8._gather_torch_or_numpy(fp8_act, "layer-01.mlp", cols)
        assert gathered.shape == (1, 2)
        assert tap_fp8.dtype_meta["source"] == "fp8_upcast_on_device"


def test_drive_telemetry_streamer_1gb_passthrough_and_2gb_hard_stop(tmp_path: Any) -> None:
    """Executes Cell 2's DriveTelemetryStreamer: passes 1GB milestone, aborts at 2GB cap."""
    nb_path = os.path.join("notebooks", "a100_stretch_test.ipynb")
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    # Cell 2 defines HardStorageCapReached and DriveTelemetryStreamer
    cell2_src = "".join(nb["cells"][2]["source"])
    tree = ast.parse(_clean_cell_source(cell2_src))
    kept_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name in ("HardStorageCapReached", "DriveTelemetryStreamer")
    ]
    mod_ast = ast.Module(body=kept_nodes, type_ignores=[])

    # Scale 1 GB -> 3 KB and 2 GB -> 12 KB so exact cap logic runs in milliseconds
    scaled_1gb = 3_000
    scaled_2gb = 12_000
    ns: dict[str, Any] = {
        "os": os,
        "json": json,
        "shutil": __import__("shutil"),
        "ONE_GB_BYTES": scaled_1gb,
        "TWO_GB_HARD_CAP_BYTES": scaled_2gb,
        "SEALED_RUNS_DRIVE_DIR": str(tmp_path / "sealed_runs"),
    }
    os.makedirs(ns["SEALED_RUNS_DRIVE_DIR"], exist_ok=True)
    exec(compile(mod_ast, filename="cell2_streamer.py", mode="exec"), ns)

    cap_exc_cls = ns["HardStorageCapReached"]
    streamer_cls = ns["DriveTelemetryStreamer"]

    session_dir = str(tmp_path / "drive_session")
    os.makedirs(session_dir, exist_ok=True)
    streamer = streamer_cls(session_dir, flush_every=5)

    # Feed tokens until scaled 1GB is crossed — must NOT raise
    crossed_milestone_at: int | None = None
    hit_cap_at: int | None = None

    for tok_i in range(200):
        rec = {
            "token_pos": tok_i,
            "phase": "decode",
            "token_str": "safety",
            "mlp_256": {"L-01": [0.125] * 24},
        }
        try:
            streamer.append_token(rec)
            if streamer.logged_1gb_milestone and crossed_milestone_at is None:
                crossed_milestone_at = tok_i
        except cap_exc_cls:
            hit_cap_at = tok_i
            break

    assert crossed_milestone_at is not None, "Should cross 1 GB milestone cleanly"
    assert hit_cap_at is not None, "Should abruptly abort at 2 GB ceiling"
    assert hit_cap_at > crossed_milestone_at, "Must continue past 1 GB before 2 GB abort"
    assert streamer.aborted_at_2gb is True

    marker_path = os.path.join(session_dir, "ABORTED_AT_2GB_LIMIT.json")
    assert os.path.isfile(marker_path)
    with open(marker_path, encoding="utf-8") as fh:
        marker_data = json.load(fh)
    assert marker_data["status"] == "HARD_STOP_2GB_REACHED"
    assert marker_data["chunks_written"] >= 1

    # Total bytes on disk (including marker) must stay safely under scaled_2gb
    total_on_disk = sum(
        os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(session_dir) for f in fs
    )
    assert total_on_disk <= scaled_2gb
