"""Unit test validating Google Colab demo notebook structure, syntax, and execution."""

from __future__ import annotations

import ast
import json
import os
from typing import Any

import torch
import torch.nn as nn
import yaml

from glassbox import GlassBox
from glassbox.lens import compute_logit_lens, format_lens_table
from glassbox.viewer import render_html


def test_quickstart_demo_notebook_validity() -> None:
    nb_path = os.path.join("notebooks", "quickstart_demo.ipynb")
    assert os.path.isfile(nb_path), f"Notebook not found at {nb_path}"

    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    assert nb.get("nbformat") == 4
    cells = nb.get("cells", [])
    assert len(cells) >= 10, f"Expected at least 10 cells, got {len(cells)}"

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    markdown_cells = [c for c in cells if c.get("cell_type") == "markdown"]

    assert len(code_cells) >= 6
    assert len(markdown_cells) >= 5

    # Check required functionality mentions
    all_source = " ".join("".join(c.get("source", [])) for c in cells)
    assert "glassbox" in all_source
    assert "Qwen/Qwen2.5-0.5B" in all_source
    assert "EleutherAI/pythia-70m" in all_source
    assert "gpt2" in all_source
    assert "Baseline PyTorch" in all_source
    assert "Naive PyTorch Hook" in all_source
    assert "GlassBox Async DMA" in all_source
    assert "format_lens_table" in all_source or "lens" in all_source
    assert "render_html" in all_source
    assert "IPython" in all_source
    assert "Qwen/Qwen2.5-3B" in all_source
    assert "Test C" in all_source
    assert "scaling_results" in all_source
    assert "t4_scaling_matrix.csv" in all_source
    assert "t4_scaling_matrix.json" in all_source

    # Validate Python syntax of all code cells
    for i, cell in enumerate(code_cells):
        raw_source = "".join(cell.get("source", []))
        cleaned_lines = []
        in_shell = False
        for line in raw_source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("!") or stripped.startswith("%") or in_shell:
                cleaned_lines.append(f"# {line}")
                in_shell = line.rstrip().endswith("\\")
            else:
                cleaned_lines.append(line)
                in_shell = False
        cleaned_source = "\n".join(cleaned_lines)
        try:
            ast.parse(cleaned_source)
        except SyntaxError as e:
            raise AssertionError(f"SyntaxError in code cell {i}:\n{raw_source}") from e


def test_model_config_fallback_robustness() -> None:
    """Verifies that intermediate size calculation handles None (like HF GPT2Config)."""

    class MockGPT2Config:
        n_layer = 12
        n_embd = 768
        n_head = 12
        n_inner = None  # Crucial: standard GPT-2 config sets n_inner to None
        vocab_size = 50257

    cfg = MockGPT2Config()
    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or 12
    hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None) or 768
    num_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None) or 12
    intermediate_size = (
        getattr(cfg, "intermediate_size", None)
        or getattr(cfg, "n_inner", None)
        or (4 * hidden_size)
    )

    assert num_layers == 12
    assert hidden_size == 768
    assert num_heads == 12
    assert intermediate_size == 3072


def test_notebook_simulation_workflow(tmp_path: Any) -> None:
    """End-to-end execution of the exact telemetry flow in quickstart_demo.ipynb."""

    class MockMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_proj = nn.Linear(16, 32)
            self.down_proj = nn.Linear(32, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.up_proj(x))

    class MockAttn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.o_proj = nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.o_proj(x)

    class MockBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = MockMLP()
            self.self_attn = MockAttn()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.self_attn(x) + self.mlp(x)

    class MockTransformerModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

            class Config:
                num_hidden_layers = 2
                hidden_size = 16
                num_attention_heads = 2
                intermediate_size = 32
                vocab_size = 100

            self.config = Config()
            self.layers = nn.ModuleList([MockBlock(), MockBlock()])
            self.norm = nn.LayerNorm(16)
            self.lm_head = nn.Linear(16, 100)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = torch.randn(1, input_ids.shape[1], 16)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(self.norm(x))

        def generate(
            self, input_ids: torch.Tensor, max_new_tokens: int = 4, **kwargs: Any
        ) -> torch.Tensor:
            curr = input_ids
            for _ in range(max_new_tokens):
                _ = self(curr)
                curr = torch.cat([curr, torch.tensor([[1]])], dim=1)
            return curr

    model = MockTransformerModel()
    manifest = {
        "model_id": "mock_demo_model",
        "checkpoint_sha256": "0" * 64,
        "architecture": {
            "layers": 2,
            "hidden": 16,
            "heads": 2,
            "mlp_per_layer": 32,
            "vocab": 100,
        },
    }

    wl_path = str(tmp_path / "mock_watchlist.yaml")
    wl_dict = {
        "version": 1,
        "model_id": manifest["model_id"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "watchlist_id": "test_nb_flow",
        "trigger": "always",
        "window": {"start_token": 0, "end_token": 128, "every_n": 1},
        "precision": "fp16",
        "entries": [{"addresses": ["N-01-0001..0004", "A-01-01..02", "L-01", "LOGITS"]}],
    }
    with open(wl_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(wl_dict, f)

    input_ids = torch.tensor([[1, 2, 3]])
    last_run_id = None
    generated_tokens_output = None

    # Benchmark-style watch loop
    for _ in range(2):
        with GlassBox.watch(
            wl_path,
            model=model,
            manifest=manifest,
            prompt="Hello world",
        ) as ctx:
            out_ids = model.generate(input_ids, max_new_tokens=4)
            generated_tokens_output = out_ids
        last_run_id = ctx.run_id

    # 1. Verify run_id is captured properly
    assert last_run_id is not None
    assert last_run_id.startswith("RUN-")
    assert generated_tokens_output is not None

    # 2. Cryptographic verification
    assert GlassBox.verify(last_run_id) is True

    # 3. DataFrame conversion & top neuron inspection
    df = GlassBox.to_dataframe(last_run_id)
    assert hasattr(df, "shape")
    assert df.shape[0] > 0
    neuron_df = df[df["address"].str.startswith("N-")]
    assert not neuron_df.empty

    # 4. Logit lens computation & table formatting
    lens_data = compute_logit_lens(last_run_id, token_pos=0)
    assert lens_data["run_id"] == last_run_id
    lens_tbl = format_lens_table(lens_data)
    assert "Logit Lens & Residual Progression" in lens_tbl

    # 5. Interactive Calm Viewer generation
    calm_html = render_html(last_run_id)
    assert "<html" in calm_html.lower() or "<!doctype html>" in calm_html.lower()
    assert last_run_id in calm_html
