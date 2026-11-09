"""Unit tests for The Eureka Experiment (v3.1) suite."""

import json
import os
import tempfile
import numpy as np
import pytest
import torch

from experiments.eureka_self_doubt.analyzer_auditor import (
    ANALYZER_SYSTEM_PROMPT,
    audit_cohort,
    audit_run,
    build_analyzer_prompt,
    format_token_trace,
    heuristic_audit,
    parse_analyzer_output,
)
from experiments.eureka_self_doubt.cdla_svd_engine import (
    CDLASvdEngine,
    benjamini_hochberg,
    compute_paired_effect_sizes,
    extract_svd_pc1,
)
from experiments.eureka_self_doubt.phase1_runner import (
    format_chat_prompt,
    load_prompt_battery,
)
from experiments.eureka_self_doubt.phase2_knockout import (
    build_ablation_targets,
    grade_completion,
)
from experiments.eureka_self_doubt.phase3_trigger import (
    detect_hesitation,
    generate_orthogonal_vector,
    is_coherent,
)


class TestAnalyzerAuditor:
    """Tests for Cell 3A Analyzer Auditor."""

    def test_format_token_trace(self):
        tokens = [(0, "<think>"), (1, "Let"), (2, "see\n")]
        formatted = format_token_trace(tokens)
        assert '[t=0] "<think>"' in formatted
        assert '[t=1] "Let"' in formatted
        assert r'[t=2] "see\n"' in formatted

    def test_build_analyzer_prompt(self):
        prompt = build_analyzer_prompt("What is 2+2?", ["4"])
        assert "USER PROMPT GIVEN TO TARGET MODEL:" in prompt
        assert "What is 2+2?" in prompt
        assert '[t=0] "4"' in prompt

    def test_parse_analyzer_output_clean(self):
        raw = """{
            "trajectory_type": "GENUINE_BACKTRACK",
            "t_commit": 12,
            "t_pivot": 24,
            "t_resolve": 28,
            "pivot_token": "Wait",
            "confidence": 0.95,
            "reasoning": "Model caught the contradiction."
        }"""
        res = parse_analyzer_output(raw)
        assert res["trajectory_type"] == "GENUINE_BACKTRACK"
        assert res["t_pivot"] == 24
        assert res["t_commit"] == 12
        assert res["pivot_token"] == "Wait"
        assert res["confidence"] == 0.95

    def test_parse_analyzer_output_markdown_fence(self):
        raw = """Here is my evaluation:
        ```json
        {
            "trajectory_type": "IMMEDIATE_SOLVE",
            "t_commit": null,
            "t_pivot": null,
            "t_resolve": null,
            "pivot_token": null,
            "confidence": 0.99,
            "reasoning": "Direct solution."
        }
        ```"""
        res = parse_analyzer_output(raw)
        assert res["trajectory_type"] == "IMMEDIATE_SOLVE"
        assert res["t_pivot"] is None

    def test_parse_analyzer_output_invalid(self):
        with pytest.raises(ValueError):
            parse_analyzer_output("not json at all")

        with pytest.raises(ValueError):
            parse_analyzer_output('{"trajectory_type": "INVALID_TYPE"}')

    def test_heuristic_audit_detection(self):
        tokens = [
            (0, "<think>"),
            (5, "So"),
            (6, "the"),
            (7, "answer"),
            (8, "is"),
            (9, "150W."),
            (10, "Wait,"),
            (11, "that"),
            (12, "cannot"),
            (13, "be"),
            (14, "right."),
        ]
        res = heuristic_audit(tokens)
        assert res["trajectory_type"] == "GENUINE_BACKTRACK"
        assert res["t_pivot"] == 10
        assert "Wait" in res["pivot_token"]

    def test_audit_cohort_execution(self, tmp_path):
        runs = [
            {
                "run_id": "RUN-01",
                "prompt_id": "p1",
                "version": "B",
                "prompt_text": "trap",
                "tokens": [(i, tok) for i, tok in enumerate(["start", "math", "Wait", "fix"])],
            },
            {
                "run_id": "RUN-02",
                "prompt_id": "p1",
                "version": "A",
                "prompt_text": "clean",
                "tokens": [(0, "direct"), (1, "answer")],
            },
        ]
        out_file = str(tmp_path / "semantic_pivots.json")
        res = audit_cohort(runs, analyzer_fn=None, output_path=out_file)
        assert res["total_runs"] == 2
        assert res["summary"]["GENUINE_BACKTRACK"] >= 0
        assert os.path.exists(out_file)


class TestCDLASvdEngine:
    """Tests for Cell 3B Contrastive DLA & SVD Slicer."""

    def test_benjamini_hochberg(self):
        p_vals = [0.001, 0.004, 0.009, 0.05, 0.20]
        sig, q = benjamini_hochberg(p_vals, alpha=0.05)
        assert sig[0]  # First should be highly significant
        assert q[0] <= 0.05
        assert not sig[-1]  # 0.20 not significant

    def test_extract_svd_pc1(self):
        torch.manual_seed(42)
        # Create 20 samples in R^128 with strong signal along first axis
        X = torch.randn(20, 128)
        X[:, 0] += 5.0

        u_eureka, ratio = extract_svd_pc1(X)
        assert u_eureka.shape == (128,)
        assert torch.isclose(torch.linalg.norm(u_eureka), torch.tensor(1.0), atol=1e-5)
        assert ratio > 0.0

    def test_engine_analyze_traces_synthetic(self, tmp_path):
        np.random.seed(42)
        # 4 layers (35..38), 4 heads, head_dim=16 => d_model = 64
        engine = CDLASvdEngine(d_model=64, num_heads=4, head_dim=16, layer_range=(35, 38))

        # Synthetic pivots manifest
        pivots = {
            "runs": [
                {
                    "run_id": "RUN-B1",
                    "version": "B",
                    "trajectory_type": "GENUINE_BACKTRACK",
                    "t_commit": 5,
                    "t_pivot": 15,
                },
                {
                    "run_id": "RUN-B2",
                    "version": "B",
                    "trajectory_type": "GENUINE_BACKTRACK",
                    "t_commit": 6,
                    "t_pivot": 16,
                },
                {
                    "run_id": "RUN-A1",
                    "version": "A",
                    "trajectory_type": "IMMEDIATE_SOLVE",
                    "t_commit": None,
                    "t_pivot": None,
                },
            ]
        }

        # Synthetic activations: inject huge surge specifically into Layer 36, Head 2
        T = 25
        run_acts = {}
        for r in ["RUN-B1", "RUN-B2"]:
            layer_map = {}
            for l in range(35, 39):
                arr = np.random.randn(T, 64).astype(np.float32)
                if l == 36:
                    # Head 2 is indices [16:32]
                    # Inject surge at pivot tokens [12..16]
                    arr[12:17, 16:32] += 10.0
                layer_map[l] = arr
            run_acts[r] = layer_map

        # Clean run
        layer_map_a = {}
        for l in range(35, 39):
            layer_map_a[l] = np.random.randn(T, 64).astype(np.float32)
        run_acts["RUN-A1"] = layer_map_a

        manifest, u_eureka = engine.analyze_traces(pivots, run_acts, top_k=2, alpha=0.05)

        # Candidate heads verification
        top_cand = manifest["top_candidate_heads"][0]
        assert top_cand["layer"] == 36
        assert top_cand["head"] == 2
        assert len(top_cand["mean_vector"]) == 16
        assert u_eureka.shape == (64,)

        # Save and verify
        m_file = str(tmp_path / "eureka_manifest.json")
        v_file = str(tmp_path / "u_eureka.pt")
        engine.save_artifacts(manifest, u_eureka, m_file, v_file)
        assert os.path.exists(m_file)
        assert os.path.exists(v_file)


class TestPhase2Knockout:
    """Tests for Phase 2 Causal Knockout and Grading."""

    def test_grade_completion_trap(self):
        text = "<think>some reasoning</think>The cooling unit draws 150W."
        grade = grade_completion(text, correct_re=r"75\s*W", trap_re=r"150\s*W")
        assert grade == "trap"

    def test_grade_completion_correct(self):
        text = "<think>Wait, 150W was wrong.</think>Therefore, it draws 75 W."
        grade = grade_completion(text, correct_re=r"75\s*W", trap_re=r"150\s*W")
        assert grade == "correct"

    def test_grade_completion_other(self):
        text = "I don't know the answer."
        grade = grade_completion(text, correct_re=r"75\s*W", trap_re=r"150\s*W")
        assert grade == "other"

    def test_build_ablation_targets(self):
        cands = [
            {"target_str": "A-36-02", "layer": 36, "head": 2, "mean_vector": [0.5] * 128},
            {"target_str": "A-41-10", "layer": 41, "head": 10, "mean_vector": [0.1] * 128},
        ]
        mean_targets = build_ablation_targets(cands, mode="mean")
        assert len(mean_targets) == 2
        assert mean_targets[0].mode == "mean"
        assert mean_targets[0].target == "A-36-02"

        zero_targets = build_ablation_targets(cands, mode="zero")
        assert zero_targets[0].mode == "zero"


class TestPhase3Trigger:
    """Tests for Phase 3 Causal Sufficiency and Steering."""

    def test_detect_hesitation(self):
        assert detect_hesitation("<think>Let's calculate. Wait, is that right?</think>42")
        assert detect_hesitation("<think>Actually, let me double-check this.</think>")
        assert not detect_hesitation("<think>The sum is straightforward: 2+2=4.</think>4")

    def test_is_coherent(self):
        assert is_coherent("This is a fluent explanation of the solution.")
        looping = "the server draws 100W " * 15
        assert not is_coherent(looping)

    def test_generate_orthogonal_vector(self):
        v = torch.randn(128)
        v = v / torch.linalg.norm(v)
        v_orth = generate_orthogonal_vector(v)

        assert torch.isclose(torch.linalg.norm(v_orth), torch.tensor(1.0), atol=1e-5)
        dot = torch.dot(v, v_orth)
        assert torch.abs(dot) < 1e-5


class TestPromptsBattery:
    """Tests for prompt battery configuration."""

    def test_load_prompt_battery(self):
        battery = load_prompt_battery("experiments/eureka_self_doubt/prompts.yaml")
        assert battery["version"] == 1
        assert "pairs" in battery
        assert len(battery["pairs"]) == 25

        domains = {p["domain"] for p in battery["pairs"]}
        assert len(domains) >= 5
        for p in battery["pairs"]:
            assert "version_a" in p and "prompt" in p["version_a"]
            assert "version_b" in p and "prompt" in p["version_b"]
            assert "trap_re" in p["version_b"]
