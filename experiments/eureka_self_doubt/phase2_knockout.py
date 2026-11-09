"""Phase 2 Causal Knockout Runner for The Eureka Experiment (v3.1).

Validates Causal Necessity (Noising) by ablating candidate self-doubt attention heads:
- Evaluates 4 experimental conditions on the 25 Version B trap prompts:
  1. Baseline (unperturbed model)
  2. Zero-Ablation on candidate heads (mode="zero")
  3. Mean-Ablation on candidate heads (mode="mean", mean_value=clean_head_vector)
  4. Control Knockout: Mean-Ablation on matched non-candidate heads (specificity control)
- Quantifies Backtracking Suppression Rate (% of outputs reverting to the intuitive trap answer).
- Compares text fluency between Mean-Ablation and Zero-Ablation.
- Emits `phase2_results.json`.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable

try:
    import torch
except ImportError:
    torch = None

from glassbox.steering import AblationTarget, SteeringController


def grade_completion(text: str, correct_re: str, trap_re: str) -> str:
    """Grade completion after </think> or across output.
    
    Returns: 'correct' | 'trap' | 'other'
    """
    eval_text = text
    # Prefer evaluating post-reasoning solution if </think> is present
    if "</think>" in text:
        eval_text = text.split("</think>")[-1]

    has_trap = bool(re.search(trap_re, eval_text, re.IGNORECASE)) if trap_re else False
    has_correct = bool(re.search(correct_re, eval_text, re.IGNORECASE)) if correct_re else False

    if has_trap and not has_correct:
        return "trap"
    if has_correct and not has_trap:
        return "correct"
    if has_trap and has_correct:
        # Check which appears first in final answer
        m_trap = re.search(trap_re, eval_text, re.IGNORECASE)
        m_corr = re.search(correct_re, eval_text, re.IGNORECASE)
        if m_trap and m_corr:
            return "trap" if m_trap.start() < m_corr.start() else "correct"
        return "correct"
    return "other"


def load_manifest(manifest_path: str = "eureka_manifest.json") -> dict[str, Any]:
    """Load eureka manifest JSON."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_ablation_targets(
    candidate_heads: list[dict[str, Any]],
    mode: str = "mean",
) -> list[AblationTarget]:
    """Build list of AblationTarget instances for candidate heads."""
    targets: list[AblationTarget] = []
    for cand in candidate_heads:
        tgt_str = cand.get("target_str") or f"A-{cand['layer']:02d}-{cand['head']:02d}"
        if mode == "mean":
            mean_vec = cand.get("mean_vector", 0.0)
            targets.append(AblationTarget(target=tgt_str, mode="mean", mean_value=mean_vec))
        else:
            targets.append(AblationTarget(target=tgt_str, mode="zero"))
    return targets


def run_phase2_eval(
    model: Any,
    tokenizer: Any,
    manifest_data: dict[str, Any] | str = "eureka_manifest.json",
    prompts_data: dict[str, Any] | str = "prompts.yaml",
    n_samples: int = 4,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    device: str = "cuda",
    output_path: str = "phase2_results.json",
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Execute Phase 2 Causal Knockout across the 4 conditions."""
    if isinstance(manifest_data, str):
        manifest = load_manifest(manifest_data)
    else:
        manifest = manifest_data

    try:
        from .phase1_runner import format_chat_prompt, load_prompt_battery
    except ImportError:
        from phase1_runner import format_chat_prompt, load_prompt_battery

    if isinstance(prompts_data, str):
        battery = load_prompt_battery(prompts_data)
    else:
        battery = prompts_data

    cand_heads = manifest.get("top_candidate_heads", [])
    if not cand_heads:
        raise ValueError("No candidate heads found in eureka_manifest.json")

    # Pick non-candidate heads for control knockout (e.g. from lowest ranked heads)
    all_ranked = manifest.get("all_ranked_heads", [])
    control_candidates = all_ranked[-len(cand_heads):] if len(all_ranked) >= len(cand_heads) else []

    conditions = ["baseline", "zero_ablation", "mean_ablation", "control_knockout"]
    pairs = battery.get("pairs", [])
    total_evals = len(conditions) * len(pairs) * n_samples
    done_evals = 0

    results_by_condition: dict[str, dict[str, Any]] = {
        cond: {"correct": 0, "trap": 0, "other": 0, "total": 0, "runs": []}
        for cond in conditions
    }

    for cond in conditions:
        # Configure steering controller
        ablate_list: list[AblationTarget] = []
        if cond == "zero_ablation":
            ablate_list = build_ablation_targets(cand_heads, mode="zero")
        elif cond == "mean_ablation":
            ablate_list = build_ablation_targets(cand_heads, mode="mean")
        elif cond == "control_knockout":
            ablate_list = build_ablation_targets(control_candidates, mode="mean")

        controller = SteeringController(model=model, ablate=ablate_list) if ablate_list else None

        for pair in pairs:
            pair_id = pair.get("id", "pair")
            ver_b = pair.get("version_b", {})
            prompt_str = ver_b.get("prompt", "")
            correct_re = ver_b.get("correct_re", "")
            trap_re = ver_b.get("trap_re", "")

            formatted = format_chat_prompt(prompt_str, tokenizer)

            for s_idx in range(n_samples):
                if progress_callback:
                    progress_callback(cond, done_evals, total_evals)

                if controller:
                    controller.attach()

                try:
                    inputs = tokenizer(formatted, return_tensors="pt").to(device)
                    in_len = inputs["input_ids"].shape[1]
                    with torch.no_grad():
                        out = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                        )
                    gen_ids = out[0, in_len:].tolist()
                    decoded = tokenizer.decode(gen_ids, skip_special_tokens=False)
                finally:
                    if controller:
                        controller.detach()

                grade = grade_completion(decoded, correct_re, trap_re)
                results_by_condition[cond][grade] += 1
                results_by_condition[cond]["total"] += 1
                results_by_condition[cond]["runs"].append({
                    "prompt_id": pair_id,
                    "sample_idx": s_idx,
                    "grade": grade,
                    "output_snippet": decoded[:200] + ("..." if len(decoded) > 200 else ""),
                })
                done_evals += 1

    # Compute summary rates
    summary: dict[str, Any] = {}
    for cond, data in results_by_condition.items():
        tot = max(1, data["total"])
        summary[cond] = {
            "total_runs": tot,
            "backtracking_success_rate": round(data["correct"] / tot * 100, 2),
            "trap_submission_rate": round(data["trap"] / tot * 100, 2),
            "other_rate": round(data["other"] / tot * 100, 2),
        }

    final_payload = {
        "candidate_heads": [c.get("target_str") for c in cand_heads],
        "summary": summary,
        "details": results_by_condition,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)

    return final_payload
