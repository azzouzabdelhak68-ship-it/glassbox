"""Phase 3 Causal Trigger Runner for The Eureka Experiment (v3.1).

Validates Causal Sufficiency by steering reasoning models with the extracted PC1 Eureka vector:
- Loads `u_eureka.pt` and primary layer from `eureka_manifest.json`.
- Applies 2-token pulsed steering (`pulse_steps=2`) during autoregressive decoding on
  clean Version A prompts (where the model normally solves immediately without doubt).
- Sweeps steering multiplier alpha in [0.0, 0.5, 1.0, 2.0, 4.0] alongside an orthogonal control vector.
- Quantifies Hesitation Induction Rate (% of clean runs forced into self-doubt/re-verification)
  vs Syntactic Coherence Rate.
- Emits `phase3_results.json`.
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

from glassbox.steering import SteeringController, SteeringVector


_HESITATION_PATTERNS = [
    r"\bwait\b",
    r"\bactually\b",
    r"\bhold on\b",
    r"\blet me re-?check\b",
    r"\blet me double-?check\b",
    r"\bis that correct\b",
    r"\bwait a second\b",
    r"\bhold up\b",
    r"\bwait,?\s+is\b",
    r"\bhowever,?\s+what if\b",
    r"\bcould there be an error\b",
    r"\bmistake\b",
    r"\brethink\b",
]


def detect_hesitation(text: str) -> bool:
    """Check if generated text exhibits self-doubt or backtracking hesitation."""
    # Look inside <think> or full text
    eval_text = text
    if "<think>" in text:
        think_part = text.split("<think>")[1]
        eval_text = think_part.split("</think>")[0] if "</think>" in think_part else think_part

    for pat in _HESITATION_PATTERNS:
        if re.search(pat, eval_text, re.IGNORECASE):
            return True
    return False


def is_coherent(text: str) -> bool:
    """Check for pathological repetition or non-fluency (syntactic collapse)."""
    if len(text.strip()) < 10:
        return False
    # Check for repeated 4-gram loop
    words = text.split()
    if len(words) >= 20:
        for i in range(len(words) - 15):
            seq = " ".join(words[i : i + 4])
            if text.count(seq) > 6:
                return False
    return True


def generate_orthogonal_vector(v: torch.Tensor) -> torch.Tensor:
    """Generate a random unit vector strictly orthogonal to v."""
    rand = torch.randn_like(v)
    # Gram-Schmidt orthogonalization: r_orth = r - (r . v / ||v||^2) * v
    dot = torch.dot(rand.flatten(), v.flatten())
    v_norm_sq = torch.dot(v.flatten(), v.flatten())
    orth = rand - (dot / v_norm_sq) * v
    return orth / torch.linalg.norm(orth)


def run_phase3_eval(
    model: Any,
    tokenizer: Any,
    manifest_data: dict[str, Any] | str = "eureka_manifest.json",
    vector_data: torch.Tensor | str = "u_eureka.pt",
    prompts_data: dict[str, Any] | str = "prompts.yaml",
    pulse_steps: int = 2,
    pulse_start_step: int = 15,
    alpha_sweep: list[float] | None = None,
    n_samples: int = 4,
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    device: str = "cuda",
    output_path: str = "phase3_results.json",
    progress_callback: Callable[[float, int, int], None] | None = None,
) -> dict[str, Any]:
    """Execute Phase 3 Pulsed Steering evaluation on clean Version A prompts."""
    if alpha_sweep is None:
        alpha_sweep = [0.0, 0.5, 1.0, 2.0, 4.0]

    try:
        from .phase1_runner import format_chat_prompt, load_prompt_battery
    except ImportError:
        from phase1_runner import format_chat_prompt, load_prompt_battery

    if isinstance(manifest_data, str):
        with open(manifest_data, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = manifest_data

    if isinstance(vector_data, str):
        u_eureka = torch.load(vector_data, map_location=device)
    else:
        u_eureka = vector_data.to(device)

    if isinstance(prompts_data, str):
        battery = load_prompt_battery(prompts_data)
    else:
        battery = prompts_data

    target_layer = int(manifest.get("primary_svd_layer", 42))
    orth_vector = generate_orthogonal_vector(u_eureka)

    pairs = battery.get("pairs", [])
    # Evaluate across alphas plus orthogonal control at alpha=2.0
    conditions: list[tuple[str, float, torch.Tensor | None]] = [
        (f"alpha_{a:.1f}", a, u_eureka) for a in alpha_sweep
    ]
    conditions.append(("orthogonal_control_2.0", 2.0, orth_vector))

    total_evals = len(conditions) * len(pairs) * n_samples
    done_evals = 0

    results_by_condition: dict[str, dict[str, Any]] = {
        name: {"hesitation_count": 0, "coherent_count": 0, "total": 0, "runs": []}
        for name, _, _ in conditions
    }

    for name, scale, vec in conditions:
        steer_list: list[SteeringVector] = []
        if vec is not None and scale > 0.0:
            steer_list.append(
                SteeringVector(
                    layer=target_layer,
                    vector=vec,
                    scale=scale,
                    mode="decode",
                    pulse_steps=pulse_steps,
                    pulse_start_step=pulse_start_step,
                )
            )

        controller = SteeringController(model=model, steer=steer_list) if steer_list else None

        for pair in pairs:
            pair_id = pair.get("id", "pair")
            ver_a = pair.get("version_a", {})
            prompt_str = ver_a.get("prompt", "")

            formatted = format_chat_prompt(prompt_str, tokenizer)

            for s_idx in range(n_samples):
                if progress_callback:
                    progress_callback(scale, done_evals, total_evals)

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

                hesitated = detect_hesitation(decoded)
                coherent = is_coherent(decoded)

                results_by_condition[name]["total"] += 1
                if hesitated:
                    results_by_condition[name]["hesitation_count"] += 1
                if coherent:
                    results_by_condition[name]["coherent_count"] += 1

                results_by_condition[name]["runs"].append({
                    "prompt_id": pair_id,
                    "sample_idx": s_idx,
                    "hesitation": hesitated,
                    "coherent": coherent,
                    "output_snippet": decoded[:200] + ("..." if len(decoded) > 200 else ""),
                })
                done_evals += 1

    summary: dict[str, Any] = {}
    for name, data in results_by_condition.items():
        tot = max(1, data["total"])
        summary[name] = {
            "total_runs": tot,
            "hesitation_induction_rate": round(data["hesitation_count"] / tot * 100, 2),
            "coherence_rate": round(data["coherent_count"] / tot * 100, 2),
        }

    final_payload = {
        "target_layer": target_layer,
        "pulse_steps": pulse_steps,
        "pulse_start_step": pulse_start_step,
        "summary": summary,
        "details": results_by_condition,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_payload, f, indent=2)

    return final_payload
