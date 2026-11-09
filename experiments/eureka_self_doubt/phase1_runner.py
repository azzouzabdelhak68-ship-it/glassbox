"""Phase 1 Flight Recording Runner for The Eureka Experiment (v3.1).

Coordinates dense flight recording of reasoning traces across 25 de-memorized prompt pairs
(Version A clean baseline vs Version B trap prompt) with GlassBox zero-overhead tap.
Stores run metadata, token traces, and attention layer activations for downstream
cognitive pivot auditing and cDLA circuit attribution.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import yaml

try:
    import torch
except ImportError:
    torch = None

from glassbox import GlassBox


def load_prompt_battery(yaml_path: str = "prompts.yaml") -> dict[str, Any]:
    """Load the 25-pair prompt battery YAML file."""
    if not os.path.exists(yaml_path):
        # Look in directory of this script if relative
        alt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), yaml_path)
        if os.path.exists(alt_path):
            yaml_path = alt_path

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def format_chat_prompt(prompt_text: str, tokenizer: Any = None) -> str:
    """Format prompt with standard chat template for reasoning models."""
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback to standard chat format
    return f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n<think>\n"


def run_phase1_cohort(
    model: Any,
    tokenizer: Any,
    prompts_data: dict[str, Any] | str = "prompts.yaml",
    n_samples: int = 8,
    layer_range: tuple[int, int] = (35, 50),
    store_dir: str = "./glassbox_store",
    max_new_tokens: int = 1024,
    temperature: float = 0.6,
    device: str = "cuda",
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[int, Any]]]:
    """Execute Phase 1 flight recording across prompt battery.
    
    Returns:
        (runs_metadata_list, run_activations_map)
    """
    if isinstance(prompts_data, str):
        battery = load_prompt_battery(prompts_data)
    else:
        battery = prompts_data

    pairs = battery.get("pairs", [])
    total_runs = len(pairs) * 2 * n_samples

    # Define GlassBox watchlist: input to o_proj across candidate layers
    watchlist = [f"layer-{l:02d}.attn" for l in range(layer_range[0], layer_range[1] + 1)]

    runs_meta: list[dict[str, Any]] = []
    run_activations: dict[str, dict[int, Any]] = {}
    completed = 0

    for pair in pairs:
        pair_id = pair.get("id", "pair")
        domain = pair.get("domain", "general")

        for version_key, ver_label in [("version_a", "A"), ("version_b", "B")]:
            ver_data = pair.get(version_key, {})
            prompt_str = ver_data.get("prompt", "")
            correct_re = ver_data.get("correct_re", "")
            trap_re = ver_data.get("trap_re", "")

            formatted_prompt = format_chat_prompt(prompt_str, tokenizer)

            for sample_idx in range(n_samples):
                run_id = f"RUN-{pair_id}-{ver_label}-s{sample_idx:02d}-{int(time.time()*1000)%1000000:06d}"

                if progress_callback:
                    progress_callback(completed, total_runs, run_id)

                # Execute with GlassBox flight recording
                with GlassBox(
                    model=model,
                    watchlist=watchlist,
                    store=store_dir,
                    precision="fp16",
                    run_id=run_id,
                ) as gb:
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                    input_len = inputs["input_ids"].shape[1]

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                        )

                    gen_tokens = outputs[0, input_len:].tolist()
                    decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

                    # Extract token strings
                    tokens_list: list[tuple[int, str]] = []
                    for t_pos, tid in enumerate(gen_tokens):
                        t_str = tokenizer.decode([tid], skip_special_tokens=False)
                        tokens_list.append((t_pos, t_str))

                    # Capture in-memory activation slices if available from tap hooks
                    act_map: dict[int, Any] = {}
                    if hasattr(gb, "activations") and isinstance(gb.activations, dict):
                        for l in range(layer_range[0], layer_range[1] + 1):
                            addr = f"layer-{l:02d}.attn"
                            if addr in gb.activations:
                                act_map[l] = gb.activations[addr]
                    run_activations[run_id] = act_map

                record = {
                    "run_id": run_id,
                    "prompt_id": pair_id,
                    "domain": domain,
                    "version": ver_label,
                    "sample_idx": sample_idx,
                    "prompt_text": prompt_str,
                    "output_text": decoded_text,
                    "tokens": tokens_list,
                    "correct_re": correct_re,
                    "trap_re": trap_re,
                }
                runs_meta.append(record)
                completed += 1

    return runs_meta, run_activations


def save_phase1_records(
    runs_meta: list[dict[str, Any]],
    output_path: str = "phase1_runs.json",
) -> None:
    """Save Phase 1 runs metadata to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(runs_meta, f, indent=2)
