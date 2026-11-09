"""Analyzer Auditor (Cell 3A Component) for The Eureka Experiment (v3.1).

Implements the independent semantic evaluator protocol:
- Formats tokenized reasoning traces into numbered [t=N] tokens for evaluation.
- Prompts the external Analyzer LLM (Qwen-27B / Qwen-32B) to locate semantic cognitive
  pivots (t_commit, t_pivot, t_resolve) and classify trajectory types.
- Safely parses and validates structured JSON outputs.
- Provides a deterministic heuristic audit fallback for offline verification / CI testing.
- Aggregates cohort results into `semantic_pivots.json`.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

ANALYZER_SYSTEM_PROMPT = """You are an expert cognitive evaluator and mechanistic interpretability auditor.
Your task is to analyze the reasoning trace (<think> ... </think>) of an LLM presented with a cognitive trap prompt.
The trace is presented with exact token indices: [t=N] "token".

You must classify the trajectory into exactly one category:
1. "GENUINE_BACKTRACK": The model initially commits to an intuitive or erroneous path, recognizes the contradiction or constraint trap, pauses or reverses, and redirects to a corrected solution.
2. "IMMEDIATE_SOLVE": The model identifies the true constraint or solution immediately without committing to the trap or displaying self-doubt.
3. "BLIND_FAILURE": The model falls for the trap and finishes generation without ever noticing the contradiction or error.

If the trajectory is "GENUINE_BACKTRACK", you must pinpoint the exact token positions:
- t_commit: The token index where the model locks into the intuitive/erroneous hypothesis.
- t_pivot: The exact token index where the model initiates self-doubt, hesitates, or pivots away from the intuitive error (the semantic cognitive pivot, e.g. 'Wait', 'Actually', 'Since 2 is...', 'Hold on').
- t_resolve: The token index where the model begins formulating the valid alternative path.

Return your evaluation STRICTLY as a valid JSON object with these exact keys:
{
  "trajectory_type": "GENUINE_BACKTRACK" | "IMMEDIATE_SOLVE" | "BLIND_FAILURE",
  "t_commit": <integer or null>,
  "t_pivot": <integer or null>,
  "t_resolve": <integer or null>,
  "pivot_token": <string or null>,
  "confidence": <float between 0.0 and 1.0>,
  "reasoning": "<concise explanation of why t_pivot was selected>"
}
"""

_COMMON_HESITATION_WORDS = [
    "wait", "actually", "hold", "re-evaluat", "recheck", "re-check",
    "mistake", "wrong", "however", "contradiction", "impossible",
    "except", "unless", "rethink", "hang on", "pause", "re-read",
]


def format_token_trace(tokens: list[tuple[int, str]] | list[str]) -> str:
    """Format token sequence into indexed lines: [t=N] "token"."""
    lines: list[str] = []
    for idx, item in enumerate(tokens):
        if isinstance(item, tuple):
            pos, tok = item
        else:
            pos, tok = idx, item
        # Escape newlines and quotes cleanly for display
        escaped_tok = tok.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n").replace("\t", "\\t")
        lines.append(f'[t={pos}] "{escaped_tok}"')
    return "\n".join(lines)


def build_analyzer_prompt(prompt_text: str, tokens: list[tuple[int, str]] | list[str]) -> str:
    """Construct user prompt for the Analyzer LLM."""
    trace_str = format_token_trace(tokens)
    return (
        f"USER PROMPT GIVEN TO TARGET MODEL:\n{prompt_text}\n\n"
        f"TARGET MODEL TOKEN TRACE (<think> tokens):\n{trace_str}\n\n"
        "Analyze the trace above and output the evaluation JSON object."
    )


def parse_analyzer_output(text: str) -> dict[str, Any]:
    """Parse and validate JSON response from Analyzer LLM."""
    cleaned = text.strip()
    # Remove markdown code block fences if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1)
    else:
        # Match first balanced JSON object
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            cleaned = cleaned[brace_start : brace_end + 1]

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse Analyzer LLM output as JSON: {exc}\nRaw: {text!r}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Analyzer output must be a JSON object, got {type(data)}")

    traj = str(data.get("trajectory_type", "")).strip().upper()
    valid_trajs = {"GENUINE_BACKTRACK", "IMMEDIATE_SOLVE", "BLIND_FAILURE"}
    if traj not in valid_trajs:
        raise ValueError(f"Invalid trajectory_type: {traj!r}. Expected one of {valid_trajs}")

    t_commit = data.get("t_commit")
    t_pivot = data.get("t_pivot")
    t_resolve = data.get("t_resolve")

    if traj == "GENUINE_BACKTRACK":
        if t_pivot is None or not isinstance(t_pivot, (int, float)):
            raise ValueError("GENUINE_BACKTRACK must specify a numeric t_pivot")
        t_pivot = int(t_pivot)
        if t_commit is not None:
            t_commit = int(t_commit)
        if t_resolve is not None:
            t_resolve = int(t_resolve)
    else:
        t_commit = None
        t_pivot = None
        t_resolve = None

    confidence = float(data.get("confidence", 1.0))
    confidence = max(0.0, min(1.0, confidence))

    return {
        "trajectory_type": traj,
        "t_commit": t_commit,
        "t_pivot": t_pivot,
        "t_resolve": t_resolve,
        "pivot_token": str(data.get("pivot_token")) if data.get("pivot_token") is not None else None,
        "confidence": confidence,
        "reasoning": str(data.get("reasoning", "")),
    }


def heuristic_audit(
    tokens: list[tuple[int, str]] | list[str],
    prompt_text: str = "",
) -> dict[str, Any]:
    """Deterministic fallback auditor for offline verification and testing.
    
    Scans token stream for hesitation / contradiction markers and structural reversals.
    """
    token_pairs: list[tuple[int, str]] = []
    for idx, item in enumerate(tokens):
        if isinstance(item, tuple):
            token_pairs.append((item[0], item[1]))
        else:
            token_pairs.append((idx, item))

    if not token_pairs:
        return {
            "trajectory_type": "BLIND_FAILURE",
            "t_commit": None,
            "t_pivot": None,
            "t_resolve": None,
            "pivot_token": None,
            "confidence": 0.5,
            "reasoning": "Empty token trace",
        }

    # Search for hesitation tokens
    pivot_pos: int | None = None
    pivot_tok: str | None = None
    commit_pos: int | None = None

    for pos, tok in token_pairs:
        tok_lower = tok.strip().lower()
        if not tok_lower:
            continue
        if commit_pos is None and pos >= 5:
            # Assume initial commitment early in reasoning
            commit_pos = pos

        for marker in _COMMON_HESITATION_WORDS:
            if marker in tok_lower and pos > 5:
                pivot_pos = pos
                pivot_tok = tok
                break
        if pivot_pos is not None:
            break

    if pivot_pos is not None:
        t_commit = max(1, pivot_pos // 2) if commit_pos is None else commit_pos
        t_resolve = min(len(token_pairs) - 1, pivot_pos + 3)
        return {
            "trajectory_type": "GENUINE_BACKTRACK",
            "t_commit": t_commit,
            "t_pivot": pivot_pos,
            "t_resolve": t_resolve,
            "pivot_token": pivot_tok,
            "confidence": 0.85,
            "reasoning": f"Found hesitation marker {pivot_tok!r} at position {pivot_pos}",
        }

    # If no hesitation found, check if trace was short or long
    if len(token_pairs) < 30:
        return {
            "trajectory_type": "IMMEDIATE_SOLVE",
            "t_commit": None,
            "t_pivot": None,
            "t_resolve": None,
            "pivot_token": None,
            "confidence": 0.70,
            "reasoning": "Short direct answer without hesitation markers",
        }

    return {
        "trajectory_type": "BLIND_FAILURE",
        "t_commit": min(10, len(token_pairs) - 1),
        "t_pivot": None,
        "t_resolve": None,
        "pivot_token": None,
        "confidence": 0.75,
        "reasoning": "Long trajectory with no hesitation or course correction detected",
    }


def audit_run(
    run_record: dict[str, Any],
    analyzer_fn: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Audit an individual flight run record using either the Analyzer LLM or heuristic fallback.
    
    `run_record` is expected to have:
      - run_id: str
      - prompt_id: str
      - version: 'A' | 'B'
      - prompt_text: str
      - tokens: list of tokens (strings or (int, str) tuples)
    """
    run_id = run_record.get("run_id", "")
    prompt_id = run_record.get("prompt_id", "")
    version = run_record.get("version", "B")
    prompt_text = run_record.get("prompt_text", "")
    tokens = run_record.get("tokens", [])

    if analyzer_fn is not None and version != "A":
        prompt = build_analyzer_prompt(prompt_text, tokens)
        try:
            response_text = analyzer_fn(prompt)
            audit_res = parse_analyzer_output(response_text)
        except Exception:
            audit_res = heuristic_audit(tokens, prompt_text)
    else:
        audit_res = heuristic_audit(tokens, prompt_text)

    audit_res.update({
        "run_id": run_id,
        "prompt_id": prompt_id,
        "version": version,
    })
    return audit_res


def audit_cohort(
    runs: list[dict[str, Any]],
    analyzer_fn: Callable[[str], str] | None = None,
    output_path: str = "semantic_pivots.json",
) -> dict[str, Any]:
    """Audit all recorded runs and output `semantic_pivots.json`."""
    results: list[dict[str, Any]] = []
    backtrack_count = 0
    immediate_count = 0
    blind_count = 0

    for run in runs:
        res = audit_run(run, analyzer_fn=analyzer_fn)
        results.append(res)
        ttype = res["trajectory_type"]
        if ttype == "GENUINE_BACKTRACK":
            backtrack_count += 1
        elif ttype == "IMMEDIATE_SOLVE":
            immediate_count += 1
        else:
            blind_count += 1

    manifest = {
        "total_runs": len(runs),
        "summary": {
            "GENUINE_BACKTRACK": backtrack_count,
            "IMMEDIATE_SOLVE": immediate_count,
            "BLIND_FAILURE": blind_count,
        },
        "runs": results,
    }

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
