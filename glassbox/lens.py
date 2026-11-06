"""Mechanistic Logit Lens & Residual Stream Projection Engine.

Analyzes intermediate layer activations (L-01..L-NN), layer norm updates, and
token prediction evolution across the model's transformer depth.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any

from glassbox.storage.layout import get_store_dir, read_deep_index, read_deep_slice

_RE_L = re.compile(r"^L-(\d+)$")


def compute_logit_lens(
    run_id: str,
    store: str | None = None,
    token_pos: int | None = None,
) -> dict[str, Any]:
    """Compute layer-by-layer residual stream progression and logit lens for ``run_id``.

    Returns:
        dict containing:
        - ``run_id``: target run identifier
        - ``layers``: list of detected layer indices
        - ``tokens``: total tokens captured
        - ``selected_token``: inspected token index
        - ``layer_norms``: {layer_idx: float norm}
        - ``layer_deltas``: {layer_idx: float delta from prev layer}
        - ``max_delta_layer``: layer index with the largest information write
        - ``top_logits``: list of top logits per token if available
    """
    root = get_store_dir(store)
    run_path = os.path.join(root, "runs", run_id)
    index = read_deep_index(run_path)

    if not index.get("triggered"):
        raise ValueError(f"E02: run {run_id} was not triggered (no deep activations)")

    watched = list(index.get("watched", []))
    tokens = int(index.get("tokens", 0))
    if tokens < 1:
        raise ValueError(f"E02: run {run_id} contains 0 captured tokens")

    # Filter residual stream layer addresses
    layer_addrs: list[tuple[int, str]] = []
    for a in watched:
        m = _RE_L.match(a)
        if m:
            layer_addrs.append((int(m.group(1)), a))
    layer_addrs.sort(key=lambda t: t[0])

    target_tok = 0 if token_pos is None else max(0, min(token_pos, tokens - 1))

    # Slice residual stream values for the selected token
    layer_norms: dict[int, float] = {}
    if layer_addrs:
        indices = [watched.index(a) for _, a in layer_addrs]
        slice_vals = read_deep_slice(run_path, target_tok, target_tok, indices)[0]
        for (l_idx, _), val in zip(layer_addrs, slice_vals):
            layer_norms[l_idx] = float(val)

    # Compute layer-to-layer delta (information written per layer)
    layer_deltas: dict[int, float] = {}
    sorted_layers = sorted(layer_norms.keys())
    for i, l_idx in enumerate(sorted_layers):
        if i == 0:
            layer_deltas[l_idx] = round(layer_norms[l_idx], 4)
        else:
            prev = sorted_layers[i - 1]
            diff = abs(layer_norms[l_idx] - layer_norms[prev])
            layer_deltas[l_idx] = round(diff, 4)

    max_delta_layer = max(layer_deltas.items(), key=lambda t: t[1])[0] if layer_deltas else None

    logits_summary = index.get("logits_summary") or []
    tok_logits = logits_summary[target_tok] if target_tok < len(logits_summary) else None

    return {
        "run_id": run_id,
        "tokens": tokens,
        "selected_token": target_tok,
        "layers": sorted_layers,
        "layer_norms": layer_norms,
        "layer_deltas": layer_deltas,
        "max_delta_layer": max_delta_layer,
        "logits_summary": tok_logits,
    }


def format_lens_table(lens_data: dict[str, Any]) -> str:
    """Format logit lens residual progression as a clean CLI table."""
    layers = lens_data.get("layers", [])
    if not layers:
        return (
            "Logit Lens: No residual stream addresses (L-01..L-NN) found in watched list.\n"
            "Tip: run with preset 'logit-lens' to record layer-by-layer residual streams."
        )

    lines = [
        f"=== Logit Lens & Residual Progression for {lens_data['run_id']} "
        f"(Token {lens_data['selected_token']}) ===",
        f"{'Layer':<8} {'Norm':<12} {'Delta':<12} {'Information Write'}",
        "-" * 55,
    ]

    norms = lens_data.get("layer_norms", {})
    deltas = lens_data.get("layer_deltas", {})
    max_d = max(deltas.values(), default=1.0) or 1.0

    for l_idx in layers:
        norm_val = norms.get(l_idx, 0.0)
        delta_val = deltas.get(l_idx, 0.0)
        bar_len = int((delta_val / max_d) * 20) if math.isfinite(delta_val) else 0
        bar = "#" * bar_len
        lines.append(f"L-{l_idx:02d}    {norm_val:<12.4f} {delta_val:<12.4f} {bar}")

    if lens_data.get("max_delta_layer") is not None:
        lines.append("-" * 55)
        lines.append(
            f"Peak computation update at Layer L-{lens_data['max_delta_layer']:02d} "
            f"(Delta = {deltas.get(lens_data['max_delta_layer'], 0.0):.4f})"
        )

    summary = lens_data.get("logits_summary")
    if summary and isinstance(summary, dict):
        lines.append(
            f"Output Head Prediction: Top Token ID {summary.get('top_token_ids', ['?'])[0]} "
            f"| Entropy: {summary.get('entropy', '?')}"
        )

    return "\n".join(lines)
