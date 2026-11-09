"""Contrastive Direct Logit Attribution (cDLA) & SVD Extraction Engine (Cell 3B Component).

Implements rigorous PyTorch/SciPy tensor mathematics for Mechanistic Circuit Attribution:
- Slices GQA Query attention heads across candidate layers from tapped `o_proj:in` activations.
- Compares pre-pivot window [t_pivot-3, t_pivot-1] against matched baseline window.
- Computes paired statistical significance across runs with Benjamini-Hochberg FDR correction.
- Ranks causal candidate heads and extracts mean vectors for Phase 2 Mean-Ablation.
- Performs SVD on the pivot activation subspace to extract 5,120-dim PC1 steering vector u_eureka.
- Emits `eureka_manifest.json` and `u_eureka.pt`.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import scipy.stats as stats
import torch


def benjamini_hochberg(p_values: np.ndarray | list[float], alpha: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Perform Benjamini-Hochberg False Discovery Rate (FDR) control.
    
    Returns:
        (significant_mask, q_values)
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # Calculate BH adjusted p-values (q-values)
    factors = m / (np.arange(1, m + 1, dtype=np.float64))
    q_raw = sorted_p * factors
    # Enforce monotonicity: q[i] = min_{j >= i}(q[j])
    q_monotonic = np.minimum.accumulate(q_raw[::-1])[::-1]
    q_monotonic = np.clip(q_monotonic, 0.0, 1.0)

    # Re-order to original indices
    q_values = np.empty_like(q_monotonic)
    q_values[sorted_indices] = q_monotonic
    significant = q_values <= alpha

    return significant, q_values


def compute_paired_effect_sizes(
    pivot_surges: dict[tuple[int, int], list[float]],
    baseline_surges: dict[tuple[int, int], list[float]],
    alpha: float = 0.01,
) -> list[dict[str, Any]]:
    """Compute paired statistical tests and Cohen's d effect sizes for all heads."""
    head_keys = list(pivot_surges.keys())
    raw_p_values: list[float] = []
    effect_sizes: list[float] = []
    mean_diffs: list[float] = []

    for key in head_keys:
        p_vals = np.array(pivot_surges[key], dtype=np.float64)
        b_vals = np.array(baseline_surges.get(key, [0.0] * len(p_vals)), dtype=np.float64)

        diff = p_vals - b_vals
        if len(diff) == 0:
            mean_d = 0.0
            d = 0.0
            pval = 1.0
        else:
            mean_d = float(np.nanmean(diff))
            if np.isnan(mean_d):
                mean_d = 0.0
            std_d = float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else 1.0
            if np.isnan(std_d) or std_d < 1e-4:
                std_d = 1e-4
            d = mean_d / std_d

            # Paired t-test
            if len(diff) >= 2 and np.any(diff != 0):
                try:
                    res = stats.ttest_rel(p_vals, b_vals)
                    pval = float(res.pvalue) if not np.isnan(res.pvalue) else 1.0
                except Exception:
                    pval = 1.0
            else:
                pval = 1.0

        raw_p_values.append(pval)
        effect_sizes.append(d)
        mean_diffs.append(mean_d)

    is_sig, q_values = benjamini_hochberg(raw_p_values, alpha=alpha)

    results: list[dict[str, Any]] = []
    for idx, (layer, head) in enumerate(head_keys):
        results.append({
            "layer": layer,
            "head": head,
            "target_str": f"A-{layer:02d}-{head:02d}",
            "cohens_d": float(effect_sizes[idx]),
            "mean_surge": float(mean_diffs[idx]),
            "p_value": float(raw_p_values[idx]),
            "q_value": float(q_values[idx]),
            "significant_fdr": bool(is_sig[idx]),
        })

    # Sort descending by effect size
    results.sort(key=lambda item: item["cohens_d"], reverse=True)
    return results


def extract_svd_pc1(
    activation_matrix: np.ndarray | torch.Tensor,
) -> tuple[torch.Tensor, float]:
    """Extract normalized PC1 steering vector and explained variance ratio via SVD.
    
    Args:
        activation_matrix: [N_samples, d_model] float matrix of pivot token activations.
    
    Returns:
        (u_eureka: torch.Tensor of shape [d_model], explained_variance_ratio: float)
    """
    if isinstance(activation_matrix, np.ndarray):
        X = torch.from_numpy(activation_matrix).to(torch.float32)
    else:
        X = activation_matrix.to(torch.float32)

    if X.ndim != 2 or X.shape[0] < 1:
        raise ValueError(f"Activation matrix must be 2D [N, d_model], got shape {X.shape}")

    # Center matrix
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean

    # Perform SVD: X = U S V^T
    # V has shape [d_model, min(N, d_model)]
    try:
        _, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        v1 = Vh[0]  # First right singular vector (PC1)
        var_total = torch.sum(S ** 2).item()
        var_pc1 = (S[0] ** 2).item()
        exp_ratio = (var_pc1 / var_total) if var_total > 1e-12 else 1.0
    except Exception:
        # Fallback to mean vector if SVD fails (e.g. zero matrix)
        v1 = X_mean.squeeze(0)
        exp_ratio = 1.0

    # Ensure unit norm
    norm = torch.linalg.norm(v1)
    if norm > 1e-8:
        u_eureka = v1 / norm
    else:
        u_eureka = v1

    return u_eureka, float(exp_ratio)


class CDLASvdEngine:
    """End-to-end attribution engine coordinating cDLA and SVD across cohort runs."""

    def __init__(
        self,
        d_model: int = 5120,
        num_heads: int = 40,
        head_dim: int = 128,
        layer_range: tuple[int, int] = (35, 50),
    ) -> None:
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.layer_range = layer_range
        assert num_heads * head_dim == d_model, f"{num_heads} * {head_dim} != {d_model}"

    def analyze_traces(
        self,
        semantic_pivots: dict[str, Any],
        run_activations: dict[str, dict[int, np.ndarray]],
        top_k: int = 4,
        alpha: float = 0.01,
    ) -> tuple[dict[str, Any], torch.Tensor]:
        """Analyze runs to find causal heads and PC1 steering vector.
        
        Args:
            semantic_pivots: Output dictionary from Analyzer Auditor (`semantic_pivots.json`).
            run_activations: Mapping from run_id to {layer_idx: activation_matrix [T, d_model]}.
            top_k: Number of candidate attention heads to select.
            alpha: FDR threshold.
            
        Returns:
            (manifest_dict, u_eureka_tensor)
        """
        runs_list = semantic_pivots.get("runs", [])
        pivots_map = {r["run_id"]: r for r in runs_list}

        layers = list(range(self.layer_range[0], self.layer_range[1] + 1))
        pivot_surges: dict[tuple[int, int], list[float]] = {}
        baseline_surges: dict[tuple[int, int], list[float]] = {}
        clean_head_vectors: dict[tuple[int, int], list[np.ndarray]] = {}

        for l in layers:
            for h in range(1, self.num_heads + 1):
                pivot_surges[(l, h)] = []
                baseline_surges[(l, h)] = []
                clean_head_vectors[(l, h)] = []

        pivot_activation_slices: list[np.ndarray] = []
        primary_svd_layer = layers[len(layers) // 2]  # Default to mid-candidate layer

        for run_id, layer_acts in run_activations.items():
            meta = pivots_map.get(run_id)
            if not meta:
                continue

            traj = meta.get("trajectory_type")
            is_clean = (meta.get("version") == "A") or (traj == "IMMEDIATE_SOLVE")

            # Collect clean activations for computing true baseline mean vector
            if is_clean:
                for l in layers:
                    if l not in layer_acts:
                        continue
                    act = layer_acts[l]  # [T, d_model]
                    for h in range(1, self.num_heads + 1):
                        h_start = (h - 1) * self.head_dim
                        h_end = h * self.head_dim
                        clean_head_vectors[(l, h)].append(act[:, h_start:h_end])
                continue

            if traj != "GENUINE_BACKTRACK":
                continue

            t_pivot = meta.get("t_pivot")
            if t_pivot is None:
                continue

            # Contrastive window: [t_pivot-3, t_pivot-1] (decision window)
            # Baseline window: [t_commit-2, t_commit+2] or earlier pre-pivot tokens
            t_commit = meta.get("t_commit") or max(2, t_pivot // 2)

            for l in layers:
                if l not in layer_acts:
                    continue
                act = layer_acts[l]
                T = act.shape[0]

                p_start = max(0, t_pivot - 3)
                p_end = max(1, min(T, t_pivot))
                b_start = max(0, t_commit - 2)
                b_end = max(1, min(T, t_commit + 2))

                p_slice = act[p_start:p_end]
                b_slice = act[b_start:b_end]

                if l == primary_svd_layer:
                    # Collect pivot activations for SVD
                    pivot_activation_slices.append(p_slice)

                for h in range(1, self.num_heads + 1):
                    h_start = (h - 1) * self.head_dim
                    h_end = h * self.head_dim

                    # Head L2 norm in pivot vs baseline
                    p_norm = float(np.mean(np.linalg.norm(p_slice[:, h_start:h_end], axis=-1)))
                    b_norm = float(np.mean(np.linalg.norm(b_slice[:, h_start:h_end], axis=-1)))

                    pivot_surges[(l, h)].append(p_norm)
                    baseline_surges[(l, h)].append(b_norm)

        # Statistical ranking
        stats_ranked = compute_paired_effect_sizes(pivot_surges, baseline_surges, alpha=alpha)

        # Select top-k candidate heads
        selected_candidates = stats_ranked[:top_k]

        # Calculate mean-ablation vectors for each candidate head
        top_heads_manifest: list[dict[str, Any]] = []
        for cand in selected_candidates:
            l = cand["layer"]
            h = cand["head"]
            clean_slices = clean_head_vectors.get((l, h), [])
            if clean_slices:
                cat_clean = np.concatenate(clean_slices, axis=0)
                mean_vec = np.mean(cat_clean, axis=0).astype(np.float32).tolist()
            else:
                mean_vec = [0.0] * self.head_dim

            cand_entry = dict(cand)
            cand_entry["mean_vector"] = mean_vec
            top_heads_manifest.append(cand_entry)

        if top_heads_manifest:
            # Anchor SVD layer to top candidate head's layer
            primary_svd_layer = top_heads_manifest[0]["layer"]

        # Compute SVD PC1 Steering Vector
        if pivot_activation_slices:
            X_svd = np.concatenate(pivot_activation_slices, axis=0)
        else:
            X_svd = np.zeros((1, self.d_model), dtype=np.float32)

        u_eureka, exp_ratio = extract_svd_pc1(X_svd)

        manifest = {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "layer_range": list(self.layer_range),
            "primary_svd_layer": primary_svd_layer,
            "explained_variance_ratio": exp_ratio,
            "top_candidate_heads": top_heads_manifest,
            "all_ranked_heads": stats_ranked,
        }

        return manifest, u_eureka

    def save_artifacts(
        self,
        manifest: dict[str, Any],
        u_eureka: torch.Tensor,
        manifest_path: str = "eureka_manifest.json",
        vector_path: str = "u_eureka.pt",
    ) -> None:
        """Write manifest and steering tensor to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        os.makedirs(os.path.dirname(os.path.abspath(vector_path)), exist_ok=True)
        torch.save(u_eureka, vector_path)
