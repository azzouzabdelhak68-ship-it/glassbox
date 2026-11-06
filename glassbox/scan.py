"""Scan weights on CPU (or live PyTorch module) into a Deep manifest (spec §A.1, §FILE-manifest).

Deep-Only Architecture:
Reads ``config.json`` + ``*.safetensors`` header on CPU (or inspects an in-memory
PyTorch ``nn.Module``) to build the structural neuron/head/layer inventory and
checkpoint SHA-256 required by Deep watchlists.

Exit mapping for CLI: E02 ``ValueError``/``FileNotFoundError`` (missing
config), E03 ``ManifestExistsError`` (same sha exists, needs ``--force``).
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
import time
from typing import Any

from glassbox.exceptions import ManifestExistsError

MODEL_ID_DEFAULT = "M-001"


def _read_safetensors_header(path: str) -> bytes:
    """Return raw header JSON bytes of a safetensors file (CPU read only)."""
    with open(path, "rb") as fh:
        prefix = fh.read(8)
        if len(prefix) < 8:
            raise ValueError(f"E02: truncated safetensors header in {path}")
        (header_len,) = struct.unpack("<Q", prefix)
        if header_len > 64 * 1024 * 1024:
            raise ValueError(f"E02: safetensors header too large in {path}")
        header = fh.read(int(header_len))
        if len(header) < int(header_len):
            raise ValueError(f"E02: truncated safetensors header in {path}")
    json.loads(header.decode("utf-8"))  # validate JSON, keep raw bytes
    return header


def _arch_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Map HF config keys to manifest architecture fields (with GQA & head_dim)."""
    layers = cfg.get("num_hidden_layers", cfg.get("n_layer", cfg.get("num_layers")))
    hidden = cfg.get("hidden_size", cfg.get("n_embd", cfg.get("d_model")))
    heads = cfg.get("num_attention_heads", cfg.get("n_head", cfg.get("num_heads")))
    mlp = cfg.get("intermediate_size", cfg.get("n_inner", cfg.get("ffn_dim")))
    if mlp is None and "n_inner" in cfg and hidden is not None:
        mlp = int(hidden) * 4  # GPT-2 standard when n_inner is explicit null
    vocab = cfg.get("vocab_size")
    family = str(cfg.get("model_type", "unknown"))
    vals: dict[str, Any] = {
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "mlp": mlp,
        "vocab": vocab,
    }
    missing = [k for k, v in vals.items() if v is None]
    if missing:
        raise ValueError(f"E02: config.json missing keys for {missing}")
    layers_i = int(str(vals["layers"]))
    hidden_i = int(str(vals["hidden"]))
    heads_i = int(str(vals["heads"]))
    mlp_i = int(str(vals["mlp"]))
    vocab_i = int(str(vals["vocab"]))
    if min(layers_i, hidden_i, heads_i, mlp_i, vocab_i) < 1:
        raise ValueError("E02: architecture dimensions in config.json must be >= 1")
    kv_heads = int(cfg.get("num_key_value_heads", heads_i) or heads_i)
    head_dim = int(cfg.get("head_dim", max(1, hidden_i // heads_i)))
    return {
        "family": family,
        "layers": layers_i,
        "hidden": hidden_i,
        "heads": heads_i,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "mlp_per_layer": mlp_i,
        "vocab": vocab_i,
    }


def _write_manifest(
    manifest: dict[str, Any],
    output: str | None,
    force: bool,
    t0: float,
) -> dict[str, Any]:
    model_id = str(manifest["model_id"])
    checkpoint_sha256 = str(manifest["checkpoint_sha256"])
    if output is None:
        out_path = os.path.join("manifests", f"{model_id}_{checkpoint_sha256[:8]}", "manifest.json")
    elif output.endswith(".json"):
        out_path = output
    else:
        out_path = os.path.join(output, "manifest.json")
    if os.path.isfile(out_path) and not force:
        try:
            with open(out_path, encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("checkpoint_sha256") == checkpoint_sha256:
                raise ManifestExistsError(
                    f"E03: manifest exists sha256:{checkpoint_sha256}. Use --force."
                )
        except ManifestExistsError:
            raise
        except (OSError, ValueError, UnicodeDecodeError):
            raise ManifestExistsError(f"E03: manifest exists at {out_path}. Use --force.") from None
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    manifest["scan"]["duration_s"] = round(time.monotonic() - t0, 2)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return manifest


def scan(
    path: str,
    output: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Scan a local weights directory on CPU and write ``manifest.json``.

    Args:
        path: weights dir containing ``config.json`` (+ optional ``*.safetensors``).
        output: manifest file or dir; default ``manifests/<MODEL>_<SHA8>/manifest.json``.
        force: overwrite same-sha manifest.

    Returns:
        Manifest dict (also written to disk).

    Raises:
        FileNotFoundError: E02 missing ``config.json``.
        ManifestExistsError: E03 same-sha manifest exists without ``--force``.
        ValueError: E02 bad config/header JSON.
    """
    t0 = time.monotonic()
    weights_dir = path
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(
            f"E02: no config.json in {weights_dir}. Deep mode requires local model weights."
        )
    cfg_path = os.path.join(weights_dir, "config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(
            f"E02: no config.json in {weights_dir}. Deep mode requires local model weights."
        )
    with open(cfg_path, "rb") as fh:
        config_bytes = fh.read()
    try:
        cfg = json.loads(config_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"E02: bad config.json in {weights_dir}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"E02: config.json in {weights_dir} must be a JSON object")
    arch = _arch_from_config(cfg)

    st_files = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
    header_bytes = b""
    if st_files:
        header_bytes = _read_safetensors_header(st_files[0])
    checkpoint_sha256 = hashlib.sha256(config_bytes + header_bytes).hexdigest()

    model_name = os.path.basename(os.path.normpath(weights_dir))
    model_id = str(cfg.get("model_id", MODEL_ID_DEFAULT))
    layers, heads, mlp = arch["layers"], arch["heads"], arch["mlp_per_layer"]
    manifest: dict[str, Any] = {
        "model_id": model_id,
        "model_name": model_name,
        "checkpoint_sha256": checkpoint_sha256,
        "mode": "deep",
        "architecture": arch,
        "inventory": {
            "layers": [f"L-{i:02d}" for i in (1, layers)] if layers > 1 else ["L-01"],
            "heads": [f"A-{i:02d}-01..{heads:02d}" for i in (1, layers)],
            "neurons_per_layer": mlp,
            "embed": "EMBED",
            "logits": "LOGITS",
            "attention": f"A-01-01..A-{layers:02d}-{heads:02d}",
            "neurons": f"N-01-0001..N-{layers:02d}-{mlp:04d}",
            "specials": ["EMBED", "LOGITS", "NORM"],
        },
        "runtime": {"framework": "PyTorch (cpu scan)", "cuda": "none"},
        "hardware": {"gpu": "none", "detected_via": "cpu-scan"},
        "adapter": {"family": arch["family"], "layout": "row_major_float16"},
        "scan": {
            "source": "config.json + weights header",
            "gpu_used": False,
            "duration_s": round(time.monotonic() - t0, 2),
        },
    }
    return _write_manifest(manifest, output, force, t0)
