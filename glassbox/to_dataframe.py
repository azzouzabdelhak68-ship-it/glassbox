"""Dataframe view over sealed Deep runs (spec section A.12).

Deep-Only Architecture:
Reads ``deep_index.json`` + ``deep.bin`` (via memory-mapped slice only, never
loading the full file).
- ``what="deep"`` (default): returns long-form ``tokens x watched`` rows with
  normative columns ``[run_id, token_pos, token, logit, address, value, watched, seal_ok]``.
- ``what="full"``: returns ``tokens x watched`` rows with run metadata columns
  (``model_id, checkpoint_sha256, prompt_text, output_text, seed``) broadcast
  onto every activation row for multi-run cohort analysis.
- Real ``logit`` column population: when ``LOGITS`` is watched in the run or
  ``logits_summary`` is recorded in ``deep_index.json``, the ``logit`` column
  carries the token's top-1 logit value.
- Partial-precise: unwatched addresses are omitted unless explicitly requested
  (then ``watched=False, value=NaN``); values are float32 logical. Broken seal
  sets ``seal_ok=False`` with a ``SealBrokenWarning``.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

DEEP_COLUMNS: list[str] = [
    "run_id",
    "token_pos",
    "token",
    "logit",
    "address",
    "value",
    "watched",
    "seal_ok",
]

FULL_COLUMNS: list[str] = [
    *DEEP_COLUMNS,
    "model_id",
    "checkpoint_sha256",
    "prompt_text",
    "output_text",
    "seed",
]


class SealBrokenWarning(UserWarning):
    """Seal failed verification; rows still returned with ``seal_ok=False``."""


def _store_root(store: str | None = None) -> str:
    if store:
        return store
    if "GLASSBOX_DIR" in os.environ:
        return os.environ["GLASSBOX_DIR"]
    if "BLACKBOX_DIR" in os.environ:
        return os.environ["BLACKBOX_DIR"]
    if os.path.exists("./blackbox_store") and not os.path.exists("./glassbox_store"):
        return "./blackbox_store"
    return "./glassbox_store"


def _verify_seal_ok(run_id: str, store: str) -> bool:
    """Seal verification check; ``False`` on any problem (never raises)."""
    try:
        from glassbox.storage.seal import verify_run

        return bool(verify_run(run_id, store))
    except Exception:
        return False


def _find_manifest(model_id: str, sha: str) -> dict[str, Any] | None:
    """Best-effort manifest lookup for address validation (None if absent)."""
    import glob

    for path in glob.glob(os.path.join("manifests", "*", "manifest.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                man = json.load(fh)
            if man.get("model_id") == model_id and man.get("checkpoint_sha256") == sha:
                return man
        except (OSError, ValueError):
            continue
    return None


def _validate_against_manifest(requested: list[str], model_id: str, sha: str) -> None:
    """Validate requested addresses vs manifest when one is on disk (E04)."""
    man = _find_manifest(model_id, sha)
    if man is None:
        return
    try:
        from glassbox.watchlist import validate_addresses
    except ImportError:
        return
    validate_addresses(requested, man)


def _frame(rows: Any, include_meta: bool = False) -> Any:
    """Build the pandas frame (or list-of-dicts fallback) with normative dtypes."""
    if not isinstance(rows, list):
        return rows
    cols = FULL_COLUMNS if include_meta else DEEP_COLUMNS
    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=cols)
        dtypes: dict[str, str] = {
            "run_id": "string",
            "token_pos": "int32",
            "token": "string",
            "logit": "float32",
            "address": "string",
            "value": "float32",
            "watched": "bool",
            "seal_ok": "bool",
        }
        if include_meta:
            dtypes.update(
                {
                    "model_id": "string",
                    "checkpoint_sha256": "string",
                    "prompt_text": "string",
                    "output_text": "string",
                    "seed": "int32",
                }
            )
        return df.astype(dtypes)
    except ImportError:
        return rows


def _index_texts(index: dict[str, Any]) -> tuple[list[str], str, str]:
    """Extract token strings, prompt text, and output text from ``deep_index.json``."""
    output = index.get("output", {})
    ids = output.get("tokens", []) if isinstance(output, dict) else []
    texts = [str(t) for t in ids] if isinstance(ids, list) else []
    prompt = index.get("prompt", {})
    prompt_text = str(prompt.get("text", "")) if isinstance(prompt, dict) else ""
    output_text = str(output.get("text", "")) if isinstance(output, dict) else ""
    return texts, prompt_text, output_text


def _token_logits(index: dict[str, Any], run_path: str, lo: int, hi: int) -> dict[int, float]:
    """Extract per-token top-1 logit from watched ``LOGITS`` column or ``logits_summary``."""
    from glassbox.storage.layout import read_deep_slice

    out: dict[int, float] = {}
    watched: list[str] = list(index.get("watched", []))
    if "LOGITS" in watched and index.get("triggered"):
        try:
            col_idx = watched.index("LOGITS")
            mat = read_deep_slice(run_path, lo, hi, [col_idx])
            for row_i, pos in enumerate(range(lo, hi + 1)):
                out[pos] = float(mat[row_i, 0])
            return out
        except Exception:
            pass
    summary = index.get("logits_summary")
    if isinstance(summary, list):
        for pos in range(lo, hi + 1):
            if 0 <= pos < len(summary) and isinstance(summary[pos], dict):
                top_logits = summary[pos].get("top_logits")
                if isinstance(top_logits, list) and top_logits:
                    out[pos] = float(top_logits[0])
    return out


def _deep_rows(
    run_id: str,
    run_path: str,
    index: dict[str, Any],
    tokens: tuple[int, int] | None,
    addresses: list[str] | None,
    seal_ok: bool,
    include_meta: bool,
) -> list[dict[str, Any]]:
    """Long-form tokens x watched rows from the Cold slice (never full file)."""
    from glassbox.storage.layout import read_deep_slice

    if not index.get("triggered"):
        raise ValueError(f"E02: no Deep for run {run_id}: triggered=false")
    watched: list[str] = list(index.get("watched", []))
    total_tokens = int(index.get("tokens", 0))
    lo, hi = (0, total_tokens - 1) if tokens is None else (tokens[0], tokens[1])
    if lo < 0 or hi < lo or hi >= total_tokens:
        raise ValueError(f"E02: bad tokens slice {(lo, hi)} for T={total_tokens}")
    model_id = str(index.get("model_id", "M-001"))
    ckpt_sha = str(index.get("checkpoint_sha256", ""))
    if addresses is None:
        cols = list(range(len(watched)))
        wanted = list(watched)
    else:
        _validate_against_manifest(addresses, model_id, ckpt_sha)
        wanted = list(addresses)
        cols = [watched.index(a) if a in watched else -1 for a in wanted]
    present = [c for c in cols if c >= 0]
    values = read_deep_slice(run_path, lo, hi, present if present else [])
    col_of = {c: k for k, c in enumerate(present)}
    texts, prompt_text, output_text = _index_texts(index)
    logits_map = _token_logits(index, run_path, lo, hi)
    replay = index.get("replay", {})
    seed = int(replay.get("seed", 42)) if isinstance(replay, dict) else 42

    try:
        import numpy as np
        import pandas as pd

        n_tokens = hi - lo + 1
        n_addrs = len(wanted)
        total = n_tokens * n_addrs

        pos_arr = np.repeat(np.arange(lo, hi + 1, dtype=np.int32), n_addrs)
        addr_arr = np.tile(np.array(wanted, dtype=object), n_tokens)
        watched_mask = np.tile(np.array([c >= 0 for c in cols], dtype=bool), n_tokens)

        flat_vals = np.full(total, np.nan, dtype=np.float32)
        for j, c in enumerate(cols):
            if c >= 0 and c in col_of:
                flat_vals[j::n_addrs] = values[:, col_of[c]].astype(np.float32)

        text_base = [texts[p] if p < len(texts) else prompt_text for p in range(lo, hi + 1)]
        text_arr = np.repeat(np.array(text_base, dtype=object), n_addrs)

        logit_base = [logits_map.get(p, float("nan")) for p in range(lo, hi + 1)]
        logit_arr = np.repeat(np.array(logit_base, dtype=np.float32), n_addrs)
        if "LOGITS" in wanted:
            l_idx = wanted.index("LOGITS")
            if cols[l_idx] >= 0:
                logit_arr[l_idx::n_addrs] = flat_vals[l_idx::n_addrs]

        data_dict: dict[str, Any] = {
            "run_id": np.full(total, run_id, dtype=object),
            "token_pos": pos_arr,
            "token": text_arr,
            "logit": logit_arr,
            "address": addr_arr,
            "value": flat_vals,
            "watched": watched_mask,
            "seal_ok": np.full(total, seal_ok, dtype=bool),
        }
        if include_meta:
            data_dict.update(
                {
                    "model_id": np.full(total, model_id, dtype=object),
                    "checkpoint_sha256": np.full(total, ckpt_sha, dtype=object),
                    "prompt_text": np.full(total, prompt_text, dtype=object),
                    "output_text": np.full(total, output_text, dtype=object),
                    "seed": np.full(total, seed, dtype=np.int32),
                }
            )
        cols_order = FULL_COLUMNS if include_meta else DEEP_COLUMNS
        df = pd.DataFrame(data_dict, columns=cols_order)
        dtypes: dict[str, str] = {
            "run_id": "string",
            "token_pos": "int32",
            "token": "string",
            "logit": "float32",
            "address": "string",
            "value": "float32",
            "watched": "bool",
            "seal_ok": "bool",
        }
        if include_meta:
            dtypes.update(
                {
                    "model_id": "string",
                    "checkpoint_sha256": "string",
                    "prompt_text": "string",
                    "output_text": "string",
                    "seed": "int32",
                }
            )
        return df.astype(dtypes)
    except ImportError:
        pass

    rows: list[dict[str, Any]] = []
    for row_i, pos in enumerate(range(lo, hi + 1)):
        text = texts[pos] if pos < len(texts) else prompt_text
        tok_logit = logits_map.get(pos, float("nan"))
        for addr, col in zip(wanted, cols):
            val = float("nan") if col < 0 else float(values[row_i, col_of[col]])
            row: dict[str, Any] = {
                "run_id": run_id,
                "token_pos": pos,
                "token": text,
                "logit": val if addr == "LOGITS" and col >= 0 else tok_logit,
                "address": addr,
                "value": val,
                "watched": col >= 0,
                "seal_ok": seal_ok,
            }
            if include_meta:
                row.update(
                    {
                        "model_id": model_id,
                        "checkpoint_sha256": ckpt_sha,
                        "prompt_text": prompt_text,
                        "output_text": output_text,
                        "seed": seed,
                    }
                )
            rows.append(row)
    return rows


def to_dataframe(
    run_id: str,
    what: str = "deep",
    tokens: tuple[int, int] | None = None,
    addresses: list[str] | None = None,
    store: str | None = None,
) -> Any:
    """Frame over a sealed Deep run (spec §API-dataframe).

    ``what="deep"`` returns long-form ``tokens x watched`` rows; ``what="full"``
    appends broadcast run metadata columns (``model_id, checkpoint_sha256,
    prompt_text, output_text, seed``). Broken seal warns (``SealBrokenWarning``)
    and sets ``seal_ok=False`` — rows are never silently dropped.

    Raises:
        ValueError: bad ``what`` or bad ``tokens`` slice or ``triggered=false``.
        RunNotFoundError: when the run directory is missing.
    """
    if what not in ("deep", "full"):
        raise ValueError(f"E02: unknown what={what!r}; expected 'deep'|'full'")
    root = _store_root(store)
    run_path = os.path.join(root, "runs", run_id)
    index_path = os.path.join(run_path, "deep_index.json")
    if not os.path.isfile(index_path):
        from glassbox.exceptions import RunNotFoundError

        raise RunNotFoundError(f"E06: run {run_id} not found in {root}/runs/.")
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    seal_ok = _verify_seal_ok(run_id, root)
    if not seal_ok:
        warnings.warn(f"seal broken for {run_id}; seal_ok=False", SealBrokenWarning, stacklevel=2)
    include_meta = what == "full"
    rows = _deep_rows(run_id, run_path, index, tokens, addresses, bool(seal_ok), include_meta)
    return _frame(rows, include_meta=include_meta)
