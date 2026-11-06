"""Hot index shards with per-address Deep aggregates (spec §SEARCH-tiers, plan §10.1).

Tier 1 Hot lives in ``$BLACKBOX_DIR/index/shard-*.jsonl`` (one JSON object
per sealed Deep run) plus a per-run copy at ``runs/<RUN>/meta.json``. Each entry
carries conservative aggregates computed from ``deep.bin`` + ``deep_index.json``
so the Hot filter never drops a true positive; Cold exact values always decide:

- ``min/max/mean/std/p95/l2`` over all watched tokens of the run,
- ``firing_rate_0.8`` = fraction of finite values ``> 0.8``,
- ``win10`` = maxima of non-overlapping 10-token chunks computed on true token
  indices (before dropping NaNs, so token-window pre-filters never shift index
  boundaries; capped at the first 128 chunks; ``n_win10`` records the total).
"""

from __future__ import annotations

import concurrent.futures
import json
import math
import os
import re
from collections.abc import Callable
from typing import Any

import numpy as np

from glassbox.storage.layout import decode_deep_array

SHARD_SIZE = 10000
WIN = 10
WIN_CAP = 128
FIRING_THRESHOLD = 0.8
_SHARD_RE = re.compile(r"^shard-(\d{4})\.jsonl$")


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


def _index_dir(store: str) -> str:
    path = os.path.join(store, "index")
    os.makedirs(path, exist_ok=True)
    return path


def _shard_for(path: str, run_id: str) -> str:
    """Current open shard in O(1) directory check; rolls every SHARD_SIZE lines."""
    _ = run_id
    existing: list[tuple[int, str]] = []
    if os.path.isdir(path):
        for name in os.listdir(path):
            m = _SHARD_RE.match(name)
            if m:
                existing.append((int(m.group(1)), os.path.join(path, name)))
    if not existing:
        return os.path.join(path, "shard-0000.jsonl")
    existing.sort(key=lambda item: item[0])
    last_idx, last_path = existing[-1]
    try:
        with open(last_path, encoding="utf-8") as fh:
            n = sum(1 for _ in fh)
    except OSError:
        return last_path
    if n < SHARD_SIZE:
        return last_path
    return os.path.join(path, f"shard-{last_idx + 1:04d}.jsonl")


def _load_matrix(run_dir: str, dtype: str, tokens: int, nwatched: int) -> np.ndarray:
    """Memory-map ``deep.bin`` row-major ``[T, W]`` and decode to float32."""
    raw_dt: Any = {"fp32": np.dtype("<f4"), "fp16": np.dtype("<f2"), "bf16": np.dtype("<u2")}[dtype]
    deep_path = os.path.join(run_dir, "deep.bin")
    size = os.path.getsize(deep_path)
    if size != tokens * nwatched * int(raw_dt.itemsize):
        raise ValueError(f"E02: deep.bin size {size} != [{tokens}x{nwatched}] {dtype}")
    mapped = np.memmap(deep_path, dtype=raw_dt, mode="r", shape=(tokens, nwatched))
    return decode_deep_array(mapped, dtype)


def _addr_stats(col: Any) -> dict[str, Any]:
    """Conservative aggregates for one watched column (preserving true token window indices)."""
    raw_vals = np.asarray(col, dtype=np.float64)
    finite = raw_vals[np.isfinite(raw_vals)]
    if finite.size == 0:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p95": None,
            "l2": None,
            "firing_rate_0.8": 0.0,
            "win10": [],
            "n_win10": 0,
        }
    total_len = int(raw_vals.size)
    chunks = int(math.ceil(total_len / WIN))
    win10: list[float] = []
    for i in range(min(chunks, WIN_CAP)):
        window_slice = raw_vals[i * WIN : (i + 1) * WIN]
        window_finite = window_slice[np.isfinite(window_slice)]
        win10.append(float(window_finite.max()) if window_finite.size > 0 else float(finite.max()))
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p95": float(np.percentile(finite, 95)),
        "l2": float(np.linalg.norm(finite, ord=2)),
        "firing_rate_0.8": float((finite > FIRING_THRESHOLD).mean()),
        "win10": win10,
        "n_win10": chunks,
    }


def build_hot_entry(run_id: str, store: str | None = None) -> dict[str, Any]:
    """Compute the Hot entry for a sealed Deep run (raises E02/E06 on bad runs)."""
    root = _store_root(store)
    run_dir = os.path.join(root, "runs", run_id)
    index_path = os.path.join(run_dir, "deep_index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"E06: {run_id} not found in {root}/runs/.")
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    if not isinstance(index, dict):
        raise ValueError(f"E02: bad deep_index.json for {run_id}")

    wl_ref = index.get("watchlist_ref")
    wl_id = (
        str(wl_ref.get("watchlist_id", ""))
        if isinstance(wl_ref, dict)
        else str(index.get("watchlist_id", ""))
    )
    prompt = index.get("prompt")
    prompt_text = str(prompt.get("text", "")) if isinstance(prompt, dict) else ""

    entry: dict[str, Any] = {
        "run_id": run_id,
        "model_id": str(index.get("model_id", "M-001")),
        "checkpoint_sha256": str(index.get("checkpoint_sha256", "")),
        "watchlist_id": wl_id,
        "tags": list(index.get("tags", []) or []),
        "flags": dict(index.get("flags", {}) or {}),
        "prompt_prefix": prompt_text[:200],
        "has_deep": False,
        "addresses": {},
    }
    if not index.get("triggered"):
        return entry
    watched: list[str] = list(index.get("watched", []))
    tokens = int(index.get("tokens", 0))
    dtype = str(index.get("dtype", "fp16"))
    if not watched or tokens < 1:
        return entry
    try:
        mat = _load_matrix(run_dir, dtype, tokens, len(watched))
    except (OSError, ValueError, KeyError):
        return entry
    entry["has_deep"] = True
    entry["dtype"] = dtype
    entry["tokens"] = tokens
    entry["addresses"] = {addr: _addr_stats(mat[:, j]) for j, addr in enumerate(watched)}
    return entry


def append_hot(run_id: str, store: str | None = None) -> dict[str, Any]:
    """Build + persist the Hot entry (shard line + ``meta.json``); idempotent."""
    root = _store_root(store)
    entry = build_hot_entry(run_id, root)
    path = _index_dir(root)
    shard = _shard_for(path, run_id)
    with open(shard, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    meta_path = os.path.join(root, "runs", run_id, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(entry, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return entry


def ensure_hot(run_id: str, store: str | None = None) -> dict[str, Any]:
    """Return the Hot entry, building it lazily when missing (never E06-lossy)."""
    root = _store_root(store)
    meta_path = os.path.join(root, "runs", run_id, "meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                entry = json.load(fh)
            if isinstance(entry, dict) and entry.get("run_id") == run_id:
                return entry
        except (OSError, ValueError):
            pass
    return append_hot(run_id, root)


def iter_hot_entries(store: str | None = None) -> list[dict[str, Any]]:
    """All Hot entries: shard lines first, then runs missing from shards."""
    root = _store_root(store)
    entries: dict[str, dict[str, Any]] = {}
    path = os.path.join(root, "index")
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if not name.startswith("shard-") or not name.endswith(".jsonl"):
                continue
            try:
                with open(os.path.join(path, name), encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(obj, dict) and obj.get("run_id"):
                            entries[str(obj["run_id"])] = obj
            except OSError:
                continue
    runs_dir = os.path.join(root, "runs")
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            if re.fullmatch(r"RUN-\d{6}", name) and name not in entries:
                try:
                    entries[name] = ensure_hot(name, root)
                except (OSError, ValueError):
                    continue
    return [entries[k] for k in sorted(entries)]


def hot_scan(
    predicate: Callable[[dict[str, Any]], bool],
    store: str | None = None,
    workers: int = 4,
) -> list[dict[str, Any]]:
    """Parallel Hot filter; predicate sees entries only (no Cold I/O)."""
    entries = iter_hot_entries(store)
    if workers < 2 or len(entries) < 2:
        return [e for e in entries if predicate(e)]
    out: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for entry, keep in zip(entries, pool.map(predicate, entries)):
            if keep:
                out.append(entry)
    return out


def cold_verify_one(run_id: str, check: Callable[[], bool], store: str | None = None) -> bool:
    """Run one Cold exact check; any exception counts as non-match (never raise)."""
    _ = (run_id, store)
    try:
        return bool(check())
    except Exception:
        return False


def reindex(store: str | None = None) -> int:
    """Rebuild shards from all sealed runs; returns entry count."""
    root = _store_root(store)
    path = _index_dir(root)
    for name in os.listdir(path):
        if name.startswith("shard-") and name.endswith(".jsonl"):
            try:
                os.remove(os.path.join(path, name))
            except OSError:
                pass
    count = 0
    runs_dir = os.path.join(root, "runs")
    if os.path.isdir(runs_dir):
        for name in sorted(os.listdir(runs_dir)):
            if re.fullmatch(r"RUN-\d{6}", name):
                try:
                    append_hot(name, root)
                    count += 1
                except (OSError, ValueError):
                    continue
    return count
