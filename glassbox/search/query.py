"""Query grammar + two-phase Hot filter then Cold verify (spec §SEARCH-grammar).

Supported query (case-insensitive keywords; string literals double-quoted):

- ``FIND RUNS WHERE <cond> AND <cond> ...`` — the leading
  ``FIND RUNS WHERE`` is optional (CLI ``find`` passes the bare condition).
- ``checkpoint="sha256:abc..."`` (``sha256:`` prefix optional).
- ``tag="good"`` — matches a ``k=v`` tag exactly, a bare value against the
  value side of any ``k=v`` tag, or ``tag:k=v`` full form.
- ``prompt CONTAINS "literal"`` — case-sensitive substring on the full prompt.
- ``flag="bad_output" | "ring_overflow" | "hash_fail"``.
- ``ADDR AT token=K OP NUM`` with ``OP`` in ``> >= < <= = == !=``.
- ``ADDR AT token A..B AGG OP NUM`` / ``ADDR AT tokens A..B AGG OP NUM``
  with ``AGG`` in ``MAX MIN MEAN STD L2 P95`` (``max_window`` is accepted as an
  alias of ``MAX`` over the window). A bare ``..`` range without ``AGG``
  defaults to ``MAX``.
- ``ADDR`` ranges like ``N-24-0001..0100`` expand; a range matches when ANY
  address in it satisfies the predicate (documented choice).

Two-phase execution (:func:`find_runs`): Hot scans ``index/`` aggregates in
a thread pool (conservative superset — a run is a candidate unless the Hot
aggregates prove it cannot match), then Cold ``mmap``s only candidate
slices via ``deep_index.json`` offsets and checks the exact predicate
(100-1000x I/O reduction). Every row carries ``hash_seal_verified``.
``COMPARE GROUPS`` is owned by the CLI flags; :func:`compare_runs` runs the
two group passes plus pooled aggregates.
"""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

import numpy as np

from glassbox.search import index as _index

_CMP_OPS = (">=", "<=", "!=", "==", ">", "<", "=")
_AGG_OPS = ("MAX", "MIN", "MEAN", "STD", "L2", "P95", "MAX_WINDOW")
_ADDR_RE = re.compile(
    r"^(N-\d+-\d+|A-\d+-\d+|L-\d+-\d+|L-\d+|SAE-\d+-\d+|F-\d+-\d+|LOGITS:\d+|LOGITS-ENTROPY|LOGITS|EMBED|NORM)$"
)
_RANGE_SUFFIX_RE = re.compile(
    r"^(N-\d+-\d+|A-\d+-\d+|L-\d+-\d+|L-\d+|SAE-\d+-\d+|F-\d+-\d+)\.\.(\d+)$"
)


def _split_and(query: str) -> list[str]:
    return [p.strip() for p in re.split(r"\bAND\b", query, flags=re.IGNORECASE) if p.strip()]


def _strip_find_prefix(query: str) -> str:
    query = query.strip()
    m = re.match(r"(?i)^FIND\s+RUNS\s+WHERE\s+(.*)$", query, flags=re.DOTALL)
    return m.group(1).strip() if m else query


def _expand_range_token(token: str) -> list[str]:
    """Expand ``N-24-0001..0100`` suffix ranges; singles pass through."""
    token = token.strip()
    if _ADDR_RE.match(token):
        return [token]
    m = _RANGE_SUFFIX_RE.match(token)
    if not m:
        raise ValueError(f"E02: bad address {token!r}.")
    base, end_s = m.group(1), m.group(2)
    if base.startswith("N-"):
        lm = re.match(r"^N-(\d+)-(\d+)$", base)
        assert lm is not None
        layer, start = int(lm.group(1)), int(lm.group(2))
        width = max(len(lm.group(2)), len(end_s))
        end = int(end_s)
        if end < start:
            raise ValueError(f"E02: bad address range {token!r}.")
        return [f"N-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    if base.startswith("A-"):
        lm = re.match(r"^A-(\d+)-(\d+)$", base)
        assert lm is not None
        layer, start = int(lm.group(1)), int(lm.group(2))
        width = max(len(lm.group(2)), len(end_s))
        end = int(end_s)
        if end < start:
            raise ValueError(f"E02: bad address range {token!r}.")
        return [f"A-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    if base.startswith("SAE-") or base.startswith("F-"):
        is_sae = base.startswith("SAE-")
        pref = "SAE" if is_sae else "F"
        lm_sae = re.match(r"^(?:SAE|F)-(\d+)-(\d+)$", base)
        assert lm_sae is not None
        layer, start = int(lm_sae.group(1)), int(lm_sae.group(2))
        width = max(len(lm_sae.group(2)), len(end_s))
        end = int(end_s)
        if end < start:
            raise ValueError(f"E02: bad address range {token!r}.")
        return [f"{pref}-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    lm_dim = re.match(r"^L-(\d+)-(\d+)$", base)
    if lm_dim is not None:
        layer, start = int(lm_dim.group(1)), int(lm_dim.group(2))
        width = max(len(lm_dim.group(2)), len(end_s))
        end = int(end_s)
        if end < start:
            raise ValueError(f"E02: bad address range {token!r}.")
        return [f"L-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    lm_l = re.match(r"^L-(\d+)$", base)
    assert lm_l is not None
    start = int(lm_l.group(1))
    width = max(len(lm_l.group(1)), len(end_s))
    end = int(end_s)
    if end < start:
        raise ValueError(f"E02: bad address range {token!r}.")
    return [f"L-{i:0{width}d}" for i in range(start, end + 1)]


def _parse_op(text: str) -> tuple[str, str]:
    for op in _CMP_OPS:
        if text.startswith(op):
            return op, text[len(op) :].strip()
    raise ValueError(f"E02: bad comparator in {text!r}.")


def _parse_neuron_cond(part: str) -> dict[str, Any]:
    """Parse one neuron predicate; raises ``ValueError`` (E02) on bad grammar."""
    m = re.match(r"(?i)^(\S+)\s+AT\s+tokens?\s*=?\s*(.+)$", part.strip(), flags=re.DOTALL)
    if not m:
        raise ValueError(f"E02: bad query grammar near {part!r}.")
    addrs = _expand_range_token(m.group(1))
    rest = m.group(2).strip()
    token: int | None = None
    lo = hi = 0
    m2 = re.match(r"^(\d+)\s*(.*)$", rest, flags=re.DOTALL)
    if not m2:
        raise ValueError(f"E02: bad query grammar near {part!r}.")
    first, after = int(m2.group(1)), m2.group(2).strip()
    m3 = re.match(r"^\.\.\s*(\d+)\s*(.*)$", after, flags=re.DOTALL)
    if m3:
        lo, hi = first, int(m3.group(1))
        after = m3.group(2).strip()
        if hi < lo:
            raise ValueError(f"E02: bad token range in {part!r}.")
    else:
        token = first
        lo = hi = first
    agg = "VALUE"
    m4 = re.match(r"(?i)^(MAX_WINDOW|MAX|MIN|MEAN|STD|L2|P95)\s+(.*)$", after, flags=re.DOTALL)
    if m4:
        agg = m4.group(1).upper()
        after = m4.group(2).strip()
    elif token is None:
        agg = "MAX"
    op, num_s = _parse_op(after)
    try:
        value = float(num_s)
    except ValueError as exc:
        raise ValueError(f"E02: bad number in {part!r}.") from exc
    if agg == "MAX_WINDOW":
        agg = "MAX"
    return {
        "kind": "neuron",
        "addresses": addrs,
        "token": token,
        "lo": lo,
        "hi": hi,
        "agg": agg,
        "op": op,
        "value": value,
    }


def parse_query(query: str) -> dict[str, Any]:
    """Parse a find query to ``{hot: [...], neuron: [...]}`` (E02 on grammar)."""
    body = _strip_find_prefix(query)
    if not body:
        raise ValueError("E02: bad query grammar near ''.")
    hot: list[dict[str, Any]] = []
    neuron: list[dict[str, Any]] = []
    for part in _split_and(body):
        m = re.fullmatch(r'(?i)checkpoint\s*=\s*"([^"]*)"', part)
        if m:
            hot.append({"kind": "checkpoint", "value": m.group(1).replace("sha256:", "")})
            continue
        m = re.fullmatch(r'(?i)tag\s*=\s*"([^"]*)"', part)
        if m:
            hot.append({"kind": "tag", "value": m.group(1)})
            continue
        m = re.fullmatch(r"(?i)tag\s*:\s*([A-Za-z0-9_]+)\s*=\s*(\S+)", part)
        if m:
            hot.append({"kind": "tag", "value": f"{m.group(1)}={m.group(2)}"})
            continue
        m = re.fullmatch(r'(?i)prompt\s+CONTAINS\s+"([^"]*)"', part)
        if m:
            hot.append({"kind": "prompt", "value": m.group(1)})
            continue
        m = re.fullmatch(r'(?i)flag\s*=\s*"(bad_output|ring_overflow|hash_fail)"', part)
        if m:
            hot.append({"kind": "flag", "value": m.group(1).lower()})
            continue
        if re.search(r"(?i)\bAT\b", part):
            neuron.append(_parse_neuron_cond(part))
            continue
        raise ValueError(f"E02: bad query grammar near {part!r}.")
    if not hot and not neuron:
        raise ValueError("E02: bad query grammar near ''.")
    return {"hot": hot, "neuron": neuron}


def _tag_match(tags: list[str], value: str) -> bool:
    if value in tags:
        return True
    for tag in tags:
        if tag == value or tag.split("=", 1)[-1] == value:
            return True
    return False


def _hot_match(entry: dict[str, Any], conds: list[dict[str, Any]], store: str) -> bool:
    for cond in conds:
        kind = cond["kind"]
        if kind == "checkpoint":
            if str(entry.get("checkpoint_sha256", "")) != cond["value"]:
                return False
        elif kind == "tag":
            if not _tag_match([str(t) for t in entry.get("tags", [])], cond["value"]):
                return False
        elif kind == "prompt":
            texts = [str(entry.get("prompt_prefix", ""))]
            if cond["value"] not in "".join(texts):
                # Prefix is 200 chars: confirm against full prompt in deep_index.json
                try:
                    with open(
                        os.path.join(store, "runs", entry["run_id"], "deep_index.json"),
                        encoding="utf-8",
                    ) as fh:
                        full = str((json.load(fh).get("prompt", {}) or {}).get("text", ""))
                    if cond["value"] not in full:
                        return False
                except (OSError, ValueError):
                    return False
        elif kind == "flag":
            if cond["value"] == "hash_fail":
                continue
            if not bool((entry.get("flags", {}) or {}).get(cond["value"], False)):
                return False
    return True


def _hot_neuron_possible(entry: dict[str, Any], pred: dict[str, Any]) -> bool:
    """Conservative Hot pre-filter: False only when aggregates rule it out."""
    addrs = entry.get("addresses", {})
    op, target = pred["op"], pred["value"]
    lo, hi = pred["lo"], pred["hi"]

    def _possible(stats: dict[str, Any]) -> bool:
        if not stats or stats.get("max") is None:
            return True
        if pred["agg"] in ("STD", "L2"):
            return True
        if op in (">", ">="):
            if pred["agg"] in ("VALUE", "MAX"):
                chunks: list[float] = stats.get("win10", []) or []
                n = stats.get("n_win10", 0)
                if chunks and n:
                    span = range(lo // 10, hi // 10 + 1)
                    cands = [chunks[c] for c in span if 0 <= c < len(chunks)]
                    if cands:
                        return max(cands) >= target
                return float(stats["max"]) >= target
            return float(stats["max"]) >= target
        if op in ("<", "<="):
            return float(stats["min"]) <= target
        if op in ("=", "=="):
            return float(stats["min"]) <= target <= float(stats["max"])
        return True

    return any(_possible(addrs.get(a, {})) for a in pred["addresses"])


def _apply_op(op: str, actual: float, target: float) -> bool:
    if op == ">":
        return actual > target
    if op == ">=":
        return actual >= target
    if op == "<":
        return actual < target
    if op == "<=":
        return actual <= target
    if op == "!=":
        return actual != target
    return actual == target


def _cold_neuron_check(
    run_path: str, pred: dict[str, Any], token_lo: int, token_hi: int
) -> tuple[bool, dict[str, Any]]:
    """Exact Cold check; returns (match, {token, value} report point).

    Uses ``valid_idx = np.flatnonzero(np.isfinite(series))`` so that NaN gaps
    from ring_overflow never shift the reported token position!
    """
    from glassbox.storage.layout import read_deep_index, read_deep_slice

    index = read_deep_index(run_path)
    if not index.get("triggered"):
        return False, {}
    watched: list[str] = list(index.get("watched", []))
    total = int(index.get("tokens", 0))
    lo = max(pred["lo"], token_lo)
    hi = min(pred["hi"], token_hi)
    if lo > hi or lo >= total:
        return False, {}
    hi = min(hi, total - 1)
    matched_addrs = [a for a in pred["addresses"] if a in watched]
    cols = [watched.index(a) for a in matched_addrs]
    if not cols:
        return False, {}
    mat = read_deep_slice(run_path, lo, hi, cols)

    best: dict[str, Any] = {}
    matched = False
    for k, _col in enumerate(cols):
        series = mat[:, k].astype(np.float64)
        valid_idx = np.flatnonzero(np.isfinite(series))
        if valid_idx.size == 0:
            continue
        finite = series[valid_idx]
        agg = pred["agg"]
        if pred["token"] is not None:
            if lo > pred["token"] or hi < pred["token"]:
                continue
            actual = float(series[pred["token"] - lo])
            if math.isnan(actual):
                continue
            point = {"token": pred["token"], "value": actual}
        elif agg == "MAX":
            j_fin = int(np.argmax(finite))
            j_true = int(valid_idx[j_fin])
            actual = float(finite[j_fin])
            point = {"token": lo + j_true, "value": actual}
        elif agg == "MIN":
            j_fin = int(np.argmin(finite))
            j_true = int(valid_idx[j_fin])
            actual = float(finite[j_fin])
            point = {"token": lo + j_true, "value": actual}
        elif agg == "MEAN":
            actual = float(finite.mean())
            j_true = int(valid_idx[int(np.argmax(finite))])
            point = {"token": lo + j_true, "value": actual}
        elif agg == "STD":
            actual = float(finite.std())
            j_true = int(valid_idx[int(np.argmax(finite))])
            point = {"token": lo + j_true, "value": actual}
        elif agg == "L2":
            actual = float(np.linalg.norm(finite, ord=2))
            j_true = int(valid_idx[int(np.argmax(finite))])
            point = {"token": lo + j_true, "value": actual}
        else:  # P95
            actual = float(np.percentile(finite, 95))
            j_true = int(valid_idx[int(np.argmax(finite))])
            point = {"token": lo + j_true, "value": actual}
        if _apply_op(pred["op"], actual, pred["value"]):
            matched = True
            if (
                not best
                or (pred["agg"] != "MIN" and point["value"] > best.get("value", -math.inf))
                or (pred["agg"] == "MIN" and point["value"] < best.get("value", math.inf))
            ):
                best = {"address": matched_addrs[k], **point}
    return matched, best


def _seal_ok(run_id: str, store: str) -> bool:
    from glassbox.storage.seal import verify_run

    try:
        return bool(verify_run(run_id, store))
    except Exception:
        return False


def find_runs(
    query: str,
    store: str | None = None,
    limit: int = 20,
    tokens: tuple[int, int] | None = None,
    workers: int = 4,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Two-phase find; returns (rows, meta{hot_ms, cold_slices}).

    Raises ``ValueError`` (E02 grammar) and ``UnknownAddressError``-shaped
    ``ValueError`` (E04) for malformed addresses.
    """
    root = _index._store_root(store)
    parsed = parse_query(query)
    for pred in parsed["neuron"]:
        for addr in pred["addresses"]:
            if not _ADDR_RE.match(addr):
                raise ValueError(
                    f"E04: unknown address {addr} not in manifest; "
                    "Valid: N-01-0001..N-32-14336, A-01-01..A-32-32, LOGITS, EMBED."
                )
    token_lo, token_hi = (tokens[0], tokens[1]) if tokens else (0, 1 << 30)
    if token_lo > token_hi:
        raise ValueError(f"E02: bad --tokens range {tokens!r}.")

    import time

    t0 = time.perf_counter()

    def _candidate(entry: dict[str, Any]) -> bool:
        if not _hot_match(entry, parsed["hot"], root):
            return False
        return all(_hot_neuron_possible(entry, p) for p in parsed["neuron"])

    candidates = _index.hot_scan(_candidate, root, workers=workers)
    hot_ms = (time.perf_counter() - t0) * 1000.0
    rows: list[dict[str, Any]] = []
    cold_slices = 0
    want_hash_fail = any(c["kind"] == "flag" and c["value"] == "hash_fail" for c in parsed["hot"])
    for entry in candidates:
        run_id = str(entry["run_id"])
        run_path = os.path.join(root, "runs", run_id)
        if not parsed["neuron"]:
            ok = _seal_ok(run_id, root)
            if want_hash_fail and ok:
                continue
            rows.append(
                {
                    "run": run_id,
                    "address": None,
                    "token": None,
                    "value": None,
                    "hash_seal_verified": ok,
                }
            )
        else:
            points = []
            for pred in parsed["neuron"]:
                try:
                    match, point = _cold_neuron_check(run_path, pred, token_lo, token_hi)
                except Exception:
                    continue
                cold_slices += 1
                if match:
                    points.append(point)
            if not points:
                continue
            ok = _seal_ok(run_id, root)
            if want_hash_fail and ok:
                continue
            for point in points:
                rows.append(
                    {
                        "run": run_id,
                        "address": point.get("address"),
                        "token": point.get("token"),
                        "value": point.get("value"),
                        "hash_seal_verified": ok,
                    }
                )
        if len(rows) >= limit:
            break
    return rows[:limit], {"hot_ms": hot_ms, "cold_slices": cold_slices}


def _parse_group(expr: str) -> dict[str, Any]:
    text = expr.strip().strip("\"'")
    m = re.fullmatch(r"tag:([A-Za-z0-9_]+)=(\S+)", text)
    if m:
        return {"kind": "tag", "value": f"{m.group(1)}={m.group(2)}"}
    m = re.fullmatch(r"tag\s*=\s*\"?([^\"]+)\"?", text)
    if m:
        return {"kind": "tag", "value": m.group(1)}
    m = re.fullmatch(r"checkpoint=(\S+)", text)
    if m:
        return {"kind": "checkpoint", "value": m.group(1).replace("sha256:", "")}
    raise ValueError(f"E02: bad group {expr!r}; expected tag:k=v or checkpoint=sha.")


def _parse_stat_list(stats: list[str] | str) -> list[tuple[str, float]]:
    if isinstance(stats, str):
        stats = [s.strip() for s in stats.split(",") if s.strip()]
    out: list[tuple[str, float]] = []
    for name in stats:
        m = re.fullmatch(r"(?i)firing_rate\(([\d.]+)\)", name.strip())
        if m:
            out.append(("firing_rate", float(m.group(1))))
            continue
        low = name.strip().lower()
        if low in ("mean", "std", "l2", "p95", "max", "min"):
            out.append((low, 0.0))
        elif low == "firing_rate":
            out.append(("firing_rate", 0.8))
        else:
            raise ValueError(
                f"E02: bad stat {name!r}; expected mean|std|l2|p95|max|min|firing_rate[(t)]."
            )
    return out


def _pooled_stats(values: list[float], thresh: float) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "n_values": 0,
            "mean": None,
            "std": None,
            "l2": None,
            "p95": None,
            "max": None,
            "min": None,
            "firing_rate": 0.0,
        }
    return {
        "n_values": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "l2": float(np.linalg.norm(arr, ord=2)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "firing_rate": float((arr > thresh).mean()),
    }


def compare_runs(
    good: str,
    bad: str,
    neurons: list[str],
    stats: list[str] | str = "mean,p95,firing_rate",
    tokens: tuple[int, int] = (15, 20),
    limit: int = 10000,
    store: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Two group passes + pooled aggregates per address (spec §SEARCH-grammar).

    Returns ``(per_address_results, meta)`` with ``meta["cold_slices"]``.
    """
    from glassbox.storage.layout import read_deep_index, read_deep_slice

    root = _index._store_root(store)
    for addr in neurons:
        for one in _expand_range_token(addr):
            if not _ADDR_RE.match(one):
                raise ValueError(
                    f"E04: unknown address {one} not in manifest; "
                    "Valid: N-01-0001..N-32-14336, A-01-01..A-32-32, LOGITS, EMBED."
                )
    expanded: list[str] = []
    for addr in neurons:
        expanded.extend(_expand_range_token(addr))
    stat_list = _parse_stat_list(stats)
    groups = {"good": _parse_group(good), "bad": _parse_group(bad)}
    lo, hi = tokens
    results: list[dict[str, Any]] = []
    cold_slices = 0
    for addr in expanded:
        per: dict[str, Any] = {"address": addr, "tokens": f"{lo}..{hi}", "good": {}, "bad": {}}
        for side in ("good", "bad"):
            cond = groups[side]
            pooled: list[float] = []
            n_runs = 0
            for entry in _index.iter_hot_entries(root):
                if n_runs >= limit:
                    break
                if cond["kind"] == "tag":
                    if not _tag_match([str(t) for t in entry.get("tags", [])], cond["value"]):
                        continue
                elif str(entry.get("checkpoint_sha256", "")) != cond["value"]:
                    continue
                run_path = os.path.join(root, "runs", str(entry["run_id"]))
                try:
                    index = read_deep_index(run_path)
                except ValueError:
                    continue
                if not index.get("triggered") or addr not in (index.get("watched", [])):
                    continue
                total = int(index.get("tokens", 0))
                slo, shi = max(lo, 0), min(hi, total - 1)
                if slo > shi:
                    continue
                try:
                    mat = read_deep_slice(run_path, slo, shi, [list(index["watched"]).index(addr)])
                    cold_slices += 1
                except ValueError:
                    continue
                pooled.extend(float(v) for v in mat[:, 0].astype(np.float64))
                n_runs += 1
            if n_runs == 0:
                per[side] = {"n": 0, "empty": True}
                continue
            thresh = next((t for s, t in stat_list if s == "firing_rate"), 0.8)
            agg = _pooled_stats(pooled, thresh)
            agg["n"] = n_runs
            per[side] = agg
        if per["good"].get("empty") or per["bad"].get("empty"):
            per["attribution"] = 0.0
            per["delta_mean"] = 0.0
        else:
            gm = float(per["good"].get("mean", 0.0) or 0.0)
            bm = float(per["bad"].get("mean", 0.0) or 0.0)
            g_fr = float(per["good"].get("firing_rate", 0.0) or 0.0)
            b_fr = float(per["bad"].get("firing_rate", 0.0) or 0.0)
            diff = gm - bm
            per["delta_mean"] = round(diff, 4)
            per["attribution"] = round(diff * (g_fr - b_fr + 1.0), 4)
        results.append(per)
    return results, {"cold_slices": cold_slices}
