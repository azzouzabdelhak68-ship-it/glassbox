"""Controller: trigger/window/size gate (spec §CTRL-triggers/estimator, plan §9.2/9.5).

Trigger is evaluated per token BEFORE arming Deep hooks. Unarmed tokens
write nothing to ring slots ``1..N``; a run that never arms seals
``deep_index.json`` with ``triggered=false`` and no ``deep.bin``.

Exit mapping for CLI: E02 ``ValueError`` (bad trigger/params), E04
``UnknownAddressError``/``CheckpointMismatchError``, E05 ``SizeFlagError``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from glassbox.exceptions import CheckpointMismatchError, SizeFlagError
from glassbox.watchlist import manifest_arch, parse_watchlist, validate_addresses

GREEN_MAX_BYTES = 50 * 1_000_000_000
RED_MIN_BYTES = 1000 * 1_000_000_000
_OVERHEAD = 1.15
_PRECISION_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

_RE_WINDOW_S = re.compile(r"^window:(\d+(?:\.\d+)?)s\.\.(\d+(?:\.\d+)?)s$")
_RE_WINDOW_TOK = re.compile(r"^window:(\d+)\.\.(\d+)$")


def estimate_size(w: int, t: int, r: int, b: int) -> tuple[float, str]:
    """Estimate store bytes and GREEN/YELLOW/RED flag.

    Formula (normative): ``bytes = W*T*R*B*1.15``. GREEN <50GB,
    YELLOW 50-1000GB, RED >1000GB.
    """
    if min(w, t, r, b) < 0:
        raise ValueError("E02: estimator inputs must be non-negative")
    total = float(w) * float(t) * float(r) * float(b) * _OVERHEAD
    if total < GREEN_MAX_BYTES:
        return (total, "GREEN")
    if total <= RED_MIN_BYTES:
        return (total, "YELLOW")
    return (total, "RED")


def precision_bytes(precision: str) -> int:
    """Map logical precision to bytes per value."""
    if precision not in _PRECISION_BYTES:
        raise ValueError(f"E02: bad precision {precision!r}")
    return _PRECISION_BYTES[precision]


def check_checkpoint(manifest: dict[str, Any], watchlist: dict[str, Any]) -> None:
    """Raise ``CheckpointMismatchError`` (E04) on model/sha drift."""
    m_id, w_id = manifest.get("model_id"), watchlist.get("model_id")
    m_sha, w_sha = manifest.get("checkpoint_sha256"), watchlist.get("checkpoint_sha256")
    if m_id != w_id or m_sha != w_sha:
        raise CheckpointMismatchError(
            f"E04: checkpoint mismatch expected {m_sha} got {w_sha}. "
            "Re-scan; old watchlists invalid."
        )


def parse_trigger(spec: str | dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a trigger spec to a dict (E02 on bad shape).

    Accepted: ``"always"``, ``"on_flag:bad_output"``, ``"window:5s..10s"``
    (elapsed seconds), ``"window:10..50"`` (inclusive token range),
    ``{"on_flag": "bad_output"}``, ``{"window": {"after_s":.., "until_s":..}}``,
    ``{"address": "N-24-0001", "lo": x, "hi": y}`` range gate.
    """
    if spec is None:
        return {"kind": "always"}
    if isinstance(spec, dict):
        if set(spec) == {"on_flag"} and isinstance(spec["on_flag"], str):
            return {"kind": "on_flag", "flag": str(spec["on_flag"])}
        if set(spec) == {"window"} and isinstance(spec["window"], dict):
            win = spec["window"]
            try:
                after = float(win["after_s"])
                until = float(win["until_s"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"E02: bad trigger window {spec!r}") from exc
            if after < 0 or until < after:
                raise ValueError(f"E02: bad trigger window {spec!r}")
            return {"kind": "time_window", "after_s": after, "until_s": until}
        if set(spec) == {"address", "lo", "hi"} and isinstance(spec["address"], str):
            try:
                lo, hi = float(spec["lo"]), float(spec["hi"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"E02: bad trigger gate {spec!r}") from exc
            if lo > hi:
                raise ValueError(f"E02: bad trigger gate {spec!r}")
            return {"kind": "address_gate", "address": spec["address"], "lo": lo, "hi": hi}
        raise ValueError(f"E02: bad trigger {spec!r}")
    if not isinstance(spec, str):
        raise ValueError(f"E02: bad trigger {spec!r}")
    if spec == "always":
        return {"kind": "always"}
    if spec.startswith("on_flag:"):
        flag = spec[len("on_flag:") :]
        if flag:
            return {"kind": "on_flag", "flag": flag}
        raise ValueError(f"E02: bad trigger {spec!r}")
    m = _RE_WINDOW_S.match(spec)
    if m:
        after, until = float(m.group(1)), float(m.group(2))
        if until < after:
            raise ValueError(f"E02: bad trigger {spec!r}")
        return {"kind": "time_window", "after_s": after, "until_s": until}
    m = _RE_WINDOW_TOK.match(spec)
    if m:
        start, end = int(m.group(1)), int(m.group(2))
        if end < start:
            raise ValueError(f"E02: bad trigger {spec!r}")
        return {"kind": "token_range", "start_token": start, "end_token": end}
    raise ValueError(
        f"E02: bad trigger {spec!r}; allowed: always, on_flag:bad_output, "
        "window:5s..10s, window:10..50"
    )


def is_armed(
    trigger: dict[str, Any],
    token_pos: int,
    window: dict[str, int] | None = None,
    elapsed_s: float | None = None,
    flags: dict[str, Any] | None = None,
    values: dict[str, float] | None = None,
) -> bool:
    """Return True when Deep hooks are armed for ``token_pos``.

    The ``window`` gate (``start_token`` inclusive, ``end_token`` exclusive,
    stepping ``every_n``) always applies first; unarmed tokens write nothing.
    """
    window = window or {"start_token": 0, "end_token": 1 << 30, "every_n": 1}
    start = int(window.get("start_token", 0))
    end = int(window.get("end_token", 1 << 30))
    every_n = int(window.get("every_n", 1))
    if token_pos < start or token_pos >= end or every_n < 1:
        return False
    if (token_pos - start) % every_n != 0:
        return False
    kind = trigger.get("kind", "always")
    if kind == "always":
        return True
    if kind == "on_flag":
        return bool((flags or {}).get(trigger["flag"], False))
    if kind == "time_window":
        if elapsed_s is None:
            return False
        return bool(trigger["after_s"] <= elapsed_s <= trigger["until_s"])
    if kind == "token_range":
        return bool(trigger["start_token"] <= token_pos <= trigger["end_token"])
    if kind == "address_gate":
        vals = values or {}
        addr = trigger["address"]
        if addr not in vals:
            return False
        return bool(trigger["lo"] <= float(vals[addr]) <= trigger["hi"])
    return False


def armed_tokens(
    trigger: dict[str, Any],
    window: dict[str, int],
    n_tokens: int,
    elapsed: list[float] | None = None,
    flags: dict[str, Any] | None = None,
    values: list[dict[str, float]] | None = None,
) -> list[int]:
    """List token positions in ``[0..n_tokens)`` where Deep is armed."""
    out: list[int] = []
    for pos in range(n_tokens):
        tick = elapsed[pos] if elapsed is not None and pos < len(elapsed) else None
        vals = values[pos] if values is not None and pos < len(values) else None
        if is_armed(trigger, pos, window, tick, flags, vals):
            out.append(pos)
    return out


def _load_manifest(manifest: dict[str, Any] | str) -> dict[str, Any]:
    """Accept a manifest dict or path to ``manifest.json``."""
    if isinstance(manifest, dict):
        return manifest
    with open(manifest, encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[no-any-return]


def validate_watchlist(
    watchlist_path: str,
    manifest: dict[str, Any] | str,
    add: list[str] | None = None,
    window: dict[str, int] | None = None,
    trigger: str | dict[str, Any] | None = None,
    precision: str | None = None,
) -> list[str]:
    """Parse watchlist, check checkpoint, validate addresses (E04 fail fast).

    ``add``/``window``/``trigger``/``precision`` overrides are expanded and
    validated exactly like file content; unknown addresses raise naming the
    first bad address before any GPU work starts.
    """
    man = _load_manifest(manifest)
    wl = parse_watchlist(
        watchlist_path, manifest=man, add=add, window=window, trigger=trigger, precision=precision
    )
    check_checkpoint(man, wl)
    if wl.get("effective_trigger") is not None:
        parse_trigger(wl["effective_trigger"])  # E02 on bad trigger shape
    manifest_arch(man)  # E02 on broken manifest arch
    validate_addresses(list(wl["addresses"]), man)
    return list(wl["addresses"])


def run_with_watchlist(
    watchlist: str,
    tags: dict[str, str] | None = None,
    manifest: dict[str, Any] | str | None = None,
    manifest_path: str | None = None,
    force: bool = False,
    trigger: str | dict[str, Any] | None = None,
    precision: str | None = None,
    tokens: int | None = None,
    runs: int = 1,
    add: list[str] | None = None,
    window: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Validate + estimate a Deep run (sealing itself is sidecar-owned).

    Validates watchlist vs manifest BEFORE any work, normalizes the trigger,
    runs the size estimator. RED without ``force`` raises ``SizeFlagError``
    and seals nothing.

    Raises:
        ValueError: E02 (manifest required, bad trigger/params).
        UnknownAddressError / CheckpointMismatchError: E04.
        SizeFlagError: E05 RED without ``--force``.
    """
    man_src = manifest if manifest is not None else manifest_path
    if man_src is None:
        raise ValueError("E02: manifest required for validation")
    man = _load_manifest(man_src)
    wl = parse_watchlist(
        watchlist, manifest=man, add=add, window=window, trigger=trigger, precision=precision
    )
    check_checkpoint(man, wl)

    eff_trigger = parse_trigger(wl.get("effective_trigger"))
    eff_precision = str(wl.get("effective_precision", "fp16"))
    b = precision_bytes(eff_precision)

    merged = wl.get("merged_window", {})
    if tokens is not None:
        t = int(tokens)
    else:
        try:
            t = max(1, int(merged["end_token"]) - int(merged["start_token"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"E02: bad window {merged!r}") from exc
    if runs < 1 or t < 1:
        raise ValueError("E02: tokens and runs must be >= 1")

    addresses = list(wl["addresses"])
    validate_addresses(addresses, man)
    total, flag = estimate_size(len(addresses), t, int(runs), b)
    if flag == "RED" and not force:
        raise SizeFlagError(
            f"E05: RED-flagged {total / 1e9:.1f}GB estimate "
            f"(W={len(addresses)} T={t} R={runs} B={b}); "
            "require --force. Nothing ran."
        )
    _ = tags
    return {
        "watchlist_id": wl.get("watchlist_id"),
        "addresses": addresses,
        "trigger": eff_trigger,
        "window": merged,
        "patch": wl.get("patch"),
        "W": len(addresses),
        "T": t,
        "R": int(runs),
        "B": b,
        "bytes": total,
        "flag": flag,
        "dry_run": True,
        "force": force,
    }
