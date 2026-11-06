"""Watchlist DSL parse + Deep presets + validator (spec §WATCH-grammar/presets, plan §9.1).

Deep-Only Architecture:
Every watchlist must resolve to at least one watched internal activation address.
Presets expand deterministically against the manifest architecture
(layers/heads/mlp/hidden/vocab). Layer selections clamp to the manifest so small
CPU test manifests and full 32/80-layer transformer manifests both work cleanly.
``pilot-sparse-1k`` uses ``random.Random(42)`` (250 neurons x 4 layers).

Supported Deep addresses:
- ``N-LL-NNNN``: MLP intermediate post-activation neuron ``1..mlp_per_layer``
- ``A-LL-HH``: Attention head ``1..heads`` (per-head reduction across ``head_dim``)
- ``L-LL``: Residual stream layer output summary (L2 norm across ``hidden``)
- ``L-LL-DDDD``: Residual stream specific hidden dimension ``1..hidden``
- ``LOGITS``: Top-1 max logit at the output head
- ``LOGITS:<id>``: Exact logit for vocabulary token ``0..vocab-1`` (IOI / circuit probing)
- ``LOGITS-ENTROPY``: Shannon entropy of the output logit distribution
- ``EMBED``: Token embedding output summary (L2 norm)
- ``NORM``: Final layer norm output summary (L2 norm)

Exit mapping for CLI: E02 ``ValueError``/``FileNotFoundError`` (bad YAML),
E04 ``UnknownAddressError`` (first bad address + valid range).
"""

from __future__ import annotations

import hashlib
import os
import random
import re
from typing import Any

import yaml

from glassbox.exceptions import UnknownAddressError

_RE_N = re.compile(r"^N-(\d+)-(\d+)$")
_RE_A = re.compile(r"^A-(\d+)-(\d+)$")
_RE_L = re.compile(r"^L-(\d+)$")
_RE_L_DIM = re.compile(r"^L-(\d+)-(\d+)$")
_RE_SAE = re.compile(r"^SAE-(\d+)-(\d+)$")
_RE_F = re.compile(r"^F-(\d+)-(\d+)$")
_RE_LOGITS_TOK = re.compile(r"^LOGITS:(\d+)$")
_RE_SUFFIX_RANGE = re.compile(
    r"^(N-\d+-\d+|A-\d+-\d+|L-\d+-\d+|L-\d+|SAE-\d+-\d+|F-\d+-\d+)\.\.(\d+)$"
)
_SPECIALS = ("LOGITS", "LOGITS-ENTROPY", "EMBED", "NORM")

_REQUIRED_FIELDS = (
    "version",
    "model_id",
    "checkpoint_sha256",
    "watchlist_id",
    "trigger",
    "window",
    "precision",
    "entries",
)

PRESETS: tuple[str, ...] = (
    "last-4-layers-full",
    "attention_only",
    "mid-mlp-wide",
    "logit-lens",
    "pilot-sparse-1k",
    "ioi-circuit-starter",
)

_PILOT_SEED = 42
_PILOT_PER_LAYER = 250
_PILOT_LAYERS = (8, 16, 24, 31)


def _valid_range_text(manifest: dict[str, Any]) -> str:
    """Human-readable valid range summary derived from manifest arch."""
    arch = manifest.get("architecture", manifest)
    layers = int(arch.get("layers", 0))
    heads = int(arch.get("heads", 0))
    mlp = int(arch.get("mlp_per_layer", manifest.get("inventory", {}).get("neurons_per_layer", 0)))
    return (
        f"Valid: N-01-0001..N-{layers:02d}-{mlp:04d}, "
        f"A-01-01..A-{layers:02d}-{heads:02d}, LOGITS, EMBED."
    )


def manifest_arch(manifest: dict[str, Any]) -> tuple[int, int, int]:
    """Return ``(layers, heads, mlp_per_layer)`` from a manifest dict.

    Raises:
        ValueError: E02 when the manifest lacks architecture fields.
    """
    arch = manifest.get("architecture", manifest)
    inv = manifest.get("inventory", {})
    try:
        layers = int(arch["layers"])
        heads = int(arch["heads"])
        mlp = int(arch.get("mlp_per_layer", inv.get("neurons_per_layer")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"E02: manifest missing architecture: {exc}") from exc
    if min(layers, heads, mlp) < 1:
        raise ValueError(f"E02: manifest has non-positive architecture {(layers, heads, mlp)}")
    return (layers, heads, mlp)


def _clamp_layers(wanted: list[int], layers: int) -> list[int]:
    """Clamp wanted 1-based layers into ``[1..layers]``, deduped in order."""
    out: list[int] = []
    for layer in wanted:
        clamped = max(1, min(int(layer), layers))
        if clamped not in out:
            out.append(clamped)
    return out


def expand_preset(name: str, manifest: dict[str, Any]) -> list[str]:
    """Expand one Deep preset name to exact addresses (deterministic).

    Args:
        name: one of the 6 names in :data:`PRESETS`.
        manifest: manifest dict (only ``architecture`` is read).

    Raises:
        ValueError: E02 on unknown preset name.
    """
    if name not in PRESETS:
        raise ValueError(f"E02: unknown preset {name!r}; expected one of {list(PRESETS)}")
    layers, heads, mlp = manifest_arch(manifest)
    if name == "last-4-layers-full":
        sel = _clamp_layers([layers - 3, layers - 2, layers - 1, layers], layers)
        out: list[str] = []
        for layer in sel:
            for head in range(1, heads + 1):
                out.append(f"A-{layer:02d}-{head:02d}")
            for neuron in range(1, mlp + 1):
                out.append(f"N-{layer:02d}-{neuron:04d}")
        out.append("LOGITS")
        return out
    if name == "attention_only":
        return [
            f"A-{layer:02d}-{head:02d}"
            for layer in range(1, layers + 1)
            for head in range(1, heads + 1)
        ]
    if name == "mid-mlp-wide":
        sel = _clamp_layers([12, 13, 14, 15, 16, 17, 18, 19, 20], layers)
        return [f"N-{layer:02d}-{i:04d}" for layer in sel for i in range(1, mlp + 1)]
    if name == "logit-lens":
        return [f"L-{layer:02d}" for layer in range(1, layers + 1)] + ["NORM", "LOGITS"]
    if name == "pilot-sparse-1k":
        sel = _clamp_layers(list(_PILOT_LAYERS), layers)
        rng = random.Random(_PILOT_SEED)
        out_pilot: list[str] = []
        for layer in sel:
            take = min(_PILOT_PER_LAYER, mlp)
            for neuron in sorted(rng.sample(range(1, mlp + 1), take)):
                out_pilot.append(f"N-{layer:02d}-{neuron:04d}")
        return out_pilot
    # ioi-circuit-starter: A-17..24 heads 04/07/12 + N-20..24 neurons 0001..0200.
    sel_attn = _clamp_layers([17, 18, 19, 20, 21, 22, 23, 24], layers)
    sel_mlp = _clamp_layers([20, 21, 22, 23, 24], layers)
    kept_heads = [h for h in (4, 7, 12) if 1 <= h <= heads]
    out_ioi: list[str] = []
    for layer in sel_attn:
        for head in kept_heads:
            out_ioi.append(f"A-{layer:02d}-{head:02d}")
    for layer in sel_mlp:
        for neuron in range(1, min(200, mlp) + 1):
            out_ioi.append(f"N-{layer:02d}-{neuron:04d}")
    return out_ioi


def preset_window_hint(name: str) -> dict[str, int]:
    """Window hint a preset carries (``mid-mlp-wide`` steps every 10th token)."""
    if name == "mid-mlp-wide":
        return {"every_n": 10}
    return {}


def is_single_address(addr: str) -> bool:
    """Return True if ``addr`` is a well-formed single Deep address token."""
    addr = addr.strip()
    return bool(
        _RE_N.match(addr)
        or _RE_A.match(addr)
        or _RE_L.match(addr)
        or _RE_L_DIM.match(addr)
        or _RE_SAE.match(addr)
        or _RE_F.match(addr)
        or _RE_LOGITS_TOK.match(addr)
        or addr in _SPECIALS
    )


def _expand_suffix_range(token: str) -> list[str]:
    """Expand ``N-24-0001..0100`` / ``A-24-01..08`` / ``L-01..04`` suffix ranges (inclusive)."""
    m = _RE_SUFFIX_RANGE.match(token.strip())
    if not m:
        return []
    base, end_s = m.group(1), m.group(2)
    if base.startswith("N-"):
        lm = _RE_N.match(base)
        assert lm is not None
        layer, start = int(lm.group(1)), int(lm.group(2))
        end = int(end_s)
        width = max(len(lm.group(2)), len(end_s))
        if end < start:
            raise ValueError(f"E02: bad range {token}")
        return [f"N-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    if base.startswith("A-"):
        lm = _RE_A.match(base)
        assert lm is not None
        layer, start = int(lm.group(1)), int(lm.group(2))
        end = int(end_s)
        width = max(len(lm.group(2)), len(end_s))
        if end < start:
            raise ValueError(f"E02: bad range {token}")
        return [f"A-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    if base.startswith("SAE-") or base.startswith("F-"):
        is_sae = base.startswith("SAE-")
        pref = "SAE" if is_sae else "F"
        lm_sae = _RE_SAE.match(base) if is_sae else _RE_F.match(base)
        assert lm_sae is not None
        layer, start = int(lm_sae.group(1)), int(lm_sae.group(2))
        end = int(end_s)
        width = max(len(lm_sae.group(2)), len(end_s))
        if end < start:
            raise ValueError(f"E02: bad range {token}")
        return [f"{pref}-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    lm_dim = _RE_L_DIM.match(base)
    if lm_dim is not None:
        layer, start = int(lm_dim.group(1)), int(lm_dim.group(2))
        end = int(end_s)
        width = max(len(lm_dim.group(2)), len(end_s))
        if end < start:
            raise ValueError(f"E02: bad range {token}")
        return [f"L-{layer:02d}-{i:0{width}d}" for i in range(start, end + 1)]
    lm_layer = _RE_L.match(base)
    if lm_layer is not None:
        start = int(lm_layer.group(1))
        end = int(end_s)
        width = max(len(lm_layer.group(1)), len(end_s))
        if end < start:
            raise ValueError(f"E02: bad range {token}")
        return [f"L-{i:0{width}d}" for i in range(start, end + 1)]
    return []


def expand_address(token: str) -> list[str]:
    """Expand one watchlist token (single or ``..`` range) to addresses."""
    token = token.strip()
    if is_single_address(token):
        return [token]
    if ".." in token:
        left, _, right = token.partition("..")
        left, right = left.strip(), right.strip()
        suffix_expanded = _expand_suffix_range(token)
        if suffix_expanded:
            return suffix_expanded
        if is_single_address(left) and is_single_address(right):
            if left == right:
                return [left]
            raise ValueError(f"E02: only same-layer suffix ranges supported, got {token}")
        raise ValueError(f"E02: bad address range {token}")
    raise ValueError(f"E02: bad address {token}")


def expand_addresses(tokens: list[str]) -> list[str]:
    """Expand + dedupe tokens preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for tok in tokens:
        for addr in expand_address(str(tok)):
            if addr not in seen:
                seen.add(addr)
                out.append(addr)
    return out


def validate_addresses(addresses: list[str], manifest: dict[str, Any]) -> None:
    """Validate every address vs manifest; E04 on first unknown.

    Raises:
        ValueError: E02 when address list is empty (Deep requires >= 1 address).
        UnknownAddressError: names first bad address + valid range.
    """
    if not addresses:
        raise ValueError("E02: Deep watchlist requires at least 1 watched address")
    layers, heads, mlp = manifest_arch(manifest)
    arch = manifest.get("architecture", manifest)
    hidden = int(arch.get("hidden", mlp))
    vocab = int(arch.get("vocab", 1 << 30))
    for addr in addresses:
        ok = False
        m = _RE_N.match(addr)
        if m:
            ok = 1 <= int(m.group(1)) <= layers and 1 <= int(m.group(2)) <= mlp
        elif _RE_A.match(addr or ""):
            m2 = _RE_A.match(addr)
            assert m2 is not None
            ok = 1 <= int(m2.group(1)) <= layers and 1 <= int(m2.group(2)) <= heads
        elif _RE_L.match(addr or ""):
            m3 = _RE_L.match(addr)
            assert m3 is not None
            ok = 1 <= int(m3.group(1)) <= layers
        elif _RE_L_DIM.match(addr or ""):
            m4 = _RE_L_DIM.match(addr)
            assert m4 is not None
            ok = 1 <= int(m4.group(1)) <= layers and 1 <= int(m4.group(2)) <= hidden
        elif _RE_LOGITS_TOK.match(addr or ""):
            m5 = _RE_LOGITS_TOK.match(addr)
            assert m5 is not None
            ok = 0 <= int(m5.group(1)) < vocab
        elif _RE_SAE.match(addr or "") or _RE_F.match(addr or ""):
            m_sae = _RE_SAE.match(addr or "") or _RE_F.match(addr or "")
            assert m_sae is not None
            ok = 1 <= int(m_sae.group(1)) <= layers and 1 <= int(m_sae.group(2))
        elif addr in _SPECIALS:
            ok = True
        if not ok:
            raise UnknownAddressError(
                f"E04: unknown address {addr} not in manifest "
                f"{manifest.get('model_id', '?')}; {_valid_range_text(manifest)}"
            )


def _validate_window(window: Any, path: str) -> dict[str, int]:
    """Validate the file ``window`` block; E02 on bad shape/values."""
    if not isinstance(window, dict) or "start_token" not in window or "end_token" not in window:
        raise ValueError(f"E02: watchlist {path}: bad window {window!r}")
    try:
        start = int(window["start_token"])
        end = int(window["end_token"])
        every_n = int(window.get("every_n", 1))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"E02: watchlist {path}: bad window {window!r}") from exc
    if start < 0 or end < 1 or end < start or every_n < 1:
        raise ValueError(f"E02: watchlist {path}: bad window {window!r}")
    return {"start_token": start, "end_token": end, "every_n": every_n}


def _validate_trigger(trigger: Any, path: str) -> None:
    """Light trigger shape check (full semantics live in controller)."""
    if isinstance(trigger, str):
        if trigger == "always" or trigger.startswith("on_flag:") or trigger.startswith("window:"):
            return
        raise ValueError(f"E02: watchlist {path}: bad trigger {trigger!r}")
    if isinstance(trigger, dict):
        if set(trigger) <= {"on_flag", "window", "address", "lo", "hi"} and trigger:
            return
        raise ValueError(f"E02: watchlist {path}: bad trigger {trigger!r}")
    raise ValueError(f"E02: watchlist {path}: bad trigger {trigger!r}")


def _expand_entries(
    entries: list[Any], path: str, manifest: dict[str, Any] | None
) -> tuple[list[str], list[str]]:
    """Expand ``entries`` to ordered tokens; also returns presets used."""
    tokens: list[str] = []
    presets_used: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"E02: watchlist {path}: bad entry {entry!r}")
        if "preset" in entry and "extends" not in entry:
            name = str(entry["preset"])
            if name not in PRESETS:
                raise ValueError(f"E02: watchlist {path}: unknown preset {name!r}")
            presets_used.append(name)
            if manifest is None:
                raise ValueError(
                    f"E02: watchlist {path}: preset {name!r} needs a manifest; "
                    "call with manifest= or validate later via controller"
                )
            tokens.extend(expand_preset(name, manifest))
        elif "addresses" in entry:
            addrs = entry["addresses"]
            if not isinstance(addrs, list):
                raise ValueError(f"E02: watchlist {path}: addresses must be a list")
            tokens.extend(str(a) for a in addrs)
        elif "extends" in entry:
            base = entry["extends"]
            names = [str(base)] if isinstance(base, str) else [str(n) for n in base]
            extra = entry.get("watch", [])
            if not isinstance(extra, list):
                raise ValueError(f"E02: watchlist {path}: watch must be a list")
            for name in names:
                if name not in PRESETS:
                    raise ValueError(f"E02: watchlist {path}: unknown preset {name!r}")
                presets_used.append(name)
                if manifest is None:
                    raise ValueError(
                        f"E02: watchlist {path}: extends {name!r} needs a manifest; "
                        "call with manifest= or validate later via controller"
                    )
                tokens.extend(expand_preset(name, manifest))
            tokens.extend(str(a) for a in extra)
        else:
            raise ValueError(f"E02: watchlist {path}: bad entry {entry!r}")
    return tokens, presets_used


def parse_watchlist(
    path: str,
    manifest: dict[str, Any] | None = None,
    add: list[str] | None = None,
    window: dict[str, int] | None = None,
    trigger: str | dict[str, Any] | None = None,
    precision: str | None = None,
) -> dict[str, Any]:
    """Parse + expand a Deep watchlist YAML file (spec §WATCH-grammar/presets).

    Preset/``extends`` entries expand against ``manifest`` when given;
    ``add``/``window``/``trigger``/``precision`` overrides are expanded and
    sealed as ``patch`` (``None`` when no override was passed).

    Returns dict with file fields plus ``addresses`` (expanded, ordered,
    deduped), ``watchlist_sha256`` (file bytes sha) and ``patch``.

    Raises:
        FileNotFoundError: E02 missing file.
        ValueError: E02 bad YAML/fields/trigger/window/precision/preset or empty addresses.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"E02: watchlist not found: {path}")
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        data = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"E02: bad watchlist YAML {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"E02: bad watchlist {path}: top level must be a map")
    for field in _REQUIRED_FIELDS:
        if field not in data:
            raise ValueError(f"E02: watchlist {path} missing required: {field}")
    if data["version"] != 1:
        raise ValueError(f"E02: watchlist {path}: version must be 1")
    file_precision = str(data["precision"])
    if file_precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"E02: watchlist {path}: bad precision {file_precision!r}")
    _validate_window(data["window"], path)
    _validate_trigger(data["trigger"], path)
    entries = data["entries"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"E02: watchlist {path}: entries must be a non-empty list")

    tokens, presets_used = _expand_entries(entries, path, manifest)
    if add:
        tokens.extend(str(a) for a in add)
    addresses = expand_addresses(tokens)
    if not addresses:
        raise ValueError(f"E02: watchlist {path}: Deep mode requires at least 1 watched address")

    merged_window = dict(data["window"])
    if window:
        for key in ("start_token", "end_token", "every_n"):
            if key in window:
                merged_window[key] = int(window[key])
    else:
        for name in presets_used:
            hint = preset_window_hint(name)
            if hint and merged_window.get("every_n", 1) == 1:
                merged_window.update(hint)
    merged_window = {
        "start_token": int(merged_window["start_token"]),
        "end_token": int(merged_window["end_token"]),
        "every_n": int(merged_window.get("every_n", 1)),
    }
    _validate_window(merged_window, path)

    eff_trigger = trigger if trigger is not None else data["trigger"]
    _validate_trigger(eff_trigger, path)
    eff_precision = precision if precision is not None else file_precision
    if eff_precision not in ("fp32", "fp16", "bf16"):
        raise ValueError(f"E02: bad precision {eff_precision!r}")

    patch: dict[str, Any] | None = None
    if add or window or trigger is not None or precision is not None:
        patch = {
            "add": list(add) if add else [],
            "window": dict(window) if window else None,
            "trigger": trigger,
            "precision": precision,
        }

    parsed: dict[str, Any] = dict(data)
    parsed["addresses"] = addresses
    parsed["presets_used"] = presets_used
    parsed["merged_window"] = merged_window
    parsed["effective_trigger"] = eff_trigger
    parsed["effective_precision"] = eff_precision
    parsed["patch"] = patch
    parsed["watchlist_sha256"] = hashlib.sha256(raw).hexdigest()
    return parsed
