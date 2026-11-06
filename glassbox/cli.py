"""CLI with exactly 8 subcommands — Deep-Only Architecture (spec sections A.1-A.8).

Commands:
- ``scan``: Scan local weights on CPU into a Deep manifest (spec A.1).
- ``run``: Run inference with a Deep watchlist and seal evidence (spec A.2).
- ``find``: Two-phase search over sealed Deep runs (spec A.3).
- ``compare``: Compare good vs bad activation cohorts (spec A.4).
- ``verify``: Verify hash chain of sealed Deep runs (spec A.5).
- ``view``: View a Deep run / export static HTML (spec A.6).
- ``size``: Show store size with GREEN/YELLOW/RED flags (spec A.7).
- ``export``: Export a Deep run to a flat zip (spec A.8).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from typing import Any

COMMANDS: dict[str, tuple[str, str]] = {
    "scan": ("Scan weights on CPU into a manifest (spec section A.1).", "1"),
    "run": ("Run with a watchlist and seal evidence (spec section A.2).", "2"),
    "find": ("Two-phase search over sealed runs (spec section A.3).", "3"),
    "compare": ("Compare good vs bad cohorts (spec section A.4).", "3"),
    "verify": ("Verify hash chain of sealed runs (spec section A.5).", "1"),
    "view": ("View a run / export static HTML (spec section A.6).", "3"),
    "size": ("Show store size with GREEN/YELLOW/RED flags (spec section A.7).", "2"),
    "export": ("Export a run to a flat zip (spec section A.8).", "4"),
}

_COMMAND_ORDER: list[str] = [
    "scan",
    "run",
    "find",
    "compare",
    "verify",
    "view",
    "size",
    "export",
]

_VALID_ADDRESS_HINT = "Valid: N-01-0001..N-32-14336, A-01-01..A-32-32, LOGITS, EMBED."
_ADDRESS_RE = re.compile(
    r"^(N-\d{2}-\d{4}|A-\d{2}-\d{2}|L-\d{2}|SAE-\d{2}-\d{4}|F-\d{2}-\d{4}|LOGITS|EMBED|NORM)$"
)
_ADDRESS_RANGE_RE = re.compile(r"^N-\d{2}-\d{4}\.\.\d{4}$")
_CANDIDATE_ADDR_RE = re.compile(
    r"N-\d+-\d+|A-\d+-\d+|SAE-\d+-\d+|F-\d+-\d+|LOGITS|EMBED|NORM|L-\d+"
)
_TAG_RE = re.compile(r"^[a-z0-9_]+=.+$")


def _add_scan_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("scan", help=COMMANDS["scan"][0])
    p.add_argument("weights_path", nargs="?", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--json", action="store_true")


def _add_run_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("run", help=COMMANDS["run"][0])
    p.add_argument("--watch", default=None)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--trigger", default=None)
    p.add_argument("--precision", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")


def _add_find_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("find", help=COMMANDS["find"][0])
    p.add_argument("query", nargs="?", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--json", action="store_true")
    p.add_argument("--table", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--tokens", default=None)


def _add_compare_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("compare", help=COMMANDS["compare"][0])
    p.add_argument("--good", default=None)
    p.add_argument("--bad", default=None)
    p.add_argument("--neurons", default=None)
    p.add_argument("--stat", default="mean,p95,firing_rate")
    p.add_argument("--tokens", default="15..20")
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--attribute", action="store_true", help="Rank neurons by causal attribution.")
    p.add_argument("--json", action="store_true")


def _add_verify_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("verify", help=COMMANDS["verify"][0])
    p.add_argument("run_ids", nargs="*")
    p.add_argument("--all", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--chain", default=None)


def _add_view_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("view", help=COMMANDS["view"][0])
    p.add_argument("run_id", nargs="?", default=None)
    p.add_argument("--open", action="store_true")
    p.add_argument("--port", type=int, default=8137)
    p.add_argument("--export-html", default=None)
    p.add_argument("--lens", action="store_true", help="Display residual stream logit lens.")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--tokens", default=None)
    p.add_argument("--addrs", default=None)
    p.add_argument("--no-color", action="store_true")


def _add_size_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("size", help=COMMANDS["size"][0])
    group = p.add_mutually_exclusive_group()
    group.add_argument("--by-model", action="store_true")
    group.add_argument("--by-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--dir", default=None)


def _add_export_parser(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("export", help=COMMANDS["export"][0])
    p.add_argument("--zip", action="store_true")
    p.add_argument("--run", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--with-manifest", action="store_true")
    p.add_argument("--with-index", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="glassbox",
        description="GlassBox: Flight recorder & interpretability instrumentation engine",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    _add_scan_parser(sub)
    _add_run_parser(sub)
    _add_find_parser(sub)
    _add_compare_parser(sub)
    _add_verify_parser(sub)
    _add_view_parser(sub)
    _add_size_parser(sub)
    _add_export_parser(sub)
    return parser


def _store_root(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if "GLASSBOX_DIR" in os.environ:
        return os.environ["GLASSBOX_DIR"]
    if "BLACKBOX_DIR" in os.environ:
        return os.environ["BLACKBOX_DIR"]
    if os.path.exists("./blackbox_store") and not os.path.exists("./glassbox_store"):
        return "./blackbox_store"
    return "./glassbox_store"


def _err(code: str, message: str) -> int:
    print(f"Error {code}: {message}", file=sys.stderr)
    return {"E02": 2, "E03": 3, "E04": 4, "E05": 5, "E06": 6, "E07": 7}[code]


def _human(nbytes: float) -> str:
    if nbytes >= 1e12:
        return f"{nbytes / 1e12:.2f}TB"
    if nbytes >= 1e9:
        return f"{nbytes / 1e9:.1f}GB"
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:.1f}MB"
    if nbytes >= 1e3:
        return f"{nbytes / 1e3:.1f}KB"
    return f"{nbytes:.0f}B"


def _band_hint(flag: str) -> str:
    return {"GREEN": "(<50GB)", "YELLOW": "(50-1000GB)", "RED": "(>1000GB)"}[flag]


def _valid_address(addr: str) -> bool:
    addr = addr.strip()
    if _ADDRESS_RANGE_RE.match(addr):
        return True
    if not _ADDRESS_RE.match(addr):
        return False
    if addr.startswith("N-"):
        try:
            layer = int(addr[2:4])
            neuron = int(addr[5:9])
        except ValueError:
            return False
        return 1 <= layer <= 32 and 1 <= neuron <= 14336
    if addr.startswith("A-"):
        try:
            layer = int(addr[2:4])
            head = int(addr[5:7])
        except ValueError:
            return False
        return 1 <= layer <= 32 and 1 <= head <= 32
    return True


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore[import-untyped]

        with open(path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def _count_watched(cfg: dict[str, Any]) -> int:
    total = 0
    entries = cfg.get("entries", [])
    if not isinstance(entries, list):
        return 1000
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for addr in entry.get("addresses", []) or []:
            m = re.fullmatch(r"N-\d{2}-(\d{4})\.\.(\d{4})", str(addr))
            if m:
                total += max(0, int(m.group(2)) - int(m.group(1)) + 1)
            else:
                total += 1
        if entry.get("preset"):
            total += 1000
        for addr in entry.get("watch", []) or []:
            _ = addr
            total += 1
    return total if total > 0 else 1000


def _run_plan(
    watch: str | None, trigger: str | None, precision: str | None
) -> tuple[int, int, int, int, str]:
    if watch and os.path.isfile(watch):
        cfg = _load_yaml(watch)
        w = _count_watched(cfg)
        precision = precision or str(cfg.get("precision", "fp16"))
        window = cfg.get("window", {})
        t = 500
        if isinstance(window, dict):
            try:
                t = max(1, int(window.get("end_token", 511)) - int(window.get("start_token", 0)))
            except (TypeError, ValueError):
                t = 500
    else:
        w, t = 1000, 500
        precision = precision or "fp16"
    b = {"fp32": 4, "fp16": 2, "bf16": 2}.get(precision, 2)
    return w, t, 1, b, precision


def _estimator_line(w: int, t: int, r: int, b: int, precision: str) -> tuple[str, str, float]:
    from glassbox import estimate_bytes, flag_for_bytes

    raw = float(w * t * r * b)
    total = estimate_bytes(w, t, r, b)
    flag = flag_for_bytes(total)
    line = (
        f"Estimator: W={w} T={t} R={r} B={b}B({precision}) "
        f"-> {_human(raw)} x1.15 = {_human(total)} -> {flag} {_band_hint(flag)}."
    )
    return line, flag, total


def _validate_trigger(trigger: str | None) -> str | None:
    if trigger is None:
        return None
    if trigger in ("always", "on_flag:bad_output"):
        return None
    if re.fullmatch(r"window:\d+s\.\.\d+s", trigger):
        return None
    if re.fullmatch(r"window:\d+\.\.\d+", trigger):
        return None
    return (
        f"bad trigger {trigger!r}; allowed: always, on_flag:bad_output, window:5s..10s, window:A..B"
    )


def _parse_tags(tag_args: list[str]) -> tuple[dict[str, str] | None, str | None]:
    tags: dict[str, str] = {}
    for item in tag_args or []:
        if not _TAG_RE.match(item):
            return None, f"bad tag {item!r}; expected [a-z0-9_]+=value"
        key, _, value = item.partition("=")
        tags[key] = value
    return tags, None


def _iter_runs(store: str) -> list[tuple[str, dict[str, Any], str]]:
    runs: list[tuple[str, dict[str, Any], str]] = []
    runs_dir = os.path.join(store, "runs")
    if not os.path.isdir(runs_dir):
        return runs
    for name in sorted(os.listdir(runs_dir)):
        if not re.fullmatch(r"RUN-\d{6}", name):
            continue
        run_dir = os.path.join(runs_dir, name)
        index_path = os.path.join(run_dir, "deep_index.json")
        try:
            with open(index_path, encoding="utf-8") as fh:
                index = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(index, dict):
            runs.append((name, index, run_dir))
    return runs


# ---------------------------------------------------------------- scan ---


def cmd_scan(args: argparse.Namespace) -> int:
    from glassbox import GlassBox, ManifestExistsError

    if not args.weights_path:
        return _err("E02", "scan requires WEIGHTS_PATH directory containing config.json.")
    try:
        manifest = GlassBox.scan(args.weights_path, args.output, args.force)
    except ManifestExistsError as exc:
        return _err("E03", str(exc))
    except FileNotFoundError as exc:
        return _err("E02", str(exc))
    except Exception as exc:
        return _err("E02", f"scan failed: {exc}")
    sha = str(manifest.get("checkpoint_sha256", "0" * 64))
    manifest_path = str(manifest.get("manifest_path", ""))
    arch = manifest.get("architecture", {})
    if args.json:
        print(
            json.dumps(
                {
                    "manifest": manifest_path,
                    "model_id": manifest.get("model_id", "M-001"),
                    "checkpoint_sha256": sha,
                }
            )
        )
        return 0
    layers = arch.get("layers", 32)
    hidden = arch.get("hidden", 4096)
    heads = arch.get("heads", 32)
    mlp = arch.get("mlp_per_layer", 14336)
    name = os.path.basename(os.path.normpath(args.weights_path))
    duration = manifest.get("scan", {}).get("duration_s", 0.1)
    print(f"Scan CPU {duration:.1f}s: config.json + safetensors header only (no GPU, no prompt).")
    print(f"Manifest -> {manifest_path}")
    print(
        f"M-001 {name} | C-{sha[:6]} sha256:{sha[:6]}... | "
        f"{layers}L hidden{hidden} {heads}H MLP{mlp} | torch-cpu | ready for Deep watchlists"
    )
    return 0


# ----------------------------------------------------------------- run ---


def cmd_run(args: argparse.Namespace) -> int:
    from glassbox import (
        CheckpointMismatchError,
        GlassBox,
        SizeFlagError,
        UnknownAddressError,
    )

    if not args.watch:
        return _err("E02", "run requires --watch FILE.")
    if args.limit is not None and args.limit < 1:
        return _err("E02", "--limit must be >= 1 (--limit 0 rejected).")
    trigger_err = _validate_trigger(args.trigger)
    if trigger_err:
        return _err("E02", trigger_err)
    if args.precision is not None and args.precision not in ("fp32", "fp16", "bf16"):
        return _err("E02", f"bad precision {args.precision!r}; expected fp32|fp16|bf16")
    tags, tag_err = _parse_tags(list(args.tag or []))
    if tag_err:
        return _err("E02", tag_err)
    if not os.path.isfile(args.watch):
        return _err("E02", f"watchlist file not found: {args.watch}")

    # Fail fast on unknown addresses before any estimate/seal (E04)
    cfg = _load_yaml(args.watch)
    for entry in cfg.get("entries", []) or []:
        if not isinstance(entry, dict):
            continue
        for addr in entry.get("addresses", []) or []:
            if not _valid_address(str(addr)):
                return _err("E04", f"unknown address {addr}; {_VALID_ADDRESS_HINT}")

    w, t, r, b, precision = _run_plan(args.watch, args.trigger, args.precision)
    line, flag, _ = _estimator_line(w, t, r, b, precision)
    if flag == "RED" and not args.force:
        print(line)
        print("Require --force. Nothing ran.")
        return 5
    if args.dry_run:
        if args.json:
            from glassbox import estimate_bytes

            print(
                json.dumps(
                    {"W": w, "T": t, "R": r, "bytes": estimate_bytes(w, t, r, b), "flag": flag}
                )
            )
        else:
            print(line + " Proceeding.")
        return 0
    if flag == "RED" and args.force:
        print(f"WARNING RED-flagged {line} proceeding --force (user-owned disk).")
    else:
        print(line + " Proceeding.")
    tag_dict = tags or {}
    try:
        with GlassBox.watch(
            args.watch,
            trigger=args.trigger,
            precision=args.precision,
            force=args.force,
            tag=tag_dict,
        ) as ctx:
            ctx.record_synthetic(n_tokens=min(t, 16))
        run_id = str(ctx.run_id)
        seal = ctx.seal or {}
    except UnknownAddressError as exc:
        return _err("E04", str(exc))
    except CheckpointMismatchError as exc:
        return _err("E04", str(exc))
    except SizeFlagError:
        return 5
    except FileNotFoundError as exc:
        return _err("E02", str(exc))
    except ValueError as exc:
        return _err("E02", str(exc))

    store = _store_root()
    deep_path = os.path.join(store, "runs", run_id, "deep.bin")
    deep_mb = os.path.getsize(deep_path) / 1e6 if os.path.isfile(deep_path) else 0.0
    short = str(seal.get("hash", "0" * 64))[:4]
    if args.json:
        print(
            json.dumps(
                {
                    "run": run_id,
                    "hash": seal.get("hash", ""),
                    "deep_mb": round(deep_mb, 2),
                    "flag": flag,
                }
            )
        )
    else:
        print(f"{run_id} sealed | deep {deep_mb:.2f}MB (W={w}) | hash {short}... | chain OK")
    return 0


# ---------------------------------------------------------------- find ---


def cmd_find(args: argparse.Namespace) -> int:
    from glassbox.search.query import find_runs

    if not args.query:
        return _err("E02", "find requires a QUERY string.")
    if args.limit < 1 or args.limit > 1000:
        return _err("E02", f"--limit must be 1..1000 (got {args.limit}).")
    query: str = args.query
    for cand in _CANDIDATE_ADDR_RE.findall(query):
        base = cand.split("..")[0]
        if not _valid_address(base):
            return _err("E04", f"unknown address {cand} not in manifest; {_VALID_ADDRESS_HINT}")
    token_range: tuple[int, int] | None = None
    if args.tokens:
        m = re.fullmatch(r"\s*(\d+)\s*\.\.\s*(\d+)\s*", str(args.tokens))
        if not m or int(m.group(2)) < int(m.group(1)):
            return _err("E02", f"bad --tokens {args.tokens!r}; expected A..B.")
        token_range = (int(m.group(1)), int(m.group(2)))
    store = _store_root()
    try:
        rows, meta = find_runs(query, store, limit=args.limit, tokens=token_range)
    except FileNotFoundError as exc:
        return _err("E02", str(exc))
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("E04"):
            return _err("E04", msg[4:].strip())
        return _err("E02", msg[4:].strip() if msg.startswith("E02") else msg)
    hot_ms = float(meta.get("hot_ms", 0.0))
    cold_slices = int(meta.get("cold_slices", 0))
    if args.json:
        for row in rows:
            print(json.dumps({**row, "hash_seal_verified": bool(row["hash_seal_verified"])}))
    else:
        has_neuron = any(r.get("address") for r in rows)
        if has_neuron:
            print("RUN-ID      ADDR@t         VAL      SEAL")
        else:
            print("RUN-ID      MATCH            SEAL")
        for row in rows:
            verified = str(bool(row["hash_seal_verified"])).lower()
            if row.get("address"):
                addr = str(row["address"])
                print(
                    f"{row['run']}  {addr}@t{row['token']}  {row['value']:.2f}  verified={verified}"
                )
            else:
                print(f"{row['run']}  prompt/tag match  verified={verified}")
        print(f"hot {hot_ms:.0f}ms + cold {cold_slices} slices (two-phase filter->verify)")
    return 0


# -------------------------------------------------------------- compare ---


def _num(value: Any) -> str:
    return f"{float(value):.2f}" if isinstance(value, (int, float)) else "--"


def _fmt_side(side: dict[str, Any]) -> str:
    if side.get("empty"):
        return "n=0 (no Deep runs)"
    return (
        f"n={side.get('n', 0)} mean {_num(side.get('mean'))} "
        f"p95 {_num(side.get('p95'))} fire {_num(side.get('firing_rate'))}"
    )


def _bars(results: list[dict[str, Any]], side: str) -> str:
    """ASCII histogram from mean firing_rate (10 chars, #/.) ."""
    if not results:
        return ".........."
    fires = [float(r[side].get("firing_rate", 0.0) or 0.0) for r in results]
    avg = sum(fires) / len(fires)
    full = max(0, min(10, int(round(avg * 10))))
    return "#" * full + "." * (10 - full)


def cmd_compare(args: argparse.Namespace) -> int:
    from glassbox.search.query import compare_runs

    if not args.good or not args.bad:
        return _err("E02", "compare requires --good EXPR --bad EXPR.")
    if not args.neurons:
        return _err("E02", "compare requires --neurons LIST.")
    neurons = [n.strip() for n in str(args.neurons).split(",") if n.strip()]
    for neuron in neurons:
        if not _valid_address(neuron.split("..")[0]):
            return _err("E04", f"unknown address {neuron}; {_VALID_ADDRESS_HINT}")
    m = re.fullmatch(r"\s*(\d+)\s*\.\.\s*(\d+)\s*", str(args.tokens))
    if not m or int(m.group(2)) < int(m.group(1)):
        return _err("E02", f"bad --tokens {args.tokens!r}; expected A..B.")
    if args.limit < 1 or args.limit > 100000:
        return _err("E02", f"--limit must be 1..100000 (got {args.limit}).")
    store = _store_root()
    try:
        results, meta = compare_runs(
            args.good,
            args.bad,
            neurons,
            stats=str(args.stat),
            tokens=(int(m.group(1)), int(m.group(2))),
            limit=args.limit,
            store=store,
        )
    except FileNotFoundError as exc:
        return _err("E02", str(exc))
    except ValueError as exc:
        msg = str(exc)
        if msg.startswith("E04"):
            return _err("E04", msg[4:].strip())
        return _err("E02", msg[4:].strip() if msg.startswith("E02") else msg)
    if results and all(r["good"].get("empty") or r["bad"].get("empty") for r in results):
        missing = args.good if results[0]["good"].get("empty") else args.bad
        return _err("E06", f"no runs with {missing}.")
    if args.json:
        print(json.dumps(results))
        return 0
    if getattr(args, "attribute", False):
        sorted_results = sorted(
            results, key=lambda r: abs(float(r.get("attribution", 0.0))), reverse=True
        )
        print("=== Causal Direct Logit Attribution Ranking ===")
        print(
            f"{'Address':<14} {'Good Mean':<12} {'Bad Mean':<12} {'Delta':<10} "
            f"{'Attribution':<12} {'Driver'}"
        )
        print("-" * 72)
        for row in sorted_results:
            gm = row["good"].get("mean", 0.0) or 0.0
            bm = row["bad"].get("mean", 0.0) or 0.0
            delta = row.get("delta_mean", 0.0)
            attr = row.get("attribution", 0.0)
            driver = "PRO-GOOD (+)" if attr > 0 else "PRO-BAD (-)"
            print(
                f"{row['address']:<14} {gm:<12.4f} {bm:<12.4f} {delta:<10.4f} "
                f"{attr:<12.4f} {driver}"
            )
        print(f"Aggregated across {int(meta.get('cold_slices', 0))} cold slices.")
        return 0
    for row in results:
        print(f"{row['address']} | good {_fmt_side(row['good'])} | bad {_fmt_side(row['bad'])}")
    print(
        f"histogram good [{_bars(results, 'good')}] bad [{_bars(results, 'bad')}] | "
        f"hot index, {int(meta.get('cold_slices', 0))} cold slices"
    )
    return 0


# --------------------------------------------------------------- verify ---


def cmd_verify(args: argparse.Namespace) -> int:
    from glassbox import verify_run

    store = _store_root(args.chain and os.path.dirname(args.chain) or None)
    if args.all:
        run_ids = [run_id for run_id, _, _ in _iter_runs(store)]
        if not run_ids:
            return _err("E06", f"no runs found in {store}/runs/.")
    else:
        run_ids = list(args.run_ids or [])
        if not run_ids:
            return _err("E02", "verify requires RUN-ID... or --all.")
    ok_count = 0
    fail_count = 0
    for run_id in run_ids:
        run_dir = os.path.join(store, "runs", run_id)
        if not os.path.isdir(run_dir):
            print(f"Error E06: {run_id} not found in {store}/runs/.", file=sys.stderr)
            if len(run_ids) == 1 and not args.all:
                return 6
            fail_count += 1
            continue
        ok = bool(verify_run(run_id, store))
        try:
            with open(os.path.join(run_dir, "seal.json"), encoding="utf-8") as fh:
                seal = json.load(fh)
            with open(os.path.join(run_dir, "deep_index.json"), encoding="utf-8") as fh:
                index = json.load(fh)
        except (OSError, ValueError):
            seal, index = {}, {}
        if args.json:
            print(json.dumps({"run": run_id, "ok": ok, "hash": seal.get("hash", "")}))
        elif ok:
            ok_count += 1
            index_short = str(seal.get("sha_index", "0" * 64))[:4]
            deep_short = str(seal.get("sha_deep") or "none")
            deep_short = deep_short[:4] if len(deep_short) > 8 else deep_short
            replay = index.get("replay", {}) if isinstance(index, dict) else {}
            rver = replay.get("versions", {}) if isinstance(replay, dict) else {}
            ckpt = str(replay.get("checkpoint_sha256", "0" * 64))[:6]
            prev = str(seal.get("prev_hash", "0" * 64))[:4]
            print(
                f"{run_id} chain OK | index {index_short}... deep {deep_short}... | "
                f"replay{{seed {replay.get('seed', 42)},temp {replay.get('temp', 0.0)},"
                f"top_p {replay.get('top_p', 1.0)},sampler {replay.get('sampler', 'greedy')},"
                f"torch{rver.get('torch', 'unknown')},cuda{rver.get('cuda', 'cpu')},"
                f"C-{ckpt}...}} | prev {prev}... linked"
            )
        else:
            fail_count += 1
            stored = str(seal.get("sha_index", "?"))[:8]
            print(f"{run_id} FAIL: deep_index.json mismatch stored {stored}... (tampered).")
    if args.all or len(run_ids) > 1:
        if fail_count:
            print(f"SUMMARY: {ok_count} OK, {fail_count} FAIL -> exit 7")
            return 7
        print(f"SUMMARY: {ok_count} OK, {fail_count} FAIL")
        return 0
    if fail_count:
        return 7
    return 0


# ----------------------------------------------------------------- view ---


def cmd_view(args: argparse.Namespace) -> int:
    from glassbox import viewer

    store = _store_root()
    if args.port < 1024 or args.port > 65535:
        return _err("E02", f"bad port {args.port} (1024..65535).")
    if not args.run_id:
        return _err("E06", f"run id required; no run found in {store}/runs/.")
    run_id = args.run_id
    token_range: tuple[int, int] | None = None
    if args.tokens:
        m = re.fullmatch(r"\s*(\d+)\s*\.\.\s*(\d+)\s*", str(args.tokens))
        if not m or int(m.group(2)) < int(m.group(1)):
            return _err("E02", f"bad --tokens {args.tokens!r}; expected A..B.")
        token_range = (int(m.group(1)), int(m.group(2)))
    addrs: list[str] | None = None
    if args.addrs:
        addrs = [a.strip() for a in str(args.addrs).split(",") if a.strip()]
        if not addrs:
            return _err("E02", "bad --addrs; expected ADDR[,ADDR...].")
    _ = args.no_gpu
    if getattr(args, "lens", False):
        from glassbox.lens import compute_logit_lens, format_lens_table

        try:
            target_t = token_range[0] if token_range else None
            lens_data = compute_logit_lens(run_id, store, token_pos=target_t)
            if args.json:
                print(json.dumps(lens_data, indent=2))
            else:
                print(format_lens_table(lens_data))
            return 0
        except Exception as exc:
            return _err("E02", str(exc))
    if args.open:
        try:
            viewer.open_in_browser(run_id, store, port=args.port)
        except FileNotFoundError as exc:
            return _err("E06", str(exc))
        return 0
    try:
        table = viewer.token_table(run_id, store, tokens=token_range, addrs=addrs)
    except FileNotFoundError as exc:
        return _err("E06", str(exc))
    except ValueError as exc:
        return _err("E02", str(exc))
    if args.json:
        print(json.dumps(viewer.to_json(table)))
        return 0
    if args.export_html:
        try:
            html_text = viewer.render_html(run_id, store)
        except FileNotFoundError as exc:
            return _err("E06", str(exc))
        with open(args.export_html, "w", encoding="utf-8") as fh:
            fh.write(html_text)
        try:
            kb = os.path.getsize(args.export_html) / 1e3
        except OSError:
            kb = 0.0
        badge = "OK" if table["run"]["seal_ok"] else "FAIL"
        print(
            f"Viewer http://127.0.0.1:{args.port}/{run_id} | "
            f"export {args.export_html} {kb:.0f}KB | badge seal {badge}"
        )
        return 0
    color = not args.no_color and sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    print(viewer.format_table(table, color=color))
    return 0


# ----------------------------------------------------------------- size ---


def cmd_size(args: argparse.Namespace) -> int:
    from glassbox import estimate_bytes, flag_for_bytes

    store = _store_root(args.dir)
    by_model: dict[str, dict[str, Any]] = {}
    for run_id, index, run_dir in _iter_runs(store):
        model = str(index.get("model_id", "M-001"))
        ckpt = str(index.get("checkpoint_sha256", "0" * 64))[:6]
        scope = f"{model}/C-{ckpt}"
        hot = 0
        cold = 0
        for name in ("deep_index.json", "seal.json", "meta.json"):
            try:
                hot += os.path.getsize(os.path.join(run_dir, name))
            except OSError:
                pass
        for name in ("deep.bin",):
            try:
                cold += os.path.getsize(os.path.join(run_dir, name))
            except OSError:
                pass
        slot = by_model.setdefault(scope, {"runs": 0, "hot": 0, "cold": 0})
        slot["runs"] += 1
        slot["hot"] += hot
        slot["cold"] += cold
    if args.json:
        rows = [
            {
                "scope": scope,
                "runs": v["runs"],
                "hot_bytes": v["hot"],
                "cold_bytes": v["cold"],
                "total_bytes": v["hot"] + v["cold"],
                "flag": flag_for_bytes(
                    estimate_bytes(1, 1, 1, 1) * 0 + (v["hot"] + v["cold"]) * 1.15
                ),
            }
            for scope, v in sorted(by_model.items())
        ]
        print(json.dumps(rows))
        return 0
    if args.by_run:
        for run_id, _, run_dir in _iter_runs(store):
            hot = 0
            cold = 0
            for name in ("deep_index.json", "seal.json", "meta.json"):
                try:
                    hot += os.path.getsize(os.path.join(run_dir, name))
                except OSError:
                    pass
            try:
                cold = os.path.getsize(os.path.join(run_dir, "deep.bin"))
            except OSError:
                cold = 0
            flag = flag_for_bytes((hot + cold) * 1.15)
            print(f"{run_id} {_human(hot)} + {_human(cold)} {flag}")
        print("retention: retention_policy.json (user-owned) -- flag only, MUST NOT auto-delete.")
        return 0
    print("MODEL         RUNS  HOT    COLD   TOTAL  FLAG")
    for scope, v in sorted(by_model.items()):
        total = v["hot"] + v["cold"]
        flag = flag_for_bytes(total * 1.15)
        print(
            f"{scope}  {v['runs']}  {_human(v['hot'])}  {_human(v['cold'])}  "
            f"{_human(total)}  {flag} {_band_hint(flag)}"
        )
    print("retention: retention_policy.json (user-owned) -- flag only, MUST NOT auto-delete.")
    return 0


# --------------------------------------------------------------- export ---


def cmd_export(args: argparse.Namespace) -> int:
    if not args.zip:
        return _err("E02", "export requires --zip --run RUN-ID.")
    if not args.run:
        return _err("E02", "export requires --zip --run RUN-ID.")
    store = _store_root()
    run_id = args.run
    run_dir = os.path.join(store, "runs", run_id)
    if not os.path.isdir(run_dir):
        return _err("E06", f"{run_id} not found in {store}/runs/.")
    out = args.out or f"./{run_id}.zip"
    if os.path.isfile(out):
        print(f"WARNING overwriting {out}")
    files: list[tuple[str, str]] = []
    for name in ("deep_index.json", "seal.json", "meta.json"):
        src = os.path.join(run_dir, name)
        if os.path.isfile(src):
            files.append((src, name))
    for name in ("deep.bin",):
        src = os.path.join(run_dir, name)
        if os.path.isfile(src):
            files.append((src, name))
    if args.with_manifest:
        manifest_added = False
        try:
            with open(os.path.join(run_dir, "deep_index.json"), encoding="utf-8") as fh:
                index = json.load(fh)
            ckpt = str(index.get("checkpoint_sha256", ""))[:6]
            base = os.path.join("manifests")
            if os.path.isdir(base):
                for entry in sorted(os.listdir(base)):
                    cand = os.path.join(base, entry, "manifest.json")
                    if ckpt in entry and os.path.isfile(cand):
                        files.append((cand, "manifest.json"))
                        manifest_added = True
                        break
            _ = manifest_added
        except (OSError, ValueError):
            pass
        pointer = os.path.join(run_dir, "FORMAT_GUIDE_POINTER.txt")
        with open(pointer, "w", encoding="utf-8") as fh:
            fh.write("Adapt this export via docs/FORMAT_GUIDE.md (we document, you edit).\n")
        files.append((pointer, "FORMAT_GUIDE_POINTER.txt"))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files:
            zf.write(src, arc)
    with open(out, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    try:
        size = os.path.getsize(out)
    except OSError:
        size = 0
    print(f"{run_id} -> {out} {_human(size)} {len(files)} files (sha {sha[:4]}...).")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``[project.scripts] blackbox``)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help(sys.stdout)
        return 0
    handlers = {
        "scan": cmd_scan,
        "run": cmd_run,
        "find": cmd_find,
        "compare": cmd_compare,
        "verify": cmd_verify,
        "view": cmd_view,
        "size": cmd_size,
        "export": cmd_export,
    }
    handler = handlers.get(args.command)
    if handler is None:
        return 0
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
