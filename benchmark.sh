#!/usr/bin/env bash
x="""
set -euo pipefail
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs|--tokens|--watch) ARGS+=("$1" "${2:-}"); shift 2;;
    --mode) ARGS+=("$1" "${2:-}"); shift 2;;
    -h|--help) echo "Usage: benchmark.sh [--watch FILE] [--tokens N] [--runs N]"; exit 0;;
    *) echo "benchmark.sh: unknown arg $1"; exit 0;;
  esac
done
exec python3 "$0" "${ARGS[@]}"
exit 0
# """
# Python implementation (cross-platform, runs on Linux/macOS/Windows).
# Deep-Only Architecture Benchmark:
# Measures DeepTap gather, float packing, and persistence for W watched x T tokens.
import datetime
import json
import os
import platform
import random
import subprocess
import sys
import tempfile
import time


def _torch_version() -> str:
    try:
        from importlib.metadata import version

        return str(version("torch")).split("+")[0]
    except Exception:
        return "unknown"


def _gpu_name() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        if proc.returncode == 0 and lines:
            return lines[0]
    except Exception:
        pass
    return "none"


def _flag(total_bytes: float) -> str:
    if total_bytes < 50e9:
        return "GREEN"
    if total_bytes <= 1000e9:
        return "YELLOW"
    return "RED"


def bench_deep(watch: str | None, tokens: int, runs: int) -> int:
    """Deep gather bench (synthetic RNG values, universal CPU execution).

    Measures DeepTap gather+encode for W watched x T tokens, writes one
    deep.bin to a temp dir to time persistence, then projects R runs via
    the estimator (bytes=W*T*R*B*1.15). Prints JSON keys per spec §BENCH,
    appends benchmarks/community/RESULTS.md.
    """
    from glassbox.tap.hooks import DeepTap

    precision = "fp16"
    b = 2
    watched: list[str] = []
    if watch and os.path.isfile(watch):
        try:
            from glassbox.watchlist import parse_watchlist

            full_man = {
                "model_id": "M-001",
                "checkpoint_sha256": "0" * 64,
                "architecture": {
                    "layers": 32,
                    "hidden": 4096,
                    "heads": 32,
                    "mlp_per_layer": 14336,
                },
            }
            parsed = parse_watchlist(watch, manifest=full_man)
            watched = [str(a) for a in parsed.get("addresses", [])]
            precision = str(parsed.get("effective_precision", "fp16"))
            b = {"fp32": 4, "fp16": 2, "bf16": 2}.get(precision, 2)
        except Exception as exc:
            print("benchmark.sh: warning: watchlist not parsed: " + str(exc), file=sys.stderr)
    if not watched:
        watched = ["N-02-%04d" % i for i in range(1, 1001)]
    w = len(watched)
    tap = DeepTap(watched, window={"start_token": 0, "end_token": tokens, "every_n": 1})
    rng = random.Random(42)
    tap._staging = {c: rng.uniform(-1.0, 1.0) for c in range(w)}  # warmup
    tap.end_token(0)
    tap2 = DeepTap(watched, window={"start_token": 0, "end_token": tokens, "every_n": 1})
    t0 = time.perf_counter()
    for pos in range(tokens):
        tap2._staging = {c: rng.uniform(-1.0, 1.0) for c in range(w)}
        tap2.end_token(pos)
    ms_tapped = (time.perf_counter() - t0) / max(1, tokens) * 1000.0
    t0 = time.perf_counter()
    blob = tap2.to_bytes()
    ms_encode = (time.perf_counter() - t0) * 1000.0
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "RUN-000001")
        os.makedirs(run_dir)
        with open(os.path.join(run_dir, "deep.bin"), "wb") as fh:
            fh.write(blob)
        deep_bytes = os.path.getsize(os.path.join(run_dir, "deep.bin"))
    total_bytes = w * tokens * runs * b * 1.15
    flag = _flag(total_bytes)
    result = {
        "mode": "deep",
        "W": w,
        "T": tokens,
        "R": runs,
        "precision": precision,
        "overhead_pct": None,
        "ms_per_token_tapped": round(ms_tapped, 4),
        "ms_encode_total": round(ms_encode, 3),
        "ms_baseline": None,
        "ms_tapped": round(ms_tapped, 4),
        "deep_bin_bytes_per_run": deep_bytes,
        "GB": round(total_bytes / 1e9, 6),
        "total_GB": round(total_bytes / 1e9, 6),
        "flag": flag,
        "torch": _torch_version(),
        "cuda": "cpu",
        "gpu": _gpu_name(),
        "os": platform.system() + " " + platform.release(),
    }
    print(json.dumps(result))
    day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    run_dir = os.path.join("benchmarks", "community", day + "-cpu-deep-W" + str(w))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(run_dir, "env.txt"), "w", encoding="utf-8") as fh:
        fh.write("torch=" + result["torch"] + "\n")
        fh.write("gpu=" + result["gpu"] + "\n")
        fh.write("os=" + result["os"] + "\n")
        fh.write("python=" + platform.python_version() + "\n")
    row = (
        "| deep | "
        + str(w)
        + " | "
        + str(tokens)
        + " | "
        + str(runs)
        + " | "
        + precision
        + " | n/a | n/a | "
        + str(result["ms_per_token_tapped"])
        + " | "
        + str(result["total_GB"])
        + " | "
        + flag
        + " | "
        + result["torch"]
        + " | cpu | "
        + result["gpu"]
        + " | "
        + result["os"]
        + " |\n"
    )
    results_md = os.path.join("benchmarks", "community", "RESULTS.md")
    if os.path.isfile(results_md):
        with open(results_md, "a", encoding="utf-8") as fh:
            fh.write(row)
    return 0


def main(argv: list) -> int:
    runs = 10
    watch: str | None = None
    tokens = 500
    i = 0
    while i < len(argv):
        if argv[i] == "--mode" and i + 1 < len(argv):
            # Mode arg kept for CLI backwards compatibility
            i += 2
        elif argv[i] == "--runs" and i + 1 < len(argv):
            try:
                runs = max(1, int(argv[i + 1]))
            except ValueError:
                runs = 10
            i += 2
        elif argv[i] == "--watch" and i + 1 < len(argv):
            watch = argv[i + 1]
            i += 2
        elif argv[i] == "--tokens" and i + 1 < len(argv):
            try:
                tokens = max(1, int(argv[i + 1]))
            except ValueError:
                tokens = 500
            i += 2
        else:
            i += 1
    return bench_deep(watch, tokens, runs)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
