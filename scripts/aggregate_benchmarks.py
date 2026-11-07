"""Aggregates community benchmark results and updates leaderboard stats.

Usage:
    python scripts/aggregate_benchmarks.py --data-dir DIR --output BENCHMARKS.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any


def load_benchmark_records(data_dir: str) -> list[dict[str, Any]]:
    """Loads all benchmark JSON and JSONL records from a directory."""
    records: list[dict[str, Any]] = []
    if not os.path.isdir(data_dir):
        return records

    # Search for json and jsonl files
    json_files = glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True)
    jsonl_files = glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)

    for path in json_files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    records.append(data)
                elif isinstance(data, list):
                    records.extend([d for d in data if isinstance(d, dict)])
        except Exception:
            continue

    for path in jsonl_files:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception:
            continue

    return records


def calculate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculates global performance averages and hardware breakdowns."""
    if not records:
        return {
            "total_runs": 0,
            "avg_overhead_pct": 0.0,
            "avg_baseline_tps": 0.0,
            "avg_glassbox_tps": 0.0,
            "devices": {},
        }

    overheads: list[float] = []
    devices: dict[str, list[float]] = {}
    baseline_tps_list: list[float] = []
    glassbox_tps_list: list[float] = []

    for r in records:
        # Extract overhead
        overhead = r.get("glassbox_overhead_percent")
        if overhead is None:
            overhead = r.get("overhead_pct")
        if overhead is not None:
            try:
                overheads.append(float(overhead))
            except (ValueError, TypeError):
                pass

        # Extract device
        dev = r.get("device", "Unknown")
        if dev not in devices:
            devices[dev] = []
        if overhead is not None:
            try:
                devices[dev].append(float(overhead))
            except (ValueError, TypeError):
                pass

        # Extract throughputs from summary list if present
        summary = r.get("summary")
        if isinstance(summary, list) and len(summary) >= 3:
            try:
                b_tps = summary[0].get("Throughput (tok/s)")
                g_tps = summary[2].get("Throughput (tok/s)")
                if b_tps is not None:
                    baseline_tps_list.append(float(b_tps))
                if g_tps is not None:
                    glassbox_tps_list.append(float(g_tps))
            except (ValueError, TypeError):
                pass
        else:
            b_tps = r.get("baseline_tok_s")
            g_tps = r.get("glassbox_tok_s")
            if b_tps is not None:
                baseline_tps_list.append(float(b_tps))
            if g_tps is not None:
                glassbox_tps_list.append(float(g_tps))

    device_stats = {}
    for dev, o_list in devices.items():
        if o_list:
            device_stats[dev] = {
                "count": len(o_list),
                "avg_overhead": round(sum(o_list) / len(o_list), 2),
            }

    return {
        "total_runs": len(records),
        "avg_overhead_pct": round(sum(overheads) / len(overheads), 2) if overheads else 0.0,
        "avg_baseline_tps": (
            round(sum(baseline_tps_list) / len(baseline_tps_list), 2) if baseline_tps_list else 0.0
        ),
        "avg_glassbox_tps": (
            round(sum(glassbox_tps_list) / len(glassbox_tps_list), 2) if glassbox_tps_list else 0.0
        ),
        "devices": device_stats,
    }


def generate_markdown_report(metrics: dict[str, Any]) -> str:
    """Formats aggregated metrics into a clean markdown leaderboard."""
    lines = [
        "# 🏆 GlassBox Community Benchmark Leaderboard",
        "",
        "Aggregated performance benchmarks submitted by community members testing GlassBox.",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Recorded Community Runs:** {metrics['total_runs']}",
        (
            f"- **Global Average GlassBox Overhead:** **{metrics['avg_overhead_pct']}%** "
            "(Target: <1.0%)"
        ),
    ]

    if metrics["avg_glassbox_tps"] > 0:
        lines.append(f"- **Average Inference Speed:** {metrics['avg_glassbox_tps']} tokens/sec")

    lines.extend(
        [
            "",
            "## Hardware Breakdown",
            "",
            "| Hardware Device | Submissions | Average Overhead (%) | Verification Status |",
            "| :--- | :--- | :--- | :--- |",
        ]
    )

    if metrics["devices"]:
        sorted_devs = sorted(metrics["devices"].items(), key=lambda x: x[1]["count"], reverse=True)
        for dev, stats in sorted_devs:
            status = "✅ Sub-1% Verified" if stats["avg_overhead"] < 1.0 else "⚡ Measured"
            lines.append(
                f"| **{dev}** | {stats['count']} | {stats['avg_overhead']:+.2f}% | {status} |"
            )
    else:
        lines.append("| *Awaiting community submissions* | 0 | N/A | Pending |")

    lines.extend(
        [
            "",
            "---",
            "",
            "### How to Submit Your Run:",
            (
                "1. Open the [Google Colab Quickstart]"
                "(https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/"
                "notebooks/quickstart_demo.ipynb)."
            ),
            "2. Run all cells to benchmark on your hardware (free GPU or CPU).",
            "3. Click **💬 Share to Discussions** in Cell 11 and drag your bundle into the post!",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate GlassBox community benchmarks")
    parser.add_argument(
        "--data-dir", default="benchmarks/community", help="Directory of benchmark records"
    )
    parser.add_argument("--output", default="BENCHMARKS.md", help="Output markdown file")
    args = parser.parse_args()

    records = load_benchmark_records(args.data_dir)
    metrics = calculate_metrics(records)
    md_content = generate_markdown_report(metrics)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md_content)
        total = metrics["total_runs"]
        avg_oh = metrics["avg_overhead_pct"]
        print(f"[OK] Benchmark report written to {args.output} (Runs: {total}, Avg: {avg_oh}%)")
    else:
        print(md_content.encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
