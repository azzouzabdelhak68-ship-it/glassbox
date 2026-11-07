"""Unit tests for the community benchmark aggregator script."""

import json

from scripts.aggregate_benchmarks import (
    calculate_metrics,
    generate_markdown_report,
    load_benchmark_records,
)


def test_calculate_metrics_empty():
    metrics = calculate_metrics([])
    assert metrics["total_runs"] == 0
    assert metrics["avg_overhead_pct"] == 0.0
    assert metrics["devices"] == {}


def test_calculate_metrics_and_markdown(tmp_path):
    records = [
        {
            "model": "Qwen/Qwen2.5-0.5B",
            "device": "Tesla T4",
            "overhead_pct": 0.72,
            "baseline_tok_s": 42.0,
            "glassbox_tok_s": 41.7,
        },
        {
            "model": "Qwen/Qwen2.5-0.5B",
            "device": "Tesla T4",
            "overhead_pct": 0.88,
            "baseline_tok_s": 43.0,
            "glassbox_tok_s": 42.6,
        },
        {
            "model": "Qwen/Qwen2.5-0.5B",
            "device": "Apple M2 Max",
            "glassbox_overhead_percent": 0.45,
            "summary": [
                {"Throughput (tok/s)": 35.0},
                {"Throughput (tok/s)": 24.0},
                {"Throughput (tok/s)": 34.8},
            ],
        },
    ]

    # Save to temp directory
    rec_file = tmp_path / "run_01.json"
    with open(rec_file, "w", encoding="utf-8") as f:
        json.dump(records, f)

    loaded = load_benchmark_records(str(tmp_path))
    assert len(loaded) == 3

    metrics = calculate_metrics(loaded)
    assert metrics["total_runs"] == 3
    # Average of 0.72, 0.88, 0.45 = 2.05 / 3 = 0.68
    assert metrics["avg_overhead_pct"] == 0.68
    assert "Tesla T4" in metrics["devices"]
    assert metrics["devices"]["Tesla T4"]["count"] == 2
    assert metrics["devices"]["Tesla T4"]["avg_overhead"] == 0.80

    report = generate_markdown_report(metrics)
    assert "# 🏆 GlassBox Community Benchmark Leaderboard" in report
    assert "Tesla T4" in report
    assert "Apple M2 Max" in report
    assert "Sub-1% Verified" in report
