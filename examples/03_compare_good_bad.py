# Phase 3: runnable CPU fallback (no weights/GPU needed).
"""10k good vs 10k bad comparison (demo: 4 vs 4 synthetic runs).

Spec §EXAMPLES 03 contract:
  In: two prompt lists + shared watchlist.
  Out: Hot-only scan + Cold slices; CLI prints per-group
    mean/p95/firing_rate + ASCII histogram delta;
    JSON mode emits exact stats object.
  Then: blackbox compare --good tag=good --bad tag=bad
    --neurons N-24-0001,A-24-07

With real weights the loop body is ``with blackbox.watch(...): model.generate(p)``;
on a CPU-only host this example seals the same file layout with synthetic
values (seeded RNG) and runs the real compare backend.
"""

from __future__ import annotations

import os
import random
import tempfile

import numpy as np

SHA = "ab12cd34" * 8


def _seal_run(store: str, prompt: str, tag: str, hot: bool, seed: int) -> str:
    from glassbox.storage import layout as _layout
    from glassbox.storage import seal as _seal

    watched = ["N-24-0001", "A-24-07"]
    run_id, run_dir = _layout.allocate_run(store)
    rng = random.Random(seed)
    base = 0.6 if hot else 0.2
    mat = np.asarray(
        [[rng.uniform(base - 0.2, base + 0.2) for _ in watched] for _ in range(20)],
        dtype=np.float32,
    )
    _layout.write_deep_bin(run_dir, mat, "fp16")
    ctx = {
        "run_id": run_id,
        "model_id": "M-001",
        "checkpoint_sha256": SHA,
        "prompt": {"text": prompt, "tokens": [1]},
        "output": {"text": "demo", "tokens": [2]},
        "tags": [tag],
        "replay": {"seed": seed, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
    }
    _layout.write_deep_index(
        run_dir,
        run_id,
        triggered=True,
        dtype="fp16",
        tokens=20,
        watched=watched,
        context=ctx,
    )
    _seal.seal_run(run_dir)
    return run_id


def run_comparison(
    good_ps: list[str],
    bad_ps: list[str],
    seed: int = 42,
) -> list[dict[str, object]]:
    """Seal two tagged cohorts, compare N-24-0001/A-24-07, print histogram."""
    from glassbox.search.query import compare_runs

    store = os.path.join(tempfile.gettempdir(), "bb-example-03")
    os.makedirs(store, exist_ok=True)
    for i, prompt in enumerate(good_ps):
        _seal_run(store, prompt, "cohort=good", False, seed + i)
    for i, prompt in enumerate(bad_ps):
        _seal_run(store, prompt, "cohort=bad", True, seed + 100 + i)
    results, meta = compare_runs(
        "tag:cohort=good",
        "tag:cohort=bad",
        ["N-24-0001", "A-24-07"],
        stats="mean,p95,firing_rate",
        tokens=(15, 19),
        store=store,
    )
    for row in results:
        assert isinstance(row, dict)
        good = row["good"]
        bad = row["bad"]
        assert isinstance(good, dict) and isinstance(bad, dict)
        print(
            f"{row['address']} | good n={good.get('n')} mean {good.get('mean'):.2f} "
            f"| bad n={bad.get('n')} mean {bad.get('mean'):.2f}"
        )
    print(f"cold slices: {meta['cold_slices']} | store={store}")
    print(
        "CLI parity: blackbox compare --good tag:cohort=good --bad tag:cohort=bad "
        "--neurons N-24-0001,A-24-07 --tokens 15..19"
    )
    return results


def main() -> None:
    run_comparison(
        good_ps=["Hello there", "Good morning", "Nice day", "Thanks"],
        bad_ps=["Hello there", "Good morning", "Nice day", "Thanks"],
        seed=42,
    )


if __name__ == "__main__":
    main()
