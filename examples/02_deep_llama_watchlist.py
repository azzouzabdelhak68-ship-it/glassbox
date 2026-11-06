# Phase 2: runnable CPU fallback (no weights/GPU needed).
"""Deep on Llama-3-8B (Linux+Docker+NVIDIA for real weights).

Spec EXAMPLES 02 contract:
  In: local weights + watchlist (pilot-sparse-1k + N-24-0001, A-17-04).
  Out: outer.json + deep.bin (~1 MB for W=1k x T~=64 fp16)
    + deep_index.json + seal.json;
    `blackbox find 'N-24-0001 AT token=17 > 0.8'` returns run;
    viewer Token 17 shows N-24-0001:0.92 blue.

On a CPU-only host this example runs the same code path with synthetic
values (seeded RNG, never real activations): DeepTap gather -> deep.bin
row-major [T, W] LE -> seal -> verify -> dataframe slice.
"""

from __future__ import annotations

import os
import random
import tempfile

import numpy as np


def run_deep_llama(prompt: str, seed: int = 42) -> str:
    """Deep-gather demo; returns the sealed run id."""
    from glassbox.storage import layout as _layout
    from glassbox.storage import seal as _seal
    from glassbox.tap.hooks import DeepTap

    _ = prompt
    watched = ["N-24-0001", "A-17-04"] + [f"N-24-{i:04d}" for i in range(2, 1001)]
    tap = DeepTap(
        watched,
        window={"start_token": 0, "end_token": 64, "every_n": 1},
        trigger="always",
        dtype="fp16",
    )
    rng = random.Random(seed)
    for pos in range(64):
        tap._staging = {c: rng.uniform(-1.0, 1.0) for c in range(len(watched))}
        if pos == 17:
            tap._staging[0] = 0.92  # N-24-0001 at token 17 (spec viewer value)
            tap._staging[1] = 0.83
        tap.end_token(pos)
    store = os.path.join(tempfile.gettempdir(), "bb-example-02")
    os.makedirs(store, exist_ok=True)
    run_id, run_dir = _layout.allocate_run(store)
    _layout.write_deep_bin(run_dir, np.asarray(tap.ring, dtype=np.float32), "fp16")
    ctx = {
        "run_id": run_id,
        "model_id": "M-001",
        "prompt": {"text": "Explain why the sky is blue", "tokens": [1, 2, 3]},
        "output": {"text": "demo", "tokens": [4, 5]},
        "replay": {"seed": seed, "temp": 0.0, "top_p": 1.0, "sampler": "greedy"},
    }
    _layout.write_deep_index(
        run_dir,
        run_id,
        triggered=True,
        dtype="fp16",
        tokens=64,
        watched=watched,
        context=ctx,
    )
    _seal.seal_run(run_dir)
    assert _seal.verify_run(run_id, store) is True
    deep_mb = os.path.getsize(os.path.join(run_dir, "deep.bin")) / 1e6
    print(f"{run_id} sealed | deep {deep_mb:.1f}MB | chain OK | store={store}")
    print("hint: blackbox find 'N-24-0001 AT token=17 > 0.8' (needs Phase-3 index)")
    return run_id


def main() -> None:
    run_deep_llama("Explain why the sky is blue", seed=42)


if __name__ == "__main__":
    main()
