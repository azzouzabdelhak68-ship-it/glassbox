# PRESETS — watchlist macros that expand to exact addresses

> Presets are address lists only (no meaning). They expand deterministically
> against the scan manifest, so the same preset on the same checkpoint always
> yields the same addresses. Mix without editing: `extends: pilot-sparse-1k`
> + `watch: [N-24-0001]`. Per-run tweaks via `GlassBox.watch(..., add=[...])`
> seal as `deep_index.json:watchlist_ref.patch`. Unknown addresses fail fast (E04)
> before any GPU work; nothing seals.

## The 7 presets

| Preset | Expands to (Llama-3-8B: 32L, 32H, 14336 MLP) | W | 10k-run fp16 est. (T=500) |
|---|---|---|---|
| `outer_only` | no Deep entries; Outer always | 0 | ~30 MB (3 KB/run) GREEN |
| `last-4-layers-full` | L-29..32 all heads + all MLP + LOGITS | 57,473 | ~660 GB YELLOW |
| `attention_only` | all heads, no MLP (1/14th cost) | 1,024 | ~12 GB GREEN |
| `mid-mlp-wide` | L-12..20 every 10th token (`every_n: 10`) | 129,024 addrs, T/10 | ~1.5 TB RED, needs `--force` |
| `logit-lens` | `L-01..32` + NORM + LOGITS (residual stream) | 34 | ~MBs GREEN |
| `pilot-sparse-1k` | 250 seeded neurons each from L08/L16/L24/L31 | 1,000 | ~11.5 GB GREEN |
| `ioi-circuit-starter` | `A-17..24-04/07/12` + `N-20..24-0001..0200` (curated, stored neutrally) | 1,024 | ~12 GB GREEN |

Formula: `bytes = W × T × R × B × 1.15`. `mid-mlp-wide` carries an
`every_n: 10` window hint, so its effective T is T/10 (already applied above).

## Exact expansion rules (match `glassbox/watchlist.py`)

- Layer selections **clamp to the manifest**: on a 4-layer test manifest,
  `last-4-layers-full` yields that manifest's last 4 layers, not L-29..32.
  Small manifests are for CPU tests; full manifests give full sizes.
- `pilot-sparse-1k` samples with `random.Random(42)`, `min(250, mlp)` neurons
  per layer, sorted. Same checkpoint → same 1,000 addresses, always.
- `ioi-circuit-starter` keeps heads `04/07/12` that exist (`1 <= h <= heads`)
  and the first `min(200, mlp)` neurons of each of L-20..24.
- Ranges (`N-24-0001..0100`) are inclusive; dedupe preserves first-seen order.
- `extends:` accepts one preset or a list, plus extra `watch: [...]`
  addresses. File `precision`/`trigger`/`window` can be overridden per run;
  call kwargs win over file, file wins over preset hint, builtins
  (`always` / all-tokens / `fp16`) lose to everything.

## Examples

```yaml
# watchlists/pilot-sparse-1k.yaml
version: 1
model_id: M-001
checkpoint_sha256: ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34ab12cd34
watchlist_id: pilot-sparse-1k-v1
trigger: always
window: { start_token: 0, end_token: 511, every_n: 1 }
precision: fp16
entries:
  - preset: pilot-sparse-1k
  - addresses: [A-02-04, N-02-0001]
```

```python
# same watchlist for 10k comparable runs + one probed run (sealed as patch)
for p in prompts:
    with GlassBox.watch("watchlists/cohort-A.yaml", tag={"cohort": "good"}):
        model.generate(p, seed=42)
with GlassBox.watch("cohort-A.yaml", add=["N-24-9999"],
                     window={"start_token": 10, "end_token": 50}):
    model.generate("Probe prompt.", seed=42)
```

## Failure modes (fail fast, nothing sealed)

- `E04 unknown address N-99-9999 ... Valid: N-01-0001..N-32-14336, ...` — typo
  or stale manifest (one weight bit changes the checkpoint sha; old
  watchlists are invalid → re-`scan`).
- `E05 RED ... require --force` — estimate >1000 GB. Reduce W, set
  `every_n: 10`, narrow the window, or pass `--force` deliberately.
