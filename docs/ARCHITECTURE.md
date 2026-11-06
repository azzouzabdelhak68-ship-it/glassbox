# ARCHITECTURE — generated from `glassbox/`

> Every module below exists in this repo.

## Data flow (one run, front to back)

```
weights/ ──scan.py──▶ manifests/<M>_<sha8>/manifest.json   (CPU, 2–10s, once)
watchlists/*.yaml ──watchlist.py──▶ addresses[]
        │ controller.py: validate vs manifest (E04) → estimate (GREEN/YELLOW/RED)
        ▼
model.generate() ──glassbox.watch──▶ prompt/tokens/timing/seed
        │ trigger armed? ──tap/hooks.py──▶ index_select watched cols (GPU)
        │                      │ copyStream → pinned (non-blocking, async)
        ▼                      ▼
ring (/dev/shm/gb.ring, triple buffer; tmp fallback on CPU hosts)
        │ tap/sidecar.py: poll → WAL append + fsync/token → seal
        ▼
glassbox_store/runs/RUN-00000N/{deep.bin, deep_index.json, seal.json}
        + chain.jsonl (hash_i = SHA256(prev||SHA256(deep_index)||SHA256(deep)||canonical(replay)))
        ▼
search/index.py (Hot shards: min/max/mean/p95/firing_rate/win10)
        ▼
search/query.py (Hot filter → Cold mmap verify) → cli find/compare (with DLA)
viewer.py → cli view (table/JSON/static HTML SVG matrix) · to_dataframe.py (vectorized DataFrame)
lens.py → cli view --lens / glassbox.lens() (Logit Lens residual stream analysis)
profile.py → glassbox.profile() (Automated calibration profiler)
```

## Module map (file → role)

| Module | Role |
|---|---|
| `__init__.py` | `GlassBox.watch/scan/verify/lens/profile`, `to_dataframe`, error types, backward aliases |
| `__main__.py` | `python -m glassbox` entry (`raise SystemExit(main())`) |
| `cli.py` | 8 subcommands, argparse only, single-line errors, exits 0/2/3/4/5/6/7 |
| `scan.py` | CPU header read (`config.json` + safetensors header) → `manifest.json` |
| `watchlist.py` | YAML DSL + 7 presets + `extends`/`add` patch + validation (SAE/F/N/A/L/LOGITS) |
| `controller.py` | triggers (`always`/`on_flag`/`window`/address gate), window gating, estimator `W·T·R·B·1.15` |
| `exceptions.py` | `ManifestExistsError/UnknownAddressError/CheckpointMismatchError/SizeFlagError/RunNotFoundError/SealBrokenError` |
| `tap/hooks.py` | **only hook code**: per-parent `register_forward_hook`, `index_select` + async gather; ring-full→flag, throw→auto-detach |
| `tap/sidecar.py` | mmap triple ring → WAL + per-token fsync → seal; `attach`+`seal_prefix` crash recovery |
| `gpu_adapters/{base,nvidia,cpu,amd,apple}.py` | 6-func port contract; CUDA priority, CPU fallback, HIP/MPS community stubs |
| `storage/layout.py` | run alloc, `deep.bin` (dense + chunked sparse), `deep_index.json`, mmap slice reads |
| `storage/seal.py` | `seal_run` + `verify_run` (cryptographic SHA-256 seal + chain walk) |
| `search/index.py` | Hot shards, parallel `hot_scan`, `reindex` |
| `search/query.py` | EBNF grammar, two-phase `find_runs`, `compare_runs` with Direct Logit Attribution |
| `to_dataframe.py` | Vectorized NumPy array construction for ultra-fast conversion to DataFrames |
| `viewer.py` | Standalone single-file HTML (Calm Viewer) with SVG heatmaps and circuit diagrams |
| `lens.py` | Logit Lens and residual stream projection engine |
| `profile.py` | Automated activation calibration and watchlist generator |
| `benchmark.sh` | Gather methodology, JSON benchmark keys |

## Key invariants (gates enforce)

- Validate watchlist vs manifest **before** any GPU work (E04, nothing sealed).
- Hooks never block: ring-full → `ring_overflow`, throw → auto-`detach()`; job never killed.
- Never auto-delete / auto-redact: flag only (`retention_state.json`), user-owned policy.
- Raw save for research; `redactor=` is a caller-owned hook point, never a shipped filter.
- Partial-but-precise: unwatched renders as null/`--`/grey, never interpolated.
- `product.md` is never edited; `spec.md` changes need owner `apply`.
