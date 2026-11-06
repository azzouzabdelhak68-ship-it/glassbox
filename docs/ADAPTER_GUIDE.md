# ADAPTER_GUIDE — port the GPU backend in one file

> Scan is CPU-only (no port needed). Copy `gpu_adapters/nvidia.py` and swap
> the stream/pinning primitives; keep the `index_select` gather. One file
> per backend; core never changes. License: MIT — adapt freely.

## 6-function interface (`gpu_adapters/base.py` is the contract)

| Function | Signature | Contract |
|---|---|---|
| `capabilities` | `() -> dict` | `{backend, streams, pinned, events, dtypes, int8_dequant}` + `available: bool` on real backends. Never raises; reports `available=False` off-device so callers degrade to `cpu`. |
| `create_context` | `(watchlist: dict) -> dict` | Group addresses by parent module (see `tap/hooks.group_by_parent`), move `idx` to the device **once**, allocate double-buffered pinned host memory, one copyStream + Event per group. |
| `attach` | `(model, ctx) -> handles` | `register_forward_hook` **only** on watched parent modules (never per neuron). Returns handles carrying the recorder. |
| `detach` | `(handles) -> None` | Idempotent, never raises; restores the model byte-identical. Hook throw must auto-`detach()`. |
| `alloc_pinned` | `(nbytes: int) -> buffer` | Pinned/page-locked host staging (double-buffer backing store). |
| `stage_copy` | `(gathered, pinned, stream) -> None` | Async device→pinned copy on the group's copyStream (`non_blocking`), overlapping the next MatMul (~80% hidden). Default stream is never blocked. |

Invariants (gates fail otherwise): gather is `activation.index_select` on the
last dim, watched columns only; int8 weights dequantize **on device**
(`gathered * scale`) before the copy — the sealed value is logical, with
`dtype_meta`; ring-full sets `ring_overflow` and drops Deep (never stalls);
`grep synchronize tap/hooks.py` must be empty; no `tensor.cpu()` in hooks.

## Backend notes

- **`nvidia.py` (priority #1, Day 1):** `torch.cuda.Stream` per group,
  `torch.cuda.Event` per group, `tensor.pin_memory()`. Requires
  Linux + Docker `--gpus all` + `--shm-size=1g`, driver 535+, CUDA 12.1–13.2,
  `torch>=2.4`. Import `torch.cuda` lazily inside functions so CPU-only hosts
  import the module cleanly (`capabilities()` reports `available=False`).
- **`cpu.py` (fallback, works everywhere):** `index_select` on CPU tensors,
  queued `clone()` into a bytearray (~0.005 ms/1k). Used by
  `GLASSBOX_GPU=cpu pytest`, Deep-only hosts, Windows/Mac.
- **`amd.py` (community port):** swap `torch.cuda.Stream/Event` for HIP
  streams (`torch.cuda` API is HIP-compatible under ROCm in most builds;
  otherwise use `torch.hip`). Keep gather + double buffering unchanged.
  Expected delta: ~30 lines.
- **`apple.py` (community stub):** MPS has no streams/pinning; use
  `torch.mps` synchronized blit per group on the test path and document the
  measured overhead honestly. Deep on Apple is best-effort, never a Day-1
  performance claim.

## Port procedure (good first issue, no SLA)

1. Copy `nvidia.py` → `<backend>.py`; implement the 6 functions.
2. Test CPU-first: `GLASSBOX_GPU=cpu pytest -q` (your backend must not break it).
3. Bench: `./benchmark.sh --mode deep --watch watchlists/pilot-sparse-1k.yaml
   --tokens 500 --runs 100` → commit the log under
   `benchmarks/community/<date>-<gpu>-<model>-W<w>/` (`result.json` + `env.txt`
   with `nvidia-smi`, `torch.__version__`, `uname -a`).
4. PR with bench numbers; estimator flags (GREEN <50GB / YELLOW / RED >1000GB)
   are unchanged — only the copy primitive moves.

## Search adapters (separate seam, same idea)

`search/index.py::hot_scan` + `cold_verify_one` are the replaceable pair for
community DB ports (SQLite/DuckDB-already-default/anything): Hot returns a
candidate **superset** from aggregates, Cold decides exact via
`deep_index.json` offsets. Never implement one-row-per-token-per-neuron
tables (5B rows / 400GB index for 10k×500×1k — rejected by design).
