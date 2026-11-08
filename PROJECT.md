# Project: GlassBox Fast Production Serving & Distributed Multi-GPU Sharding

## Architecture
GlassBox expands from a single-GPU PyTorch instrument into a production-grade, distributed mechanistic interpretability engine:
- **Engine Backend Layer (`glassbox/backends/`)**: Uniform `BaseEngineAdapter` protocol isolating engine-specific hooks. `PyTorchEngineAdapter` preserves legacy execution. `VLLMEngineAdapter` injects into vLLM V1 worker subprocesses via `GlassBoxWorkerExtension`. `SGLangEngineAdapter` hooks SGLang SRT. `StepManifest` generates CPU metadata from continuous batching `InputBatch`.
- **Distributed Sharding Layer (`glassbox/tap/distributed.py`)**: `TPShardRouter` manages zero-NVLink tensor parallel coordination:
  - Residual Stream (`L-LL`, `NORM`, `EMBED`): Passive recording on Rank 0 only; Active steering on ALL TP ranks.
  - Attention Heads (`A-LL-HH`) & MLP Neurons (`N-LL-NNNN`): Modulo partitioned across TP ranks.
  - Logits Entropy (`LOGITS-ENTROPY`): Local $(m_r, s_r, w_r)$ 3-scalar reduction (12 B/tok) merged by collator for exact global Shannon entropy with zero NVLink communication.
- **In-Graph Wiretap & Acceleration Layer (`csrc/`, `glassbox/csrc_fallback.py`)**: Two-stage CUDA Graph-safe tap:
  - Stage 1: In-graph gather op `glassbox::tap` writing to pre-allocated static HBM staging slab governed by device control tensor `ctl` (kill switch `ctl[0]`, `actual_num_tokens` `ctl[1]`, slot `ctl[2]`).
  - Stage 2: Post-replay non-blocking D2H copy on `copyStream` with event synchronization.
  - Fallbacks: Pure-Python NumPy and Triton implementations in `glassbox/csrc_fallback.py` registered via `@torch.library.custom_op` ensuring 100% test pass on non-CUDA machines.
- **Storage & Collator Layer (`csrc/ring_buffer.h`, `crates/glassbox_collator/`, `glassbox/storage/collator.py`)**:
  - Wait-free SPSC pinned host ring buffer with fail-open `DROP_NEWEST` policy and drop tombstones.
  - High-speed Rust crate `glassbox-collator` writing 256MB size-rotated segmented WALs with `posix_fadvise(DONTNEED)`.
  - Virtual multi-rank indexing joining `(global_step_id, req_id)` across ranks without physical file merging.
  - Offline compactor to `GBSPR001` format (`deep.bin`).
  - Python interface with ctypes shared library loader and pure-Python NumPy fallback reader.

---

## Feature Inventory
Every feature from the Survey phase is mapped to an assigned milestone:

| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | `BaseEngineAdapter` Protocol | Formal ABC/protocol for engine adapters (`validate_engine`, `attach`, `detach`, `flush`) | M1 | Survey |
| 2 | Backend Registry & Auto-Detection | Registry mapping backend names (`pytorch`, `vllm`, `sglang`) and `auto` detection | M1 | Survey |
| 3 | `PyTorchEngineAdapter` | Preserves 100% legacy single-GPU PyTorch/HuggingFace execution via `DeepTap` & `SteeringController` | M1 | Survey |
| 4 | `VLLMEngineAdapter` & `GlassBoxWorkerExtension` | vLLM V1 worker extension for continuous batching, tapping hooks, and manifest harvesting | M1 | Survey |
| 5 | `SGLangEngineAdapter` | SGLang SRT adapter extracting `extend_start_loc` and `req_pool_indices` for RadixAttention | M1 | Survey |
| 6 | `StepManifest` Generation | CPU-side lightweight per-step manifest capturing continuous batching token metadata | M1 | Survey |
| 7 | `watch()` Signature Update | Extends `watch()` and `GlassBox.watch` with `backend="auto"`, `tp_size=None`, `tp_rank=None` | M1 | Survey |
| 8 | Backward Compatibility Shim | Expose new submodules in `blackbox/__init__.py` for legacy package callers | M1 | Survey |
| 9 | Residual Stream Passive Recording | Probes on `L-LL`, `NORM`, `EMBED` recorded on Rank 0 only; skipped on ranks 1..N | M2 | Survey |
| 10 | Residual Stream Active Steering | Steering interventions on residual stream applied identically to ALL TP ranks | M2 | Survey |
| 11 | Attention Head TP Sharding | Maps `A-LL-HH` to rank $r = \lfloor (HH-1) / H_{\text{rank}} \rfloor$ and local index | M2 | Survey |
| 12 | MLP Neuron TP Sharding | Maps `N-LL-NNNN` to rank $r = \lfloor (NNNN-1) / N_{\text{rank}} \rfloor$ and local index | M2 | Survey |
| 13 | Logits Entropy 3-Scalar Reduction | Ranks compute $(m_r, s_r, w_r)$ local log-sum-exp scalars per token row (12 B/tok) | M2 | Survey |
| 14 | Global Shannon Entropy Reconstruct | Math reconstruction: $M = \max m_r$, $S = \sum s_r e^{m_r-M}$, $W = \sum w_r e^{m_r-M}$, $H = M + \ln S - W/S$ | M2 | Survey |
| 15 | `glassbox::tap` Custom Op Registration | Register `glassbox::tap` in C++ via `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL` (CPU & CUDA) | M3 | Survey |
| 16 | Device Control Tensor `ctl` Semantics | Obeys `ctl[0]` (kill switch), `ctl[1]` (`actual_num_tokens`), `ctl[2]` (slot index) | M3 | Survey |
| 17 | Out-of-Graph Stream Copy | Asynchronous D2H transfer on dedicated `copyStream` with `cudaEventRecord` sync | M3 | Survey |
| 18 | Pure-Python & Triton Fallbacks | Pure-Python NumPy and Triton `@torch.library.custom_op` fallbacks in `glassbox/csrc_fallback.py` | M3 | Survey |
| 19 | Wait-Free SPSC Host Ring Buffer | Pinned host memory ring buffer in `csrc/ring_buffer.h` and Python equivalent | M4 | Survey |
| 20 | `DROP_NEWEST` Fail-Open Policy | SPSC drop policy on disk stall: increments dropped counter and emits `{step_id, status: DROPPED}` tombstone | M4 | Survey |
| 21 | Segmented WAL Format | 256MB size-rotated rank-local files (`runs/<ID>/ranks/rank_<r>/segment_<seq>.bin`) with 4MB alignment | M4 | Survey |
| 22 | OS Cache Management (`posix_fadvise`) | Calls `posix_fadvise(DONTNEED)` on completed WAL chunks; gracefully no-ops on non-Linux | M4 | Survey |
| 23 | Virtual Multi-Rank Indexing | Dynamic index engine joining multi-rank slices on `(global_step_id, req_id)` without physical merge | M4 | Survey |
| 24 | Offline `GBSPR001` Compactor | Background tool compacting segmented WALs into standard compressed `deep.bin` + `deep_index.json` | M4 | Survey |
| 25 | Collator Python & NumPy Fallback | Python interface `glassbox/storage/collator.py` with ctypes shared lib loader and pure-Python NumPy fallback | M4 | Survey |
| 26 | Baseline 100% Backward Compatibility | 77 passing unit tests in `tests/` preserved with 0 regressions; add `pandas>=2.0` to dependencies | M5 | Survey |
| 27 | E2E Test Suite Validation | Pass 100% of E2E Test Suite published by E2E Testing Track (`TEST_READY.md`) | M5 | Survey |
| 28 | Adversarial Coverage Hardening | Tier 5 white-box coverage hardening with adversarial edge-case stress tests | M5 | Survey |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Engine Backend Abstraction & Worker Extensions | `glassbox/backends/base.py`, `pytorch.py`, `vllm.py`, `sglang.py`, `__init__.py`, `glassbox/__init__.py`, `blackbox/__init__.py` | None | IN_PROGRESS |
| M2 | Distributed Sharding & Zero-NVLink Router | `glassbox/tap/distributed.py` (`TPShardRouter`, residual passive/active rules, head/neuron sharding, 3-scalar reduction and global Shannon entropy) | None | IN_PROGRESS |
| M3 | CUDA Graph Safe Wiretap & Fallback | `csrc/glassbox_ops.cpp`, `csrc/ring_gather_cuda.cu`, `glassbox/csrc_fallback.py` (`glassbox::tap`, `ctl` tensor, pure-Python/NumPy and Triton fallbacks) | None | IN_PROGRESS |
| M4 | Storage, Fail-Open Ring Buffer & Virtual Indexing | `csrc/ring_buffer.h`, `crates/glassbox_collator/`, `glassbox/storage/collator.py` (ring buffer, segmented WAL, virtual index, compactor, ctypes loader + fallback) | None | IN_PROGRESS |
| M5 | Final Milestone: E2E Integration & Adversarial Hardening | Phase 1: 100% E2E test pass (Tiers 1-4) & 0 baseline regressions; Phase 2: Tier 5 adversarial hardening | M1, M2, M3, M4, TEST_READY.md | PLANNED |

*Note*: In parallel with M1-M4, the **E2E Testing Track Orchestrator** is dispatched to design the test harness and implement opaque-box test suites (Tiers 1-4), publishing `TEST_READY.md`.

---

## Interface Contracts

### 1. `BaseEngineAdapter` ↔ Serving Engines & `watch()`
- Location: `glassbox/backends/base.py`
```python
class StepManifest:
    step_id: int
    req_ids: list[str | int]
    query_start_loc: list[int] | np.ndarray
    logical_positions: list[int] | np.ndarray | None = None
    actual_num_tokens: int = 0
    timestamp_ns: int = 0
    flags: dict[str, Any] = field(default_factory=dict)
    def to_dict(self) -> dict[str, Any]: ...

class BaseEngineAdapter(ABC):
    backend_name: ClassVar[str]
    def __init__(self, tp_rank: int = 0, tp_size: int = 1, **kwargs: Any) -> None: ...
    @classmethod
    @abstractmethod
    def validate_engine(cls, engine: Any) -> bool: ...
    @abstractmethod
    def attach(self, engine: Any, watchlist: dict[str, Any], **kwargs: Any) -> Any: ...
    @abstractmethod
    def detach(self) -> None: ...
    @abstractmethod
    def flush(self) -> None: ...
    @property
    def is_attached(self) -> bool: ...
```

### 2. `TPShardRouter` ↔ Distributed Workers
- Location: `glassbox/tap/distributed.py`
```python
class TPShardRouter:
    def __init__(self, tp_rank: int, tp_size: int, architecture: dict[str, Any] | None = None) -> None: ...
    def is_owner(self, address: str) -> bool: ...
    def should_steer(self, address: str) -> bool: ...
    def route_address(self, address: str) -> tuple[int, int]: ... # (owner_rank, local_index)
    def compute_local_logits_reduction(self, local_logits: torch.Tensor | np.ndarray) -> tuple[float, float, float]: ... # (m_r, s_r, w_r)
    @staticmethod
    def reconstruct_global_entropy(scalars: list[tuple[float, float, float]]) -> float: ... # global Shannon entropy
```

### 3. `glassbox::tap` Operator
- Location: `csrc/glassbox_ops.cpp` & `glassbox/csrc_fallback.py`
```cpp
// PyTorch Custom Op Schema:
"glassbox::tap(Tensor activations, Tensor feature_indices, Tensor ctl, Tensor(a!) staging_slab, int col_offset) -> ()"
// ctl tensor format (int32 or int64):
// ctl[0]: kill switch (0 = disabled, 1 = active)
// ctl[1]: actual_num_tokens (gathered rows: 0..actual_num_tokens-1)
// ctl[2]: active staging slot index (0..K-1)
```

### 4. Storage & Virtual Index Reader ↔ Query API
- Location: `glassbox/storage/collator.py` & `crates/glassbox_collator/`
```python
class VirtualIndexReader:
    def __init__(self, run_dir: str | Path) -> None: ...
    def query_slice(self, global_step_id: int, req_id: str | int, address: str) -> np.ndarray | None: ...
    def compact_to_deep_bin(self, output_path: str | Path) -> None: ...
```

---

## Code Layout
- `glassbox/backends/`:
  - `__init__.py`: Registry, auto-detection, and export.
  - `base.py`: `BaseEngineAdapter` protocol and `StepManifest`.
  - `pytorch.py`: `PyTorchEngineAdapter` wrapping `DeepTap` & `SteeringController`.
  - `vllm.py`: `VLLMEngineAdapter` & `GlassBoxWorkerExtension`.
  - `sglang.py`: `SGLangEngineAdapter`.
- `glassbox/tap/`:
  - `distributed.py`: `TPShardRouter` and reduction math.
  - `hooks.py`: Existing `DeepTap` (unchanged for backward compatibility).
  - `sidecar.py`: Existing `Sidecar` (unchanged).
- `csrc/`:
  - `glassbox_ops.cpp`: PyTorch C++ custom op definitions.
  - `ring_gather_cuda.cu`: CUDA kernel for in-graph gather.
  - `ring_buffer.h`: Header-only SPSC pinned host ring buffer.
- `glassbox/csrc_fallback.py`: Pure-Python / NumPy and Triton fallbacks for `glassbox::tap`.
- `crates/glassbox_collator/`:
  - `Cargo.toml`: Crate manifest (`cdylib`, `bin`).
  - `src/lib.rs`: C-ABI and Rust API exports.
  - `src/wal.rs`: Segmented WAL writer/reader with 256MB rotation and `posix_fadvise`.
  - `src/index.rs`: Virtual index joining `(global_step_id, req_id)`.
  - `src/compactor.rs`: Offline compactor to `GBSPR001`.
  - `src/main.rs`: CLI binary entry point.
- `glassbox/storage/`:
  - `collator.py`: Python interface to Rust collator with ctypes and pure-Python NumPy fallback.
  - `layout.py`: Existing storage layout and `read_deep_slice` (extended to query virtual index).
- `tests/`:
  - `test_backends.py`: Backend abstraction and worker extension unit tests.
  - `test_distributed_tp.py`: Distributed TP router and logits entropy reduction tests.
  - `test_graph_tap.py`: CUDA Graph safe wiretap and `ctl` tensor tests.
  - `test_collator.py`: Storage, ring buffer, segmented WAL, and virtual index tests.
