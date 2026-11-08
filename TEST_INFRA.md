# GlassBox E2E and Unit Test Infrastructure (`TEST_INFRA.md`)

## 1. Overview & Architectural Principles

This document specifies the test architecture, methodologies, and verification harnesses for GlassBox Fast Production Serving and Distributed Multi-GPU Sharding.

The test infrastructure enforces three foundational principles:
1. **Zero Regression Guarantee**: All 77 existing baseline unit and integration tests must pass with 100% backward compatibility.
2. **Deterministic & Authoritative Expected Outputs**: Every test asserts against mathematical ground truths (e.g. `torch.distributions.Categorical.entropy()`), formal binary schemas (e.g. `GBWAL001`, `GBSPR001`), or explicit protocol interfaces. No facade tests or dummy passes are permitted.
3. **Tri-Language Verification**: Python, C++/CUDA (`csrc/ring_buffer.h`), and Rust (`crates/glassbox_collator/`) components are tested both in isolation and across integrated boundaries.

---

## 2. The 4-Tier Test Hierarchy

Each subsystem is tested across four orthogonal tiers:

| Tier | Category | Purpose | Typical Coverage |
| :--- | :--- | :--- | :--- |
| **Tier 1** | **Feature Coverage** | Validates the primary behavior (happy path) and interface contracts for all 25 features. | >= 5 test cases per functional milestone. |
| **Tier 2** | **Boundary & Corner Cases** | Exercises degenerate, empty, single-element, and extreme inputs. | Clamped bounds, zero-tokens, single-slot buffers, extreme vocab sizes. |
| **Tier 3** | **Cross-Feature Combinations** | Tests pairwise interactions across disparate subsystems. | Precision dtypes (fp16/fp32/bf16), dynamic backend resolution, routing vs steering. |
| **Tier 4** | **Real-World Scenarios** | Simulates end-to-end serving workloads under production failure modes. | Multi-turn continuous batching, CUDA Graph dynamic `ctl` updates, writer backpressure stalls. |

---

## 3. Test Suite Matrix Across All 25 Features

### 3.1. Engine Backends & Worker Extensions (`tests/test_backends.py`)
- **Feature 1 (`BaseEngineAdapter` Protocol)**: Abstract protocol enforcement, lifecycle (`validate_engine`, `attach`, `detach`, `flush`, `is_attached`).
- **Feature 2 (Backend Registry & Discovery)**: Dynamic registration decorator `@register_backend`, case-insensitive lookup via `get_backend`, engine auto-detection (`detect_backend`, `resolve_backend`).
- **Feature 3 (`PyTorchEngineAdapter`)**: Preserves legacy single-GPU and HuggingFace execution wrapping `DeepTap` & `SteeringController`.
- **Feature 4 (`VLLMEngineAdapter` & `GlassBoxWorkerExtension`)**: Continuous batching `InputBatch` metadata harvesting, worker hooks.
- **Feature 5 (`SGLangEngineAdapter`)**: RadixAttention metadata extraction (`extend_start_loc`, `req_pool_indices`).
- **Feature 6 (`StepManifest` Generation)**: CPU-side per-step execution manifest, auto-token calculation, JSON primitive serialization.
- **Feature 7 (`watch()` Signature Update)**: Backward-compatible support for `backend="auto"`, `tp_size`, `tp_rank` in `watch()` and `GlassBox.watch`.
- **Feature 8 (Backward Compatibility Shim)**: Legacy package shims in `blackbox/__init__.py` forwarding to `glassbox.backends`.

### 3.2. Distributed TP Sharding & Router (`tests/test_distributed_tp.py`)
- **Feature 9 (Residual Stream Passive Recording)**: Passive recording on Rank 0 only (`L-LL`, `NORM`, `EMBED`); skipped on ranks 1..N.
- **Feature 10 (Residual Stream Active Steering)**: Active interventions applied to ALL TP ranks identically to prevent forward divergence.
- **Feature 11 (Attention Head Modulo Sharding)**: Column-parallel mapping `A-LL-HH` to owner rank $r = \lfloor (HH - 1) / H_{\text{rank}} \rfloor$ and local index.
- **Feature 12 (MLP Neuron Modulo Sharding)**: Column-parallel mapping `N-LL-NNNN` to owner rank $r = \lfloor (NNNN - 1) / N_{\text{rank}} \rfloor$ and local index.
- **Feature 13 (Logits Entropy 3-Scalar Reduction)**: Local $(m_r, s_r, w_r)$ log-sum-exp reductions per token row.
- **Feature 14 (Global Shannon Entropy Reconstruction)**: Mathematical reconstruction matching `torch.distributions.Categorical(logits=...).entropy()` within precision tolerance ($< 10^{-5}$) with zero NVLink traffic.

### 3.3. CUDA Graph Safe Wiretap & Op (`tests/test_graph_tap.py`)
- **Feature 15 (`glassbox::tap` Custom Op Registration)**: Op registration via `torch.library.custom_op` / C++ runtime.
- **Feature 16 (Device Control Tensor `ctl` Semantics)**: In-graph device tensor evaluation:
  - `ctl[0] == 0`: Kill switch active (staging slab unmodified).
  - `ctl[0] == 1`: Active gather of `ctl[1]` actual tokens into slot `ctl[2]`.
  - `col_offset`: Column placement within staging slab.
- **Feature 17 (Out-of-Graph Stream Copy)**: Asynchronous D2H transfer (`glassbox_async_d2h_copy`) using dedicated copy stream and non-blocking event recording.
- **Feature 18 (Pure-Python & Triton Fallbacks)**: NumPy and Triton implementations in `glassbox/csrc_fallback.py` with `@tap.register_fake` shape transfer for Dynamo/CUDA Graph tracing without graph breaks.

### 3.4. Storage, Ring Buffer & Collator (`tests/test_collator.py`)
- **Feature 19 (Wait-Free SPSC Host Ring Buffer)**: Pinned host ring buffer in `csrc/ring_buffer.h` with wait-free SPSC semantics and head/tail wrapping.
- **Feature 20 (`DROP_NEWEST` Fail-Open Policy)**: Drop newest on backpressure: producer thread never stalls, increments dropped count, emits tombstone.
- **Feature 21 (Segmented WAL Format)**: 256MB size-rotated segment files (`segment_<seq>.bin`), `GBWAL001` headers, `WALR` record frames with CRC32 integrity.
- **Feature 22 (OS Cache Management)**: `posix_fadvise(POSIX_FADV_DONTNEED)` cache eviction on completed segments.
- **Feature 23 (Virtual Multi-Rank Indexing)**: `VirtualIndexReader` joining multi-rank slices on `(global_step_id, req_id)` without physical merge.
- **Feature 24 (Offline `GBSPR001` Compactor)**: Offline compaction of segmented WALs into unified `deep.bin` + `deep_index.json`.
- **Feature 25 (Collator Python & NumPy Fallback)**: Pure-Python ctypes loader with robust NumPy fallback for multi-rank index reading.

---

## 4. Test Execution & Verification Commands

All tests are designed to execute synchronously using `uv run pytest` under Python 3.11:

```bash
# Run new test suites across all 25 features (105 tests)
uv run pytest tests/test_backends.py tests/test_distributed_tp.py tests/test_graph_tap.py tests/test_collator.py -v

# Run entire repository test suite (182 tests: 77 baseline + 105 new)
uv run pytest tests/ -v
```
