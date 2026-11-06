<div align="center">

```
   ____ _                         ____             
  / ___| | __ _ ___ ___          | __ )  _____  __ 
 | |  _| |/ _` / __/ __|  _____  |  _ \ / _ \ \/ / 
 | |_| | | (_| \__ \__ \ |_____| | |_) | (_) >  <  
  \____|_|\__,_|___/___/         |____/ \___/_/\_\ 
```

### Zero-Overhead Flight Recorder & Mechanistic Interpretability Engine for PyTorch LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Ruff: Compliant](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: Mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![Status: v0.1.0](https://img.shields.io/badge/release-v0.1.0-brightgreen.svg)]()

<p align="center">
  <b>GlassBox makes LLM internals clear as glass.</b><br>
  Don't just observe what a model outputs — record tamper-evident mathematical evidence to reconstruct internal neuron firings, attention dynamics, residual stream progressions, and causal decision pathways token-by-token.
</p>

[Key Features](#key-features) •
[Why GlassBox?](#why-glassbox) •
[Architecture](#architecture) •
[Quickstart](#quickstart) •
[Superpowers](#mechanistic-superpowers) •
[CLI Reference](#cli-reference) •
[Watchlist DSL](#addressing-scheme--watchlist-dsl)

---

</div>

## Why GlassBox?

Traditional LLM observability tools (LangSmith, OpenTelemetry, W&B) operate outside the model weights: they record prompts, completions, latency, and token counts. But when a model produces a hallucination, safety violation, or reasoning breakdown, **external logs cannot tell you why**.

To debug models mechanistically, researchers need internal states. However, naive PyTorch hooks introduce crippling GPU synchronization bubbles (`.cpu()` / `.item()`) that stall compute pipelines by 15–40%.

**GlassBox solves this.** It acts as an aircraft flight recorder:

| Capability | Standard LLM Observability | GlassBox |
|---|---|---|
| **Capture Layer** | External API / Token strings | Internal activations, attention heads, residual streams, logits |
| **GPU Impact** | None (doesn't touch GPU) | **<1% overhead**: Zero-sync async `copyStream` DMA transfers |
| **Storage Engine** | Text / JSON traces | High-density binary (`deep.bin`) with chunked sparse compression |
| **Integrity** | Mutable database rows | **Cryptographic SHA-256 seal** + append-only provenance chain |
| **Interpretability** | Surface prompt analysis | **Logit Lens**, **Direct Logit Attribution (DLA)**, **SAE Latents** |
| **Visualization** | Heavy web dashboards (React/Node) | **Calm Viewer**: 100% self-contained single-file HTML & inline SVG |

---

## Architecture

GlassBox hooks directly into PyTorch forward passes with non-blocking DMA copies to pinned host memory, completely decoupling inference execution from storage.

```mermaid
flowchart TD
    subgraph GPU ["GPU Compute Pipeline (Forward Pass)"]
        M["PyTorch LLM Forward Pass"] -->|Layer Hook| T["DeepTap Async Gather"]
        T -->|Non-blocking copy_| CS["Dedicated CUDA copyStream"]
    end

    subgraph Host ["Zero-Stall Host Memory"]
        CS -->|Async DMA| PIN["Pinned Double Buffer (PBO)"]
        PIN -->|Ring Swap| RING["Triple Ring Buffer (/dev/shm/gb.ring)"]
    end

    subgraph Storage ["Tamper-Evident Flight Store"]
        RING -->|Flush / fsync| WAL["WAL Sidecar Worker"]
        WAL --> DBIN["deep.bin (Dense / Chunked Sparse)"]
        WAL --> DIDX["deep_index.json (Unified Hot Metadata)"]
        WAL --> SEAL["seal.json (SHA-256 Hash Link)"]
        SEAL --> CHAIN["chain.jsonl (Immutable Audit Trail)"]
    end

    subgraph Analysis ["Analysis & Visualization Engine"]
        DIDX & DBIN --> QUERY["Two-Phase Search (glassbox find / compare)"]
        DIDX & DBIN --> LENS["Logit Lens (glassbox view --lens)"]
        DIDX & DBIN --> DF["Vectorized to_dataframe() -> Pandas/Polars"]
        DIDX & DBIN --> HTML["Calm HTML Viewer (Zero NPM, Inline SVG)"]
    end

    classDef gpu fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef host fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef store fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef analysis fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;

    class M,T,CS gpu;
    class PIN,RING host;
    class WAL,DBIN,DIDX,SEAL,CHAIN store;
    class QUERY,LENS,DF,HTML analysis;
```

---

## Key Features

- ⚡ **Zero-Stall Asynchronous Tap**: Captures activations on a dedicated CUDA `copyStream` without CPU synchronization barriers.
- 🗜️ **Chunked Sparse Compression**: Transparent chunked compression (`zlib` / RLE) for zero-heavy activations, reducing disk footprint by 2× to 10× with random-access slice reads (`read_deep_slice`).
- 🔭 **Logit Lens**: Project intermediate layer representations ($L_0 \dots L_N$) through the unembedding matrix to see token predictions form in real time across model depth.
- 🎯 **Direct Logit Attribution (DLA)**: Quantify causal feature salience between prompt cohorts (`glassbox compare --attribute`).
- 🧬 **Sparse Autoencoder (SAE) Latents**: Full first-class support for monosemantic feature addresses (`SAE-LL-FFFF` and `F-LL-FFFF`).
- 🤖 **Automated Watchlist Profiler**: Automatically profiles models over sample prompts and exports tailored, high-variance watchlists.
- 🔒 **Cryptographic Hash Sealing**: SHA-256 seal covering unified metadata and binary activations linked into an append-only provenance chain (`chain.jsonl`).
- 🎨 **Calm Viewer**: Self-contained static HTML reports with inline SVG activation heatmap matrices and co-occurrence circuit graphs (zero npm, zero React, zero CDN).

---

## Installation

```bash
# Install via pip
pip install glassbox

# Or install from source
git clone https://github.com/<your-username>/glassbox.git
cd glassbox
pip install -e ".[dev,search]"
```

> **Backward Compatibility**: Full backward-compatible shims are included. Existing code using `import blackbox` or running the `blackbox` CLI command continues to work without modification.

---

## Quickstart

### 1. Scan Model Weights (CPU, 2–10 seconds, once per checkpoint)

Inspect safetensors metadata and architecture config to generate an inventory manifest:

```bash
glassbox scan /path/to/llama-3-8b
```
```
Scan CPU 0.3s: config.json + safetensors header only (no GPU, no prompt).
Manifest -> manifests/M-001_49666014/manifest.json
M-001 llama-3-8b | C-496660 sha256:496660... | 32L hidden4096 32H MLP14336 | torch-cpu | ready for Deep watchlists
```

### 2. Instrument Model with a Declarative Watchlist

Define exactly which neurons, attention heads, or residual streams to record:

```yaml
# watchlists/pilot.yaml
version: 1
model_id: M-001
checkpoint_sha256: 496660148f94...
watchlist_id: pilot-v1
trigger: always
window: { start_token: 0, end_token: 512, every_n: 1 }
precision: fp16
entries:
  - addresses: [N-24-0001..0100, A-17-04, LOGITS]
```

Use the context manager in Python:

```python
from glassbox import GlassBox
import torch

model = load_model(...)

# Watchlist validated and memory pre-allocated before forward pass begins
with GlassBox.watch("watchlists/pilot.yaml", manifest="manifests/M-001/manifest.json"):
    output = model.generate("The capital of France is", max_new_tokens=20)
```

### 3. Verify Cryptographic Integrity

Every run is sealed on context exit. Verify that neither the metadata nor binary activations have been altered:

```bash
glassbox verify RUN-000001
# Recomputes SHA256(prev || SHA256(deep_index) || SHA256(deep) || canonical(replay))
# Validates unbroken chain continuity
```
```
RUN-000001: VERIFIED ok (chain link #1 unbroken)
```

---

## Mechanistic Superpowers

### 1. The Logit Lens

Project intermediate residual streams directly into vocabulary space to detect the exact layer where knowledge or hallucinations commit:

```bash
glassbox view RUN-000001 --lens
```

```
Layer Progression: RUN-000001 Token #17 ("Paris")
Layer   Top Predicted Token   Top Logit   Entropy
-------------------------------------------------
L-00    " "                   2.1400      5.4200
L-04    "a"                   3.0200      4.8900
L-08    "the"                 4.1200      4.2100
L-16    "city"                6.8700      2.8400
L-24    "Paris"               11.4500     1.0200
L-31    "Paris"               16.8900     0.3200
```

Programmatic access:
```python
import glassbox

lens_data = glassbox.lens("RUN-000001", token_pos=17)
print(lens_data["layers"])
```

### 2. Direct Logit Attribution (DLA)

Rank which specific neurons or attention heads caused a model's behavior to diverge between two cohorts:

```bash
glassbox compare --good tag:cohort=truthful --bad tag:cohort=hallucinated \
  --neurons N-24-0001..0100,A-17-04 --attribute
```

```
Direct Logit Attribution (DLA) Ranking:
Rank  Address    Layer  Component  Mean Diff   Attribution Score  Causal Impact
---------------------------------------------------------------------------
#1    N-24-0042  24     MLP-Neuron +1.8420     0.9412             HIGH
#2    A-17-04    17     Attn-Head  +1.1200     0.7810             MEDIUM
#3    N-24-0019  24     MLP-Neuron -0.9520     0.6540             INHIBITORY
```

### 3. Automated Watchlist Calibration Profiler

Don't guess which neurons matter. Automatically calibrate against representative inputs:

```python
import glassbox

# Discovers the top-500 highest-variance components across sample prompts
active_addrs = glassbox.profile(
    model,
    manifest,
    sample_inputs=["Reasoning prompt A", "Factual recall B"],
    top_k=500,
    output_yaml="watchlists/auto_calibrated.yaml",
)
```

### 4. Vectorized DataFrame Export

Export runs directly into Pandas / Polars with memory-mapped zero-copy slice reads:

```python
import glassbox

# Instantaneous vector load — no row-by-row dict iteration
df = glassbox.to_dataframe("RUN-000001", tokens=(0, 64))
print(df.head())
```
```
   token_pos  token    address     value  seal_ok
0          0      1  N-24-0001  0.041200     True
1          0      1  N-24-0002 -0.891400     True
2          0      1    A-17-04  1.421000     True
```

### 5. Calm HTML Viewer (Zero NPM, Zero React)

Export a fully self-contained HTML visualization including an **interactive SVG activation matrix** and **co-occurrence circuit diagram**:

```bash
glassbox view RUN-000001 --export-html ./report.html
```

Open `report.html` in any browser — zero network requests, zero CDN dependencies, 100% portable.

---

## Addressing Scheme & Watchlist DSL

GlassBox uses standardized coordinate addresses:

| Syntax | Description | Example |
|---|---|---|
| `N-LL-IIII` | MLP intermediate neuron at layer `LL`, feature index `IIII` | `N-24-0042` |
| `A-LL-HH` | Attention head L2 activation norm at layer `LL`, head `HH` | `A-17-04` |
| `L-LL` | Residual stream layer norm / progression at layer `LL` | `L-16` |
| `L-LL-DDDD` | Specific hidden dimension `DDDD` at layer `LL` | `L-16-0512` |
| `SAE-LL-FFFF` / `F-LL-FFFF` | Sparse Autoencoder monosemantic feature latent | `SAE-24-0412` |
| `LOGITS` | Top-1 predicted token and confidence per step | `LOGITS` |
| `LOGITS:<token_id>` | Logit tracking for a specific vocabulary token ID | `LOGITS:1042` |
| `LOGITS-ENTROPY` | Prediction entropy across vocabulary distribution | `LOGITS-ENTROPY` |
| `X..Y` | Inclusive range expansion | `N-24-0001..0100` |

### Built-In Presets

GlassBox includes 7 deterministic presets:

- `outer_only`: Metadata, prompts, output tokens, seed, and timing ($W=0$).
- `pilot-sparse-1k`: 1,000 deterministic seeded neurons across layers 8, 16, 24, and 31.
- `attention_only`: All attention heads across all layers ($W=1,024$).
- `logit-lens`: Residual streams $L_0 \dots L_N$, layer norms, and logits ($W=34$).
- `last-4-layers-full`: All heads, neurons, and logits across the final 4 layers.
- `ioi-circuit-starter`: Canonical heads and MLP neurons for Indirect Object Identification circuits.
- `mid-mlp-wide`: Broad mid-layer MLP monitoring with every-10th-token subsampling.

---

## CLI Reference

GlassBox provides exactly 8 focused subcommands:

```
usage: glassbox <command> [options]

Commands:
  scan       Scan weights on CPU into a manifest.json inventory
  run        Execute run with a declared watchlist and hash-seal output
  find       Two-phase Hot/Cold query across runs with SQL-like predicates
  compare    Statistical comparison & Direct Logit Attribution across cohorts
  verify     Verify cryptographic SHA-256 seal and chain continuity
  view       Terminal table view, Logit Lens (--lens), or HTML report export
  size       Show storage footprint with GREEN/YELLOW/RED capacity flags
  export     Export run to portable flat ZIP archive
```

### CLI Examples

```bash
# Scan weights
glassbox scan ./weights/llama-3-8b

# Run surveillance with size safety gate
glassbox run --watch watchlists/pilot.yaml --tag cohort=eval --seed 42

# Two-phase search: filter 100k runs in Hot index, verify in Cold mmap
glassbox find 'N-24-0001 AT token=17 > 0.8' --verify

# Compare 10,000 good vs 10,000 bad runs with DLA ranking
glassbox compare --good tag:cohort=good --bad tag:cohort=bad \
  --neurons N-24-0001,A-17-04 --attribute

# Verify integrity across all runs
glassbox verify --all

# View run in terminal or export self-contained HTML
glassbox view RUN-000001
glassbox view RUN-000001 --lens
glassbox view RUN-000001 --export-html ./run_000001.html

# Check disk usage against capacity safety thresholds
glassbox size
```

---

## Storage Layout & Provenance Chain

Runs are stored in flat, self-contained directories under `glassbox_store/`:

```
glassbox_store/
├── chain.jsonl                # Append-only hash chain linking all runs
├── index/                     # Tier-1 Hot search index shards
│   └── shard-0000.jsonl
└── runs/
    └── RUN-000001/
        ├── deep.bin           # Raw binary activations (row-major [T, W] LE)
        ├── deep_index.json    # Unified metadata, prompt, output, replay context
        └── seal.json          # Cryptographic SHA-256 seal:
                               # hash_i = SHA256(prev || SHA256(deep_index) || SHA256(deep) || replay)
```

If even a single byte of `deep.bin` or `deep_index.json` is modified post-execution, `glassbox verify` flags the tampering immediately.

---

## Testing & Quality Assurance

GlassBox is tested with 100% CPU-executable test suites (Linux, macOS, Windows):

```bash
# Run full test suite (59 unit, integration, and feature tests)
pytest -q

# Run static type checking
mypy glassbox/

# Run linter & formatter
ruff check .
ruff format --check .
```

---

## License

GlassBox is open-source software licensed under the [Apache License, Version 2.0](LICENSE).
