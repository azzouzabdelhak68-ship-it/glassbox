# GlassBox Documentation & Guide Hub

Welcome to the **GlassBox** documentation portal. Whether you are using GlassBox for zero-overhead flight recording during training/serving, conducting mechanistic interpretability research, or performing real-time activation steering and causal ablation, this guide provides a structured index of all concepts, specifications, and tutorials.

---

## 🗺️ Documentation Sitemap

```
docs/
├── README.md                   <-- You are here (Master Documentation Hub)
├── STEERING_GUIDE.md           <-- Real-time Activation Steering & Causal Ablation
├── DISCOVERY_QWEN_SAFETY.md    <-- Empirical Case Study: Qwen3.8-27B on NVIDIA A100
├── ARCHITECTURE.md             <-- System Architecture: Async DMA, copyStreams, Ring Buffers
├── FORMAT_GUIDE.md             <-- Binary Layout (deep.bin), Index Manifest, & SHA-256 Chains
├── ADAPTER_GUIDE.md            <-- Hardware Adapters: CUDA, CPU, ROCm, Apple MPS
├── PRESETS.md                  <-- Watchlist Presets & Addressing DSL
├── THREAT_MODEL.md             <-- Cryptographic Integrity & Anti-Tamper Security
└── Distributed Wiki Guide      <-- ../wiki/Distributed-Serving-and-Multi-GPU-Architecture.md
```

---

## 🚀 1. Getting Started & Core Concepts

* **[Quickstart & Installation](../README.md#-quickstart)**: Install GlassBox, run CPU/GPU verification, and record your first flight in under 60 seconds.
* **[Watchlist Addressing DSL](PRESETS.md)**: Master GlassBox's unified coordinate addressing:
  * `N-LL-NNNN`: Intermediate MLP post-activation neurons.
  * `A-LL-HH`: Attention head activation norms.
  * `L-LL`: Residual stream hidden-state progressions.
  * `LOGITS` / `LOGITS-ENTROPY`: Vocabulary projections and Shannon entropy.
* **[Preset Libraries](PRESETS.md)**: Explore built-in presets (`pilot`, `attention_only`, `all_layers_sparse`) for plug-and-play probing.

---

## 🔬 2. Real-Time Activation Steering & Causal Ablation (Feature 1)

* **[Steering & Ablation Guide](STEERING_GUIDE.md)**: Transition from passive observation to active causal intervention.
  * **Additive Directional Steering**: Inject steering vectors $\mathbf{x} \leftarrow \mathbf{x} + \alpha \mathbf{v}$ into residual streams, MLPs, or attention blocks.
  * **Norm-Preserving Steering**: Automatic renormalization to maintain activation magnitude stability.
  * **Orthogonal Subspace Ablation**: Project out refusal, sycophancy, or bias directions $\mathbf{x} \leftarrow \mathbf{x} - \frac{\mathbf{x} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$.
  * **Neuron Clamping & Head Zeroing**: Clamp or silence specific MLP neurons (`N-64-0024`) or attention heads (`A-18-03`) mid-flight.
  * **Token Mode Conditioning**: Steer only during autoregressive decode (`mode="decode"`), prompt prefill (`mode="prefill"`), or at specific token positions.

---

## 📑 3. Featured Research & Case Studies

* **[Caught in the Act: Mechanistic Discoveries in Qwen3.8-27B](DISCOVERY_QWEN_SAFETY.md)**:
  * Full empirical whitepaper documenting our NVIDIA A100 SXM4-80GB stretch run.
  * Proves refusal decisions lock at the user `<|im_end|>` token *before* `<think>` starts.
  * Details the causal proof of why the model emitted `"Need final in English."` (latent Chinese RLHF circuit collision at Layer 56).
  * Layer-18 Gated DeltaNet threat tripwires and 16-layer entropy double crossover.
* **[1-Click Replication Notebook](../notebooks/a100_stretch_test.ipynb)**: Open directly in Google Colab to reproduce our A100 findings in ~2–4 minutes.

---

## ⚙️ 4. Technical Specifications & Internals

* **[System Architecture](ARCHITECTURE.md)**: How GlassBox achieves sub-1% runtime overhead:
  * Dedicated CUDA `copyStream` asynchronous DMA transfers.
  * Pinned host memory (PBO) double-buffering.
  * Non-blocking triple ring buffers (`/dev/shm/gb.ring`).
  * Background Write-Ahead Log (WAL) sidecar process.
* **[Flight Store Binary Layout](FORMAT_GUIDE.md)**:
  * `deep.bin`: Row-major token-major compressed sparse binary matrix (`float16` / `bfloat16` / `float32`).
  * `deep_index.json`: Unified hot manifest carrying execution replay parameters and column offsets.
  * `seal.json`: Cryptographic SHA-256 hash chains connecting runs into an immutable provenance ledger (`chain.jsonl`).
* **[Hardware Adapter Guide](ADAPTER_GUIDE.md)**: Device-specific pipelines for NVIDIA CUDA, AMD ROCm, Apple Metal (MPS), and CPU fallback.
* **[Threat Model & Security](THREAT_MODEL.md)**: Tamper resistance, audit-trail validation, and verifying integrity with `glassbox verify RUN-ID`.

---

## 📊 5. Visualization & Analysis Tools

* **Calm HTML Viewer**: Generates 100% self-contained, single-file HTML dashboards with inline SVG activation heatmaps and circuit co-occurrence graphs (zero npm, zero CDN dependencies).
* **Parallel 16-Layer Logit Lens**: Project hidden states directly into vocabulary space on-device across 16 layers simultaneously (`glassbox view RUN-ID --lens`).
* **Direct Logit Attribution (DLA)**: Compare good vs. bad cohorts to rank which specific neurons or heads drove behavioral divergence (`glassbox compare --attribute`).
* **Vectorized DataFrame Export**: Memory-mapped zero-copy slice reads into Pandas and Polars (`glassbox.to_dataframe("RUN-ID")`).

---

## 🌐 6. Distributed Production Serving & Multi-GPU (v2.0)

* **[Distributed Serving Architecture](../wiki/Distributed-Serving-and-Multi-GPU-Architecture.md)**:
  * **Serving Backends**: Continuous batching worker extensions for vLLM V1 (`GlassBoxWorkerExtension`) and SGLang SRT (`SGLangEngineAdapter`).
  * **Zero-NVLink TP Sharding**: Local 3-scalar $(m_r, s_r, w_r)$ reduction per token (12 B/tok) reconstructing exact global Shannon entropy with 0 NVLink inter-GPU traffic.
  * **CUDA Graph-Safe Wiretap**: Two-stage in-graph op (`glassbox::tap`) writing to static HBM slabs via `ctl` device tensor, paired with out-of-graph asynchronous D2H DMA copy.
  * **Rust Collator & SPSC Ring Buffer**: Pinned host memory wait-free SPSC ring buffer with `DROP_NEWEST` fail-open policy, 256MB segmented WALs, and dynamic virtual multi-rank indexing.

---

## 💬 Community & Support

* **[Discussions](https://github.com/azzouzabdelhak68-ship-it/glassbox/discussions)**: Share mechanistic findings, ask questions, and propose new features.
* **[Issue Tracker](https://github.com/azzouzabdelhak68-ship-it/glassbox/issues)**: Report bugs or request architectural extensions.
* **[Support Guide](../SUPPORT.md)**: How to get help and connect with the development team.
