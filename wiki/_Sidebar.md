# [GlassBox Wiki](Home.md)

**Flight Recorder & Mechanistic Engine**

---

### Navigation

* **[Home](Home.md)**
  * [Overview & Value Proposition](Home.md#welcome-to-the-glassbox-wiki)
  * [Architecture Diagrams](Home.md#visual-architecture-overview)
  * [Quickstart](Home.md#quickstart-guide)

---

### Core Documentation

* **[User Guide](User-Guide.md)**
  * [Installation & Hardware Setup](User-Guide.md#installation-and-hardware-tiers)
  * [Watchlist DSL & Presets](User-Guide.md#watchlist-addressing-dsl-and-presets)
  * [Flight Recording Workflow](User-Guide.md#flight-recording-workflow)
  * [CLI Manual & Exit Codes](User-Guide.md#cli-reference-manual)

* **[Research Playbook](Research-Playbook.md)**
  * [5-Step Mechanistic Protocol](Research-Playbook.md#mechanistic-interpretability-workflow)
  * [Hypothesis Testing Templates](Research-Playbook.md#hypothesis-formulation-template)
  * [Direct Logit Attribution (DLA)](Research-Playbook.md#direct-logit-attribution-dla)
  * [The 4-Module Safety Battery](Research-Playbook.md#the-4-module-safety-battery)

---

### Theory & Control Plane

* **[Activation Steering & Ablation](Activation-Steering-and-Ablation-Guide.md)**
  * [Additive Vector Steering](Activation-Steering-and-Ablation-Guide.md#additive-vector-steering)
  * [Norm Preservation Math](Activation-Steering-and-Ablation-Guide.md#norm-preservation-mathematics)
  * [Orthogonal Subspace Ablation](Activation-Steering-and-Ablation-Guide.md#directional-orthogonal-subspace-ablation)
  * [Neuron & Head Silencing](Activation-Steering-and-Ablation-Guide.md#neuron-clamping-and-attention-head-silencing)

* **[Architecture & Internals](Architecture-and-Internals.md)**
  * [Non-Blocking DMA copyStream](Architecture-and-Internals.md#non-blocking-dma-staging-and-copystream)
  * [SPSC Ring Buffer WAL](Architecture-and-Internals.md#lockless-spsc-triple-buffer-ring-and-wal)
  * [Chunked Sparse Format](Architecture-and-Internals.md#chunked-sparse-binary-storage-deepbin)
  * [SHA-256 Provenance Ledger](Architecture-and-Internals.md#run-provenance-and-cryptographic-sealing)

---

### Empirical Research

* **[Discovery & Hypothesis Catalog](Discovery-and-Hypothesis-Catalog.md)**
  * [Qwen3.8-27B Empirical Case Study](Discovery-and-Hypothesis-Catalog.md#frontier-case-study-qwen38-27b)
  * [Pre-Thought Refusal Lock](Discovery-and-Hypothesis-Catalog.md#discovery-1-pre-thought-refusal-lock)
  * [Latent Chinese RLHF Circuit](Discovery-and-Hypothesis-Catalog.md#discovery-2-latent-chinese-rlhf-circuit)
  * [Layer-18 DeltaNet Sentinels](Discovery-and-Hypothesis-Catalog.md#discovery-4-the-multi-stage-safety-control-circuit)
  * [Hypothesis Catalog](Discovery-and-Hypothesis-Catalog.md#catalog-of-open-research-hypotheses)

---

### Practical Cookbooks

* **[Colab & Solo Testing Cookbook](Cookbook-Colab-and-Solo-Testing.md)**
  * [Colab 20-Minute Budget](Cookbook-Colab-and-Solo-Testing.md#google-colab-20-minute-budget-management)
  * [16.5-Minute Watchdog](Cookbook-Colab-and-Solo-Testing.md#the-165-minute-safety-watchdog)
  * [Two-Tier Drive Streaming Bridge](Cookbook-Colab-and-Solo-Testing.md#two-tier-google-drive-streaming-bridge)
  * [DeltaNet Kernel Fallbacks](Cookbook-Colab-and-Solo-Testing.md#deltanet-fast-kernel-guards-and-fallbacks)

* **[Troubleshooting & FAQ](Cookbook-Troubleshooting-and-FAQ.md)**
  * [Exit Codes E02–E07](Cookbook-Troubleshooting-and-FAQ.md#master-exit-code-diagnostic-table)
  * [GPU OOM Mitigations](Cookbook-Troubleshooting-and-FAQ.md#gpu-memory-and-oom-mitigations)
  * [Verification Failure Diagnosis](Cookbook-Troubleshooting-and-FAQ.md#cryptographic-verification-failure-diagnostics)
  * [Frequently Asked Questions](Cookbook-Troubleshooting-and-FAQ.md#frequently-asked-questions)

---

### External Links

* [GitHub Repository](https://github.com/azzouzabdelhak68-ship-it/glassbox)
* [Issue Tracker](https://github.com/azzouzabdelhak68-ship-it/glassbox/issues)
* [Benchmark Leaderboard](https://github.com/azzouzabdelhak68-ship-it/glassbox/blob/main/BENCHMARKS.md)
* [Quickstart Google Colab](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/quickstart_demo.ipynb)

---
<div align="center">
  <sub>GlassBox v0.1.0 • PolyForm Noncommercial</sub>
</div>
