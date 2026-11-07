# ⚡ Test GlassBox in Google Colab (Zero Setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/quickstart_demo.ipynb)

Don't want to install anything locally or deal with Python environments and CUDA drivers? You can test GlassBox right now in your browser.

---

## What this notebook does:
When you click the badge above and hit **Run all**, it runs the whole pipeline in the cloud:

1. **Loads a small open model in seconds:** Automatically downloads `Qwen2.5-0.5B` on Colab's free CPU or GPU.
2. **Runs 3-tier speed benchmarks:**
   - **Test A (Micro-Scale):** Interactive single-prompt test ($B=1$, 0.5B model).
   - **Test B (Batched Stress):** Batched multi-layer harvesting ($B=4$, 0.5B model).
   - **Test C (T4 Tight-Fit Scaling Matrix):** Stresses a larger model (`Qwen2.5-3B` in fp16,
     ~6.5 GB weights) fitting tightly into the 15 GB T4 GPU across three distinct sub-phases
     (Small prompt $B=1$, Medium context $B=1$, and Batched $B=4$).

> 💡 **Why Scale Matters:** At micro-scale (0.5B at batch=1), GPU forward passes take `<3ms`,
> so Python hook dispatch dominates. GlassBox's sub-1% asynchronous DMA advantage shines
> at production scale ($B \ge 4$ or models $\ge 3\text{B}$) where activation payloads reach
> tens of megabytes and synchronous `.cpu()` stalls the CUDA pipeline.
3. **Inspects what neurons are doing:** Shows layer-by-layer predictions (Logit Lens) and top-firing neurons as tokens pass through the prompt.
4. **Opens the visual report:** Renders the interactive single-file HTML report right inside the notebook.
5. **Exports complete run artifacts & benchmark data:** Packages `token_level_telemetry.json`,
   `frontier_llm_prompt.md` (embedding empirical scaling matrices), `benchmark_metrics.json`
   (covering Test A, Test B, and Test C), `t4_scaling_matrix.csv`, and `calm_viewer.html`
   into a single `.zip` bundle for 1-click browser download and Google Drive auto-save.
6. **1-Click Download & Community Discussions:** Auto-downloads your artifacts (.ZIP), auto-saves
   to Google Drive, and generates a pre-formatted post with all benchmark tables for the
   GitHub Discussions community board.

---

## How to Run:
1. Click the **[Open in Colab](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/quickstart_demo.ipynb)** badge.
2. In Google Colab, click **Runtime** $\rightarrow$ **Run all** (or press `Ctrl + F9`).
3. Scroll through the live benchmark, Logit Lens table, and interactive visualizer.

No cloning, no setup. Try it out and see how it performs!

---

## Need Heavy-Duty A100 (40 GB) Research Runs?

For large-model mechanistic interpretability (`Qwen/Qwen3-14B` in native BF16 ~30 GB VRAM or `Qwen3-32B` 4-bit NF4 across a 4-module Safety & Steering circuit battery, token-by-token streaming straight to Google Drive in ~3–5 minutes under a `< 20 min` wall-clock cap), see **[A100_STRETCH_NOTE.md](A100_STRETCH_NOTE.md)** or launch directly:

[![Open A100 Stretch Test In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/a100_stretch_test.ipynb)
