# ⚡ Test GlassBox in Google Colab (Zero Setup)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/quickstart_demo.ipynb)

Don't want to install anything locally or deal with Python environments and CUDA drivers? You can test GlassBox right now in your browser.

---

## What this notebook does:
When you click the badge above and hit **Run all**, it runs the whole pipeline in the cloud:

1. **Loads a small open model in seconds:** Automatically downloads `Qwen2.5-0.5B` on Colab's free CPU or GPU.
2. **Runs a live speed test:** Compares normal PyTorch vs. standard hooks vs. GlassBox async DMA so you can see the **<1% overhead** live on your screen.
3. **Inspects what neurons are doing:** Shows layer-by-layer predictions (Logit Lens) and top-firing neurons as tokens pass through the prompt.
4. **Opens the visual report:** Renders the interactive single-file HTML report right inside the notebook.
5. **Exports token-by-token telemetry & Frontier-LLM prompt:** Generates a general analysis prompt ready to paste into Claude, GPT-4o, or Gemini to analyze circuit behavior on any prompt.
6. **1-Click Download & Community Discussions:** Packages your run artifacts (.ZIP), auto-saves to Google Drive, and lets you share results to the GitHub Discussions community board.

---

## How to Run:
1. Click the **[Open in Colab](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/quickstart_demo.ipynb)** badge.
2. In Google Colab, click **Runtime** $\rightarrow$ **Run all** (or press `Ctrl + F9`).
3. Scroll through the live benchmark, Logit Lens table, and interactive visualizer.

No cloning, no setup. Try it out and see how it performs!
