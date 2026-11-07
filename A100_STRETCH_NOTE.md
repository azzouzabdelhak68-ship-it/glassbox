# Personal Note: A100 (40 GB) Safety & Steering Stretch Test

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/a100_stretch_test.ipynb)

This is your personal runbook and architecture note for running a heavy, research-grade mechanistic interpretability stretch test on a **Google Colab NVIDIA A100 (40 GB VRAM)** in **under 20 minutes**, streaming token-by-token internal states straight into your Google Drive.

---

## 1. Quick Launch Checklist (Colab A100)

1. Click **[Open in Colab](https://colab.research.google.com/github/azzouzabdelhak68-ship-it/glassbox/blob/main/notebooks/a100_stretch_test.ipynb)** above.
2. In Colab, go to **Runtime -> Change runtime type -> Hardware accelerator: A100 GPU** (40 GB VRAM).
3. Click **Runtime -> Run all** (`Ctrl + F9`) and approve the Google Drive mount prompt when it pops up in Cell 2.
4. Sit back. The watchdog timer and storage guard manage the rest automatically.

---

## 2. Locked-In Specs & Guardrails

| Parameter | Configuration | Why It Is Set This Way |
| :--- | :--- | :--- |
| **Target Model (Default)** | `Qwen/Qwen3-14B` (**Native BF16**, ~29.6 GB VRAM) | Uses **100% hardware A100 BF16 Tensor Cores + FlashAttention-2 (`sdpa`)** at **50–70 tok/s (~2.5s per prompt)** with ~10 GB VRAM headroom and zero quantization distortion on safety neurons. One-click dropdown also supports **`unsloth/Qwen3-32B-bnb-4bit`** (32B NF4, ~19 GB VRAM, ~30 tok/s). |
| **Why `Qwen3.8-27B` Was Guarded** | Auto-detects `fla` + `causal_conv1d` CUDA wheels | `Qwen/Qwen3.8-27B` uses 48 Gated DeltaNet (`linear_attn`) layers that **silently fall back to pure-Python token loops (< 0.5 tok/s)** on Colab when `flash-linear-attention` / `causal-conv1d` C++ wheels aren't compiled, and `bitsandbytes` `load_in_8bit` outlier splitting stalls Tensor Cores 5x–10x. The loader detects missing kernels automatically so Cell 5 never hangs. |
| **Research Task** | **Safety & Steering Battery** | Contrastive Refusal vs. Benign lexical twins, Sycophancy vs. Neutral stance pairs, TruthfulQA myth-traps, and Hypothetical/Roleplay framing probes (completes in **~2–4 minutes** on A100, and supports clicking **Stop** early at any prompt with clean auto-flush to Drive). |
| **Telemetry Density** | **High-Density (~25 KB/token)** | 16 depth-spaced layers, **256 MLP neurons/layer** (4,096 neurons/tok), all attention/DeltaNet head norms, residual norms + 16 hidden dims, and **Top-5 Logit Lens** per layer for every prompt and generated token. |
| **Runtime Cap** | **< 20 Minutes Hard Limit** | Wall-clock watchdog starts in Cell 1 and stops launching new prompts at **16.5 minutes** (with dynamic token trimming near the cutoff) so analysis & sealing always finish under 20 minutes. |
| **Drive Storage Cap** | **Pass 1 GB -> Abrupt Stop at 2 GB** | Streams `telemetry_chunk_XXXX.jsonl` directly to Drive every 50 tokens (`os.fsync`). Crosses **1.0 GB** without interruption; if total session folder hits **2.0 GB (`2,147,483,648` bytes)**, it **aborts immediately** mid-stream. |

---

## 3. How Data Streams Directly to Your Google Drive

Instead of holding gigabytes in RAM until the end (where a disconnect loses everything) or writing one token at a time over cloud FUSE (which chokes throughput), the notebook runs a **Two-Tier Streaming Bridge**:

```text
[ A100 GPU: Qwen3.8-27B ]
        │  GlassBox DeepTap (Async CUDA gather across 16 layers)
        ▼
[ Local RAM Ring Buffer ] ──(Every 50 tokens / ~1.25 MB)──► [ fsync Chunk Writer ]
                                                                     │
                                                                     ▼
                        /content/drive/MyDrive/GlassBox_Safety_Stretch/<SESSION_ID>/
                           ├── session_config.json
                           ├── telemetry_chunk_0001.jsonl   (~1.25 MB, 50 tokens)
                           ├── telemetry_chunk_0002.jsonl   (~1.25 MB, 50 tokens)
                           ├── ...
                           ├── sealed_runs/RUN-XXXXXX/      (deep.bin + seal.json)
                           ├── safety_circuit_neurons.csv   (Refusal vs Benign neuron deltas)
                           ├── layer_divergence_curve.csv   (Layer-by-layer norm shifts)
                           ├── safety_circuit_plots.png     (3-panel diagnostic chart)
                           ├── safety_battery_summary.json  (Speeds, refusal rates, storage)
                           ├── frontier_llm_safety_prompt.md(Ready for Gemini/Claude/GPT)
                           └── calm_viewer_latest.html      (Interactive browser viewer)
```

### What Every Single Line in `telemetry_chunk_XXXX.jsonl` Contains
Each line is one token (covering **both** input prompt tokens and generated completion tokens) so you can trace exact prefill circuit build-up right into autoregressive refusal/compliance:
- `prompt_id`, `pair_id`, `suite` (`refusal_pair`, `sycophancy_pair`, `truthfulness_probe`, `framing_resilience`), `role` (`harmful_trigger` vs `benign_twin`, `sycophantic_push` vs `neutral_control`, etc.)
- `token_pos`, `phase` (`prefill` or `decode`), `token_id`, `token_str`
- `residual_norms`: L2 norm of the residual stream at each of the 16 monitored layers
- `hidden_dims`: First 16 coordinates of the residual stream per layer (tracks directional steering shifts)
- `head_norms`: Per-head L2 norms across `self_attn` and Gated DeltaNet `linear_attn` layers
- `logit_lens`: Top-5 predicted vocabulary tokens, probabilities, and Shannon entropy at each of the 16 layers
- `top_neurons`: Top-16 highest-magnitude `(address, value)` neuron firings at this token
- `mlp_256`: Full 256-neuron activation slice across all 16 layers (`4,096` floats per token)

---

## 4. What to Look For in the Safety & Steering Results

1. **The Decision Token (`phase == "prefill"`, last prompt token):**
   - In `safety_circuit_neurons.csv`, check the neurons with the largest `abs_delta` and `cohens_d` between `harmful_trigger` and `benign_twin` pairs. Those are candidate **refusal feature / inhibition neurons**.
2. **The Refusal Flip Layer (Logit Lens):**
   - Look at `logit_lens` across the 16 layers on `harmful_trigger` vs `framing_resilience` prompts. Early layers usually predict topical completion tokens; middle-to-late layers abruptly promote refusal tokens (`"I"`, `"Sorry"`, `"cannot"`, `"unable"`).
3. **Sycophancy Drift:**
   - Compare `sycophantic_push` against `neutral_control` for the same `pair_id`. Watch where the authoritative user framing suppresses the factual token that appeared in middle layers.
4. **TruthfulQA Myth Suppression:**
   - On myth-trap prompts, check whether middle layers surface the popular misconception token in Top-5 Logit Lens before late layers override it.

---

## 5. Pulling Results Back Into Antigravity (`/gdrive`)

Once your Colab A100 run finishes writing to `/content/drive/MyDrive/GlassBox_Safety_Stretch/<SESSION_ID>/`:
- Every chunk and summary CSV/JSON is already in your Google Drive (`azzouzabdelhak68@gmail.com`).
- You can either feed `frontier_llm_safety_prompt.md` + `safety_circuit_neurons.csv` into any frontier model, or ask here in chat using `/gdrive` to inspect and analyze the latest folder directly from Drive.
