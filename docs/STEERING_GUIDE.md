# Real-Time Activation Steering & Causal Ablation Guide

With GlassBox **Feature 1**, researchers and engineers can transition from passive flight recording to **active causal intervention**.

GlassBox hooks directly into PyTorch execution pipelines to perform:
1. **Additive Steering**: Inject steering vectors $\mathbf{x} \leftarrow \mathbf{x} + \alpha \mathbf{v}$ at specific layers (residual stream, MLP, attention) with optional L2 norm preservation.
2. **Directional Subspace Ablation**: Project out refusal, sycophancy, or bias directions $\mathbf{x} \leftarrow \mathbf{x} - \frac{\mathbf{x} \cdot \mathbf{v}}{\|\mathbf{v}\|^2} \mathbf{v}$.
3. **Neuron Clamping / Zeroing**: Zero out, clamp, or replace individual MLP neurons (`N-LL-NNNN`) mid-generation.
4. **Attention Head Ablation**: Silence or clamp specific attention heads (`A-LL-HH`).
5. **Token Filtering**: Restrict steering to prompt prefill (`mode="prefill"`), autoregressive decode (`mode="decode"`), or specific token index positions.

---

## 1. Quickstart: Standalone Steering Controller

```python
import torch
from glassbox import SteeringController, SteeringVector, AblationTarget, steer, ablate

# Load your PyTorch model
model = ...

# Initialize the active controller
controller = SteeringController(model)

# 1. Additive Steering: inject a refusal steering vector (or subtract to bypass refusal)
controller.add_steering(
    layer=64,
    vector=u_refusal,
    scale=-1.5,           # negative scale steers AWAY from refusal
    component="residual", # "residual" | "mlp" | "attn"
    mode="decode",        # steer only during generated tokens
    norm_preserve=True,   # preserve original L2 norm to maintain perplexity stability
)

# 2. Causal Neuron Ablation: silence a specific compliance-suppressor neuron
controller.add_ablation(
    target="N-64-0024",
    mode="zero",          # "zero" | "clamp_max" | "clamp_min" | "clamp" | "mean"
)

# 3. Attention Head Ablation: silence an early tripwire head
controller.add_ablation(
    target="A-18-03",
    mode="zero",
)

# Run generation with active intervention
with controller:
    output = model.generate(...)

# Inspect intervention telemetry
print(controller.stats)
# {
#   "invocations": 64,
#   "tokens_steered": 224,
#   "tokens_ablated": 448,
#   "hooks_installed": 3
# }
```

---

## 2. Unified Watch & Steer: Simultaneous Recording and Intervention

You can record flight telemetry with `GlassBox.watch` while simultaneously applying active steering rules:

```python
from glassbox import GlassBox, steer, ablate

with GlassBox.watch(
    "watchlists/safety_steering.yaml",
    model=model,
    steer=[
        steer(layer=64, vector=u_ref, scale=-2.0, norm_preserve=True),
        steer(layer=56, vector=u_chinese_policy, scale=-1.0),
    ],
    ablate=[
        ablate("N-64-0024", mode="zero"),
        ablate("A-18-03", mode="zero"),
    ],
) as gb:
    output = model.generate(...)

# The run is cryptographically sealed in deep.bin with SHA-256 integrity,
# recording the exact steered activations!
```

---

## 3. Directional Orthogonal Subspace Ablation

To remove the model's capacity to represent a specific direction (such as refusal or sycophancy) without adding a static bias:

```python
controller = SteeringController(model)

# Mathematically guarantees: dot(x_ablated, u_refusal) == 0
controller.add_ablation(
    target="layer-64",
    mode="orthogonal",
    direction=u_refusal,
    token_mode="all",
)

with controller:
    output = model.generate(...)
```

---

## 4. Performance & Zero-Sync Design

- **In-Place Non-Blocking Transformations**: Steering and ablation hooks execute directly on GPU VRAM tensors in PyTorch's native compute stream.
- **Combined Module Multi-Plexing**: If multiple steering and ablation rules target the same layer or module, GlassBox compiles them into a single hook invocation per module, preventing execution overhead.
- **Zero Host Synchronization**: Modifying tensors on device introduces zero GPU-to-CPU stalls.
