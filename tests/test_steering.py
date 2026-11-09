from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from glassbox import (
    AblationTarget,
    GlassBox,
    SteeringController,
    SteeringVector,
    ablate,
    steer,
)


class DummyMLP(nn.Module):
    def __init__(self, hidden_dim: int = 16, intermediate_dim: int = 32):
        super().__init__()
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        nn.init.constant_(self.up_proj.weight, 1.0)
        nn.init.constant_(self.down_proj.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(x)))


class DummyAttention(nn.Module):
    def __init__(self, hidden_dim: int = 16, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.constant_(self.o_proj.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass through o_proj
        return self.o_proj(x)


class DummyLayer(nn.Module):
    def __init__(self, hidden_dim: int = 16, num_heads: int = 4, intermediate_dim: int = 32):
        super().__init__()
        self.self_attn = DummyAttention(hidden_dim, num_heads)
        self.mlp = DummyMLP(hidden_dim, intermediate_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return (x,)


class DummyTransformer(nn.Module):
    def __init__(
        self,
        num_layers: int = 2,
        hidden_dim: int = 16,
        num_heads: int = 4,
        intermediate_dim: int = 32,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [DummyLayer(hidden_dim, num_heads, intermediate_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            out = layer(x)
            x = out[0]
        return x


def test_steering_vector_validation() -> None:
    # Valid construction
    sv = SteeringVector(layer=1, vector=[1.0, 2.0], scale=1.5)
    assert sv.layer == 1
    assert sv.scale == 1.5

    # Invalid layer
    with pytest.raises(ValueError, match="layer index must be >= 1"):
        SteeringVector(layer=0, vector=[1.0])

    # Invalid component
    with pytest.raises(ValueError, match="bad component"):
        SteeringVector(layer=1, vector=[1.0], component="invalid")

    # Invalid mode
    with pytest.raises(ValueError, match="bad mode"):
        SteeringVector(layer=1, vector=[1.0], mode="invalid")


def test_ablation_target_validation() -> None:
    # Valid construction
    at = AblationTarget(target="N-01-0005", mode="zero")
    assert at.target == "N-01-0005"
    assert at.mode == "zero"

    # Invalid mode
    with pytest.raises(ValueError, match="bad ablation mode"):
        AblationTarget(target="N-01-0005", mode="bogus")

    # Orthogonal without direction
    with pytest.raises(ValueError, match="orthogonal ablation requires direction"):
        AblationTarget(target="L-01", mode="orthogonal")


def test_additive_steering_residual() -> None:
    model = DummyTransformer(num_layers=2, hidden_dim=8)
    x = torch.ones(1, 3, 8)  # [batch=1, seq=3, dim=8]

    # Baseline output
    orig_out = model(x).clone()

    # Apply steering to Layer 1 residual
    v_steer = torch.ones(8) * 5.0
    controller = SteeringController(model)
    controller.add_steering(layer=1, vector=v_steer, scale=2.0)

    with controller:
        steered_out = model(x)
        stats = controller.stats
        assert stats["invocations"] > 0
        assert stats["tokens_steered"] == 3

    # Output should change significantly
    diff = (steered_out - orig_out).abs().sum().item()
    assert diff > 1.0

    # Once detached, output reverts to baseline
    reverted_out = model(x)
    assert torch.allclose(reverted_out, orig_out)


def test_norm_preserving_steering() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=8)
    x = torch.randn(1, 4, 8)

    v_steer = torch.randn(8)
    controller = SteeringController(model)
    # Norm preserving on layer 1
    controller.add_steering(layer=1, vector=v_steer, scale=10.0, norm_preserve=True)

    with controller:
        out = model(x)
        assert out.shape == x.shape


def test_orthogonal_subspace_ablation() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=8)
    # Fix weights to identity for exact arithmetic verification
    layer = model.layers[0]
    nn.init.zeros_(layer.self_attn.o_proj.weight)
    nn.init.zeros_(layer.mlp.up_proj.weight)

    x = torch.tensor([[[2.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    v_dir = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    controller = SteeringController(model)
    controller.add_ablation("layer-01", mode="orthogonal", direction=v_dir)

    with controller:
        out = model(x)
        # First dimension along v_dir should be completely projected out (0.0)
        assert abs(out[0, 0, 0].item()) < 1e-5
        # Second dimension should remain untouched (4.0)
        assert abs(out[0, 0, 1].item() - 4.0) < 1e-5


def test_neuron_ablation_modes() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=4, intermediate_dim=8)
    x = torch.ones(1, 2, 4)

    # 1. Zero ablation on neuron 3 (N-01-0003)
    controller = SteeringController(model)
    controller.add_ablation("N-01-0003", mode="zero")

    orig_out = model(x).clone()
    with controller:
        abl_out = model(x)
        assert not torch.allclose(abl_out, orig_out)
        assert controller.stats["tokens_ablated"] == 2

    # 2. Clamping ablation
    controller.clear()
    controller.add_ablation("N-01-0001", mode="clamp_max", clamp_max=0.5)
    with controller:
        clamp_out = model(x)
        assert clamp_out.shape == x.shape


def test_attention_head_ablation() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=8, num_heads=2)  # head_dim = 4
    x = torch.ones(1, 2, 8)

    orig_out = model(x).clone()

    controller = SteeringController(model)
    # Ablate Head 1 (A-01-01)
    controller.add_ablation("A-01-01", mode="zero")

    with controller:
        abl_out = model(x)
        assert not torch.allclose(abl_out, orig_out)


def test_token_mode_filtering() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=4)

    # Prefill tensor (seq_len = 3) and Decode tensor (seq_len = 1)
    x_prefill = torch.ones(1, 3, 4)
    x_decode = torch.ones(1, 1, 4)

    v = torch.ones(4) * 2.0
    controller = SteeringController(model)
    # Steer only decode steps
    controller.add_steering(layer=1, vector=v, scale=1.0, mode="decode")

    with controller:
        # In prefill (seq=3), steering should be skipped
        _ = model(x_prefill)
        assert controller.stats["tokens_steered"] == 0

        # In decode (seq=1), steering should engage
        _ = model(x_decode)
        assert controller.stats["tokens_steered"] == 1


def test_convenience_factories() -> None:
    sv = steer(layer=2, vector=[0.5, 0.5], scale=-1.0, norm_preserve=True)
    assert isinstance(sv, SteeringVector)
    assert sv.layer == 2
    assert sv.scale == -1.0
    assert sv.norm_preserve is True

    at = ablate("N-02-0010", mode="zero")
    assert isinstance(at, AblationTarget)
    assert at.target == "N-02-0010"
    assert at.mode == "zero"


def test_unified_watch_and_steer(tmp_path: Any) -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=4)
    x = torch.ones(1, 2, 4)

    wl_file = tmp_path / "test_wl.yaml"
    wl_file.write_text(
        "version: 1\n"
        "model_id: M-001\n"
        "watchlist_id: test-wl\n"
        "checkpoint_sha256: '0000'\n"
        "trigger: always\n"
        "window: {start_token: 0, end_token: 511, every_n: 1}\n"
        "precision: fp16\n"
        "entries:\n"
        "  - addresses: [N-01-0001, L-01]\n"
    )

    v = [1.0, 1.0, 1.0, 1.0]
    # Test GlassBox.watch with simultaneous steer and ablate arguments
    with GlassBox.watch(
        str(wl_file),
        model=model,
        steer=steer(layer=1, vector=v, scale=0.5),
        ablate=ablate("N-01-0001", mode="zero"),
        force=True,
    ) as gb:
        st = gb.steering
        assert st is not None
        assert st._attached is True
        out = model(x)
        assert out.shape == x.shape

    # After context exit, steering is cleanly detached
    assert st._attached is False
    assert gb.steering is None


# ---------------------------------------------------------------------------
# v2.1 additions: per-head mean-ablation vectors and pulsed steering
# ---------------------------------------------------------------------------


def test_head_mean_ablation_with_vector() -> None:
    """mode='mean' accepts a [head_dim] vector and writes it into the head's slice."""
    model = DummyTransformer(num_layers=1, hidden_dim=8, num_heads=2)  # head_dim = 4
    x = torch.ones(1, 2, 8)
    mean_vec = torch.tensor([1.0, 2.0, 3.0, 4.0])

    controller = SteeringController(model)
    controller.add_ablation("A-01-01", mode="mean", mean_value=mean_vec)

    captured: dict[str, torch.Tensor] = {}
    with controller:
        # Registered AFTER the steering hook so it observes the already-ablated input.
        h = model.layers[0].self_attn.o_proj.register_forward_pre_hook(
            lambda _m, inputs: captured.setdefault("o_proj_in", inputs[0].clone())
        )
        try:
            model(x)
        finally:
            h.remove()

    o_in = captured["o_proj_in"]
    assert torch.allclose(o_in[0, 0, :4], mean_vec)  # head 1 replaced by mean vector
    assert torch.allclose(o_in[0, 1, :4], mean_vec)  # every position
    assert torch.allclose(o_in[0, :, 4:], torch.ones(2, 4))  # head 2 untouched


def test_head_mean_ablation_scalar_still_works() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=8, num_heads=2)
    x = torch.ones(1, 2, 8)
    controller = SteeringController(model)
    controller.add_ablation("A-01-02", mode="mean", mean_value=0.25)

    captured: dict[str, torch.Tensor] = {}
    with controller:
        h = model.layers[0].self_attn.o_proj.register_forward_pre_hook(
            lambda _m, inputs: captured.setdefault("o_proj_in", inputs[0].clone())
        )
        try:
            model(x)
        finally:
            h.remove()
    assert torch.allclose(captured["o_proj_in"][0, :, 4:], torch.full((2, 4), 0.25))


def test_pulse_validation() -> None:
    with pytest.raises(ValueError, match="pulse_steps must be >= 1"):
        SteeringVector(layer=1, vector=[1.0], pulse_steps=0)
    with pytest.raises(ValueError, match="pulse_start_step requires pulse_steps"):
        SteeringVector(layer=1, vector=[1.0], pulse_start_step=3)
    with pytest.raises(ValueError, match="pulse_start_step must be >= 1"):
        SteeringVector(layer=1, vector=[1.0], pulse_steps=2, pulse_start_step=0)
    sv = steer(layer=1, vector=[1.0], pulse_steps=2, pulse_start_step=4)
    assert sv.pulse_steps == 2 and sv.pulse_start_step == 4


def test_pulse_fixed_start_step_fires_exactly_n_decode_steps() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=4)
    controller = SteeringController(model)
    controller.add_steering(
        layer=1, vector=torch.ones(4), scale=1.0, pulse_steps=2, pulse_start_step=2
    )
    with controller:
        model(torch.ones(1, 5, 4))  # prefill: never steered
        assert controller.stats["tokens_steered"] == 0
        applied_per_step = []
        for _ in range(5):  # 5 decode steps
            before = controller.stats["tokens_steered"]
            model(torch.ones(1, 1, 4))
            applied_per_step.append(controller.stats["tokens_steered"] - before)
    # decode step 1: idle, steps 2-3: pulse, steps 4-5: disarmed
    assert applied_per_step == [0, 1, 1, 0, 0]
    assert controller.stats["pulses_applied"] == 2


def test_pulse_external_fire_and_prefill_reset() -> None:
    model = DummyTransformer(num_layers=1, hidden_dim=4)
    controller = SteeringController(model)
    controller.add_steering(layer=1, vector=torch.ones(4), scale=1.0, pulse_steps=3)
    with controller:
        model(torch.ones(1, 4, 4))  # prefill
        model(torch.ones(1, 1, 4))  # decode 1: not armed
        model(torch.ones(1, 1, 4))  # decode 2: not armed
        assert controller.stats["tokens_steered"] == 0

        assert controller.fire_pulse() == 1  # e.g. LogitsProcessor saw the trigger text
        for _ in range(5):
            model(torch.ones(1, 1, 4))
        assert controller.stats["tokens_steered"] == 3  # exactly pulse_steps, then disarmed

        # A new prefill is a new generation: arm state is cleared.
        controller.fire_pulse()
        model(torch.ones(1, 4, 4))
        before = controller.stats["tokens_steered"]
        model(torch.ones(1, 1, 4))
        assert controller.stats["tokens_steered"] == before


def test_non_pulsed_steering_unchanged() -> None:
    """Vectors without pulse_steps keep the original always-on behaviour."""
    model = DummyTransformer(num_layers=1, hidden_dim=4)
    controller = SteeringController(model)
    controller.add_steering(layer=1, vector=torch.ones(4), scale=1.0, mode="decode")
    with controller:
        for _ in range(3):
            model(torch.ones(1, 1, 4))
    assert controller.stats["tokens_steered"] == 3
    assert controller.fire_pulse() == 0
