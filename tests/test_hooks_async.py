"""Deep gather async tests (spec §TAP-fsst Deep, §FILE-deepbin; plan §9.2/13).

CPU-only fakes: no GPU, no weights. Run with
``BLACKBOX_GPU=cpu pytest tests/test_hooks_async.py -q``.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from glassbox import controller
from glassbox.tap import hooks
from glassbox.tap.hooks import DeepTap, attach_hooks, detach_hooks
from glassbox.watchlist import expand_preset

SMALL_MANIFEST: dict[str, Any] = {
    "model_id": "M-001",
    "checkpoint_sha256": "ab12cd34" * 8,
    "architecture": {"layers": 4, "hidden": 512, "heads": 8, "mlp_per_layer": 1024},
}

FULL_MANIFEST: dict[str, Any] = {
    "model_id": "M-001",
    "checkpoint_sha256": "ab12cd34" * 8,
    "architecture": {"layers": 32, "hidden": 4096, "heads": 32, "mlp_per_layer": 14336},
}


class FakeTensor:
    """Minimal tensor stand-in exposing ``index_select``/``tolist``/``clone``."""

    def __init__(self, values: list[float], dtype: str = "fp16") -> None:
        self._v: list[float] = list(values)
        self.dtype: str = dtype

    def index_select(self, dim: int, idx: Any) -> FakeTensor:
        assert dim == -1
        return FakeTensor([self._v[int(i)] for i in idx], dtype=self.dtype)

    def tolist(self) -> list[float]:
        return list(self._v)

    def clone(self) -> FakeTensor:
        return FakeTensor(list(self._v), dtype=self.dtype)

    def __mul__(self, scale: float) -> FakeTensor:
        return FakeTensor([v * float(scale) for v in self._v], dtype="fp16")

    def __len__(self) -> int:
        return len(self._v)


class BoomTensor(FakeTensor):
    """Tensor whose gather explodes (exercises the hook-throw path)."""

    def index_select(self, dim: int, idx: Any) -> FakeTensor:
        raise RuntimeError("boom in gather")


class FakeHandle:
    """Torch-like removable hook handle."""

    def __init__(self, module: FakeModule, hid: int) -> None:
        self._module = module
        self._hid = hid
        self.removed = False

    def remove(self) -> None:
        self.removed = True
        self._module._hooks.pop(self._hid, None)


class FakeModule:
    """Parent module stand-in with ``register_forward_hook``."""

    def __init__(self, name: str, width: int) -> None:
        self.name = name
        self.width = width
        self._hooks: dict[int, Any] = {}
        self._next = 0

    def register_forward_hook(self, fn: Any) -> FakeHandle:
        hid = self._next
        self._next += 1
        self._hooks[hid] = fn
        return FakeHandle(self, hid)

    def forward(self, values: Any, dtype: str = "fp16") -> Any:
        if hasattr(values, "ndim") or hasattr(values, "shape"):
            out = values
        else:
            out = FakeTensor(list(values), dtype=dtype)
        for fn in list(self._hooks.values()):
            fn(self, None, out)
        return out


class FakeModel:
    """Toy model: parents resolved via torch-like ``get_submodule``."""

    def __init__(self) -> None:
        self.mods: dict[str, FakeModule] = {}

    def add(self, name: str, width: int) -> FakeModule:
        module = FakeModule(name, width)
        self.mods[name] = module
        return module

    def get_submodule(self, name: str) -> FakeModule:
        return self.mods[name]


def _watchlist(addresses: list[str], **over: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "addresses": list(addresses),
        "precision": "fp16",
        "merged_window": {"start_token": 0, "end_token": 1 << 30, "every_n": 1},
        "effective_trigger": "always",
    }
    cfg.update(over)
    return cfg


def test_deep_bytes_exact_1k_x10_row_major() -> None:
    """Fake 1k watched x 10 fp16 tokens -> 20000 bytes, reshape(T, W) exact."""
    watched = [f"N-01-{i:04d}" for i in range(1, 1001)]
    model = FakeModel()
    mlp = model.add("layer-01.mlp", 4096)
    attached = attach_hooks(model, _watchlist(watched))
    tap = attached.recorder
    assert len(mlp._hooks) == 1  # one hook on the watched parent only
    for tok in range(10):
        mlp.forward([float((i % 64) + tok * 64) for i in range(4096)])
        assert tap.end_token(tok) is True
    raw = tap.to_bytes()
    assert len(raw) == 1000 * 10 * 2 == 20000
    grid = np.frombuffer(raw, dtype=np.dtype("<f2")).reshape(10, 1000)
    expected = np.asarray(
        [[float((j % 64) + tok * 64) for j in range(1000)] for tok in range(10)],
        dtype=np.float16,
    )
    assert np.array_equal(grid, expected)
    index = tap.deep_index("RUN-000184")
    assert index["triggered"] is True and index["tokens"] == 10
    assert index["columns"]["N-01-0001"] == {"offset": 0, "dtype": "fp16", "shape": [10]}
    detach_hooks(attached)
    assert len(mlp._hooks) == 0


def test_hooks_source_has_no_banned_calls() -> None:
    """No blocking host-copy call; ``synchronize`` only in the ban comment."""
    path = os.path.join(os.path.dirname(hooks.__file__), "hooks.py")
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    assert ".cpu()" not in "\n".join(lines)
    for line in lines:
        if "synchronize" in line:
            assert line.lstrip().startswith("#"), f"non-comment synchronize: {line}"


def test_ring_overflow_drops_deep_keeps_outer() -> None:
    """Ring-full flags ``ring_overflow`` and drops later Deep payloads."""
    watched = ["N-01-0001", "N-01-0002"]
    model = FakeModel()
    mlp = model.add("layer-01.mlp", 8)
    tap = DeepTap(watched, capacity_tokens=3)
    tap.attach(model)
    for tok in range(5):
        mlp.forward([float(tok), 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        tap.end_token(tok)
    assert tap.flags["ring_overflow"] is True
    assert len(tap.ring) == 3
    assert len(tap.to_bytes()) == 2 * 3 * 2
    tap.detach()


def test_hook_throw_auto_detaches_and_run_continues() -> None:
    """Hook exception auto-detaches (model restored) and Outer continues."""
    watched = ["N-01-0001"]
    model = FakeModel()
    mlp = model.add("layer-01.mlp", 4)
    attached = attach_hooks(model, _watchlist(watched))
    tap = attached.recorder
    mlp.forward([0.5, 0.0, 0.0, 0.0])
    assert tap.end_token(0) is True
    out = FakeTensor([0.0], dtype="fp16")
    boom = BoomTensor([9.0], dtype="fp16")
    for fn in list(mlp._hooks.values()):
        fn(mlp, None, boom)  # hook throws internally -> auto detach
    assert tap.flags["hook_detached"] is True
    assert len(mlp._hooks) == 0
    assert tap.detached is True
    out2 = mlp.forward([0.7, 0.0, 0.0, 0.0])  # model runs clean afterwards
    assert out2.tolist()[0] == 0.7
    assert tap.end_token(1) is False  # staging was dropped with the throw
    assert out.tolist() == [0.0]
    detach_hooks(attached)  # idempotent after auto-detach
    detach_hooks(attached)


def test_int8_dequant_logical_value() -> None:
    """int8 weights dequantize on device (int8 * scale) before the copy."""
    watched = ["N-02-0001", "N-02-0002"]
    model = FakeModel()
    mlp = model.add("layer-02.mlp", 4)
    tap = DeepTap(watched, scale=0.5)
    tap.attach(model)
    mlp.forward([4.0, -6.0, 0.0, 0.0], dtype="int8")
    assert tap.end_token(0) is True
    assert tap.ring == [[2.0, -3.0]]
    assert tap.dtype_meta["source"] == "int8_dequantized_on_device"
    assert tap.dtype_meta["stored"] == "fp16"
    tap.detach()


def test_every_n_subsamples_window() -> None:
    """``every_n=2`` seals every 2nd token only (T=5 of 10)."""
    watched = ["N-01-0001"]
    model = FakeModel()
    mlp = model.add("layer-01.mlp", 4)
    window = {"start_token": 0, "end_token": 10, "every_n": 2}
    tap = DeepTap(watched, window=window)
    tap.attach(model)
    for tok in range(10):
        mlp.forward([float(tok), 0.0, 0.0, 0.0])
        tap.end_token(tok)
    assert [row[0] for row in tap.ring] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert len(tap.to_bytes()) == 1 * 5 * 2
    tap.detach()


def test_unarmed_tokens_write_nothing() -> None:
    """``on_flag`` trigger with no flag armed seals triggered=false, no bytes."""
    watched = ["N-01-0001"]
    model = FakeModel()
    mlp = model.add("layer-01.mlp", 4)
    tap = DeepTap(watched, trigger="on_flag:bad_output")
    tap.attach(model)
    for tok in range(4):
        mlp.forward([float(tok), 0.0, 0.0, 0.0])
        assert tap.end_token(tok, flags={"bad_output": False}) is False
    assert tap.to_bytes() == b""
    assert tap.deep_index("RUN-000184")["triggered"] is False
    mlp.forward([1.0, 0.0, 0.0, 0.0])
    assert tap.end_token(4, flags={"bad_output": True}) is True
    assert tap.deep_index("RUN-000184")["triggered"] is True
    tap.detach()


def test_address_gate_range_trigger() -> None:
    """Certain-elements gate arms only while the watched value sits in range."""
    watched = ["N-24-0001"] if False else ["N-01-0001"]
    model = FakeModel()
    mlp = model.add("layer-01.mlp", 4)
    tap = DeepTap(watched, trigger={"address": watched[0], "lo": 0.8, "hi": 1.2})
    tap.attach(model)
    for tok, val in enumerate([0.1, 0.9, 2.0]):
        mlp.forward([val, 0.0, 0.0, 0.0])
        tap.end_token(tok)
    assert [row[0] for row in tap.ring] == [0.9]
    tap.detach()


def test_pilot_sparse_1k_seeded_250x4() -> None:
    """Pilot preset expands deterministically: 250 seeded neurons x 4 layers."""
    first = expand_preset("pilot-sparse-1k", FULL_MANIFEST)
    second = expand_preset("pilot-sparse-1k", FULL_MANIFEST)
    assert first == second
    assert len(first) == 1000
    for layer in (8, 16, 24, 31):
        got = [a for a in first if a.startswith(f"N-{layer:02d}-")]
        assert len(got) == 250, f"layer {layer}: {len(got)}"
    controller.validate_watchlist  # controller validates presets vs manifest
    from glassbox.watchlist import validate_addresses

    validate_addresses(first, FULL_MANIFEST)


def test_validate_watchlist_with_preset_and_add_patch(tmp_path: Any) -> None:
    """YAML preset + ``add:`` patch validates vs the manifest (E04 fail fast)."""
    import yaml

    path = os.path.join(str(tmp_path), "w.yaml")
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {
                "version": 1,
                "model_id": "M-001",
                "checkpoint_sha256": SMALL_MANIFEST["checkpoint_sha256"],
                "watchlist_id": "test-v1",
                "trigger": "always",
                "window": {"start_token": 0, "end_token": 10, "every_n": 1},
                "precision": "fp16",
                "entries": [
                    {"extends": "attention_only", "watch": ["N-02-0001..0003"]},
                    {"addresses": ["LOGITS"]},
                ],
            },
            fh,
        )
    addrs = controller.validate_watchlist(path, SMALL_MANIFEST, add=["N-02-0004"])
    assert "N-02-0001" in addrs and "N-02-0004" in addrs and "LOGITS" in addrs
    assert addrs == sorted(set(addrs), key=addrs.index)  # order kept, deduped


def test_zero_sync_logits_hook() -> None:
    """Logits hook stores raw tensors without hot-path sync; materializes lazily on CPU."""
    import torch

    tap = hooks.DeepTap(["LOGITS", "LOGITS:42", "LOGITS-ENTROPY"])
    model = FakeModel()
    logits_mod = model.add("logits", 100)
    tap.attach(model)

    logits_tensor = torch.zeros((1, 100), dtype=torch.float32)
    logits_tensor[0, 42] = 5.0
    logits_tensor[0, 7] = 3.0
    logits_tensor[0, 99] = 1.0

    logits_mod.forward(logits_tensor)
    tap.end_token(0)

    # Hot path verification: raw tensor stored, cache still empty before property read
    assert len(tap._raw_logits_summary) == 1
    assert len(tap._cached_logits_summary) == 0

    # Lazy read materializes Python numbers without forward-pass barrier
    summary = tap.logits_summary
    assert len(summary) == 1
    assert summary[0]["top_token_ids"][0] == 42
    assert summary[0]["top_logits"][0] == 5.0
    assert summary[0]["entropy"] > 0.0
    assert len(tap._raw_logits_summary) == 0  # drained into cached summary


def test_batched_inference_selection() -> None:
    """Batched input tensor [B, S, D] captures target batch_index cleanly."""
    import torch

    # Watch neuron N-01-0001 with batch_index=2
    tap = hooks.DeepTap(["N-01-0001"], batch_index=2)
    model = FakeModel()
    mlp = model.add("layer-01.mlp", 16)
    tap.attach(model)

    # 4 batches, 1 token, 16 hidden dim
    batch_tensor = torch.zeros((4, 1, 16), dtype=torch.float32)
    batch_tensor[0, 0, 0] = 10.0
    batch_tensor[1, 0, 0] = 20.0
    batch_tensor[2, 0, 0] = 99.5  # target batch item
    batch_tensor[3, 0, 0] = 40.0

    mlp.forward(batch_tensor)
    tap.end_token(0)

    assert len(tap.ring) == 1
    assert tap.ring[0][0] == 99.5
