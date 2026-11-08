"""Comprehensive 4-tier unit and integration tests for engine backends (Features 1-8).

Covers:
- Tier 1: Feature Coverage (BaseEngineAdapter protocol, StepManifest, PyTorchEngineAdapter,
          VLLMEngineAdapter, GlassBoxWorkerExtension, SGLangEngineAdapter, Registry,
          watch(), blackbox)
- Tier 2: Boundary & Corner Cases (empty/single/extreme manifests, duplicate attach/detach,
          invalid backends)
- Tier 3: Cross-Feature Combinations (custom backend registration, JSON roundtrip,
          adapter with steering)
- Tier 4: Real-World Scenarios (continuous batching simulation, multi-turn decoding manifests,
          worker lifecycle)
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn

import blackbox
from glassbox import GlassBox, watch
from glassbox.backends import (
    BaseEngineAdapter,
    GlassBoxWorkerExtension,
    PyTorchEngineAdapter,
    SGLangEngineAdapter,
    StepManifest,
    VLLMEngineAdapter,
    detect_backend,
    get_backend,
    list_backends,
    register_backend,
    resolve_backend,
)
from glassbox.steering import SteeringVector

# ==============================================================================
# Helper Mock Objects and Models
# ==============================================================================


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
    """Small transformer model matching GlassBox address conventions (L-01, N-01-xxxx, etc.)."""

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


class DummyVLLMEngine:
    """Mock representing a vLLM LLM or LLMEngine instance."""

    def __init__(self) -> None:
        self.llm_engine = MagicMock()
        self.model_executor = MagicMock()


class DummySGLangEngine:
    """Mock representing an SGLang SRTServer or Engine instance."""

    def __init__(self) -> None:
        self.tokenizer_manager = MagicMock()
        self.server_args = MagicMock()


class CustomTestAdapter(BaseEngineAdapter):
    """Custom adapter for testing registration mechanics."""

    backend_name: ClassVar[str] = "custom_test"

    @classmethod
    def validate_engine(cls, engine: Any) -> bool:
        return getattr(engine, "is_custom_mock", False) is True

    def attach(self, engine: Any, watchlist: dict[str, Any], **kwargs: Any) -> Any:
        self._is_attached = True
        return self

    def detach(self) -> None:
        self._is_attached = False

    def flush(self) -> None:
        pass


# ==============================================================================
# Tier 1: Feature Coverage (Features 1-8)
# ==============================================================================


class TestFeature1BaseEngineAdapterProtocol:
    """Feature 1: BaseEngineAdapter protocol and lifecycle."""

    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            BaseEngineAdapter()  # type: ignore[abstract]

    def test_concrete_subclass_lifecycle(self):
        adapter = CustomTestAdapter(tp_rank=1, tp_size=4, extra_flag=True)
        assert adapter.tp_rank == 1
        assert adapter.tp_size == 4
        assert adapter.kwargs.get("extra_flag") is True
        assert not adapter.is_attached
        assert adapter.step_manifests == []

        dummy_engine = MagicMock()
        dummy_engine.is_custom_mock = True
        assert CustomTestAdapter.validate_engine(dummy_engine)

        adapter.attach(dummy_engine, {"watchlist": "test"})
        assert adapter.is_attached

        adapter.flush()
        adapter.detach()
        assert not adapter.is_attached

    def test_default_rank_and_size_clamping(self):
        adapter = CustomTestAdapter(tp_rank=0, tp_size=0)
        assert adapter.tp_size == 1  # clamped to max(1, ...)
        assert adapter.tp_rank == 0


class TestFeature2BackendRegistryAndAutoDetection:
    """Feature 2: Backend registry, discovery, and auto-detection."""

    def test_builtin_backends_are_registered(self):
        registered = list_backends()
        assert "pytorch" in registered
        assert "vllm" in registered
        assert "sglang" in registered

    def test_get_backend_retrieval(self):
        assert get_backend("pytorch") is PyTorchEngineAdapter
        assert get_backend("PyTorch") is PyTorchEngineAdapter
        assert get_backend("VLLM") is VLLMEngineAdapter
        assert get_backend("SGLANG") is SGLangEngineAdapter

    def test_get_backend_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown backend"):
            get_backend("nonexistent_serving_backend_xyz")

    def test_custom_backend_registration_and_decorator(self):
        @register_backend("tier1_custom")
        class SubAdapter(BaseEngineAdapter):
            backend_name: ClassVar[str] = "tier1_custom"

            @classmethod
            def validate_engine(cls, engine: Any) -> bool:
                return False

            def attach(self, engine: Any, watchlist: dict[str, Any], **kwargs: Any) -> Any:
                return self

            def detach(self) -> None:
                pass

            def flush(self) -> None:
                pass

        assert "tier1_custom" in list_backends()
        assert get_backend("tier1_custom") is SubAdapter

    def test_detect_backend_auto(self):
        torch_mod = DummyTransformer()
        vllm_eng = DummyVLLMEngine()
        sglang_eng = DummySGLangEngine()

        assert detect_backend(torch_mod) == "pytorch"
        assert detect_backend(vllm_eng) == "vllm"
        assert detect_backend(sglang_eng) == "sglang"
        assert detect_backend(None) == "pytorch"

    def test_resolve_backend_with_auto(self):
        vllm_eng = DummyVLLMEngine()
        cls_resolved = resolve_backend("auto", vllm_eng)
        assert cls_resolved is VLLMEngineAdapter

        torch_mod = DummyTransformer()
        cls_resolved = resolve_backend("auto", torch_mod)
        assert cls_resolved is PyTorchEngineAdapter

        explicit_cls = resolve_backend("sglang")
        assert explicit_cls is SGLangEngineAdapter


class TestFeature3PyTorchEngineAdapter:
    """Feature 3: PyTorchEngineAdapter preserving single-GPU execution."""

    def test_validate_engine(self):
        model = DummyTransformer()
        assert PyTorchEngineAdapter.validate_engine(model)
        assert not PyTorchEngineAdapter.validate_engine(None)
        assert not PyTorchEngineAdapter.validate_engine(DummyVLLMEngine())
        assert not PyTorchEngineAdapter.validate_engine(DummySGLangEngine())

    def test_attach_and_detach_single_gpu(self):
        model = DummyTransformer()
        adapter = PyTorchEngineAdapter(tp_rank=0, tp_size=1)
        adapter.attach(model, {"trigger": "always", "window": "all", "addresses": ["L-01"]})
        assert adapter.is_attached
        assert adapter.engine is model
        assert adapter.tap is not None

        # Verify forward execution still works identically
        inp = torch.randn(2, 16)
        out1 = model(inp)
        assert out1.shape == (2, 16)

        adapter.flush()
        adapter.detach()
        assert not adapter.is_attached

    def test_attach_with_steering_vectors(self):
        model = DummyTransformer()
        adapter = PyTorchEngineAdapter(tp_rank=0, tp_size=1)
        s_vec = SteeringVector(layer=1, vector=[0.1] * 16, scale=1.5)
        adapter.attach(model, {"addresses": ["L-01"]}, steer=[s_vec])
        assert adapter.steering is not None
        assert adapter.is_attached
        adapter.detach()


class TestFeature4VLLMEngineAdapterAndWorkerExtension:
    """Feature 4: VLLMEngineAdapter and GlassBoxWorkerExtension."""

    def test_vllm_validate_engine(self):
        engine = DummyVLLMEngine()
        assert VLLMEngineAdapter.validate_engine(engine)
        assert not VLLMEngineAdapter.validate_engine(DummyTransformer())
        assert not VLLMEngineAdapter.validate_engine(None)

    def test_worker_extension_initialization(self):
        ext = GlassBoxWorkerExtension(
            watchlist={"addresses": ["L-01", "A-01-01", "N-01-0001"]},
            tp_rank=0,
            tp_size=2,
        )
        assert ext.tp_rank == 0
        assert ext.tp_size == 2
        assert ext.router is not None

    def test_worker_extension_step_manifest_extraction(self):
        ext = GlassBoxWorkerExtension(
            watchlist={"addresses": ["L-01"]},
            tp_rank=0,
            tp_size=1,
        )

        batch_meta = SimpleNamespace(
            req_ids=["req_1", "req_2"],
            query_start_loc=torch.tensor([0, 16, 32], dtype=torch.int32),
            num_tokens=32,
        )

        manifest = ext.extract_step_manifest(batch_meta, step_id=42)
        assert isinstance(manifest, StepManifest)
        assert manifest.step_id == 42
        assert manifest.req_ids == ["req_1", "req_2"]
        assert list(manifest.query_start_loc) == [0, 16, 32]
        assert manifest.actual_num_tokens == 32

    def test_worker_extension_harvest_manifests(self):
        ext = GlassBoxWorkerExtension(watchlist={}, tp_rank=0, tp_size=1)
        ext.extract_step_manifest(
            {"req_ids": ["r1"], "query_start_loc": [0, 5], "num_tokens": 5}, step_id=1
        )
        ext.extract_step_manifest(
            {"req_ids": ["r2"], "query_start_loc": [0, 8], "num_tokens": 8}, step_id=2
        )

        harvested = list(ext.step_manifests)
        assert len(harvested) == 2
        assert harvested[0].step_id == 1
        assert harvested[1].step_id == 2


class TestFeature5SGLangEngineAdapter:
    """Feature 5: SGLangEngineAdapter and RadixAttention metadata extraction."""

    def test_sglang_validate_engine(self):
        engine = DummySGLangEngine()
        assert SGLangEngineAdapter.validate_engine(engine)
        assert not SGLangEngineAdapter.validate_engine(DummyTransformer())
        assert not SGLangEngineAdapter.validate_engine(None)

    def test_extract_step_manifest_from_radix_dict(self):
        adapter = SGLangEngineAdapter(tp_rank=0, tp_size=2)
        batch_meta = {
            "req_pool_indices": [101, 102],
            "extend_start_loc": [0, 12, 28],
            "seq_lens": [12, 16],
        }
        manifest = adapter.extract_step_manifest(batch_meta, step_id=10)
        assert manifest.step_id == 10
        assert manifest.req_ids == [101, 102]
        assert list(manifest.query_start_loc) == [0, 12, 28]
        assert manifest.actual_num_tokens == 28

    def test_extract_step_manifest_from_object_attributes(self):
        adapter = SGLangEngineAdapter(tp_rank=0, tp_size=1)
        mock_meta = SimpleNamespace(
            req_pool_indices=["prompt_alpha", "prompt_beta"],
            extend_start_loc=np.array([0, 5, 15], dtype=np.int64),
            logical_positions=list(range(15)),
            num_tokens=15,
        )

        manifest = adapter.extract_step_manifest(mock_meta, step_id=7)
        assert manifest.step_id == 7
        assert manifest.req_ids == ["prompt_alpha", "prompt_beta"]
        assert manifest.actual_num_tokens == 15


class TestFeature6StepManifestGeneration:
    """Feature 6: StepManifest dataclass fields, auto-inference, and serialization."""

    def test_step_manifest_fields_and_auto_num_tokens(self):
        m = StepManifest(
            step_id=100,
            req_ids=["req_a", "req_b"],
            query_start_loc=[0, 8, 20],
        )
        assert m.step_id == 100
        assert m.req_ids == ["req_a", "req_b"]
        assert m.actual_num_tokens == 20
        assert m.timestamp_ns > 0
        assert isinstance(m.flags, dict)

    def test_step_manifest_from_logical_positions(self):
        m = StepManifest(
            step_id=101,
            req_ids=["req_c"],
            logical_positions=[0, 1, 2, 3],
        )
        assert m.actual_num_tokens == 4

    def test_step_manifest_to_dict_json_serializable(self):
        m = StepManifest(
            step_id=102,
            req_ids=[1, 2],
            query_start_loc=np.array([0, 4, 10], dtype=np.int32),
            logical_positions=np.arange(10, dtype=np.int64),
            flags={"status": "ok", "nested": {"val": np.float32(3.14)}},
        )
        d = m.to_dict()
        assert isinstance(d["step_id"], int)
        assert d["query_start_loc"] == [0, 4, 10]
        assert d["actual_num_tokens"] == 10
        # Assert pure JSON serializability
        dumped = json.dumps(d)
        loaded = json.loads(dumped)
        assert loaded["step_id"] == 102
        assert loaded["flags"]["nested"]["val"] == pytest.approx(3.14, rel=1e-4)


class TestFeature7WatchSignatureUpdate:
    """Feature 7: watch() and GlassBox.watch signature extensions."""

    def test_watch_accepts_backend_tp_size_tp_rank(self):
        ctx = watch(
            "L-00",
            backend="pytorch",
            tp_size=2,
            tp_rank=1,
            force=True,
        )
        assert ctx.backend == "pytorch"
        assert ctx.tp_size == 2
        assert ctx.tp_rank == 1

    def test_watch_defaults_to_auto(self):
        ctx = watch("L-00", force=True)
        assert ctx.backend == "auto"
        assert ctx.tp_size is None
        assert ctx.tp_rank is None

    def test_glassbox_class_watch_accepts_parameters(self):
        ctx = GlassBox.watch(
            "L-00",
            backend="auto",
            tp_size=4,
            tp_rank=0,
            force=True,
        )
        assert ctx.backend == "auto"
        assert ctx.tp_size == 4
        assert ctx.tp_rank == 0


class TestFeature8BackwardCompatibilityBlackBoxShim:
    """Feature 8: Backward compatibility shim in blackbox/__init__.py."""

    def test_blackbox_exports_match_glassbox(self):
        assert hasattr(blackbox, "watch")
        assert hasattr(blackbox, "scan")
        assert hasattr(blackbox, "BlackBox")
        assert hasattr(blackbox, "GlassBox")

    def test_blackbox_backends_submodules_exist(self):
        import blackbox.backends as bb_backends
        import blackbox.backends.base as bb_base
        import blackbox.backends.pytorch as bb_pytorch
        import blackbox.backends.sglang as bb_sglang
        import blackbox.backends.vllm as bb_vllm

        assert bb_backends.BaseEngineAdapter is BaseEngineAdapter
        assert bb_base.StepManifest is StepManifest
        assert bb_pytorch.PyTorchEngineAdapter is PyTorchEngineAdapter
        assert bb_vllm.VLLMEngineAdapter is VLLMEngineAdapter
        assert bb_sglang.SGLangEngineAdapter is SGLangEngineAdapter

    def test_legacy_watch_invocation_via_blackbox(self):
        ctx = blackbox.watch("L-00", force=True)
        assert ctx.backend == "auto"
        assert ctx.watchlist == "L-00"


# ==============================================================================
# Tier 2: Boundary & Corner Cases
# ==============================================================================


class TestTier2BoundaryAndCornerCases:
    """Boundary and corner cases for engine backends."""

    def test_empty_step_manifest(self):
        m = StepManifest(step_id=0)
        assert m.step_id == 0
        assert m.req_ids == []
        assert m.query_start_loc == []
        assert m.actual_num_tokens == 0
        d = m.to_dict()
        assert d["actual_num_tokens"] == 0

    def test_single_token_step_manifest(self):
        m = StepManifest(step_id=1, req_ids=["single_req"], query_start_loc=[0, 1])
        assert m.actual_num_tokens == 1
        assert m.to_dict()["actual_num_tokens"] == 1

    def test_large_continuous_batch_manifest(self):
        num_reqs = 500
        locs = [i * 4 for i in range(num_reqs + 1)]
        req_ids = [f"req_{i}" for i in range(num_reqs)]
        m = StepManifest(step_id=999, req_ids=req_ids, query_start_loc=locs)
        assert m.actual_num_tokens == 2000
        assert len(m.req_ids) == 500

    def test_duplicate_attach_idempotence(self):
        model = DummyTransformer()
        adapter = PyTorchEngineAdapter()
        adapter.attach(model, {"addresses": ["L-01"]})
        assert adapter.is_attached
        # Calling attach again on already attached adapter re-attaches safely
        adapter.attach(model, {"addresses": ["L-01"]})
        assert adapter.is_attached
        adapter.detach()
        assert not adapter.is_attached

    def test_detach_without_attach_is_safe_noop(self):
        adapter = PyTorchEngineAdapter()
        assert not adapter.is_attached
        adapter.detach()
        assert not adapter.is_attached

    def test_custom_backend_name_case_insensitivity(self):
        @register_backend("mixed_case_BACKEND")
        class CaseAdapter(BaseEngineAdapter):
            backend_name: ClassVar[str] = "mixed_case_backend"

            @classmethod
            def validate_engine(cls, engine: Any) -> bool:
                return False

            def attach(self, engine: Any, watchlist: dict[str, Any], **kwargs: Any) -> Any:
                return self

            def detach(self) -> None:
                pass

            def flush(self) -> None:
                pass

        assert get_backend("MIXED_CASE_backend") is CaseAdapter
        assert get_backend("mixed_case_backend") is CaseAdapter


# ==============================================================================
# Tier 3: Cross-Feature Combinations
# ==============================================================================


class TestTier3CrossFeatureCombinations:
    """Pairwise combinations of backend components."""

    def test_custom_adapter_with_auto_resolve(self):
        class UniqueEngine:
            is_unique_flavor = True

        @register_backend("unique_flavor")
        class FlavorAdapter(BaseEngineAdapter):
            backend_name: ClassVar[str] = "unique_flavor"

            @classmethod
            def validate_engine(cls, engine: Any) -> bool:
                return getattr(engine, "is_unique_flavor", False) is True

            def attach(self, engine: Any, watchlist: dict[str, Any], **kwargs: Any) -> Any:
                self._is_attached = True
                return self

            def detach(self) -> None:
                self._is_attached = False

            def flush(self) -> None:
                pass

        eng = UniqueEngine()
        resolved_cls = resolve_backend("unique_flavor", eng)
        assert resolved_cls is FlavorAdapter

        adapter_inst = resolved_cls(tp_rank=0, tp_size=1)
        adapter_inst.attach(eng, {})
        assert adapter_inst.is_attached
        adapter_inst.detach()

    def test_step_manifest_nested_torch_tensors_in_flags(self):
        m = StepManifest(
            step_id=5,
            req_ids=["r1"],
            query_start_loc=[0, 3],
            flags={
                "tensor_scalar": torch.tensor(42.5),
                "tensor_list": torch.tensor([1, 2, 3]),
                "numpy_scalar": np.float64(1.234),
            },
        )
        d = m.to_dict()
        assert d["flags"]["tensor_scalar"] == pytest.approx(42.5)
        assert d["flags"]["tensor_list"] == [1, 2, 3]
        assert d["flags"]["numpy_scalar"] == pytest.approx(1.234)
        dumped = json.dumps(d)
        assert "42.5" in dumped


# ==============================================================================
# Tier 4: Real-World Scenarios
# ==============================================================================


class TestTier4RealWorldScenarios:
    """Simulated serving workflows and continuous batching lifecycles."""

    def test_continuous_batching_multi_turn_sequence(self):
        """Simulate prefill phase followed by 2 decode iterations."""
        ext = GlassBoxWorkerExtension(watchlist={"addresses": ["L-01"]}, tp_rank=0, tp_size=1)

        # Step 0: Prefill with 2 requests of prompts length 10 and 15
        ext.extract_step_manifest(
            {
                "req_ids": ["req_alpha", "req_beta"],
                "query_start_loc": [0, 10, 25],
                "num_tokens": 25,
            },
            step_id=0,
        )

        # Step 1: Decode 1 token per request
        ext.extract_step_manifest(
            {"req_ids": ["req_alpha", "req_beta"], "query_start_loc": [0, 1, 2], "num_tokens": 2},
            step_id=1,
        )

        # Step 2: req_alpha finished, req_gamma joined (1 prompt + 1 decode)
        ext.extract_step_manifest(
            {"req_ids": ["req_beta", "req_gamma"], "query_start_loc": [0, 1, 13], "num_tokens": 13},
            step_id=2,
        )

        harvested = list(ext.step_manifests)
        assert len(harvested) == 3
        assert harvested[0].actual_num_tokens == 25
        assert harvested[1].actual_num_tokens == 2
        assert harvested[2].actual_num_tokens == 13

        # All harvested manifests serialize to JSON cleanly
        for h in harvested:
            serialized = json.dumps(h.to_dict())
            assert "step_id" in serialized

    def test_pytorch_adapter_e2e_watch_execution(self, tmp_path):
        """Verify full execution through PyTorch adapter under watch context."""
        import yaml

        wl_data = {
            "version": 1,
            "model_id": "test_m",
            "checkpoint_sha256": "0" * 64,
            "watchlist_id": "test_wl",
            "trigger": "always",
            "window": {"start_token": 0, "end_token": 511, "every_n": 1},
            "precision": "fp16",
            "entries": [{"addresses": ["L-01"]}],
        }
        wl_path = tmp_path / "test_wl.yaml"
        wl_path.write_text(yaml.safe_dump(wl_data))

        model = DummyTransformer()
        with watch(str(wl_path), model=model, backend="pytorch", force=True) as ctx:
            adapter = ctx.adapter
            assert adapter is not None
            assert adapter.is_attached
            x = torch.randn(1, 16)
            _ = model(x)
        assert not adapter.is_attached
