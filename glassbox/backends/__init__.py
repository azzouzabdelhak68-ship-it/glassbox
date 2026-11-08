"""Engine backend registry and dynamic dispatcher (spec §BACKENDS).

Registers, discovers, and resolves execution adapters for PyTorch, vLLM,
and SGLang serving runtimes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, overload

from glassbox.backends.base import BaseEngineAdapter, StepManifest
from glassbox.backends.pytorch import PyTorchEngineAdapter
from glassbox.backends.sglang import SGLangEngineAdapter
from glassbox.backends.vllm import GlassBoxWorkerExtension, VLLMEngineAdapter

_REGISTRY: dict[str, type[BaseEngineAdapter]] = {}


@overload
def register_backend(name: type[BaseEngineAdapter]) -> type[BaseEngineAdapter]: ...


@overload
def register_backend(
    name: str | None = None,
) -> Callable[[type[BaseEngineAdapter]], type[BaseEngineAdapter]]: ...


def register_backend(
    name: str | type[BaseEngineAdapter] | None = None,
) -> Any:
    """Register an engine adapter class under one or more backend names."""
    if isinstance(name, type) and issubclass(name, BaseEngineAdapter):
        # Used as @register_backend without parentheses
        cls = name
        key = getattr(cls, "backend_name", cls.__name__.lower()).strip().lower()
        _REGISTRY[key] = cls
        return cls

    def _decorator(cls: type[BaseEngineAdapter]) -> type[BaseEngineAdapter]:
        key = (
            str(name).strip().lower()
            if name is not None
            else getattr(cls, "backend_name", cls.__name__.lower()).strip().lower()
        )
        _REGISTRY[key] = cls
        return cls

    return _decorator


def get_backend(name: str) -> type[BaseEngineAdapter]:
    """Look up an engine adapter by name (case-insensitive)."""
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        avail = ", ".join(repr(k) for k in sorted(_REGISTRY.keys()))
        raise KeyError(f"E02: Unknown backend {name!r}. Available backends: {avail}")
    return _REGISTRY[key]


def list_backends() -> list[str]:
    """Return a list of all currently registered backend names."""
    return sorted(_REGISTRY.keys())


def detect_backend(engine: Any) -> str:
    """Auto-detect the appropriate backend name for the given engine object."""
    if engine is None:
        return "pytorch"

    if VLLMEngineAdapter.validate_engine(engine):
        return "vllm"
    if SGLangEngineAdapter.validate_engine(engine):
        return "sglang"
    if PyTorchEngineAdapter.validate_engine(engine):
        return "pytorch"

    return "pytorch"


def resolve_backend(
    backend: str = "auto", engine: Any = None
) -> type[BaseEngineAdapter]:
    """Resolve backend name or 'auto' string to the concrete adapter class."""
    b = str(backend).strip().lower()
    if b == "auto":
        detected = detect_backend(engine)
        return get_backend(detected)
    return get_backend(b)


# Register default standard backends
register_backend("pytorch")(PyTorchEngineAdapter)
register_backend("vllm")(VLLMEngineAdapter)
register_backend("sglang")(SGLangEngineAdapter)

__all__ = [
    "BaseEngineAdapter",
    "GlassBoxWorkerExtension",
    "PyTorchEngineAdapter",
    "SGLangEngineAdapter",
    "StepManifest",
    "VLLMEngineAdapter",
    "detect_backend",
    "get_backend",
    "list_backends",
    "register_backend",
    "resolve_backend",
]
