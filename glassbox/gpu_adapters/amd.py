"""AMD backend stub: HIP streams port (community adapter, plan §9.4).

Not implemented Day 1 (Linux+NVIDIA is priority #1). Port guide: copy
``blackbox/gpu_adapters/nvidia.py``, swap CUDA Streams/pinning for HIP
streams + ``hipHostMalloc`` pinned memory, keep the ``index_select`` gather
and the 6-function contract. See ``docs/ADAPTER_GUIDE.md`` (§amd).
"""

from __future__ import annotations

from typing import Any

_GUIDE = "docs/ADAPTER_GUIDE.md (§amd): copy nvidia.py, swap CUDA Stream+pin for HIP streams"


def _missing(name: str) -> NotImplementedError:
    return NotImplementedError(f"amd adapter {name}() not implemented; {_GUIDE}")


def capabilities() -> dict[str, Any]:
    """Report AMD HIP capabilities (stub: raises until ported)."""
    raise _missing("capabilities")


def create_context(watchlist: dict[str, Any]) -> dict[str, Any]:
    """Create an AMD HIP copy context (stub: raises until ported)."""
    _ = watchlist
    raise _missing("create_context")


def attach(model: Any, ctx: dict[str, Any]) -> list[Any]:
    """Attach AMD gather (stub: raises until ported)."""
    _ = (model, ctx)
    raise _missing("attach")


def detach(handles: list[Any]) -> None:
    """Detach AMD gather (stub: raises until ported)."""
    _ = handles
    raise _missing("detach")


def alloc_pinned(nbytes: int) -> Any:
    """Allocate AMD pinned host memory (stub: raises until ported)."""
    _ = nbytes
    raise _missing("alloc_pinned")


def stage_copy(gathered: Any, pinned: Any, stream: Any = None) -> None:
    """Async HIP stage copy (stub: raises until ported)."""
    _ = (gathered, pinned, stream)
    raise _missing("stage_copy")
