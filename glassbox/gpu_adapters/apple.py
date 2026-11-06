"""Apple backend stub: MPS blit port (community adapter, plan §9.4).

Not implemented Day 1 (Linux+NVIDIA is priority #1). Port guide: copy
``blackbox/gpu_adapters/nvidia.py``, swap CUDA Streams/pinning for MPS
command-buffer blits + shared/unified host memory, keep the
``index_select`` gather and the 6-function contract.
See ``docs/ADAPTER_GUIDE.md`` (§apple).
"""

from __future__ import annotations

from typing import Any

_GUIDE = "docs/ADAPTER_GUIDE.md (§apple): copy nvidia.py, swap CUDA Stream+pin for MPS blits"


def _missing(name: str) -> NotImplementedError:
    return NotImplementedError(f"apple adapter {name}() not implemented; {_GUIDE}")


def capabilities() -> dict[str, Any]:
    """Report Apple MPS capabilities (stub: raises until ported)."""
    raise _missing("capabilities")


def create_context(watchlist: dict[str, Any]) -> dict[str, Any]:
    """Create an Apple MPS blit context (stub: raises until ported)."""
    _ = watchlist
    raise _missing("create_context")


def attach(model: Any, ctx: dict[str, Any]) -> list[Any]:
    """Attach Apple gather (stub: raises until ported)."""
    _ = (model, ctx)
    raise _missing("attach")


def detach(handles: list[Any]) -> None:
    """Detach Apple gather (stub: raises until ported)."""
    _ = handles
    raise _missing("detach")


def alloc_pinned(nbytes: int) -> Any:
    """Allocate Apple shared host memory (stub: raises until ported)."""
    _ = nbytes
    raise _missing("alloc_pinned")


def stage_copy(gathered: Any, pinned: Any, stream: Any = None) -> None:
    """Async MPS blit stage copy (stub: raises until ported)."""
    _ = (gathered, pinned, stream)
    raise _missing("stage_copy")
