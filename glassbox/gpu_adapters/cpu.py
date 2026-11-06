"""CPU fallback backend: clone-to-queue, ~0.005 ms for 1k (plan §9.4).

Works everywhere (no CUDA, no device streams): gather stays an
``index_select`` on the CPU tensor and the copy is a queued ``clone`` into
host staging. Used for tests (``BLACKBOX_GPU=cpu pytest``) and CPU
hosts. No ``torch`` import at module top; torch tensors are handled by
duck-typing when present.
"""

from __future__ import annotations

import collections
from typing import Any


def capabilities() -> dict[str, Any]:
    """Report CPU fallback capabilities (always available)."""
    return {
        "backend": "cpu",
        "available": True,
        "streams": False,
        "pinned": False,
        "events": False,
        "dtypes": ["fp32", "fp16", "bf16"],
        "int8_dequant": "on_device_before_copy",
        "queue_ms_per_1k": 0.005,
    }


def create_context(watchlist: dict[str, Any]) -> dict[str, Any]:
    """Create a CPU queue context (a deque per watched parent group)."""
    from glassbox.tap.hooks import group_by_parent

    addresses = [str(a) for a in watchlist.get("addresses", [])]
    groups = group_by_parent(addresses)
    return {
        "backend": "cpu",
        "watched": addresses,
        "W": len(addresses),
        "double_buffered": True,
        "queues": {parent: collections.deque() for parent in groups},
    }


def attach(model: Any, ctx: dict[str, Any]) -> list[Any]:
    """Attach Deep hooks draining into the CPU queues."""
    from glassbox.tap.hooks import attach_hooks

    watchlist = {"addresses": list(ctx.get("watched", [])), "precision": "fp16"}
    attached = attach_hooks(model, watchlist)
    ctx["handles"] = attached
    return list(attached)


def detach(handles: list[Any]) -> None:
    """Detach CPU gather handles; idempotent, never raises."""
    from glassbox.tap.hooks import detach_hooks

    detach_hooks(handles)


def alloc_pinned(nbytes: int) -> bytearray:
    """Allocate host staging (plain bytearray; nothing to pin on CPU)."""
    if nbytes < 0:
        raise ValueError("E02: nbytes must be non-negative")
    return bytearray(nbytes)


def stage_copy(gathered: Any, pinned: Any, stream: Any = None) -> None:
    """Queue a clone of ``gathered`` into host staging (~0.005 ms / 1k)."""
    _ = stream
    clone = getattr(gathered, "clone", None)
    staged = clone() if callable(clone) else gathered
    if hasattr(pinned, "copy_"):
        pinned.copy_(staged, non_blocking=True)
        return
    if isinstance(pinned, bytearray) and hasattr(staged, "tobytes"):
        pinned[:] = staged.tobytes()[: len(pinned)]
        return
    raise ValueError("E02: pinned buffer must expose copy_ or be a bytearray")
