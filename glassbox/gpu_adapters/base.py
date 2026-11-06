"""Adapter interface: 6 functions every backend implements (spec §GPU-adapters, plan §9.4).

Contract: group the watchlist by parent module, move ``idx`` to the device
once, use double-buffered pinned host memory with one copyStream + Event
per group, gather with ``index_select`` on device, and never block the
default stream. int8 weights dequantize on device (logical value stored +
``dtype_meta``). ``torch`` is imported lazily inside functions so CPU-only
hosts stay clean.
"""

from __future__ import annotations

from typing import Any


def capabilities() -> dict[str, Any]:
    """Describe backend capabilities (portable base: CPU drain semantics)."""
    return {
        "backend": "base",
        "streams": False,
        "pinned": False,
        "events": False,
        "dtypes": ["fp32", "fp16", "bf16"],
        "int8_dequant": "on_device_before_copy",
    }


def create_context(watchlist: dict[str, Any]) -> dict[str, Any]:
    """Create a backend copy context for a parsed watchlist dict."""
    addresses = [str(a) for a in watchlist.get("addresses", [])]
    return {
        "backend": "base",
        "watched": addresses,
        "W": len(addresses),
        "double_buffered": True,
        "streams": {},
        "events": {},
    }


def attach(model: Any, ctx: dict[str, Any]) -> list[Any]:
    """Attach backend gather to watched parent modules (lazy hooks import)."""
    from glassbox.tap.hooks import attach_hooks

    watchlist = {"addresses": list(ctx.get("watched", [])), "precision": "fp16"}
    attached = attach_hooks(model, watchlist)
    ctx["handles"] = attached
    return list(attached)


def detach(handles: list[Any]) -> None:
    """Detach backend gather handles; idempotent, never raises."""
    from glassbox.tap.hooks import detach_hooks

    detach_hooks(handles)


def alloc_pinned(nbytes: int) -> bytearray:
    """Allocate pinned host memory (base: portable bytearray staging)."""
    if nbytes < 0:
        raise ValueError("E02: nbytes must be non-negative")
    return bytearray(nbytes)


def stage_copy(gathered: Any, pinned: Any, stream: Any = None) -> None:
    """Async stage a device buffer into pinned host memory (base: queued copy)."""
    _ = stream
    if hasattr(pinned, "copy_"):
        pinned.copy_(gathered, non_blocking=True)
        return
    if isinstance(pinned, bytearray) and hasattr(gathered, "tobytes"):
        pinned[:] = gathered.tobytes()[: len(pinned)]
        return
    raise ValueError("E02: pinned buffer must expose copy_ or be a bytearray")
