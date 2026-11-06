"""NVIDIA backend: CUDA Stream + Event barrier + pinned memory (plan §9.4 priority #1).

``torch.cuda`` is imported lazily inside functions only, so CPU-only hosts
import this module cleanly. When CUDA is unavailable every entry point that
needs the device raises an honest ``NotImplementedError`` pointing at
``docs/ADAPTER_GUIDE.md``; :func:`capabilities` always answers (reporting
``available=False``) so callers can degrade to the ``cpu`` adapter.
"""

from __future__ import annotations

from typing import Any

_GUIDE = "docs/ADAPTER_GUIDE.md (§nvidia)"


def _require_cuda() -> Any:
    """Import torch+cuda lazily; raise honest NotImplementedError when absent."""
    try:
        import torch
    except ImportError as exc:
        raise NotImplementedError(
            f"nvidia adapter needs torch with CUDA; see {_GUIDE} (cpu adapter covers tests)"
        ) from exc
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        raise NotImplementedError(
            f"nvidia adapter needs a CUDA device; see {_GUIDE} (cpu adapter covers tests)"
        )
    return torch


def capabilities() -> dict[str, Any]:
    """Report NVIDIA capabilities; ``available`` is False on CPU-only hosts."""
    try:
        torch = _require_cuda()
        name = torch.cuda.get_device_name(0)
        return {
            "backend": "nvidia",
            "available": True,
            "device": name,
            "streams": True,
            "pinned": True,
            "events": True,
            "dtypes": ["fp32", "fp16", "bf16"],
            "int8_dequant": "on_device_before_copy",
        }
    except NotImplementedError:
        return {
            "backend": "nvidia",
            "available": False,
            "streams": True,
            "pinned": True,
            "events": True,
            "dtypes": ["fp32", "fp16", "bf16"],
            "int8_dequant": "on_device_before_copy",
        }


def create_context(watchlist: dict[str, Any]) -> dict[str, Any]:
    """Create per-group copyStreams + Events + double-buffered pinned host buffers."""
    torch = _require_cuda()
    from glassbox.tap.hooks import group_by_parent

    addresses = [str(a) for a in watchlist.get("addresses", [])]
    groups = group_by_parent(addresses)
    streams: dict[str, Any] = {}
    events: dict[str, Any] = {}
    pinned_buffers: dict[str, list[Any]] = {}
    device_idx: dict[str, Any] = {}
    for parent, cols in groups.items():
        streams[parent] = torch.cuda.Stream()
        events[parent] = torch.cuda.Event()
        pinned_buffers[parent] = [
            alloc_pinned(len(cols) * 4),
            alloc_pinned(len(cols) * 4),
        ]
        device_idx[parent] = torch.tensor(
            [local for _, local, _ in cols], dtype=torch.long, device="cuda"
        )
    return {
        "backend": "nvidia",
        "watched": addresses,
        "W": len(addresses),
        "double_buffered": True,
        "streams": streams,
        "events": events,
        "pinned_buffers": pinned_buffers,
        "device_idx": device_idx,
    }


def attach(model: Any, ctx: dict[str, Any]) -> list[Any]:
    """Attach Deep hooks with the context's copyStreams."""
    _require_cuda()
    from glassbox.tap.hooks import attach_hooks

    first_stream = next(iter(ctx.get("streams", {}).values()), None)
    watchlist = {"addresses": list(ctx.get("watched", [])), "precision": "fp16"}
    attached = attach_hooks(model, watchlist, stream=first_stream)
    ctx["handles"] = attached
    return list(attached)


def detach(handles: list[Any]) -> None:
    """Detach NVIDIA gather handles; idempotent, never raises."""
    from glassbox.tap.hooks import detach_hooks

    detach_hooks(handles)


def alloc_pinned(nbytes: int) -> Any:
    """Allocate pinned host memory (double-buffer backing store)."""
    torch = _require_cuda()
    if nbytes < 0:
        raise ValueError("E02: nbytes must be non-negative")
    return torch.empty(nbytes, dtype=torch.uint8).pin_memory()


def stage_copy(
    gathered: Any,
    pinned: Any,
    stream: Any = None,
    event: Any = None,
) -> None:
    """Async device->pinned copy on the group's copyStream with Event ordering."""
    torch = _require_cuda()
    if stream is not None:
        ready_event = event if event is not None else torch.cuda.Event()
        ready_event.record(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            stream.wait_event(ready_event)
            pinned.copy_(gathered, non_blocking=True)
            if event is not None:
                event.record(stream)
    else:
        pinned.copy_(gathered, non_blocking=True)
