"""Tap package: Deep gather hooks + sidecar (spec §TAP-*, plan.md §9)."""

from __future__ import annotations

from glassbox.tap.hooks import AttachedTaps, DeepTap, attach_hooks, detach_hooks
from glassbox.tap.sidecar import Sidecar, SidecarError

__all__ = [
    "AttachedTaps",
    "DeepTap",
    "Sidecar",
    "SidecarError",
    "attach_hooks",
    "detach_hooks",
]
