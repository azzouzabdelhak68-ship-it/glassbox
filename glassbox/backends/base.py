"""Base abstractions for engine adapters and execution manifests (spec §BACKENDS).

Provides the abstract base adapter protocol (:class:`BaseEngineAdapter`) and
the continuous batching step manifest (:class:`StepManifest`).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


def _to_json_primitive(val: Any) -> Any:
    """Recursively convert NumPy arrays, tensors, and scalars to pure JSON primitives."""
    if val is None:
        return None
    if isinstance(val, (bool, str)):
        return val
    if isinstance(val, (int, np.integer)):
        return int(val)
    if isinstance(val, (float, np.floating)):
        return float(val)
    if hasattr(val, "tolist") and callable(val.tolist):
        return _to_json_primitive(val.tolist())
    if isinstance(val, (list, tuple)):
        return [_to_json_primitive(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _to_json_primitive(v) for k, v in val.items()}
    if hasattr(val, "item") and callable(val.item):
        return _to_json_primitive(val.item())
    return val


@dataclass
class StepManifest:
    """Per-step continuous batching execution manifest.

    Captures prompt/decode boundary indices, active request IDs, and logical
    token positions across dynamic continuous batching engines (vLLM, SGLang)
    without synchronizing or replaying CUDA Graphs.
    """

    step_id: int = 0
    req_ids: list[str | int] = field(default_factory=list)
    query_start_loc: list[int] | np.ndarray = field(default_factory=list)
    logical_positions: list[int] | np.ndarray | None = None
    actual_num_tokens: int = 0
    timestamp_ns: int = 0
    flags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp_ns == 0:
            self.timestamp_ns = time.time_ns()
        if self.actual_num_tokens == 0:
            if (
                isinstance(self.query_start_loc, (list, tuple, np.ndarray))
                and len(self.query_start_loc) > 1
            ):
                self.actual_num_tokens = int(self.query_start_loc[-1])
            elif self.logical_positions is not None and len(self.logical_positions) > 0:
                self.actual_num_tokens = len(self.logical_positions)

    def to_dict(self) -> dict[str, Any]:
        """Convert manifest to pure JSON-serializable primitives."""
        return {
            "step_id": int(self.step_id),
            "req_ids": _to_json_primitive(self.req_ids),
            "query_start_loc": _to_json_primitive(self.query_start_loc),
            "logical_positions": _to_json_primitive(self.logical_positions),
            "actual_num_tokens": int(self.actual_num_tokens),
            "timestamp_ns": int(self.timestamp_ns),
            "flags": _to_json_primitive(self.flags),
        }


class BaseEngineAdapter(ABC):
    """Abstract base protocol for engine execution adapters."""

    backend_name: ClassVar[str] = "base"

    def __init__(self, tp_rank: int = 0, tp_size: int = 1, **kwargs: Any) -> None:
        self.tp_rank = int(tp_rank)
        self.tp_size = max(1, int(tp_size))
        self.kwargs = dict(kwargs)
        self._is_attached: bool = False
        self._step_manifests: list[StepManifest] = []

    @classmethod
    @abstractmethod
    def validate_engine(cls, engine: Any) -> bool:
        """Return True if this adapter can instrument the given engine object."""
        ...

    @abstractmethod
    def attach(self, engine: Any, watchlist: dict[str, Any], **kwargs: Any) -> Any:
        """Attach instrumentation hooks/extensions to the engine."""
        ...

    @abstractmethod
    def detach(self) -> None:
        """Detach all instrumentation hooks/extensions and restore engine state."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending staging sequences or activation records."""
        ...

    @property
    def is_attached(self) -> bool:
        """Whether the adapter is currently attached to an engine."""
        return self._is_attached

    @property
    def step_manifests(self) -> list[StepManifest]:
        """List of step manifests recorded during execution."""
        return self._step_manifests

    def record_step(self, manifest: StepManifest) -> None:
        """Record an execution step manifest."""
        self._step_manifests.append(manifest)
