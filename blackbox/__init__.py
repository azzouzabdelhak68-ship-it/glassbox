"""Backward-compatibility layer forwarding to `glassbox`."""

from __future__ import annotations

import sys

import glassbox
import glassbox.cli as _cli
import glassbox.controller as _controller
import glassbox.exceptions as _exceptions
import glassbox.gpu_adapters as _gpu_adapters
import glassbox.gpu_adapters.amd as _amd
import glassbox.gpu_adapters.apple as _apple
import glassbox.gpu_adapters.base as _base
import glassbox.gpu_adapters.cpu as _cpu
import glassbox.gpu_adapters.nvidia as _nvidia
import glassbox.lens as _lens
import glassbox.profile as _profile
import glassbox.scan as _scan
import glassbox.search as _search
import glassbox.search.index as _index
import glassbox.search.query as _query
import glassbox.storage as _storage
import glassbox.storage.layout as _layout
import glassbox.storage.seal as _seal
import glassbox.tap as _tap
import glassbox.tap.hooks as _hooks
import glassbox.tap.sidecar as _sidecar
import glassbox.to_dataframe as _to_dataframe
import glassbox.viewer as _viewer
import glassbox.watchlist as _watchlist
from glassbox import *  # noqa: F403

sys.modules["blackbox.cli"] = _cli
sys.modules["blackbox.controller"] = _controller
sys.modules["blackbox.exceptions"] = _exceptions
sys.modules["blackbox.gpu_adapters"] = _gpu_adapters
sys.modules["blackbox.gpu_adapters.amd"] = _amd
sys.modules["blackbox.gpu_adapters.apple"] = _apple
sys.modules["blackbox.gpu_adapters.base"] = _base
sys.modules["blackbox.gpu_adapters.cpu"] = _cpu
sys.modules["blackbox.gpu_adapters.nvidia"] = _nvidia
sys.modules["blackbox.lens"] = _lens
sys.modules["blackbox.profile"] = _profile
sys.modules["blackbox.scan"] = _scan
sys.modules["blackbox.search"] = _search
sys.modules["blackbox.search.index"] = _index
sys.modules["blackbox.search.query"] = _query
sys.modules["blackbox.storage"] = _storage
sys.modules["blackbox.storage.layout"] = _layout
sys.modules["blackbox.storage.seal"] = _seal
sys.modules["blackbox.tap"] = _tap
sys.modules["blackbox.tap.hooks"] = _hooks
sys.modules["blackbox.tap.sidecar"] = _sidecar
sys.modules["blackbox.to_dataframe"] = _to_dataframe
sys.modules["blackbox.viewer"] = _viewer
sys.modules["blackbox.watchlist"] = _watchlist

__all__ = list(glassbox.__all__)
