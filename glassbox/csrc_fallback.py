"""Pure-Python / NumPy and Triton fallbacks for glassbox::tap operator (spec §GRAPH-WIRE-M3).

Provides 100% numerical and behavioral parity in environments without CUDA/nvcc,
while registering the custom op via torch.library for CUDA Graph / Torch Dynamo tracing
without graph breaks.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Try importing Triton for GPU kernel execution
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


# -----------------------------------------------------------------------------
# Triton In-Graph Gather Kernel (CUDA Graph safe)
# -----------------------------------------------------------------------------
if _TRITON_AVAILABLE:

    @triton.jit
    def _tap_gather_triton_kernel(
        activations_ptr,
        feature_indices_ptr,
        staging_slab_ptr,
        ctl_ptr,
        col_offset,
        stride_act_row,
        stride_act_col,
        stride_slab_slot,
        stride_slab_row,
        stride_slab_col,
        stride_feat,
        act_rows,
        act_cols,
        slab_slots,
        slab_rows,
        slab_cols,
        num_features,
        BLOCK_SIZE_ROW: tl.constexpr,
        BLOCK_SIZE_FEAT: tl.constexpr,
    ):
        """Triton kernel reading ctl device tensor dynamically in-graph."""
        # 1. Device-side kill switch check
        kill_switch = tl.load(ctl_ptr)
        if kill_switch == 0:
            return

        # 2. Extract actual tokens and slot index
        actual_tokens = tl.load(ctl_ptr + 1)
        slot_idx = tl.load(ctl_ptr + 2)

        if actual_tokens <= 0 or slot_idx < 0 or slot_idx >= slab_slots:
            return

        pid_row = tl.program_id(0)
        pid_feat = tl.program_id(1)

        row_offsets = pid_row * BLOCK_SIZE_ROW + tl.arange(0, BLOCK_SIZE_ROW)
        feat_offsets = pid_feat * BLOCK_SIZE_FEAT + tl.arange(0, BLOCK_SIZE_FEAT)

        valid_rows = actual_tokens
        if valid_rows > act_rows:
            valid_rows = act_rows
        if valid_rows > slab_rows:
            valid_rows = slab_rows

        row_mask = row_offsets < valid_rows
        feat_mask = feat_offsets < num_features

        # Load feature column indices
        feat_ptrs = feature_indices_ptr + feat_offsets * stride_feat
        col_indices = tl.load(feat_ptrs, mask=feat_mask, other=0)

        # Activations pointer calculation: [rows, cols]
        act_ptrs = (
            activations_ptr
            + row_offsets[:, None] * stride_act_row
            + col_indices[None, :] * stride_act_col
        )
        col_valid_mask = col_indices < act_cols
        read_mask = row_mask[:, None] & feat_mask[None, :] & col_valid_mask[None, :]
        vals = tl.load(act_ptrs, mask=read_mask, other=0.0)

        # Staging slab pointer calculation: [slot, row, col_offset + feat]
        out_cols = col_offset + feat_offsets
        slab_col_valid_mask = (out_cols >= 0) & (out_cols < slab_cols)
        slab_ptrs = (
            staging_slab_ptr
            + slot_idx * stride_slab_slot
            + row_offsets[:, None] * stride_slab_row
            + out_cols[None, :] * stride_slab_col
        )
        write_mask = row_mask[:, None] & feat_mask[None, :] & slab_col_valid_mask[None, :]
        tl.store(slab_ptrs, vals, mask=write_mask)


def tap_triton(
    activations: torch.Tensor,
    feature_indices: torch.Tensor,
    ctl: torch.Tensor,
    staging_slab: torch.Tensor,
    col_offset: int,
) -> None:
    """Triton launcher for tap gather operator."""
    if not _TRITON_AVAILABLE or not activations.is_cuda:
        tap_fallback(activations, feature_indices, ctl, staging_slab, col_offset)
        return

    if activations.dim() > 2:
        act_2d = activations.reshape(-1, activations.shape[-1])
    else:
        act_2d = activations

    num_features = feature_indices.numel()
    if num_features == 0:
        return

    slab_cols = staging_slab.shape[2]
    if col_offset < 0 or col_offset >= slab_cols:
        return

    grid = (
        triton.cdiv(int(act_2d.shape[0]), 32),
        triton.cdiv(int(num_features), 32),
    )

    _tap_gather_triton_kernel[grid](
        activations_ptr=act_2d,
        feature_indices_ptr=feature_indices,
        staging_slab_ptr=staging_slab,
        ctl_ptr=ctl,
        col_offset=col_offset,
        stride_act_row=act_2d.stride(0),
        stride_act_col=act_2d.stride(1),
        stride_slab_slot=staging_slab.stride(0),
        stride_slab_row=staging_slab.stride(1),
        stride_slab_col=staging_slab.stride(2),
        stride_feat=feature_indices.stride(0) if feature_indices.dim() > 0 else 1,
        act_rows=act_2d.shape[0],
        act_cols=act_2d.shape[1],
        slab_slots=staging_slab.shape[0],
        slab_rows=staging_slab.shape[1],
        slab_cols=staging_slab.shape[2],
        num_features=num_features,
        BLOCK_SIZE_ROW=32,
        BLOCK_SIZE_FEAT=32,
    )


# -----------------------------------------------------------------------------
# Pure NumPy Fallback
# -----------------------------------------------------------------------------
def tap_numpy(
    activations: np.ndarray,
    feature_indices: np.ndarray | list[int],
    ctl: np.ndarray | list[int],
    staging_slab: np.ndarray,
    col_offset: int,
) -> None:
    """Pure-NumPy implementation of glassbox::tap."""
    ctl_arr = np.asarray(ctl, dtype=np.int64)
    if ctl_arr.size < 3:
        raise ValueError("ctl array must have at least 3 elements [kill, actual_tokens, slot]")

    kill_switch = int(ctl_arr[0])
    if kill_switch == 0:
        return

    actual_num_tokens = int(ctl_arr[1])
    slot_idx = int(ctl_arr[2])

    if actual_num_tokens <= 0:
        return

    feat_indices = np.asarray(feature_indices, dtype=np.int64)
    if feat_indices.size == 0:
        return

    if activations.ndim > 2:
        act_2d = activations.reshape(-1, activations.shape[-1])
    else:
        act_2d = activations

    max_rows = min(actual_num_tokens, act_2d.shape[0], staging_slab.shape[1])
    if max_rows <= 0:
        return

    if slot_idx < 0 or slot_idx >= staging_slab.shape[0]:
        raise IndexError(
            f"slot_idx {slot_idx} out of bounds for staging_slab with {staging_slab.shape[0]} slots"
        )

    slab_cols = staging_slab.shape[2]
    if col_offset < 0 or col_offset >= slab_cols:
        return

    end_col = min(col_offset + feat_indices.size, slab_cols)
    valid_features = feat_indices[: end_col - col_offset]

    gathered = act_2d[:max_rows, valid_features]
    staging_slab[slot_idx, :max_rows, col_offset:end_col] = gathered


# -----------------------------------------------------------------------------
# PyTorch Fallback
# -----------------------------------------------------------------------------
def tap_fallback(
    activations: Any,
    feature_indices: Any,
    ctl: Any,
    staging_slab: Any,
    col_offset: int,
) -> None:
    """Pure-Python / PyTorch / NumPy fallback implementation of glassbox::tap."""
    # NumPy array dispatch
    if isinstance(activations, np.ndarray) or isinstance(staging_slab, np.ndarray):
        tap_numpy(
            activations=np.asarray(activations),
            feature_indices=feature_indices,
            ctl=ctl,
            staging_slab=np.asarray(staging_slab),
            col_offset=col_offset,
        )
        return

    if not isinstance(ctl, torch.Tensor):
        ctl = torch.as_tensor(ctl, dtype=torch.long)
    if not isinstance(feature_indices, torch.Tensor):
        feature_indices = torch.as_tensor(feature_indices, dtype=torch.long)

    # 1. Kill switch check (ctl[0])
    kill_switch = int(ctl[0].item())
    if kill_switch == 0:
        return

    # 2. Extract actual tokens (ctl[1]) and slot index (ctl[2])
    actual_num_tokens = int(ctl[1].item())
    slot_idx = int(ctl[2].item())

    if actual_num_tokens <= 0:
        return

    num_features = feature_indices.numel()
    if num_features == 0:
        return

    # Flatten activations to 2D [tokens, hidden_dim]
    if activations.dim() > 2:
        act_2d = activations.reshape(-1, activations.shape[-1])
    else:
        act_2d = activations

    max_rows = min(actual_num_tokens, act_2d.shape[0], staging_slab.shape[1])
    if max_rows <= 0:
        return

    if slot_idx < 0 or slot_idx >= staging_slab.shape[0]:
        raise IndexError(
            f"slot_idx {slot_idx} out of bounds for staging_slab with {staging_slab.shape[0]} slots"
        )

    slab_cols = staging_slab.shape[2]
    if col_offset < 0 or col_offset >= slab_cols:
        return

    end_col = min(col_offset + num_features, slab_cols)
    valid_features = feature_indices[: end_col - col_offset]

    # Gather rows [0 .. max_rows - 1] and columns valid_features
    # into staging_slab[slot_idx, 0 .. max_rows - 1, col_offset .. end_col]
    gathered = act_2d[:max_rows, valid_features]
    if gathered.dtype != staging_slab.dtype:
        gathered = gathered.to(staging_slab.dtype)

    staging_slab[slot_idx, :max_rows, col_offset:end_col] = gathered


# -----------------------------------------------------------------------------
# Out-of-Graph Asynchronous D2H Transfer Helper
# -----------------------------------------------------------------------------
def glassbox_async_d2h_copy(
    src: torch.Tensor,
    dst: torch.Tensor,
    stream: Any = None,
    event: Any = None,
    slot_idx: int | None = None,
    actual_num_tokens: int | None = None,
) -> Any:
    """Asynchronous D2H copy from device staging slab to host pinned memory.

    Parameters:
        src: Device tensor (staging slab or slot slice).
        dst: Host tensor (pinned host memory ring slot).
        stream: Optional dedicated CUDA copyStream.
        event: Optional CUDA event to record completion.
        slot_idx: Optional slot index to copy from 3D slab.
        actual_num_tokens: Optional number of rows to copy.

    Returns:
        Recorded CUDA event if on CUDA, otherwise None.
    """
    if src.is_cuda and torch.cuda.is_available():
        copy_stream = stream or torch.cuda.Stream()
        with torch.cuda.stream(copy_stream):
            if slot_idx is not None and actual_num_tokens is not None:
                max_dst_rows = dst.shape[1] if dst.dim() > 1 else dst.shape[0]
                rows = min(actual_num_tokens, src.shape[1], max_dst_rows)
                dst[slot_idx, :rows].copy_(src[slot_idx, :rows], non_blocking=True)
            elif slot_idx is not None:
                dst[slot_idx].copy_(src[slot_idx], non_blocking=True)
            elif actual_num_tokens is not None:
                rows = min(actual_num_tokens, src.shape[0], dst.shape[0])
                dst[:rows].copy_(src[:rows], non_blocking=True)
            else:
                dst.copy_(src, non_blocking=True)

            ev = event or torch.cuda.Event()
            ev.record(copy_stream)
            return ev
    else:
        # Host/CPU fallback
        if slot_idx is not None and actual_num_tokens is not None:
            max_dst_rows = dst.shape[1] if dst.dim() > 1 else dst.shape[0]
            rows = min(actual_num_tokens, src.shape[1], max_dst_rows)
            dst[slot_idx, :rows].copy_(src[slot_idx, :rows])
        elif slot_idx is not None:
            dst[slot_idx].copy_(src[slot_idx])
        elif actual_num_tokens is not None:
            rows = min(actual_num_tokens, src.shape[0], dst.shape[0])
            dst[:rows].copy_(src[:rows])
        else:
            dst.copy_(src)
        return None


# -----------------------------------------------------------------------------
# PyTorch Custom Op Registration & Abstract Fake Tensor Registration
# -----------------------------------------------------------------------------
# Check if glassbox::tap is already registered (e.g. from C++ extension)
_TAP_REGISTERED = False
try:
    if hasattr(torch.ops, "glassbox") and hasattr(torch.ops.glassbox, "tap"):
        _TAP_REGISTERED = True
except Exception:
    pass

if not _TAP_REGISTERED:
    # Schema: "glassbox::tap(Tensor activations, Tensor feature_indices,
    #         Tensor ctl, Tensor(a!) staging_slab, int col_offset) -> ()"
    @torch.library.custom_op("glassbox::tap", mutates_args={"staging_slab"})
    def tap(
        activations: torch.Tensor,
        feature_indices: torch.Tensor,
        ctl: torch.Tensor,
        staging_slab: torch.Tensor,
        col_offset: int,
    ) -> None:
        """In-graph wiretap gather operator."""
        tap_fallback(activations, feature_indices, ctl, staging_slab, col_offset)

    @tap.register_fake
    def _tap_fake(
        activations: torch.Tensor,
        feature_indices: torch.Tensor,
        ctl: torch.Tensor,
        staging_slab: torch.Tensor,
        col_offset: int,
    ) -> None:
        """Abstract fake tensor transfer function for PyTorch Dynamo / CUDA Graphs."""
        return
else:
    tap = torch.ops.glassbox.tap

__all__ = [
    "tap",
    "tap_fallback",
    "tap_numpy",
    "tap_triton",
    "glassbox_async_d2h_copy",
]
