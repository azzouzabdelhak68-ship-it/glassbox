/**
 * ring_gather_cuda.cu - Optimized CUDA Kernel for In-Graph Wiretap Gather (spec §GRAPH-WIRE-M3)
 *
 * Implements:
 * 1. 2D grid-stride loop gathering column features into staging slab.
 * 2. Device control tensor `ctl` loaded into shared memory / registers:
 *    - ctl[0]: kill switch (0 = disabled / immediate return, 1 = active)
 *    - ctl[1]: actual_num_tokens (rows gathered: 0 .. actual_num_tokens - 1; padding ignored)
 *    - ctl[2]: active staging slot index (0 .. K-1)
 * 3. Support for float32, float16 (Half), bfloat16 (BFloat16) and int32/int64 indices.
 * 4. Safe CUDA Graph capture and replay semantics without host synchronization.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>

namespace glassbox {

template <typename scalar_t, typename index_t, typename ctl_t>
__global__ void ring_gather_cuda_kernel(
    const scalar_t* __restrict__ activations,
    const index_t* __restrict__ feature_indices,
    const ctl_t* __restrict__ ctl,
    scalar_t* __restrict__ staging_slab,
    int64_t col_offset,
    int64_t act_stride_row,
    int64_t act_stride_col,
    int64_t slab_stride_slot,
    int64_t slab_stride_row,
    int64_t slab_stride_col,
    int64_t act_num_rows,
    int64_t act_num_cols,
    int64_t slab_max_slots,
    int64_t slab_max_rows,
    int64_t slab_max_cols,
    int64_t num_features
) {
    // 1. Cooperative load of ctl into shared memory
    __shared__ ctl_t s_ctl[3];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_ctl[0] = ctl[0]; // kill switch
        s_ctl[1] = ctl[1]; // actual_num_tokens
        s_ctl[2] = ctl[2]; // active staging slot index
    }
    __syncthreads();

    // 2. Kill switch check: if 0, immediate return
    ctl_t kill_switch = s_ctl[0];
    if (kill_switch == 0) {
        return;
    }

    ctl_t actual_tokens = s_ctl[1];
    ctl_t slot_idx = s_ctl[2];

    if (actual_tokens <= 0 || slot_idx < 0 || slot_idx >= slab_max_slots) {
        return;
    }

    int64_t rows_to_gather = (actual_tokens < act_num_rows) ? actual_tokens : act_num_rows;
    if (rows_to_gather > slab_max_rows) {
        rows_to_gather = slab_max_rows;
    }

    // 3. 2D grid-stride loop over (row, feature)
    for (int64_t row = blockIdx.y * blockDim.y + threadIdx.y;
         row < rows_to_gather;
         row += blockDim.y * gridDim.y) {

        const scalar_t* act_row = activations + row * act_stride_row;
        scalar_t* slab_row = staging_slab + slot_idx * slab_stride_slot + row * slab_stride_row;

        for (int64_t feat = blockIdx.x * blockDim.x + threadIdx.x;
             feat < num_features;
             feat += blockDim.x * gridDim.x) {

            int64_t out_col = col_offset + feat;
            if (out_col >= 0 && out_col < slab_max_cols) {
                index_t in_col = feature_indices[feat];
                if (in_col >= 0 && in_col < act_num_cols) {
                    slab_row[out_col * slab_stride_col] = act_row[in_col * act_stride_col];
                }
            }
        }
    }
}

void glassbox_tap_cuda(
    const at::Tensor& activations,
    const at::Tensor& feature_indices,
    const at::Tensor& ctl,
    at::Tensor staging_slab,
    int64_t col_offset
) {
    TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");
    TORCH_CHECK(staging_slab.is_cuda(), "staging_slab must be a CUDA tensor");
    TORCH_CHECK(ctl.is_cuda(), "ctl must be a CUDA tensor");
    TORCH_CHECK(feature_indices.is_cuda(), "feature_indices must be a CUDA tensor");

    TORCH_CHECK(ctl.numel() >= 3, "ctl tensor must contain at least 3 elements [kill, actual_tokens, slot]");
    TORCH_CHECK(staging_slab.dim() == 3, "staging_slab must be 3D [slots, max_tokens, features]");

    int64_t num_features = feature_indices.numel();
    if (num_features <= 0) {
        return;
    }

    at::Tensor act_2d = (activations.dim() > 2) ? activations.reshape({-1, activations.size(-1)}) : activations;
    TORCH_CHECK(act_2d.dim() == 2, "activations must be 2D or reducible to 2D");

    int64_t act_num_rows = act_2d.size(0);
    int64_t act_num_cols = act_2d.size(1);
    int64_t slab_max_slots = staging_slab.size(0);
    int64_t slab_max_rows = staging_slab.size(1);
    int64_t slab_max_cols = staging_slab.size(2);

    dim3 block(16, 16);
    dim3 grid(
        static_cast<uint32_t>(std::min<int64_t>((num_features + block.x - 1) / block.x, 64)),
        static_cast<uint32_t>(std::min<int64_t>((act_num_rows + block.y - 1) / block.y, 64))
    );
    if (grid.x == 0) grid.x = 1;
    if (grid.y == 0) grid.y = 1;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Dispatch floating scalar types: float32, Half, BFloat16
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, activations.scalar_type(), "ring_gather_cuda_kernel", [&] {
        using scalar_type = scalar_t;
        if (feature_indices.scalar_type() == at::kLong) {
            using index_type = int64_t;
            if (ctl.scalar_type() == at::kLong) {
                using ctl_type = int64_t;
                ring_gather_cuda_kernel<scalar_type, index_type, ctl_type><<<grid, block, 0, stream>>>(
                    act_2d.data_ptr<scalar_type>(),
                    feature_indices.data_ptr<index_type>(),
                    ctl.data_ptr<ctl_type>(),
                    staging_slab.data_ptr<scalar_type>(),
                    col_offset,
                    act_2d.stride(0),
                    act_2d.stride(1),
                    staging_slab.stride(0),
                    staging_slab.stride(1),
                    staging_slab.stride(2),
                    act_num_rows,
                    act_num_cols,
                    slab_max_slots,
                    slab_max_rows,
                    slab_max_cols,
                    num_features
                );
            } else {
                using ctl_type = int32_t;
                ring_gather_cuda_kernel<scalar_type, index_type, ctl_type><<<grid, block, 0, stream>>>(
                    act_2d.data_ptr<scalar_type>(),
                    feature_indices.data_ptr<index_type>(),
                    ctl.data_ptr<ctl_type>(),
                    staging_slab.data_ptr<scalar_type>(),
                    col_offset,
                    act_2d.stride(0),
                    act_2d.stride(1),
                    staging_slab.stride(0),
                    staging_slab.stride(1),
                    staging_slab.stride(2),
                    act_num_rows,
                    act_num_cols,
                    slab_max_slots,
                    slab_max_rows,
                    slab_max_cols,
                    num_features
                );
            }
        } else {
            using index_type = int32_t;
            if (ctl.scalar_type() == at::kLong) {
                using ctl_type = int64_t;
                ring_gather_cuda_kernel<scalar_type, index_type, ctl_type><<<grid, block, 0, stream>>>(
                    act_2d.data_ptr<scalar_type>(),
                    feature_indices.data_ptr<index_type>(),
                    ctl.data_ptr<ctl_type>(),
                    staging_slab.data_ptr<scalar_type>(),
                    col_offset,
                    act_2d.stride(0),
                    act_2d.stride(1),
                    staging_slab.stride(0),
                    staging_slab.stride(1),
                    staging_slab.stride(2),
                    act_num_rows,
                    act_num_cols,
                    slab_max_slots,
                    slab_max_rows,
                    slab_max_cols,
                    num_features
                );
            } else {
                using ctl_type = int32_t;
                ring_gather_cuda_kernel<scalar_type, index_type, ctl_type><<<grid, block, 0, stream>>>(
                    act_2d.data_ptr<scalar_type>(),
                    feature_indices.data_ptr<index_type>(),
                    ctl.data_ptr<ctl_type>(),
                    staging_slab.data_ptr<scalar_type>(),
                    col_offset,
                    act_2d.stride(0),
                    act_2d.stride(1),
                    staging_slab.stride(0),
                    staging_slab.stride(1),
                    staging_slab.stride(2),
                    act_num_rows,
                    act_num_cols,
                    slab_max_slots,
                    slab_max_rows,
                    slab_max_cols,
                    num_features
                );
            }
        }
    });
}

} // namespace glassbox
