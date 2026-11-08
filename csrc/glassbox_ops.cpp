/**
 * glassbox_ops.cpp - PyTorch Custom C++ Operator Registration & CPU Dispatch (spec §GRAPH-WIRE-M3)
 *
 * Implements:
 * 1. PyTorch custom operator schema:
 *    "glassbox::tap(Tensor activations, Tensor feature_indices, Tensor ctl, Tensor(a!) staging_slab, int col_offset) -> ()"
 * 2. CPU gather implementation obeying device control tensor `ctl`:
 *    - ctl[0]: kill switch (0 = disabled / no-op, 1 = active)
 *    - ctl[1]: actual_num_tokens (rows copied: 0 .. actual_num_tokens - 1; padding tokens ignored)
 *    - ctl[2]: active staging slot index (0 .. K-1)
 * 3. CUDA operator forward declaration and dispatch registration.
 * 4. Out-of-graph copy stream helper `glassbox_async_d2h_copy` with dedicated copy stream and cudaEvent sync.
 */

#include <torch/extension.h>
#include <torch/library.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <vector>

#if defined(USE_CUDA) || defined(GLASSBOX_HAS_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAFunctions.h>
#define GLASSBOX_WITH_CUDA 1
#endif

namespace glassbox {

// ----------------------------------------------------------------------------
// Forward Declaration for CUDA implementation (defined in ring_gather_cuda.cu)
// ----------------------------------------------------------------------------
#if defined(GLASSBOX_WITH_CUDA)
void glassbox_tap_cuda(
    const at::Tensor& activations,
    const at::Tensor& feature_indices,
    const at::Tensor& ctl,
    at::Tensor staging_slab,
    int64_t col_offset
);
#endif

// ----------------------------------------------------------------------------
// CPU Gather Implementation
// ----------------------------------------------------------------------------
void glassbox_tap_cpu(
    const at::Tensor& activations,
    const at::Tensor& feature_indices,
    const at::Tensor& ctl,
    at::Tensor staging_slab,
    int64_t col_offset
) {
    TORCH_CHECK(ctl.defined() && ctl.numel() >= 3, "ctl tensor must contain at least 3 elements [kill, actual_tokens, slot]");
    at::Tensor ctl_contig = ctl.to(at::kCPU).contiguous();

    int64_t kill_switch = 0;
    int64_t actual_tokens = 0;
    int64_t slot_idx = 0;

    if (ctl_contig.scalar_type() == at::kLong) {
        const int64_t* p = ctl_contig.data_ptr<int64_t>();
        kill_switch = p[0];
        actual_tokens = p[1];
        slot_idx = p[2];
    } else if (ctl_contig.scalar_type() == at::kInt) {
        const int32_t* p = ctl_contig.data_ptr<int32_t>();
        kill_switch = p[0];
        actual_tokens = p[1];
        slot_idx = p[2];
    } else {
        TORCH_CHECK(false, "ctl tensor must be int32 or int64");
    }

    // 1. Kill switch: if 0, immediate no-op
    if (kill_switch == 0) {
        return;
    }

    // 2. Validate token count and slot index
    if (actual_tokens <= 0) {
        return;
    }

    int64_t num_features = feature_indices.numel();
    if (num_features <= 0) {
        return;
    }

    TORCH_CHECK(staging_slab.defined() && staging_slab.dim() == 3, "staging_slab must be 3D [slots, max_tokens, features]");
    TORCH_CHECK(slot_idx >= 0 && slot_idx < staging_slab.size(0), "slot index out of range for staging_slab");

    // Flatten activations if 3D [batch, seq, hidden] -> 2D [tokens, hidden]
    at::Tensor act_2d = (activations.dim() > 2) ? activations.reshape({-1, activations.size(-1)}) : activations;
    TORCH_CHECK(act_2d.dim() == 2, "activations must be 2D or reducible to 2D");

    int64_t max_rows = std::min({actual_tokens, act_2d.size(0), staging_slab.size(1)});
    if (max_rows <= 0) {
        return;
    }

    at::Tensor indices_contig = feature_indices.to(at::kCPU).contiguous();
    at::Tensor slab_contig = staging_slab; // mutable in-place

    int64_t act_stride_row = act_2d.stride(0);
    int64_t act_stride_col = act_2d.stride(1);
    int64_t slab_stride_slot = slab_contig.stride(0);
    int64_t slab_stride_row = slab_contig.stride(1);
    int64_t slab_stride_col = slab_contig.stride(2);
    int64_t slab_cols = slab_contig.size(2);
    int64_t act_cols = act_2d.size(1);

    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, activations.scalar_type(), "glassbox_tap_cpu", [&] {
        const scalar_t* act_ptr = act_2d.data_ptr<scalar_t>();
        scalar_t* slab_ptr = slab_contig.data_ptr<scalar_t>();

        if (indices_contig.scalar_type() == at::kLong) {
            const int64_t* idx_ptr = indices_contig.data_ptr<int64_t>();
            for (int64_t r = 0; r < max_rows; ++r) {
                const scalar_t* act_row = act_ptr + r * act_stride_row;
                scalar_t* slab_row = slab_ptr + slot_idx * slab_stride_slot + r * slab_stride_row;
                for (int64_t f = 0; f < num_features; ++f) {
                    int64_t out_c = col_offset + f;
                    if (out_c >= 0 && out_c < slab_cols) {
                        int64_t in_c = idx_ptr[f];
                        if (in_c >= 0 && in_c < act_cols) {
                            slab_row[out_c * slab_stride_col] = act_row[in_c * act_stride_col];
                        }
                    }
                }
            }
        } else {
            const int32_t* idx_ptr = indices_contig.data_ptr<int32_t>();
            for (int64_t r = 0; r < max_rows; ++r) {
                const scalar_t* act_row = act_ptr + r * act_stride_row;
                scalar_t* slab_row = slab_ptr + slot_idx * slab_stride_slot + r * slab_stride_row;
                for (int64_t f = 0; f < num_features; ++f) {
                    int64_t out_c = col_offset + f;
                    if (out_c >= 0 && out_c < slab_cols) {
                        int64_t in_c = idx_ptr[f];
                        if (in_c >= 0 && in_c < act_cols) {
                            slab_row[out_c * slab_stride_col] = act_row[in_c * act_stride_col];
                        }
                    }
                }
            }
        }
    });
}

// ----------------------------------------------------------------------------
// Out-of-Graph Asynchronous D2H Transfer Helper
// ----------------------------------------------------------------------------
#if defined(GLASSBOX_WITH_CUDA)
static cudaStream_t g_copy_stream = nullptr;
static cudaEvent_t g_copy_event = nullptr;
static std::once_flag g_stream_init_flag;

static void init_copy_stream() {
    std::call_once(g_stream_init_flag, []() {
        cudaStreamCreateWithFlags(&g_copy_stream, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&g_copy_event, cudaEventDisableTiming);
    });
}

void glassbox_async_d2h_copy_cuda(
    const at::Tensor& src,
    at::Tensor dst,
    int64_t slot_idx,
    int64_t num_tokens
) {
    init_copy_stream();
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(!dst.is_cuda(), "dst must be a host/CPU tensor");

    // Synchronize copy stream with current compute stream so data is ready
    cudaStream_t compute_stream = c10::cuda::getCurrentCUDAStream();
    cudaEvent_t compute_event;
    cudaEventCreateWithFlags(&compute_event, cudaEventDisableTiming);
    cudaEventRecord(compute_event, compute_stream);
    cudaStreamWaitEvent(g_copy_stream, compute_event, 0);
    cudaEventDestroy(compute_event);

    at::Tensor src_slice = src;
    at::Tensor dst_slice = dst;

    if (slot_idx >= 0) {
        TORCH_CHECK(slot_idx < src.size(0) && slot_idx < dst.size(0), "slot index out of bounds");
        src_slice = src[slot_idx];
        dst_slice = dst[slot_idx];
    }
    if (num_tokens >= 0) {
        int64_t rows = std::min({num_tokens, src_slice.size(0), dst_slice.size(0)});
        src_slice = src_slice.slice(0, 0, rows);
        dst_slice = dst_slice.slice(0, 0, rows);
    }

    src_slice = src_slice.contiguous();
    dst_slice = dst_slice.contiguous();

    size_t copy_bytes = src_slice.numel() * src_slice.element_size();
    if (copy_bytes > 0) {
        cudaMemcpyAsync(
            dst_slice.data_ptr(),
            src_slice.data_ptr(),
            copy_bytes,
            cudaMemcpyDeviceToHost,
            g_copy_stream
        );
        cudaEventRecord(g_copy_event, g_copy_stream);
    }
}

void glassbox_sync_copy_event() {
    if (g_copy_event) {
        cudaEventSynchronize(g_copy_event);
    }
}

bool glassbox_query_copy_event() {
    if (!g_copy_event) return true;
    cudaError_t err = cudaEventQuery(g_copy_event);
    return err == cudaSuccess;
}
#endif

void glassbox_async_d2h_copy_cpu(
    const at::Tensor& src,
    at::Tensor dst,
    int64_t slot_idx,
    int64_t num_tokens
) {
    at::Tensor src_slice = src;
    at::Tensor dst_slice = dst;

    if (slot_idx >= 0) {
        TORCH_CHECK(slot_idx < src.size(0) && slot_idx < dst.size(0), "slot index out of bounds");
        src_slice = src[slot_idx];
        dst_slice = dst[slot_idx];
    }
    if (num_tokens >= 0) {
        int64_t rows = std::min({num_tokens, src_slice.size(0), dst_slice.size(0)});
        src_slice = src_slice.slice(0, 0, rows);
        dst_slice = dst_slice.slice(0, 0, rows);
    }
    dst_slice.copy_(src_slice, /*non_blocking=*/false);
}

// Global dispatcher function matching schema
void glassbox_tap(
    const at::Tensor& activations,
    const at::Tensor& feature_indices,
    const at::Tensor& ctl,
    at::Tensor staging_slab,
    int64_t col_offset
) {
    if (activations.is_cuda()) {
#if defined(GLASSBOX_WITH_CUDA)
        glassbox_tap_cuda(activations, feature_indices, ctl, staging_slab, col_offset);
#else
        TORCH_CHECK(false, "glassbox::tap called with CUDA tensor but compiled without CUDA support");
#endif
    } else {
        glassbox_tap_cpu(activations, feature_indices, ctl, staging_slab, col_offset);
    }
}

void glassbox_async_d2h_copy(
    const at::Tensor& src,
    at::Tensor dst,
    int64_t slot_idx,
    int64_t num_tokens
) {
    if (src.is_cuda()) {
#if defined(GLASSBOX_WITH_CUDA)
        glassbox_async_d2h_copy_cuda(src, dst, slot_idx, num_tokens);
#else
        TORCH_CHECK(false, "glassbox::async_d2h_copy called with CUDA tensor but compiled without CUDA support");
#endif
    } else {
        glassbox_async_d2h_copy_cpu(src, dst, slot_idx, num_tokens);
    }
}

} // namespace glassbox

// ----------------------------------------------------------------------------
// PyTorch Library Registrations
// ----------------------------------------------------------------------------
TORCH_LIBRARY(glassbox, m) {
    m.def("tap(Tensor activations, Tensor feature_indices, Tensor ctl, Tensor(a!) staging_slab, int col_offset) -> ()");
    m.def("async_d2h_copy(Tensor src, Tensor(a!) dst, int slot_idx, int num_tokens) -> ()");
}

TORCH_LIBRARY_IMPL(glassbox, CPU, m) {
    m.impl("tap", &glassbox::glassbox_tap_cpu);
    m.impl("async_d2h_copy", &glassbox::glassbox_async_d2h_copy_cpu);
}

#if defined(GLASSBOX_WITH_CUDA)
TORCH_LIBRARY_IMPL(glassbox, CUDA, m) {
    m.impl("tap", &glassbox::glassbox_tap_cuda);
    m.impl("async_d2h_copy", &glassbox::glassbox_async_d2h_copy_cuda);
}
#endif

// ----------------------------------------------------------------------------
// Python Extension Bindings
// ----------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "GlassBox custom C++/CUDA operators for in-graph wiretap and D2H async copy";
    m.def("tap", &glassbox::glassbox_tap, "In-graph wiretap gather operator (CPU & CUDA)");
    m.def("tap_cpu", &glassbox::glassbox_tap_cpu, "In-graph wiretap CPU gather operator");
    m.def("async_d2h_copy", &glassbox::glassbox_async_d2h_copy, "Out-of-graph D2H copy helper");
    m.def("async_d2h_copy_cpu", &glassbox::glassbox_async_d2h_copy_cpu, "Out-of-graph D2H copy CPU helper");
#if defined(GLASSBOX_WITH_CUDA)
    m.def("tap_cuda", &glassbox::glassbox_tap_cuda, "In-graph wiretap CUDA gather operator");
    m.def("async_d2h_copy_cuda", &glassbox::glassbox_async_d2h_copy_cuda, "Out-of-graph D2H copy CUDA helper");
    m.def("sync_copy_event", &glassbox::glassbox_sync_copy_event, "Synchronize copy event");
    m.def("query_copy_event", &glassbox::glassbox_query_copy_event, "Query copy event status");
#endif
}
