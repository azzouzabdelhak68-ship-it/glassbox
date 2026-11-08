#ifndef GLASSBOX_RING_BUFFER_H
#define GLASSBOX_RING_BUFFER_H

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>
#include <memory>
#include <algorithm>

#if defined(__has_include)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define GLASSBOX_HAS_CUDA 1
#endif
#endif

namespace glassbox {

// Slot status flags
enum SlotStatus : uint32_t {
    SLOT_EMPTY = 0,
    SLOT_COMMITTING = 1,
    SLOT_READY = 2,
    SLOT_DROPPED = 3,
    SLOT_READING = 4,
};

// Drop reason codes
enum DropReason : uint32_t {
    DROP_REASON_UNKNOWN = 0,
    DROP_REASON_RING_FULL = 1,
    DROP_REASON_IO_TIMEOUT = 2,
    DROP_REASON_INVALID_PAYLOAD = 3,
    DROP_REASON_USER_REQUESTED = 4,
};

// Tombstone entry for fail-open dropped steps
struct TombstoneEntry {
    uint64_t step_id{0};
    uint32_t reason_code{DROP_REASON_UNKNOWN};
    uint64_t timestamp_ns{0};
};

// Ring buffer slot with cache-line alignment
struct alignas(64) RingSlot {
    std::atomic<uint32_t> status{SLOT_EMPTY};
    uint64_t step_id{0};
    uint32_t actual_num_tokens{0};
    uint32_t payload_bytes{0};
    void* data_ptr{nullptr};
    char manifest[8192]{};
    void* ready_event{nullptr};
    uint64_t timestamp_ns{0};

    // Internal buffer management
    void* owned_data_buf{nullptr};
    size_t owned_data_size{0};
    bool is_cuda_pinned{false};

    ~RingSlot() {
        if (owned_data_buf) {
#if defined(GLASSBOX_HAS_CUDA)
            if (is_cuda_pinned) {
                cudaFreeHost(owned_data_buf);
                owned_data_buf = nullptr;
                return;
            }
#endif
            std::free(owned_data_buf);
            owned_data_buf = nullptr;
        }
    }
};

inline uint64_t current_time_ns() {
    auto now = std::chrono::steady_clock::now();
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count()
    );
}

// Single-Producer Single-Consumer (SPSC) Host Ring Buffer with DROP_NEWEST fail-open policy
class HostRingBuffer {
public:
    static constexpr uint32_t DEFAULT_CAPACITY = 16;
    static constexpr uint32_t TOMBSTONE_CAPACITY = 256;

    explicit HostRingBuffer(size_t slot_payload_bytes = 0, uint32_t capacity = DEFAULT_CAPACITY)
        : capacity_(capacity > 0 ? capacity : DEFAULT_CAPACITY),
          slot_payload_bytes_(slot_payload_bytes),
          slots_(capacity_) {
        // Allocate pinned / page-aligned memory for each slot if payload bytes requested
        for (uint32_t i = 0; i < capacity_; ++i) {
            slots_[i].status.store(SLOT_EMPTY, std::memory_order_relaxed);
            slots_[i].payload_bytes = static_cast<uint32_t>(slot_payload_bytes);
            if (slot_payload_bytes > 0) {
                void* ptr = nullptr;
                bool is_cuda = false;
#if defined(GLASSBOX_HAS_CUDA)
                if (cudaHostAlloc(&ptr, slot_payload_bytes, cudaHostAllocDefault) == cudaSuccess) {
                    is_cuda = true;
                } else {
                    ptr = nullptr;
                }
#endif
                if (!ptr) {
                    if (posix_memalign(&ptr, 4096, slot_payload_bytes) != 0) {
                        ptr = std::malloc(slot_payload_bytes);
                    }
                }
                slots_[i].owned_data_buf = ptr;
                slots_[i].owned_data_size = slot_payload_bytes;
                slots_[i].is_cuda_pinned = is_cuda;
                slots_[i].data_ptr = ptr;
            }
        }

        // Initialize tombstone ring
        tombstones_.resize(TOMBSTONE_CAPACITY);
    }

    ~HostRingBuffer() = default;

    // Disallow copy/move to guarantee pointer stability and thread safety
    HostRingBuffer(const HostRingBuffer&) = delete;
    HostRingBuffer& operator=(const HostRingBuffer&) = delete;
    HostRingBuffer(HostRingBuffer&&) = delete;
    HostRingBuffer& operator=(HostRingBuffer&&) = delete;

    // Producer API: Non-blocking acquire with fail-open DROP_NEWEST policy
    RingSlot* try_acquire(uint64_t step_id, uint32_t actual_tokens) noexcept {
        const uint64_t head = head_seq_.load(std::memory_order_relaxed);

        // Fast-path capacity check using cached tail
        if (head - cached_tail_ >= capacity_) {
            // Refresh cached tail with acquire semantics
            cached_tail_ = tail_seq_.load(std::memory_order_acquire);
            if (head - cached_tail_ >= capacity_) {
                // Buffer is FULL: Fail-open DROP_NEWEST!
                // Producer thread NEVER blocks; records tombstone and returns nullptr.
                record_tombstone(step_id, DROP_REASON_RING_FULL);
                dropped_steps_count_.fetch_add(1, std::memory_order_relaxed);
                return nullptr;
            }
        }

        const uint32_t idx = static_cast<uint32_t>(head % capacity_);
        RingSlot* slot = &slots_[idx];
        slot->status.store(SLOT_COMMITTING, std::memory_order_relaxed);
        slot->step_id = step_id;
        slot->actual_num_tokens = actual_tokens;
        slot->timestamp_ns = current_time_ns();
        slot->manifest[0] = '\0';
        return slot;
    }

    // Producer API: Commit slot making it visible to consumer
    void commit(RingSlot* slot) noexcept {
        if (!slot) return;
        slot->status.store(SLOT_READY, std::memory_order_release);
        head_seq_.fetch_add(1, std::memory_order_release);
    }

    // Producer API: Push copy convenience
    bool push(uint64_t step_id, uint32_t actual_tokens, const void* data, size_t bytes, const char* manifest) noexcept {
        RingSlot* slot = try_acquire(step_id, actual_tokens);
        if (!slot) {
            return false;
        }
        if (data && bytes > 0 && slot->data_ptr) {
            size_t copy_size = std::min(bytes, static_cast<size_t>(slot->payload_bytes));
            std::memcpy(slot->data_ptr, data, copy_size);
        }
        if (manifest) {
            std::strncpy(slot->manifest, manifest, sizeof(slot->manifest) - 1);
            slot->manifest[sizeof(slot->manifest) - 1] = '\0';
        } else {
            slot->manifest[0] = '\0';
        }
        commit(slot);
        return true;
    }

    // Producer API: Explicit step drop (e.g., validation failure or DMA error)
    void drop_step(uint64_t step_id, uint32_t reason_code = DROP_REASON_UNKNOWN) noexcept {
        record_tombstone(step_id, reason_code);
        dropped_steps_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Consumer API: Peek at next available slot without advancing tail
    const RingSlot* peek() noexcept {
        const uint64_t tail = tail_seq_.load(std::memory_order_relaxed);

        // Fast-path check using cached head
        if (cached_head_ <= tail) {
            cached_head_ = head_seq_.load(std::memory_order_acquire);
            if (cached_head_ <= tail) {
                return nullptr; // Empty
            }
        }

        const uint32_t idx = static_cast<uint32_t>(tail % capacity_);
        RingSlot* slot = &slots_[idx];
        if (slot->status.load(std::memory_order_acquire) != SLOT_READY) {
            return nullptr;
        }
        slot->status.store(SLOT_READING, std::memory_order_relaxed);
        return slot;
    }

    // Consumer API: Release slot after reading, advancing tail
    void release(const RingSlot* slot) noexcept {
        if (!slot) return;
        const_cast<RingSlot*>(slot)->status.store(SLOT_EMPTY, std::memory_order_release);
        tail_seq_.fetch_add(1, std::memory_order_release);
    }

    // Consumer API: Drain tombstone entries
    uint32_t drain_tombstones(TombstoneEntry* out_entries, uint32_t max_entries) noexcept {
        if (!out_entries || max_entries == 0) return 0;

        uint32_t drained = 0;
        uint64_t t_tail = tombstone_tail_.load(std::memory_order_relaxed);
        const uint64_t t_head = tombstone_head_.load(std::memory_order_acquire);

        while (t_tail < t_head && drained < max_entries) {
            uint32_t idx = static_cast<uint32_t>(t_tail % TOMBSTONE_CAPACITY);
            out_entries[drained++] = tombstones_[idx];
            t_tail++;
        }
        tombstone_tail_.store(t_tail, std::memory_order_release);
        return drained;
    }

    // Statistics
    uint64_t get_dropped_count() const noexcept {
        return dropped_steps_count_.load(std::memory_order_relaxed);
    }

    uint32_t occupancy() const noexcept {
        uint64_t head = head_seq_.load(std::memory_order_relaxed);
        uint64_t tail = tail_seq_.load(std::memory_order_relaxed);
        return (head >= tail) ? static_cast<uint32_t>(head - tail) : 0;
    }

    uint32_t capacity() const noexcept {
        return capacity_;
    }

    size_t slot_payload_bytes() const noexcept {
        return slot_payload_bytes_;
    }

private:
    void record_tombstone(uint64_t step_id, uint32_t reason_code) noexcept {
        uint64_t t_head = tombstone_head_.load(std::memory_order_relaxed);
        uint64_t t_tail = tombstone_tail_.load(std::memory_order_acquire);

        if (t_head - t_tail < TOMBSTONE_CAPACITY) {
            uint32_t idx = static_cast<uint32_t>(t_head % TOMBSTONE_CAPACITY);
            tombstones_[idx].step_id = step_id;
            tombstones_[idx].reason_code = reason_code;
            tombstones_[idx].timestamp_ns = current_time_ns();
            tombstone_head_.store(t_head + 1, std::memory_order_release);
        }
    }

    const uint32_t capacity_;
    const size_t slot_payload_bytes_;
    std::vector<RingSlot> slots_;

    // Producer cache line
    alignas(64) std::atomic<uint64_t> head_seq_{0};
    alignas(64) uint64_t cached_tail_{0};

    // Consumer cache line
    alignas(64) std::atomic<uint64_t> tail_seq_{0};
    alignas(64) uint64_t cached_head_{0};

    // Drops counter cache line
    alignas(64) std::atomic<uint64_t> dropped_steps_count_{0};

    // Auxiliary tombstone ring
    alignas(64) std::vector<TombstoneEntry> tombstones_;
    alignas(64) std::atomic<uint64_t> tombstone_head_{0};
    alignas(64) std::atomic<uint64_t> tombstone_tail_{0};
};

} // namespace glassbox

// C-ABI exports
extern "C" {

typedef void* HostRingBufferHandle;

inline HostRingBufferHandle glassbox_ring_create(size_t slot_payload_bytes, uint32_t capacity) {
    try {
        return new glassbox::HostRingBuffer(slot_payload_bytes, capacity);
    } catch (...) {
        return nullptr;
    }
}

inline void glassbox_ring_destroy(HostRingBufferHandle handle) {
    if (handle) {
        delete static_cast<glassbox::HostRingBuffer*>(handle);
    }
}

inline glassbox::RingSlot* glassbox_ring_try_acquire(HostRingBufferHandle handle, uint64_t step_id, uint32_t actual_tokens) {
    if (!handle) return nullptr;
    return static_cast<glassbox::HostRingBuffer*>(handle)->try_acquire(step_id, actual_tokens);
}

inline void glassbox_ring_commit(HostRingBufferHandle handle, glassbox::RingSlot* slot) {
    if (handle && slot) {
        static_cast<glassbox::HostRingBuffer*>(handle)->commit(slot);
    }
}

inline bool glassbox_ring_push(
    HostRingBufferHandle handle,
    uint64_t step_id,
    uint32_t actual_tokens,
    const void* data,
    size_t bytes,
    const char* manifest
) {
    if (!handle) return false;
    return static_cast<glassbox::HostRingBuffer*>(handle)->push(step_id, actual_tokens, data, bytes, manifest);
}

inline void glassbox_ring_drop_step(HostRingBufferHandle handle, uint64_t step_id, uint32_t reason_code) {
    if (handle) {
        static_cast<glassbox::HostRingBuffer*>(handle)->drop_step(step_id, reason_code);
    }
}

inline const glassbox::RingSlot* glassbox_ring_peek(HostRingBufferHandle handle) {
    if (!handle) return nullptr;
    return static_cast<glassbox::HostRingBuffer*>(handle)->peek();
}

inline void glassbox_ring_release(HostRingBufferHandle handle, const glassbox::RingSlot* slot) {
    if (handle && slot) {
        static_cast<glassbox::HostRingBuffer*>(handle)->release(slot);
    }
}

inline uint32_t glassbox_ring_drain_tombstones(
    HostRingBufferHandle handle,
    glassbox::TombstoneEntry* out_entries,
    uint32_t max_entries
) {
    if (!handle) return 0;
    return static_cast<glassbox::HostRingBuffer*>(handle)->drain_tombstones(out_entries, max_entries);
}

inline uint64_t glassbox_ring_get_dropped_count(HostRingBufferHandle handle) {
    if (!handle) return 0;
    return static_cast<glassbox::HostRingBuffer*>(handle)->get_dropped_count();
}

inline uint32_t glassbox_ring_occupancy(HostRingBufferHandle handle) {
    if (!handle) return 0;
    return static_cast<glassbox::HostRingBuffer*>(handle)->occupancy();
}

} // extern "C"

#endif // GLASSBOX_RING_BUFFER_H
