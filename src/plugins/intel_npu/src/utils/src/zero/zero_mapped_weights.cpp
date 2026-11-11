// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_mapped_weights.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

#include "intel_npu/utils/utils.hpp"
#include "intel_npu/utils/zero/zero_mem.hpp"
#include "intel_npu/utils/zero/zero_mem_pool.hpp"
#include "openvino/core/except.hpp"

namespace intel_npu {

namespace {

// Bytes from base that belong to one mapped section. An import may not span two sections, and ov::MappedMemory backs
// the unaligned tail of a file with a separate anonymous section placed right after the file view.
size_t contiguous_mapped_bytes(const void* base, size_t size) {
#ifdef _WIN32
    MEMORY_BASIC_INFORMATION mbi{};
    if (VirtualQuery(base, &mbi, sizeof(mbi)) == 0 || mbi.State != MEM_COMMIT) {
        return size;
    }

    const auto region_end = reinterpret_cast<uintptr_t>(mbi.BaseAddress) + mbi.RegionSize;
    const auto base_addr = reinterpret_cast<uintptr_t>(base);
    return region_end <= base_addr ? size : std::min(size, static_cast<size_t>(region_end - base_addr));
#else
    // not needed on Linux so far; an import that crosses a section boundary fails and the caller falls back
    (void)base;
    return size;
#endif
}

}  // namespace

bool ZeroMappedWeights::is_enabled() {
    // unset or any non-zero value keeps zero-copy weights on
    const char* env = std::getenv("NPUW_WEIGHTS_ZERO_COPY");
    return env == nullptr || std::atoi(env) != 0;
}

ZeroMappedWeights::ZeroMappedWeights(const std::shared_ptr<ZeroInitStructsHolder>& init_structs,
                                     const void* cpu_ptr,
                                     size_t size)
    : _logger("ZeroMappedWeights", Logger::global().level()) {
    OPENVINO_ASSERT(init_structs != nullptr && init_structs->isExternalMemoryStandardAllocationSupported() &&
                        init_structs->isMemUsageHintSupported() && init_structs->isGraphArgumentSizeSupported(),
                    "Zero-copy weights need standard allocation import, the NPU memory usage hint and graph "
                    "argument sizes");
    OPENVINO_ASSERT(cpu_ptr != nullptr && size != 0, "Can not import a null or zero-sized memory range");
    OPENVINO_ASSERT(reinterpret_cast<uintptr_t>(cpu_ptr) % utils::STANDARD_PAGE_SIZE == 0,
                    "Memory range must be aligned to the standard page size");

    // Only whole pages of one mapped section are imported; the unaligned tail of a file is left to the caller.
    _size = (contiguous_mapped_bytes(cpu_ptr, size) / utils::STANDARD_PAGE_SIZE) * utils::STANDARD_PAGE_SIZE;
    OPENVINO_ASSERT(_size != 0, "Memory range is smaller than one page");

    _mem = zero_mem::import_weights_memory(init_structs, cpu_ptr, _size);

    _logger.info("Mapped %zu of %zu bytes", _size, size);
}

bool ZeroMappedWeights::covers(size_t offset, size_t byte_size) const {
    return offset <= _size && byte_size <= _size - offset;
}

// the pool entry's deleter frees the import
ZeroMappedWeights::~ZeroMappedWeights() = default;

}  // namespace intel_npu
