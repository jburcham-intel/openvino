// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_npu/utils/logger/logger.hpp"
#include "intel_npu/utils/zero/zero_init.hpp"

namespace intel_npu {

class ZeroMem;

/**
 * @brief Maps an existing CPU memory range (e.g. a memory mapped weights file) for the device without copying it.
 * @details The range is imported as one host allocation tagged as graph weights (ZE_extension_mem_npu). The driver
 * splits a large weights import into slices it makes resident or evicts independently, so only the weights a submission
 * uses need to be resident. The import is registered with the level zero memory pool, so a tensor pointing anywhere
 * into it is bound in place. Only the page-aligned prefix of the range is mapped; check covers() before using a
 * sub-range. Throws if the mapping can not be created.
 */
class ZeroMappedWeights final {
public:
    ZeroMappedWeights(const std::shared_ptr<ZeroInitStructsHolder>& init_structs, const void* cpu_ptr, size_t size);
    ZeroMappedWeights(const ZeroMappedWeights&) = delete;
    ZeroMappedWeights& operator=(const ZeroMappedWeights&) = delete;
    ~ZeroMappedWeights();

    /**
     * @brief Whether zero-copy weights are enabled; NPUW_WEIGHTS_ZERO_COPY=0 turns them off.
     */
    static bool is_enabled();

    /**
     * @brief Whether [offset, offset + byte_size) of the CPU range is mapped.
     */
    bool covers(size_t offset, size_t byte_size) const;

private:
    Logger _logger;
    size_t _size = 0;
    std::shared_ptr<ZeroMem> _mem;
};

}  // namespace intel_npu
