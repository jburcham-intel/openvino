// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/common/network_metadata.hpp"

#include <unordered_set>

#include "openvino/core/except.hpp"

namespace intel_npu {

void NetworkMetadata::bindRelatedDescriptors() {
    size_t ioIndex = 0;

    for (IODescriptor& input : inputs) {
        if (input.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (input.isStateInput) {
            const auto relatedDescriptorIterator =
                std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& output) {
                    return output.isStateOutput && (output.nameFromCompiler == input.nameFromCompiler);
                });

            if (relatedDescriptorIterator != outputs.end()) {
                input.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;

                // A state input and its state output are two views of the same variable, so their shape and
                // precision must match. Otherwise the buffer shared between them, which is sized from the input,
                // would be undersized for the output.
                OPENVINO_ASSERT(relatedDescriptorIterator->shapeFromCompiler == input.shapeFromCompiler &&
                                    relatedDescriptorIterator->precision == input.precision,
                                "State input '",
                                input.nameFromCompiler,
                                "' does not match its state output in shape or precision.");
            }
        } else if (input.isShapeTensor) {
            const auto relatedDescriptorIterator =
                std::find_if(inputs.begin(), inputs.end(), [&](const IODescriptor& candidate) {
                    return !candidate.isShapeTensor && (candidate.nameFromCompiler == input.nameFromCompiler);
                });

            if (relatedDescriptorIterator != inputs.end()) {
                input.relatedDescriptorIndex = std::distance(inputs.begin(), relatedDescriptorIterator);
                inputs.at(*input.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        }

        ++ioIndex;
    }

    ioIndex = 0;

    for (IODescriptor& output : outputs) {
        if (output.relatedDescriptorIndex.has_value()) {
            ++ioIndex;
            continue;
        }

        if (output.isShapeTensor) {
            const auto relatedDescriptorIterator =
                std::find_if(outputs.begin(), outputs.end(), [&](const IODescriptor& candidate) {
                    return !candidate.isShapeTensor && (candidate.nameFromCompiler == output.nameFromCompiler);
                });

            if (relatedDescriptorIterator != outputs.end()) {
                output.relatedDescriptorIndex = std::distance(outputs.begin(), relatedDescriptorIterator);
                outputs.at(*output.relatedDescriptorIndex).relatedDescriptorIndex = ioIndex;
            }
        }

        ++ioIndex;
    }
}

}  // namespace intel_npu

void intel_npu::NetworkMetadata::mark_host_only_inputs(const std::shared_ptr<const ov::Model>& model) {
    std::unordered_set<const ov::Node*> reachable;
    std::vector<std::shared_ptr<ov::Node>> pending;
    for (const auto& result : model->get_results()) {
        pending.push_back(result);
    }
    for (const auto& sink : model->get_sinks()) {
        pending.push_back(sink);
    }
    while (!pending.empty()) {
        const auto node = pending.back();
        pending.pop_back();
        if (!reachable.insert(node.get()).second) {
            continue;
        }
        for (const auto& input : node->input_values()) {
            pending.push_back(input.get_node_shared_ptr());
        }
        for (const auto& dependency : node->get_control_dependencies()) {
            pending.push_back(dependency);
        }
    }
    for (auto& desc : inputs) {
        if (desc.isStateInput || desc.isShapeTensor || desc.isMainInputWeights || desc.isInitInputWeights) {
            continue;
        }
        // Compiler-generated inputs remain device-accessed. Match a unique original parameter.
        const ov::Node* matched = nullptr;
        bool ambiguous = false;
        for (const auto& parameter : model->get_parameters()) {
            if (desc.nodeFriendlyName == parameter->get_friendly_name() ||
                desc.nameFromCompiler == parameter->get_friendly_name() ||
                parameter->output(0).get_names().count(desc.nameFromCompiler)) {
                ambiguous = ambiguous || matched != nullptr;
                matched = parameter.get();
            }
        }
        desc.isHostOnly = matched != nullptr && !ambiguous && !reachable.count(matched);
    }
}
