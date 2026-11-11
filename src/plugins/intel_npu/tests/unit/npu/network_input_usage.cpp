// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "intel_npu/common/network_metadata.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"

namespace {
intel_npu::IODescriptor input_descriptor(const std::string& name, uint32_t index) {
    intel_npu::IODescriptor desc{};
    desc.nameFromCompiler = name;
    desc.nodeFriendlyName = name;
    desc.indexUsedByDriver = index;
    return desc;
}
}  // namespace

TEST(NetworkInputUsage, HostGatherSourceWithDeadConsumersHasNoDeviceAccess) {
    auto vocab = std::make_shared<ov::op::v0::Parameter>(ov::element::u4, ov::Shape{65536, 4096});
    vocab->set_friendly_name("vocab");
    auto gathered = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 1, 4096});
    gathered->set_friendly_name("gathered");
    // A rewrite can leave a consumer alive in memory without a path to a result.
    auto dead = std::make_shared<ov::op::v0::Relu>(vocab);
    auto result = std::make_shared<ov::op::v0::Result>(gathered);
    auto head = std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{vocab, gathered});
    ASSERT_FALSE(vocab->output(0).get_target_inputs().empty());
    intel_npu::NetworkMetadata metadata;
    metadata.inputs = {input_descriptor("vocab", 7), input_descriptor("gathered", 2)};
    metadata.mark_host_only_inputs(head);
    EXPECT_TRUE(metadata.inputs[0].isHostOnly);
    EXPECT_FALSE(metadata.inputs[1].isHostOnly);
    // The same source must remain device-accessed in the tail graph that actually reads it.
    auto tail = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(vocab)},
                                            ov::ParameterVector{vocab});
    metadata.mark_host_only_inputs(tail);
    EXPECT_FALSE(metadata.inputs[0].isHostOnly);
}

TEST(NetworkInputUsage, CompilerGeneratedAndShapeInputsRemainDeviceAccessed) {
    auto unused = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    unused->set_friendly_name("unused");
    auto used = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4});
    auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(used)},
                                             ov::ParameterVector{unused, used});
    intel_npu::NetworkMetadata metadata;
    metadata.inputs = {input_descriptor("generated", 1), input_descriptor("unused", 2)};
    metadata.inputs[1].isShapeTensor = true;
    metadata.mark_host_only_inputs(model);
    EXPECT_FALSE(metadata.inputs[0].isHostOnly);
    EXPECT_FALSE(metadata.inputs[1].isHostOnly);
}
