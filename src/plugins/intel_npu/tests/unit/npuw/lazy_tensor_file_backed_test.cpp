// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lazy_tensor.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "common_test_utils/file_utils.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/mmap_object.hpp"
#include "serialization.hpp"

using ov::npuw::s11n::Weights;
using ov::npuw::s11n::WeightsContext;
using ov::npuw::weights::LazyTensor;

namespace {

std::shared_ptr<ov::op::v0::Constant> make_weightless_constant(const ov::element::Type& type,
                                                                const ov::Shape& shape,
                                                                const std::vector<float>& data,
                                                                std::size_t offset) {
    auto constant = std::make_shared<ov::op::v0::Constant>(type, shape, data);
    constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(constant->get_byte_size(), offset, type);
    return constant;
}

// Round-trips a LazyTensor through serialize()/deserialize() so the resulting Const has m_node reset,
// mirroring the state a Const is really in when read_weight() runs during blob import.
LazyTensor roundtrip(const LazyTensor& var) {
    using namespace ov::npuw::s11n;
    LazyTensor res;
    std::stringstream ss;
    write(ss, var);
    read(ss, res);
    return res;
}

// Builds a real on-disk weights file and returns a WeightsContext wired to it via ctx.weights + ctx.weights_path,
// the same "mmap the whole file, save the path" branch Const::read_weight() takes for a plain weightless Const.
WeightsContext make_file_backed_context(const std::filesystem::path& file_path, std::size_t file_size) {
    {
        std::ofstream os(file_path, std::ios::binary);
        std::vector<uint8_t> bytes(file_size, 0xAB);
        os.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    auto mapped = ov::load_mmap_object(file_path);
    auto weights = std::make_shared<Weights>(reinterpret_cast<char*>(mapped->data()), mapped->size(), mapped);
    return WeightsContext(weights, file_path.string(), {}, {});
}

}  // namespace

TEST(LazyTensorFileBackedTest, PlainFileBackedConst_ReturnsCorrectInfo) {
    auto constant = make_weightless_constant(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f}, 0);
    LazyTensor res = roundtrip(LazyTensor(constant));

    std::filesystem::path file_path = ov::test::utils::generateTestFilePrefix() + "_lazy_tensor_file_backed.bin";
    res.read_weight(make_file_backed_context(file_path, constant->get_byte_size()));

    auto info = res.get_file_backed_info();
    ASSERT_TRUE(info.has_value());
    EXPECT_EQ(info->weights_path, file_path.string());
    EXPECT_EQ(info->offset, 0u);
    EXPECT_EQ(info->byte_size, constant->get_byte_size());

    std::filesystem::remove(file_path);
}

TEST(LazyTensorFileBackedTest, HandleProviderConst_ReturnsNullopt) {
    auto constant = make_weightless_constant(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f}, 0);
    LazyTensor res = roundtrip(LazyTensor(constant));

    std::filesystem::path file_path = ov::test::utils::generateTestFilePrefix() + "_lazy_tensor_handle_provider.bin";
    WeightsContext ctx = make_file_backed_context(file_path, constant->get_byte_size());
    // Never invoked: get_file_backed_info() doesn't call eval(), it just has to see a non-null handle_provider.
    ctx.handle_provider = []() -> ov::FileHandle {
        return ov::FileHandle{};
    };
    res.read_weight(ctx);

    EXPECT_FALSE(res.get_file_backed_info().has_value());

    std::filesystem::remove(file_path);
}

TEST(LazyTensorFileBackedTest, Bf16ConvertedConst_ReturnsNullopt) {
    std::vector<ov::float16> f16_values = {ov::float16(1.0f), ov::float16(-2.5f)};
    auto constant = std::make_shared<ov::op::v0::Constant>(ov::element::f16, ov::Shape{2}, f16_values);
    constant->get_rt_info()[ov::WeightlessCacheAttribute::get_type_info_static()] =
        ov::WeightlessCacheAttribute(constant->get_byte_size(), 0, ov::element::f16);
    LazyTensor res = roundtrip(LazyTensor(constant));

    std::filesystem::path file_path = ov::test::utils::generateTestFilePrefix() + "_lazy_tensor_bf16.bin";
    {
        std::ofstream os(file_path, std::ios::binary);
        std::vector<ov::bfloat16> bf16_values = {ov::bfloat16(1.0f), ov::bfloat16(-2.5f)};
        os.write(reinterpret_cast<const char*>(bf16_values.data()),
                 static_cast<std::streamsize>(bf16_values.size() * sizeof(ov::bfloat16)));
    }
    auto mapped = ov::load_mmap_object(file_path);
    auto weights = std::make_shared<Weights>(reinterpret_cast<char*>(mapped->data()), mapped->size(), mapped);
    WeightsContext ctx(weights, file_path.string(), {}, {{0, constant->get_byte_size()}});
    res.read_weight(ctx);

    EXPECT_FALSE(res.get_file_backed_info().has_value());

    std::filesystem::remove(file_path);
}

TEST(LazyTensorFileBackedTest, ConstsCacheDeserializedConst_ReturnsNullopt) {
    auto constant = make_weightless_constant(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f}, 0);
    LazyTensor res = roundtrip(LazyTensor(constant));

    WeightsContext::ConstsCache consts_cache;
    consts_cache[{0, constant->get_byte_size()}] = constant;
    res.read_weight(WeightsContext(nullptr, "", consts_cache, {}));

    EXPECT_FALSE(res.get_file_backed_info().has_value());
}

TEST(LazyTensorFileBackedTest, ModelAttachedConst_ReturnsNullopt) {
    // Never round-tripped through serialize/deserialize, so m_node stays set - the "live model node" case.
    auto constant =
        std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{2}, std::vector<float>{1.0f, 2.0f});
    LazyTensor var(constant);

    EXPECT_FALSE(var.get_file_backed_info().has_value());
}

TEST(LazyTensorFileBackedTest, PermuteWrappedFileBackedConst_ReturnsNulloptAtOuterLevel) {
    auto constant = make_weightless_constant(ov::element::f32, ov::Shape{1, 2}, {1.0f, 2.0f}, 0);
    LazyTensor inner = roundtrip(LazyTensor(constant));

    std::filesystem::path file_path = ov::test::utils::generateTestFilePrefix() + "_lazy_tensor_permute.bin";
    inner.read_weight(make_file_backed_context(file_path, constant->get_byte_size()));
    ASSERT_TRUE(inner.get_file_backed_info().has_value());

    LazyTensor permuted = inner.permute({1, 0});
    EXPECT_FALSE(permuted.get_file_backed_info().has_value());

    std::filesystem::remove(file_path);
}

TEST(LazyTensorFileBackedTest, ConvertWrappedFileBackedConst_ReturnsNulloptAtOuterLevel) {
    auto constant = make_weightless_constant(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f}, 0);
    LazyTensor inner = roundtrip(LazyTensor(constant));

    std::filesystem::path file_path = ov::test::utils::generateTestFilePrefix() + "_lazy_tensor_convert.bin";
    inner.read_weight(make_file_backed_context(file_path, constant->get_byte_size()));
    ASSERT_TRUE(inner.get_file_backed_info().has_value());

    LazyTensor converted = inner.convert(ov::element::f16);
    EXPECT_FALSE(converted.get_file_backed_info().has_value());

    std::filesystem::remove(file_path);
}
