#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
enum class ShaderResourceKind : uint8_t
{
    Sampler,
    SampledImage,
    StorageImage,
    CombinedImageSampler,
    UniformBuffer,
    StorageBuffer,
    AccelerationStructure,
    Unknown
};

enum class ShaderImageDimension : uint8_t
{
    Unknown, OneD, TwoD, ThreeD, Cube, Rect, Buffer, Subpass
};

struct ShaderImageShape
{
    ShaderImageDimension dimension{ShaderImageDimension::Unknown};
    bool arrayed{};
    bool multisampled{};
};

struct ShaderDescriptor
{
    uint32_t set{};
    uint32_t binding{};
    ShaderResourceKind kind{ShaderResourceKind::Unknown};
    // Zero denotes a runtime or unresolved descriptor array length. For fixed
    // arrays, count reflects the module defaults; resolve it again when a
    // pipeline supplies specialization values.
    uint32_t count{1};
    std::vector<uint32_t> array_length_ids;
    bool runtime_sized{};
    ShaderImageShape image{};
};

struct ShaderConstantExpression
{
    uint32_t result_id{};
    uint32_t specialization_id{~0u};
    uint32_t operation{}; // Zero is a literal; otherwise a SPIR-V opcode.
    uint32_t bit_width{};
    uint32_t signedness{};
    uint64_t default_value{};
    std::vector<uint32_t> operands;
};

struct ShaderSpecializationValue
{
    uint32_t constant_id{};
    uint32_t byte_size{};
    uint64_t value{};
};

struct ShaderInterfaceVariable
{
    uint32_t location{};
    uint32_t component{};
    uint32_t scalar_opcode{};
    uint32_t width{};
    uint32_t signedness{};
    uint32_t vector_size{};
    bool type_known{};
};

struct ShaderReflection
{
    uint32_t stage{};
    std::vector<ShaderDescriptor> descriptors;
    std::vector<ShaderConstantExpression> constants;
    std::vector<ShaderInterfaceVariable> inputs, outputs;
    bool interface_complete{true};
    bool push_constants{};
    bool push_constant_extent_known{};
    uint32_t push_constant_bytes{};
};

// Reflects the resource declarations of one SPIR-V module. This is a structural
// inventory. Fixed-size arrays, matrices, structure members, and 32/64-bit
// interface values are flattened for stage linking. Descriptor-array extents
// retain their SPIR-V constant expressions and can be resolved with pipeline
// specialization values. Unsupported interface types remain explicitly
// incomplete and fail stage-link validation.
bool reflect_spirv_resources(const void* bytes, size_t size, uint32_t stage,
    const char* entry, ShaderReflection& result, std::string& error);
// Resolves fixed descriptor-array extents using Vulkan specialization IDs.
// Runtime descriptor arrays resolve to zero. Values must match the scalar
// widths declared by their specialization constants.
bool resolve_shader_descriptor_counts(const ShaderReflection& reflection,
    const std::vector<ShaderSpecializationValue>& values,
    std::vector<uint32_t>& counts, std::string& error);
bool validate_shader_stage_link(const ShaderReflection& producer,
    const ShaderReflection& consumer, std::string& error);
}
