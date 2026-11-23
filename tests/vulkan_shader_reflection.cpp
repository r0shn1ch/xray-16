#include "src/Layers/xrRenderVK/ShaderReflection.h"
#include "src/Layers/xrRenderVK/VulkanGameShaders.h"
#include "src/Layers/xrRenderVK/DeferredShaders.h"
#include "src/Layers/xrRenderVK/PostProcessShaders.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

int main()
{
    using namespace xray::render::vulkan;
    const std::vector<uint32_t> words{
        0x07230203, 0x00010000, 0, 16, 0,
        0x0005000f, 4, 1, 0x6e69616d, 0, // fragment main
        0x00030016, 2, 32, // float
        0x00090019, 3, 2, 1, 0, 0, 0, 1, 0, // sampled 2D image
        0x0003001b, 4, 3, // combined image sampler
        0x00040020, 5, 0, 4, // UniformConstant pointer
        0x0003001e, 6, 2, // struct
        0x00040020, 7, 2, 6, // Uniform pointer
        0x00040020, 8, 9, 6, // PushConstant pointer
        0x00030047, 6, 2, // Block decoration
        0x00050048, 6, 0, 35, 0, // member 0 offset 0
        0x00040047, 10, 34, 0, // set 0
        0x00040047, 10, 33, 3, // binding 3
        0x00040047, 11, 34, 2, // set 2
        0x00040047, 11, 33, 1, // binding 1
        0x0004003b, 5, 10, 0, // combined sampler variable
        0x0004003b, 7, 11, 2, // uniform buffer variable
        0x0004003b, 8, 12, 9, // push constant variable
    };
    ShaderReflection reflection;
    std::string error;
    assert(reflect_spirv_resources(words.data(), words.size() * 4, 4, "main", reflection, error));
    assert(reflection.stage == 4 && reflection.push_constants && reflection.descriptors.size() == 2);
    assert(reflection.push_constant_extent_known && reflection.push_constant_bytes == 4);
    assert(reflection.descriptors[0].set == 0 && reflection.descriptors[0].binding == 3 &&
        reflection.descriptors[0].kind == ShaderResourceKind::CombinedImageSampler);
    assert(reflection.descriptors[1].set == 2 && reflection.descriptors[1].binding == 1 &&
        reflection.descriptors[1].kind == ShaderResourceKind::UniformBuffer);
    auto storage = words;
    for (size_t i = 5; i < storage.size(); i += storage[i] >> 16)
    {
        if (storage[i] == 0x00090019 && storage[i + 1] == 3) storage[i + 7] = 2;
        if (storage[i] == 0x00040020 && storage[i + 1] == 5) storage[i + 3] = 3;
        if (storage[i] == 0x00030047 && storage[i + 1] == 6) storage[i + 2] = 3;
    }
    assert(reflect_spirv_resources(storage.data(), storage.size() * 4, 4, "main", reflection, error));
    assert(reflection.descriptors[0].kind == ShaderResourceKind::StorageImage);
    assert(reflection.descriptors[1].kind == ShaderResourceKind::StorageBuffer);
    auto array = words;
    array.insert(array.end(), {0x0004002b, 2, 14, 2, // constant array length 2
        0x0004001c, 13, 4, 14}); // array of combined samplers
    for (size_t i = 5; i < array.size(); i += array[i] >> 16)
        if (array[i] == 0x00040020 && array[i + 1] == 5) array[i + 3] = 13;
    assert(reflect_spirv_resources(array.data(), array.size() * 4, 4, "main", reflection, error));
    assert(reflection.descriptors[0].count == 2);
    auto specialized_array = words;
    specialized_array[3] = 18;
    specialized_array.insert(specialized_array.end(), {
        0x00040015, 15, 32, 0, // unsigned 32-bit integer
        0x00040047, 14, 1, 7, // SpecId 7
        0x00040032, 15, 14, 2, // specialization constant default 2
        0x0004002b, 15, 16, 1, // literal integer 1
        0x00060034, 15, 17, 128, 14, 16, // OpSpecConstantOp IAdd
        0x0004001c, 13, 4, 17}); // array of combined samplers
    for (size_t i = 5; i < specialized_array.size(); i += specialized_array[i] >> 16)
        if (specialized_array[i] == 0x00040020 && specialized_array[i + 1] == 5)
            specialized_array[i + 3] = 13;
    assert(reflect_spirv_resources(specialized_array.data(), specialized_array.size() * 4,
        4, "main", reflection, error));
    assert(reflection.descriptors[0].count == 3);
    std::vector<uint32_t> descriptor_counts;
    assert(resolve_shader_descriptor_counts(reflection, {}, descriptor_counts, error));
    assert(descriptor_counts.size() == 2 && descriptor_counts[0] == 3);
    assert(resolve_shader_descriptor_counts(reflection, {{7, 4, 4}}, descriptor_counts, error));
    assert(descriptor_counts[0] == 5);
    assert(!resolve_shader_descriptor_counts(reflection, {{7, 8, 4}}, descriptor_counts, error));
    assert(error.find("size does not match") != std::string::npos);
    assert(!resolve_shader_descriptor_counts(reflection,
        {{7, 4, UINT32_MAX}}, descriptor_counts, error));
    assert(error.find("length could not be resolved") != std::string::npos);
    auto matrix_push = words;
    matrix_push[3] = 18;
    matrix_push.insert(matrix_push.end(), {
        0x00040017, 16, 2, 4, // vec4 of float
        0x00040018, 17, 16, 4, // mat4 of vec4 columns
        0x00050048, 6, 0, 7, 16}); // member MatrixStride 16
    for (size_t i = 5; i < matrix_push.size(); i += matrix_push[i] >> 16)
        if (matrix_push[i] == 0x0003001e && matrix_push[i + 1] == 6)
            matrix_push[i + 2] = 17;
    assert(reflect_spirv_resources(matrix_push.data(), matrix_push.size() * 4,
        4, "main", reflection, error));
    assert(reflection.push_constant_extent_known && reflection.push_constant_bytes == 64);
    assert(!reflect_spirv_resources(words.data(), words.size() * 4, 0, "main", reflection, error));
    assert(reflection.descriptors.empty());

    auto missing_binding = words;
    for (size_t i = 5; i < missing_binding.size(); i += missing_binding[i] >> 16)
        if (missing_binding[i] == 0x00040047 && missing_binding[i + 1] == 10 &&
            missing_binding[i + 2] == 33)
            missing_binding[i + 2] = 0;
    assert(!reflect_spirv_resources(missing_binding.data(), missing_binding.size() * 4,
        4, "main", reflection, error));
    assert(error.find("missing set or binding") != std::string::npos);
    auto unbound_unused = words;
    for (size_t i = 5; i < unbound_unused.size(); i += unbound_unused[i] >> 16)
        if (unbound_unused[i] == 0x00040047 && unbound_unused[i + 1] == 10 &&
            (unbound_unused[i + 2] == 33 || unbound_unused[i + 2] == 34))
            unbound_unused[i + 2] = 0;
    assert(reflect_spirv_resources(unbound_unused.data(), unbound_unused.size() * 4,
        4, "main", reflection, error));
    assert(reflection.descriptors.size() == 1);

    auto malformed = words;
    malformed[5] = 0;
    assert(!reflect_spirv_resources(malformed.data(), malformed.size() * 4,
        4, "main", reflection, error));

    // Exercise a shipped binary as well as the small structural fixtures.
    assert(reflect_spirv_resources(game_shaders::CutoutFragment,
        sizeof(game_shaders::CutoutFragment), 4, "main", reflection, error));
    assert(!reflection.descriptors.empty() && reflection.push_constants);
    assert(reflection.push_constant_extent_known && reflection.push_constant_bytes <= 256);

    const auto io_module = [](uint32_t stage, uint32_t storage_class,
        uint32_t components, uint32_t location) {
        return std::vector<uint32_t>{
            0x07230203, 0x00010000, 0, 6, 0,
            0x0006000f, stage, 1, 0x6e69616d, 0, 5, // main interface variable 5
            0x00030016, 2, 32, // float
            0x00040017, 3, 2, components, // float vector
            0x00040020, 4, storage_class, 3, // pointer
            0x0004003b, 4, 5, storage_class, // interface variable
            0x00040047, 5, 30, location}; // Location
    };
    auto vertex_io = io_module(0, 3, 3, 0);
    auto fragment_io = io_module(4, 1, 3, 0);
    ShaderReflection vertex, fragment;
    assert(reflect_spirv_resources(vertex_io.data(), vertex_io.size() * 4,
        0, "main", vertex, error));
    assert(reflect_spirv_resources(fragment_io.data(), fragment_io.size() * 4,
        4, "main", fragment, error));
    assert(vertex.outputs.size() == 1 && fragment.inputs.size() == 1);
    assert(validate_shader_stage_link(vertex, fragment, error));
    fragment_io = io_module(4, 1, 2, 0);
    assert(reflect_spirv_resources(fragment_io.data(), fragment_io.size() * 4,
        4, "main", fragment, error));
    assert(!validate_shader_stage_link(vertex, fragment, error));
    assert(error.find("type mismatch") != std::string::npos);
    fragment_io = io_module(4, 1, 3, 1);
    assert(reflect_spirv_resources(fragment_io.data(), fragment_io.size() * 4,
        4, "main", fragment, error));
    assert(!validate_shader_stage_link(vertex, fragment, error));
    assert(error.find("no producer") != std::string::npos);
    auto duplicate_io = vertex_io;
    duplicate_io[3] = 7;
    duplicate_io[5] = 0x0007000f;
    duplicate_io.insert(duplicate_io.begin() + 11, 6);
    duplicate_io.insert(duplicate_io.end(), {0x0004003b, 4, 6, 3,
        0x00040047, 6, 30, 0});
    assert(!reflect_spirv_resources(duplicate_io.data(), duplicate_io.size() * 4,
        0, "main", vertex, error));
    assert(error.find("duplicate interface") != std::string::npos);

    const auto block_io_module = [](uint32_t stage, uint32_t storage_class,
        uint32_t location, uint32_t component) {
        return std::vector<uint32_t>{
            0x07230203, 0x00010000, 0, 7, 0,
            0x0006000f, stage, 1, 0x6e69616d, 0, 6, // main interface block 6
            0x00030016, 2, 32, // float
            0x00040017, 3, 2, 3, // vec3
            0x0004001e, 4, 3, 3, // struct: builtin, user varying
            0x00040020, 5, storage_class, 4,
            0x0004003b, 5, 6, storage_class,
            0x00050048, 4, 0, 11, 0, // member 0: BuiltIn Position
            0x00050048, 4, 1, 30, location, // member 1: Location
            0x00050048, 4, 1, 31, component}; // member 1: Component
    };
    auto block_vertex = block_io_module(0, 3, 0, 1);
    auto block_fragment = block_io_module(4, 1, 0, 1);
    assert(reflect_spirv_resources(block_vertex.data(), block_vertex.size() * 4,
        0, "main", vertex, error));
    assert(reflect_spirv_resources(block_fragment.data(), block_fragment.size() * 4,
        4, "main", fragment, error));
    assert(vertex.interface_complete && fragment.interface_complete);
    assert(vertex.outputs.size() == 1 && fragment.inputs.size() == 1);
    assert(vertex.outputs[0].component == 1 && fragment.inputs[0].component == 1);
    assert(validate_shader_stage_link(vertex, fragment, error));
    block_fragment = block_io_module(4, 1, 1, 1);
    assert(reflect_spirv_resources(block_fragment.data(), block_fragment.size() * 4,
        4, "main", fragment, error));
    assert(!validate_shader_stage_link(vertex, fragment, error));
    assert(error.find("no producer") != std::string::npos);

    assert(reflect_spirv_resources(deferred_shaders::GBufferVertex,
        sizeof(deferred_shaders::GBufferVertex), 0, "main", vertex, error));
    assert(reflect_spirv_resources(deferred_shaders::GBufferFragment,
        sizeof(deferred_shaders::GBufferFragment), 4, "main", fragment, error));
    assert(vertex.interface_complete && fragment.interface_complete);
    assert(!fragment.inputs.empty() && validate_shader_stage_link(vertex, fragment, error));

    assert(reflect_spirv_resources(postprocess_shaders::Fragment,
        sizeof(postprocess_shaders::Fragment), 4, "main", fragment, error));
    assert(fragment.descriptors.size() == 8);
    assert(fragment.descriptors.back().set == 0 && fragment.descriptors.back().binding == 7 &&
        fragment.descriptors.back().kind == ShaderResourceKind::SampledImage);
    assert(fragment.push_constant_extent_known && fragment.push_constant_bytes <= 128);
    assert(reflect_spirv_resources(postprocess_shaders::BrightFragment,
        sizeof(postprocess_shaders::BrightFragment), 4, "main", fragment, error));
    assert(fragment.descriptors.size() == 2 && fragment.push_constants);
    assert(reflect_spirv_resources(postprocess_shaders::UpsampleFragment,
        sizeof(postprocess_shaders::UpsampleFragment), 4, "main", fragment, error));
    assert(fragment.descriptors.size() == 2 && fragment.push_constants);
}
