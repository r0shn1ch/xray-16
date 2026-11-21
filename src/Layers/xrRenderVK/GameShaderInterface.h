#pragma once

#include "DeferredPass.h"
#include "ShaderReflection.h"

#include <string>

namespace xray::render::vulkan
{
inline const char* shader_resource_kind_name(ShaderResourceKind kind)
{
    switch (kind)
    {
    case ShaderResourceKind::Sampler: return "sampler";
    case ShaderResourceKind::SampledImage: return "sampled-image";
    case ShaderResourceKind::StorageImage: return "storage-image";
    case ShaderResourceKind::CombinedImageSampler: return "combined-image-sampler";
    case ShaderResourceKind::UniformBuffer: return "uniform-buffer";
    case ShaderResourceKind::StorageBuffer: return "storage-buffer";
    case ShaderResourceKind::AccelerationStructure: return "acceleration-structure";
    case ShaderResourceKind::Unknown: return "unknown";
    }
    return "invalid";
}

// Matches the descriptor layouts created by DeferredPass for game materials.
inline bool validate_game_shader_interface(const ShaderReflection& reflection,
    bool skinned, const std::vector<ShaderSpecializationValue>& specialization_values,
    std::string& error, bool allow_weather_arrays = false)
{
    const uint32_t weather_set = skinned ? 3 : 2;
    const uint32_t forward_set = skinned ? 2 : 1;
    if (reflection.push_constants && !reflection.push_constant_extent_known)
    {
        error = "game shader push constant extent could not be reflected";
        return false;
    }
    if (reflection.push_constants && reflection.push_constant_bytes > sizeof(float) * 64)
    {
        error = "game shader push constants exceed the Vulkan material pipeline range";
        return false;
    }
    std::vector<uint32_t> descriptor_counts;
    if (!resolve_shader_descriptor_counts(reflection, specialization_values, descriptor_counts, error))
        return false;
    for (size_t descriptor_index = 0; descriptor_index < reflection.descriptors.size(); ++descriptor_index)
    {
        const ShaderDescriptor& descriptor = reflection.descriptors[descriptor_index];
        bool accepted = false;
        if (descriptor.set == 0)
            accepted = descriptor.binding < SurfaceMaterialTextureCount &&
                descriptor.kind == ShaderResourceKind::CombinedImageSampler;
        else if (skinned && descriptor.set == 1)
            accepted = descriptor.binding == 0 && descriptor.kind == ShaderResourceKind::StorageBuffer;
        else if (descriptor.set == forward_set)
            accepted = descriptor.binding == 0 && descriptor.kind == ShaderResourceKind::UniformBuffer;
        else if (descriptor.set == weather_set)
            accepted = descriptor.binding < 6 &&
                descriptor.kind == ShaderResourceKind::CombinedImageSampler;
        const bool count_available = descriptor_index < descriptor_counts.size();
        const uint32_t descriptor_count = count_available ? descriptor_counts[descriptor_index] : 0;
        const bool material_array_supported = descriptor.set == 0 && count_available &&
            !descriptor.runtime_sized && descriptor.binding < SurfaceMaterialTextureCount && descriptor_count > 0 &&
            descriptor_count <= SurfaceMaterialTextureCount - descriptor.binding;
        const bool weather_array_supported = descriptor.set == weather_set && count_available &&
            !descriptor.runtime_sized && descriptor.binding < 6 && descriptor_count > 0 &&
            descriptor_count <= 6 - descriptor.binding && (descriptor_count == 1 || allow_weather_arrays);
        const bool count_supported = descriptor.set == 0 ? material_array_supported :
            descriptor.set == weather_set ? weather_array_supported :
                count_available && descriptor_count == 1;
        if (!accepted || !count_supported)
        {
            error = "game shader descriptor set " + std::to_string(descriptor.set) +
                " binding " + std::to_string(descriptor.binding) +
                " does not match the Vulkan material pipeline layout (kind=" +
                shader_resource_kind_name(descriptor.kind) + ", count=" +
                std::to_string(descriptor_count) + ", count-reflected=" +
                (count_available ? "yes" : "no") + ", runtime-array=" +
                (descriptor.runtime_sized ? "yes" : "no") + ")";
            return false;
        }
    }
    error.clear();
    return true;
}

inline bool validate_game_shader_interface(const ShaderReflection& reflection,
    bool skinned, std::string& error, bool allow_weather_arrays = false)
{
    return validate_game_shader_interface(reflection, skinned, {}, error, allow_weather_arrays);
}
}
