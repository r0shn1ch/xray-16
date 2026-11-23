#include "GameShaderResources.h"
#include "DeferredPass.h"
#include "GameShaderInterface.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>
#include <vector>

namespace xray::render::vulkan
{
namespace
{
bool specialization_values(const VkSpecializationInfo* info,
    std::vector<ShaderSpecializationValue>& values, std::string& error)
{
    values.clear();
    if (!info)
    {
        error.clear();
        return true;
    }
    if (info->mapEntryCount && (!info->pMapEntries || !info->pData))
    {
        error = "invalid Vulkan shader specialization info";
        return false;
    }
    values.reserve(info->mapEntryCount);
    for (uint32_t index = 0; index < info->mapEntryCount; ++index)
    {
        const VkSpecializationMapEntry& entry = info->pMapEntries[index];
        if (!entry.size || entry.size > sizeof(uint64_t) || entry.offset > info->dataSize ||
            entry.size > info->dataSize - entry.offset)
        {
            error = "Vulkan shader specialization entry is outside its data block";
            return false;
        }
        uint64_t value{};
        std::memcpy(&value, static_cast<const uint8_t*>(info->pData) + entry.offset, entry.size);
        values.push_back({entry.constantID, static_cast<uint32_t>(entry.size), value});
    }
    error.clear();
    return true;
}

uint64_t hash_specializations(const std::array<const VkSpecializationInfo*, 5>& infos)
{
    struct Entry
    {
        uint32_t id{}, size{};
        uint64_t value{};
        bool operator<(const Entry& other) const
        {
            return std::tie(id, size, value) < std::tie(other.id, other.size, other.value);
        }
    };
    bool any = false;
    uint64_t hash = 14695981039346656037ull;
    const auto append = [&hash](const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    for (size_t stage = 0; stage < infos.size(); ++stage)
    {
        const VkSpecializationInfo* info = infos[stage];
        if (!info || !info->mapEntryCount)
            continue;
        any = true;
        append(&stage, sizeof(stage));
        std::vector<Entry> entries;
        entries.reserve(info->mapEntryCount);
        for (uint32_t i = 0; i < info->mapEntryCount; ++i)
        {
            const VkSpecializationMapEntry& source = info->pMapEntries[i];
            Entry entry{source.constantID, static_cast<uint32_t>(source.size), 0};
            std::memcpy(&entry.value, static_cast<const uint8_t*>(info->pData) + source.offset, source.size);
            entries.push_back(entry);
        }
        std::sort(entries.begin(), entries.end());
        for (const Entry& entry : entries)
        {
            append(&entry.id, sizeof(entry.id));
            append(&entry.size, sizeof(entry.size));
            append(&entry.value, entry.size);
        }
    }
    return any ? hash : 0;
}

uint64_t hash_shader_resources(const std::array<const GameShaderResource*, 5>& resources)
{
    uint64_t hash = 14695981039346656037ull;
    const auto append = [&hash](const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    for (size_t stage = 0; stage < resources.size(); ++stage)
    {
        const bool present = resources[stage] != nullptr;
        append(&stage, sizeof(stage));
        append(&present, sizeof(present));
        if (!present)
            continue;
        append(&resources[stage]->stage, sizeof(resources[stage]->stage));
        append(&resources[stage]->content_fingerprint, sizeof(resources[stage]->content_fingerprint));
        const size_t name_size = resources[stage]->name.size();
        append(&name_size, sizeof(name_size));
        append(resources[stage]->name.data(), name_size);
    }
    return hash;
}

uint64_t hash_shader_defines(const std::vector<std::string>& defines)
{
    if (defines.empty())
        return 0;
    uint64_t hash = 14695981039346656037ull;
    const auto append = [&hash](const void* data, size_t size) {
        const auto* bytes = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < size; ++i)
        {
            hash ^= bytes[i];
            hash *= 1099511628211ull;
        }
    };
    for (const std::string& define : defines)
    {
        const size_t size = define.size();
        append(&size, sizeof(size));
        append(define.data(), size);
    }
    return hash;
}

uint64_t hash_pipeline_variant(uint64_t shader_variant, uint64_t specialization_variant)
{
    if (!specialization_variant)
        return shader_variant;
    uint64_t hash = 14695981039346656037ull;
    for (uint64_t value : {shader_variant, specialization_variant})
    {
        for (unsigned shift = 0; shift < 64; shift += 8)
        {
            hash ^= static_cast<uint8_t>(value >> shift);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}
}

bool GameShaderResources::pipeline(DeferredPass &pass, const std::string &svs, const std::string &sps,
    SurfaceMode mode, bool hud, bool skinned, std::string &error,
    const std::string &shs, const std::string &sds, const std::string &sgs,
    const VkSpecializationInfo* vertex_specialization,
    const VkSpecializationInfo* hull_specialization,
    const VkSpecializationInfo* domain_specialization,
    const VkSpecializationInfo* geometry_specialization,
    const VkSpecializationInfo* pixel_specialization,
    const std::vector<std::string>& defines, uint64_t* variant_hash)
{
    std::string vertex, pixel;
    if (!canonical_name(svs, GameShaderStage::Vertex, vertex, error) ||
        !canonical_name(sps, GameShaderStage::Pixel, pixel, error))
        return false;
    if (mode != SurfaceMode::Opaque && mode != SurfaceMode::AlphaTest && mode != SurfaceMode::Transparent)
    {
        error = "invalid legacy SVS/SPS surface mode: " + vertex + " / " + pixel;
        return false;
    }
    if (hud && mode != SurfaceMode::Opaque)
    {
        error = "invalid legacy SVS/SPS HUD pass parameters: " + vertex + " / " + pixel;
        return false;
    }
    if (shs.empty() != sds.empty())
    {
        error = "game tessellation requires both hull and domain shaders";
        return false;
    }
    DeferredPass::GamePipelineStages stages;
    if (!shs.empty() &&
        (!canonical_name(shs, GameShaderStage::Hull, stages.hull_name, error) ||
            !canonical_name(sds, GameShaderStage::Domain, stages.domain_name, error)))
        return false;
    if (!sgs.empty() && !canonical_name(sgs, GameShaderStage::Geometry, stages.geometry_name, error))
        return false;
    if ((!shs.empty() && !tessellation_supported_) || (!sgs.empty() && !geometry_supported_))
    {
        error = "game shader pipeline requests unsupported Vulkan tessellation/geometry features";
        return false;
    }
    std::vector<ShaderSpecializationValue> vertex_values, hull_values, domain_values,
        geometry_values, pixel_values;
    if (!specialization_values(vertex_specialization, vertex_values, error) ||
        !specialization_values(hull_specialization, hull_values, error) ||
        !specialization_values(domain_specialization, domain_values, error) ||
        !specialization_values(geometry_specialization, geometry_values, error) ||
        !specialization_values(pixel_specialization, pixel_values, error))
        return false;
    const auto resolve_defines = [this, &defines, &error](const std::string& name,
        GameShaderStage stage, std::vector<std::string>& effective) {
        return pipeline_defines(name, stage, "main", defines, effective, error);
    };
    std::vector<std::string> vertex_defines, pixel_defines, hull_defines, domain_defines,
        geometry_defines;
    if (!resolve_defines(vertex, GameShaderStage::Vertex, vertex_defines) ||
        !resolve_defines(pixel, GameShaderStage::Pixel, pixel_defines) ||
        (!stages.hull_name.empty() &&
            (!resolve_defines(stages.hull_name, GameShaderStage::Hull, hull_defines) ||
                !resolve_defines(stages.domain_name, GameShaderStage::Domain, domain_defines))) ||
        (!stages.geometry_name.empty() &&
            !resolve_defines(stages.geometry_name, GameShaderStage::Geometry, geometry_defines)))
        return false;
    std::vector<std::string> variant_defines;
    const auto add_stage_defines = [&variant_defines](const char* stage,
        const std::vector<std::string>& stage_defines) {
        for (const std::string& define : stage_defines)
            variant_defines.emplace_back(std::string(stage) + ":" + define);
    };
    add_stage_defines("vs", vertex_defines);
    add_stage_defines("hs", hull_defines);
    add_stage_defines("ds", domain_defines);
    add_stage_defines("gs", geometry_defines);
    add_stage_defines("ps", pixel_defines);
    const uint64_t shader_variant_hash = hash_shader_defines(variant_defines);
    if (variant_hash)
        *variant_hash = shader_variant_hash;
    GameShaderResource *vs = nullptr, *ps = nullptr;
    if (!shader(vertex, GameShaderStage::Vertex, "main", 0, vs, error,
                nullptr, 0, false, vertex_defines) ||
        !shader(pixel, GameShaderStage::Pixel, "main", 0, ps, error,
                nullptr, 0, false, pixel_defines))
        return false;
    GameShaderResource *hs = nullptr, *ds = nullptr, *gs = nullptr;
    if (!shs.empty() &&
        (!shader(stages.hull_name, GameShaderStage::Hull, "main", 0, hs, error,
                nullptr, 0, false, hull_defines) ||
            !shader(stages.domain_name, GameShaderStage::Domain, "main", 0, ds, error,
                nullptr, 0, false, domain_defines)))
        return false;
    if (!sgs.empty() &&
        !shader(stages.geometry_name, GameShaderStage::Geometry, "main", 0, gs, error,
            nullptr, 0, false, geometry_defines))
        return false;
    if ((!hs && hull_specialization && hull_specialization->mapEntryCount) ||
        (!ds && domain_specialization && domain_specialization->mapEntryCount) ||
        (!gs && geometry_specialization && geometry_specialization->mapEntryCount))
    {
        error = "Vulkan specialization info was supplied for a missing shader stage";
        return false;
    }
    stages.hull = hs ? hs->module.handle() : VK_NULL_HANDLE;
    stages.domain = ds ? ds->module.handle() : VK_NULL_HANDLE;
    stages.geometry = gs ? gs->module.handle() : VK_NULL_HANDLE;
    stages.vertex_specialization = vertex_specialization;
    stages.hull_specialization = hull_specialization;
    stages.domain_specialization = domain_specialization;
    stages.geometry_specialization = geometry_specialization;
    stages.fragment_specialization = pixel_specialization;
    stages.specialization_hash = hash_specializations({vertex_specialization, hull_specialization,
        domain_specialization, geometry_specialization, pixel_specialization});
    stages.shader_variant_hash = hash_pipeline_variant(shader_variant_hash, stages.specialization_hash);
    if (variant_hash)
        *variant_hash = stages.shader_variant_hash;
    struct StageInterface
    {
        GameShaderResource** resource;
        const std::vector<ShaderSpecializationValue>* specialization_values;
    };
    const std::array<StageInterface, 5> interfaces{{
        {&vs, &vertex_values}, {&hs, &hull_values}, {&ds, &domain_values},
        {&gs, &geometry_values}, {&ps, &pixel_values}}};
    std::array<bool, SurfaceMaterialTextureCount> material_bindings_seen{};
    std::array<bool, 6> weather_bindings_seen{};
    const uint32_t weather_set_index = skinned ? 3u : 2u;
    const bool allow_weather_arrays = mode == SurfaceMode::Transparent && !hud;
    for (const StageInterface& stage_interface : interfaces)
    {
        GameShaderResource* resource = *stage_interface.resource;
        if (!resource || validate_game_shader_interface(resource->reflection,
                skinned, *stage_interface.specialization_values, error, allow_weather_arrays))
            continue;

        const std::string precompiled_interface_error = error;
        if (!resource->precompiled_rejection.empty())
        {
            error = "legacy shader '" + resource->name + "': " + precompiled_interface_error +
                "; runtime HLSL fallback was already used: " + resource->precompiled_rejection;
            return false;
        }

        // A valid SPIR-V module can still describe a layout that the legacy
        // material pipeline cannot bind. Retry the exact active HLSL source
        // before failing device initialization, especially when a non-manifest
        // game .spv bypassed the sidecar fingerprint checks.
        GameShaderResource* runtime_resource = nullptr;
        std::string runtime_error;
        const uint8_t* runtime_source = resource->runtime_source.empty() ? nullptr :
            resource->runtime_source.data();
        if (shader(resource->name, resource->stage, resource->entry_point.c_str(),
                resource->compile_flags, runtime_resource,
                runtime_error, runtime_source, resource->runtime_source.size(), true,
                resource->runtime_defines) && runtime_resource &&
            validate_game_shader_interface(runtime_resource->reflection, skinned,
                *stage_interface.specialization_values, runtime_error, allow_weather_arrays))
        {
            *stage_interface.resource = runtime_resource;
            continue;
        }
        error = "legacy shader '" + resource->name + "': " + precompiled_interface_error +
            "; runtime HLSL fallback failed: " +
            (runtime_error.empty() ? "shader compilation or interface validation failed" : runtime_error);
        return false;
    }
    for (const StageInterface& stage_interface : interfaces)
    {
        const GameShaderResource* resource = *stage_interface.resource;
        if (!resource)
            continue;
        std::vector<uint32_t> descriptor_counts;
        if (!resolve_shader_descriptor_counts(resource->reflection,
                *stage_interface.specialization_values, descriptor_counts, error))
            return false;
        for (size_t index = 0; index < resource->reflection.descriptors.size(); ++index)
        {
            const ShaderDescriptor& descriptor = resource->reflection.descriptors[index];
            const bool material = descriptor.set == 0 &&
                descriptor.kind == ShaderResourceKind::CombinedImageSampler;
            const bool weather = descriptor.set == weather_set_index &&
                descriptor.kind == ShaderResourceKind::CombinedImageSampler;
            if (!material && !weather)
                continue;
            const uint32_t binding_limit = material ? SurfaceMaterialTextureCount : 6u;
            if (descriptor.binding >= binding_limit)
            {
                error = "legacy shader '" + resource->name + "' uses an invalid " +
                    (material ? "material" : "weather") + " descriptor binding " +
                    std::to_string(descriptor.binding);
                return false;
            }
            if (index >= descriptor_counts.size() || descriptor_counts[index] == 0 ||
                descriptor_counts[index] > binding_limit - descriptor.binding)
            {
                error = "legacy shader '" + resource->name +
                    "' uses an unsupported " + (material ? "material" : "weather") +
                    " descriptor array at binding " +
                    std::to_string(descriptor.binding);
                return false;
            }
            const uint32_t descriptor_count = descriptor_counts[index];
            if (material && material_bindings_seen[descriptor.binding] &&
                stages.material_descriptor_counts[descriptor.binding] != descriptor_count)
            {
                error = "legacy shader stages disagree on material descriptor count at binding " +
                    std::to_string(descriptor.binding);
                return false;
            }
            if (weather && weather_bindings_seen[descriptor.binding] &&
                stages.weather_descriptor_counts[descriptor.binding] != descriptor_count)
            {
                error = "legacy shader stages disagree on weather descriptor count at binding " +
                    std::to_string(descriptor.binding);
                return false;
            }
            if (material)
            {
                material_bindings_seen[descriptor.binding] = true;
                stages.material_descriptor_counts[descriptor.binding] = descriptor_count;
            }
            else
            {
                weather_bindings_seen[descriptor.binding] = true;
                stages.weather_descriptor_counts[descriptor.binding] = descriptor_count;
            }
            if (descriptor.image.dimension == ShaderImageDimension::Unknown)
                continue;
            for (uint32_t element = 0; element < descriptor_count; ++element)
            {
                const uint32_t slot = descriptor.binding + element;
                ShaderImageShape& existing_shape = material ? stages.material_image_shapes[slot] :
                    stages.weather_image_shapes[slot];
                if (existing_shape.dimension != ShaderImageDimension::Unknown &&
                    (existing_shape.dimension != descriptor.image.dimension ||
                        existing_shape.arrayed != descriptor.image.arrayed ||
                        existing_shape.multisampled != descriptor.image.multisampled))
                {
                    error = std::string("legacy shader stages disagree on ") +
                        (material ? "material" : "weather") +
                        " image shape at binding " +
                        std::to_string(slot) + " (descriptor array starts at binding " +
                        std::to_string(descriptor.binding) + ")";
                    return false;
                }
                existing_shape = descriptor.image;
            }
        }
    }
    stages.shader_fingerprint = hash_shader_resources({vs, hs, ds, gs, ps});
    stages.hull = hs ? hs->module.handle() : VK_NULL_HANDLE;
    stages.domain = ds ? ds->module.handle() : VK_NULL_HANDLE;
    stages.geometry = gs ? gs->module.handle() : VK_NULL_HANDLE;
    const GameShaderResource* previous = vs;
    for (const GameShaderResource* next : {hs, ds, gs, ps})
    {
        if (!next) continue;
        if (!validate_shader_stage_link(previous->reflection, next->reflection, error))
        {
            error = "legacy shader stage link '" + previous->name + "' -> '" +
                next->name + "': " + error;
            return false;
        }
        previous = next;
    }
    if (!pass.create_game_pipeline(vs->name, ps->name, vs->module.handle(), ps->module.handle(),
            mode, hud, error, skinned, stages))
        return false;
    std::string key = vs->name;
    key.push_back('\0');
    key += ps->name;
    key.push_back('\0');
    key += std::to_string(stages.shader_variant_hash);
    if (pipeline_recipes_.find(key) == pipeline_recipes_.end())
    {
        PipelineRecipe recipe{vs->name, ps->name, vs->module.handle(), ps->module.handle(), mode, hud, skinned, stages};
        const VkSpecializationInfo** fields[]{&recipe.stages.vertex_specialization, &recipe.stages.hull_specialization,
            &recipe.stages.domain_specialization, &recipe.stages.geometry_specialization, &recipe.stages.fragment_specialization};
        for (size_t stage = 0; stage < recipe.specialization.size(); ++stage)
        {
            const auto* info = *fields[stage];
            *fields[stage] = nullptr;
            if (!info || !info->mapEntryCount)
                continue;
            auto& data = recipe.specialization[stage];
            data.entries.assign(info->pMapEntries, info->pMapEntries + info->mapEntryCount);
            const auto* bytes = static_cast<const uint8_t*>(info->pData);
            data.bytes.assign(bytes, bytes + info->dataSize);
        }
        pipeline_recipes_.emplace(std::move(key), std::move(recipe));
    }
    return true;
}

bool GameShaderResources::rebuild_pipelines(DeferredPass& pass, std::string& error) const
{
    pass.begin_game_pipeline_reload();
    for (const auto& entry : pipeline_recipes_)
    {
        const auto& recipe = entry.second;
        auto stages = recipe.stages;
        std::array<VkSpecializationInfo, 5> info{};
        const VkSpecializationInfo** fields[]{&stages.vertex_specialization, &stages.hull_specialization,
            &stages.domain_specialization, &stages.geometry_specialization, &stages.fragment_specialization};
        for (size_t stage = 0; stage < info.size(); ++stage)
        {
            const auto& data = recipe.specialization[stage];
            if (data.entries.empty())
                continue;
            info[stage] = {static_cast<uint32_t>(data.entries.size()), data.entries.data(),
                data.bytes.size(), data.bytes.data()};
            *fields[stage] = &info[stage];
        }
        if (!pass.create_game_pipeline(recipe.vertex_name, recipe.fragment_name, recipe.vertex, recipe.fragment,
                recipe.mode, recipe.hud, error, recipe.skinned, stages))
        {
            pass.abort_game_pipeline_reload();
            return false;
        }
    }
    pass.commit_game_pipeline_reload();
    error.clear();
    return true;
}
} // namespace xray::render::vulkan
