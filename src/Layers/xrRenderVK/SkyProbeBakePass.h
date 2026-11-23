#pragma once

#include "../xrRenderGraph/SkyProbeLayout.h"
#include "SkyProbeReducePass.h"

namespace xray::render::vulkan
{
inline constexpr uint32_t SkyProbeRaysPerFace = 8;

inline bool sample_sky_probes(const framegraph::SkyProbeLayout& layout,
    const std::vector<SkyProbeVisibility>& probes, const std::array<float, 3>& position,
    std::array<float, 6>& faces)
{
    if (!layout.count || probes.empty() || probes.size() > layout.count)
        return false;
    std::array<uint32_t, 3> base{};
    std::array<float, 3> fraction{};
    for (unsigned axis = 0; axis < 3; ++axis)
    {
        const uint32_t last = layout.dimensions[axis] - 1;
        const double end = double(layout.origin[axis]) + double(layout.spacing) * last;
        if (!std::isfinite(position[axis]) || position[axis] < layout.origin[axis] ||
            position[axis] > static_cast<float>(end))
            return false;
        double coordinate = std::min((double(position[axis]) - layout.origin[axis]) / layout.spacing,
            double(last));
        // Positions produced by the bake are rounded to float. At an exact
        // baked position, avoid weighting an adjacent, potentially unbaked probe.
        const double nearest = std::round(coordinate);
        if (position[axis] == static_cast<float>(double(layout.origin[axis]) + double(layout.spacing) * nearest))
            coordinate = nearest;
        base[axis] = std::min(static_cast<uint32_t>(coordinate), last - 1);
        fraction[axis] = static_cast<float>(coordinate - base[axis]);
    }
    std::array<float, 6> result{};
    for (unsigned corner = 0; corner < 8; ++corner)
    {
        const uint32_t x = base[0] + (corner & 1u);
        const uint32_t y = base[1] + ((corner >> 1) & 1u);
        const uint32_t z = base[2] + ((corner >> 2) & 1u);
        float weight = 1.f;
        for (unsigned axis = 0; axis < 3; ++axis)
            weight *= (corner & (1u << axis)) ? fraction[axis] : 1.f - fraction[axis];
        if (weight == 0.f)
            continue;
        const uint32_t index = (z * layout.dimensions[1] + y) * layout.dimensions[0] + x;
        if (index >= probes.size())
            return false;
        for (unsigned face = 0; face < result.size(); ++face)
        {
            const float visibility = probes[index].faces[face];
            if (!(visibility >= 0.f && visibility <= 1.f))
                return false;
            result[face] += visibility * weight;
        }
    }
    faces = result;
    return true;
}

inline bool generate_sky_probe_rays(const framegraph::SkyProbeLayout& layout, uint32_t first,
    uint32_t count, float distance, std::vector<RayQueryInput>& rays)
{
    if (!count || uint64_t(first) + count > layout.count || !std::isfinite(distance) || distance <= .05f)
        return false;
    rays.clear();
    rays.reserve(size_t(count) * 6 * SkyProbeRaysPerFace);
    static const auto directions = [] {
        std::array<std::array<float, 3>, 6 * SkyProbeRaysPerFace> result{};
        constexpr float golden_angle = 2.39996323f;
        for (unsigned face = 0; face < 6; ++face)
        {
            const unsigned axis = face / 2;
            for (unsigned ray = 0; ray < SkyProbeRaysPerFace; ++ray)
            {
                const float radius = .7f * std::sqrt((float(ray) + .5f) / SkyProbeRaysPerFace);
                const float angle = ray * golden_angle;
                auto& direction = result[face * SkyProbeRaysPerFace + ray];
                direction[axis] = ((face & 1) ? -1.f : 1.f) * std::sqrt(1.f - radius * radius);
                direction[(axis + 1) % 3] = radius * std::cos(angle);
                direction[(axis + 2) % 3] = radius * std::sin(angle);
            }
        }
        return result;
    }();
    for (uint32_t probe = first; probe < first + count; ++probe)
    {
        std::array<float, 3> origin{};
        if (!layout.position(probe, origin))
            return false;
        for (const auto& direction : directions)
        {
            RayQueryInput value{};
            std::copy(origin.begin(), origin.end(), value.origin_min);
            value.origin_min[3] = .05f;
            std::copy(direction.begin(), direction.end(), value.direction_max);
            value.direction_max[3] = distance;
            rays.push_back(value);
        }
    }
    return true;
}

// Each slice is a serial query and reduction. The caller advances the slice
// only after committing submission and fences each descriptor slot before reuse.
class SkyProbeBakePass
{
public:
    static constexpr uint32_t ProbesPerSlice = 32;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        uint32_t slots, VkPipelineCache cache, const BindlessTextureTable& textures,
        const SkyProbeReducePass::ShaderSource& source, std::string& error)
    {
        slots_.clear();
        if (!query_.initialize(device, get, slots, cache, textures, error) ||
            !reduce_.initialize(device, get, limits, slots, cache, source, error))
            return false;

        constexpr auto input_bytes = ProbesPerSlice * 6 * SkyProbeRaysPerFace * sizeof(RayQueryInput);
        constexpr auto output_bytes = ProbesPerSlice * 6 * SkyProbeRaysPerFace * sizeof(RayQueryHit);
        constexpr auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        slots_.resize(slots);
        for (auto& slot : slots_)
            if (!slot.rays.initialize(device, input_bytes, usage,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, dispatch, error) ||
                !slot.hits.initialize(device, output_bytes, usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, dispatch, error))
            {
                slots_.clear();
                return false;
            }
        return true;
    }

    bool record(VkCommandBuffer command, uint32_t slot, const LevelRayScene& scene,
        const framegraph::SkyProbeLayout& layout, const BufferResource& probes, uint32_t first,
        uint32_t count, float ray_distance, std::string& error)
    {
        if (slot >= slots_.size() || count > ProbesPerSlice ||
            !generate_sky_probe_rays(layout, first, count, ray_distance, scratch_))
        {
            error = "invalid sky probe bake slice";
            return false;
        }
        auto& buffers = slots_[slot];
        return buffers.rays.write(0, scratch_.data(), scratch_.size() * sizeof(RayQueryInput), error) &&
            query_.record(command, slot, scene, buffers.rays, buffers.hits,
                static_cast<uint32_t>(scratch_.size()), error) &&
            reduce_.record(command, slot, buffers.hits, probes, first, count, SkyProbeRaysPerFace, error);
    }

private:
    struct Slot { BufferResource rays, hits; };
    RayQueryBatch query_;
    SkyProbeReducePass reduce_;
    std::vector<Slot> slots_;
    std::vector<RayQueryInput> scratch_;
};
}
