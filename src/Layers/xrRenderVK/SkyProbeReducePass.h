#pragma once

#include "RayQueryBatch.h"
#include "StorageComputePass.h"
#include <array>
#include <cstring>
#include <functional>

namespace xray::render::vulkan
{
struct SkyProbeVisibility
{
    std::array<float, 6> faces{};
    std::array<float, 2> padding{};
};
static_assert(sizeof(SkyProbeVisibility) == 32);

class SkyProbeReducePass
{
public:
    using ShaderSource = std::function<bool(const char*, std::vector<uint8_t>&, std::string&)>;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t frame_slots, VkPipelineCache cache, const ShaderSource& source, std::string& error)
    {
        std::vector<uint8_t> binary;
        if (!source("vk/sky_probe_reduce.cs.spv", binary, error))
            return false;
        if (binary.size() < 20 || binary.size() % sizeof(uint32_t))
        {
            error = "invalid sky probe reduction shader";
            return false;
        }
        std::vector<uint32_t> words(binary.size() / sizeof(uint32_t));
        std::memcpy(words.data(), binary.data(), binary.size());
        max_groups_ = limits.maxComputeWorkGroupCount[0];
        return pass_.initialize(device, get, frame_slots, 2, sizeof(Slice), limits.maxStorageBufferRange,
            cache, words.data(), binary.size(), error);
    }

    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& hits,
        const BufferResource& probes, uint32_t first, uint32_t count, uint32_t rays_per_face,
        std::string& error) const
    {
        const uint64_t groups = (uint64_t(count) + 63) / 64;
        const uint64_t ray_stride = uint64_t(rays_per_face) * 6;
        if (!count || !ray_stride || groups > max_groups_ ||
            ray_stride > hits.size() / sizeof(RayQueryHit) / count ||
            uint64_t(first) + count > probes.size() / sizeof(SkyProbeVisibility))
        {
            error = "invalid sky probe reduction slice";
            return false;
        }
        const Slice slice{first, count, rays_per_face};
        return pass_.record(command, slot, {{hits.handle(), 0, hits.size()}, {probes.handle(), 0, probes.size()}},
            &slice, static_cast<uint32_t>(groups), 1, error);
    }

private:
    struct Slice { uint32_t first, count, rays_per_face; };
    StorageComputePass pass_;
    uint32_t max_groups_{};
};
}
