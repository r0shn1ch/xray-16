#pragma once

#include "DetailCullPass.h"
#include <array>

namespace xray::render::vulkan
{
struct DetailGenerateSlot
{
    float bounds[4]{};
    uint32_t data[4]{};
    uint32_t palettes[4]{};
};
struct DetailGroundTriangle { float a[4]{}, b[4]{}, c[4]{}; };
struct GeneratedDetail
{
    float position_scale[4]{};
    float yaw{};
    uint32_t model{UINT32_MAX};
    uint32_t padding[2]{};
};
static_assert(sizeof(DetailGenerateSlot) == 48 && sizeof(DetailGroundTriangle) == 48 &&
    sizeof(GeneratedDetail) == 32);

class DetailGeneratePass
{
public:
    struct Parameters
    {
        uint32_t slot_count{}, grid_size{};
        float sample_spacing{}, density{};
    };

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, const DetailCullPass::Source& source, std::string& error)
    {
        std::vector<uint8_t> binary;
        if (!source("vk/detail_generate.cs.spv", binary, error))
            return false;
        if (binary.size() < 20 || binary.size() % sizeof(uint32_t))
        {
            error = "invalid GPU detail generation shader";
            return false;
        }
        std::vector<uint32_t> words(binary.size() / sizeof(uint32_t));
        std::memcpy(words.data(), binary.data(), binary.size());
        max_groups_x_ = limits.maxComputeWorkGroupCount[0];
        max_groups_y_ = limits.maxComputeWorkGroupCount[1];
        return compute_.initialize(device, get, slots, 4, sizeof(Parameters), limits.maxStorageBufferRange,
            cache, words.data(), binary.size(), error);
    }

    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& slots,
        const BufferResource& ground, const BufferResource& prototypes, const BufferResource& output,
        const Parameters& params, std::string& error) const
    {
        const uint64_t candidates = uint64_t(params.grid_size) * params.grid_size;
        if (!candidates || !params.slot_count || params.slot_count > slots.size() / sizeof(DetailGenerateSlot) ||
            candidates > output.size() / sizeof(GeneratedDetail) / params.slot_count)
        {
            error = "invalid GPU detail generation range";
            return false;
        }
        const uint64_t count = candidates * params.slot_count;
        const uint64_t groups = (count - 1) / 64 + 1;
        const uint32_t groups_x = static_cast<uint32_t>(std::min<uint64_t>(groups, max_groups_x_));
        const uint64_t groups_y = groups_x ? (groups - 1) / groups_x + 1 : 0;
        if (!groups_y || groups_y > max_groups_y_ || count > UINT32_MAX)
        {
            error = "GPU detail generation exceeds dispatch capacity";
            return false;
        }
        return compute_.record(command, slot, {{slots.handle(), 0, slots.size()},
            {ground.handle(), 0, ground.size()}, {prototypes.handle(), 0, prototypes.size()},
            {output.handle(), 0, count * sizeof(GeneratedDetail)}}, &params,
            groups_x, static_cast<uint32_t>(groups_y), error);
    }

private:
    StorageComputePass compute_;
    uint32_t max_groups_x_{}, max_groups_y_{};
};
}
