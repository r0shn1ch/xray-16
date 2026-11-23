#pragma once
#include <cstring>

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
        uint32_t compact_cells{};
    };

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, const DetailCullPass::Source& source, std::string& error)
    {
        if (limits.maxPerStageDescriptorStorageBuffers < 5 || limits.maxDescriptorSetStorageBuffers < 5)
        { error = "GPU detail generation needs five storage bindings"; return false; }
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
        max_groups_x_ = std::min(limits.maxComputeWorkGroupCount[0], 65535u);
        max_groups_y_ = std::min(limits.maxComputeWorkGroupCount[1], 65535u);
        max_candidates_ = limits.maxStorageBufferRange / sizeof(GeneratedDetail);
        return compute_.initialize(device, get, slots, 5, sizeof(Parameters), limits.maxStorageBufferRange,
            cache, words.data(), binary.size(), error);
    }

    bool candidate_count(const Parameters& params, uint32_t& count, std::string& error) const
    {
        const uint64_t per_cell = uint64_t(params.grid_size) * params.grid_size;
        if (!per_cell || !params.slot_count || per_cell > max_candidates_ / params.slot_count)
        {
            error = "GPU detail candidates exceed storage buffer capacity";
            return false;
        }
        count = static_cast<uint32_t>(per_cell * params.slot_count);
        return true;
    }

    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& slots,
        const BufferResource& ground, const BufferResource& prototypes, const BufferResource& output,
        const Parameters& params, std::string& error, const BufferResource* visible = nullptr,
        const BufferResource* indirect_arguments = nullptr) const
    {
        uint32_t count{};
        if (!candidate_count(params, count, error)) return false;
        if (params.slot_count > slots.size() / sizeof(DetailGenerateSlot) ||
            count > output.size() / sizeof(GeneratedDetail) ||
            (params.compact_cells && (!visible || !indirect_arguments ||
                visible->size() / sizeof(uint32_t) < uint64_t(params.slot_count) + 1)) ||
            (!params.compact_cells && (visible || indirect_arguments)))
        {
            error = "invalid GPU detail generation range";
            return false;
        }
        const uint64_t groups = (count - 1) / 64 + 1;
        const uint32_t groups_x = static_cast<uint32_t>(std::min<uint64_t>(groups, max_groups_x_));
        const uint64_t groups_y = groups_x ? (groups - 1) / groups_x + 1 : 0;
        if (!groups_y || groups_y > max_groups_y_)
        {
            error = "GPU detail generation exceeds dispatch capacity";
            return false;
        }
        const auto& selection = visible ? *visible : slots;
        const std::vector<VkDescriptorBufferInfo> descriptors{{slots.handle(), 0, slots.size()},
            {ground.handle(), 0, ground.size()}, {prototypes.handle(), 0, prototypes.size()},
            {output.handle(), 0, count * sizeof(GeneratedDetail)},
            {selection.handle(), 0, selection.size()}};
        if (indirect_arguments)
            return compute_.record_indirect(command, slot, descriptors, &params, *indirect_arguments, 0, error);
        return compute_.record(command, slot, descriptors, &params, groups_x, static_cast<uint32_t>(groups_y), error);
    }

private:
    StorageComputePass compute_;
    uint32_t max_groups_x_{}, max_groups_y_{};
    uint32_t max_candidates_{};
};
}
