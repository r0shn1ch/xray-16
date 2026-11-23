#pragma once
#include "DetailGeneratePass.h"
#include "DetailCellCullShaders.h"

namespace xray::render::vulkan
{
// Select resident cells and build the generation dispatch entirely on GPU.
class DetailCellCullPass
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, std::string& error)
    {
        update_ = reinterpret_cast<PFN_vkCmdUpdateBuffer>(get(device, "vkCmdUpdateBuffer"));
        fill_ = reinterpret_cast<PFN_vkCmdFillBuffer>(get(device, "vkCmdFillBuffer"));
        barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
        max_groups_x_ = std::min(limits.maxComputeWorkGroupCount[0], 65535u);
        max_groups_y_ = std::min(limits.maxComputeWorkGroupCount[1], 65535u);
        if (!update_ || !fill_ || !barrier_ || !max_groups_x_ || !max_groups_y_ ||
            limits.maxPerStageDescriptorStorageBuffers < 3 || limits.maxDescriptorSetStorageBuffers < 3 ||
            limits.maxPushConstantsSize < sizeof(Parameters))
        { if (error.empty()) error = "GPU detail cell culling unavailable"; return false; }
        return compute_.initialize(device, get, slots, 3, sizeof(Parameters), limits.maxStorageBufferRange,
            cache, detail_cell_cull_shaders::Compute, sizeof(detail_cell_cull_shaders::Compute), error);
    }

    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& cells,
        const BufferResource& visible, const BufferResource& arguments,
        const float (&mvp)[16], const float* camera_distance, float margin,
        const DetailGeneratePass::Parameters& generation, std::string& error) const
    {
        const uint64_t per_cell = uint64_t(generation.grid_size) * generation.grid_size;
        const uint64_t candidates = uint64_t(generation.slot_count) * per_cell;
        const uint64_t groups = (uint64_t(generation.slot_count) + 63) / 64;
        const uint32_t x = static_cast<uint32_t>(std::min<uint64_t>(groups, max_groups_x_));
        const uint64_t y = x ? (groups + x - 1) / x : 0;
        if (!command || !per_cell || per_cell > UINT32_MAX || !candidates || candidates > UINT32_MAX ||
            !y || y > max_groups_y_ || !camera_distance || !std::isfinite(margin) || margin < 0.f ||
            !cells.handle() || uint64_t(generation.slot_count) * sizeof(DetailGenerateSlot) > cells.size() ||
            !visible.handle() || (uint64_t(generation.slot_count) + 1) * 4 > visible.size() ||
            !arguments.handle() || arguments.size() < sizeof(VkDispatchIndirectCommand))
        { error = "invalid GPU detail cell culling range"; return false; }
        for (unsigned i = 0; i < 4; ++i)
            if (!std::isfinite(camera_distance[i]) || (i == 3 && camera_distance[i] < 0.f))
            { error = "invalid GPU detail cell camera"; return false; }
        Parameters params{};
        std::copy_n(mvp, 16, params.mvp);
        std::copy_n(camera_distance, 4, params.camera_distance);
        params.slot_count = generation.slot_count;
        params.per_cell = static_cast<uint32_t>(per_cell);
        params.max_groups_x = max_groups_x_;
        params.margin = margin;
        const VkDispatchIndirectCommand initial{0, 1, 1};
        fill_(command, visible.handle(), 0, 4, 0);
        update_(command, arguments.handle(), 0, sizeof(initial), &initial);
        VkMemoryBarrier dependency{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        dependency.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier_(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &dependency, 0, nullptr, 0, nullptr);
        if (!compute_.record(command, slot, {{cells.handle(), 0, cells.size()},
                {visible.handle(), 0, visible.size()}, {arguments.handle(), 0, sizeof(initial)}}, &params, x,
                static_cast<uint32_t>(y), error)) return false;
        VkBufferMemoryBarrier indirect{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        indirect.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        indirect.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        indirect.srcQueueFamilyIndex = indirect.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        indirect.buffer = arguments.handle(); indirect.offset = 0; indirect.size = sizeof(initial);
        barrier_(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            0, 0, nullptr, 1, &indirect, 0, nullptr);
        error.clear(); return true;
    }

private:
    struct Parameters
    {
        float mvp[16], camera_distance[4];
        uint32_t slot_count, per_cell, max_groups_x;
        float margin;
    };
    static_assert(sizeof(Parameters) == 96);
    StorageComputePass compute_;
    PFN_vkCmdUpdateBuffer update_{};
    PFN_vkCmdFillBuffer fill_{};
    PFN_vkCmdPipelineBarrier barrier_{};
    uint32_t max_groups_x_{}, max_groups_y_{};
};
}
