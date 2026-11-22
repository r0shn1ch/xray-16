#pragma once

#include "DetailInstance.h"
#include "StorageComputePass.h"
#include <cstring>
#include <functional>

namespace xray::render::vulkan
{
struct DetailCullInput
{
    float x{}, y{}, z{}, yaw{}, scale{};
};
static_assert(sizeof(DetailCullInput) == 20);

class DetailCullPass
{
public:
    using Source = std::function<bool(const char*, std::vector<uint8_t>&, std::string&)>;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, const Source& source, std::string& error)
    {
        update_ = reinterpret_cast<PFN_vkCmdUpdateBuffer>(get(device, "vkCmdUpdateBuffer"));
        barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
        max_groups_ = limits.maxComputeWorkGroupCount[0];
        std::vector<uint8_t> binary;
        if (!update_ || !barrier_ || !source("vk/detail_cull.cs.spv", binary, error) ||
            binary.size() < 20 || binary.size() % sizeof(uint32_t))
        {
            if (error.empty())
                error = "invalid GPU detail culling shader or procedures";
            return false;
        }
        std::vector<uint32_t> words(binary.size() / sizeof(uint32_t));
        std::memcpy(words.data(), binary.data(), binary.size());
        return compute_.initialize(device, get, slots, 3, sizeof(Parameters), limits.maxStorageBufferRange,
            cache, words.data(), binary.size(), error);
    }

    // Keep each slot fenced, and use one descriptor slot per prototype dispatch.
    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& input,
        const BufferResource& output, const BufferResource& args, const float (&mvp)[16],
        uint32_t count, uint32_t index_count, float model_radius, std::string& error,
        uint32_t generated_model = UINT32_MAX,
        uint32_t capacity = 0) const
    {
        const uint64_t groups = (uint64_t(count) + 63) / 64;
        const uint32_t source_stride = generated_model == UINT32_MAX ? 5 : 8;
        const VkDeviceSize input_bytes = VkDeviceSize(count) * source_stride * sizeof(float);
        if (!capacity)
            capacity = count;
        const VkDeviceSize output_bytes = VkDeviceSize(capacity) * sizeof(DetailInstance);
        if (!command || !count || !index_count || !update_ || !barrier_ ||
            !input.handle() || !output.handle() || !args.handle() || groups > max_groups_ ||
            input_bytes > input.size() || output_bytes > output.size() ||
            args.size() < sizeof(VkDrawIndexedIndirectCommand))
        {
            error = "invalid GPU detail culling buffers";
            return false;
        }
        const VkDrawIndexedIndirectCommand initial{index_count, 0, 0, 0, 0};
        update_(command, args.handle(), 0, sizeof(initial), &initial);
        VkMemoryBarrier dependency{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        dependency.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier_(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &dependency, 0, nullptr, 0, nullptr);

        Parameters constants{};
        std::memcpy(constants.view_projection, mvp, sizeof(mvp));
        constants.placement_count = count;
        constants.instance_capacity = capacity;
        constants.radius = model_radius;
        constants.source_stride = source_stride;
        constants.model = generated_model;
        if (!compute_.record(command, slot, {{input.handle(), 0, input_bytes},
                {output.handle(), 0, output_bytes}, {args.handle(), 0, sizeof(initial)}},
                &constants, static_cast<uint32_t>(groups), 1, error))
            return false;

        dependency.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        dependency.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        barrier_(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0, 1, &dependency, 0, nullptr, 0, nullptr);
        return true;
    }

private:
    struct Parameters
    {
        float view_projection[16];
        uint32_t placement_count, instance_capacity;
        float radius;
        uint32_t source_stride, model;
    };
    StorageComputePass compute_;
    PFN_vkCmdUpdateBuffer update_{};
    PFN_vkCmdPipelineBarrier barrier_{};
    uint32_t max_groups_{};
};
}
