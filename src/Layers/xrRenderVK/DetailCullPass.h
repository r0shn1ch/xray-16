#pragma once

#include "DetailInstance.h"
#include "StorageComputePass.h"
#include <algorithm>
#include <cmath>
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
        max_groups_x_ = std::min(limits.maxComputeWorkGroupCount[0], 65535u);
        max_groups_y_ = std::min(limits.maxComputeWorkGroupCount[1], 65535u);
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
        const uint32_t groups_x = static_cast<uint32_t>(std::min<uint64_t>(groups, max_groups_x_));
        const uint64_t groups_y = groups_x ? (groups - 1) / groups_x + 1 : 0;
        const uint32_t source_stride = generated_model == UINT32_MAX ? 5 : 8;
        const VkDeviceSize input_bytes = VkDeviceSize(count) * source_stride * sizeof(float);
        if (!capacity)
            capacity = count;
        const VkDeviceSize output_bytes = VkDeviceSize(capacity) * sizeof(DetailInstance);
        if (!command || !count || !index_count || !update_ || !barrier_ ||
            !input.handle() || !output.handle() || !args.handle() || !groups_y || groups_y > max_groups_y_ ||
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
        // The projection matrix is column-major. Normalize the six planes
        // once per dispatch so the shader compares distances in world units.
        for (unsigned component = 0; component < 4; ++component)
        {
            const float* column = mvp + component * 4;
            constants.planes[0][component] = column[3] + column[0];
            constants.planes[1][component] = column[3] - column[0];
            constants.planes[2][component] = column[3] + column[1];
            constants.planes[3][component] = column[3] - column[1];
            constants.planes[4][component] = column[2];
            constants.planes[5][component] = column[3] - column[2];
        }
        for (auto& plane : constants.planes)
        {
            const float length = std::hypot(plane[0], plane[1], plane[2]);
            if (length > 0.f && std::isfinite(length))
                for (float& component : plane)
                    component /= length;
            else
                plane[3] = plane[3] < 0.f ? -INFINITY : INFINITY;
        }
        constants.placement_count = count;
        constants.instance_capacity = capacity;
        constants.radius = model_radius;
        constants.source_stride = source_stride;
        constants.model = generated_model;
        if (!compute_.record(command, slot, {{input.handle(), 0, input_bytes},
                {output.handle(), 0, output_bytes}, {args.handle(), 0, sizeof(initial)}},
                &constants, groups_x, static_cast<uint32_t>(groups_y), error))
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
        float planes[6][4];
        uint32_t placement_count, instance_capacity;
        float radius;
        uint32_t source_stride, model;
    };
    static_assert(sizeof(Parameters) == 116);
    StorageComputePass compute_;
    PFN_vkCmdUpdateBuffer update_{};
    PFN_vkCmdPipelineBarrier barrier_{};
    uint32_t max_groups_x_{}, max_groups_y_{};
};
}
