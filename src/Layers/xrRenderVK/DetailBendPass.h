#pragma once

#include "DetailInstance.h"
#include "DetailInteractionField.h"
#include "DetailBendShaders.h"

namespace xray::render::vulkan
{
// Add current and previous field displacement to compacted instance buffers.
// Record after culling on the same queue; each prototype uses a fenced slot.
class DetailBendPass
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, std::string& error)
    {
        if (!get || limits.maxPushConstantsSize < sizeof(Parameters))
        { error = "detail bend compute limits unavailable"; return false; }
        barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
        groups_x_ = std::min(limits.maxComputeWorkGroupCount[0], 65535u);
        groups_y_ = std::min(limits.maxComputeWorkGroupCount[1], 65535u);
        if (!barrier_ || !groups_x_ || !groups_y_)
        { error = "detail bend compute procedures unavailable"; return false; }
        return pass_.initialize(device,get,slots,{{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
            {3,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}},sizeof(Parameters),limits.maxStorageBufferRange,
            limits.maxUniformBufferRange,cache,detail_bend_shaders::Compute,
            sizeof(detail_bend_shaders::Compute),error);
    }

    // arguments may be null for a CPU-prepared, directly drawn instance list.
    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& instances,
        const BufferResource* arguments, uint32_t capacity, const DetailInteractionField::Output& field,
        std::string& error) const
    {
        const uint64_t groups=(uint64_t(capacity)+63)/64;
        const uint32_t x=static_cast<uint32_t>(std::min<uint64_t>(groups,groups_x_));
        const uint64_t y=x ? (groups-1)/x+1 : 0;
        const VkDeviceSize bytes=VkDeviceSize(capacity)*sizeof(DetailInstance);
        if (!command || !barrier_ || !capacity || !y || y>groups_y_ || !instances.handle() ||
            bytes>instances.size() || (arguments && (!arguments->handle() ||
                arguments->size()<sizeof(VkDrawIndexedIndirectCommand))) ||
            !field.view || !field.previous_view || !field.sampler || !field.size ||
            !std::isfinite(field.texel_size) || field.texel_size<=0.f)
        { error="invalid detail bend inputs"; return false; }
        Parameters constants{};
        std::copy_n(field.origin,2,constants.origin);
        std::copy_n(field.previous_origin,2,constants.previous_origin);
        constants.texel_size=field.texel_size; constants.size=field.size; constants.capacity=capacity;
        constants.indirect=arguments!=nullptr; constants.previous_valid=field.previous_valid;
        for (unsigned i=0; i<2; ++i)
            if (!std::isfinite(constants.origin[i]) || !std::isfinite(constants.previous_origin[i]))
            { error="invalid detail bend window"; return false; }
        VkMemoryBarrier dependency{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        dependency.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT;
        dependency.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        barrier_(command,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT|VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&dependency,0,nullptr,0,nullptr);
        const BufferResource& count=arguments ? *arguments : instances;
        if (!pass_.record_mixed(command,slot,{{instances.handle(),0,bytes},
                {count.handle(),0,arguments ? sizeof(VkDrawIndexedIndirectCommand) : bytes}},
                {{field.sampler,field.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                 {field.sampler,field.previous_view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
                &constants,x,static_cast<uint32_t>(y),error)) return false;
        dependency.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        dependency.dstAccessMask=VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
        barrier_(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
            0,1,&dependency,0,nullptr,0,nullptr);
        return true;
    }
private:
    struct Parameters
    {
        float origin[2],previous_origin[2],texel_size;
        uint32_t size,capacity,indirect,previous_valid;
    };
    static_assert(sizeof(Parameters)==36);
    StorageComputePass pass_;
    PFN_vkCmdPipelineBarrier barrier_{};
    uint32_t groups_x_{},groups_y_{};
};
}
