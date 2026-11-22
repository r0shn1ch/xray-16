#pragma once
#include "DetailInteraction.h"
#include "DetailInteractionShaders.h"
#include "StorageComputePass.h"

namespace xray::render::vulkan
{
// Images and layouts belong to the frame graph: previous/current are GENERAL,
// the heightmap is SHADER_READ_ONLY_OPTIMAL. Descriptor slots must be fenced.
class DetailInteractionPass
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, std::string& error)
    {
        if (limits.maxPushConstantsSize < sizeof(DetailInteractionParameters) ||
            limits.maxComputeWorkGroupCount[0] < 128 || limits.maxComputeWorkGroupCount[1] < 128)
        { error = "GPU detail interaction limits are insufficient"; return false; }
        return pass_.initialize(device, get, slots, {{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {1,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, {2,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
            {3,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}}, sizeof(DetailInteractionParameters),
            limits.maxStorageBufferRange, limits.maxUniformBufferRange, cache,
            detail_interaction_shaders::Compute, sizeof(detail_interaction_shaders::Compute), error);
    }
    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& entities,
        VkImageView previous, VkImageView heightmap, VkSampler height_sampler, VkImageView current,
        const DetailInteractionParameters& params, std::string& error) const
    {
        const VkDeviceSize bytes = VkDeviceSize((std::max)(1u,params.entity_count)) * sizeof(DetailInteractionEntity);
        if (!params.size || params.size > 1024 || params.entity_count > 64 || bytes > entities.size() ||
            !previous || previous == current || !current || !heightmap || !height_sampler ||
            !std::isfinite(params.texel_size) || params.texel_size <= 0.f ||
            !std::isfinite(params.height_texel_size) || params.height_texel_size <= 0.f ||
            !std::isfinite(params.delta) || params.delta < 0.f || params.delta > .1f ||
            !std::isfinite(params.contact_rate) || params.contact_rate < 0.f || params.contact_rate > 1.f ||
            !std::isfinite(params.max_velocity) || params.max_velocity < 0.f)
        { error = "invalid GPU detail interaction inputs"; return false; }
        for (float value : params.spring)
            if (!std::isfinite(value)) { error = "invalid GPU detail interaction spring"; return false; }
        for (unsigned axis=0; axis<2; ++axis)
            if (!std::isfinite(params.origin[axis]) || !std::isfinite(params.height_origin[axis]) ||
                (params.previous_valid && std::abs(int64_t(params.shift[axis])) >= params.size))
            { error = "invalid GPU detail interaction window"; return false; }
        return pass_.record_mixed(command, slot, {{entities.handle(),0,bytes}},
            {{VK_NULL_HANDLE,previous,VK_IMAGE_LAYOUT_GENERAL},
             {height_sampler,heightmap,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
             {VK_NULL_HANDLE,current,VK_IMAGE_LAYOUT_GENERAL}}, &params,
            (params.size+7)/8, (params.size+7)/8, error);
    }
private:
    StorageComputePass pass_;
};
}
