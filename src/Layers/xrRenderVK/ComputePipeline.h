#pragma once

#include <vulkan/vulkan.h>
#include <string>

namespace xray::render::vulkan
{
// The caller owns the returned pipeline and destroys it after submitted work.
bool create_compute_pipeline(VkDevice device, VkPipelineCache cache, VkPipelineLayout layout,
    VkShaderModule module, const char *entry, PFN_vkCreateComputePipelines create,
    PFN_vkDestroyPipeline destroy, VkPipeline &result, std::string &error,
    const VkSpecializationInfo *specialization = nullptr);
}
