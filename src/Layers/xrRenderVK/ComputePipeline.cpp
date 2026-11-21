#include "ComputePipeline.h"

namespace xray::render::vulkan
{
bool create_compute_pipeline(VkDevice device, VkPipelineCache cache, VkPipelineLayout layout,
    VkShaderModule module, const char *entry, PFN_vkCreateComputePipelines create,
    PFN_vkDestroyPipeline destroy, VkPipeline &result, std::string &error,
    const VkSpecializationInfo *specialization)
{
    result = VK_NULL_HANDLE;
    if (!device || !layout || !module || !entry || !*entry || !create || !destroy)
    {
        error = "invalid Vulkan compute pipeline input";
        return false;
    }
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = entry;
    info.stage.pSpecializationInfo = specialization;
    info.layout = layout;
    const VkResult status = create(device, cache, 1, &info, nullptr, &result);
    if (status != VK_SUCCESS || !result)
    {
        if (result)
            destroy(device, result, nullptr);
        result = VK_NULL_HANDLE;
        error = "vkCreateComputePipelines failed: VkResult=" + std::to_string(static_cast<int>(status));
        return false;
    }
    error.clear();
    return true;
}
}
