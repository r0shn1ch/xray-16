#include "GameShaderResources.h"
#include "ComputePipeline.h"

#include <cstring>

namespace xray::render::vulkan
{
bool GameShaderResources::compute_pipeline(const std::string &name, VkPipelineCache cache,
    VkPipelineLayout layout, PFN_vkCreateComputePipelines create, PFN_vkDestroyPipeline destroy, VkPipeline &result,
    std::string &error, const VkSpecializationInfo *specialization)
{
    result = VK_NULL_HANDLE;
    std::vector<std::string> defines;
    GameShaderResource *compute = nullptr;
    if (!pipeline_defines(name, GameShaderStage::Compute, "main", {}, defines, error) ||
        !shader(name, GameShaderStage::Compute, "main", 0, compute, error,
            nullptr, 0, false, defines))
        return false;
    std::vector<ShaderSpecializationValue> values;
    if (specialization)
    {
        if (specialization->mapEntryCount && (!specialization->pMapEntries || !specialization->pData))
        {
            error = "invalid Vulkan compute specialization info";
            return false;
        }
        values.reserve(specialization->mapEntryCount);
        for (uint32_t index = 0; index < specialization->mapEntryCount; ++index)
        {
            const VkSpecializationMapEntry& entry = specialization->pMapEntries[index];
            if (!entry.size || entry.size > sizeof(uint64_t) || entry.offset > specialization->dataSize ||
                entry.size > specialization->dataSize - entry.offset)
            {
                error = "Vulkan compute specialization entry is outside its data block";
                return false;
            }
            uint64_t value{};
            std::memcpy(&value, static_cast<const uint8_t*>(specialization->pData) + entry.offset, entry.size);
            values.push_back({entry.constantID, static_cast<uint32_t>(entry.size), value});
        }
    }
    std::vector<uint32_t> descriptor_counts;
    if (!resolve_shader_descriptor_counts(compute->reflection, values, descriptor_counts, error))
        return false;
    return create_compute_pipeline(device_, cache, layout, compute->module.handle(), "main",
        create, destroy, result, error, specialization);
}
}
