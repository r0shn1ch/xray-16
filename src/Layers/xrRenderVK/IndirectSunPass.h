#pragma once
#include "RaySurface.h"

namespace xray::render::vulkan
{
// Each descriptor slot may be rewritten only after its frame fence completes.
class IndirectSunPass
{
public:
    ~IndirectSunPass() { destroy(); }
    IndirectSunPass() = default;
    IndirectSunPass(const IndirectSunPass&) = delete;
    IndirectSunPass& operator=(const IndirectSunPass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t slots,
        VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error);
    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& surfaces,
        const BufferResource& rays, uint32_t count, const float* direction_distance,
        std::string& error) const;
    void destroy();
private:
    VkDevice device_{};
    VkDeviceSize max_storage_range_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkUpdateDescriptorSets update_sets_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
