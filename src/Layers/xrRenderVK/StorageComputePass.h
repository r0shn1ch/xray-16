#pragma once
#include "BufferResource.h"
#include <cstdint>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// Simple storage-buffer compute dispatch with separately synchronized frame slots.
class StorageComputePass
{
public:
    struct Binding { uint32_t number; VkDescriptorType type; };
    ~StorageComputePass() { destroy(); }
    StorageComputePass() = default;
    StorageComputePass(const StorageComputePass&) = delete;
    StorageComputePass& operator=(const StorageComputePass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t slots, uint32_t bindings,
        uint32_t push_size, VkDeviceSize max_storage_range, VkPipelineCache cache,
        const uint32_t* code, size_t code_size, std::string& error);
    // Descriptor sets are per frame slot; the caller must wait for the slot
    // before updating it. Bindings and buffers use matching vector order.
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t slots,
        const std::vector<Binding>& bindings, uint32_t push_size, VkDeviceSize max_storage_range,
        VkDeviceSize max_uniform_range, VkPipelineCache cache, const uint32_t* code,
        size_t code_size, std::string& error, VkDescriptorSetLayout external_layout = VK_NULL_HANDLE);
    bool record(VkCommandBuffer command, uint32_t slot, const std::vector<VkDescriptorBufferInfo>& buffers,
        const void* constants, uint32_t groups_x, uint32_t groups_y, std::string& error,
        VkDescriptorSet external_set = VK_NULL_HANDLE) const;
    bool record_mixed(VkCommandBuffer command, uint32_t slot,
        const std::vector<VkDescriptorBufferInfo>& buffers,
        const std::vector<VkDescriptorImageInfo>& images, const void* constants,
        uint32_t groups_x, uint32_t groups_y, std::string& error,
        VkDescriptorSet external_set = VK_NULL_HANDLE) const;
    void destroy();
private:
    VkDevice device_{};
    VkDeviceSize max_storage_range_{};
    VkDeviceSize max_uniform_range_{};
    uint32_t push_size_{};
    bool external_layout_{};
    std::vector<Binding> bindings_;
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
