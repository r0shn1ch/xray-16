#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>

namespace xray::render::vulkan
{
// A level-scoped sampled-texture table. Call write() before submitting commands
// that use the slot, and keep every image and sampler alive until those commands
// complete. The owner waits for the device before destroy().
class BindlessTextureTable
{
public:
    ~BindlessTextureTable() { destroy(); }
    BindlessTextureTable() = default;
    BindlessTextureTable(const BindlessTextureTable&) = delete;
    BindlessTextureTable& operator=(const BindlessTextureTable&) = delete;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
        uint32_t capacity, std::string& error);
    bool write(uint32_t index, VkImageView view, VkSampler sampler, std::string& error);
    void destroy();

    VkDescriptorSetLayout layout() const { return layout_; }
    VkDescriptorSet set() const { return set_; }
    uint32_t capacity() const { return capacity_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    uint32_t capacity_{};
    PFN_vkDestroyDescriptorSetLayout destroy_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkUpdateDescriptorSets update_sets_{};
};
}
