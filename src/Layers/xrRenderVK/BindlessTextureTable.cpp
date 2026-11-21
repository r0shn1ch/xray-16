#include "BindlessTextureTable.h"

namespace xray::render::vulkan
{
namespace
{
template <typename T> T procedure(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{
    return reinterpret_cast<T>(get(device, name));
}
}

bool BindlessTextureTable::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    uint32_t capacity, std::string& error)
{
    destroy();
    if (!device || !get || !capacity)
    {
        error = "bindless texture table needs a device and nonzero capacity";
        return false;
    }
    device_ = device;
    const auto create_layout = procedure<PFN_vkCreateDescriptorSetLayout>(device, get,
        "vkCreateDescriptorSetLayout");
    destroy_layout_ = procedure<PFN_vkDestroyDescriptorSetLayout>(device, get,
        "vkDestroyDescriptorSetLayout");
    const auto create_pool = procedure<PFN_vkCreateDescriptorPool>(device, get,
        "vkCreateDescriptorPool");
    destroy_pool_ = procedure<PFN_vkDestroyDescriptorPool>(device, get,
        "vkDestroyDescriptorPool");
    const auto allocate = procedure<PFN_vkAllocateDescriptorSets>(device, get,
        "vkAllocateDescriptorSets");
    update_sets_ = procedure<PFN_vkUpdateDescriptorSets>(device, get,
        "vkUpdateDescriptorSets");
    const auto fail = [&](const char* message)
    {
        error = message;
        destroy();
        return false;
    };
    if (!create_layout || !destroy_layout_ || !create_pool || !destroy_pool_ ||
        !allocate || !update_sets_)
        return fail("bindless texture descriptor procedures unavailable");

    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = capacity;
    binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    const VkDescriptorBindingFlags flags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
    VkDescriptorSetLayoutBindingFlagsCreateInfo binding_flags{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    binding_flags.bindingCount = 1;
    binding_flags.pBindingFlags = &flags;
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.pNext = &binding_flags;
    layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("bindless texture descriptor layout creation failed");

    VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, capacity};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("bindless texture descriptor pool creation failed");

    VkDescriptorSetAllocateInfo allocate_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate_info.descriptorPool = pool_;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &layout_;
    if (allocate(device, &allocate_info, &set_) != VK_SUCCESS)
        return fail("bindless texture descriptor allocation failed");
    capacity_ = capacity;
    error.clear();
    return true;
}

bool BindlessTextureTable::write(uint32_t index, VkImageView view, VkSampler sampler,
    std::string& error, VkImageLayout layout)
{
    if (!set_ || index >= capacity_ || !view || !sampler ||
        (layout != VK_IMAGE_LAYOUT_GENERAL && layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
    {
        error = "bindless texture descriptor write is invalid";
        return false;
    }
    VkDescriptorImageInfo image{sampler, view, layout};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set_;
    write.dstBinding = 0;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    update_sets_(device_, 1, &write, 0, nullptr);
    error.clear();
    return true;
}

void BindlessTextureTable::destroy()
{
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    device_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    set_ = VK_NULL_HANDLE;
    capacity_ = 0;
    destroy_layout_ = nullptr;
    destroy_pool_ = nullptr;
    update_sets_ = nullptr;
}
}
