#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>

// NVRHI is linked statically and uses Vulkan-Hpp's dynamic dispatcher. Keep
// its single storage definition in the Vulkan renderer module.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace xray::render::vulkan
{
void initialize_nvrhi_vulkan_dispatch(VkInstance instance,
    PFN_vkGetInstanceProcAddr get_instance_proc, VkDevice device)
{
    auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
    dispatcher.init(get_instance_proc);
    dispatcher.init(vk::Instance(instance));
    dispatcher.init(vk::Device(device));
}
}
