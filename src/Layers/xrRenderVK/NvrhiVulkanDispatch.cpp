#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#include <vulkan/vulkan.hpp>
#include <nvrhi/vulkan.h>

// NVRHI is linked statically and uses Vulkan-Hpp's dynamic dispatcher. Keep
// its single storage definition in the Vulkan renderer module.
VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

namespace xray::render::vulkan
{
nvrhi::vulkan::DeviceHandle create_nvrhi_vulkan_device(const nvrhi::vulkan::DeviceDesc& description,
    PFN_vkGetInstanceProcAddr get_instance_proc)
{
    auto& dispatcher = VULKAN_HPP_DEFAULT_DISPATCHER;
    dispatcher.init(get_instance_proc);
    dispatcher.init(vk::Instance(description.instance));
    dispatcher.init(vk::Device(description.device));
    return nvrhi::vulkan::createDevice(description);
}
}
