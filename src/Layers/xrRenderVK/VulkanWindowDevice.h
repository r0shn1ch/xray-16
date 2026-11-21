#pragma once

#include "FrameContext.h"
#include "VulkanHardware.h"
#include <nvrhi/vulkan.h>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

struct SDL_Window;

namespace xray::render::vulkan
{
// Owns an SDL window's Vulkan instance, surface, logical device and swapchain.
// The caller owns the window and all resources built from device() and must
// destroy those resources (after waiting for idle) before destroy().
class VulkanWindowDevice
{
public:
    VulkanWindowDevice() = default;
    ~VulkanWindowDevice() { destroy(); }
    VulkanWindowDevice(const VulkanWindowDevice&) = delete;
    VulkanWindowDevice& operator=(const VulkanWindowDevice&) = delete;

    bool initialize(SDL_Window* window, VkExtent2D extent, bool allow_readback,
        std::string& error, bool use_depth = false, bool preserve_prepass_depth = false,
        bool postprocess = false, VkExtent2D scene_extent = {},
        FrameSwapchainCreate swapchain_create = nullptr,
        FrameSwapchainDestroyed swapchain_destroyed = nullptr, void* swapchain_data = nullptr);
    bool recreate_frame(VkExtent2D extent, std::string& error);
    bool recreate_frame(VkExtent2D extent, VkSampleCountFlagBits scene_samples, std::string& error);
    bool recreate_frame(VkExtent2D extent, VkExtent2D scene_extent, std::string& error);
    bool recreate_surface(VkExtent2D extent, std::string& error);
    bool recreate_surface(VkExtent2D extent, VkExtent2D scene_extent, std::string& error);
    void destroy();
    VkDevice device() const { return m_device; }
    const nvrhi::vulkan::DeviceHandle& nvrhi_device() const { return m_nvrhi_device; }
    VkInstance instance() const { return m_instance; }
    VkQueue queue() const { return m_queue; }
    VkQueue compute_queue() const { return m_compute_queue; }
    VkQueue transfer_queue() const { return m_transfer_queue; }
    const FrameGenerationQueues& frame_generation_queues() const { return m_frame_generation_queues; }
    bool ray_tracing_enabled() const { return m_ray_tracing_enabled; }
    bool bindless_textures_enabled() const { return m_bindless_textures_enabled; }
    bool gbuffer_msaa_resolve_available() const { return m_gbuffer_msaa_resolve_available; }
    PFN_vkGetPhysicalDeviceImageFormatProperties image_format_properties() const { return m_image_format_properties; }
    PFN_vkCreateRenderPass2 create_render_pass2() const { return m_create_render_pass2; }
    const VkPhysicalDeviceDepthStencilResolveProperties& depth_stencil_resolve_properties() const
    { return m_depth_stencil_resolve_properties; }
    const PhysicalDevice& physical() const { return m_physical; }
    FrameContext& frame() { return m_frame; }
    const FrameContext& frame() const { return m_frame; }
    PFN_vkGetInstanceProcAddr instance_proc() const { return m_instance_proc; }
    PFN_vkGetDeviceProcAddr device_proc() const { return m_device_proc; }

private:
    static VKAPI_ATTR VkBool32 VKAPI_CALL validation_callback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
        VkDebugUtilsMessageTypeFlagsEXT type, const VkDebugUtilsMessengerCallbackDataEXT* data, void* user);
    VkInstance m_instance = VK_NULL_HANDLE;
    SDL_Window* m_window{};
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    nvrhi::vulkan::DeviceHandle m_nvrhi_device;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkQueue m_compute_queue = VK_NULL_HANDLE;
    VkQueue m_transfer_queue = VK_NULL_HANDLE;
    FrameGenerationQueues m_frame_generation_queues{};
    bool m_ray_tracing_enabled = false;
    bool m_bindless_textures_enabled = false;
    bool m_gbuffer_msaa_resolve_available = false;
    PhysicalDevice m_physical{};
    VkPhysicalDeviceDepthStencilResolveProperties m_depth_stencil_resolve_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    PFN_vkGetPhysicalDeviceImageFormatProperties m_image_format_properties{};
    PFN_vkCreateRenderPass2 m_create_render_pass2{};
    FrameContext m_frame;
    PFN_vkGetInstanceProcAddr m_instance_proc{};
    PFN_vkGetDeviceProcAddr m_device_proc{};
    PFN_vkDestroyInstance m_destroy_instance{};
    PFN_vkDestroySurfaceKHR m_destroy_surface{};
    PFN_vkDestroyDevice m_destroy_device{};
    PFN_vkDeviceWaitIdle m_wait_idle{};
    VkDebugUtilsMessengerEXT m_debug_messenger = VK_NULL_HANDLE;
    PFN_vkDestroyDebugUtilsMessengerEXT m_destroy_debug_messenger{};
    std::FILE* m_validation_log{};
    std::mutex m_validation_mutex;
    std::unordered_map<std::string, unsigned> m_validation_repeats;
};
}
