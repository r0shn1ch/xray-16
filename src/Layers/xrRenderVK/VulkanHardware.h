#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct HardwareDispatch
{
    PFN_vkEnumeratePhysicalDevices enumerate_physical_devices{};
    PFN_vkGetPhysicalDeviceQueueFamilyProperties get_queue_families{};
    PFN_vkGetPhysicalDeviceSurfaceSupportKHR get_surface_support{};
    PFN_vkEnumerateDeviceExtensionProperties enumerate_device_extensions{};
    PFN_vkGetPhysicalDeviceProperties get_physical_properties{};
    PFN_vkGetPhysicalDeviceFeatures get_physical_features{};
    PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties{};
    PFN_vkGetPhysicalDeviceFormatProperties get_format_properties{};
    PFN_vkGetPhysicalDeviceFeatures2KHR get_physical_features2{};
};

// Advertised support only. Device extensions and features are enabled when a
// ray-tracing renderer is created, not by the current raster path.
struct RayTracingAvailability
{
    bool acceleration_structure{};
    bool ray_query{};
    bool ray_tracing_pipeline{};
    bool vulkan11_dependencies{};
};

struct PhysicalDevice
{
    VkPhysicalDevice handle = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceMemoryProperties memory{};
    uint32_t graphics_present_family = UINT32_MAX;
    uint32_t compute_queue_family = UINT32_MAX;
    uint32_t compute_queue_index = 0;
    uint32_t transfer_queue_family = UINT32_MAX;
    uint32_t transfer_queue_index = 0;
    uint32_t frame_generation_present_family = UINT32_MAX;
    uint32_t frame_generation_present_index = UINT32_MAX;
    uint32_t frame_generation_compute_family = UINT32_MAX;
    uint32_t frame_generation_compute_index = UINT32_MAX;
    uint32_t frame_generation_acquire_family = UINT32_MAX;
    uint32_t frame_generation_acquire_index = UINT32_MAX;
    VkQueueFlags graphics_queue_flags{};
    VkQueueFlags compute_queue_flags{};
    VkQueueFlags transfer_queue_flags{};
    uint64_t local_memory_bytes{};
    int64_t score{};
    RayTracingAvailability ray_tracing{};
    bool bindless_textures{};
    bool timeline_semaphore{};
    bool create_render_pass2_extension{};
    bool depth_stencil_resolve_extension{};
    bool frame_generation_queues{};
};

struct FrameGenerationQueues
{
    VkQueue present = VK_NULL_HANDLE;
    VkQueue async_compute = VK_NULL_HANDLE;
    VkQueue image_acquire = VK_NULL_HANDLE;
    bool enabled{};
};

struct DeviceDispatch
{
    PFN_vkCreateDevice create_device{};
    PFN_vkGetDeviceProcAddr get_device_proc{};
};

struct EnabledDeviceInfo
{
    std::vector<const char*> extensions;
    bool buffer_device_address{};
    bool descriptor_binding_uniform_buffer_update_after_bind{};
    bool timeline_semaphore{};
};

bool select_physical_device(VkInstance instance, VkSurfaceKHR surface,
    const HardwareDispatch& vk, PhysicalDevice& selected, std::string& error);
bool create_logical_device(const PhysicalDevice& physical_device, const DeviceDispatch& vk,
    VkDevice& device, VkQueue& graphics_queue, std::string& error,
    bool enable_ray_tracing = false, uint32_t instance_api_version = VK_API_VERSION_1_0,
    bool enable_bindless_textures = false, EnabledDeviceInfo* enabled_info = nullptr,
    VkQueue* compute_queue = nullptr, FrameGenerationQueues* frame_generation_queues = nullptr,
    VkQueue* transfer_queue = nullptr);
// Requirements shared by the capability probe and gameplay device selection.
bool supports_game_formats(VkPhysicalDevice device, PFN_vkGetPhysicalDeviceFormatProperties get,
    const VkPhysicalDeviceProperties& properties, std::string& error);
RayTracingAvailability probe_ray_tracing_availability(VkPhysicalDevice device,
    const std::vector<VkExtensionProperties>& extensions, uint32_t api_version,
    PFN_vkGetPhysicalDeviceFeatures2KHR get_features2);
bool probe_bindless_texture_availability(VkPhysicalDevice device,
    const std::vector<VkExtensionProperties>& extensions, uint32_t api_version,
    PFN_vkGetPhysicalDeviceFeatures2KHR get_features2);
// Upper bound for material slots in the current ray-query pipeline; zero
// means required buffer or acceleration-structure descriptors are unavailable.
uint32_t ray_query_texture_capacity(const VkPhysicalDeviceProperties& properties,
    const VkPhysicalDeviceDescriptorIndexingProperties& indexing,
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& acceleration);
}
