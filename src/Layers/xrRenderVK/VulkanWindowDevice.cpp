#include "xrCore/stdafx.h"
#include "VulkanWindowDevice.h"
#include "VulkanSettings.h"
#include "xrCore/log.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

namespace xray::render::vulkan
{
namespace
{
bool requested_hdr10()
{
    const char* override_value = SDL_getenv("XRAY_VK_HDR10");
    if (override_value && std::strcmp(override_value, "1") == 0)
        return true;
    return hdr10_enabled != 0;
}
}
#if defined(XRAY_HAS_NVRHI_VULKAN)
void initialize_nvrhi_vulkan_dispatch(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc,
    VkDevice device);
#endif

namespace
{
template <typename T> T load_instance_proc(VkInstance instance, PFN_vkGetInstanceProcAddr get, const char* name)
{
    return reinterpret_cast<T>(get(instance, name));
}
template <typename T> bool contains_name(const char* expected, const std::vector<T>& items)
{
    return std::any_of(items.begin(), items.end(), [&](const auto& item)
        { return std::strcmp(expected, item.extensionName) == 0; });
}
}

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanWindowDevice::validation_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void* user)
{
    auto* owner = static_cast<VulkanWindowDevice*>(user);
    const char* id = data && data->pMessageIdName ? data->pMessageIdName : "Vulkan";
    const char* message = data && data->pMessage ? data->pMessage : "(no message)";
    std::lock_guard<std::mutex> lock(owner->m_validation_mutex);
    const std::string key = std::string(id) + ": " + message;
    const unsigned repeats = ++owner->m_validation_repeats[key];
    // Preserve the first warning and exponentially spaced occurrence counts.
    // Per-draw driver hints otherwise bury errors and crash context in logs.
    if ((repeats & (repeats - 1)) != 0)
        return VK_FALSE;
    Msg("! [renderer-vulkan] validation occurrence=%u %s: %s", repeats, id, message);
    if (owner->m_validation_log)
    {
        std::fprintf(owner->m_validation_log, "occurrence=%u %s: %s\n", repeats, id, message);
        std::fflush(owner->m_validation_log);
    }
    return VK_FALSE;
}

bool VulkanWindowDevice::initialize(SDL_Window* window, VkExtent2D extent,
    bool allow_readback, std::string& error, bool use_depth, bool preserve_prepass_depth,
    bool postprocess, VkExtent2D scene_extent, FrameSwapchainCreate swapchain_create,
    FrameSwapchainDestroyed swapchain_destroyed, void* swapchain_data)
{
    destroy();
    if (!window || !extent.width || !extent.height)
    {
        error = "Vulkan device requires an SDL window and a nonzero extent";
        return false;
    }
    m_window = window;
    const auto fail = [&]() { destroy(); return false; };
    // SDL loads the platform Vulkan loader for SDL_WINDOW_VULKAN windows and
    // selects the matching Win32/X11/Wayland/Android surface implementation.
    m_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(SDL_Vulkan_GetVkGetInstanceProcAddr());
    if (!m_instance_proc)
    {
        error = "SDL has no Vulkan loader for this window";
        return fail();
    }
    unsigned int extension_count = 0;
    if (!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, nullptr) || !extension_count)
    {
        error = "SDL returned no Vulkan instance extensions";
        return fail();
    }
    std::vector<const char*> extensions(extension_count);
    if (!SDL_Vulkan_GetInstanceExtensions(window, &extension_count, extensions.data()))
    {
        error = "SDL could not enumerate Vulkan instance extensions";
        return fail();
    }
    extensions.resize(extension_count);
    if (std::none_of(extensions.begin(), extensions.end(), [](const char* name)
        { return std::strcmp(name, VK_KHR_SURFACE_EXTENSION_NAME) == 0; }))
    {
        error = "SDL Vulkan extensions do not include VK_KHR_surface";
        return fail();
    }
    std::vector<const char*> layers_enabled;
    VkValidationFeatureEnableEXT sync_feature = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT validation_features{VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT};
    const auto enumerate_extensions = load_instance_proc<PFN_vkEnumerateInstanceExtensionProperties>(
        VK_NULL_HANDLE, m_instance_proc, "vkEnumerateInstanceExtensionProperties");
    const auto collect_extensions = [&](const char* layer)
    {
        uint32_t count = 0;
        if (!enumerate_extensions || enumerate_extensions(layer, &count, nullptr) != VK_SUCCESS)
            return std::vector<VkExtensionProperties>{};
        std::vector<VkExtensionProperties> result(count);
        if (enumerate_extensions(layer, &count, result.data()) != VK_SUCCESS)
            result.clear();
        return result;
    };
    const auto global_extensions = collect_extensions(nullptr);
    if (contains_name(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME, global_extensions) &&
        std::none_of(extensions.begin(), extensions.end(), [](const char* name) {
            return std::strcmp(name, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME) == 0;
        }))
        extensions.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    const char* rt_requested_env = SDL_getenv("XRAY_VK_RAY_TRACING");
#if defined(XR_PLATFORM_ANDROID)
    // Mobile Vulkan requests RT features when available; devices without them
    // continue through the raster path. The environment setting can opt out.
    const bool rt_requested = !rt_requested_env || std::strcmp(rt_requested_env, "0") != 0;
#else
    const bool rt_requested = rt_requested_env && std::strcmp(rt_requested_env, "0") != 0;
#endif
    uint32_t loader_api_version = VK_API_VERSION_1_0;
    const auto enumerate_version = load_instance_proc<PFN_vkEnumerateInstanceVersion>(
        VK_NULL_HANDLE, m_instance_proc, "vkEnumerateInstanceVersion");
    if (enumerate_version && enumerate_version(&loader_api_version) != VK_SUCCESS)
        loader_api_version = VK_API_VERSION_1_0;
    const bool rt_instance = rt_requested && loader_api_version >= VK_API_VERSION_1_1;
    const bool features2_enabled = contains_name(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        global_extensions);
    if (features2_enabled && std::none_of(extensions.begin(), extensions.end(), [](const char* name)
        { return std::strcmp(name, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0; }))
        extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    // Android's debug-signed APK uses ReleaseMasterGold native code. Keep the
    // explicit opt-in available there; without a packaged layer it is inert.
    const char* requested = SDL_getenv("XRAY_VK_VALIDATION");
    if (requested && std::strcmp(requested, "0") != 0)
    {
        const auto enumerate_layers = load_instance_proc<PFN_vkEnumerateInstanceLayerProperties>(
            VK_NULL_HANDLE, m_instance_proc, "vkEnumerateInstanceLayerProperties");
        uint32_t count = 0;
        if (enumerate_layers && enumerate_layers(&count, nullptr) == VK_SUCCESS)
        {
            std::vector<VkLayerProperties> available(count);
            if (enumerate_layers(&count, available.data()) == VK_SUCCESS &&
                std::any_of(available.begin(), available.end(), [](const auto& layer)
                { return std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0; }))
            {
                layers_enabled.push_back("VK_LAYER_KHRONOS_validation");
                const auto layer_extensions = collect_extensions("VK_LAYER_KHRONOS_validation");
                const auto available = [&](const char* name)
                { return contains_name(name, global_extensions) || contains_name(name, layer_extensions); };
                if (available(VK_EXT_DEBUG_UTILS_EXTENSION_NAME))
                    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
                if (available(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME))
                {
                    extensions.push_back(VK_EXT_VALIDATION_FEATURES_EXTENSION_NAME);
                    validation_features.enabledValidationFeatureCount = 1;
                    validation_features.pEnabledValidationFeatures = &sync_feature;
                }
            }
        }
        SDL_Log("[renderer-vulkan] validation %s",
            layers_enabled.empty() ? "requested but VK_LAYER_KHRONOS_validation unavailable" : "enabled");
    }
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = app.pEngineName = "OpenXRay";
    app.apiVersion = loader_api_version >= VK_API_VERSION_1_2 ? VK_API_VERSION_1_2 :
        (loader_api_version >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1 : VK_API_VERSION_1_0);
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    instance_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instance_info.ppEnabledExtensionNames = extensions.data();
    instance_info.enabledLayerCount = static_cast<uint32_t>(layers_enabled.size());
    instance_info.ppEnabledLayerNames = layers_enabled.data();
    if (validation_features.enabledValidationFeatureCount)
        instance_info.pNext = &validation_features;
    const auto create_instance = load_instance_proc<PFN_vkCreateInstance>(VK_NULL_HANDLE,
        m_instance_proc, "vkCreateInstance");
    if (!create_instance || create_instance(&instance_info, nullptr, &m_instance) != VK_SUCCESS)
    {
        error = "vkCreateInstance failed";
        return fail();
    }
    m_destroy_instance = load_instance_proc<PFN_vkDestroyInstance>(m_instance, m_instance_proc,
        "vkDestroyInstance");
    m_destroy_surface = load_instance_proc<PFN_vkDestroySurfaceKHR>(m_instance, m_instance_proc,
        "vkDestroySurfaceKHR");
    m_device_proc = load_instance_proc<PFN_vkGetDeviceProcAddr>(m_instance, m_instance_proc,
        "vkGetDeviceProcAddr");
    if (!m_destroy_instance || !m_destroy_surface || !m_device_proc)
    {
        error = "required Vulkan instance procedures are unavailable";
        return fail();
    }
    if (!layers_enabled.empty() && std::any_of(extensions.begin(), extensions.end(),
            [](const char* name) { return std::strcmp(name, VK_EXT_DEBUG_UTILS_EXTENSION_NAME) == 0; }))
    {
        const auto create = load_instance_proc<PFN_vkCreateDebugUtilsMessengerEXT>(m_instance,
            m_instance_proc, "vkCreateDebugUtilsMessengerEXT");
        m_destroy_debug_messenger = load_instance_proc<PFN_vkDestroyDebugUtilsMessengerEXT>(m_instance,
            m_instance_proc, "vkDestroyDebugUtilsMessengerEXT");
        VkDebugUtilsMessengerCreateInfoEXT messenger{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        messenger.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messenger.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
            VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messenger.pfnUserCallback = validation_callback;
        messenger.pUserData = this;
        if (create && m_destroy_debug_messenger)
            create(m_instance, &messenger, nullptr, &m_debug_messenger);
        char* log_dir = SDL_GetPrefPath("OpenXRay", "validation");
        if (log_dir)
        {
            const std::string path = std::string(log_dir) + "vulkan-vuid.log";
            m_validation_log = std::fopen(path.c_str(), "a");
            SDL_free(log_dir);
        }
    }
    if (!SDL_Vulkan_CreateSurface(window, m_instance, &m_surface))
    {
        error = "SDL could not create a Vulkan window surface";
        return fail();
    }
    HardwareDispatch hardware{
        load_instance_proc<PFN_vkEnumeratePhysicalDevices>(m_instance, m_instance_proc, "vkEnumeratePhysicalDevices"),
        load_instance_proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(m_instance, m_instance_proc,
            "vkGetPhysicalDeviceQueueFamilyProperties"),
        load_instance_proc<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(m_instance, m_instance_proc,
            "vkGetPhysicalDeviceSurfaceSupportKHR"),
        load_instance_proc<PFN_vkEnumerateDeviceExtensionProperties>(m_instance, m_instance_proc,
            "vkEnumerateDeviceExtensionProperties"),
        load_instance_proc<PFN_vkGetPhysicalDeviceProperties>(m_instance, m_instance_proc,
            "vkGetPhysicalDeviceProperties"),
        load_instance_proc<PFN_vkGetPhysicalDeviceFeatures>(m_instance, m_instance_proc,
            "vkGetPhysicalDeviceFeatures"),
        load_instance_proc<PFN_vkGetPhysicalDeviceMemoryProperties>(m_instance, m_instance_proc,
            "vkGetPhysicalDeviceMemoryProperties"),
        load_instance_proc<PFN_vkGetPhysicalDeviceFormatProperties>(m_instance, m_instance_proc,
            "vkGetPhysicalDeviceFormatProperties"),
        app.apiVersion >= VK_API_VERSION_1_1 ?
            load_instance_proc<PFN_vkGetPhysicalDeviceFeatures2KHR>(m_instance, m_instance_proc,
            "vkGetPhysicalDeviceFeatures2") :
            (features2_enabled ? load_instance_proc<PFN_vkGetPhysicalDeviceFeatures2KHR>(m_instance,
                m_instance_proc, "vkGetPhysicalDeviceFeatures2KHR") : nullptr)
    };
    if (!select_physical_device(m_instance, m_surface, hardware, m_physical, error))
        return fail();
    const uint32_t effective_api_version = std::min(app.apiVersion, m_physical.properties.apiVersion);
    const bool core_msaa_resolve = effective_api_version >= VK_API_VERSION_1_2;
    const bool extension_msaa_resolve = effective_api_version >= VK_API_VERSION_1_1 &&
        m_physical.create_render_pass2_extension && m_physical.depth_stencil_resolve_extension;
    if (core_msaa_resolve || extension_msaa_resolve)
    {
        const char* properties2_name = app.apiVersion >= VK_API_VERSION_1_1 ?
            "vkGetPhysicalDeviceProperties2" : "vkGetPhysicalDeviceProperties2KHR";
        const auto get_properties2 = load_instance_proc<PFN_vkGetPhysicalDeviceProperties2KHR>(
            m_instance, m_instance_proc, properties2_name);
        if (get_properties2)
        {
            VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties2.pNext = &m_depth_stencil_resolve_properties;
            get_properties2(m_physical.handle, &properties2);
        }
    }
    DeviceDispatch devices{load_instance_proc<PFN_vkCreateDevice>(m_instance, m_instance_proc,
        "vkCreateDevice"), m_device_proc};
    EnabledDeviceInfo enabled_device_info;
    FrameGenerationQueues* frame_generation_queues = nullptr;
#if defined(XRAY_HAS_FSR3_ANDROID)
    frame_generation_queues = &m_frame_generation_queues;
#endif
    if (rt_instance && (m_physical.ray_tracing.ray_query ||
            m_physical.ray_tracing.ray_tracing_pipeline))
    {
        m_ray_tracing_enabled = create_logical_device(m_physical, devices, m_device,
            m_queue, error, true, app.apiVersion, m_physical.bindless_textures, &enabled_device_info,
            &m_compute_queue, frame_generation_queues, &m_transfer_queue);
        m_bindless_textures_enabled = m_ray_tracing_enabled && m_physical.bindless_textures;
        if (!m_ray_tracing_enabled && m_physical.bindless_textures)
            m_ray_tracing_enabled = create_logical_device(m_physical, devices, m_device,
                m_queue, error, true, app.apiVersion, false, &enabled_device_info, &m_compute_queue,
                frame_generation_queues, &m_transfer_queue);
        if (!m_ray_tracing_enabled)
            SDL_Log("[renderer-vulkan] ray tracing device unavailable: %s; using raster device",
                error.c_str());
    }
    if (!m_ray_tracing_enabled && m_physical.bindless_textures)
        m_bindless_textures_enabled = create_logical_device(m_physical, devices, m_device,
            m_queue, error, false, app.apiVersion, true, &enabled_device_info, &m_compute_queue,
            frame_generation_queues, &m_transfer_queue);
    if (!m_ray_tracing_enabled && !m_bindless_textures_enabled &&
        !create_logical_device(m_physical, devices, m_device, m_queue, error, false, app.apiVersion,
            false, &enabled_device_info, &m_compute_queue, frame_generation_queues,
            &m_transfer_queue)) return fail();
#if defined(XRAY_HAS_NVRHI_VULKAN)
    initialize_nvrhi_vulkan_dispatch(m_instance, m_instance_proc, m_device);
    nvrhi::vulkan::DeviceDesc nvrhi_desc{};
    nvrhi_desc.instance = m_instance;
    nvrhi_desc.physicalDevice = m_physical.handle;
    nvrhi_desc.device = m_device;
    nvrhi_desc.graphicsQueue = m_queue;
    nvrhi_desc.graphicsQueueIndex = static_cast<int>(m_physical.graphics_present_family);
    // NVRHI tracks submissions independently for each advertised queue.
    // A shared VkQueue must have one owner; the frame graph records its
    // synchronous compute work on the graphics command list.
    const bool separate_compute = m_compute_queue && m_compute_queue != m_queue;
    const bool separate_transfer = m_transfer_queue && m_transfer_queue != m_queue &&
        m_transfer_queue != m_compute_queue;
    if (separate_compute)
    {
        nvrhi_desc.computeQueue = m_compute_queue;
        nvrhi_desc.computeQueueIndex = static_cast<int>(m_physical.compute_queue_family);
    }
    if (separate_transfer)
    {
        nvrhi_desc.transferQueue = m_transfer_queue;
        nvrhi_desc.transferQueueIndex = static_cast<int>(m_physical.transfer_queue_family);
    }
    nvrhi_desc.instanceExtensions = extensions.data();
    nvrhi_desc.numInstanceExtensions = extensions.size();
    nvrhi_desc.deviceExtensions = enabled_device_info.extensions.data();
    nvrhi_desc.numDeviceExtensions = enabled_device_info.extensions.size();
    nvrhi_desc.bufferDeviceAddressSupported = enabled_device_info.buffer_device_address;
    nvrhi_desc.descriptorBindingUniformBufferUpdateAfterBind =
        enabled_device_info.descriptor_binding_uniform_buffer_update_after_bind;
    if (enabled_device_info.timeline_semaphore)
    {
        try
        {
            m_nvrhi_device = nvrhi::vulkan::createDevice(nvrhi_desc);
        }
        catch (const std::exception& exception)
        {
            SDL_Log("[renderer-vulkan] NVRHI device creation failed: %s; direct Vulkan remains active",
                exception.what());
        }
        if (m_nvrhi_device)
            SDL_Log("[renderer-vulkan] nvrhi.device.ready graphics-family=%u compute-family=%u transfer-family=%u async-compute=%d copy-queue=%d",
                m_physical.graphics_present_family, m_physical.compute_queue_family,
                m_physical.transfer_queue_family,
                int(m_compute_queue && m_physical.compute_queue_family == m_physical.graphics_present_family &&
                    m_physical.compute_queue_index > 0), int(m_transfer_queue != VK_NULL_HANDLE));
        else
            SDL_Log("[renderer-vulkan] NVRHI could not wrap the Vulkan device; direct Vulkan remains active");
    }
    else
        SDL_Log("[renderer-vulkan] NVRHI requires timeline semaphores; direct Vulkan remains active");
#endif
    SDL_Log("[renderer-vulkan] gpu.selected name='%s' vendor=0x%04x device=0x%04x driver=0x%x api=%u.%u queue=%u",
        m_physical.properties.deviceName, m_physical.properties.vendorID, m_physical.properties.deviceID,
        m_physical.properties.driverVersion, VK_VERSION_MAJOR(m_physical.properties.apiVersion),
        VK_VERSION_MINOR(m_physical.properties.apiVersion), m_physical.graphics_present_family);
    SDL_Log("[renderer-vulkan] gpu.ray_tracing advertised acceleration=%d query=%d pipeline=%d enabled=%d",
        int(m_physical.ray_tracing.acceleration_structure), int(m_physical.ray_tracing.ray_query),
        int(m_physical.ray_tracing.ray_tracing_pipeline), int(m_ray_tracing_enabled));
    if (frame_generation_queues)
        SDL_Log("[renderer-vulkan] gpu.frame_generation distinct_queues=%d present=%u:%u compute=%u:%u acquire=%u:%u",
            int(m_frame_generation_queues.enabled), m_physical.frame_generation_present_family,
            m_physical.frame_generation_present_index, m_physical.frame_generation_compute_family,
            m_physical.frame_generation_compute_index, m_physical.frame_generation_acquire_family,
            m_physical.frame_generation_acquire_index);
    SDL_Log("[renderer-vulkan] gpu.bindless_textures advertised=%d enabled=%d",
        int(m_physical.bindless_textures), int(m_bindless_textures_enabled));
    m_destroy_device = reinterpret_cast<PFN_vkDestroyDevice>(m_device_proc(m_device, "vkDestroyDevice"));
    m_image_format_properties = load_instance_proc<PFN_vkGetPhysicalDeviceImageFormatProperties>(
        m_instance, m_instance_proc, "vkGetPhysicalDeviceImageFormatProperties");
    const char* render_pass2_name = effective_api_version >= VK_API_VERSION_1_2 ?
        "vkCreateRenderPass2" : "vkCreateRenderPass2KHR";
    m_create_render_pass2 = reinterpret_cast<PFN_vkCreateRenderPass2>(m_device_proc(m_device, render_pass2_name));
    m_gbuffer_msaa_resolve_available = (core_msaa_resolve || extension_msaa_resolve) &&
        m_image_format_properties && m_create_render_pass2 &&
        m_depth_stencil_resolve_properties.supportedDepthResolveModes != VK_RESOLVE_MODE_NONE;
    SDL_Log("[renderer-vulkan] gbuffer.msaa_resolve api=%u.%u available=%d depth-modes=0x%x stencil-modes=0x%x",
        VK_VERSION_MAJOR(effective_api_version), VK_VERSION_MINOR(effective_api_version),
        int(m_gbuffer_msaa_resolve_available),
        m_depth_stencil_resolve_properties.supportedDepthResolveModes,
        m_depth_stencil_resolve_properties.supportedStencilResolveModes);
    FrameDispatch frame_dispatch;
    if (!m_destroy_device || !load_frame_dispatch(m_instance, m_instance_proc, m_device,
            m_device_proc, frame_dispatch, error))
    {
        if (error.empty()) error = "vkDestroyDevice is unavailable";
        return fail();
    }
    m_wait_idle = frame_dispatch.device_wait_idle;
    uint32_t timestamp_bits = 0;
    const auto queue_properties = load_instance_proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        m_instance, m_instance_proc, "vkGetPhysicalDeviceQueueFamilyProperties");
    if (queue_properties)
    {
        uint32_t count = 0;
        queue_properties(m_physical.handle, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        if (count) queue_properties(m_physical.handle, &count, families.data());
        if (m_physical.graphics_present_family < count)
            timestamp_bits = families[m_physical.graphics_present_family].timestampValidBits;
    }
    FrameSwapchainCreateContext swapchain_context{};
    if (swapchain_create && m_frame_generation_queues.enabled)
    {
        swapchain_context.instance = m_instance;
        swapchain_context.get_instance_proc = m_instance_proc;
        swapchain_context.physical_device = m_physical.handle;
        swapchain_context.device = m_device;
        swapchain_context.get_device_proc = m_device_proc;
        swapchain_context.game_queue = m_queue;
        swapchain_context.game_queue_family = m_physical.graphics_present_family;
        swapchain_context.game_queue_flags = m_physical.graphics_queue_flags;
        swapchain_context.present_queue = m_frame_generation_queues.present;
        swapchain_context.present_queue_family = m_physical.frame_generation_present_family;
        swapchain_context.compute_queue = m_frame_generation_queues.async_compute;
        swapchain_context.compute_queue_family = m_physical.frame_generation_compute_family;
        swapchain_context.image_acquire_queue = m_frame_generation_queues.image_acquire;
        swapchain_context.image_acquire_queue_family = m_physical.frame_generation_acquire_family;
        swapchain_context.api_version = m_physical.properties.apiVersion;
        swapchain_context.scene_extent = scene_extent.width ? scene_extent : extent;
    }
    const char* linear_scene = SDL_getenv("XRAY_VK_HDR_LINEAR_SCENE");
    if (!m_frame.initialize(m_physical.handle, m_device, m_surface, m_queue,
            m_physical.graphics_present_family, extent, frame_dispatch, error,
            allow_readback, use_depth, preserve_prepass_depth, postprocess,
            m_physical.properties.limits.timestampPeriod, timestamp_bits, scene_extent,
            m_nvrhi_device, separate_compute,
            m_physical.graphics_queue_flags, m_physical.compute_queue_flags,
            swapchain_context.physical_device ? &swapchain_context : nullptr,
            swapchain_create, swapchain_destroyed, swapchain_data,
            separate_transfer,
            m_physical.transfer_queue_flags, m_physical.compute_queue_family,
            m_physical.transfer_queue_family, linear_scene && std::strcmp(linear_scene, "1") == 0,
            requested_hdr10()))
        return fail();
    SDL_Log("[renderer-vulkan] swapchain.ready extent=%ux%u scene=%ux%u images=%zu format=%d scene_format=%d HDR10=%d timestamps=%u bits",
        m_frame.presentation_extent().width, m_frame.presentation_extent().height,
        m_frame.extent().width, m_frame.extent().height, m_frame.image_count(),
        static_cast<int>(m_frame.format()), static_cast<int>(m_frame.scene_format()), int(m_frame.hdr10_active()), timestamp_bits);
    error.clear();
    return true;
}

bool VulkanWindowDevice::recreate_frame(VkExtent2D extent, std::string& error)
{
    return recreate_frame(extent, m_frame.scene_samples(), error);
}

bool VulkanWindowDevice::recreate_frame(VkExtent2D extent, VkSampleCountFlagBits scene_samples,
    std::string& error)
{
    if (!m_device || !m_surface || !m_queue || !m_instance || !m_instance_proc || !m_device_proc ||
        !extent.width || !extent.height)
    {
        error = "Vulkan window device cannot recreate a frame for an invalid window or extent";
        return false;
    }
    if (!m_frame.request_scene_samples(scene_samples, error) ||
        !m_frame.recreate(m_physical.handle, m_surface, extent, error))
        return false;
    error.clear();
    return true;
}

bool VulkanWindowDevice::recreate_frame(VkExtent2D extent, VkExtent2D scene_extent,
    std::string& error)
{
    if (!m_frame.request_scene_extent(scene_extent, error))
        return false;
    return recreate_frame(extent, m_frame.scene_samples(), error);
}

bool VulkanWindowDevice::recreate_surface(VkExtent2D extent, std::string& error)
{
    SDL_Log("[renderer-vulkan] surface.recreate begin extent=%ux%u", extent.width, extent.height);
    if (!m_window || !m_instance || !m_device ||
        !m_physical.handle || !extent.width || !extent.height)
    {
        error = "Vulkan surface recreation requires a live window, device and drawable extent";
        return false;
    }
    // The native Android window may only be associated with one live Vulkan
    // surface/swapchain. Retire the old chain and surface before creating its
    // replacement from the resumed SDL window.
    const VkSurfaceKHR previous = m_surface;
    if (!m_frame.release_swapchain())
    {
        error = "could not release the old Vulkan swapchain before surface recreation";
        return false;
    }
    if (previous && m_destroy_surface)
        m_destroy_surface(m_instance, previous, nullptr);
    m_surface = VK_NULL_HANDLE;

    VkSurfaceKHR replacement = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(m_window, m_instance, &replacement) || !replacement)
    {
        error = "SDL could not create a replacement Vulkan surface";
        return false;
    }
    m_surface = replacement;
    const auto get_surface_support = load_instance_proc<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
        m_instance, m_instance_proc, "vkGetPhysicalDeviceSurfaceSupportKHR");
    VkBool32 supported = VK_FALSE;
    if (!get_surface_support || get_surface_support(m_physical.handle,
            m_physical.graphics_present_family, replacement, &supported) != VK_SUCCESS || !supported)
    {
        if (m_destroy_surface)
            m_destroy_surface(m_instance, replacement, nullptr);
        m_surface = VK_NULL_HANDLE;
        error = "selected Vulkan queue cannot present to the replacement surface";
        return false;
    }

    if (!m_frame.recreate(m_physical.handle, replacement, extent, error))
        return false;
    error.clear();
    return true;
}

bool VulkanWindowDevice::recreate_surface(VkExtent2D extent, VkExtent2D scene_extent,
    std::string& error)
{
    if (!m_frame.request_scene_extent(scene_extent, error))
        return false;
    return recreate_surface(extent, error);
}

void VulkanWindowDevice::destroy()
{
    if (m_device && m_wait_idle)
        m_wait_idle(m_device);
    m_frame.destroy();
    m_nvrhi_device = nullptr;
    if (m_device && m_destroy_device)
        m_destroy_device(m_device, nullptr);
    if (m_surface && m_instance && m_destroy_surface)
        m_destroy_surface(m_instance, m_surface, nullptr);
    if (m_debug_messenger && m_destroy_debug_messenger)
        m_destroy_debug_messenger(m_instance, m_debug_messenger, nullptr);
    m_debug_messenger = VK_NULL_HANDLE;
    m_destroy_debug_messenger = nullptr;
    if (m_validation_log)
        std::fclose(m_validation_log);
    m_validation_log = nullptr;
    m_validation_repeats.clear();
    if (m_instance && m_destroy_instance)
        m_destroy_instance(m_instance, nullptr);
    m_instance = VK_NULL_HANDLE;
    m_surface = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_ray_tracing_enabled = false;
    m_bindless_textures_enabled = false;
    m_gbuffer_msaa_resolve_available = false;
    m_queue = VK_NULL_HANDLE;
    m_compute_queue = VK_NULL_HANDLE;
    m_transfer_queue = VK_NULL_HANDLE;
    m_frame_generation_queues = {};
    m_physical = {};
    m_depth_stencil_resolve_properties = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    m_image_format_properties = nullptr;
    m_create_render_pass2 = nullptr;
    m_instance_proc = nullptr;
    m_device_proc = nullptr;
    m_destroy_instance = nullptr;
    m_destroy_surface = nullptr;
    m_destroy_device = nullptr;
    m_wait_idle = nullptr;
    m_window = nullptr;
}
}
