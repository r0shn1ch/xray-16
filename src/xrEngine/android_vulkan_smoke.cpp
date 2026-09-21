#include "stdafx.h"

#if defined(XR_PLATFORM_ANDROID)

#include "android_vulkan_smoke.h"

#include <SDL.h>

#if __has_include(<vulkan/vulkan.h>)
#define XRAY_ANDROID_HAS_VULKAN_HEADERS 1
#include <vulkan/vulkan.h>
#endif

#include <dlfcn.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#ifndef SDL_WINDOW_VULKAN
#define SDL_WINDOW_VULKAN 0x10000000u
#endif

namespace AndroidVulkanSmoke
{
#if defined(XRAY_ANDROID_HAS_VULKAN_HEADERS)
namespace
{
using GetInstanceExtensions = SDL_bool (*)(SDL_Window*, unsigned int*, const char**);
using CreateSurface = SDL_bool (*)(SDL_Window*, VkInstance, VkSurfaceKHR*);

template <typename T>
T load_instance_proc(VkInstance instance, PFN_vkGetInstanceProcAddr get_proc, const char* name)
{
    return reinterpret_cast<T>(get_proc(instance, name));
}

template <typename T>
T load_device_proc(VkDevice device, PFN_vkGetDeviceProcAddr get_proc, const char* name)
{
    return reinterpret_cast<T>(get_proc(device, name));
}

bool has_extension(const std::vector<const char*>& extensions, const char* name)
{
    return std::any_of(extensions.begin(), extensions.end(), [name](const char* extension)
    {
        return std::strcmp(extension, name) == 0;
    });
}

template <typename T>
bool has_device_extension(const std::vector<T>& extensions, const char* name)
{
    return std::any_of(extensions.begin(), extensions.end(), [name](const T& extension)
    {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}
}
#endif

bool Run(std::string& reason)
{
#if !defined(XRAY_ANDROID_HAS_VULKAN_HEADERS)
    reason = "Vulkan headers are not provided by the Android toolchain";
    Msg("! [renderer-vulkan] %s", reason.c_str());
    return false;
#else
    void* vulkan_library = dlopen("libvulkan.so", RTLD_NOW | RTLD_LOCAL);
    if (!vulkan_library)
    {
        reason = "libvulkan.so is unavailable";
        Msg("! [renderer-vulkan] %s", reason.c_str());
        return false;
    }

    SDL_Window* window = nullptr;
    VkInstance instance = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    VkSemaphore acquire_semaphore = VK_NULL_HANDLE;

    PFN_vkDestroyInstance destroy_instance = nullptr;
    PFN_vkDestroySurfaceKHR destroy_surface = nullptr;
    PFN_vkDestroyDevice destroy_device = nullptr;
    PFN_vkDeviceWaitIdle device_wait_idle = nullptr;
    PFN_vkDestroySwapchainKHR destroy_swapchain = nullptr;
    PFN_vkDestroySemaphore destroy_semaphore = nullptr;

    auto cleanup = [&]
    {
        if (device && device_wait_idle)
            device_wait_idle(device);
        if (device && acquire_semaphore && destroy_semaphore)
            destroy_semaphore(device, acquire_semaphore, nullptr);
        if (device && swapchain && destroy_swapchain)
            destroy_swapchain(device, swapchain, nullptr);
        if (device && destroy_device)
            destroy_device(device, nullptr);
        if (instance && surface && destroy_surface)
            destroy_surface(instance, surface, nullptr);
        if (instance && destroy_instance)
            destroy_instance(instance, nullptr);
        if (window)
            SDL_DestroyWindow(window);
        dlclose(vulkan_library);
    };

    auto fail = [&](const std::string& message)
    {
        reason = message;
        Msg("! [renderer-vulkan] %s", reason.c_str());
        cleanup();
        return false;
    };

    const auto get_instance_proc = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
        dlsym(vulkan_library, "vkGetInstanceProcAddr"));
    if (!get_instance_proc)
        return fail("vkGetInstanceProcAddr is unavailable");

    const auto get_instance_extensions = reinterpret_cast<GetInstanceExtensions>(
        dlsym(RTLD_DEFAULT, "SDL_Vulkan_GetInstanceExtensions"));
    const auto create_surface = reinterpret_cast<CreateSurface>(
        dlsym(RTLD_DEFAULT, "SDL_Vulkan_CreateSurface"));
    if (!get_instance_extensions || !create_surface)
        return fail("SDL was built without Vulkan window support");

    window = SDL_CreateWindow("OpenXRay Vulkan surface smoke", SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED, 960, 540, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    if (!window)
    {
        reason = SDL_GetError();
        return fail(reason);
    }

    unsigned int extension_count = 0;
    if (!get_instance_extensions(window, &extension_count, nullptr) || extension_count == 0)
        return fail("SDL returned no Vulkan instance extensions");

    std::vector<const char*> extensions(extension_count);
    if (!get_instance_extensions(window, &extension_count, extensions.data()))
        return fail("SDL could not enumerate Vulkan instance extensions");
    if (!has_extension(extensions, VK_KHR_SURFACE_EXTENSION_NAME))
        return fail("SDL Vulkan extensions do not include VK_KHR_surface");

    VkApplicationInfo application_info{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application_info.pApplicationName = "OpenXRay";
    application_info.applicationVersion = VK_MAKE_VERSION(0, 9, 0);
    application_info.pEngineName = "OpenXRay";
    application_info.engineVersion = VK_MAKE_VERSION(0, 9, 0);
    application_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application_info;
    instance_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    instance_info.ppEnabledExtensionNames = extensions.data();

    const auto create_instance = reinterpret_cast<PFN_vkCreateInstance>(
        get_instance_proc(VK_NULL_HANDLE, "vkCreateInstance"));
    if (!create_instance || create_instance(&instance_info, nullptr, &instance) != VK_SUCCESS)
        return fail("vkCreateInstance failed");

    destroy_instance = load_instance_proc<PFN_vkDestroyInstance>(instance, get_instance_proc, "vkDestroyInstance");
    destroy_surface = load_instance_proc<PFN_vkDestroySurfaceKHR>(instance, get_instance_proc, "vkDestroySurfaceKHR");
    const auto enumerate_physical_devices = load_instance_proc<PFN_vkEnumeratePhysicalDevices>(
        instance, get_instance_proc, "vkEnumeratePhysicalDevices");
    const auto get_queue_families = load_instance_proc<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        instance, get_instance_proc, "vkGetPhysicalDeviceQueueFamilyProperties");
    const auto get_surface_support = load_instance_proc<PFN_vkGetPhysicalDeviceSurfaceSupportKHR>(
        instance, get_instance_proc, "vkGetPhysicalDeviceSurfaceSupportKHR");
    const auto get_surface_capabilities = load_instance_proc<PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR>(
        instance, get_instance_proc, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    const auto get_surface_formats = load_instance_proc<PFN_vkGetPhysicalDeviceSurfaceFormatsKHR>(
        instance, get_instance_proc, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    const auto get_present_modes = load_instance_proc<PFN_vkGetPhysicalDeviceSurfacePresentModesKHR>(
        instance, get_instance_proc, "vkGetPhysicalDeviceSurfacePresentModesKHR");
    const auto enumerate_device_extensions = load_instance_proc<PFN_vkEnumerateDeviceExtensionProperties>(
        instance, get_instance_proc, "vkEnumerateDeviceExtensionProperties");
    const auto get_device_proc = load_instance_proc<PFN_vkGetDeviceProcAddr>(
        instance, get_instance_proc, "vkGetDeviceProcAddr");
    if (!destroy_instance || !destroy_surface || !enumerate_physical_devices || !get_queue_families ||
        !get_surface_support || !get_surface_capabilities || !get_surface_formats || !get_present_modes ||
        !enumerate_device_extensions || !get_device_proc)
        return fail("required Vulkan instance procedures are unavailable");

    if (!create_surface(window, instance, &surface))
        return fail("SDL could not create a Vulkan window surface");

    uint32_t physical_count = 0;
    if (enumerate_physical_devices(instance, &physical_count, nullptr) != VK_SUCCESS || physical_count == 0)
        return fail("no Vulkan physical device is available");
    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    if (enumerate_physical_devices(instance, &physical_count, physical_devices.data()) != VK_SUCCESS)
        return fail("could not enumerate Vulkan physical devices");

    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t queue_family = std::numeric_limits<uint32_t>::max();
    VkPhysicalDeviceProperties physical_properties{};
    const auto get_physical_properties = load_instance_proc<PFN_vkGetPhysicalDeviceProperties>(
        instance, get_instance_proc, "vkGetPhysicalDeviceProperties");
    for (VkPhysicalDevice candidate : physical_devices)
    {
        uint32_t family_count = 0;
        get_queue_families(candidate, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        get_queue_families(candidate, &family_count, families.data());
        for (uint32_t index = 0; index < family_count; ++index)
        {
            VkBool32 presentation_supported = VK_FALSE;
            if ((families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                get_surface_support(candidate, index, surface, &presentation_supported) == VK_SUCCESS &&
                presentation_supported)
            {
                uint32_t device_extension_count = 0;
                enumerate_device_extensions(candidate, nullptr, &device_extension_count, nullptr);
                std::vector<VkExtensionProperties> device_extensions(device_extension_count);
                enumerate_device_extensions(candidate, nullptr, &device_extension_count, device_extensions.data());
                if (has_device_extension(device_extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME))
                {
                    physical_device = candidate;
                    queue_family = index;
                    if (get_physical_properties)
                        get_physical_properties(candidate, &physical_properties);
                    break;
                }
            }
        }
        if (physical_device)
            break;
    }
    if (!physical_device)
        return fail("no Vulkan graphics queue supports the Android surface and swapchain");

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;
    const char* device_extensions[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 1;
    device_info.ppEnabledExtensionNames = device_extensions;

    const auto create_device = load_instance_proc<PFN_vkCreateDevice>(instance, get_instance_proc, "vkCreateDevice");
    if (!create_device || create_device(physical_device, &device_info, nullptr, &device) != VK_SUCCESS)
        return fail("vkCreateDevice failed");

    destroy_device = load_device_proc<PFN_vkDestroyDevice>(device, get_device_proc, "vkDestroyDevice");
    device_wait_idle = load_device_proc<PFN_vkDeviceWaitIdle>(device, get_device_proc, "vkDeviceWaitIdle");
    const auto get_device_queue = load_device_proc<PFN_vkGetDeviceQueue>(device, get_device_proc, "vkGetDeviceQueue");
    const auto create_swapchain = load_device_proc<PFN_vkCreateSwapchainKHR>(device, get_device_proc, "vkCreateSwapchainKHR");
    const auto get_swapchain_images = load_device_proc<PFN_vkGetSwapchainImagesKHR>(device, get_device_proc, "vkGetSwapchainImagesKHR");
    const auto acquire_next_image = load_device_proc<PFN_vkAcquireNextImageKHR>(device, get_device_proc, "vkAcquireNextImageKHR");
    const auto queue_present = load_device_proc<PFN_vkQueuePresentKHR>(device, get_device_proc, "vkQueuePresentKHR");
    destroy_swapchain = load_device_proc<PFN_vkDestroySwapchainKHR>(device, get_device_proc, "vkDestroySwapchainKHR");
    const auto create_semaphore = load_device_proc<PFN_vkCreateSemaphore>(device, get_device_proc, "vkCreateSemaphore");
    destroy_semaphore = load_device_proc<PFN_vkDestroySemaphore>(device, get_device_proc, "vkDestroySemaphore");
    if (!destroy_device || !device_wait_idle || !get_device_queue || !create_swapchain || !get_swapchain_images ||
        !acquire_next_image || !queue_present || !destroy_swapchain || !create_semaphore || !destroy_semaphore)
        return fail("required Vulkan device procedures are unavailable");

    VkSurfaceCapabilitiesKHR capabilities{};
    if (get_surface_capabilities(physical_device, surface, &capabilities) != VK_SUCCESS)
        return fail("could not query Vulkan surface capabilities");

    uint32_t format_count = 0;
    if (get_surface_formats(physical_device, surface, &format_count, nullptr) != VK_SUCCESS || format_count == 0)
        return fail("Vulkan surface has no supported formats");
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    get_surface_formats(physical_device, surface, &format_count, formats.data());

    uint32_t present_mode_count = 0;
    if (get_present_modes(physical_device, surface, &present_mode_count, nullptr) != VK_SUCCESS || present_mode_count == 0)
        return fail("Vulkan surface has no present modes");
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    get_present_modes(physical_device, surface, &present_mode_count, present_modes.data());

    VkExtent2D extent = capabilities.currentExtent;
    if (extent.width == std::numeric_limits<uint32_t>::max())
    {
        extent.width = 960;
        extent.height = 540;
        extent.width = std::clamp(extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        extent.height = std::clamp(extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(capabilities.supportedCompositeAlpha & composite_alpha))
    {
        const VkCompositeAlphaFlagBitsKHR alternatives[] = {
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        };
        for (VkCompositeAlphaFlagBitsKHR candidate : alternatives)
            if (capabilities.supportedCompositeAlpha & candidate)
            {
                composite_alpha = candidate;
                break;
            }
    }

    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;
    if (std::find(present_modes.begin(), present_modes.end(), present_mode) == present_modes.end())
        present_mode = present_modes.front();

    uint32_t image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount && image_count > capabilities.maxImageCount)
        image_count = capabilities.maxImageCount;
    VkSwapchainCreateInfoKHR swapchain_info{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    swapchain_info.surface = surface;
    swapchain_info.minImageCount = image_count;
    swapchain_info.imageFormat = formats.front().format;
    swapchain_info.imageColorSpace = formats.front().colorSpace;
    swapchain_info.imageExtent = extent;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.preTransform = capabilities.currentTransform;
    swapchain_info.compositeAlpha = composite_alpha;
    swapchain_info.presentMode = present_mode;
    swapchain_info.clipped = VK_TRUE;
    if (create_swapchain(device, &swapchain_info, nullptr, &swapchain) != VK_SUCCESS)
        return fail("vkCreateSwapchainKHR failed");

    uint32_t swapchain_image_count = 0;
    if (get_swapchain_images(device, swapchain, &swapchain_image_count, nullptr) != VK_SUCCESS || swapchain_image_count == 0)
        return fail("Vulkan swapchain has no images");

    VkSemaphoreCreateInfo semaphore_info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    if (create_semaphore(device, &semaphore_info, nullptr, &acquire_semaphore) != VK_SUCCESS)
        return fail("vkCreateSemaphore failed");

    VkQueue queue = VK_NULL_HANDLE;
    get_device_queue(device, queue_family, 0, &queue);
    uint32_t image_index = 0;
    const VkResult acquire_result = acquire_next_image(device, swapchain, UINT64_MAX, acquire_semaphore,
        VK_NULL_HANDLE, &image_index);
    if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
        return fail("vkAcquireNextImageKHR failed");

    VkPresentInfoKHR present_info{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &acquire_semaphore;
    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain;
    present_info.pImageIndices = &image_index;
    const VkResult present_result = queue_present(queue, &present_info);
    if (present_result != VK_SUCCESS && present_result != VK_SUBOPTIMAL_KHR)
        return fail("vkQueuePresentKHR failed");
    if (device_wait_idle(device) != VK_SUCCESS)
        return fail("Vulkan queue did not become idle after present");

    Msg("[renderer-vulkan] surface/swapchain present PASS: %s, Vulkan %u.%u.%u",
        physical_properties.deviceName,
        VK_VERSION_MAJOR(physical_properties.apiVersion), VK_VERSION_MINOR(physical_properties.apiVersion),
        VK_VERSION_PATCH(physical_properties.apiVersion));
    reason = "Vulkan surface, device, swapchain and present path passed";
    cleanup();
    return true;
#endif
}
}

#endif
