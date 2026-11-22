#include "FrameContext.h"
#include "FrameImageGraph.h"
#include "FrameSubmissionGraph.h"
#if defined(XR_PLATFORM_ANDROID)
#include "xrEngine/x_ray.h"
#endif

#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace xray::render::vulkan
{
namespace
{
VkCommandBuffer nvrhi_command_buffer(nvrhi::ICommandList* command_list)
{
    if (!command_list) return VK_NULL_HANDLE;
    const nvrhi::Object native = command_list->getNativeObject(nvrhi::ObjectTypes::VK_CommandBuffer);
    return static_cast<VkCommandBuffer>(native.pointer);
}

struct FrameSubmissionGuard
{
    bool& submission_failed;
    nvrhi::ICommandList* command_list{};
    bool command_list_open{};
    nvrhi::ICommandList* compute_command_list{};
    bool compute_command_list_open{};
    std::vector<nvrhi::ICommandList*> generated_command_lists_open;
    bool submitted{};

    void track_generated(nvrhi::ICommandList* list)
    {
        if (list)
            generated_command_lists_open.push_back(list);
    }

    void mark_generated_closed(nvrhi::ICommandList* list)
    {
        generated_command_lists_open.erase(std::remove(generated_command_lists_open.begin(),
            generated_command_lists_open.end(), list), generated_command_lists_open.end());
    }

    ~FrameSubmissionGuard()
    {
        if (submitted) return;
        if (command_list_open && command_list)
        {
            try { command_list->close(); }
            catch (...) {}
        }
        if (compute_command_list_open && compute_command_list)
        {
            try { compute_command_list->close(); }
            catch (...) {}
        }
        for (nvrhi::ICommandList* list : generated_command_lists_open)
        {
            try { list->close(); }
            catch (...) {}
        }
        submission_failed = true;
    }
};

void crash_stage(const char* stage)
{
#if defined(XR_PLATFORM_ANDROID)
    android_set_load_context(stage);
    android_set_vulkan_stage(stage);
#else
    (void)stage;
#endif
}
}
bool mark_surface_lost(VkResult result, bool& surface_lost, FrameStatus& status)
{
    if (result != VK_ERROR_SURFACE_LOST_KHR)
        return false;
    surface_lost = true;
    status = FrameStatus::RecreateRequired;
    return true;
}

namespace
{
bool complete(const FrameDispatch& vk)
{
    return vk.get_surface_capabilities && vk.get_surface_formats && vk.create_swapchain && vk.destroy_swapchain &&
        vk.get_swapchain_images && vk.acquire_next_image && vk.queue_present && vk.create_image_view &&
        vk.destroy_image_view && vk.get_format_properties && vk.get_memory_properties &&
        vk.create_image && vk.destroy_image &&
        vk.get_image_memory_requirements && vk.allocate_memory && vk.free_memory && vk.bind_image_memory &&
        vk.create_render_pass && vk.destroy_render_pass && vk.create_framebuffer &&
        vk.destroy_framebuffer && vk.create_command_pool && vk.destroy_command_pool && vk.allocate_command_buffers &&
        vk.free_command_buffers &&
        vk.reset_command_buffer && vk.begin_command_buffer && vk.cmd_begin_render_pass && vk.cmd_end_render_pass &&
        vk.cmd_clear_attachments &&
        vk.cmd_copy_image_to_buffer && vk.cmd_copy_image && vk.cmd_pipeline_barrier &&
        vk.end_command_buffer && vk.create_semaphore && vk.destroy_semaphore && vk.create_fence && vk.destroy_fence &&
        vk.wait_for_fences && vk.reset_fences && vk.queue_submit && vk.device_wait_idle;
}

template <typename T, typename Query>
bool enumerate(Query query, std::vector<T>& values)
{
    for (unsigned attempt = 0; attempt != 3; ++attempt)
    {
        uint32_t count = 0;
        if (query(&count, nullptr) != VK_SUCCESS || count == 0)
            return false;
        values.resize(count);
        const VkResult result = query(&count, values.data());
        if (result == VK_SUCCESS)
        {
            values.resize(count);
            return !values.empty();
        }
        if (result != VK_INCOMPLETE)
            return false;
    }
    return false;
}

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
}

bool load_frame_dispatch(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc,
    VkDevice device, PFN_vkGetDeviceProcAddr get_device_proc, FrameDispatch& dispatch, std::string& error)
{
    dispatch = {};
    if (!instance || !get_instance_proc || !device || !get_device_proc)
    {
        error = "Vulkan frame dispatch requires valid instance and device handles";
        return false;
    }

#define XRAY_LOAD_INSTANCE(member, name) \
    dispatch.member = load_instance_proc<decltype(dispatch.member)>(instance, get_instance_proc, name)
#define XRAY_LOAD_DEVICE(member, name) \
    dispatch.member = load_device_proc<decltype(dispatch.member)>(device, get_device_proc, name)
    XRAY_LOAD_INSTANCE(get_surface_capabilities, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    XRAY_LOAD_INSTANCE(get_surface_formats, "vkGetPhysicalDeviceSurfaceFormatsKHR");
    XRAY_LOAD_INSTANCE(get_format_properties, "vkGetPhysicalDeviceFormatProperties");
    XRAY_LOAD_INSTANCE(get_memory_properties, "vkGetPhysicalDeviceMemoryProperties");
    XRAY_LOAD_DEVICE(create_swapchain, "vkCreateSwapchainKHR");
    XRAY_LOAD_DEVICE(destroy_swapchain, "vkDestroySwapchainKHR");
    XRAY_LOAD_DEVICE(get_swapchain_images, "vkGetSwapchainImagesKHR");
    XRAY_LOAD_DEVICE(acquire_next_image, "vkAcquireNextImageKHR");
    XRAY_LOAD_DEVICE(queue_present, "vkQueuePresentKHR");
    XRAY_LOAD_DEVICE(set_hdr_metadata, "vkSetHdrMetadataEXT");
    XRAY_LOAD_DEVICE(create_image_view, "vkCreateImageView");
    XRAY_LOAD_DEVICE(destroy_image_view, "vkDestroyImageView");
    XRAY_LOAD_DEVICE(create_image, "vkCreateImage");
    XRAY_LOAD_DEVICE(destroy_image, "vkDestroyImage");
    XRAY_LOAD_DEVICE(get_image_memory_requirements, "vkGetImageMemoryRequirements");
    XRAY_LOAD_DEVICE(allocate_memory, "vkAllocateMemory");
    XRAY_LOAD_DEVICE(free_memory, "vkFreeMemory");
    XRAY_LOAD_DEVICE(bind_image_memory, "vkBindImageMemory");
    XRAY_LOAD_DEVICE(create_render_pass, "vkCreateRenderPass");
    XRAY_LOAD_DEVICE(destroy_render_pass, "vkDestroyRenderPass");
    XRAY_LOAD_DEVICE(create_framebuffer, "vkCreateFramebuffer");
    XRAY_LOAD_DEVICE(destroy_framebuffer, "vkDestroyFramebuffer");
    XRAY_LOAD_DEVICE(create_command_pool, "vkCreateCommandPool");
    XRAY_LOAD_DEVICE(destroy_command_pool, "vkDestroyCommandPool");
    XRAY_LOAD_DEVICE(allocate_command_buffers, "vkAllocateCommandBuffers");
    XRAY_LOAD_DEVICE(free_command_buffers, "vkFreeCommandBuffers");
    XRAY_LOAD_DEVICE(reset_command_buffer, "vkResetCommandBuffer");
    XRAY_LOAD_DEVICE(begin_command_buffer, "vkBeginCommandBuffer");
    XRAY_LOAD_DEVICE(cmd_begin_render_pass, "vkCmdBeginRenderPass");
    XRAY_LOAD_DEVICE(cmd_end_render_pass, "vkCmdEndRenderPass");
    XRAY_LOAD_DEVICE(cmd_clear_attachments, "vkCmdClearAttachments");
    XRAY_LOAD_DEVICE(cmd_copy_image_to_buffer, "vkCmdCopyImageToBuffer");
    XRAY_LOAD_DEVICE(cmd_copy_image, "vkCmdCopyImage");
    XRAY_LOAD_DEVICE(cmd_blit_image, "vkCmdBlitImage");
    XRAY_LOAD_DEVICE(cmd_pipeline_barrier, "vkCmdPipelineBarrier");
    XRAY_LOAD_DEVICE(end_command_buffer, "vkEndCommandBuffer");
    XRAY_LOAD_DEVICE(create_semaphore, "vkCreateSemaphore");
    XRAY_LOAD_DEVICE(destroy_semaphore, "vkDestroySemaphore");
    XRAY_LOAD_DEVICE(create_fence, "vkCreateFence");
    XRAY_LOAD_DEVICE(destroy_fence, "vkDestroyFence");
    XRAY_LOAD_DEVICE(wait_for_fences, "vkWaitForFences");
    XRAY_LOAD_DEVICE(reset_fences, "vkResetFences");
    XRAY_LOAD_DEVICE(queue_submit, "vkQueueSubmit");
    XRAY_LOAD_DEVICE(device_wait_idle, "vkDeviceWaitIdle");
    XRAY_LOAD_DEVICE(create_query_pool, "vkCreateQueryPool");
    XRAY_LOAD_DEVICE(destroy_query_pool, "vkDestroyQueryPool");
    XRAY_LOAD_DEVICE(cmd_reset_query_pool, "vkCmdResetQueryPool");
    XRAY_LOAD_DEVICE(cmd_write_timestamp, "vkCmdWriteTimestamp");
    XRAY_LOAD_DEVICE(get_query_pool_results, "vkGetQueryPoolResults");
#undef XRAY_LOAD_DEVICE
#undef XRAY_LOAD_INSTANCE

    if (!complete(dispatch))
    {
        dispatch = {};
        error = "required Vulkan frame procedures are unavailable";
        return false;
    }
    error.clear();
    return true;
}

bool FrameContext::recreate(VkPhysicalDevice physical_device, VkSurfaceKHR surface,
    VkExtent2D requested_extent, std::string& error)
{
    if (!m_device || !m_queue || !m_command_pool || !physical_device || !surface)
    {
        error = "Vulkan frame context cannot recreate an uninitialized swapchain";
        return false;
    }
    if (!wait_idle())
    {
        error = "could not wait for Vulkan device before swapchain recreation";
        return false;
    }

    const VkFormat previous_format = m_format;
    if (!m_commands.empty())
        m_vk.free_command_buffers(m_device, m_command_pool,
            static_cast<uint32_t>(m_commands.size()), m_commands.data());
    m_commands.clear();
    destroy_swapchain_resources();
    m_vk = m_native_vk;
    m_device_lost = false;
    m_submission_failed = false;

    if (!create_swapchain(physical_device, surface, requested_extent, error))
        goto failed;
    if (m_format != previous_format)
    {
        error = "Vulkan surface format changed; existing renderer pipelines must be rebuilt";
        goto failed;
    }
    if (!create_render_targets(error) || !create_commands(error) || !create_sync(error))
        goto failed;
    m_current_frame = 0;
    m_surface_lost = false;
    error.clear();
    return true;

failed:
    destroy_swapchain_resources();
    m_format = previous_format;
    return false;
}

bool FrameContext::release_swapchain()
{
    if (!m_device || !m_command_pool || !wait_idle())
        return false;
    if (!m_commands.empty())
        m_vk.free_command_buffers(m_device, m_command_pool,
            static_cast<uint32_t>(m_commands.size()), m_commands.data());
    m_commands.clear();
    destroy_swapchain_resources();
    return true;
}

bool FrameContext::wait_idle()
{
    if (!m_device || !m_vk.device_wait_idle)
        return false;
    const VkResult result = m_vk.device_wait_idle(m_device);
    m_device_lost = result == VK_ERROR_DEVICE_LOST;
    if (result == VK_SUCCESS && m_nvrhi_device)
    {
        try { m_nvrhi_device->runGarbageCollection(); }
        catch (...) { return false; }
    }
    return result == VK_SUCCESS;
}

FrameContext::~FrameContext()
{
    destroy();
}

bool FrameContext::initialize(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface,
    VkQueue queue, uint32_t queue_family, VkExtent2D requested_extent,
    const FrameDispatch& dispatch, std::string& error, bool allow_readback, bool use_depth,
    bool preserve_prepass_depth, bool postprocess,
    float timestamp_period_ns, uint32_t timestamp_valid_bits, VkExtent2D scene_extent,
    nvrhi::vulkan::DeviceHandle nvrhi_device, bool async_compute_queue,
    VkQueueFlags graphics_queue_flags, VkQueueFlags compute_queue_flags,
    const FrameSwapchainCreateContext* swapchain_context, FrameSwapchainCreate swapchain_create,
    FrameSwapchainDestroyed swapchain_destroyed, void* swapchain_data,
    bool async_copy_queue, VkQueueFlags copy_queue_flags,
    uint32_t compute_queue_family, uint32_t copy_queue_family, bool linear_scene, bool hdr10)
{
    destroy();
    if (!physical_device || !device || !surface || !queue || queue_family == UINT32_MAX)
    {
        error = "Vulkan frame context requires valid device, surface and queue handles";
        return false;
    }
    if (!complete(dispatch))
    {
        error = "Vulkan frame context dispatch is incomplete";
        return false;
    }

    m_device = device;
    m_nvrhi_device = std::move(nvrhi_device);
    m_async_compute_available = async_compute_queue && bool(m_nvrhi_device);
    m_async_copy_available = async_copy_queue && bool(m_nvrhi_device);
    m_graphics_queue_flags = graphics_queue_flags;
    m_compute_queue_flags = compute_queue_flags;
    m_copy_queue_flags = copy_queue_flags;
    m_queue = queue;
    m_queue_family = queue_family;
    m_compute_queue_family = compute_queue_family == VK_QUEUE_FAMILY_IGNORED ?
        queue_family : compute_queue_family;
    m_copy_queue_family = copy_queue_family == VK_QUEUE_FAMILY_IGNORED ?
        queue_family : copy_queue_family;
    m_frame_image_sharing = frame_image_sharing(m_queue_family, m_compute_queue_family);
    m_vk = dispatch;
    m_native_vk = dispatch;
    m_swapchain_context = swapchain_context ? *swapchain_context : FrameSwapchainCreateContext{};
    m_swapchain_create = swapchain_create;
    m_swapchain_destroyed = swapchain_destroyed;
    m_swapchain_data = swapchain_data;
    m_frame_interpolation_swapchain = false;
    m_device_lost = false;
    m_submission_failed = false;
    m_surface_lost = false;
    m_allow_readback = allow_readback;
    m_preserve_prepass_depth = preserve_prepass_depth;
    m_postprocess = postprocess;
    m_linear_scene = linear_scene;
    m_hdr10_requested = hdr10;
    if ((scene_extent.width || scene_extent.height) &&
        (!scene_extent.width || !scene_extent.height || !postprocess))
    {
        error = "independent scene extent requires nonzero dimensions and postprocessing";
        destroy();
        return false;
    }
    m_requested_scene_extent = scene_extent;
    if (timestamp_period_ns > 0.f && timestamp_valid_bits &&
        m_vk.create_query_pool && m_vk.destroy_query_pool && m_vk.cmd_reset_query_pool &&
        m_vk.cmd_write_timestamp && m_vk.get_query_pool_results)
    {
        VkQueryPoolCreateInfo query_info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_info.queryCount = 2 * FramesInFlight;
        if (m_vk.create_query_pool(m_device, &query_info, nullptr, &m_timestamp_pool) == VK_SUCCESS)
        {
            m_timestamp_period_ns = timestamp_period_ns;
            m_timestamp_valid_bits = timestamp_valid_bits;
        }
    }
    if (postprocess && !allow_readback)
    {
        error = "Vulkan postprocess requires swapchain transfer-source support";
        destroy();
        return false;
    }
    if (preserve_prepass_depth && !use_depth)
    {
        error = "prepass depth requires a depth attachment";
        destroy();
        return false;
    }
    if (use_depth || postprocess)
        m_vk.get_memory_properties(physical_device, &m_memory_properties);
    if (postprocess && !m_frame_image_pool.initialize(device, m_memory_properties,
            {m_vk.create_image, m_vk.destroy_image, m_vk.get_image_memory_requirements,
             m_vk.allocate_memory, m_vk.free_memory, m_vk.bind_image_memory},
            m_nvrhi_device))
    {
        error = "Vulkan frame image pool procedures are unavailable";
        destroy();
        return false;
    }
    if (use_depth)
    {
        const VkFormat candidates[]{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D16_UNORM};
        for (VkFormat candidate : candidates)
        {
            VkFormatProperties properties{};
            m_vk.get_format_properties(physical_device, candidate, &properties);
            const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if ((properties.optimalTilingFeatures & required) == required)
            {
                m_depth_format = candidate;
                break;
            }
        }
        if (m_depth_format == VK_FORMAT_UNDEFINED)
        {
            error = "Vulkan device has no depth format that supports attachment and sampling";
            destroy();
            return false;
        }
    }

    if (!create_swapchain(physical_device, surface, requested_extent, error) ||
        !create_render_targets(error) || !create_commands(error) || !create_sync(error))
    {
        destroy();
        return false;
    }
    error.clear();
    return true;
}

bool FrameContext::create_swapchain(VkPhysicalDevice physical_device, VkSurfaceKHR surface,
    VkExtent2D requested_extent, std::string& error)
{
    m_physical_device = physical_device;
    VkSurfaceCapabilitiesKHR capabilities{};
    const auto record_surface_error = [&](VkResult result)
    {
        m_device_lost = result == VK_ERROR_DEVICE_LOST;
        m_surface_lost = result == VK_ERROR_SURFACE_LOST_KHR;
    };
    const VkResult capabilities_result = m_vk.get_surface_capabilities(physical_device, surface, &capabilities);
    if (capabilities_result != VK_SUCCESS)
    {
        record_surface_error(capabilities_result);
        error = "could not query Vulkan surface capabilities";
        return false;
    }
    if (!(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT))
    {
        error = "Vulkan surface does not support color attachments";
        return false;
    }
    std::vector<VkSurfaceFormatKHR> formats;
    VkResult formats_result = VK_SUCCESS;
    if (!enumerate<VkSurfaceFormatKHR>([&](uint32_t* count, VkSurfaceFormatKHR* values)
        { return formats_result = m_vk.get_surface_formats(physical_device, surface, count, values); }, formats))
    {
        record_surface_error(formats_result);
        error = "Vulkan surface has no supported formats";
        return false;
    }

    const auto hdr10 = std::find_if(formats.begin(), formats.end(), [](const VkSurfaceFormatKHR& format) {
        return format.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT &&
            (format.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || format.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    });
    VkFormatProperties linear_format{};
    if (hdr10 != formats.end() && m_postprocess && m_requested_scene_extent.width)
        m_vk.get_format_properties(m_physical_device, VK_FORMAT_R16G16B16A16_SFLOAT, &linear_format);
    constexpr VkFormatFeatureFlags linear_required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    const bool hdr10_available = hdr10 != formats.end() &&
        (linear_format.optimalTilingFeatures & linear_required) == linear_required;
    const bool use_hdr10 = m_hdr10_requested && hdr10_available;
    if (!use_hdr10 && m_allow_readback && !(capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
    {
        error = "Vulkan surface does not support transfer-source swapchain images";
        return false;
    }
    VkSurfaceFormatKHR surface_format{};
    if (use_hdr10)
        surface_format = *hdr10;
    else if (formats.size() == 1 && formats.front().format == VK_FORMAT_UNDEFINED)
    {
        surface_format = {m_format != VK_FORMAT_UNDEFINED ? m_format : VK_FORMAT_B8G8R8A8_UNORM,
            formats.front().colorSpace};
    }
    else
    {
        const auto score = [&](const VkSurfaceFormatKHR& format) {
            if (format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
                return 0;
            const bool rgba8 = format.format == VK_FORMAT_R8G8B8A8_UNORM || format.format == VK_FORMAT_B8G8R8A8_UNORM;
            if (m_allow_readback && !rgba8)
                return 0;
            if (format.format == m_format && format.colorSpace == m_color_space)
                return 4;
            return format.format == VK_FORMAT_B8G8R8A8_UNORM ? 3 : rgba8 ? 2 : 1;
        };
        const auto preferred = std::max_element(formats.begin(), formats.end(), [&](const auto& a, const auto& b) {
            return score(a) < score(b);
        });
        if (score(*preferred) != 0)
            surface_format = *preferred;
    }
    if (!use_hdr10 && (surface_format.format == VK_FORMAT_UNDEFINED || surface_format.colorSpace != VK_COLOR_SPACE_SRGB_NONLINEAR_KHR ||
        (m_allow_readback && surface_format.format != VK_FORMAT_B8G8R8A8_UNORM && surface_format.format != VK_FORMAT_R8G8B8A8_UNORM)))
    {
        error = m_allow_readback ? "Vulkan surface has no 8-bit SDR RGBA swapchain format for readback" :
            "Vulkan surface has no SDR swapchain color space";
        return false;
    }

    if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        m_presentation_extent = capabilities.currentExtent;
    else
    {
        m_presentation_extent.width = std::clamp(requested_extent.width,
            capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        m_presentation_extent.height = std::clamp(requested_extent.height,
            capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }
    if (!m_presentation_extent.width || !m_presentation_extent.height)
    {
        error = "Vulkan surface has zero extent";
        return false;
    }

    m_extent = m_requested_scene_extent.width ? m_requested_scene_extent : m_presentation_extent;

    VkCompositeAlphaFlagBitsKHR composite_alpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(capabilities.supportedCompositeAlpha & composite_alpha))
    {
        constexpr std::array<VkCompositeAlphaFlagBitsKHR, 3> alternatives{
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR
        };
        const auto supported = std::find_if(alternatives.begin(), alternatives.end(), [&](auto candidate)
        {
            return capabilities.supportedCompositeAlpha & candidate;
        });
        if (supported == alternatives.end())
        {
            error = "Vulkan surface has no supported composite alpha mode";
            return false;
        }
        composite_alpha = *supported;
    }

    uint32_t image_count = capabilities.minImageCount;
    if (image_count < std::numeric_limits<uint32_t>::max())
        ++image_count;
    if (capabilities.maxImageCount && image_count > capabilities.maxImageCount)
        image_count = capabilities.maxImageCount;

    VkSwapchainCreateInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    info.surface = surface;
    info.minImageCount = image_count;
    info.imageFormat = surface_format.format;
    info.imageColorSpace = surface_format.colorSpace;
    info.imageExtent = m_presentation_extent;
    info.imageArrayLayers = 1;
    info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
        (m_allow_readback && !use_hdr10 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // The engine renders upright in the Android window's logical orientation.
    // With currentTransform on rotated displays, the compositor interprets
    // those pixels as already rotated, yielding a sideways, reduced frame.
    info.preTransform = (capabilities.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR) ?
        VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR : capabilities.currentTransform;
    info.compositeAlpha = composite_alpha;
    info.presentMode = VK_PRESENT_MODE_FIFO_KHR;
    info.clipped = VK_TRUE;
    constexpr VkImageUsageFlags frame_interpolation_usage =
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!use_hdr10 && m_swapchain_create && m_swapchain_context.device == m_device &&
        (capabilities.supportedUsageFlags & frame_interpolation_usage) == frame_interpolation_usage)
    {
        const VkSwapchainCreateInfoKHR native_info = info;
        const FrameDispatch native_dispatch = m_native_vk;
        FrameSwapchainCreateContext callback_context = m_swapchain_context;
        callback_context.scene_extent = m_extent;
        VkSwapchainKHR replacement = VK_NULL_HANDLE;
        bool frame_interpolation = false;
        std::string callback_error;
        if (m_swapchain_create(info, callback_context, m_vk, replacement,
            frame_interpolation, m_swapchain_data, callback_error))
        {
            if (!replacement || !m_vk.destroy_swapchain || !m_vk.get_swapchain_images ||
                !m_vk.acquire_next_image || !m_vk.queue_present)
            {
                error = "frame interpolation returned an incomplete replacement swapchain";
                if (replacement && m_vk.destroy_swapchain)
                    m_vk.destroy_swapchain(m_device, replacement, nullptr);
                m_vk = native_dispatch;
                return false;
            }
            m_swapchain = replacement;
            m_frame_interpolation_swapchain = frame_interpolation;
        }
        else
        {
            // Frame interpolation is optional. Restore the ordinary SDL
            // swapchain path when the provider cannot initialize its backend.
            info = native_info;
            m_vk = native_dispatch;
        }
    }
    if (!m_swapchain)
    {
        const VkResult result = m_vk.create_swapchain(m_device, &info, nullptr, &m_swapchain);
        if (result != VK_SUCCESS)
        {
            record_surface_error(result);
            error = "vkCreateSwapchainKHR failed: " + std::to_string(result);
            return false;
        }
    }
    m_format = surface_format.format;
    m_color_space = surface_format.colorSpace;
    m_hdr10_surface_available = hdr10_available;
    m_hdr_metadata_peak = m_hdr_metadata_black = -1.f;

    if (!enumerate<VkImage>([&](uint32_t* count, VkImage* values)
        { return m_vk.get_swapchain_images(m_device, m_swapchain, count, values); }, m_images))
    {
        error = "could not enumerate Vulkan swapchain images";
        return false;
    }
    return true;
}

void FrameContext::update_hdr_metadata(float paper_white, float peak, float black)
{
    if (!hdr10_active() || !m_swapchain || !m_vk.set_hdr_metadata ||
        !std::isfinite(paper_white) || !std::isfinite(peak) || !std::isfinite(black))
        return;
    peak = std::clamp(peak, 80.f, 10000.f);
    paper_white = std::clamp(paper_white, 80.f, peak);
    black = std::clamp(black, 0.f, paper_white - 1.f);
    if (peak == m_hdr_metadata_peak && black == m_hdr_metadata_black)
        return;
    VkHdrMetadataEXT metadata{};
    metadata.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;
    metadata.displayPrimaryRed = {0.708f, 0.292f};
    metadata.displayPrimaryGreen = {0.170f, 0.797f};
    metadata.displayPrimaryBlue = {0.131f, 0.046f};
    metadata.whitePoint = {0.3127f, 0.3290f};
    metadata.maxLuminance = peak;
    metadata.minLuminance = black;
    metadata.maxContentLightLevel = peak;
    // Frame-average luminance is unknown without measuring the encoded frames.
    m_vk.set_hdr_metadata(m_device, 1, &m_swapchain, &metadata);
    m_hdr_metadata_peak = peak;
    m_hdr_metadata_black = black;
}

bool FrameContext::create_scene_colors(std::string& error)
{
    for (size_t i = 0; i < m_images.size(); ++i)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = m_scene_format;
        info.extent = {m_extent.width, m_extent.height, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkImage image = VK_NULL_HANDLE;
        if (!m_frame_image_pool.acquire(info, image, error))
        { error = "independent scene color: " + error; return false; }
        m_scene_images.push_back(image);
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = m_scene_format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = view.subresourceRange.layerCount = 1;
        VkImageView output = VK_NULL_HANDLE;
        if (m_vk.create_image_view(m_device, &view, nullptr, &output) != VK_SUCCESS)
        { error = "could not create independent scene color view"; return false; }
        m_scene_views.push_back(output);
    }
    return true;
}

bool FrameContext::create_multisampled_scene_colors(std::string& error)
{
    if (m_scene_samples == VK_SAMPLE_COUNT_1_BIT)
        return true;
    if (!m_postprocess)
    {
        error = "multisampled scene color requires the frame image pool";
        return false;
    }
    for (size_t i = 0; i < m_images.size(); ++i)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = m_scene_format;
        info.extent = {m_extent.width, m_extent.height, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = m_scene_samples;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkImage image = VK_NULL_HANDLE;
        if (!m_frame_image_pool.acquire(info, image, error))
        { error = "multisampled scene color: " + error; return false; }
        m_multisampled_scene_images.push_back(image);
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = m_scene_format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = view.subresourceRange.layerCount = 1;
        VkImageView output = VK_NULL_HANDLE;
        if (m_vk.create_image_view(m_device, &view, nullptr, &output) != VK_SUCCESS)
        { error = "could not create multisampled scene color view"; return false; }
        m_multisampled_scene_views.push_back(output);
    }
    return true;
}

bool FrameContext::request_scene_samples(VkSampleCountFlagBits samples, std::string& error)
{
    if (samples != VK_SAMPLE_COUNT_1_BIT && samples != VK_SAMPLE_COUNT_2_BIT &&
        samples != VK_SAMPLE_COUNT_4_BIT)
    {
        error = "scene sample count must be 1x, 2x or 4x";
        return false;
    }
    if (samples != VK_SAMPLE_COUNT_1_BIT && (!m_postprocess || m_depth_format == VK_FORMAT_UNDEFINED))
    {
        error = "multisampled scene rendering requires postprocess targets and a depth attachment";
        return false;
    }
    m_scene_samples = samples;
    error.clear();
    return true;
}

bool FrameContext::request_scene_extent(VkExtent2D extent, std::string& error)
{
    if ((extent.width == 0) != (extent.height == 0) ||
        ((extent.width || extent.height) && (!m_postprocess || !m_device)))
    {
        error = "independent scene extent requires a live postprocess frame context";
        return false;
    }
    m_requested_scene_extent = extent;
    error.clear();
    return true;
}

bool FrameContext::create_render_targets(std::string& error)
{
    m_scene_format = m_format;
    if ((m_linear_scene || hdr10_active()) && m_postprocess && m_requested_scene_extent.width &&
        !m_frame_interpolation_swapchain)
    {
        VkFormatProperties properties{};
        m_vk.get_format_properties(m_physical_device, VK_FORMAT_R16G16B16A16_SFLOAT, &properties);
        constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if ((properties.optimalTilingFeatures & required) == required)
            m_scene_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    m_scene_depth_attached = false;
    m_scene_depth_images.clear();
    if (m_requested_scene_extent.width && !create_scene_colors(error)) return false;
    if (!create_multisampled_scene_colors(error)) return false;
    VkAttachmentDescription attachments[3]{};
    attachments[0].format = m_scene_format;
    attachments[0].samples = m_scene_samples;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachments[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[0].finalLayout = m_scene_samples == VK_SAMPLE_COUNT_1_BIT ?
        (m_allow_readback ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR) :
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    uint32_t depth_attachment = 1;
    uint32_t resolve_attachment = UINT32_MAX;
    VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resolve{};
    if (m_scene_samples != VK_SAMPLE_COUNT_1_BIT)
    {
        resolve_attachment = 1;
        attachments[resolve_attachment].format = m_scene_format;
        attachments[resolve_attachment].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[resolve_attachment].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[resolve_attachment].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[resolve_attachment].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[resolve_attachment].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[resolve_attachment].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[resolve_attachment].finalLayout = m_allow_readback ?
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        resolve = {resolve_attachment, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        depth_attachment = 2;
    }
    attachments[depth_attachment].format = m_depth_format;
    attachments[depth_attachment].samples = m_scene_samples;
    attachments[depth_attachment].loadOp = m_preserve_prepass_depth ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachments[depth_attachment].storeOp = m_preserve_prepass_depth ?
        VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachments[depth_attachment].initialLayout = m_preserve_prepass_depth ?
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    attachments[depth_attachment].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    VkAttachmentReference depth{depth_attachment, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color;
    if (resolve_attachment != UINT32_MAX)
        subpass.pResolveAttachments = &resolve;
    if (m_depth_format != VK_FORMAT_UNDEFINED)
        subpass.pDepthStencilAttachment = &depth;

    std::array<VkSubpassDependency, 2> dependencies{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    if (m_depth_format != VK_FORMAT_UNDEFINED)
    {
        dependencies[0].srcStageMask |= m_preserve_prepass_depth ?
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT : VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].dstStageMask |= VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[0].srcAccessMask |= m_preserve_prepass_depth ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : 0;
        dependencies[0].dstAccessMask |= VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = m_allow_readback ? VK_PIPELINE_STAGE_TRANSFER_BIT :
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = m_allow_readback ? VK_ACCESS_TRANSFER_READ_BIT : 0;
    if (m_scene_samples != VK_SAMPLE_COUNT_1_BIT)
    {
        dependencies[1].dstStageMask |= VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[1].dstAccessMask |= VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }

    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = m_depth_format != VK_FORMAT_UNDEFINED ?
        (m_scene_samples == VK_SAMPLE_COUNT_1_BIT ? 2u : 3u) : 1u;
    render_pass_info.pAttachments = attachments;
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = static_cast<uint32_t>(dependencies.size());
    render_pass_info.pDependencies = dependencies.data();
    if (m_vk.create_render_pass(m_device, &render_pass_info, nullptr, &m_render_pass) != VK_SUCCESS)
    {
        error = "vkCreateRenderPass failed";
        return false;
    }

    m_image_views.reserve(m_images.size());
    m_depth_images.reserve(m_images.size());
    m_depth_memories.reserve(m_images.size());
    m_depth_views.reserve(m_images.size());
    m_framebuffers.reserve(m_images.size());
    for (size_t image_index = 0; image_index < m_images.size(); ++image_index)
    {
        const VkImage image = m_images[image_index];
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = m_format;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.layerCount = 1;
        VkImageView view = VK_NULL_HANDLE;
        if (m_vk.create_image_view(m_device, &view_info, nullptr, &view) != VK_SUCCESS)
        {
            error = "vkCreateImageView failed";
            return false;
        }
        m_image_views.push_back(view);

        VkImageView framebuffer_views[3]{view, VK_NULL_HANDLE, VK_NULL_HANDLE};
        const bool borrowed_depth_pending = m_preserve_prepass_depth &&
            m_scene_samples != VK_SAMPLE_COUNT_1_BIT;
        if (m_depth_format != VK_FORMAT_UNDEFINED && !borrowed_depth_pending)
        {
            VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.format = m_depth_format;
            image_info.extent = {m_extent.width, m_extent.height, 1};
            image_info.mipLevels = image_info.arrayLayers = 1;
            image_info.samples = m_scene_samples;
            image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            image_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkImage depth_image = VK_NULL_HANDLE;
            if (m_postprocess)
            {
                if (!m_frame_image_pool.acquire(image_info, depth_image, error))
                {
                    error = "depth attachment: " + error;
                    return false;
                }
            }
            else
            {
                if (m_vk.create_image(m_device, &image_info, nullptr, &depth_image) != VK_SUCCESS)
                {
                    error = "vkCreateImage failed for depth attachment";
                    return false;
                }
            }
            m_depth_images.push_back(depth_image);
            if (!m_postprocess)
            {
                VkMemoryRequirements requirements{};
                m_vk.get_image_memory_requirements(m_device, depth_image, &requirements);
                uint32_t memory_type = UINT32_MAX;
                for (uint32_t i = 0; i < m_memory_properties.memoryTypeCount; ++i)
                    if ((requirements.memoryTypeBits & (1u << i)) &&
                        (m_memory_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                    {
                        memory_type = i;
                        break;
                    }
                if (memory_type == UINT32_MAX)
                {
                    error = "no device-local memory type for depth attachment";
                    return false;
                }
                VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                allocation.allocationSize = requirements.size;
                allocation.memoryTypeIndex = memory_type;
                VkDeviceMemory memory = VK_NULL_HANDLE;
                if (m_vk.allocate_memory(m_device, &allocation, nullptr, &memory) != VK_SUCCESS)
                {
                    error = "vkAllocateMemory failed for depth attachment";
                    return false;
                }
                if (m_vk.bind_image_memory(m_device, depth_image, memory, 0) != VK_SUCCESS)
                {
                    m_vk.free_memory(m_device, memory, nullptr);
                    error = "vkBindImageMemory failed for depth attachment";
                    return false;
                }
                m_depth_memories.push_back(memory);
            }
            VkImageViewCreateInfo depth_view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            depth_view_info.image = depth_image;
            depth_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            depth_view_info.format = m_depth_format;
            depth_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            depth_view_info.subresourceRange.levelCount = depth_view_info.subresourceRange.layerCount = 1;
            VkImageView depth_view = VK_NULL_HANDLE;
            if (m_vk.create_image_view(m_device, &depth_view_info, nullptr, &depth_view) != VK_SUCCESS)
            {
                error = "vkCreateImageView failed for depth attachment";
                return false;
            }
            m_depth_views.push_back(depth_view);
            framebuffer_views[m_scene_samples == VK_SAMPLE_COUNT_1_BIT ? 1 : 2] = depth_view;
        }

        if (m_scene_samples != VK_SAMPLE_COUNT_1_BIT)
        {
            framebuffer_views[0] = m_multisampled_scene_views[image_index];
            framebuffer_views[1] = m_scene_views.empty() ? view : m_scene_views[image_index];
        }
        else if (!m_scene_views.empty())
            framebuffer_views[0] = m_scene_views[image_index];
        // The multisampled G-buffer owns this depth image. The game attaches
        // it after G-buffer target creation, before any frame can be recorded.
        if (borrowed_depth_pending)
            continue;
        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebuffer_info.renderPass = m_render_pass;
        framebuffer_info.attachmentCount = m_depth_format != VK_FORMAT_UNDEFINED ?
            (m_scene_samples == VK_SAMPLE_COUNT_1_BIT ? 2u : 3u) : 1u;
        framebuffer_info.pAttachments = framebuffer_views;
        framebuffer_info.width = m_extent.width;
        framebuffer_info.height = m_extent.height;
        framebuffer_info.layers = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        if (m_vk.create_framebuffer(m_device, &framebuffer_info, nullptr, &framebuffer) != VK_SUCCESS)
        {
            error = "vkCreateFramebuffer failed";
            return false;
        }
        m_framebuffers.push_back(framebuffer);
    }
    if (m_postprocess && !create_postprocess_targets(error,
            hdr10_active() ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_UNDEFINED)) return false;
    return true;
}

VkImageView FrameContext::composite_view(uint32_t index) const
{
    const auto& views = m_composite_views.empty() ? m_image_views : m_composite_views;
    return index < views.size() ? views[index] : VK_NULL_HANDLE;
}

bool FrameContext::create_postprocess_targets(std::string& error, VkFormat composite_format)
{
    m_composite_format = composite_format == VK_FORMAT_UNDEFINED ? m_format : composite_format;
    const bool independent_composite = m_composite_format != m_format;
    VkFormatProperties format_properties{};
    m_vk.get_format_properties(m_physical_device, m_scene_format, &format_properties);
    const auto required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    m_post_mip_levels = m_vk.cmd_blit_image &&
        (format_properties.optimalTilingFeatures & required) == required ?
        1u + uint32_t(std::floor(std::log2(std::max(m_extent.width, m_extent.height)))) : 1u;
    m_bloom_extent = {std::max(1u, m_extent.width / 2), std::max(1u, m_extent.height / 2)};
    VkFormatProperties bloom_properties{};
    m_vk.get_format_properties(m_physical_device, VK_FORMAT_R16G16B16A16_SFLOAT, &bloom_properties);
    const VkFormatFeatureFlags bloom_required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | required;
    m_bloom_format = (bloom_properties.optimalTilingFeatures & bloom_required) == bloom_required ?
        VK_FORMAT_R16G16B16A16_SFLOAT : m_format;
    m_bloom_mip_levels = 1;
    if (m_post_mip_levels > 1)
        while (m_bloom_mip_levels < 8 &&
            (std::min(m_bloom_extent.width, m_bloom_extent.height) >> m_bloom_mip_levels) >= 8)
            ++m_bloom_mip_levels;
    VkAttachmentDescription color{};
    color.format = m_composite_format;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout = m_scene_images.empty() ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = m_allow_readback || independent_composite ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL :
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    const VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkSubpassDependency deps[2]{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1;
    pass.pAttachments = &color;
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = 2;
    pass.pDependencies = deps;
    if (m_vk.create_render_pass(m_device, &pass, nullptr, &m_composite_pass) != VK_SUCCESS)
    {
        error = "could not create Vulkan postprocess render pass";
        return false;
    }
    if (independent_composite)
    {
        VkAttachmentDescription output = color;
        output.format = m_format;
        output.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        output.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkRenderPassCreateInfo output_pass = pass;
        output_pass.pAttachments = &output;
        output_pass.dependencyCount = 0;
        output_pass.pDependencies = nullptr;
        if (m_vk.create_render_pass(m_device, &output_pass, nullptr, &m_display_output_pass) != VK_SUCCESS)
        {
            error = "could not create Vulkan display encoding render pass";
            return false;
        }
    }
    if (!m_scene_images.empty())
    {
        color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        color.initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
        deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
        deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        if (m_vk.create_render_pass(m_device, &pass, nullptr, &m_presentation_pass) != VK_SUCCESS)
        {
            error = "could not create native-resolution UI load pass";
            return false;
        }
    }
    if (m_frame_interpolation_swapchain && m_presentation_pass)
    {
        VkAttachmentDescription ui_color{};
        ui_color.format = m_composite_format;
        ui_color.samples = VK_SAMPLE_COUNT_1_BIT;
        ui_color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ui_color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ui_color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ui_color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        ui_color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        ui_color.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        const VkAttachmentReference ui_reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription ui_subpass{};
        ui_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        ui_subpass.colorAttachmentCount = 1;
        ui_subpass.pColorAttachments = &ui_reference;
        VkSubpassDependency ui_dependencies[2]{};
        ui_dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
        ui_dependencies[0].dstSubpass = 0;
        ui_dependencies[0].srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        ui_dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        ui_dependencies[0].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        ui_dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        ui_dependencies[1].srcSubpass = 0;
        ui_dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
        ui_dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        ui_dependencies[1].dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
        ui_dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        ui_dependencies[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        VkRenderPassCreateInfo ui_pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        ui_pass.attachmentCount = 1;
        ui_pass.pAttachments = &ui_color;
        ui_pass.subpassCount = 1;
        ui_pass.pSubpasses = &ui_subpass;
        ui_pass.dependencyCount = 2;
        ui_pass.pDependencies = ui_dependencies;
        if (m_vk.create_render_pass(m_device, &ui_pass, nullptr, &m_ui_clear_pass) != VK_SUCCESS)
        {
            error = "could not create FSR UI composition clear render pass";
            return false;
        }
    }
    VkAttachmentDescription bloom_color{};
    bloom_color.format = m_bloom_format;
    bloom_color.samples = VK_SAMPLE_COUNT_1_BIT;
    bloom_color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    bloom_color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    bloom_color.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    bloom_color.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkRenderPassCreateInfo bloom_pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    bloom_pass.attachmentCount = 1;
    bloom_pass.pAttachments = &bloom_color;
    bloom_pass.subpassCount = 1;
    bloom_pass.pSubpasses = &subpass;
    if (m_vk.create_render_pass(m_device, &bloom_pass, nullptr, &m_bloom_pass) != VK_SUCCESS)
    {
        error = "could not create Vulkan bloom render pass";
        return false;
    }
    bloom_color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    if (m_vk.create_render_pass(m_device, &bloom_pass, nullptr, &m_bloom_upsample_pass) != VK_SUCCESS)
    {
        error = "could not create Vulkan bloom upsample render pass";
        return false;
    }
    for (size_t i = 0; i < m_images.size(); ++i)
    {
        const VkImage scene_image = m_scene_images.empty() ? m_images[i] : m_scene_images[i];
        const VkImageSubresourceRange scene_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        // The scene and overlay render passes leave this target in transfer-src
        // layout and synchronize their color writes with the subsequent copy.
        if (!m_post_image_states.register_image(scene_image, scene_range, ImageUse::TransferSource, error))
            return false;
        if (!m_multisampled_scene_images.empty() &&
            !m_post_image_states.register_image(m_multisampled_scene_images[i], scene_range,
                ImageUse::ColorAttachment, error))
            return false;
        if (!m_scene_images.empty() &&
            !m_post_image_states.register_image(m_images[i], scene_range,
                independent_composite ? ImageUse::Undefined : ImageUse::Present, error))
            return false;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = m_scene_format;
        info.extent = {m_extent.width, m_extent.height, 1};
        info.mipLevels = m_post_mip_levels;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            (m_post_mip_levels > 1 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
        m_frame_image_sharing.apply(info);
        VkImage image = VK_NULL_HANDLE;
        if (!m_frame_image_pool.acquire(info, image, error))
        {
            error = "Vulkan postprocess source image: " + error;
            return false;
        }
        m_post_images.push_back(image);
        const VkImageSubresourceRange post_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_post_mip_levels, 0, 1};
        if (!m_post_image_states.register_image(image, post_range, ImageUse::Undefined, error))
            return false;
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = m_scene_format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.levelCount = m_post_mip_levels;
        view.subresourceRange.layerCount = 1;
        VkImageView source = VK_NULL_HANDLE;
        if (m_vk.create_image_view(m_device, &view, nullptr, &source) != VK_SUCCESS)
        {
            error = "could not create Vulkan postprocess source view";
            return false;
        }
        m_post_views.push_back(source);
        view.subresourceRange.levelCount = 1;
        VkImageView base_source = VK_NULL_HANDLE;
        if (m_vk.create_image_view(m_device, &view, nullptr, &base_source) != VK_SUCCESS)
        {
            error = "could not create Vulkan postprocess base-mip view";
            return false;
        }
        m_post_base_views.push_back(base_source);
        view.subresourceRange.levelCount = m_post_mip_levels;
        VkImageCreateInfo bloom_info = info;
        bloom_info.format = m_bloom_format;
        bloom_info.extent = {m_bloom_extent.width, m_bloom_extent.height, 1};
        bloom_info.mipLevels = m_bloom_mip_levels;
        bloom_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        FrameImageSharing{}.apply(bloom_info);
        VkImage bloom_image = VK_NULL_HANDLE;
        if (!m_frame_image_pool.acquire(bloom_info, bloom_image, error))
        {
            error = "Vulkan bloom image: " + error;
            return false;
        }
        m_bloom_images.push_back(bloom_image);
        const VkImageSubresourceRange bloom_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, m_bloom_mip_levels, 0, 1};
        if (!m_post_image_states.register_image(bloom_image, bloom_range, ImageUse::Undefined, error))
            return false;
        view.image = bloom_image;
        view.format = m_bloom_format;
        view.subresourceRange.levelCount = m_bloom_mip_levels;
        VkImageView bloom_view = VK_NULL_HANDLE;
        if (m_vk.create_image_view(m_device, &view, nullptr, &bloom_view) != VK_SUCCESS)
        {
            error = "could not create Vulkan bloom source view";
            return false;
        }
        m_bloom_views.push_back(bloom_view);
        view.subresourceRange.levelCount = 1;
        VkImageView bloom_target_view = VK_NULL_HANDLE;
        if (m_vk.create_image_view(m_device, &view, nullptr, &bloom_target_view) != VK_SUCCESS)
        {
            error = "could not create Vulkan bloom target view";
            return false;
        }
        m_bloom_target_views.push_back(bloom_target_view);
        VkFramebufferCreateInfo bloom_framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        bloom_framebuffer.renderPass = m_bloom_pass;
        bloom_framebuffer.attachmentCount = 1;
        bloom_framebuffer.pAttachments = &m_bloom_target_views.back();
        bloom_framebuffer.width = m_bloom_extent.width;
        bloom_framebuffer.height = m_bloom_extent.height;
        bloom_framebuffer.layers = 1;
        VkFramebuffer bloom_output = VK_NULL_HANDLE;
        if (m_vk.create_framebuffer(m_device, &bloom_framebuffer, nullptr, &bloom_output) != VK_SUCCESS)
        {
            error = "could not create Vulkan bloom framebuffer";
            return false;
        }
        m_bloom_framebuffers.push_back(bloom_output);
        m_bloom_level_views.emplace_back();
        m_bloom_upsample_framebuffers.emplace_back();
        for (uint32_t mip = 0; mip < m_bloom_mip_levels; ++mip)
        {
            view.subresourceRange.baseMipLevel = mip;
            VkImageView level_view = VK_NULL_HANDLE;
            if (m_vk.create_image_view(m_device, &view, nullptr, &level_view) != VK_SUCCESS)
            {
                error = "could not create Vulkan bloom mip view";
                return false;
            }
            m_bloom_level_views.back().push_back(level_view);
            if (mip + 1 == m_bloom_mip_levels) continue;
            VkFramebufferCreateInfo level_framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            level_framebuffer.renderPass = m_bloom_upsample_pass;
            level_framebuffer.attachmentCount = 1;
            level_framebuffer.pAttachments = &m_bloom_level_views.back().back();
            level_framebuffer.width = std::max(1u, m_bloom_extent.width >> mip);
            level_framebuffer.height = std::max(1u, m_bloom_extent.height >> mip);
            level_framebuffer.layers = 1;
            VkFramebuffer level_output = VK_NULL_HANDLE;
            if (m_vk.create_framebuffer(m_device, &level_framebuffer, nullptr, &level_output) != VK_SUCCESS)
            {
                error = "could not create Vulkan bloom upsample framebuffer";
                return false;
            }
            m_bloom_upsample_framebuffers.back().push_back(level_output);
        }
        if (independent_composite)
        {
            VkImageCreateInfo composition{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            composition.imageType = VK_IMAGE_TYPE_2D;
            composition.format = m_composite_format;
            composition.extent = {m_presentation_extent.width, m_presentation_extent.height, 1};
            composition.mipLevels = composition.arrayLayers = 1;
            composition.samples = VK_SAMPLE_COUNT_1_BIT;
            composition.tiling = VK_IMAGE_TILING_OPTIMAL;
            composition.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT;
            composition.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkImage image{};
            if (!m_frame_image_pool.acquire(composition, image, error))
                return false;
            m_composite_images.push_back(image);
            if (!m_post_image_states.register_image(image, scene_range, ImageUse::Undefined, error))
                return false;
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = image;
            view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = m_composite_format;
            view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            VkImageView output{};
            if (m_vk.create_image_view(m_device, &view, nullptr, &output) != VK_SUCCESS)
            {
                error = "could not create independent presentation composition view";
                return false;
            }
            m_composite_views.push_back(output);
        }
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = m_composite_pass;
        framebuffer.attachmentCount = 1;
        const VkImageView composite_attachment = composite_view(static_cast<uint32_t>(i));
        framebuffer.pAttachments = &composite_attachment;
        framebuffer.width = m_presentation_extent.width;
        framebuffer.height = m_presentation_extent.height;
        framebuffer.layers = 1;
        VkFramebuffer output = VK_NULL_HANDLE;
        if (m_vk.create_framebuffer(m_device, &framebuffer, nullptr, &output) != VK_SUCCESS)
        {
            error = "could not create Vulkan postprocess framebuffer";
            return false;
        }
        m_composite_framebuffers.push_back(output);
        if (m_display_output_pass)
        {
            framebuffer.renderPass = m_display_output_pass;
            framebuffer.pAttachments = &m_image_views[i];
            VkFramebuffer encoded{};
            if (m_vk.create_framebuffer(m_device, &framebuffer, nullptr, &encoded) != VK_SUCCESS)
            {
                error = "could not create Vulkan display encoding framebuffer";
                return false;
            }
            m_display_output_framebuffers.push_back(encoded);
        }

        if (m_ui_clear_pass)
        {
            VkImageCreateInfo ui_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            ui_info.imageType = VK_IMAGE_TYPE_2D;
            ui_info.format = m_composite_format;
            ui_info.extent = {m_presentation_extent.width, m_presentation_extent.height, 1};
            ui_info.mipLevels = ui_info.arrayLayers = 1;
            ui_info.samples = VK_SAMPLE_COUNT_1_BIT;
            ui_info.tiling = VK_IMAGE_TILING_OPTIMAL;
            ui_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT;
            ui_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkImage ui_image = VK_NULL_HANDLE;
            if (!m_frame_image_pool.acquire(ui_info, ui_image, error))
            {
                error = "FSR UI composition image: " + error;
                return false;
            }
            m_frame_generation_ui_images.push_back(ui_image);
            VkImageViewCreateInfo ui_view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            ui_view.image = ui_image;
            ui_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            ui_view.format = m_composite_format;
            ui_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            ui_view.subresourceRange.levelCount = 1;
            ui_view.subresourceRange.layerCount = 1;
            VkImageView image_view = VK_NULL_HANDLE;
            if (m_vk.create_image_view(m_device, &ui_view, nullptr, &image_view) != VK_SUCCESS)
            {
                error = "could not create FSR UI composition image view";
                return false;
            }
            m_frame_generation_ui_views.push_back(image_view);
            VkFramebufferCreateInfo ui_framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
            ui_framebuffer.renderPass = m_ui_clear_pass;
            ui_framebuffer.attachmentCount = 1;
            ui_framebuffer.pAttachments = &m_frame_generation_ui_views.back();
            ui_framebuffer.width = m_presentation_extent.width;
            ui_framebuffer.height = m_presentation_extent.height;
            ui_framebuffer.layers = 1;
            VkFramebuffer ui_output = VK_NULL_HANDLE;
            if (m_vk.create_framebuffer(m_device, &ui_framebuffer, nullptr, &ui_output) != VK_SUCCESS)
            {
                error = "could not create FSR UI composition framebuffer";
                return false;
            }
            m_frame_generation_ui_framebuffers.push_back(ui_output);
        }
    }
    return true;
}

bool FrameContext::attach_scene_depth(const std::vector<VkImageView>& views,
    const std::vector<VkImage>& images, std::string& error)
{
    if (!m_preserve_prepass_depth || !m_render_pass ||
        (views.empty() != images.empty()) ||
        (!views.empty() && (views.size() != m_image_views.size() || images.size() != views.size() ||
            std::any_of(views.begin(), views.end(), [](VkImageView view) { return !view; }) ||
            std::any_of(images.begin(), images.end(), [](VkImage image) { return !image; }))))
    {
        error = "invalid borrowed Vulkan scene depth views";
        return false;
    }
    std::vector<VkFramebuffer> replacements;
    replacements.reserve(m_image_views.size());
    for (size_t i = 0; i < m_image_views.size(); ++i)
    {
        VkImageView attachments[3]{};
        uint32_t attachment_count = 2;
        if (m_scene_samples == VK_SAMPLE_COUNT_1_BIT)
        {
            attachments[0] = m_scene_views.empty() ? m_image_views[i] : m_scene_views[i];
            attachments[1] = views.empty() ? m_depth_views[i] : views[i];
        }
        else
        {
            attachments[0] = m_multisampled_scene_views[i];
            attachments[1] = m_scene_views.empty() ? m_image_views[i] : m_scene_views[i];
            attachments[2] = views.empty() ? m_depth_views[i] : views[i];
            attachment_count = 3;
        }
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = m_render_pass;
        info.attachmentCount = attachment_count;
        info.pAttachments = attachments;
        info.width = m_extent.width;
        info.height = m_extent.height;
        info.layers = 1;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
        if (m_vk.create_framebuffer(m_device, &info, nullptr, &framebuffer) != VK_SUCCESS)
        {
            for (auto created : replacements) m_vk.destroy_framebuffer(m_device, created, nullptr);
            error = "could not attach Vulkan scene depth to swapchain";
            return false;
        }
        replacements.push_back(framebuffer);
    }
    for (auto old : m_framebuffers) m_vk.destroy_framebuffer(m_device, old, nullptr);
    m_framebuffers.swap(replacements);
    m_scene_depth_images = images;
    m_scene_depth_attached = !views.empty();
    error.clear();
    return true;
}

bool FrameContext::attach_temporal_exposure(VkImage image, FrameRecorder recorder,
    FrameHistoryValid history_valid, void* data, std::string& error)
{
    if (!m_postprocess || !image || !recorder || !history_valid || m_exposure_image)
    {
        error = "temporal exposure requires an unused postprocess history slot";
        return false;
    }
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (!m_post_image_states.register_image(image, range, ImageUse::Undefined, error))
        return false;
    m_exposure_image = image;
    m_exposure_recorder = recorder;
    m_exposure_history_valid = history_valid;
    m_exposure_data = data;
    error.clear();
    return true;
}

bool FrameContext::attach_temporal_resolve(const std::array<VkImage, 2>& history_images,
    const std::vector<VkImage>& motion_images, FrameRecorder recorder,
    FrameHistoryValid active, FrameOutputSlot output_slot, void* data,
    std::string& error, VkExtent2D history_extent)
{
    if (!m_postprocess || m_post_mip_levels < 2 || m_post_depth_images.empty() || m_temporal_images[0] ||
        !history_images[0] || !history_images[1] || history_images[0] == history_images[1] ||
        motion_images.size() != m_images.size() || !recorder || !active || !output_slot)
    {
        error = "temporal resolve requires two histories and one motion image per frame target";
        return false;
    }
    ImageStateTracker candidate = m_post_image_states;
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (VkImage image : history_images)
        if (!candidate.register_image(image, range, ImageUse::Undefined, error)) return false;
    for (VkImage image : motion_images)
        if (!candidate.has_image(image) &&
            !candidate.register_image(image, range, ImageUse::Sampled, error)) return false;
    m_post_image_states = std::move(candidate);
    m_temporal_images = history_images;
    m_temporal_extent = history_extent.width && history_extent.height ? history_extent : m_extent;
    m_temporal_motion_images = motion_images;
    m_temporal_recorder = recorder;
    m_temporal_active = active;
    m_temporal_output_slot = output_slot;
    m_temporal_data = data;
    error.clear();
    return true;
}

bool FrameContext::attach_fsr3_upscale(const std::vector<VkImage>& output_images,
    const std::vector<VkImage>& motion_images, FrameCheckedRecorder recorder,
    FrameHistoryValid active, void* data, std::string& error)
{
    if (!m_postprocess || m_post_depth_images.empty() || !m_fsr3_output_images.empty() ||
        output_images.size() != m_images.size() || motion_images.size() != m_images.size() ||
        !recorder || !active ||
        std::any_of(output_images.begin(), output_images.end(), [](VkImage image) { return !image; }) ||
        std::any_of(motion_images.begin(), motion_images.end(), [](VkImage image) { return !image; }))
    {
        error = "FSR3 upscale requires one output and motion image per frame target";
        return false;
    }
    ImageStateTracker candidate = m_post_image_states;
    const VkImageSubresourceRange color_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (VkImage image : output_images)
        if (!candidate.register_image(image, color_range, ImageUse::Undefined, error)) return false;
    for (VkImage image : motion_images)
        if (!candidate.has_image(image) &&
            !candidate.register_image(image, color_range, ImageUse::Sampled, error)) return false;
    m_post_image_states = std::move(candidate);
    m_fsr3_output_images = output_images;
    m_fsr3_motion_images = motion_images;
    m_fsr3_recorder = recorder;
    m_fsr3_active = active;
    m_fsr3_data = data;
    error.clear();
    return true;
}

bool FrameContext::attach_postprocess_depth(const std::vector<VkImage>& images, std::string& error)
{
    if (!m_postprocess || !m_post_depth_images.empty() || images.size() != m_images.size() ||
        images.empty())
    {
        error = "postprocess depth needs one borrowed image per swapchain image";
        return false;
    }
    ImageStateTracker candidate = m_post_image_states;
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    for (VkImage image : images)
        if (!image || !candidate.register_image(image, range, ImageUse::Sampled, error))
        {
            if (error.empty()) error = "postprocess depth image is null";
            return false;
        }
    m_post_image_states = std::move(candidate);
    m_post_depth_images = images;
    error.clear();
    return true;
}

bool FrameContext::attach_postprocess_motion(const std::vector<VkImage>& images, std::string& error)
{
    if (!m_postprocess || !m_post_motion_images.empty() || images.size() != m_images.size() || images.empty() ||
        std::any_of(images.begin(), images.end(), [](VkImage image) { return !image; }))
    {
        error = "postprocess motion needs one borrowed image per swapchain image";
        return false;
    }
    if ((!m_temporal_motion_images.empty() && m_temporal_motion_images != images) ||
        (!m_fsr3_motion_images.empty() && m_fsr3_motion_images != images))
    {
        error = "postprocess motion images must match the temporal-upscaler motion inputs";
        return false;
    }
    ImageStateTracker candidate = m_post_image_states;
    if (m_temporal_motion_images.empty() && m_fsr3_motion_images.empty())
    {
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        for (VkImage image : images)
            if (!candidate.register_image(image, range, ImageUse::Sampled, error))
                return false;
        m_post_image_states = std::move(candidate);
    }
    m_post_motion_images = images;
    error.clear();
    return true;
}

bool FrameContext::enable_interpass(std::string& error)
{
    if (m_overlay_pass) return true;
    if (!m_device || !m_render_pass || !m_allow_readback || !m_scene_depth_attached ||
        m_depth_format == VK_FORMAT_UNDEFINED)
    { error = "water interpass requires a readable color target and scene depth"; return false; }
    VkAttachmentDescription attachments[3]{};
    attachments[0].format = m_scene_format;
    attachments[0].samples = m_scene_samples;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout = m_scene_samples == VK_SAMPLE_COUNT_1_BIT ?
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    attachments[0].finalLayout = m_scene_samples == VK_SAMPLE_COUNT_1_BIT ?
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    uint32_t depth_attachment = 1;
    uint32_t resolve_attachment = UINT32_MAX;
    if (m_scene_samples != VK_SAMPLE_COUNT_1_BIT)
    {
        resolve_attachment = 1;
        attachments[resolve_attachment].format = m_scene_format;
        attachments[resolve_attachment].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[resolve_attachment].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[resolve_attachment].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[resolve_attachment].initialLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        attachments[resolve_attachment].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        depth_attachment = 2;
    }
    attachments[depth_attachment].format = m_depth_format;
    attachments[depth_attachment].samples = m_scene_samples;
    attachments[depth_attachment].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[depth_attachment].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[depth_attachment].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    attachments[depth_attachment].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference resolve{resolve_attachment, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference depth{depth_attachment, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color;
    if (resolve_attachment != UINT32_MAX)
        subpass.pResolveAttachments = &resolve;
    subpass.pDepthStencilAttachment = &depth;
    const VkSubpassDependency dependencies[]{
    {
        VK_SUBPASS_EXTERNAL, 0,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        0},
    {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT, 0}};
    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = m_scene_samples == VK_SAMPLE_COUNT_1_BIT ? 2u : 3u;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 2;
    info.pDependencies = dependencies;
    if (m_vk.create_render_pass(m_device, &info, nullptr, &m_overlay_pass) != VK_SUCCESS)
    { error = "cannot create Vulkan water overlay pass"; return false; }
    error.clear();
    return true;
}

void FrameContext::record_distortion_mask(VkCommandBuffer command, uint32_t image_index,
    FrameRecorder recorder, void* data) const
{
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = m_overlay_pass;
    begin.framebuffer = m_framebuffers[image_index];
    begin.renderArea.extent = m_extent;
    m_vk.cmd_begin_render_pass(command, &begin, VK_SUBPASS_CONTENTS_INLINE);
    VkClearAttachment clear{};
    clear.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    clear.clearValue.color = {{127.f / 255.f, 127.f / 255.f, 0.f, 127.f / 255.f}};
    VkClearRect rect{};
    rect.rect.extent = m_extent;
    rect.layerCount = 1;
    m_vk.cmd_clear_attachments(command, 1, &clear, 1, &rect);
    recorder({command, m_overlay_pass, begin.framebuffer, m_extent, image_index, m_current_frame}, data);
    m_vk.cmd_end_render_pass(command);
}

void FrameContext::clear_depth(VkCommandBuffer command) const
{
    VkClearAttachment attachment{};
    attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    attachment.clearValue.depthStencil = {1.0f, 0};
    VkClearRect rect{};
    rect.rect.extent = m_extent;
    rect.layerCount = 1;
    m_vk.cmd_clear_attachments(command, 1, &attachment, 1, &rect);
}

bool FrameContext::create_commands(std::string& error)
{
    // A previous command-list allocation failure may have been transient.
    m_async_compute_enabled = m_async_compute_available;
    m_async_copy_enabled = m_async_copy_available;
    if (!m_command_pool)
    {
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.queueFamilyIndex = m_queue_family;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (m_vk.create_command_pool(m_device, &pool_info, nullptr, &m_command_pool) != VK_SUCCESS)
        {
            error = "vkCreateCommandPool failed";
            return false;
        }
    }

    if (m_nvrhi_device)
    {
        const auto create_optional_graphics_list = [&]() -> nvrhi::CommandListHandle
        {
            try
            {
                return m_nvrhi_device->createCommandList();
            }
            catch (...)
            {
                // Optional split submissions fall back to the main list.
                return {};
            }
        };
        try
        {
            for (auto& frame_lists : m_nvrhi_command_lists)
            {
                if (!frame_lists[0])
                    frame_lists[0] = m_nvrhi_device->createCommandList();
                if (!frame_lists[0])
                {
                    error = "NVRHI could not create a frame command list";
                    m_nvrhi_command_lists = {};
                    return false;
                }
                if ((m_async_compute_enabled || m_async_copy_enabled) && !frame_lists[1])
                    frame_lists[1] = create_optional_graphics_list();
                if ((m_async_compute_enabled || m_async_copy_enabled) && !frame_lists[1])
                {
                    m_async_compute_enabled = false;
                    m_async_copy_enabled = false;
                }
                if (m_async_compute_enabled && !frame_lists[2])
                    frame_lists[2] = create_optional_graphics_list();
                if (m_async_compute_enabled && !frame_lists[2])
                    m_async_compute_enabled = false;
            }
        }
        catch (const std::exception& exception)
        {
            m_nvrhi_command_lists = {};
            m_nvrhi_compute_command_lists = {};
            m_async_compute_enabled = false;
            m_async_copy_enabled = false;
            error = std::string("NVRHI frame command-list creation failed: ") + exception.what();
            return false;
        }
        if (m_async_compute_enabled)
        {
            try
            {
                nvrhi::CommandListParameters parameters;
                parameters.enableImmediateExecution = false;
                parameters.queueType = nvrhi::CommandQueue::Compute;
                for (auto& command_list : m_nvrhi_compute_command_lists)
                {
                    if (!command_list)
                        command_list = m_nvrhi_device->createCommandList(parameters);
                    if (!command_list)
                    {
                        m_async_compute_enabled = false;
                        break;
                    }
                }
            }
            catch (...)
            {
                m_async_compute_enabled = false;
            }
        }
        if (!m_async_compute_enabled)
        {
            m_nvrhi_compute_command_lists = {};
            for (auto& frame_lists : m_nvrhi_command_lists)
            {
                frame_lists[2] = nullptr;
                if (!m_async_copy_enabled)
                    frame_lists[1] = nullptr;
            }
        }
    }
    else
    {
        m_commands.resize(m_images.size());
        VkCommandBufferAllocateInfo allocate_info{};
        allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocate_info.commandPool = m_command_pool;
        allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate_info.commandBufferCount = static_cast<uint32_t>(m_commands.size());
        if (m_vk.allocate_command_buffers(m_device, &allocate_info, m_commands.data()) != VK_SUCCESS)
        {
            error = "vkAllocateCommandBuffers failed";
            return false;
        }
    }

    return true;
}

bool FrameContext::create_sync(std::string& error)
{
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t frame = 0; frame < FramesInFlight; ++frame)
    {
        if (m_vk.create_semaphore(m_device, &semaphore_info, nullptr, &m_image_available[frame]) != VK_SUCCESS ||
            m_vk.create_fence(m_device, &fence_info, nullptr, &m_frame_fences[frame]) != VK_SUCCESS)
        {
            error = "could not create Vulkan frame synchronization objects";
            return false;
        }
    }

    m_render_finished.resize(m_images.size());
    for (VkSemaphore& semaphore : m_render_finished)
        if (m_vk.create_semaphore(m_device, &semaphore_info, nullptr, &semaphore) != VK_SUCCESS)
        {
            error = "could not create Vulkan presentation semaphores";
            return false;
        }
    m_image_fences.resize(m_images.size(), VK_NULL_HANDLE);
    return true;
}

bool FrameContext::render_frame(const VkClearColorValue& clear, FrameStatus& status, std::string& error,
    FrameRecorder recorder, void* user_data, FrameReadbackRecorder readback, void* readback_data,
    FramePrepassRecorder prepass, void* prepass_data, bool clear_target,
    FrameRecorder compositor, void* compositor_data,
    FrameInterpassRecorder interpass, void* interpass_data,
    FrameRecorder overlay, void* overlay_data,
    FrameInterpassRecorder final_capture, void* final_capture_data,
    FrameRecorder final_overlay, void* final_overlay_data,
    FrameInterpassRecorder refraction_capture, void* refraction_capture_data,
    FrameRecorder refraction_overlay, void* refraction_overlay_data,
    FrameRecorder presentation_overlay, void* presentation_overlay_data,
    FrameInterpassRecorder presentation_capture, void* presentation_capture_data,
    FrameRecorder presentation_distortion, void* presentation_distortion_data,
    FrameInterpassRecorder display_capture, void* display_capture_data,
    FrameRecorder display_grade, void* display_grade_data, bool async_compute_requested,
    FrameRecordValidation validate_recording, void* validation_data,
    FrameSampledImageProvider sampled_images, void* sampled_images_data)
{
    using Clock = std::chrono::steady_clock;
    const auto milliseconds = [](Clock::time_point begin) {
        return std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    };
    m_cpu_timing = {};
    status = FrameStatus::Presented;
    if (!m_device || !m_swapchain)
    {
        error = "Vulkan frame context is not initialized";
        return false;
    }
    if (m_submission_failed)
    {
        // A failed submit leaves the acquired image and reset fence unusable
        // for another frame. Swapchain recreation restores both.
        error = "Vulkan frame submission failed; recreate the swapchain before rendering";
        return false;
    }
    if (m_preserve_prepass_depth && !m_scene_depth_attached)
    {
        error = "Vulkan swapchain has no attached scene depth";
        return false;
    }
    if ((final_capture || final_overlay) && (!m_overlay_pass || !final_capture || !final_overlay))
    { error = "UI distortion requires capture and overlay recorders"; return false; }
    if ((interpass || overlay) && (!m_overlay_pass || !interpass || !overlay))
    { error = "water overlay pass requires capture and overlay recorders"; return false; }
    if ((refraction_capture || refraction_overlay) &&
        (!m_overlay_pass || !interpass || !refraction_capture || !refraction_overlay))
    { error = "particle refraction requires completed world capture and overlay recorders"; return false; }
    if ((presentation_overlay || presentation_capture || presentation_distortion || display_capture || display_grade) &&
        (!m_presentation_pass || !presentation_overlay ||
            bool(presentation_capture) != bool(presentation_distortion) ||
            bool(display_capture) != bool(display_grade)))
    { error = "presentation UI requires its load pass and paired distortion capture/overlay"; return false; }
    const bool separate_frame_generation_ui = m_frame_interpolation_swapchain &&
        m_frame_generation_ui_active && m_frame_generation_ui_active(m_frame_generation_ui_data);
    if (separate_frame_generation_ui &&
        (!m_ui_clear_pass || !m_frame_generation_ui_register ||
            m_frame_generation_ui_images.size() != m_images.size() ||
            m_frame_generation_ui_framebuffers.size() != m_images.size()))
    {
        error = "FSR frame-generation UI targets are unavailable";
        return false;
    }

    crash_stage("vulkan frame: wait fence");
    const auto fence_begin = Clock::now();
    VkFence frame_fence = m_frame_fences[m_current_frame];
    const VkResult fence_wait = m_vk.wait_for_fences(m_device, 1, &frame_fence, VK_TRUE, UINT64_MAX);
    if (fence_wait != VK_SUCCESS)
    {
        m_device_lost = fence_wait == VK_ERROR_DEVICE_LOST;
        error = "could not wait for Vulkan frame fence";
        return false;
    }
    m_cpu_timing.fence_ms = milliseconds(fence_begin);
    if (m_nvrhi_device)
    {
        try { m_nvrhi_device->runGarbageCollection(); }
        catch (const std::exception& exception)
        {
            error = std::string("NVRHI frame garbage collection failed: ") + exception.what();
            return false;
        }
    }
    if (m_completion_pending[m_current_frame])
    {
        if (m_frame_completed && !m_frame_completed(m_current_frame, m_completion_images[m_current_frame],
                m_frame_completed_data, error))
        {
            if (error.empty()) error = "Vulkan frame completion callback failed";
            return false;
        }
        if (m_observer_completed && !m_observer_completed(m_current_frame, m_completion_images[m_current_frame],
                m_frame_observer_data, error))
        {
            if (error.empty()) error = "Vulkan frame observer completion failed";
            return false;
        }
        m_completion_pending[m_current_frame] = false;
    }
    if (m_timestamp_pool && m_timestamp_submitted[m_current_frame])
    {
        uint64_t ticks[2]{};
        if (m_vk.get_query_pool_results(m_device, m_timestamp_pool, 2 * m_current_frame, 2,
                sizeof(ticks), ticks, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
        {
            const uint64_t mask = m_timestamp_valid_bits == 64 ? UINT64_MAX :
                ((uint64_t{1} << m_timestamp_valid_bits) - 1);
            const uint64_t elapsed = (ticks[1] - ticks[0]) & mask;
            m_last_gpu_ms = static_cast<double>(elapsed) * m_timestamp_period_ns / 1000000.0;
        }
        m_timestamp_submitted[m_current_frame] = false;
    }

    crash_stage("vulkan frame: acquire image");
    const auto acquire_begin = Clock::now();
    uint32_t image_index = 0;
    const VkResult acquire = m_vk.acquire_next_image(m_device, m_swapchain, UINT64_MAX,
        m_image_available[m_current_frame], VK_NULL_HANDLE, &image_index);
    m_cpu_timing.acquire_ms = milliseconds(acquire_begin);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
    {
        status = FrameStatus::RecreateRequired;
        error.clear();
        return true;
    }
    if (mark_surface_lost(acquire, m_surface_lost, status))
    {
        error.clear();
        return true;
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
    {
        m_device_lost = acquire == VK_ERROR_DEVICE_LOST;
        error = "vkAcquireNextImageKHR failed";
        return false;
    }
    FrameSubmissionGuard submission_guard{m_submission_failed};
    if (image_index >= m_images.size())
    {
        error = "Vulkan returned an invalid swapchain image index";
        return false;
    }

    const auto image_fence_begin = Clock::now();
    VkFence image_fence = m_image_fences[image_index];
    if (image_fence && image_fence != frame_fence)
    {
        const VkResult image_fence_wait = m_vk.wait_for_fences(
            m_device, 1, &image_fence, VK_TRUE, UINT64_MAX);
        if (image_fence_wait != VK_SUCCESS)
        {
            m_device_lost = image_fence_wait == VK_ERROR_DEVICE_LOST;
            error = "could not wait for the previous Vulkan image submission";
            return false;
        }
    }
    m_cpu_timing.fence_ms += milliseconds(image_fence_begin);
    const auto record_begin = Clock::now();
    crash_stage("vulkan frame: begin command buffer");
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkCommandBuffer compute_command = VK_NULL_HANDLE;
    const bool record_async_compute = async_compute_requested && m_async_compute_enabled && prepass;
    const bool record_async_copy = prepass && m_async_copy_enabled;
    bool split_async_frame = record_async_compute;
    FrameSubmissionGraph async_submissions;
    FrameSubmissionGraph::Pass compute_submission_pass = FrameSubmissionGraph::InvalidPass;
    FrameSubmissionGraph::Pass graphics_prepass_pass = FrameSubmissionGraph::InvalidPass;
    FrameSubmissionGraph::Pass graphics_continuation_pass = FrameSubmissionGraph::InvalidPass;
    FrameSubmissionGraph::Pass final_graphics_pass = FrameSubmissionGraph::InvalidPass;
    std::vector<FrameSubmissionGraph::Pass> submission_order;
    std::vector<std::pair<FrameSubmissionGraph::Pass, FrameSubmissionGraph::Execute>> group_executors;
    std::vector<FrameSubmissionGraph::Pass> generated_group_passes;
    std::array<size_t, 3> generated_group_cursors{};
    bool generated_frame_graph_groups = false;
    bool acquire_wait_queued = false;
    const auto to_nvrhi_queue = [](FrameSubmissionGraph::Queue queue) {
        switch (queue)
        {
        case FrameSubmissionGraph::Queue::Graphics: return nvrhi::CommandQueue::Graphics;
        case FrameSubmissionGraph::Queue::Compute: return nvrhi::CommandQueue::Compute;
        case FrameSubmissionGraph::Queue::Copy: return nvrhi::CommandQueue::Copy;
        }
        return nvrhi::CommandQueue::Graphics;
    };
    const FrameSubmissionGraph::QueueWait queue_wait = [this, &to_nvrhi_queue](
        FrameSubmissionGraph::Queue destination, FrameSubmissionGraph::Queue source,
        uint64_t submission) {
        if (!m_nvrhi_device)
            throw std::runtime_error("NVRHI device is unavailable");
        m_nvrhi_device->queueWaitForCommandList(to_nvrhi_queue(destination),
            to_nvrhi_queue(source), submission);
    };
    const auto request_submission_command_buffer = [this, &async_submissions,
        &group_executors, &generated_group_passes, &generated_group_cursors,
        &submission_guard, &to_nvrhi_queue, &generated_frame_graph_groups,
        record_async_compute, record_async_copy, &compute_submission_pass,
        &graphics_prepass_pass](FrameSubmissionGraph::Pass pass,
            FrameSubmissionGraph::Queue queue, VkCommandBuffer& command_buffer,
            std::string& request_error) {
        if (!m_nvrhi_device || pass == FrameSubmissionGraph::InvalidPass ||
            async_submissions.queue(pass) != queue ||
            async_submissions.queue_family(pass) == FrameSubmissionGraph::InvalidQueueFamily)
        {
            request_error = "Vulkan frame graph: generated submission group is invalid";
            return false;
        }
        if (FrameSubmissionGraph::is_fixed_submission_pass(pass,
                compute_submission_pass, graphics_prepass_pass))
        {
            request_error = "Vulkan frame graph: generated group cannot replace a fixed frame submission";
            return false;
        }
        if ((queue == FrameSubmissionGraph::Queue::Compute && !record_async_compute) ||
            (queue == FrameSubmissionGraph::Queue::Copy && !record_async_copy))
        {
            request_error = "Vulkan frame graph: requested queue class is not enabled for this frame";
            return false;
        }
        const size_t queue_index = static_cast<size_t>(queue);
        if (queue_index >= generated_group_cursors.size())
        {
            request_error = "Vulkan frame graph: generated submission queue is invalid";
            return false;
        }
        if (std::find(generated_group_passes.begin(), generated_group_passes.end(), pass) !=
            generated_group_passes.end())
        {
            request_error = "Vulkan frame graph: generated submission group was registered twice";
            return false;
        }

        auto& lists = m_nvrhi_frame_graph_command_lists[m_current_frame][queue_index];
        const size_t list_index = generated_group_cursors[queue_index]++;
        try
        {
            if (list_index == lists.size())
            {
                nvrhi::CommandListParameters parameters;
                parameters.enableImmediateExecution = false;
                parameters.queueType = to_nvrhi_queue(queue);
                nvrhi::CommandListHandle created = m_nvrhi_device->createCommandList(parameters);
                if (!created)
                {
                    request_error = "NVRHI could not create a generated frame graph command list";
                    return false;
                }
                lists.push_back(std::move(created));
            }
            nvrhi::ICommandList* list = lists[list_index].Get();
            if (!list)
            {
                request_error = "NVRHI could not create a generated frame graph command list";
                return false;
            }
            submission_guard.track_generated(list);
            list->open();
            command_buffer = nvrhi_command_buffer(list);
            if (!command_buffer)
            {
                list->close();
                submission_guard.mark_generated_closed(list);
                request_error = "NVRHI generated command list has no Vulkan command buffer";
                return false;
            }
            group_executors.emplace_back(pass,
                [this, &submission_guard, &to_nvrhi_queue, list](FrameSubmissionGraph::Queue submitted_queue) {
                    list->close();
                    submission_guard.mark_generated_closed(list);
                    return m_nvrhi_device->executeCommandList(list, to_nvrhi_queue(submitted_queue));
                });
            generated_group_passes.push_back(pass);
            generated_frame_graph_groups = true;
        }
        catch (const std::exception& exception)
        {
            request_error = std::string("NVRHI generated command-list recording failed: ") + exception.what();
            return false;
        }
        request_error.clear();
        return true;
    };
    if (record_async_compute || record_async_copy)
    {
        if (record_async_compute)
            compute_submission_pass = async_submissions.add_pass("pre-geometry compute",
                FrameSubmissionGraph::Queue::Compute, m_compute_queue_family);
        graphics_prepass_pass = async_submissions.add_pass("graphics shadow prepass",
            FrameSubmissionGraph::Queue::Graphics, m_queue_family);
        if (record_async_compute || record_async_copy)
            graphics_continuation_pass = async_submissions.add_pass("graphics frame continuation",
                FrameSubmissionGraph::Queue::Graphics, m_queue_family);
        if (record_async_compute)
            final_graphics_pass = graphics_continuation_pass;
        else if (record_async_copy)
            final_graphics_pass = graphics_continuation_pass;
        if ((record_async_compute && m_last_graphics_submission &&
                !async_submissions.depends_on_submission(compute_submission_pass,
                    FrameSubmissionGraph::Queue::Graphics, m_last_graphics_submission, error)) ||
            (record_async_compute &&
                (!async_submissions.depends_on(graphics_continuation_pass, compute_submission_pass, error) ||
                 !async_submissions.depends_on(graphics_continuation_pass, graphics_prepass_pass, error))))
            return false;
    }
    m_nvrhi_graphics_list_index = 0;
    if (m_nvrhi_device)
    {
        nvrhi::ICommandList* command_list = m_nvrhi_command_lists[m_current_frame][0].Get();
        if (!command_list)
        {
            error = "NVRHI frame command list is unavailable";
            return false;
        }
        submission_guard.command_list = command_list;
        submission_guard.command_list_open = true;
        try
        {
            command_list->open();
            command = nvrhi_command_buffer(command_list);
        }
        catch (const std::exception& exception)
        {
            error = std::string("NVRHI could not begin the frame command list: ") + exception.what();
            return false;
        }
        if (!command)
        {
            error = "NVRHI frame command list has no Vulkan command buffer";
            return false;
        }
        if (record_async_compute)
        {
            nvrhi::ICommandList* compute_list = m_nvrhi_compute_command_lists[m_current_frame].Get();
            if (!compute_list)
            {
                error = "NVRHI asynchronous compute command list is unavailable";
                return false;
            }
            submission_guard.compute_command_list = compute_list;
            submission_guard.compute_command_list_open = true;
            try
            {
                if (!async_submissions.prepare(compute_submission_pass, queue_wait, error))
                    return false;
                compute_list->open();
                compute_command = nvrhi_command_buffer(compute_list);
            }
            catch (const std::exception& exception)
            {
                error = std::string("NVRHI could not begin the compute command list: ") + exception.what();
                return false;
            }
            if (!compute_command)
            {
                error = "NVRHI compute command list has no Vulkan command buffer";
                return false;
            }
        }
    }
    else
    {
        command = m_commands[image_index];
        const VkResult reset_command = m_vk.reset_command_buffer(command, 0);
        if (reset_command != VK_SUCCESS)
        {
            m_device_lost = reset_command == VK_ERROR_DEVICE_LOST;
            error = "could not reset Vulkan command buffer";
            return false;
        }
        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        const VkResult begin_result = m_vk.begin_command_buffer(command, &begin_info);
        if (begin_result != VK_SUCCESS)
        {
            m_device_lost = begin_result == VK_ERROR_DEVICE_LOST;
            error = "vkBeginCommandBuffer failed";
            return false;
        }
    }
    if (m_timestamp_pool)
    {
        m_vk.cmd_reset_query_pool(command, m_timestamp_pool, 2 * m_current_frame, 2);
        m_vk.cmd_write_timestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            m_timestamp_pool, 2 * m_current_frame);
    }
    FrameRecordingContext callback_context{};
    callback_context.command_buffer = command;
    callback_context.extent = m_extent;
    callback_context.image_index = image_index;
    callback_context.frame_index = m_current_frame;
    callback_context.asynchronous_copy_available = record_async_copy;
    callback_context.command_queue_flags = m_graphics_queue_flags;
    callback_context.graphics_queue_flags = m_graphics_queue_flags;
    callback_context.compute_queue_flags = m_compute_queue_flags;
    callback_context.copy_queue_flags = m_copy_queue_flags;
    callback_context.semaphore_handoff = record_async_compute || record_async_copy;
    callback_context.previous_graphics_submission = m_last_graphics_submission;
    callback_context.graphics_queue_family = m_queue_family;
    callback_context.compute_queue_family = m_compute_queue_family;
    callback_context.copy_queue_family = m_copy_queue_family;
    callback_context.command_queue = FrameSubmissionGraph::Queue::Graphics;
    callback_context.cmd_pipeline_barrier = m_vk.cmd_pipeline_barrier;
    if (prepass)
    {
        crash_stage("vulkan frame: record prepass");
        FrameRecordingContext frame = callback_context;
        frame.compute_command_buffer = compute_command;
        if (record_async_compute || record_async_copy)
        {
            frame.submission_graph = &async_submissions;
            frame.compute_submission_pass = compute_submission_pass;
            frame.graphics_prepass_pass = graphics_prepass_pass;
            frame.graphics_continuation_pass = graphics_continuation_pass;
            frame.previous_graphics_submission = m_last_graphics_submission;
            frame.request_submission_command_buffer = request_submission_command_buffer;
        }
        prepass(frame, prepass_data);
        callback_context = frame;
    }
    if (generated_frame_graph_groups)
    {
        if (graphics_continuation_pass == FrameSubmissionGraph::InvalidPass)
        {
            graphics_continuation_pass = async_submissions.add_pass("graphics frame continuation",
                FrameSubmissionGraph::Queue::Graphics, m_queue_family);
            final_graphics_pass = graphics_continuation_pass;
            if (graphics_prepass_pass != FrameSubmissionGraph::InvalidPass &&
                !async_submissions.depends_on(graphics_continuation_pass,
                    graphics_prepass_pass, error))
                return false;
        }
        for (FrameSubmissionGraph::Pass pass : generated_group_passes)
        {
            if (async_submissions.queue(pass) == FrameSubmissionGraph::Queue::Graphics &&
                graphics_prepass_pass != FrameSubmissionGraph::InvalidPass &&
                !async_submissions.depends_on(pass, graphics_prepass_pass, error))
                return false;
            if (!async_submissions.depends_on(graphics_continuation_pass, pass, error))
                return false;
        }
    }
    split_async_frame = record_async_compute || generated_frame_graph_groups;
    if (split_async_frame)
    {
        std::vector<FrameSubmissionGraph::Pass> deferred_groups{graphics_continuation_pass};
        if (compute_submission_pass != FrameSubmissionGraph::InvalidPass)
            group_executors.emplace_back(compute_submission_pass,
                [this, &submission_guard, &to_nvrhi_queue](FrameSubmissionGraph::Queue queue) {
                    submission_guard.compute_command_list->close();
                    submission_guard.compute_command_list_open = false;
                    return m_nvrhi_device->executeCommandList(
                        submission_guard.compute_command_list, to_nvrhi_queue(queue));
                });
        if (graphics_prepass_pass != FrameSubmissionGraph::InvalidPass)
            group_executors.emplace_back(graphics_prepass_pass,
                [this, &submission_guard, &to_nvrhi_queue](FrameSubmissionGraph::Queue queue) {
                    submission_guard.command_list->close();
                    submission_guard.command_list_open = false;
                    return m_nvrhi_device->executeCommandList(
                        submission_guard.command_list, to_nvrhi_queue(queue));
                });
        // The prepass callback may add resource-derived queue dependencies.
        // Compute submission order only after those edges are present, then
        // submit the recorded command lists in that topological order.
        if (!async_submissions.execution_order(submission_order, error))
            return false;
        try
        {
            for (const FrameSubmissionGraph::Pass pass : submission_order)
            {
                if (std::find(deferred_groups.begin(), deferred_groups.end(), pass) != deferred_groups.end())
                    continue;
                const auto executor = std::find_if(group_executors.begin(), group_executors.end(),
                    [pass](const auto& group) { return group.first == pass; });
                if (executor == group_executors.end())
                {
                    error = "Vulkan frame submission graph pass has no recorded command group";
                    return false;
                }
                if (!async_submissions.submit(pass, executor->second, queue_wait, error))
                {
                    m_device_lost = true;
                    return false;
                }
            }
            // Install continuation waits after submitting its producers.
            // Independent graphics, compute, and copy work can overlap when
            // the graph contains no dependency between their submissions.
            if (!async_submissions.prepare(graphics_continuation_pass, queue_wait, error))
            {
                m_device_lost = true;
                return false;
            }
            m_nvrhi_graphics_list_index = 1;
            nvrhi::ICommandList* continuation = m_nvrhi_command_lists[m_current_frame][1].Get();
            if (!continuation)
            {
                error = "NVRHI graphics continuation command list is unavailable";
                return false;
            }
            submission_guard.command_list = continuation;
            submission_guard.command_list_open = true;
            continuation->open();
            command = nvrhi_command_buffer(continuation);
            if (!command)
            {
                error = "NVRHI graphics continuation has no Vulkan command buffer";
                return false;
            }
        }
        catch (const std::exception& exception)
        {
            m_device_lost = true;
            error = std::string("NVRHI asynchronous frame split failed: ") + exception.what();
            return false;
        }
    }
    const VkImage scene_image = m_scene_images.empty() ? m_images[image_index] : m_scene_images[image_index];
    const bool independent_composite = !m_composite_images.empty();
    const VkImage composite_image = independent_composite ? m_composite_images[image_index] : m_images[image_index];
    const ImageUse composite_final_use = m_allow_readback || independent_composite
        ? ImageUse::TransferSource : ImageUse::Present;
    ImageStateTracker submitted_states = m_post_image_states;
    VkClearValue clear_values[3]{};
    clear_values[0].color = clear;
    const uint32_t depth_clear_index = m_scene_samples == VK_SAMPLE_COUNT_1_BIT ? 1u : 2u;
    clear_values[depth_clear_index].depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo render_info{};
    render_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_info.renderPass = m_render_pass;
    render_info.framebuffer = m_framebuffers[image_index];
    render_info.renderArea.extent = m_extent;
    render_info.clearValueCount = m_depth_format != VK_FORMAT_UNDEFINED ?
        (m_scene_samples == VK_SAMPLE_COUNT_1_BIT ? 2u : 3u) : 1u;
    render_info.pClearValues = clear_values;
    const auto record_scene_pass = [&](VkCommandBuffer recording_command, std::string& pass_error) {
        m_vk.cmd_begin_render_pass(recording_command, &render_info, VK_SUBPASS_CONTENTS_INLINE);
        if (clear_target)
        {
            VkClearAttachment attachment{};
            attachment.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            attachment.colorAttachment = 0;
            attachment.clearValue.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
            VkClearRect rect{};
            rect.rect.extent = m_extent;
            rect.layerCount = 1;
            m_vk.cmd_clear_attachments(recording_command, 1, &attachment, 1, &rect);
        }
        if (recorder)
        {
            crash_stage("vulkan frame: record scene");
            const FrameRecordingContext frame = make_frame_recording_context(callback_context,
                recording_command, m_render_pass, m_framebuffers[image_index], m_extent, 0, {},
                FrameSubmissionGraph::Queue::Graphics);
            recorder(frame, user_data);
        }
        m_vk.cmd_end_render_pass(recording_command);
        pass_error.clear();
        return true;
    };
    bool scene_pass_recorded = false;
    bool scene_requires_generated_ownership_transfer = false;
    FrameImageGraph scene_graph(FrameSubmissionGraph::Queue::Graphics);
    FrameImageGraph::Resource scene_color{};
    FrameImageGraph::Resource multisampled_scene_color{};
    FrameImageGraph::Resource scene_depth{};
    const VkImageSubresourceRange scene_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    ImageStateTracker scene_states = submitted_states;
    std::string scene_graph_error;
    const ImageUse initial_scene_use = m_postprocess ? ImageUse::TransferSource : ImageUse::Present;
    const bool has_scene_depth = m_depth_format != VK_FORMAT_UNDEFINED;
    const bool borrowed_scene_depth = m_preserve_prepass_depth && m_scene_depth_attached;
    const VkImage scene_depth_image = has_scene_depth ?
        (borrowed_scene_depth ? m_scene_depth_images[image_index] :
            (m_depth_images.empty() ? VK_NULL_HANDLE : m_depth_images[image_index])) : VK_NULL_HANDLE;
    const VkImageSubresourceRange scene_depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    const ImageUse initial_scene_depth_use = borrowed_scene_depth ?
        ImageUse::DepthStencilReadOnly : ImageUse::DepthStencilAttachment;
    scene_pass_recorded = m_postprocess || scene_states.register_image(scene_image, scene_range,
        initial_scene_use, scene_graph_error);
    if (scene_pass_recorded && has_scene_depth)
    {
        scene_pass_recorded = scene_depth_image != VK_NULL_HANDLE;
        if (scene_pass_recorded && !scene_states.has_image(scene_depth_image))
            scene_pass_recorded = scene_states.register_image(scene_depth_image, scene_depth_range,
                initial_scene_depth_use, scene_graph_error);
        // The G-buffer render pass above is the external producer and ends with
        // the borrowed depth image in read-only layout on every acquired frame.
        if (scene_pass_recorded && borrowed_scene_depth)
            scene_pass_recorded = scene_states.set_render_pass_final_state(scene_depth_image,
                scene_depth_range, ImageUse::DepthStencilReadOnly, scene_graph_error);
    }
    if (scene_pass_recorded)
    {
        scene_pass_recorded = scene_graph.import_image(scene_image, scene_range, scene_color,
                scene_graph_error, FrameSubmissionGraph::Queue::Graphics, 0,
                VK_SHARING_MODE_EXCLUSIVE, {}, initial_scene_use);
        if (scene_pass_recorded && has_scene_depth)
            scene_pass_recorded = scene_graph.import_image(scene_depth_image, scene_depth_range,
                scene_depth, scene_graph_error, FrameSubmissionGraph::Queue::Graphics, 0,
                borrowed_scene_depth ? m_frame_image_sharing.mode : VK_SHARING_MODE_EXCLUSIVE,
                borrowed_scene_depth ? m_frame_image_sharing.queue_families : std::vector<uint32_t>{},
                initial_scene_depth_use);
        if (scene_pass_recorded && !m_multisampled_scene_images.empty())
            scene_pass_recorded = scene_graph.import_image(m_multisampled_scene_images[image_index],
                scene_range, multisampled_scene_color, scene_graph_error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
                ImageUse::ColorAttachment);
    }
    const auto scene_pass = scene_pass_recorded ? scene_graph.add_checked_command_pass(
        "scene color and depth", FrameSubmissionGraph::Queue::Graphics,
        [&](VkCommandBuffer recording_command, std::string& pass_error) {
            return record_scene_pass(recording_command, pass_error);
        }) : ~0u;
    FrameImageGraph::Resource rendered_scene{};
    FrameImageGraph::Resource rendered_multisampled_scene{};
    FrameImageGraph::Resource rendered_scene_depth{};
    if (scene_pass_recorded && !m_multisampled_scene_images.empty())
        scene_pass_recorded = scene_graph.write_render_pass(scene_pass, multisampled_scene_color,
            ImageUse::ColorAttachment, ImageUse::ColorAttachment,
            rendered_multisampled_scene, scene_graph_error, ImageUse::ColorAttachment);
    if (scene_pass_recorded && has_scene_depth)
        scene_pass_recorded = scene_graph.write_render_pass(scene_pass, scene_depth,
            ImageUse::DepthStencilAttachment, ImageUse::DepthStencilAttachment,
            rendered_scene_depth, scene_graph_error, initial_scene_depth_use);
    if (scene_pass_recorded)
        scene_pass_recorded = scene_graph.write_render_pass(scene_pass, scene_color,
            ImageUse::ColorAttachment,
            m_allow_readback ? ImageUse::TransferSource : ImageUse::Present,
            rendered_scene, scene_graph_error);
    const FrameSampledImageSink import_scene_input = [&](const FrameSampledImage& input,
        std::string& input_error) {
        const bool has_producer = input.producer_pass != FrameSubmissionGraph::InvalidPass;
        FrameSubmissionGraph::Queue producer_queue = FrameSubmissionGraph::Queue::Graphics;
        const ImageUse producer_use = has_producer ? input.producer_use : input.use;
        if (has_producer)
        {
            if (input.producer_use == ImageUse::Undefined ||
                graphics_continuation_pass == FrameSubmissionGraph::InvalidPass)
            {
                input_error = "Vulkan scene graph: sampled input has incomplete producer metadata";
                return false;
            }
            producer_queue = async_submissions.queue(input.producer_pass);
            if (async_submissions.queue_family(input.producer_pass) ==
                FrameSubmissionGraph::InvalidQueueFamily)
            {
                input_error = "Vulkan scene graph: sampled input producer has no queue family";
                return false;
            }
            const uint32_t producer_family = async_submissions.queue_family(input.producer_pass);
            if (input.sharing.mode == VK_SHARING_MODE_EXCLUSIVE && producer_family != m_queue_family)
                scene_requires_generated_ownership_transfer = true;
            scene_states.forget_image(input.image);
        }
        if (!scene_states.has_image(input.image) &&
            !scene_states.register_image(input.image, input.range, producer_use, input_error))
            return false;
        FrameImageGraph::Resource resource{};
        const bool imported = scene_graph.import_image(input.image, input.range, resource, input_error,
                producer_queue, 0, input.sharing.mode,
                input.sharing.queue_families, producer_use, input.producer_pass) &&
            scene_graph.read(scene_pass, resource, input.use, input_error);
        return imported;
    };
    if (scene_pass_recorded && sampled_images)
        scene_pass_recorded = sampled_images(image_index, import_scene_input,
            sampled_images_data, scene_graph_error);
    if (scene_pass_recorded && graphics_continuation_pass != FrameSubmissionGraph::InvalidPass)
    {
        const auto queue_family_for_queue = [this](FrameSubmissionGraph::Queue queue) {
            switch (queue)
            {
            case FrameSubmissionGraph::Queue::Graphics: return m_queue_family;
            case FrameSubmissionGraph::Queue::Compute: return m_compute_queue_family;
            case FrameSubmissionGraph::Queue::Copy: return m_copy_queue_family;
            }
            return FrameSubmissionGraph::InvalidQueueFamily;
        };
        if (scene_requires_generated_ownership_transfer)
        {
            std::vector<FrameImageGraph::QueueGroup> scene_groups;
            std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
            FrameImageGraph::ExecutionState execution;
            if (!scene_graph.generate_submission_groups(async_submissions, queue_family_for_queue,
                    scene_groups, submission_for_pass, scene_graph_error) ||
                !scene_graph.begin_execution(execution, scene_graph_error))
                scene_pass_recorded = false;

            const auto queue_flags_for_queue = [this](FrameSubmissionGraph::Queue queue) {
                switch (queue)
                {
                case FrameSubmissionGraph::Queue::Graphics: return m_graphics_queue_flags;
                case FrameSubmissionGraph::Queue::Compute: return m_compute_queue_flags;
                case FrameSubmissionGraph::Queue::Copy: return m_copy_queue_flags;
                }
                return VkQueueFlags{};
            };
            for (const FrameImageGraph::QueueGroup& group : scene_groups)
            {
                if (!scene_pass_recorded)
                    break;
                VkCommandBuffer recording_command = VK_NULL_HANDLE;
                if (!request_submission_command_buffer(group.submission, group.queue,
                        recording_command, scene_graph_error) ||
                    !scene_graph.execute_group(group, recording_command, scene_states, execution,
                        {m_vk.cmd_pipeline_barrier}, scene_graph_error,
                        queue_flags_for_queue(group.queue), callback_context.semaphore_handoff))
                    scene_pass_recorded = false;
            }
            for (const FrameImageGraph::QueueGroup& group : scene_groups)
            {
                if (!scene_pass_recorded)
                    break;
                const auto executor = std::find_if(group_executors.begin(), group_executors.end(),
                    [&group](const auto& candidate) { return candidate.first == group.submission; });
                if (executor == group_executors.end() ||
                    !async_submissions.submit(group.submission, executor->second, queue_wait,
                        scene_graph_error))
                    scene_pass_recorded = false;
            }
            if (scene_pass_recorded &&
                (scene_pass >= submission_for_pass.size() ||
                    submission_for_pass[scene_pass] == FrameSubmissionGraph::InvalidPass ||
                    !async_submissions.depends_on(graphics_continuation_pass,
                        submission_for_pass[scene_pass], scene_graph_error) ||
                    !async_submissions.prepare(graphics_continuation_pass, queue_wait,
                        scene_graph_error)))
                scene_pass_recorded = false;
        }
        else
        {
            const auto submission_for_pass = [&](FrameImageGraph::Pass pass) {
                return pass == scene_pass ? graphics_continuation_pass : FrameSubmissionGraph::InvalidPass;
            };
            scene_pass_recorded = scene_graph.connect_submission_dependencies(async_submissions,
                submission_for_pass, queue_family_for_queue, scene_graph_error);
            if (scene_pass_recorded)
                scene_pass_recorded = async_submissions.prepare(graphics_continuation_pass,
                    queue_wait, scene_graph_error);
        }
    }
    if (scene_pass_recorded && !scene_requires_generated_ownership_transfer)
        scene_pass_recorded = scene_graph.execute(command, scene_states,
            {m_vk.cmd_pipeline_barrier}, scene_graph_error, m_graphics_queue_flags,
            callback_context.semaphore_handoff, FrameSubmissionGraph::Queue::Graphics);
    if (!scene_pass_recorded)
    {
        error = "Vulkan scene graph recording failed";
        if (!scene_graph_error.empty())
            error += ": " + scene_graph_error;
        return false;
    }
    if (interpass || refraction_capture || final_capture)
    {
        crash_stage("vulkan frame: record overlay");
        FrameImageGraph overlay_graph(FrameSubmissionGraph::Queue::Graphics);
        FrameImageGraph::Resource overlay_scene{};
        FrameImageGraph::Resource overlay_depth{};
        FrameImageGraph::Resource overlay_multisampled_scene{};
        ImageStateTracker overlay_states = scene_states;
        std::string overlay_error;
        bool overlay_graph_ready = has_scene_depth && m_overlay_pass && m_allow_readback;
        if (!overlay_graph_ready)
            overlay_error = "scene overlays require readable color and depth targets";
        if (overlay_graph_ready)
            overlay_graph_ready = overlay_graph.import_image(scene_image, scene_range, overlay_scene,
                    overlay_error, FrameSubmissionGraph::Queue::Graphics, 0,
                    VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::TransferSource) &&
                overlay_graph.import_image(scene_depth_image, scene_depth_range, overlay_depth,
                    overlay_error, FrameSubmissionGraph::Queue::Graphics, 0,
                    borrowed_scene_depth ? m_frame_image_sharing.mode : VK_SHARING_MODE_EXCLUSIVE,
                    borrowed_scene_depth ? m_frame_image_sharing.queue_families : std::vector<uint32_t>{},
                    ImageUse::DepthStencilAttachment);
        if (overlay_graph_ready && !m_multisampled_scene_images.empty())
            overlay_graph_ready = overlay_graph.import_image(
                m_multisampled_scene_images[image_index], scene_range, overlay_multisampled_scene,
                overlay_error, FrameSubmissionGraph::Queue::Graphics, 0,
                VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::ColorAttachment);

        const auto add_capture_pass = [&](const char* name, FrameInterpassRecorder capture,
            void* capture_data) {
            const auto pass = overlay_graph.add_checked_command_pass(name,
                FrameSubmissionGraph::Queue::Graphics,
                [&, capture, capture_data](VkCommandBuffer recording_command, std::string& pass_error) {
                    return capture(recording_command, scene_image, image_index, capture_data,
                        pass_error);
                });
            return overlay_graph_ready && overlay_graph.read(pass, overlay_scene,
                ImageUse::TransferSource, overlay_error);
        };
        const auto add_overlay_pass = [&](const char* name, FrameInterpassRecorder capture,
            void* capture_data, FrameRecorder draw, void* draw_data) {
            const auto pass = overlay_graph.add_checked_command_pass(name,
                FrameSubmissionGraph::Queue::Graphics,
                [&, capture, capture_data, draw, draw_data](VkCommandBuffer recording_command,
                    std::string& pass_error) {
                    if (capture)
                        if (!capture(recording_command, scene_image, image_index, capture_data,
                                pass_error))
                            return false;
                    VkRenderPassBeginInfo overlay_info = render_info;
                    overlay_info.renderPass = m_overlay_pass;
                    m_vk.cmd_begin_render_pass(recording_command, &overlay_info,
                        VK_SUBPASS_CONTENTS_INLINE);
                    const FrameRecordingContext overlay_frame = make_frame_recording_context(
                        callback_context, recording_command, m_overlay_pass, m_framebuffers[image_index],
                        m_extent, 0, {}, FrameSubmissionGraph::Queue::Graphics);
                    draw(overlay_frame, draw_data);
                    m_vk.cmd_end_render_pass(recording_command);
                    pass_error.clear();
                    return true;
                });
            FrameImageGraph::Resource next_scene{};
            FrameImageGraph::Resource next_depth{};
            FrameImageGraph::Resource next_multisampled_scene{};
            bool ready = overlay_graph_ready && overlay_graph.write_render_pass(pass, overlay_scene,
                ImageUse::ColorAttachment, ImageUse::TransferSource, next_scene, overlay_error,
                ImageUse::TransferSource) &&
                overlay_graph.write_render_pass(pass, overlay_depth,
                    ImageUse::DepthStencilAttachment, ImageUse::DepthStencilAttachment,
                    next_depth, overlay_error, ImageUse::DepthStencilAttachment);
            if (ready && !m_multisampled_scene_images.empty())
                ready = overlay_graph.write_render_pass(pass, overlay_multisampled_scene,
                    ImageUse::ColorAttachment, ImageUse::ColorAttachment,
                    next_multisampled_scene, overlay_error, ImageUse::ColorAttachment);
            if (ready)
            {
                overlay_scene = next_scene;
                overlay_depth = next_depth;
                if (!m_multisampled_scene_images.empty())
                    overlay_multisampled_scene = next_multisampled_scene;
            }
            return ready;
        };

        if (overlay_graph_ready && interpass)
            overlay_graph_ready = add_capture_pass("water scene capture", interpass, interpass_data) &&
                add_overlay_pass("water scene overlay", nullptr, nullptr, overlay, overlay_data);
        if (overlay_graph_ready && refraction_capture)
            overlay_graph_ready = add_overlay_pass("refraction capture and overlay",
                refraction_capture, refraction_capture_data, refraction_overlay,
                refraction_overlay_data);
        if (overlay_graph_ready && final_capture)
            overlay_graph_ready = add_capture_pass("final scene capture", final_capture,
                    final_capture_data) &&
                add_overlay_pass("final scene overlay", nullptr, nullptr,
                    final_overlay, final_overlay_data);
        if (overlay_graph_ready)
            overlay_graph_ready = overlay_graph.execute(command, overlay_states,
                {m_vk.cmd_pipeline_barrier}, overlay_error, m_graphics_queue_flags,
                record_async_compute, FrameSubmissionGraph::Queue::Graphics);
        if (!overlay_graph_ready)
        {
            error = "Vulkan scene overlay graph recording failed";
            if (!overlay_error.empty())
                error += ": " + overlay_error;
            return false;
        }
        scene_states = std::move(overlay_states);
    }
    submitted_states = std::move(scene_states);
    if (m_postprocess)
    {
        crash_stage("vulkan frame: record postprocess");
        if (!compositor)
        {
            error = "Vulkan postprocess compositor was not provided";
            return false;
        }
        const VkImage post_image = m_post_images[image_index];
        const VkImage bloom_image = m_bloom_images[image_index];
        const ImageStateDispatch state_dispatch{m_vk.cmd_pipeline_barrier};
        const auto mip_range = [](uint32_t mip) {
            return VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
        };
        FrameImageGraph graph;
        FrameImageGraph::Resource scene_color{};
        const VkImageSubresourceRange scene_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const auto import_compute_shared_image = [&](VkImage image,
            const VkImageSubresourceRange& range, FrameImageGraph::Resource& resource) {
            return graph.import_image(image, range, resource, error,
                FrameSubmissionGraph::Queue::Graphics, 0,
                m_frame_image_sharing.mode, m_frame_image_sharing.queue_families);
        };
        if (!graph.import_image(scene_image, scene_range, scene_color, error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
                ImageUse::TransferSource))
            return false;
        FrameImageGraph::Resource presentation_color = scene_color;
        if (composite_image != scene_image &&
            !graph.import_image(composite_image, scene_range, presentation_color, error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
                independent_composite ? ImageUse::Undefined : ImageUse::Present))
            return false;
        std::vector<FrameImageGraph::Resource> mips;
        std::vector<FrameImageGraph::Resource> bloom_mips;
        mips.reserve(m_post_mip_levels);
        bloom_mips.reserve(m_bloom_mip_levels);
        for (uint32_t mip = 0; mip < m_post_mip_levels; ++mip)
        {
            FrameImageGraph::Resource resource;
            if (!import_compute_shared_image(post_image, mip_range(mip), resource))
                return false;
            mips.push_back(resource);
        }
        for (uint32_t mip = 0; mip < m_bloom_mip_levels; ++mip)
        {
            FrameImageGraph::Resource resource;
            if (!graph.import_image(bloom_image, mip_range(mip), resource, error))
                return false;
            bloom_mips.push_back(resource);
        }
        FrameImageGraph::Resource exposure{};
        FrameImageGraph::Resource sampled_depth{};
        FrameImageGraph::Resource postprocess_motion{};
        const bool has_postprocess_motion = !m_post_motion_images.empty();
        if (has_postprocess_motion &&
            !import_compute_shared_image(m_post_motion_images[image_index], scene_range, postprocess_motion))
            return false;
        if (!m_post_depth_images.empty())
        {
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            if (!import_compute_shared_image(m_post_depth_images[image_index], range, sampled_depth))
                return false;
        }
        if (m_exposure_recorder)
        {
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (!import_compute_shared_image(m_exposure_image, range, exposure))
                return false;
        }
        const auto copy = graph.add_command_pass("postprocess scene copy", [&](VkCommandBuffer recording_command) {
            VkImageCopy region{};
            region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1;
            region.extent = {m_extent.width, m_extent.height, 1};
            m_vk.cmd_copy_image(recording_command, scene_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                post_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        });
        if (!graph.read(copy, scene_color, ImageUse::TransferSource, error) ||
            !graph.write(copy, mips[0], ImageUse::TransferDestination, mips[0], error))
            return false;
        FrameImageGraph::Resource temporal_output{};
        bool temporal_output_valid = false;
        FrameImageGraph::Resource fsr3_output{};
        bool fsr3_output_valid = false;
        if (m_fsr3_recorder && m_fsr3_active(m_fsr3_data))
        {
            FrameImageGraph::Resource output_image, motion = postprocess_motion;
            const VkImageSubresourceRange output_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (!import_compute_shared_image(m_fsr3_output_images[image_index], output_range, output_image) ||
                (!has_postprocess_motion &&
                    !import_compute_shared_image(m_fsr3_motion_images[image_index], scene_range, motion)))
                return false;
            const auto upscale_queue = record_async_compute ? FrameSubmissionGraph::Queue::Compute :
                FrameSubmissionGraph::Queue::Graphics;
            const auto upscale = graph.add_checked_command_pass("FSR3 temporal upscale", upscale_queue,
                [&](VkCommandBuffer recording_command, std::string& record_error) {
                const FrameRecordingContext context = make_frame_recording_context(callback_context,
                    recording_command, VK_NULL_HANDLE, VK_NULL_HANDLE, m_extent, 0, {}, upscale_queue);
                return m_fsr3_recorder(context, m_fsr3_data, record_error);
            });
            if (!graph.read(upscale, mips[0], ImageUse::ComputeSampled, error) ||
                !graph.read(upscale, motion, ImageUse::ComputeSampled, error) ||
                !graph.read(upscale, sampled_depth, ImageUse::ComputeSampled, error) ||
                !graph.write(upscale, output_image, ImageUse::StorageWrite, fsr3_output, error))
                return false;
            fsr3_output_valid = true;
        }
        if (m_temporal_recorder && m_temporal_active(m_temporal_data))
        {
            const uint32_t output_slot = m_temporal_output_slot(m_temporal_data);
            if (output_slot > 1)
            {
                error = "temporal resolve returned an invalid history slot";
                return false;
            }
            FrameImageGraph::Resource previous, output, motion = postprocess_motion;
            if (!import_compute_shared_image(m_temporal_images[output_slot ^ 1], scene_range, previous) ||
                !import_compute_shared_image(m_temporal_images[output_slot], scene_range, output) ||
                (!has_postprocess_motion &&
                    !import_compute_shared_image(m_temporal_motion_images[image_index], scene_range, motion)))
                return false;
            const auto resolve_queue = record_async_compute ? FrameSubmissionGraph::Queue::Compute :
                FrameSubmissionGraph::Queue::Graphics;
            const auto resolve = graph.add_command_pass("temporal color resolve", resolve_queue,
                [&](VkCommandBuffer recording_command) {
                const FrameRecordingContext context = make_frame_recording_context(callback_context,
                    recording_command, VK_NULL_HANDLE, VK_NULL_HANDLE, m_extent, 0, {}, resolve_queue);
                m_temporal_recorder(context, m_temporal_data);
            });
            if (!graph.read(resolve, mips[0], ImageUse::ComputeSampled, error) ||
                !graph.read(resolve, motion, ImageUse::ComputeSampled, error) ||
                !graph.read(resolve, sampled_depth, ImageUse::ComputeSampled, error) ||
                !graph.read(resolve, previous, ImageUse::ComputeSampled, error) ||
                !graph.write(resolve, output, ImageUse::StorageWrite, output, error))
                return false;
            const auto copy_resolved = graph.add_command_pass("temporal color copy",
                [&, output_slot](VkCommandBuffer recording_command) {
                VkImageBlit region{};
                region.srcSubresource.aspectMask = region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.srcSubresource.layerCount = region.dstSubresource.layerCount = 1;
                region.srcOffsets[1] = {int32_t(m_temporal_extent.width), int32_t(m_temporal_extent.height), 1};
                region.dstOffsets[1] = {int32_t(m_extent.width), int32_t(m_extent.height), 1};
                m_vk.cmd_blit_image(recording_command, m_temporal_images[output_slot], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    post_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_NEAREST);
            });
            if (!graph.read(copy_resolved, output, ImageUse::TransferSource, error) ||
                !graph.write(copy_resolved, mips[0], ImageUse::TransferDestination, mips[0], error))
                return false;
            temporal_output = output;
            temporal_output_valid = m_temporal_extent.width != m_extent.width || m_temporal_extent.height != m_extent.height;
        }
        uint32_t width = m_extent.width, height = m_extent.height;
        for (uint32_t mip = 1; mip < m_post_mip_levels; ++mip)
        {
            const uint32_t next_width = std::max(1u, width / 2), next_height = std::max(1u, height / 2);
            const auto blit_pass = graph.add_command_pass("postprocess mip blit",
                [&, mip, width, height, next_width, next_height](VkCommandBuffer recording_command) {
                VkImageBlit blit{};
                blit.srcSubresource.aspectMask = blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blit.srcSubresource.mipLevel = mip - 1;
                blit.dstSubresource.mipLevel = mip;
                blit.srcSubresource.layerCount = blit.dstSubresource.layerCount = 1;
                blit.srcOffsets[1] = {int32_t(width), int32_t(height), 1};
                blit.dstOffsets[1] = {int32_t(next_width), int32_t(next_height), 1};
                m_vk.cmd_blit_image(recording_command, post_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    post_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            });
            if (!graph.read(blit_pass, mips[mip - 1], ImageUse::TransferSource, error) ||
                !graph.write(blit_pass, mips[mip], ImageUse::TransferDestination, mips[mip], error))
                return false;
            width = next_width;
            height = next_height;
        }
        if (m_exposure_recorder)
        {
            const auto adapt_queue = record_async_compute ? FrameSubmissionGraph::Queue::Compute :
                FrameSubmissionGraph::Queue::Graphics;
            const auto adapt = graph.add_command_pass("temporal exposure", adapt_queue,
                [&](VkCommandBuffer recording_command) {
                const FrameRecordingContext context = make_frame_recording_context(callback_context,
                    recording_command, VK_NULL_HANDLE, VK_NULL_HANDLE, m_extent, 0, {}, adapt_queue);
                m_exposure_recorder(context, m_exposure_data);
            });
            if (!graph.read(adapt, mips[0], ImageUse::ComputeSampled, error))
                return false;
            if (m_exposure_history_valid(m_exposure_data))
            {
                if (!graph.read_write(adapt, exposure, exposure, error, ImageUse::StorageReadWrite))
                    return false;
            }
            else if (!graph.write(adapt, exposure, ImageUse::StorageWrite, exposure, error))
                return false;
        }
        const auto bright_pass = graph.add_command_pass("bloom bright pass",
            [&](VkCommandBuffer recording_command) {
            VkClearValue clear{};
            VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            info.renderPass = m_bloom_pass;
            info.framebuffer = m_bloom_framebuffers[image_index];
            info.renderArea.extent = m_bloom_extent;
            info.clearValueCount = 1;
            info.pClearValues = &clear;
            m_vk.cmd_begin_render_pass(recording_command, &info, VK_SUBPASS_CONTENTS_INLINE);
            const FrameRecordingContext context = make_frame_recording_context(callback_context,
                recording_command, m_bloom_pass, m_bloom_framebuffers[image_index], m_bloom_extent,
                0, {}, FrameSubmissionGraph::Queue::Graphics);
            compositor(context, compositor_data);
            m_vk.cmd_end_render_pass(recording_command);
        });
        if (!graph.read(bright_pass, mips[0], ImageUse::Sampled, error) ||
            !graph.write(bright_pass, bloom_mips[0], ImageUse::ColorAttachment,
                bloom_mips[0], error))
            return false;
        width = m_bloom_extent.width;
        height = m_bloom_extent.height;
        for (uint32_t mip = 1; mip < m_bloom_mip_levels; ++mip)
        {
            const uint32_t next_width = std::max(1u, width / 2), next_height = std::max(1u, height / 2);
            const auto blit_pass = graph.add_command_pass("bloom mip blit",
                [&, mip, width, height, next_width, next_height](VkCommandBuffer recording_command) {
                VkImageBlit blit{};
                blit.srcSubresource.aspectMask = blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                blit.srcSubresource.mipLevel = mip - 1;
                blit.dstSubresource.mipLevel = mip;
                blit.srcSubresource.layerCount = blit.dstSubresource.layerCount = 1;
                blit.srcOffsets[1] = {int32_t(width), int32_t(height), 1};
                blit.dstOffsets[1] = {int32_t(next_width), int32_t(next_height), 1};
                m_vk.cmd_blit_image(recording_command, bloom_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    bloom_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
            });
            if (!graph.read(blit_pass, bloom_mips[mip - 1], ImageUse::TransferSource, error) ||
                !graph.write(blit_pass, bloom_mips[mip], ImageUse::TransferDestination,
                    bloom_mips[mip], error))
                return false;
            width = next_width;
            height = next_height;
        }
        for (uint32_t level = m_bloom_mip_levels; level > 1; --level)
        {
            const uint32_t target_mip = level - 2;
            const VkExtent2D target_extent{
                std::max(1u, m_bloom_extent.width >> target_mip),
                std::max(1u, m_bloom_extent.height >> target_mip)};
            const auto upsample_pass = graph.add_command_pass("bloom upsample",
                [&, target_mip, target_extent](VkCommandBuffer recording_command) {
                VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                info.renderPass = m_bloom_upsample_pass;
                info.framebuffer = m_bloom_upsample_framebuffers[image_index][target_mip];
                info.renderArea.extent = target_extent;
                m_vk.cmd_begin_render_pass(recording_command, &info, VK_SUBPASS_CONTENTS_INLINE);
                const FrameRecordingContext context = make_frame_recording_context(callback_context,
                    recording_command, m_bloom_upsample_pass, info.framebuffer, target_extent,
                    target_mip, {}, FrameSubmissionGraph::Queue::Graphics);
                compositor(context, compositor_data);
                m_vk.cmd_end_render_pass(recording_command);
            });
            if (!graph.read(upsample_pass, bloom_mips[target_mip + 1], ImageUse::Sampled, error) ||
                !graph.read_write(upsample_pass, bloom_mips[target_mip], bloom_mips[target_mip],
                    error, ImageUse::ColorAttachment))
                return false;
        }
        const auto composite_pass = graph.add_command_pass("postprocess composite",
            [&](VkCommandBuffer recording_command) {
            VkRenderPassBeginInfo composite{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            composite.renderPass = m_composite_pass;
            composite.framebuffer = m_composite_framebuffers[image_index];
            composite.renderArea.extent = m_presentation_extent;
            m_vk.cmd_begin_render_pass(recording_command, &composite, VK_SUBPASS_CONTENTS_INLINE);
            const FrameRecordingContext context = make_frame_recording_context(callback_context,
                recording_command, m_composite_pass, m_composite_framebuffers[image_index],
                m_presentation_extent, 0, {}, FrameSubmissionGraph::Queue::Graphics);
            compositor(context, compositor_data);
            m_vk.cmd_end_render_pass(recording_command);
        });
        for (const auto mip : mips)
            if (!graph.read(composite_pass, mip, ImageUse::Sampled, error))
                return false;
        for (const auto mip : bloom_mips)
            if (!graph.read(composite_pass, mip, ImageUse::Sampled, error))
                return false;
        FrameImageGraph::Resource composited_color{};
        if (!graph.write_render_pass(composite_pass, presentation_color,
                ImageUse::ColorAttachment,
                composite_final_use,
                composited_color, error,
                composite_image == scene_image ? ImageUse::TransferSource : ImageUse::ColorAttachment))
            return false;
        if (temporal_output_valid && !graph.read(composite_pass, temporal_output, ImageUse::Sampled, error))
            return false;
        if (fsr3_output_valid && !graph.read(composite_pass, fsr3_output, ImageUse::Sampled, error))
            return false;
        if (m_exposure_recorder && !graph.read(composite_pass, exposure, ImageUse::Sampled, error))
            return false;
        if (!m_post_depth_images.empty() &&
            !graph.read(composite_pass, sampled_depth, ImageUse::Sampled, error))
            return false;
        if (has_postprocess_motion &&
            !graph.read(composite_pass, postprocess_motion, ImageUse::Sampled, error))
            return false;
        std::vector<FrameImageGraph::QueueGroup> postprocess_groups;
        FrameImageGraph::ExecutionState graph_execution;
        if (!graph.build_queue_groups(postprocess_groups, error) ||
            !graph.begin_execution(graph_execution, error))
            return false;
        const bool has_compute_group = std::any_of(postprocess_groups.begin(), postprocess_groups.end(),
            [](const FrameImageGraph::QueueGroup& group) {
                return group.queue == FrameSubmissionGraph::Queue::Compute;
            });
        if (has_compute_group)
        {
            if (!record_async_compute || !split_async_frame || !m_nvrhi_device ||
                graphics_continuation_pass == FrameSubmissionGraph::InvalidPass ||
                m_nvrhi_graphics_list_index != 1)
            {
                error = "Vulkan postprocess compute groups require an asynchronous frame continuation";
                return false;
            }

            const auto queue_family_for_queue = [this](FrameSubmissionGraph::Queue queue) {
                switch (queue)
                {
                case FrameSubmissionGraph::Queue::Graphics: return m_queue_family;
                case FrameSubmissionGraph::Queue::Compute: return m_compute_queue_family;
                case FrameSubmissionGraph::Queue::Copy: return m_copy_queue_family;
                }
                return FrameSubmissionGraph::InvalidQueueFamily;
            };
            std::vector<FrameSubmissionGraph::Pass> postprocess_passes;
            if (!graph.generate_submission_groups(async_submissions, queue_family_for_queue,
                    postprocess_groups, postprocess_passes, error))
                return false;
            if (std::find(postprocess_passes.begin(), postprocess_passes.end(),
                    FrameSubmissionGraph::InvalidPass) != postprocess_passes.end())
            {
                error = "Vulkan postprocess graph left a pass without a submission group";
                return false;
            }

            final_graphics_pass = async_submissions.add_pass("graphics presentation continuation",
                FrameSubmissionGraph::Queue::Graphics, m_queue_family);
            if (postprocess_groups.empty() ||
                !async_submissions.depends_on(postprocess_groups.front().submission,
                    graphics_continuation_pass, error))
            {
                if (error.empty())
                    error = "Vulkan postprocess graph has no initial graphics group";
                return false;
            }
            for (const FrameImageGraph::QueueGroup& group : postprocess_groups)
            {
                if (!async_submissions.depends_on(final_graphics_pass, group.submission, error))
                    return false;
            }

            // The current graphics list owns the rendered scene. Submit it
            // before the generated postprocess groups so their resource edges
            // can wait on this producer. The acquire semaphore must be consumed
            // here because a non-upscaled scene can render directly to the
            // acquired swapchain image.
            nvrhi::ICommandList* scene_list = submission_guard.command_list;
            try
            {
                scene_list->close();
                submission_guard.command_list_open = false;
                m_nvrhi_device->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics,
                    m_image_available[m_current_frame], 0);
                acquire_wait_queued = true;
            }
            catch (const std::exception& exception)
            {
                error = std::string("NVRHI could not close the scene continuation: ") + exception.what();
                return false;
            }
            if (!async_submissions.submit(graphics_continuation_pass,
                    [this, scene_list, &to_nvrhi_queue](FrameSubmissionGraph::Queue queue) {
                        return m_nvrhi_device->executeCommandList(scene_list, to_nvrhi_queue(queue));
                    }, queue_wait, error))
            {
                m_device_lost = true;
                return false;
            }

            for (const FrameImageGraph::QueueGroup& group : postprocess_groups)
            {
                VkCommandBuffer group_command = VK_NULL_HANDLE;
                if (!request_submission_command_buffer(group.submission, group.queue,
                        group_command, error))
                    return false;
                const VkQueueFlags group_flags = group.queue == FrameSubmissionGraph::Queue::Compute ?
                    m_compute_queue_flags : group.queue == FrameSubmissionGraph::Queue::Copy ?
                        m_copy_queue_flags : m_graphics_queue_flags;
                if (!graph.execute_group(group, group_command, submitted_states, graph_execution,
                        state_dispatch, error, group_flags, true))
                    return false;
            }
            for (const FrameImageGraph::QueueGroup& group : postprocess_groups)
            {
                const auto executor = std::find_if(group_executors.begin(), group_executors.end(),
                    [&group](const auto& candidate) { return candidate.first == group.submission; });
                if (executor == group_executors.end() ||
                    !async_submissions.submit(group.submission, executor->second, queue_wait, error))
                {
                    if (error.empty())
                        error = "Vulkan postprocess group has no NVRHI command-list executor";
                    m_device_lost = true;
                    return false;
                }
            }

            if (!async_submissions.prepare(final_graphics_pass, queue_wait, error))
            {
                m_device_lost = true;
                return false;
            }
            m_nvrhi_graphics_list_index = 2;
            nvrhi::ICommandList* continuation = m_nvrhi_command_lists[m_current_frame][2].Get();
            if (!continuation)
            {
                error = "NVRHI presentation continuation command list is unavailable";
                return false;
            }
            submission_guard.command_list = continuation;
            submission_guard.command_list_open = true;
            try
            {
                continuation->open();
                command = nvrhi_command_buffer(continuation);
            }
            catch (const std::exception& exception)
            {
                error = std::string("NVRHI could not begin the presentation continuation: ") + exception.what();
                return false;
            }
            if (!command)
            {
                error = "NVRHI presentation continuation has no Vulkan command buffer";
                return false;
            }
        }
        else
        {
            for (const FrameImageGraph::QueueGroup& group : postprocess_groups)
            {
                if (group.queue != FrameSubmissionGraph::Queue::Graphics)
                {
                    error = "Vulkan postprocess contains an unsubmitted non-graphics queue group";
                    return false;
                }
                if (!graph.execute_group(group, command, submitted_states, graph_execution,
                        state_dispatch, error, m_graphics_queue_flags))
                    return false;
            }
        }
    }
    if (presentation_overlay)
    {
        // The scene has already been scaled and postprocessed. When frame
        // generation is active, render 2D UI into a transparent native-size
        // layer so FidelityFX can composite it after interpolation.
        const VkFramebuffer overlay_framebuffer = separate_frame_generation_ui
            ? m_frame_generation_ui_framebuffers[image_index] : m_composite_framebuffers[image_index];
        const VkImage overlay_image = separate_frame_generation_ui
            ? m_frame_generation_ui_images[image_index] : composite_image;
        const VkImageSubresourceRange overlay_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        ImageStateTracker overlay_states = submitted_states;
        if (separate_frame_generation_ui &&
            !overlay_states.register_image(overlay_image, overlay_range, ImageUse::Undefined, error))
            return false;
        FrameImageGraph overlay_graph(FrameSubmissionGraph::Queue::Graphics);
        FrameImageGraph::Resource overlay_color{};
        const ImageUse overlay_initial_use = separate_frame_generation_ui ? ImageUse::Undefined : composite_final_use;
        if (!overlay_graph.import_image(overlay_image, overlay_range, overlay_color, error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
                overlay_initial_use))
            return false;
        FrameImageGraph::Resource overlay_input = overlay_color;
        if (separate_frame_generation_ui)
        {
            const auto clear_ui = overlay_graph.add_command_pass("frame-generation UI clear",
                [&](VkCommandBuffer recording_command) {
                VkClearValue transparent{};
                transparent.color = {{0.f, 0.f, 0.f, 0.f}};
                VkRenderPassBeginInfo clear_info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                clear_info.renderPass = m_ui_clear_pass;
                clear_info.framebuffer = overlay_framebuffer;
                clear_info.renderArea.extent = m_presentation_extent;
                clear_info.clearValueCount = 1;
                clear_info.pClearValues = &transparent;
                m_vk.cmd_begin_render_pass(recording_command, &clear_info, VK_SUBPASS_CONTENTS_INLINE);
                m_vk.cmd_end_render_pass(recording_command);
            });
            if (!overlay_graph.write_render_pass(clear_ui, overlay_color,
                    ImageUse::ColorAttachment, ImageUse::TransferSource,
                    overlay_input, error))
                return false;
        }
        const auto overlay_pass = overlay_graph.add_checked_command_pass("presentation UI overlay",
            FrameSubmissionGraph::Queue::Graphics,
            [&](VkCommandBuffer recording_command, std::string& pass_error) {
            VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            info.renderPass = m_presentation_pass;
            info.framebuffer = overlay_framebuffer;
            info.renderArea.extent = m_presentation_extent;
            m_vk.cmd_begin_render_pass(recording_command, &info, VK_SUBPASS_CONTENTS_INLINE);
            const FrameRecordingContext output = make_frame_recording_context(callback_context,
                recording_command, m_presentation_pass, overlay_framebuffer, m_presentation_extent,
                0, {}, FrameSubmissionGraph::Queue::Graphics);
            presentation_overlay(output, presentation_overlay_data);
            m_vk.cmd_end_render_pass(recording_command);
            pass_error.clear();
            return true;
        });
        FrameImageGraph::Resource overlay_output{};
        if (!overlay_graph.write_render_pass(overlay_pass, overlay_input,
                ImageUse::ColorAttachment,
                composite_final_use,
                overlay_output, error, ImageUse::TransferSource) ||
            !overlay_graph.execute(command, overlay_states, {m_vk.cmd_pipeline_barrier}, error,
                m_graphics_queue_flags, record_async_compute,
                FrameSubmissionGraph::Queue::Graphics))
        {
            if (error.empty())
                error = "Vulkan presentation UI graph recording failed";
            return false;
        }
        if (!separate_frame_generation_ui)
            submitted_states = std::move(overlay_states);
        if (presentation_capture || display_capture ||
            (m_frame_interpolation_swapchain && m_frame_generation_ui_register))
        {
            FrameImageGraph display_graph(FrameSubmissionGraph::Queue::Graphics);
            ImageStateTracker display_states = submitted_states;
            const VkImageSubresourceRange display_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            const ImageUse display_initial_use = composite_final_use;
            const ImageUse display_final_use = display_initial_use;
            FrameImageGraph::Resource scene_output{};
            std::string display_error;
            bool display_graph_ready = display_graph.import_image(composite_image,
                display_range, scene_output, display_error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
                display_initial_use);
            FrameImageGraph::Resource ui_output{};
            if (display_graph_ready && separate_frame_generation_ui)
            {
                const VkImage ui_image = m_frame_generation_ui_images[image_index];
                const bool ui_registered = display_states.has_image(ui_image) ||
                    display_states.register_image(ui_image, display_range, display_initial_use,
                        display_error);
                display_graph_ready = ui_registered && display_graph.import_image(ui_image,
                    display_range, ui_output, display_error,
                    FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
                    display_initial_use);
            }
            const auto add_capture = [&](const char* name, FrameInterpassRecorder capture,
                void* capture_data, VkImage source, FrameImageGraph::Resource image) {
                const auto pass = display_graph.add_checked_command_pass(name,
                    FrameSubmissionGraph::Queue::Graphics,
                    [&, capture, capture_data, source](VkCommandBuffer recording_command,
                        std::string& pass_error) {
                        return capture(recording_command, source, image_index, capture_data,
                            pass_error);
                    });
                return display_graph_ready && display_graph.read(pass, image,
                    ImageUse::TransferSource, display_error);
            };
            const auto add_grade = [&](const char* name, FrameRecorder grade, void* grade_data,
                VkFramebuffer framebuffer, FrameImageGraph::Resource& image) {
                const auto pass = display_graph.add_checked_command_pass(name,
                    FrameSubmissionGraph::Queue::Graphics,
                    [&, grade, grade_data, framebuffer](VkCommandBuffer recording_command,
                        std::string& pass_error) {
                        VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
                        info.renderPass = m_presentation_pass;
                        info.framebuffer = framebuffer;
                        info.renderArea.extent = m_presentation_extent;
                        m_vk.cmd_begin_render_pass(recording_command, &info,
                            VK_SUBPASS_CONTENTS_INLINE);
                        const FrameRecordingContext output = make_frame_recording_context(callback_context,
                            recording_command, m_presentation_pass, framebuffer, m_presentation_extent,
                            0, {}, FrameSubmissionGraph::Queue::Graphics);
                        grade(output, grade_data);
                        m_vk.cmd_end_render_pass(recording_command);
                        pass_error.clear();
                        return true;
                    });
                FrameImageGraph::Resource next{};
                if (!display_graph_ready || !display_graph.write_render_pass(pass, image,
                        ImageUse::ColorAttachment, display_final_use, next, display_error,
                        ImageUse::TransferSource))
                    return false;
                image = next;
                return true;
            };

            if (display_graph_ready && presentation_capture)
                display_graph_ready = add_capture("presentation distortion capture",
                        presentation_capture, presentation_capture_data,
                        composite_image, scene_output) &&
                    add_grade("presentation distortion overlay", presentation_distortion,
                        presentation_distortion_data, m_composite_framebuffers[image_index],
                        scene_output);
            if (display_graph_ready && display_capture && separate_frame_generation_ui)
                display_graph_ready = add_capture("frame-generation UI display capture",
                        display_capture, display_capture_data,
                        m_frame_generation_ui_images[image_index], ui_output) &&
                    add_grade("frame-generation UI display grade", display_grade,
                        display_grade_data, m_frame_generation_ui_framebuffers[image_index],
                        ui_output);
            if (display_graph_ready && display_capture)
                display_graph_ready = add_capture("scene display capture", display_capture,
                        display_capture_data, composite_image, scene_output) &&
                    add_grade("scene display grade", display_grade, display_grade_data,
                        m_composite_framebuffers[image_index], scene_output);
            if (display_graph_ready && m_frame_interpolation_swapchain && m_frame_generation_ui_register)
            {
                constexpr VkImageUsageFlags ui_usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
                const VkImage ui_image = separate_frame_generation_ui
                    ? m_frame_generation_ui_images[image_index] : VK_NULL_HANDLE;
                const auto register_ui = display_graph.add_checked_command_pass(
                    "frame-generation UI registration", FrameSubmissionGraph::Queue::Graphics,
                    [&, ui_image, ui_usage](VkCommandBuffer, std::string& pass_error) {
                        const bool registered = m_frame_generation_ui_register(ui_image,
                            ui_image ? m_composite_format : VK_FORMAT_UNDEFINED,
                            ui_image ? m_presentation_extent : VkExtent2D{},
                            ui_image ? ui_usage : 0, m_frame_generation_ui_data, pass_error);
                        if (!registered && pass_error.empty())
                            pass_error = "frame-generation UI registration failed";
                        return registered;
                    });
                if (separate_frame_generation_ui)
                    display_graph_ready = display_graph.read(register_ui, ui_output,
                        ImageUse::TransferSource, display_error);
            }
            if (display_graph_ready)
                display_graph_ready = display_graph.execute(command, display_states,
                    {m_vk.cmd_pipeline_barrier}, display_error, m_graphics_queue_flags,
                    record_async_compute, FrameSubmissionGraph::Queue::Graphics);
            if (!display_graph_ready)
            {
                error = "Vulkan presentation capture graph recording failed";
                if (!display_error.empty())
                    error += ": " + display_error;
                return false;
            }
            if (!separate_frame_generation_ui)
                submitted_states = std::move(display_states);
        }
    }
    if (independent_composite)
    {
        if (!m_display_output_recorder || !m_display_output_pass || image_index >= m_display_output_framebuffers.size())
        {
            error = "independent presentation requires a display encoding recorder";
            return false;
        }
        FrameImageGraph output_graph(FrameSubmissionGraph::Queue::Graphics);
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        FrameImageGraph::Resource linear, swapchain, encoded;
        if (!output_graph.import_image(composite_image, range, linear, error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {}, composite_final_use) ||
            !output_graph.import_image(m_images[image_index], range, swapchain, error,
                FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::Undefined))
            return false;
        const auto output = output_graph.add_command_pass("display encoding", [&](VkCommandBuffer cb) {
            VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            begin.renderPass = m_display_output_pass;
            begin.framebuffer = m_display_output_framebuffers[image_index];
            begin.renderArea.extent = m_presentation_extent;
            m_vk.cmd_begin_render_pass(cb, &begin, VK_SUBPASS_CONTENTS_INLINE);
            const auto recording = make_frame_recording_context(callback_context, cb, begin.renderPass, begin.framebuffer,
                m_presentation_extent, 0, {}, FrameSubmissionGraph::Queue::Graphics);
            m_display_output_recorder(recording, m_display_output_data);
            m_vk.cmd_end_render_pass(cb);
        });
        if (!output_graph.read(output, linear, ImageUse::Sampled, error) ||
            !output_graph.write_render_pass(output, swapchain, ImageUse::ColorAttachment, ImageUse::Present, encoded, error) ||
            !output_graph.execute(command, submitted_states, {m_vk.cmd_pipeline_barrier}, error,
                m_graphics_queue_flags, record_async_compute, FrameSubmissionGraph::Queue::Graphics))
            return false;
    }
    else if (m_allow_readback)
    {
        if (readback)
        {
            FrameImageGraph readback_graph(FrameSubmissionGraph::Queue::Graphics);
            FrameImageGraph::Resource readback_source{};
            const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            std::string readback_error;
            ImageStateTracker readback_states = submitted_states;
            if (!readback_graph.import_image(m_images[image_index], range, readback_source,
                    readback_error, FrameSubmissionGraph::Queue::Graphics, 0,
                    VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::TransferSource))
                return false;
            const auto readback_pass = readback_graph.add_checked_command_pass("frame readback",
                FrameSubmissionGraph::Queue::Graphics,
                [&](VkCommandBuffer recording_command, std::string& pass_error) {
                    readback(recording_command, m_images[image_index], m_presentation_extent,
                        readback_data);
                    pass_error.clear();
                    return true;
                });
            if (!readback_graph.read(readback_pass, readback_source, ImageUse::TransferSource,
                    readback_error) ||
                !readback_graph.execute(command, readback_states,
                    {m_vk.cmd_pipeline_barrier}, readback_error, m_graphics_queue_flags,
                    record_async_compute, FrameSubmissionGraph::Queue::Graphics))
            {
                error = "Vulkan frame readback graph recording failed";
                if (!readback_error.empty())
                    error += ": " + readback_error;
                return false;
            }
            submitted_states = std::move(readback_states);
        }
        FrameImageGraph presentation_graph(FrameSubmissionGraph::Queue::Graphics);
        ImageStateTracker presentation_states = submitted_states;
        FrameImageGraph::Resource presentation_image{};
        std::string presentation_error;
        const VkImageSubresourceRange presentation_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        const ImageUse presentation_initial_use = readback ? ImageUse::TransferSource :
            ImageUse::ColorAttachmentWriteTransferSource;
        bool presentation_ready = readback || presentation_states.set_render_pass_final_state(
            m_images[image_index], presentation_range, presentation_initial_use, presentation_error);
        presentation_ready = presentation_ready && presentation_graph.import_image(m_images[image_index],
            presentation_range, presentation_image, presentation_error,
            FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
            presentation_initial_use);
        const auto handoff_pass = presentation_graph.add_checked_command_pass(
            "swapchain presentation handoff", FrameSubmissionGraph::Queue::Graphics,
            [](VkCommandBuffer, std::string& pass_error) {
                pass_error.clear();
                return true;
            });
        presentation_ready = presentation_ready && presentation_graph.read(handoff_pass,
            presentation_image, m_frame_interpolation_swapchain ? ImageUse::Sampled :
                ImageUse::Present, presentation_error);
        if (presentation_ready)
            presentation_ready = presentation_graph.execute(command, presentation_states,
                {m_vk.cmd_pipeline_barrier}, presentation_error, m_graphics_queue_flags,
                record_async_compute, FrameSubmissionGraph::Queue::Graphics);
        if (!presentation_ready)
        {
            error = "Vulkan presentation handoff graph recording failed";
            if (!presentation_error.empty())
                error += ": " + presentation_error;
            return false;
        }
        submitted_states = std::move(presentation_states);
    }
    if (validate_recording && !validate_recording(validation_data, error))
    {
        if (error.empty())
            error = "Vulkan frame command recording failed";
        return false;
    }
    if (m_timestamp_pool)
        m_vk.cmd_write_timestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            m_timestamp_pool, 2 * m_current_frame + 1);
    crash_stage("vulkan frame: end command buffer");
    if (m_nvrhi_device)
    {
        try
        {
            submission_guard.command_list->close();
            submission_guard.command_list_open = false;
        }
        catch (const std::exception& exception)
        {
            error = std::string("NVRHI could not close the frame command list: ") + exception.what();
            return false;
        }
    }
    else
    {
        const VkResult end_result = m_vk.end_command_buffer(command);
        if (end_result != VK_SUCCESS)
        {
            m_device_lost = end_result == VK_ERROR_DEVICE_LOST;
            error = "vkEndCommandBuffer failed";
            return false;
        }
    }
    m_cpu_timing.record_ms = milliseconds(record_begin);
    const VkResult reset_fence = m_vk.reset_fences(m_device, 1, &frame_fence);
    if (reset_fence != VK_SUCCESS)
    {
        m_device_lost = reset_fence == VK_ERROR_DEVICE_LOST;
        m_submission_failed = true;
        error = "could not reset Vulkan frame fence";
        return false;
    }

    crash_stage("vulkan frame: queue submit");
    const auto submit_begin = Clock::now();
    VkResult submit_result = VK_SUCCESS;
    if (m_nvrhi_device)
    {
        try
        {
            if (!acquire_wait_queued)
            {
                m_nvrhi_device->queueWaitForSemaphore(nvrhi::CommandQueue::Graphics,
                    m_image_available[m_current_frame], 0);
                acquire_wait_queued = true;
            }
            m_nvrhi_device->queueSignalSemaphore(nvrhi::CommandQueue::Graphics,
                m_render_finished[image_index], 0);
            nvrhi::ICommandList* command_list = m_nvrhi_command_lists[m_current_frame][m_nvrhi_graphics_list_index].Get();
            uint64_t graphics_submission = 0;
            if (split_async_frame)
            {
                if (final_graphics_pass == FrameSubmissionGraph::InvalidPass)
                {
                    error = "Vulkan frame submission graph has no final graphics pass";
                    return false;
                }
                if (!async_submissions.submit(final_graphics_pass,
                        [this, command_list, &to_nvrhi_queue](FrameSubmissionGraph::Queue queue) {
                            return m_nvrhi_device->executeCommandList(command_list,
                                to_nvrhi_queue(queue));
                        }, queue_wait, error))
                {
                    m_device_lost = true;
                    return false;
                }
                graphics_submission = async_submissions.submission(graphics_continuation_pass);
            }
            else
                graphics_submission = m_nvrhi_device->executeCommandList(command_list);
            if (!graphics_submission)
            {
                error = "NVRHI graphics frame submission failed";
                return false;
            }
            m_last_graphics_submission = graphics_submission;
        }
        catch (const std::exception& exception)
        {
            m_device_lost = true;
            error = std::string("NVRHI frame submission failed: ") + exception.what();
            return false;
        }

        // NVRHI submits on this same graphics queue. An empty Vulkan submit
        // ordered after it attaches the frame fence to the NVRHI workload.
        VkSubmitInfo fence_submit{};
        fence_submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_result = m_vk.queue_submit(m_queue, 1, &fence_submit, frame_fence);
    }
    else
    {
        VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo submit{};
        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &m_image_available[m_current_frame];
        submit.pWaitDstStageMask = &wait_stage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &m_render_finished[image_index];
        submit_result = m_vk.queue_submit(m_queue, 1, &submit, frame_fence);
    }
    m_cpu_timing.submit_ms = milliseconds(submit_begin);
    if (submit_result != VK_SUCCESS)
    {
        m_device_lost = submit_result == VK_ERROR_DEVICE_LOST;
        m_submission_failed = true;
        error = "vkQueueSubmit failed";
        return false;
    }
    submission_guard.submitted = true;
    if (m_postprocess)
        m_post_image_states = std::move(submitted_states);
    if (m_postprocess)
        m_frame_image_pool.tick();
    m_image_fences[image_index] = frame_fence;
    m_completion_images[m_current_frame] = image_index;
    m_completion_pending[m_current_frame] = m_frame_completed != nullptr || m_observer_completed != nullptr;
    if (m_frame_submitted) m_frame_submitted(m_current_frame, image_index, m_frame_observer_data);
    if (m_timestamp_pool) m_timestamp_submitted[m_current_frame] = true;

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &m_render_finished[image_index];
    present.swapchainCount = 1;
    present.pSwapchains = &m_swapchain;
    present.pImageIndices = &image_index;
    crash_stage("vulkan frame: queue present");
    const auto present_begin = Clock::now();
    const VkResult result = m_vk.queue_present(m_queue, &present);
    m_cpu_timing.present_ms = milliseconds(present_begin);
    crash_stage("vulkan frame: presented");
    m_current_frame = (m_current_frame + 1) % FramesInFlight;
    if (mark_surface_lost(result, m_surface_lost, status))
    {
        error.clear();
        return true;
    }
    if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR && result != VK_ERROR_OUT_OF_DATE_KHR)
    {
        m_device_lost = result == VK_ERROR_DEVICE_LOST;
        error = "vkQueuePresentKHR failed";
        return false;
    }
    // Android may report SUBOPTIMAL every frame when the supported identity
    // transform differs from the display's natural rotation. Presentation is
    // valid; rebuilding the same swapchain cannot resolve that condition.
    if (result == VK_ERROR_OUT_OF_DATE_KHR)
        status = FrameStatus::RecreateRequired;
    error.clear();
    return true;
}

void FrameContext::destroy()
{
    if (m_device && m_vk.device_wait_idle)
    {
        const VkResult result = m_vk.device_wait_idle(m_device);
        m_device_lost = result == VK_ERROR_DEVICE_LOST;
        if (result == VK_SUCCESS && m_nvrhi_device)
        {
            try { m_nvrhi_device->runGarbageCollection(); }
            catch (...) {}
        }
    }
    m_nvrhi_command_lists = {};
    m_nvrhi_compute_command_lists = {};
    m_nvrhi_frame_graph_command_lists = {};
    m_async_compute_available = m_async_compute_enabled = false;
    m_async_copy_available = m_async_copy_enabled = false;
    m_last_graphics_submission = 0;
    m_nvrhi_graphics_list_index = 0;
    if (m_device)
    {
        destroy_swapchain_resources();
        m_frame_image_pool.reset();
        if (m_timestamp_pool) m_vk.destroy_query_pool(m_device, m_timestamp_pool, nullptr);
        if (m_command_pool)
            m_vk.destroy_command_pool(m_device, m_command_pool, nullptr);
    }
    m_device = VK_NULL_HANDLE;
    m_nvrhi_device = nullptr;
    m_queue = VK_NULL_HANDLE;
    m_queue_family = UINT32_MAX;
    m_compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
    m_copy_queue_family = VK_QUEUE_FAMILY_IGNORED;
    m_frame_image_sharing = {};
    m_vk = {};
    m_native_vk = {};
    m_swapchain_context = {};
    m_swapchain_create = nullptr;
    m_swapchain_destroyed = nullptr;
    m_swapchain_data = nullptr;
    m_frame_generation_ui_active = nullptr;
    m_frame_generation_ui_register = nullptr;
    m_frame_generation_ui_data = nullptr;
    m_frame_interpolation_swapchain = false;
    m_swapchain = VK_NULL_HANDLE;
    m_render_pass = VK_NULL_HANDLE;
    m_command_pool = VK_NULL_HANDLE;
    m_timestamp_pool = VK_NULL_HANDLE;
    m_timestamp_period_ns = 0.f;
    m_timestamp_valid_bits = 0;
    m_timestamp_submitted = {};
    m_last_gpu_ms.reset();
    m_extent = {};
    m_presentation_extent = {};
    m_requested_scene_extent = {};
    m_scene_samples = VK_SAMPLE_COUNT_1_BIT;
    m_format = VK_FORMAT_UNDEFINED;
    m_scene_format = VK_FORMAT_UNDEFINED;
    m_composite_format = VK_FORMAT_UNDEFINED;
    m_linear_scene = false;
    m_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    m_hdr10_surface_available = false;
    m_hdr10_requested = false;
    m_depth_format = VK_FORMAT_UNDEFINED;
    m_memory_properties = {};
    m_allow_readback = false;
    m_preserve_prepass_depth = false;
    m_postprocess = false;
    m_submission_failed = false;
    m_scene_depth_attached = false;
    m_scene_depth_images.clear();
    m_images.clear();
    m_image_views.clear();
    m_scene_images.clear();
    m_scene_views.clear();
    m_multisampled_scene_images.clear();
    m_multisampled_scene_views.clear();
    m_depth_images.clear();
    m_depth_memories.clear();
    m_depth_views.clear();
    m_framebuffers.clear();
    m_post_images.clear();
    m_post_depth_images.clear();
    m_post_motion_images.clear();
    m_post_image_states = {};
    m_bloom_mip_levels = 1;
    m_bloom_extent = {};
    m_bloom_format = VK_FORMAT_UNDEFINED;
    m_post_views.clear();
    m_post_base_views.clear();
    m_bloom_images.clear();
    m_bloom_views.clear();
    m_bloom_target_views.clear();
    m_bloom_framebuffers.clear();
    m_bloom_level_views.clear();
    m_bloom_upsample_framebuffers.clear();
    m_composite_framebuffers.clear();
    m_display_output_framebuffers.clear();
    m_display_output_pass = VK_NULL_HANDLE;
    m_display_output_recorder = nullptr;
    m_display_output_data = nullptr;
    m_composite_images.clear();
    m_composite_views.clear();
    m_frame_generation_ui_images.clear();
    m_frame_generation_ui_views.clear();
    m_frame_generation_ui_framebuffers.clear();
    m_composite_pass = VK_NULL_HANDLE;
    m_bloom_pass = VK_NULL_HANDLE;
    m_bloom_upsample_pass = VK_NULL_HANDLE;
    m_presentation_pass = VK_NULL_HANDLE;
    m_ui_clear_pass = VK_NULL_HANDLE;
    m_commands.clear();
    m_render_finished.clear();
    m_image_available = {};
    m_frame_fences = {};
    m_completion_pending = {};
    m_completion_images = {};
    m_frame_completed = nullptr;
    m_frame_completed_data = nullptr;
    m_frame_submitted = nullptr;
    m_observer_completed = nullptr;
    m_frame_observer_data = nullptr;
    m_image_fences.clear();
    m_current_frame = 0;
    m_surface_lost = false;
}

void FrameContext::destroy_swapchain_resources()
{
    if (!m_device)
        return;
    m_completion_pending = {};
    m_completion_images = {};
    for (VkFramebuffer framebuffer : m_framebuffers)
        if (framebuffer) m_vk.destroy_framebuffer(m_device, framebuffer, nullptr);
    for (VkFramebuffer framebuffer : m_composite_framebuffers)
        if (framebuffer) m_vk.destroy_framebuffer(m_device, framebuffer, nullptr);
    for (VkFramebuffer framebuffer : m_display_output_framebuffers)
        if (framebuffer) m_vk.destroy_framebuffer(m_device, framebuffer, nullptr);
    if (m_display_output_pass) m_vk.destroy_render_pass(m_device, m_display_output_pass, nullptr);
    m_display_output_framebuffers.clear();
    m_display_output_pass = VK_NULL_HANDLE;
    for (VkImageView view : m_composite_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImage image : m_composite_images)
        if (image) m_frame_image_pool.release(image);
    for (VkFramebuffer framebuffer : m_frame_generation_ui_framebuffers)
        if (framebuffer) m_vk.destroy_framebuffer(m_device, framebuffer, nullptr);
    for (VkFramebuffer framebuffer : m_bloom_framebuffers)
        if (framebuffer) m_vk.destroy_framebuffer(m_device, framebuffer, nullptr);
    for (const auto& image_framebuffers : m_bloom_upsample_framebuffers)
        for (VkFramebuffer framebuffer : image_framebuffers)
            if (framebuffer) m_vk.destroy_framebuffer(m_device, framebuffer, nullptr);
    for (VkImageView view : m_scene_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImageView view : m_frame_generation_ui_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImage image : m_frame_generation_ui_images)
        if (image) m_frame_image_pool.release(image);
    for (VkImage image : m_scene_images)
        if (image) m_frame_image_pool.release(image);
    for (VkImageView view : m_multisampled_scene_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImage image : m_multisampled_scene_images)
        if (image) m_frame_image_pool.release(image);
    for (VkImageView view : m_post_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImageView view : m_post_base_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImage image : m_post_images)
        if (image) m_frame_image_pool.release(image);
    for (VkImageView view : m_bloom_target_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImageView view : m_bloom_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (const auto& image_views : m_bloom_level_views)
        for (VkImageView view : image_views)
            if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImage image : m_bloom_images)
        if (image) m_frame_image_pool.release(image);
    if (m_bloom_pass) m_vk.destroy_render_pass(m_device, m_bloom_pass, nullptr);
    if (m_bloom_upsample_pass) m_vk.destroy_render_pass(m_device, m_bloom_upsample_pass, nullptr);
    if (m_composite_pass) m_vk.destroy_render_pass(m_device, m_composite_pass, nullptr);
    if (m_presentation_pass) m_vk.destroy_render_pass(m_device, m_presentation_pass, nullptr);
    if (m_ui_clear_pass) m_vk.destroy_render_pass(m_device, m_ui_clear_pass, nullptr);
    if (m_overlay_pass) m_vk.destroy_render_pass(m_device, m_overlay_pass, nullptr);
    for (VkImageView view : m_image_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImageView view : m_depth_views)
        if (view) m_vk.destroy_image_view(m_device, view, nullptr);
    for (VkImage image : m_depth_images)
        if (image)
        {
            if (m_postprocess) m_frame_image_pool.release(image);
            else m_vk.destroy_image(m_device, image, nullptr);
        }
    for (VkDeviceMemory memory : m_depth_memories)
        if (memory) m_vk.free_memory(m_device, memory, nullptr);
    if (m_render_pass) m_vk.destroy_render_pass(m_device, m_render_pass, nullptr);
    for (VkSemaphore semaphore : m_render_finished)
        if (semaphore) m_vk.destroy_semaphore(m_device, semaphore, nullptr);
    for (VkSemaphore semaphore : m_image_available)
        if (semaphore) m_vk.destroy_semaphore(m_device, semaphore, nullptr);
    for (VkFence fence : m_frame_fences)
        if (fence) m_vk.destroy_fence(m_device, fence, nullptr);
    if (m_frame_interpolation_swapchain && m_swapchain_destroyed)
        m_swapchain_destroyed(m_swapchain_data);
    if (m_swapchain) m_vk.destroy_swapchain(m_device, m_swapchain, nullptr);
    m_swapchain = VK_NULL_HANDLE;
    m_frame_interpolation_swapchain = false;
    m_vk = m_native_vk;
    m_render_pass = VK_NULL_HANDLE;
    m_overlay_pass = VK_NULL_HANDLE;
    m_composite_pass = VK_NULL_HANDLE;
    m_ui_clear_pass = VK_NULL_HANDLE;
    m_bloom_pass = VK_NULL_HANDLE;
    m_bloom_upsample_pass = VK_NULL_HANDLE;
    m_presentation_pass = VK_NULL_HANDLE;
    m_frame_generation_ui_images.clear();
    m_frame_generation_ui_views.clear();
    m_frame_generation_ui_framebuffers.clear();
    m_extent = {};
    m_presentation_extent = {};
    m_bloom_extent = {};
    m_bloom_mip_levels = 1;
    m_bloom_format = VK_FORMAT_UNDEFINED;
    m_images.clear();
    m_image_views.clear();
    m_scene_images.clear();
    m_scene_views.clear();
    m_multisampled_scene_images.clear();
    m_multisampled_scene_views.clear();
    m_depth_images.clear();
    m_depth_memories.clear();
    m_depth_views.clear();
    m_framebuffers.clear();
    m_post_images.clear();
    m_post_depth_images.clear();
    m_post_motion_images.clear();
    m_post_image_states = {};
    m_exposure_image = VK_NULL_HANDLE;
    m_exposure_recorder = nullptr;
    m_exposure_history_valid = nullptr;
    m_exposure_data = nullptr;
    m_temporal_images = {};
    m_temporal_extent = {};
    m_temporal_motion_images.clear();
    m_temporal_recorder = nullptr;
    m_temporal_active = nullptr;
    m_temporal_output_slot = nullptr;
    m_temporal_data = nullptr;
    m_fsr3_output_images.clear();
    m_fsr3_motion_images.clear();
    m_fsr3_recorder = nullptr;
    m_fsr3_active = nullptr;
    m_fsr3_data = nullptr;
    m_post_views.clear();
    m_post_base_views.clear();
    m_bloom_images.clear();
    m_bloom_views.clear();
    m_bloom_target_views.clear();
    m_bloom_framebuffers.clear();
    m_bloom_level_views.clear();
    m_bloom_upsample_framebuffers.clear();
    m_composite_framebuffers.clear();
    m_composite_images.clear();
    m_composite_views.clear();
    m_scene_depth_attached = false;
    m_scene_depth_images.clear();
    m_render_finished.clear();
    m_image_available = {};
    m_frame_fences = {};
    m_image_fences.clear();
}
}
