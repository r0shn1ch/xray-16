#include "xrCore/stdafx.h"
#include "Fsr3Upscaler.h"

#include "xrCore/log.h"

#include <algorithm>
#include <cmath>
#include <utility>

#if defined(XRAY_HAS_FSR3_ANDROID)
namespace
{
// The FidelityFX Vulkan backend uses one device context for all its effects.
PFN_vkGetPhysicalDeviceProperties2 fsr_get_properties2 = nullptr;
PFN_vkGetPhysicalDeviceFeatures2 fsr_get_features2 = nullptr;
}

extern "C" VKAPI_ATTR void VKAPI_CALL xrayFsrGetPhysicalDeviceProperties2(
    VkPhysicalDevice physical, VkPhysicalDeviceProperties2* properties)
{
    fsr_get_properties2(physical, properties);
}

extern "C" VKAPI_ATTR void VKAPI_CALL xrayFsrGetPhysicalDeviceFeatures2(
    VkPhysicalDevice physical, VkPhysicalDeviceFeatures2* features)
{
    fsr_get_features2(physical, features);
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL xrayFsrWaitSemaphores(
    VkDevice device, const VkSemaphoreWaitInfo* info, uint64_t timeout)
{
    auto wait = reinterpret_cast<PFN_vkWaitSemaphores>(
        vkGetDeviceProcAddr(device, "vkWaitSemaphores"));
    if (!wait)
        wait = reinterpret_cast<PFN_vkWaitSemaphores>(
            vkGetDeviceProcAddr(device, "vkWaitSemaphoresKHR"));
    return wait ? wait(device, info, timeout) : VK_ERROR_EXTENSION_NOT_PRESENT;
}

extern "C" VKAPI_ATTR VkResult VKAPI_CALL xrayFsrGetSemaphoreCounterValue(
    VkDevice device, VkSemaphore semaphore, uint64_t* value)
{
    auto get_counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
        vkGetDeviceProcAddr(device, "vkGetSemaphoreCounterValue"));
    if (!get_counter)
        get_counter = reinterpret_cast<PFN_vkGetSemaphoreCounterValue>(
            vkGetDeviceProcAddr(device, "vkGetSemaphoreCounterValueKHR"));
    return get_counter ? get_counter(device, semaphore, value) : VK_ERROR_EXTENSION_NOT_PRESENT;
}
#endif

namespace xray::render::vulkan
{
namespace
{
template <typename T> T device_proc(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{
    return reinterpret_cast<T>(get ? get(device, name) : nullptr);
}

#if defined(XRAY_HAS_FSR3_ANDROID)
void fsr3_message(FfxMsgType type, const wchar_t* message)
{
    if (type == FFX_MESSAGE_TYPE_ERROR)
        Msg("! [renderer-vulkan][FSR3] %ls", message ? message : L"(no message)");
}
#endif
}

bool Fsr3Upscaler::initialize(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance,
    VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    uint32_t api_version, PFN_vkGetPhysicalDeviceFormatProperties get_format_properties,
    VkExtent2D render_extent, VkExtent2D display_extent, VkFormat scene_format,
    VkFormat display_format, VkFormat depth_format, VkFormat motion_format, uint32_t scene_mip_levels,
    const std::vector<VkImage>& color_images, const std::vector<VkImage>& depth_images,
    const std::vector<VkImage>& motion_images, FrameImagePool& image_pool, std::string& error,
    FrameImageSharing sharing)
{
    if (frame_generation_context_)
        release_output_resources();
    else
        destroy();
#if !defined(XRAY_HAS_FSR3_ANDROID)
    (void)instance; (void)get_instance; (void)physical; (void)device; (void)get; (void)api_version; (void)get_format_properties;
    (void)render_extent; (void)display_extent;
    (void)scene_format; (void)display_format; (void)depth_format; (void)motion_format;
    (void)scene_mip_levels; (void)color_images; (void)depth_images; (void)motion_images; (void)image_pool;
    (void)sharing;
    error = "FSR3 is available only in Android ARM64 Vulkan builds";
    return false;
#else
    if (!instance || !get_instance || !physical || !device || !get || !get_format_properties || !render_extent.width || !render_extent.height ||
        !display_extent.width || !display_extent.height || color_images.empty() ||
        color_images.size() != depth_images.size() || color_images.size() != motion_images.size() ||
        color_images.size() > 16 || scene_format == VK_FORMAT_UNDEFINED ||
        display_format == VK_FORMAT_UNDEFINED || depth_format == VK_FORMAT_UNDEFINED ||
        motion_format == VK_FORMAT_UNDEFINED || !scene_mip_levels || !image_pool.initialized() ||
        std::any_of(color_images.begin(), color_images.end(), [](VkImage image) { return !image; }) ||
        std::any_of(depth_images.begin(), depth_images.end(), [](VkImage image) { return !image; }) ||
        std::any_of(motion_images.begin(), motion_images.end(), [](VkImage image) { return !image; }))
    {
        error = "FSR3 requires live images, formats and a Vulkan frame image pool";
        return false;
    }
    if (api_version < VK_API_VERSION_1_1)
    {
        error = "FSR3 needs Vulkan 1.1 or newer";
        return false;
    }

    if (frame_generation_context_ &&
        (physical_ != physical || device_ != device || render_extent_.width != render_extent.width ||
            render_extent_.height != render_extent.height || display_extent_.width != display_extent.width ||
            display_extent_.height != display_extent.height))
    {
        error = "FSR frame-generation context does not match the renderer extents";
        return false;
    }
    instance_ = instance;
    get_instance_ = get_instance;
    physical_ = physical;
    device_ = device;
    get_ = get;
    image_pool_ = &image_pool;
    sharing_ = std::move(sharing);
    render_extent_ = render_extent;
    display_extent_ = display_extent;
    scene_format_ = scene_format;
    display_format_ = display_format;
    depth_format_ = depth_format;
    motion_format_ = motion_format;
    scene_mip_levels_ = scene_mip_levels;
    color_images_ = color_images;
    depth_images_ = depth_images;
    create_image_view_ = device_proc<PFN_vkCreateImageView>(device_, get_, "vkCreateImageView");
    destroy_image_view_ = device_proc<PFN_vkDestroyImageView>(device_, get_, "vkDestroyImageView");
    if (!create_image_view_ || !destroy_image_view_)
    {
        error = "FSR3 Vulkan image view procedures are unavailable";
        release_output_resources();
        return false;
    }

    VkFormatProperties output_properties{};
    const VkFormat output_format = upscale_output_format();
    get_format_properties(physical_, output_format, &output_properties);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((output_properties.optimalTilingFeatures & required) != required)
    {
        error = "FSR3 output format does not support both storage and sampling";
        release_output_resources();
        return false;
    }

    output_images_.reserve(color_images_.size());
    output_views_.reserve(color_images_.size());
    for (size_t i = 0; i < color_images_.size(); ++i)
    {
        VkImageCreateInfo output_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        output_info.imageType = VK_IMAGE_TYPE_2D;
        output_info.format = output_format;
        output_info.extent = {display_extent_.width, display_extent_.height, 1};
        output_info.mipLevels = output_info.arrayLayers = 1;
        output_info.samples = VK_SAMPLE_COUNT_1_BIT;
        output_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        output_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        sharing_.apply(output_info);
        VkImage output = VK_NULL_HANDLE;
        if (!image_pool_->acquire(output_info, output, error))
        {
            error = "FSR3 output image: " + error;
            release_output_resources();
            return false;
        }
        output_images_.push_back(output);
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = output;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = output_info.format;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VkImageView view = VK_NULL_HANDLE;
        if (create_image_view_(device_, &view_info, nullptr, &view) != VK_SUCCESS)
        {
            error = "FSR3 output image view creation failed";
            release_output_resources();
            return false;
        }
        output_views_.push_back(view);
    }

    if (!initialize_backend_interfaces(error))
    {
        destroy();
        return false;
    }
    if (!context_initialized_ && !create_context(render_extent_, display_extent_,
            display_format_, false, scene_format_ == VK_FORMAT_R16G16B16A16_SFLOAT, error))
    {
        destroy();
        return false;
    }
    if (!frame_generation_context_)
        frame_generation_requested_ = false;
    history_valid_ = false;
    active_ = false;
    error.clear();
    return true;
#endif
}

void Fsr3Upscaler::release_output_resources()
{
    if (device_ && destroy_image_view_)
        for (VkImageView view : output_views_)
            if (view) destroy_image_view_(device_, view, nullptr);
    if (image_pool_)
        for (VkImage image : output_images_)
            if (image) image_pool_->release(image);
    image_pool_ = nullptr;
    sharing_ = {};
    color_images_.clear();
    depth_images_.clear();
    motion_images_.clear();
    output_images_.clear();
    output_views_.clear();
}

void Fsr3Upscaler::reset_for_reinitialize()
{
    if (frame_generation_requested_)
        release_output_resources();
    else
        destroy();
}

void Fsr3Upscaler::destroy()
{
#if defined(XRAY_HAS_FSR3_ANDROID)
    if (frame_interpolation_swapchain_)
        (void)ffxWaitForPresents(ffxGetSwapchainVK(frame_interpolation_swapchain_));
    destroy_context();
#endif
    release_output_resources();
    physical_ = VK_NULL_HANDLE;
    instance_ = VK_NULL_HANDLE;
    get_instance_ = nullptr;
    device_ = VK_NULL_HANDLE;
    get_ = nullptr;
    render_extent_ = display_extent_ = {};
    scene_format_ = display_format_ = depth_format_ = motion_format_ = VK_FORMAT_UNDEFINED;
    scene_mip_levels_ = 1;
    create_image_view_ = nullptr;
    destroy_image_view_ = nullptr;
#if defined(XRAY_HAS_FSR3_ANDROID)
    scratch_shared_.clear();
    scratch_upscaling_.clear();
    scratch_frame_interpolation_.clear();
    backend_shared_ = {};
    backend_upscaling_ = {};
    backend_frame_interpolation_ = {};
    context_ = {};
#endif
    backend_interfaces_initialized_ = false;
    frame_generation_requested_ = false;
    frame_generation_context_ = false;
    frame_generation_frame_enabled_ = false;
    frame_interpolation_swapchain_ = VK_NULL_HANDLE;
    frame_id_ = 0;
    delta_ms_ = 16.67f;
    jitter_ = {};
    jitter_phase_count_ = 1;
    active_ = false;
    history_valid_ = false;
}

void Fsr3Upscaler::set_frame(uint64_t frame_id, float delta_ms, float camera_near,
    float camera_far, float camera_fov_vertical)
{
    if (history_valid_ && frame_id - history_frame_id_ != 1)
        reset_history();
    frame_id_ = frame_id;
    delta_ms_ = std::isfinite(delta_ms) ? std::clamp(delta_ms, 0.1f, 1000.f) : 16.67f;
    camera_near_ = std::isfinite(camera_near) && camera_near > 0.f ? camera_near : .1f;
    camera_far_ = std::isfinite(camera_far) && camera_far > camera_near_ ? camera_far : 500.f;
    camera_fov_vertical_ = std::isfinite(camera_fov_vertical) && camera_fov_vertical > 0.f &&
        camera_fov_vertical < 3.13f ? camera_fov_vertical : 1.04719755f;
#if defined(XRAY_HAS_FSR3_ANDROID)
    if (frame_generation_context_ && frame_interpolation_swapchain_)
        configure_frame_generation(active_ && frame_generation_requested_ && history_valid_);
#endif
}

#if defined(XRAY_HAS_FSR3_ANDROID)
void Fsr3Upscaler::configure_frame_generation(bool enabled)
{
    FfxFrameGenerationConfig config{};
    config.swapChain = ffxGetSwapchainVK(frame_interpolation_swapchain_);
    config.frameGenerationCallback = [](const FfxFrameGenerationDispatchDescription* desc, void*)
    {
        return ffxFsr3DispatchFrameGeneration(desc);
    };
    config.frameGenerationEnabled = enabled;
    config.allowAsyncWorkloads = false;
    config.onlyPresentInterpolated = false;
    config.interpolationRect = {0, 0, static_cast<int32_t>(display_extent_.width),
        static_cast<int32_t>(display_extent_.height)};
    config.frameID = frame_id_;
    const FfxErrorCode result = ffxFsr3ConfigureFrameGeneration(&context_, &config);
    frame_generation_frame_enabled_ = result == FFX_OK && config.frameGenerationEnabled;
    frame_generation_config_valid_ = result == FFX_OK;
    if (result != FFX_OK)
    {
        Msg("! [renderer-vulkan][FSR3] frame-generation configuration failed with code %d",
            static_cast<int>(result));
        if (enabled) configure_frame_generation(false);
    }
}
#endif

bool Fsr3Upscaler::jitter_offset(uint32_t sample, float& x, float& y) const
{
#if defined(XRAY_HAS_FSR3_ANDROID)
    if (!ready()) return false;
    const FfxErrorCode result = ffxFsr3GetJitterOffset(&x, &y,
        static_cast<int32_t>(sample % static_cast<uint32_t>(jitter_phase_count_)), jitter_phase_count_);
    return result == FFX_OK;
#else
    (void)sample; (void)x; (void)y;
    return false;
#endif
}

VkImage Fsr3Upscaler::output_image(uint32_t image_index) const
{
    return image_index < output_images_.size() ? output_images_[image_index] : VK_NULL_HANDLE;
}

VkImageView Fsr3Upscaler::output_view(uint32_t image_index) const
{
    return image_index < output_views_.size() ? output_views_[image_index] : VK_NULL_HANDLE;
}

bool Fsr3Upscaler::active_callback(void* user)
{
    return user && static_cast<Fsr3Upscaler*>(user)->active();
}

bool Fsr3Upscaler::record_callback(const FrameRecordingContext& frame, void* user, std::string& error)
{
    return user && static_cast<Fsr3Upscaler*>(user)->record(frame, error);
}

bool Fsr3Upscaler::frame_generation_ui_active_callback(void* user)
{
#if defined(XRAY_HAS_FSR3_ANDROID)
    const auto* self = static_cast<const Fsr3Upscaler*>(user);
    return self && self->frame_generation_context_ && self->frame_generation_frame_enabled_ &&
        self->frame_interpolation_swapchain_ != VK_NULL_HANDLE;
#else
    (void)user;
    return false;
#endif
}

bool Fsr3Upscaler::register_frame_generation_ui_callback(VkImage image, VkFormat format,
    VkExtent2D extent, VkImageUsageFlags usage, void* user, std::string& error)
{
    if (!user)
    {
        error = "FSR frame-generation UI callback has no owner";
        return false;
    }
    return static_cast<Fsr3Upscaler*>(user)->register_frame_generation_ui(image, format, extent, usage, error);
}

bool Fsr3Upscaler::register_frame_generation_ui(VkImage image, VkFormat format, VkExtent2D extent,
    VkImageUsageFlags usage, std::string& error)
{
#if defined(XRAY_HAS_FSR3_ANDROID)
    if (!frame_interpolation_swapchain_ || !frame_generation_context_)
    {
        error = "FSR frame-generation UI requires an initialized interpolation swapchain";
        return false;
    }
    FfxResource ui_resource{};
    uint32_t flags = 0;
    if (image)
    {
        if (format == VK_FORMAT_UNDEFINED || !extent.width || !extent.height ||
            !(usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        {
            error = "invalid FSR frame-generation UI image";
            return false;
        }
        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = format;
        image_info.extent = {extent.width, extent.height, 1};
        image_info.mipLevels = image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = usage;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        const FfxResourceDescription description = ffxGetImageResourceDescriptionVK(image, image_info);
        ui_resource = ffxGetResourceVK(image, description, L"XRayFrameGenerationUI",
            FFX_RESOURCE_STATE_COPY_SRC);
        flags = FFX_UI_COMPOSITION_FLAG_ENABLE_INTERNAL_UI_DOUBLE_BUFFERING;
    }
    const FfxErrorCode result = ffxRegisterFrameinterpolationUiResourceVK(
        ffxGetSwapchainVK(frame_interpolation_swapchain_), ui_resource, flags);
    if (result != FFX_OK)
    {
        error = "ffxRegisterFrameinterpolationUiResourceVK failed with code " +
            std::to_string(static_cast<int>(result));
        return false;
    }
    error.clear();
    return true;
#else
    (void)image;
    (void)format;
    (void)extent;
    (void)usage;
    error = "FSR frame generation is unavailable in this build";
    return false;
#endif
}

bool Fsr3Upscaler::record(const FrameRecordingContext& frame, std::string& error)
{
#if !defined(XRAY_HAS_FSR3_ANDROID)
    (void)frame;
    error = "FSR3 is not compiled for this platform";
    return false;
#else
    if (!ready() || !active_ || !frame.command_buffer || frame.image_index >= color_images_.size())
    {
        error = "FSR3 dispatch has invalid frame inputs";
        return false;
    }
    if (frame_generation_context_ && !frame_generation_config_valid_)
    {
        error = "FSR could not configure or disable interpolation for this frame";
        return false;
    }
    const auto image_info = [this](VkExtent2D extent, VkFormat format, VkImageUsageFlags usage) {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {extent.width, extent.height, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        sharing_.apply(info);
        return info;
    };
    VkImageCreateInfo color_info = image_info(render_extent_, scene_format_,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    color_info.mipLevels = scene_mip_levels_;
    if (scene_mip_levels_ > 1) color_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    const VkImageCreateInfo depth_info = image_info(render_extent_, depth_format_,
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
    const VkImageCreateInfo motion_info = image_info(render_extent_, motion_format_,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const VkImageCreateInfo output_info = image_info(display_extent_, upscale_output_format(),
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const uint32_t index = frame.image_index;
    const FfxResource color = ffxGetResourceVK(color_images_[index],
        ffxGetImageResourceDescriptionVK(color_images_[index], color_info), L"XRaySceneColor",
        FFX_RESOURCE_STATE_COMPUTE_READ);
    const FfxResource depth = ffxGetResourceVK(depth_images_[index],
        ffxGetImageResourceDescriptionVK(depth_images_[index], depth_info), L"XRaySceneDepth",
        FFX_RESOURCE_STATE_COMPUTE_READ);
    const FfxResource motion = ffxGetResourceVK(motion_images_[index],
        ffxGetImageResourceDescriptionVK(motion_images_[index], motion_info), L"XRayMotionVectors",
        FFX_RESOURCE_STATE_COMPUTE_READ);
    const FfxResource output = ffxGetResourceVK(output_images_[index],
        ffxGetImageResourceDescriptionVK(output_images_[index], output_info, FFX_RESOURCE_USAGE_UAV),
        L"XRayUpscaleOutput", FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    FfxFsr3DispatchUpscaleDescription dispatch{};
    dispatch.commandList = ffxGetCommandListVK(frame.command_buffer);
    dispatch.color = color;
    dispatch.depth = depth;
    dispatch.motionVectors = motion;
    dispatch.upscaleOutput = output;
    dispatch.jitterOffset = {jitter_[0], jitter_[1]};
    // XRay motion vectors are previous-UV minus current-UV. FSR expects NDC
    // motion scaled by half the render extent; UV already contributes the 0.5,
    // and the Vulkan screen-space Y axis is inverted.
    dispatch.motionVectorScale = {static_cast<float>(render_extent_.width),
        -static_cast<float>(render_extent_.height)};
    dispatch.renderSize = {render_extent_.width, render_extent_.height};
    dispatch.upscaleSize = {display_extent_.width, display_extent_.height};
    dispatch.enableSharpening = true;
    dispatch.sharpness = .2f;
    dispatch.frameTimeDelta = delta_ms_;
    dispatch.preExposure = 1.f;
    dispatch.reset = !history_valid_;
    dispatch.cameraNear = camera_near_;
    dispatch.cameraFar = camera_far_;
    dispatch.cameraFovAngleVertical = camera_fov_vertical_;
    dispatch.viewSpaceToMetersFactor = 1.f;
    dispatch.frameID = frame_id_;
    const FfxErrorCode result = ffxFsr3ContextDispatchUpscale(&context_, &dispatch);
    if (result != FFX_OK)
    {
        error = "ffxFsr3ContextDispatchUpscale failed with code " +
            std::to_string(static_cast<int>(result));
        history_valid_ = false;
        if (frame_generation_frame_enabled_) configure_frame_generation(false);
        return false;
    }
    if (frame_generation_frame_enabled_)
    {
        FfxFsr3DispatchFrameGenerationPrepareDescription prepare{};
        prepare.commandList = ffxGetCommandListVK(frame.command_buffer);
        prepare.depth = depth;
        prepare.motionVectors = motion;
        prepare.jitterOffset = {jitter_[0], jitter_[1]};
        prepare.motionVectorScale = dispatch.motionVectorScale;
        prepare.renderSize = {render_extent_.width, render_extent_.height};
        prepare.frameTimeDelta = delta_ms_;
        prepare.cameraNear = camera_near_;
        prepare.cameraFar = camera_far_;
        prepare.viewSpaceToMetersFactor = 1.f;
        prepare.cameraFovAngleVertical = camera_fov_vertical_;
        prepare.frameID = frame_id_;
        const FfxErrorCode prepare_result =
            ffxFsr3ContextDispatchFrameGenerationPrepare(&context_, &prepare);
        if (prepare_result != FFX_OK)
        {
            Msg("! [renderer-vulkan][FSR3] frame-generation prepare failed with code %d",
                static_cast<int>(prepare_result));
            history_valid_ = false;
            configure_frame_generation(false);
            if (!frame_generation_config_valid_)
            {
                error = "FSR could not disable interpolation after frame-generation prepare failed";
                return false;
            }
        }
    }
    error.clear();
    return true;
#endif
}

#if defined(XRAY_HAS_FSR3_ANDROID)
bool Fsr3Upscaler::create_backend_interface(FfxInterface& iface, std::vector<uint8_t>& scratch,
    size_t max_contexts, std::string& error)
{
    const size_t size = ffxGetScratchMemorySizeVK(physical_, max_contexts);
    if (!size)
    {
        error = "FSR3 Vulkan backend scratch memory size is zero";
        return false;
    }
    scratch.resize(size);
    VkDeviceContext context{device_, physical_, get_};
    const FfxErrorCode result = ffxGetInterfaceVK(&iface, ffxGetDeviceVK(&context),
        scratch.data(), scratch.size(), max_contexts);
    if (result != FFX_OK)
    {
        error = "ffxGetInterfaceVK failed with code " + std::to_string(static_cast<int>(result));
        return false;
    }
    return true;
}

bool Fsr3Upscaler::initialize_backend_interfaces(std::string& error)
{
    if (backend_interfaces_initialized_)
        return true;
    fsr_get_properties2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        get_instance_(instance_, "vkGetPhysicalDeviceProperties2"));
    fsr_get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
        get_instance_(instance_, "vkGetPhysicalDeviceFeatures2"));
    if (!fsr_get_properties2 || !fsr_get_features2)
    {
        error = "FSR3 Vulkan physical-device queries are unavailable";
        return false;
    }
    if (!create_backend_interface(backend_shared_, scratch_shared_, 2, error) ||
        !create_backend_interface(backend_upscaling_, scratch_upscaling_, 2, error) ||
        !create_backend_interface(backend_frame_interpolation_, scratch_frame_interpolation_, 3, error))
        return false;
    backend_interfaces_initialized_ = true;
    return true;
}

bool Fsr3Upscaler::create_context(VkExtent2D render_extent, VkExtent2D display_extent,
    VkFormat back_buffer_format, bool frame_generation, bool hdr_input, std::string& error)
{
    if (!backend_interfaces_initialized_ || context_initialized_ || !render_extent.width ||
        !render_extent.height || !display_extent.width || !display_extent.height)
    {
        error = "FSR3 context creation has invalid backend state or extents";
        return false;
    }
    FfxFsr3ContextDescription description{};
    description.flags = FFX_FSR3_ENABLE_AUTO_EXPOSURE |
        FFX_FSR3_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION;
    if (hdr_input)
        description.flags |= FFX_FSR3_ENABLE_HIGH_DYNAMIC_RANGE;
    if (!frame_generation)
        description.flags |= FFX_FSR3_ENABLE_UPSCALING_ONLY;
    description.maxRenderSize = {render_extent.width, render_extent.height};
    description.maxUpscaleSize = {display_extent.width, display_extent.height};
    description.displaySize = {display_extent.width, display_extent.height};
    description.backendInterfaceSharedResources = backend_shared_;
    description.backendInterfaceUpscaling = backend_upscaling_;
    description.backendInterfaceFrameInterpolation = backend_frame_interpolation_;
    description.fpMessage = fsr3_message;
    description.backBufferFormat = ffxGetSurfaceFormatVK(back_buffer_format);
    const FfxErrorCode result = ffxFsr3ContextCreate(&context_, &description);
    if (result != FFX_OK)
    {
        context_ = {};
        error = "ffxFsr3ContextCreate failed with code " + std::to_string(static_cast<int>(result));
        return false;
    }
    render_extent_ = render_extent;
    display_extent_ = display_extent;
    context_initialized_ = true;
    frame_generation_context_ = frame_generation;
    frame_generation_frame_enabled_ = false;
    jitter_phase_count_ = std::max(1, ffxFsr3GetJitterPhaseCount(
        static_cast<int32_t>(render_extent_.width), static_cast<int32_t>(display_extent_.width)));
    history_valid_ = false;
    error.clear();
    return true;
}

void Fsr3Upscaler::destroy_context()
{
    if (context_initialized_)
    {
        ffxFsr3ContextDestroy(&context_);
        context_initialized_ = false;
    }
    context_ = {};
    frame_generation_context_ = false;
    frame_generation_frame_enabled_ = false;
}

bool Fsr3Upscaler::create_frame_generation_swapchain(VkSwapchainCreateInfoKHR& create_info,
    const FrameSwapchainCreateContext& frame_context, FrameDispatch& dispatch,
    VkSwapchainKHR& swapchain, std::string& error)
{
    if (!frame_generation_requested_ || !frame_context.instance || !frame_context.get_instance_proc ||
        !frame_context.physical_device || !frame_context.device ||
        !frame_context.get_device_proc || !frame_context.game_queue || !frame_context.present_queue ||
        !frame_context.compute_queue || !frame_context.image_acquire_queue ||
        (frame_context.game_queue_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) !=
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) ||
        frame_context.game_queue_family == UINT32_MAX || frame_context.present_queue_family == UINT32_MAX ||
        frame_context.compute_queue_family == UINT32_MAX || frame_context.image_acquire_queue_family == UINT32_MAX ||
        !frame_context.scene_extent.width || !frame_context.scene_extent.height ||
        !create_info.imageExtent.width || !create_info.imageExtent.height)
    {
        error = "FSR frame generation requires four Vulkan queue roles and valid extents";
        return false;
    }
    if (frame_context.api_version < VK_API_VERSION_1_1)
    {
        error = "FSR frame generation requires Vulkan 1.1 or newer";
        return false;
    }
    if ((device_ && device_ != frame_context.device) ||
        (physical_ && physical_ != frame_context.physical_device))
    {
        const bool requested = frame_generation_requested_;
        destroy();
        frame_generation_requested_ = requested;
    }
    instance_ = frame_context.instance;
    get_instance_ = frame_context.get_instance_proc;
    physical_ = frame_context.physical_device;
    device_ = frame_context.device;
    get_ = frame_context.get_device_proc;
    render_extent_ = frame_context.scene_extent;
    display_extent_ = create_info.imageExtent;
    if (!initialize_backend_interfaces(error))
        return false;

    VkDeviceContext device_context{device_, physical_, get_};
    const FfxDevice ffx_device = ffxGetDeviceVK(&device_context);
    FfxSwapchainReplacementFunctions replacement_functions{};
    FfxErrorCode result = ffxGetSwapchainReplacementFunctionsVK(ffx_device, &replacement_functions);
    if (result != FFX_OK || !replacement_functions.destroySwapchainKHR ||
        !replacement_functions.getSwapchainImagesKHR || !replacement_functions.acquireNextImageKHR ||
        !replacement_functions.queuePresentKHR)
    {
        error = "FSR Vulkan swapchain replacement procedures are unavailable";
        return false;
    }

    create_info.imageUsage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkFrameInterpolationInfoFFX interpolation_info{};
    interpolation_info.physicalDevice = physical_;
    interpolation_info.device = device_;
    interpolation_info.gameQueue = {frame_context.game_queue, frame_context.game_queue_family, nullptr};
    interpolation_info.asyncComputeQueue =
        {frame_context.compute_queue, frame_context.compute_queue_family, nullptr};
    interpolation_info.presentQueue =
        {frame_context.present_queue, frame_context.present_queue_family, nullptr};
    interpolation_info.imageAcquireQueue =
        {frame_context.image_acquire_queue, frame_context.image_acquire_queue_family, nullptr};
    interpolation_info.compositionMode = VK_COMPOSITION_MODE_GAME_QUEUE_FFX;

    FfxSwapchain ffx_swapchain = nullptr;
    result = ffxReplaceSwapchainForFrameinterpolationVK(
        ffxGetCommandQueueVK(frame_context.game_queue), ffx_swapchain, &create_info, &interpolation_info);
    if (result != FFX_OK || !ffx_swapchain)
    {
        if (ffx_swapchain)
            replacement_functions.destroySwapchainKHR(device_, ffxGetVKSwapchain(ffx_swapchain), nullptr);
        error = "ffxReplaceSwapchainForFrameinterpolationVK failed with code " +
            std::to_string(static_cast<int>(result));
        return false;
    }

    swapchain = ffxGetVKSwapchain(ffx_swapchain);
    dispatch.destroy_swapchain = replacement_functions.destroySwapchainKHR;
    dispatch.get_swapchain_images = replacement_functions.getSwapchainImagesKHR;
    dispatch.acquire_next_image = replacement_functions.acquireNextImageKHR;
    dispatch.queue_present = replacement_functions.queuePresentKHR;
    frame_interpolation_swapchain_ = swapchain;
    if (!create_context(frame_context.scene_extent, create_info.imageExtent,
            create_info.imageFormat, true, false, error))
    {
        replacement_functions.destroySwapchainKHR(device_, swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
        frame_interpolation_swapchain_ = VK_NULL_HANDLE;
        return false;
    }
    error.clear();
    return true;
}
#endif

bool Fsr3Upscaler::swapchain_create_callback(VkSwapchainCreateInfoKHR& create_info,
    const FrameSwapchainCreateContext& context, FrameDispatch& dispatch,
    VkSwapchainKHR& swapchain, bool& frame_interpolation, void* user_data, std::string& error)
{
#if defined(XRAY_HAS_FSR3_ANDROID)
    auto* self = static_cast<Fsr3Upscaler*>(user_data);
    if (!self)
    {
        error = "FSR frame-generation callback has no owner";
        return false;
    }
    const bool created = self->create_frame_generation_swapchain(
        create_info, context, dispatch, swapchain, error);
    frame_interpolation = created;
    if (!created)
        Msg("! [renderer-vulkan][FSR3] frame generation unavailable; using upscaling only: %s",
            error.c_str());
    return created;
#else
    (void)create_info;
    (void)context;
    (void)dispatch;
    (void)swapchain;
    (void)frame_interpolation;
    (void)user_data;
    error = "FSR frame generation is unavailable in this build";
    return false;
#endif
}

void Fsr3Upscaler::swapchain_destroyed_callback(void* user_data)
{
#if defined(XRAY_HAS_FSR3_ANDROID)
    auto* self = static_cast<Fsr3Upscaler*>(user_data);
    if (!self) return;
    if (self->frame_interpolation_swapchain_)
        (void)ffxWaitForPresents(ffxGetSwapchainVK(self->frame_interpolation_swapchain_));
    self->destroy_context();
    self->frame_interpolation_swapchain_ = VK_NULL_HANDLE;
#else
    (void)user_data;
#endif
}
}
