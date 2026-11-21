#pragma once

#include "FrameContext.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#if defined(XRAY_HAS_FSR3_ANDROID)
#include <FidelityFX/host/ffx_fsr3.h>
#include <FidelityFX/host/backends/vk/ffx_vk.h>
#endif

namespace xray::render::vulkan
{
class Fsr3Upscaler
{
public:
    Fsr3Upscaler() = default;
    ~Fsr3Upscaler() { destroy(); }
    Fsr3Upscaler(const Fsr3Upscaler&) = delete;
    Fsr3Upscaler& operator=(const Fsr3Upscaler&) = delete;

    bool initialize(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance,
        VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        uint32_t api_version, PFN_vkGetPhysicalDeviceFormatProperties get_format_properties,
        VkExtent2D render_extent, VkExtent2D display_extent, VkFormat scene_format,
        VkFormat display_format, VkFormat depth_format, VkFormat motion_format, uint32_t scene_mip_levels,
        const std::vector<VkImage>& color_images, const std::vector<VkImage>& depth_images,
        const std::vector<VkImage>& motion_images, FrameImagePool& image_pool,
        std::string& error, FrameImageSharing sharing = {});
    void request_frame_generation(bool requested) { frame_generation_requested_ = requested; }
    bool frame_generation_active() const { return frame_generation_context_; }
    void reset_for_reinitialize();
    void destroy();
    bool ready() const { return context_initialized_; }
    bool active() const { return active_; }
    void set_active(bool active) { active_ = ready() && active; }
    void reset_history() { history_valid_ = false; }
    void commit_frame() { if (active_) history_valid_ = true; }
    void set_frame(uint64_t frame_id, float delta_ms, float camera_near, float camera_far,
        float camera_fov_vertical);
    void set_jitter(float x, float y) { jitter_ = {x, y}; }
    bool jitter_offset(uint32_t sample, float& x, float& y) const;
    VkImage output_image(uint32_t image_index) const;
    VkImageView output_view(uint32_t image_index) const;
    static bool active_callback(void* user);
    static bool record_callback(const FrameRecordingContext& frame, void* user, std::string& error);
    static bool frame_generation_ui_active_callback(void* user);
    static bool register_frame_generation_ui_callback(VkImage image, VkFormat format, VkExtent2D extent,
        VkImageUsageFlags usage, void* user, std::string& error);
    static bool swapchain_create_callback(VkSwapchainCreateInfoKHR& create_info,
        const FrameSwapchainCreateContext& context, FrameDispatch& dispatch,
        VkSwapchainKHR& swapchain, bool& frame_interpolation, void* user_data,
        std::string& error);
    static void swapchain_destroyed_callback(void* user_data);

private:
    bool record(const FrameRecordingContext& frame, std::string& error);
    bool register_frame_generation_ui(VkImage image, VkFormat format, VkExtent2D extent,
        VkImageUsageFlags usage, std::string& error);
    void release_output_resources();
#if defined(XRAY_HAS_FSR3_ANDROID)
    bool create_backend_interface(FfxInterface& iface, std::vector<uint8_t>& scratch,
        size_t max_contexts, std::string& error);
    bool initialize_backend_interfaces(std::string& error);
    bool create_context(VkExtent2D render_extent, VkExtent2D display_extent,
        VkFormat back_buffer_format, bool frame_generation, bool hdr_input, std::string& error);
    bool create_frame_generation_swapchain(VkSwapchainCreateInfoKHR& create_info,
        const FrameSwapchainCreateContext& context, FrameDispatch& dispatch,
        VkSwapchainKHR& swapchain, std::string& error);
    void destroy_context();
#endif

    VkInstance instance_{};
    PFN_vkGetInstanceProcAddr get_instance_{};
    VkPhysicalDevice physical_{};
    VkDevice device_{};
    PFN_vkGetDeviceProcAddr get_{};
    FrameImagePool* image_pool_{};
    FrameImageSharing sharing_{};
    VkExtent2D render_extent_{};
    VkExtent2D display_extent_{};
    VkFormat scene_format_ = VK_FORMAT_UNDEFINED;
    VkFormat display_format_ = VK_FORMAT_UNDEFINED;
    VkFormat depth_format_ = VK_FORMAT_UNDEFINED;
    VkFormat motion_format_ = VK_FORMAT_UNDEFINED;
    uint32_t scene_mip_levels_{1};
    PFN_vkCreateImageView create_image_view_{};
    PFN_vkDestroyImageView destroy_image_view_{};
    std::vector<VkImage> color_images_;
    std::vector<VkImage> depth_images_;
    std::vector<VkImage> motion_images_;
    std::vector<VkImage> output_images_;
    std::vector<VkImageView> output_views_;
#if defined(XRAY_HAS_FSR3_ANDROID)
    std::vector<uint8_t> scratch_shared_;
    std::vector<uint8_t> scratch_upscaling_;
    std::vector<uint8_t> scratch_frame_interpolation_;
    FfxInterface backend_shared_{};
    FfxInterface backend_upscaling_{};
    FfxInterface backend_frame_interpolation_{};
    FfxFsr3Context context_{};
#endif
    uint64_t frame_id_{};
    float delta_ms_{16.67f};
    float camera_near_{0.1f};
    float camera_far_{500.f};
    float camera_fov_vertical_{1.04719755f};
    std::array<float, 2> jitter_{};
    int32_t jitter_phase_count_{1};
    bool context_initialized_{};
    bool backend_interfaces_initialized_{};
    bool frame_generation_requested_{};
    bool frame_generation_context_{};
    bool frame_generation_frame_enabled_{};
    VkSwapchainKHR frame_interpolation_swapchain_ = VK_NULL_HANDLE;
    bool active_{};
    bool history_valid_{};
};
}
