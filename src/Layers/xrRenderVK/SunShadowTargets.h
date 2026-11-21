#pragma once

#include "BufferResource.h"
#include "FrameContext.h"

namespace xray::render::vulkan
{
struct alignas(16) SunShadowUniform
{
    float inverse_view_projection[16]{};
    float sun_view_projection[16]{};
    float options[4]{0.0015f, 0.f, 0.f, 0.f};
    float ambient_color[4]{}; // RGB ambient; alpha enables RT visibility sampling
    float hemi_color[4]{};
    float rain_view_projection[16]{};
    // Density, near/far distance falloff, and animation time.
    float rain_params[4]{};
    // Legacy fog visibility plane and its weather color.
    float fog_plane[4]{0.f, 0.f, 0.f, 1.f};
    float fog_color[4]{};
    float ao_settings[4]{};
};
static_assert(sizeof(SunShadowUniform) == 304);

// Each acquired swapchain image owns a depth map and its uniform buffer;
// neither is overwritten while that image may still be in flight.
class SunShadowTargets
{
public:
    ~SunShadowTargets() { destroy(); }
    bool initialize(VkDevice device, VkFormat depth_format, uint32_t image_count,
        const VkPhysicalDeviceMemoryProperties& memory, const FrameDispatch& dispatch,
        const BufferResourceDispatch& buffers, PFN_vkCreateSampler create_sampler,
        PFN_vkDestroySampler destroy_sampler, std::string& error, FrameImagePool* image_pool = nullptr,
        uint32_t size = Size);
    bool begin(const FrameRecordingContext& frame, const SunShadowUniform& constants,
        FrameRecordingContext& shadow_frame, std::string& error);
    void end(VkCommandBuffer command) const;
    bool set_visibility_enabled(uint32_t image, bool enabled, std::string& error);
    VkRenderPass render_pass() const { return pass_; }
    VkImage image(uint32_t index) const { return index < targets_.size() ? targets_[index].image : VK_NULL_HANDLE; }
    VkImageView view(uint32_t index) const { return index < targets_.size() ? targets_[index].view : VK_NULL_HANDLE; }
    VkBuffer uniform(uint32_t index) const { return index < targets_.size() ? targets_[index].constants.handle() : VK_NULL_HANDLE; }
    VkSampler sampler() const { return sampler_; }
    uint32_t size() const { return size_; }
    void swap(SunShadowTargets& other) noexcept;
    void destroy();
    static constexpr uint32_t Size = 1024;

private:
    struct Target
    {
        VkImage image{};
        VkImageView view{};
        VkDeviceMemory memory{};
        VkFramebuffer framebuffer{};
        BufferResource constants;
    };
    VkDevice device_{};
    VkRenderPass pass_{};
    VkSampler sampler_{};
    uint32_t size_{Size};
    FrameDispatch vk_{};
    PFN_vkDestroySampler destroy_sampler_{};
    FrameImagePool* image_pool_{};
    std::vector<Target> targets_;
};
}
