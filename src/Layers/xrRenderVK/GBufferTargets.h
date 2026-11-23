#pragma once

#include "DeferredPass.h"
#include "FrameImagePool.h"

namespace xray::render::vulkan
{
class SunShadowTargets;
struct GBufferMsaaConfig
{
    VkSampleCountFlagBits samples{VK_SAMPLE_COUNT_1_BIT};
    VkResolveModeFlagBits depth_resolve{VK_RESOLVE_MODE_NONE};
    VkResolveModeFlagBits stencil_resolve{VK_RESOLVE_MODE_NONE};
    PFN_vkCreateRenderPass2 create_render_pass{};
};
// Queries the actual attachment format/usage combinations and chooses the
// highest supported count no greater than the request. The caller must have
// enabled Vulkan 1.2 or the equivalent render-pass/depth-resolve extensions.
bool negotiate_gbuffer_msaa(VkPhysicalDevice physical, const VkPhysicalDeviceProperties& properties,
    VkFormat color_format, VkFormat depth_format, VkSampleCountFlagBits requested,
    PFN_vkGetPhysicalDeviceImageFormatProperties image_properties,
    const VkPhysicalDeviceDepthStencilResolveProperties& resolve_properties,
    PFN_vkCreateRenderPass2 create_render_pass, GBufferMsaaConfig& result, std::string& error,
    VkFormat scene_color_format = VK_FORMAT_UNDEFINED);
// Offscreen target triplets are indexed by the acquired swapchain image, so
// their reuse follows FrameContext's image fences. Destroy after GPU idle and
// before destroying the device, command pool or DeferredPass descriptor pool.
class GBufferTargets
{
public:
    GBufferTargets() = default;
    ~GBufferTargets() { destroy(); }
    GBufferTargets(const GBufferTargets&) = delete;
    GBufferTargets& operator=(const GBufferTargets&) = delete;

    bool initialize(VkPhysicalDevice physical_device, VkDevice device, VkExtent2D extent, uint32_t image_count,
        VkFormat depth_format, const VkPhysicalDeviceMemoryProperties& memory,
        const FrameDispatch& dispatch, PFN_vkCreateSampler create_sampler,
        PFN_vkDestroySampler destroy_sampler, std::string& error,
        // The pool outlives these targets. Destroy targets after GPU idle and
        // before resetting the pool or destroying its Vulkan device.
        FrameImagePool* image_pool = nullptr, bool enable_depth_readback = false,
        const GBufferMsaaConfig* msaa = nullptr,
        const FrameImageSharing* sharing = nullptr);
    bool bind_lighting(DeferredPass& pass, std::string& error);
    void bind_sun_shadow(DeferredPass& pass, const SunShadowTargets& shadow);
    void bind_rain_shadow(DeferredPass& pass, const SunShadowTargets& shadow);
    void release_lighting(DeferredPass& pass);
    bool begin(const FrameRecordingContext& frame, FrameRecordingContext& geometry_frame) const;
    void end(VkCommandBuffer command) const;
    // Returns the render-pass attachments in declaration order: the three
    // multisampled colors and depth followed by their resolved attachments,
    // or the four single-sample attachments when MSAA is disabled.
    uint32_t graph_attachment_count() const;
    VkImage graph_attachment_image(uint32_t image_index, uint32_t attachment_index) const;
    VkImageAspectFlags graph_attachment_aspect(uint32_t attachment_index) const;
    bool lighting_samples_attachment(uint32_t attachment_index) const;
    bool copy_depth(const FrameRecordingContext& frame) const;
    bool copy_scene_depth(const FrameRecordingContext& frame) const;
    bool record_depth_copy(VkCommandBuffer command, uint32_t image_index) const;
    VkDescriptorSet lighting_set(uint32_t image_index) const;
    VkDescriptorSet multisampled_lighting_set(uint32_t image_index) const;
    // Valid after the geometry pass; the image is stored in read-only depth layout.
    VkImageView depth_view(uint32_t image_index) const;
    VkImage depth_image(uint32_t image_index) const;
    // The MSAA scene pass writes a separate depth copy so lighting can sample
    // the original multisampled G-buffer depth in the same frame.
    VkImageView scene_depth_view(uint32_t image_index) const;
    VkImage scene_depth_image(uint32_t image_index) const;
    VkImageView sampled_depth_view(uint32_t image_index) const;
    VkImage sampled_depth_image(uint32_t image_index) const;
    // Available after the geometry pass in SHADER_READ_ONLY_OPTIMAL layout.
    VkImageView albedo_view(uint32_t image_index) const;
    VkImage albedo_image(uint32_t image_index) const;
    VkImageView normal_view(uint32_t image_index) const;
    VkImage normal_image(uint32_t image_index) const;
    VkImageView surface_view(uint32_t image_index) const;
    VkImage surface_image(uint32_t image_index) const;
    VkSampler sampler() const { return sampler_; }
    const FrameImageSharing& image_sharing() const { return sharing_; }
    VkRenderPass render_pass() const { return pass_; }
    VkSampleCountFlagBits samples() const { return samples_; }
    void destroy();

private:
    static constexpr size_t MultisampledDepthAttachment = 3;
    struct Attachment
    {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
    };
    struct Target
    {
        Attachment albedo, normal, surface, depth, sampled_depth, scene_depth;
        std::array<Attachment, 4> multisampled;
        VkFramebuffer framebuffer{};
        VkDescriptorSet lighting{}, multisampled_lighting{};
    };
    bool create_attachment(VkFormat format, VkImageUsageFlags usage,
        VkImageAspectFlags aspect, Attachment& attachment, std::string& error,
        VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);

    VkDevice device_{};
    VkExtent2D extent_{};
    VkSampleCountFlagBits samples_{VK_SAMPLE_COUNT_1_BIT};
    VkRenderPass pass_{};
    VkSampler sampler_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    FrameDispatch vk_{};
    PFN_vkDestroySampler destroy_sampler_{};
    FrameImagePool* image_pool_{};
    FrameImageSharing sharing_{};
    std::vector<Target> targets_;
};
}
