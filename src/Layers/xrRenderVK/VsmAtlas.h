#pragma once
#include "FrameContext.h"

namespace xray::render::vulkan
{
// One persistent physical depth atlas. Caller owns queue/fence synchronization:
// atlas sampling and page writes must not overlap. initialize_layout() must be
// submitted and completed before mark_initialized() and the first page pass.
class VsmAtlas
{
public:
    ~VsmAtlas() { destroy(); }
    VsmAtlas() = default;
    VsmAtlas(const VsmAtlas&) = delete;
    VsmAtlas& operator=(const VsmAtlas&) = delete;
    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        VkFormat format, uint32_t page_size, uint32_t columns, uint32_t rows,
        const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
        const FrameDispatch& dispatch, std::string& error, bool allow_readback = false,
        FrameImagePool* image_pool = nullptr);
    bool initialize_layout(VkCommandBuffer command, std::string& error) const;
    void mark_initialized() { initialized_ = true; }
    bool begin_page(const FrameRecordingContext& frame, uint32_t slot, FrameRecordingContext& page_frame,
        std::string& error) const;
    void end_page(VkCommandBuffer command) const;
    void destroy();
    VkImage image() const { return image_; }
    VkImageView view() const { return view_; }
    VkSampler sampler() const { return sampler_; }
    VkRenderPass render_pass() const { return pass_; }
    VkExtent2D extent() const { return extent_; }
    uint32_t page_size() const { return page_size_; }
    uint64_t page_count() const { return uint64_t(columns_) * rows_; }
private:
    VkDevice device_{};
    FrameDispatch vk_{};
    VkImage image_{};
    VkDeviceMemory memory_{};
    FrameImagePool* image_pool_{};
    VkImageView view_{};
    VkFramebuffer framebuffer_{};
    VkRenderPass pass_{};
    VkSampler sampler_{};
    VkExtent2D extent_{};
    uint32_t page_size_{}, columns_{}, rows_{};
    bool initialized_{};
    PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkCmdClearDepthStencilImage clear_depth_{};
};
}
