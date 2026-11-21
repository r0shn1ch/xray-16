#pragma once

#include "FrameContext.h"
#include <array>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// Reprojects opaque depth through the previous camera into half-float UV velocity.
// RGBA16F also stores projected previous-camera depth in its third channel.
// A depth-equal raster overlay replaces camera velocity for moving rigid and
// CPU-skinned opaque models and moved linked level visuals; cutouts, GPU
// skinning, particles and LOD billboards remain separate.
// An optional image pool must outlive the targets; destroy after GPU idle.
class MotionVectors
{
public:
    MotionVectors() = default;
    ~MotionVectors() { destroy(); }
    MotionVectors(const MotionVectors&) = delete;
    MotionVectors& operator=(const MotionVectors&) = delete;

    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        VkExtent2D extent, const std::vector<VkImageView>& depth_views,
        const std::vector<VkImage>& depth_images,
        const VkPhysicalDeviceMemoryProperties& memory, const FrameDispatch& images,
        VkPipelineCache cache, bool storage_extended_formats, std::string& error,
        FrameImagePool* image_pool = nullptr, bool prefer_reprojected_depth = false,
        const FrameImageSharing* sharing = nullptr);
    void set_view(const float* view_projection, const float* inverse_view_projection);
    void commit_view();
    void reset_history();
    bool history_valid() const { return history_valid_; }
    const float* previous_view_projection() const { return previous_vp_.data(); }
    bool record(const FrameRecordingContext& frame, std::string* error = nullptr) const;
    VkImage image(uint32_t index) const;
    VkImageView view(uint32_t index) const;
    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }
    VkFormat format() const { return format_; }
    bool color_attachment_ready() const { return color_attachment_ready_; }
    void destroy();

private:
    struct Target
    {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
        VkDescriptorSet set{};
    };
    bool create_target(Target& target, std::string& error);
    VkDevice device_{};
    VkExtent2D extent_{};
    VkFormat format_{VK_FORMAT_UNDEFINED};
    bool color_attachment_ready_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    FrameDispatch images_{};
    FrameImagePool* image_pool_{};
    FrameImageSharing sharing_{};
    std::vector<Target> targets_;
    std::vector<VkImage> depth_images_;
    std::array<float, 16> current_vp_{};
    std::array<float, 16> current_inverse_{};
    std::array<float, 16> previous_vp_{};
    bool history_valid_{};
    VkSampler sampler_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDispatch dispatch_{};
};
} // namespace xray::render::vulkan
