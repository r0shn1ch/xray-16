#pragma once

#include "FrameContext.h"
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// Single-queue GPU adaptation. The compute pass samples the complete scene,
// accumulates a full-resolution histogram, then updates the shared 1x1 image.
class TemporalExposure
{
public:
    TemporalExposure() = default;
    ~TemporalExposure() { destroy(); }
    TemporalExposure(const TemporalExposure&) = delete;
    TemporalExposure& operator=(const TemporalExposure&) = delete;

    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        const std::vector<VkImageView>& scene_views, const VkPhysicalDeviceMemoryProperties& memory,
        const FrameDispatch& images, VkPipelineCache cache, std::string& error,
        FrameImagePool* image_pool = nullptr, FrameImageSharing sharing = {});
    void set_parameters(float middle_gray, float low_luminance, float delta_seconds, float adaptation_speed);
    void commit_frame() { history_valid_ = true; }
    void reset_history() { history_valid_ = false; }
    bool record(const FrameRecordingContext& frame) const;
    static bool record_callback(const FrameRecordingContext& frame, void* user, std::string& error);
    static bool history_valid_callback(void* user);
    bool ready() const { return pipeline_ != VK_NULL_HANDLE && histogram_pipeline_ != VK_NULL_HANDLE; }
    VkImage image() const { return image_; }
    VkImageView view() const { return view_; }
    VkSampler sampler() const { return sampler_; }
    void destroy();

private:
    VkDevice device_{};
    VkImage image_{};
    VkDeviceMemory memory_{};
    VkImageView view_{};
    VkSampler sampler_{};
    VkBuffer histogram_{};
    VkDeviceMemory histogram_memory_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
    FrameDispatch images_{};
    FrameImagePool* image_pool_{};
    FrameImageSharing sharing_{};
    float middle_gray_{1.f};
    float low_luminance_{0.1f};
    float blend_{1.f};
    bool history_valid_{};
    PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkDestroyBuffer destroy_buffer_{};
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdFillBuffer fill_buffer_{};
    PFN_vkCmdPipelineBarrier barrier_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDispatch dispatch_{};
    VkPipeline histogram_pipeline_{};
};
} // namespace xray::render::vulkan
