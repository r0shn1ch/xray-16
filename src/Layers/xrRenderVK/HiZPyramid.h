#pragma once

#include "FrameContext.h"
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// Maximum depth reduction, with one half-resolution R32F pyramid per acquired
// image. The caller waits for GPU idle before destroying or resizing it.
// An optional image pool must outlive the pyramid and its Vulkan device.
class HiZPyramid
{
public:
    ~HiZPyramid() { destroy(); }
    HiZPyramid(const HiZPyramid&) = delete;
    HiZPyramid& operator=(const HiZPyramid&) = delete;
    HiZPyramid() = default;

    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        VkExtent2D scene_extent, const std::vector<VkImageView>& depth_views,
        const std::vector<VkImage>& depth_images,
        const std::vector<VkImageView>& motion_views, const std::vector<VkImage>& motion_images,
        const VkPhysicalDeviceMemoryProperties& memory, const FrameDispatch& images,
        VkPipelineCache cache, std::string& error, FrameImagePool* image_pool = nullptr,
        const FrameImageSharing* sharing = nullptr);
    bool record(const FrameRecordingContext& frame, bool reject_dynamic_occluders);
    void commit_history(bool dynamic_mask_recorded = true);
    VkImage image(uint32_t image_index) const;
    VkImageView view(uint32_t image_index) const;
    VkImage previous_image() const { return history_valid_ ? image(last_image_index_) : VK_NULL_HANDLE; }
    VkImageView previous_view() const { return history_valid_ ? view(last_image_index_) : VK_NULL_HANDLE; }
    const FrameImageSharing& sharing() const { return sharing_; }
    VkSampler sampler() const { return sampler_; }
    bool history_valid() const { return history_valid_ && last_image_index_ < targets_.size(); }
    bool previous_view_is_dynamic_safe() const { return history_valid() && last_dynamic_safe_; }
    void reset_history()
    {
        history_valid_ = false;
        last_dynamic_safe_ = pending_dynamic_safe_ = false;
        last_image_index_ = pending_image_index_ = UINT32_MAX;
    }
    uint32_t levels() const { return levels_; }
    VkExtent2D extent() const { return extent_; }
    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }
    void destroy();

private:
    struct Target
    {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
        std::vector<VkImageView> mips;
        std::vector<VkDescriptorSet> sets;
        bool initialized{};
    };
    bool create_target(Target& target, std::string& error);
    VkDevice device_{};
    VkExtent2D extent_{};
    uint32_t levels_{};
    VkSampler sampler_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    FrameDispatch images_{};
    FrameImagePool* image_pool_{};
    FrameImageSharing sharing_{};
    PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDispatch dispatch_{};
    std::vector<Target> targets_;
    std::vector<VkImage> depth_images_;
    std::vector<VkImage> motion_images_;
    uint32_t last_image_index_{UINT32_MAX};
    uint32_t pending_image_index_{UINT32_MAX};
    bool pending_dynamic_safe_{};
    bool history_valid_{};
    bool last_dynamic_safe_{};
};
} // namespace xray::render::vulkan
