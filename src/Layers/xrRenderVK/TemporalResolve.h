#pragma once

#include "FrameContext.h"
#include <array>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// Ping-pong full-resolution color history. Caller waits for GPU idle before
// destroy/reinitialize and commits the output slot only after a successful submit.
class TemporalResolve
{
public:
    TemporalResolve() = default;
    ~TemporalResolve() { destroy(); }
    TemporalResolve(const TemporalResolve&) = delete;
    TemporalResolve& operator=(const TemporalResolve&) = delete;

    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        VkExtent2D extent, const std::vector<VkImageView>& scene_views,
        const std::vector<VkImageView>& motion_views,
        const std::vector<VkImageView>& depth_views,
        const VkPhysicalDeviceMemoryProperties& memory, const FrameDispatch& images,
        VkPipelineCache cache, std::string& error, FrameImagePool* image_pool = nullptr,
        bool reprojected_depth = false, bool indirect = false, bool upscale = false,
        FrameImageSharing sharing = {});
    bool record(const FrameRecordingContext& frame);
    // Standalone mode owns history-image transitions; commit after submission.
    bool record_standalone(const FrameRecordingContext& frame);
    void commit_standalone() { layouts_initialized_ |= recorded_; commit_frame(); }
    static void record_callback(const FrameRecordingContext& frame, void* user);
    static bool history_valid_callback(void* user);
    static bool active_callback(void* user);
    static uint32_t output_slot_callback(void* user);
    void commit_frame()
    {
        history_valid_ = recorded_;
        if (!recorded_) return;
        recorded_ = false;
        previous_jitter_ = current_jitter_;
        output_slot_ ^= 1;
    }
    void reset_history() { history_valid_ = recorded_ = false; }
    void set_active(bool active) { if (!active) reset_history(); active_ = active; }
    void set_history_weight(float weight);
    void set_jitter(float x, float y);
    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }
    bool active() const { return active_; }
    bool history_valid() const { return history_valid_; }
    uint32_t output_slot() const { return output_slot_; }
    VkImage image(uint32_t slot) const { return slot < 2 ? targets_[slot].image : VK_NULL_HANDLE; }
    VkImageView view(uint32_t slot) const { return slot < 2 ? targets_[slot].view : VK_NULL_HANDLE; }
    void destroy();

private:
    struct Target { VkImage image{}; VkDeviceMemory memory{}; VkImageView view{}; };
    bool create_target(Target& target, const VkPhysicalDeviceMemoryProperties& memory, std::string& error);
    VkDevice device_{};
    VkExtent2D extent_{};
    std::array<Target, 2> targets_{};
    FrameDispatch images_{};
    FrameImagePool* image_pool_{};
    FrameImageSharing sharing_{};
    VkSampler sampler_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
    uint32_t output_slot_{};
    bool history_valid_{};
    bool recorded_{};
    bool active_{};
    bool reprojected_depth_{};
    bool indirect_{};
    bool upscale_{};
    bool layouts_initialized_{};
    std::array<float, 2> current_jitter_{}, previous_jitter_{};
    float history_weight_{.9f};
    PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
