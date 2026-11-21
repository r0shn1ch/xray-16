#pragma once
#include "RayQueryBatch.h"
#include "FrameContext.h"

namespace xray::render::vulkan
{
// Depth and normal views, sampler, and output buffers outlive this pass.
// Descriptor sets are per swapchain image; callers synchronize buffer reuse.
class RayGeneratePass
{
public:
    enum class Mode { Sun, Indirect };
    ~RayGeneratePass() { destroy(); }
    RayGeneratePass() = default;
    RayGeneratePass(const RayGeneratePass&) = delete;
    RayGeneratePass& operator=(const RayGeneratePass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
        const std::vector<VkImageView>& depth, const std::vector<VkImageView>& normal,
        VkSampler sampler, const std::vector<BufferResource*>& outputs,
        VkPipelineCache cache, std::string& error, Mode mode = Mode::Sun,
        const std::vector<BufferResource*>& primary_surface_outputs = {});
    // In Indirect mode .x carries a bit-cast frame seed, .yz the output extent and .w the maximum distance.
    bool record(const FrameRecordingContext& frame, const float* inverse_view_projection,
        const float* direction_distance) const;
    void destroy();
private:
    VkDevice device_{};
    VkExtent2D extent_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
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
