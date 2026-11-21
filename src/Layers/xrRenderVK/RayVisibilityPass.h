#pragma once
#include "RayQueryBatch.h"
#include "FrameContext.h"

namespace xray::render::vulkan
{
// Image and hit buffer lifetimes and the initial image transition belong to the caller.
// record() leaves the image in SHADER_READ_ONLY_OPTIMAL for fragment sampling.
class RayVisibilityPass
{
public:
    enum class Mode { Visibility, IndirectRadiance };
    ~RayVisibilityPass() { destroy(); }
    RayVisibilityPass() = default;
    RayVisibilityPass(const RayVisibilityPass&) = delete;
    RayVisibilityPass& operator=(const RayVisibilityPass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
        const std::vector<VkImage>& images, const std::vector<VkImageView>& views,
        const std::vector<BufferResource*>& hits, VkPipelineCache cache, std::string& error,
        Mode mode = Mode::Visibility);
    bool record(const FrameRecordingContext& frame, VkImageLayout old_layout) const;
    void destroy();
private:
    VkDevice device_{};
    VkExtent2D extent_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
    std::vector<VkImage> images_;
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
