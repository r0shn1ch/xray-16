#pragma once
#include "BufferResource.h"
#include "FrameContext.h"
#include <vector>

namespace xray::render::vulkan
{
struct VsmDemandParameters
{
    float inverse_view_projection[16]{};
    float light_view[16]{};
    float level[6][4]{}; // light-space origin xy, width, reserved
    uint32_t axis{}, levels{}, step{}, lod_bias{};
    uint32_t page_size{}; // Zero requests only receiver pages.
    float near_depth{}, inverse_depth_range{}; // Zero inverse range disables depth clipping.
    uint32_t reserved{};
};
static_assert(sizeof(VsmDemandParameters) == 256);

// Each image has its own writable parameter and demand buffers. The caller
// waits for the matching frame/image fence before reusing them; needed buffers
// require TRANSFER_DST and STORAGE usage and all resources outlive this pass.
class VsmDemandPass
{
public:
    ~VsmDemandPass() { destroy(); }
    VsmDemandPass() = default;
    VsmDemandPass(const VsmDemandPass&) = delete;
    VsmDemandPass& operator=(const VsmDemandPass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
        const std::vector<VkImageView>& depths, VkSampler sampler,
        const std::vector<BufferResource*>& parameters, const std::vector<BufferResource*>& needed,
        VkPipelineCache cache, std::string& error);
    bool record(const FrameRecordingContext& frame, const VsmDemandParameters& parameters, std::string& error) const;
    void destroy();
private:
    VkDevice device_{};
    VkExtent2D extent_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
    std::vector<BufferResource*> parameters_, needed_;
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdFillBuffer fill_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
