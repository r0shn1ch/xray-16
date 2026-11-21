#pragma once
#include "BufferResource.h"
#include "FrameContext.h"
#include <vector>

namespace xray::render::vulkan
{
struct VsmResolveParameters
{
    float inverse_view_projection[16]{};
    float light_view[16]{};
    float level[6][4]{};
    uint32_t axis{}, levels{}, page_size{}, static_columns{};
    uint32_t dynamic_columns{}, dynamic_enabled{}, static_slots{}, dynamic_slots{};
    float near_depth{}, inverse_depth_range{}, bias{}, reserved{};
};
static_assert(sizeof(VsmResolveParameters) == 272);

// R32_SFLOAT output images need STORAGE and SAMPLED usage. Atlas sampling,
// page table writes and per-image parameter updates are synchronized by the caller.
class VsmResolvePass
{
public:
    ~VsmResolvePass() { destroy(); }
    VsmResolvePass() = default;
    VsmResolvePass(const VsmResolvePass&) = delete;
    VsmResolvePass& operator=(const VsmResolvePass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
        const std::vector<VkImageView>& depths, VkImageView static_atlas, VkImageView dynamic_atlas,
        VkSampler sampler, const std::vector<BufferResource*>& static_tables,
        const std::vector<BufferResource*>& dynamic_tables, const std::vector<BufferResource*>& parameters,
        const std::vector<VkImage>& images, const std::vector<VkImageView>& views,
        VkPipelineCache cache, std::string& error);
    bool record(const FrameRecordingContext& frame, const VsmResolveParameters& parameters,
        VkImageLayout old_layout, std::string& error, uint32_t table_image = UINT32_MAX,
        const BufferResource* static_override = nullptr, const BufferResource* dynamic_override = nullptr,
        VkImageView ray_view = VK_NULL_HANDLE, VkSampler ray_sampler = VK_NULL_HANDLE) const;
    void destroy();
private:
    VkDevice device_{}; VkExtent2D extent_{};
    VkDescriptorSetLayout set_layout_{}; VkDescriptorPool pool_{};
    VkPipelineLayout layout_{}; VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
    std::vector<VkImage> images_;
    std::vector<VkImageView> fallback_views_;
    VkSampler fallback_sampler_{};
    std::vector<BufferResource*> static_tables_, dynamic_tables_, parameters_;
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdPipelineBarrier barrier_{};
    PFN_vkUpdateDescriptorSets update_sets_{};
};
}
