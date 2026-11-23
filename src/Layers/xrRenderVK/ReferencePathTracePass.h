#pragma once

#include "RaySurface.h"
#include "BindlessTextureTable.h"

namespace xray::render::vulkan
{
struct ReferencePathTraceParameters
{
    float sky[4]{};
    float sun_direction_distance[4]{};
    float sun_color[4]{};
    uint32_t dimensions[4]{};
    uint32_t control[4]{};
    float limits[4]{};
};
static_assert(sizeof(ReferencePathTraceParameters) == 96);

struct ReferencePathTraceHistoryCell
{
    float radiance_count[4]{};
    float position[3]{};
    uint32_t packed_normal{};
};
static_assert(sizeof(ReferencePathTraceHistoryCell) == 32);

class ReferencePathTracePass
{
public:
    ~ReferencePathTracePass() { destroy(); }
    ReferencePathTracePass() = default;
    ReferencePathTracePass(const ReferencePathTracePass&) = delete;
    ReferencePathTracePass& operator=(const ReferencePathTracePass&) = delete;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
        VkDeviceSize max_storage_range, VkPipelineCache cache,
        const BindlessTextureTable& textures, VkDescriptorSetLayout projection_layout,
        std::string& error);
    bool record(VkCommandBuffer command, uint32_t frame_slot, const LevelRayScene& scene,
        const BufferResource& primary_rays, const BufferResource& primary_surfaces,
        const BufferResource& previous_history, const BufferResource& next_history,
        const BufferResource& local_lights,
        VkDescriptorSet projection_set,
        VkImageView motion_view, VkSampler motion_sampler, const BufferResource& radiance,
        const ReferencePathTraceParameters& parameters,
        std::string& error) const;
    void destroy();

private:
    VkDevice device_{};
    VkDeviceSize max_storage_range_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    VkDescriptorSet texture_set_{};
    BindlessTextureTable projection_fallback_;
    uint32_t texture_capacity_{};
    std::vector<VkDescriptorSet> sets_;
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkUpdateDescriptorSets update_sets_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDispatch dispatch_{};
};
}
