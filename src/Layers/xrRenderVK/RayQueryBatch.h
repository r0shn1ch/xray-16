#pragma once

#include "BufferResource.h"
#include "LevelRayScene.h"
#include "BindlessTextureTable.h"
#include <vector>

namespace xray::render::vulkan
{
struct RayQueryInput
{
    float origin_min[4]{};
    float direction_max[4]{};
};
static_assert(sizeof(RayQueryInput) == 32);

struct RayQueryHit
{
    uint32_t found{};
    float distance{};
    uint32_t instance{};
    uint32_t geometry{};
    uint32_t primitive{};
    uint32_t material_id{};
    float barycentric[2]{};
};
static_assert(sizeof(RayQueryHit) == 32);

// Query result indices address LevelRayScene's batch and geometry tables.
// The caller reuses a descriptor slot only after its frame fence completes.
class RayQueryBatch
{
public:
    static constexpr uint32_t MaxGroupsX = 4096;
    ~RayQueryBatch() { destroy(); }
    RayQueryBatch() = default;
    RayQueryBatch(const RayQueryBatch&) = delete;
    RayQueryBatch& operator=(const RayQueryBatch&) = delete;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
        uint32_t frame_slots, VkPipelineCache cache,
        const BindlessTextureTable& textures, std::string& error);
    bool record(VkCommandBuffer command, uint32_t frame_slot,
        const LevelRayScene& scene, const BufferResource& rays,
        const BufferResource& hits, uint32_t ray_count, std::string& error) const;
    void destroy();
    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkDescriptorSet texture_set_ = VK_NULL_HANDLE;
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
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
