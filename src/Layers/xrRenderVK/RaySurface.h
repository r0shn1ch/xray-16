#pragma once

#include <cstdint>
#include <vulkan/vulkan.h>
#include <string>
#include <vector>

#include "RayQueryBatch.h"

namespace xray::render::vulkan
{
// std430 output: position_valid[3] is 1 for a reconstructed hit, 0 for a miss,
// and -1 for a hit with unavailable shading attributes.
struct RaySurface
{
    float position_valid[4]{};
    float normal[4]{};
    float uv[2]{};
    uint32_t material_id{UINT32_MAX};
    uint32_t texture_slot{UINT32_MAX}; // Index into GpuLevel::ray_diffuse_texture_names().
    float albedo[4]{1.f, 1.f, 1.f, 1.f};
    // metallic, roughness, AO, flags (bit 0 = packed PBR map, bit 1 = emissive)
    float pbr[4]{0.f, 1.f, 1.f, 0.f};
    float geometric_normal[4]{}; // Face normal oriented against the incoming ray.
};
static_assert(sizeof(RaySurface) == 96);

// Descriptor slots may only be reused after the corresponding frame fence.
class RaySurfacePass
{
public:
    ~RaySurfacePass() { destroy(); }
    RaySurfacePass() = default;
    RaySurfacePass(const RaySurfacePass&) = delete;
    RaySurfacePass& operator=(const RaySurfacePass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
        const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceDescriptorIndexingProperties& indexing,
        const BindlessTextureTable& textures, VkPipelineCache cache, std::string& error);
    bool record(VkCommandBuffer command, uint32_t frame_slot, const LevelRayScene& scene,
        const BufferResource& rays, const BufferResource& hits, const BufferResource& surfaces,
        uint32_t count, std::string& error) const;
    void destroy();

private:
    VkDevice device_{};
    VkDeviceSize max_storage_range_{};
    VkDescriptorSet texture_set_{};
    uint32_t texture_capacity_{};
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
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
