#pragma once
#include "GpuParticleBuffers.h"
#include <array>

namespace xray::render::vulkan
{
struct GpuParticleDrawParams
{
    std::array<float, 4> camera_top{}, camera_right{}, camera_direction{}, eye_position{};
    uint32_t bucket{};
    float padding[3]{};
    std::array<float, 4> water_origin{}, water_normal{};
    std::array<float, 4> screen{};
};
static_assert(sizeof(GpuParticleDrawParams) == 128);

// Set 1 is reserved for particle buffers; set 0 remains the game material.
// A separate set and uniform range are required per in-flight draw bucket.
class GpuParticleDrawBindings
{
public:
    ~GpuParticleDrawBindings() { destroy(); }
    GpuParticleDrawBindings() = default;
    GpuParticleDrawBindings(const GpuParticleDrawBindings&) = delete;
    GpuParticleDrawBindings& operator=(const GpuParticleDrawBindings&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, std::string& error);
    bool bind(VkCommandBuffer command, VkPipelineLayout pipeline_layout, uint32_t slot,
        const GpuParticleBuffers& buffers, VkDescriptorBufferInfo params,
        VkDescriptorBufferInfo material_params, std::string& error) const;
    VkDescriptorSetLayout layout() const { return layout_; }
    void destroy();
private:
    VkDevice device_{};
    VkDescriptorSetLayout layout_{};
    VkDescriptorPool pool_{};
    std::vector<VkDescriptorSet> sets_;
    VkDeviceSize max_storage_{}, max_uniform_{}, uniform_alignment_{};
    PFN_vkDestroyDescriptorSetLayout destroy_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkUpdateDescriptorSets update_{};
    PFN_vkCmdBindDescriptorSets bind_{};
};
}
