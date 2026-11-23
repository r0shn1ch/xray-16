#pragma once
#include "GpuParticleDrawBindings.h"
#include "FrameContext.h"

namespace xray::render::vulkan
{
// One graphics pipeline for one compatible render pass and blend/depth state.
// Material uniforms preserve the existing 256-byte Vulkan game ABI; only the
// 64-byte view-projection matrix uses push constants.
class GpuParticleDrawPass
{
public:
    ~GpuParticleDrawPass() { destroy(); }
    GpuParticleDrawPass() = default;
    GpuParticleDrawPass(const GpuParticleDrawPass&) = delete;
    GpuParticleDrawPass& operator=(const GpuParticleDrawPass&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        VkRenderPass render_pass, VkDescriptorSetLayout material_layout, VkDescriptorSetLayout particle_layout,
        VkShaderModule vertex, VkShaderModule fragment, VkPipelineCache cache, uint32_t blend,
        bool hud, bool gbuffer, std::string& error, VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE,
        VkSampleCountFlagBits gbuffer_samples = VK_SAMPLE_COUNT_1_BIT,
        bool velocity = false, bool transparent_velocity = false);
    bool record(const FrameRecordingContext& frame, const GpuParticleDrawBindings& bindings,
        const GpuParticleBuffers& buffers, VkDescriptorSet material, VkDescriptorBufferInfo camera,
        VkDescriptorBufferInfo material_params, uint32_t descriptor_slot, uint32_t bucket,
        const void* view_projection, std::string& error, VkDescriptorSet scene = VK_NULL_HANDLE) const;
    void destroy();
private:
    VkDevice device_{}; VkRenderPass render_pass_{}; VkPipelineLayout layout_{}; VkPipeline pipeline_{};
    bool has_scene_{};
    bool velocity_{};
    VkSampleCountFlagBits samples_{VK_SAMPLE_COUNT_1_BIT};
    PFN_vkDestroyPipelineLayout destroy_layout_{}; PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBindPipeline bind_pipeline_{}; PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdSetViewport viewport_{}; PFN_vkCmdSetScissor scissor_{};
    PFN_vkCmdPushConstants push_{}; PFN_vkCmdDrawIndirect draw_{};
};
}
