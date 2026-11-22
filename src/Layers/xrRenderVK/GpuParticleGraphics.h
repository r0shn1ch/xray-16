#pragma once
#include "GpuParticleDrawPass.h"
#include "GpuParticleUniforms.h"
#include "GpuParticleCompute.h"
#include <memory>
#include <unordered_map>

namespace xray::render::vulkan
{
// Per-frame descriptor sets and uniforms cannot be reused until the frame fence
// completes. Pipelines are retained until reset after all submitted work ends.
class GpuParticleGraphics
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        VkDescriptorSetLayout material_layout, VkPipelineCache cache, uint32_t frames,
        uint32_t draws_per_frame, GpuParticleCompute::Source source, std::string& error,
        VkSampleCountFlagBits gbuffer_samples = VK_SAMPLE_COUNT_1_BIT,
        VkRenderPass velocity_render_pass = VK_NULL_HANDLE);
    bool begin_frame(uint32_t frame, std::string& error);
    bool record(const FrameRecordingContext& frame, const GpuParticleBuffers& buffers,
        VkDescriptorSet material, const std::string& fragment, uint32_t blend, bool hud, bool gbuffer,
        const GpuParticleDrawParams& camera, const float* material_constants, const float* mvp,
        std::string& error, VkDescriptorSetLayout scene_layout = VK_NULL_HANDLE, VkDescriptorSet scene = VK_NULL_HANDLE);
    bool record_velocity(const FrameRecordingContext& frame, const GpuParticleBuffers& buffers,
        VkDescriptorSet material, const GpuParticleDrawParams& camera, const float* material_constants,
        const float* current_mvp, const float* previous_mvp, bool transparent, std::string& error);
    void clear_pipelines(); // Caller must wait for GPU idle first.
    void rebind_gbuffer_samples(VkSampleCountFlagBits samples);
    void destroy(); // Caller must wait for GPU idle first.
    ~GpuParticleGraphics() { destroy(); }
private:
    struct Slot { GpuParticleUniforms uniforms; uint32_t draws{}; };
    VkDevice device_{};
    PFN_vkGetDeviceProcAddr get_{};
    VkPhysicalDeviceLimits limits_{};
    VkDescriptorSetLayout material_layout_{};
    VkPipelineCache cache_{};
    uint32_t draws_per_frame_{};
    VkSampleCountFlagBits gbuffer_samples_{VK_SAMPLE_COUNT_1_BIT};
    GpuParticleCompute::Source source_;
    GpuParticleDrawBindings bindings_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::unordered_map<std::string, std::unique_ptr<GpuParticleDrawPass>> pipelines_;
    GpuParticleDrawPass velocity_opaque_, velocity_transparent_;
};
}
