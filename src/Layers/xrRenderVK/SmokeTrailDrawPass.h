#pragma once
#include "SmokeTrailCompute.h"
#include "FrameContext.h"
#include "TrailState.h"

namespace xray::render::vulkan
{
struct SmokeTrailDrawParams
{
    uint32_t point_count{}, subdivisions{}; float texcoord_factor{}; uint32_t uv_policy{};
    float total_distance{}; uint32_t tail_fade{}, smoothing{}, edge_policy{};
    uint32_t flip_x{}, flip_y{}, rotate90{}, material_id{};
    uint32_t use_gpu_state{1}; float turbulence{}, frequency{}, evolution{};
    float center_x{}, center_y{}, center_z{}, sphere_radius{};
    float camera_x{}, camera_y{}, camera_z{}; uint32_t camera_facing{};
};
static_assert(sizeof(SmokeTrailDrawParams) == 96);

// Smoke mode reserves one set per frame; trail mode reserves one per group.
// Record at most once per frame slot and wait for its fence before reuse.
// The caller waits for the device before destroy.
class SmokeTrailDrawPass
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        VkRenderPass render_pass, VkPipelineCache cache, uint32_t frames,
        const GpuParticleCompute::Source& source, std::string& error, bool trail_mode = false,
        VkSampleCountFlagBits scene_samples = VK_SAMPLE_COUNT_1_BIT);
    bool record(const FrameRecordingContext& frame, const SmokeTrailCompute& smoke, VkImageView noise,
        VkSampler sampler, const SmokeTrailDrawParams& params, const float* mvp, std::string& error);
    bool record_trail(const FrameRecordingContext& frame, const std::vector<TrailState::Group>& groups,
        const SmokeTrailDrawParams& params, const float* mvp, VkImageView texture, VkSampler sampler, std::string& error);
    void destroy();
    ~SmokeTrailDrawPass() { destroy(); }
private:
    VkDevice device_{}; VkRenderPass render_pass_{};
    VkDescriptorSetLayout set_layout_{}; VkDescriptorPool pool_{};
    VkPipelineLayout layout_{}; VkPipeline pipeline_{};
    std::vector<VkDescriptorSet> sets_;
    std::vector<std::unique_ptr<GpuParticleUniforms>> uniforms_;
    std::vector<std::unique_ptr<BufferResource>> trail_buffers_;
    VkDeviceSize trail_alignment_{};
    bool trail_mode_{};
    static constexpr uint32_t MaxTrailGroups = 128;
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{}; PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{}; PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkUpdateDescriptorSets update_{}; PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{}; PFN_vkCmdSetViewport viewport_{}; PFN_vkCmdSetScissor scissor_{};
    PFN_vkCmdPushConstants push_{}; PFN_vkCmdDrawIndirect draw_{};
    PFN_vkCmdDraw draw_direct_{};
};
}
