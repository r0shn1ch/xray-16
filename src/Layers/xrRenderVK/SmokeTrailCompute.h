#pragma once
#include "FrameBufferPool.h"
#include "FrameSubmissionGraph.h"
#include "GpuParticleCompute.h"
#include "GpuParticleUniforms.h"
#include <array>
#include <functional>
#include <memory>

namespace xray::render::vulkan
{
struct SmokeTrailEmitParams
{
    float previous_x{}, previous_y{}, previous_z{}, current_x{};
    float current_y{}, current_z{}, direction_x{}, direction_y{};
    float direction_z{}, lifetime{}, lifetime_variance{}, max_width{};
    uint32_t count{}, max_points{}; float seed{}, padding{};
};
struct SmokeTrailSimParams
{
    float dt{}, gravity{}, buoyancy{}, turbulence{}, drag{}, time{};
    uint32_t max_points{}; float heat{};
};
struct SmokeTrailCompactParams
{
    uint32_t max_points{}, subdivisions{}; float max_width{}, turbulence{};
    float frequency{}, evolution{}, sphere_radius{}, padding{};
    float center_x{}, center_y{}, center_z{}, padding2{};
};
static_assert(sizeof(SmokeTrailEmitParams) == 64 && sizeof(SmokeTrailSimParams) == 32 &&
    sizeof(SmokeTrailCompactParams) == 48);

// One emitter's persistent ring. The caller owns fences and supplies a frame
// slot only after its previous submission completes. Recording can target a
// single queue or generated frame submissions; discard all recorded work if
// record fails.
class SmokeTrailCompute
{
public:
    static constexpr uint32_t MaxPoints = 256;
    using RequestSubmissionCommandBuffer = std::function<bool(FrameSubmissionGraph::Pass,
        FrameSubmissionGraph::Queue, VkCommandBuffer&, std::string&)>;
    enum Resource { Simulation, ControlPoints, State, DrawArguments, ResourceCount };
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        uint32_t frames, VkPipelineCache cache, const GpuParticleCompute::Source& source, std::string& error,
        FrameBufferPool* buffer_pool = nullptr);
    bool record(VkCommandBuffer command, uint32_t slot, const SmokeTrailEmitParams& emit,
        const SmokeTrailSimParams& simulation, const SmokeTrailCompactParams& compact,
        bool clear_buffers, std::string& error,
        VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
        bool semaphore_handoff = false,
        FrameSubmissionGraph* submission_graph = nullptr,
        FrameSubmissionGraph::Pass compute_submission_pass = FrameSubmissionGraph::InvalidPass,
        FrameSubmissionGraph::Pass graphics_continuation_pass = FrameSubmissionGraph::InvalidPass,
        uint64_t previous_graphics_submission = 0,
        uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED,
        uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED,
        uint32_t copy_queue_family = VK_QUEUE_FAMILY_IGNORED,
        FrameSubmissionGraph::Queue recording_queue = FrameSubmissionGraph::Queue::Compute,
        VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT,
        RequestSubmissionCommandBuffer request_submission_command_buffer = {});
    const BufferResource& get(Resource resource) const
    { return buffers_[resource] ? *buffers_[resource] : empty_buffers_[resource]; }
    void destroy(); // Requires GPU idle.
private:
    FrameBufferPool* buffer_pool_{};
    std::array<BufferResource*, ResourceCount> buffers_{};
    std::array<std::unique_ptr<BufferResource>, ResourceCount> owned_buffers_{};
    std::array<BufferResource, ResourceCount> empty_buffers_{};
    std::array<StorageComputePass, 3> passes_;
    std::vector<std::unique_ptr<GpuParticleUniforms>> uniforms_;
    PFN_vkCmdFillBuffer fill_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
