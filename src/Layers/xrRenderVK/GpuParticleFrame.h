#pragma once
#include "FrameSubmissionGraph.h"
#include "GpuParticleBuffers.h"
#include "GpuParticleUpload.h"
#include "GpuParticleUniforms.h"
#include "gpu_particles/gpu_particle_types.h"
#include <functional>

namespace xray::render::vulkan
{
// Records initialization, simulation and the three compaction phases.
// Upload commands must precede this on the same queue. Each in-flight frame
// needs a separate uniform arena and three descriptor slots beginning at base.
class GpuParticleFrame
{
public:
    using RequestSubmissionCommandBuffer = std::function<bool(FrameSubmissionGraph::Pass,
        FrameSubmissionGraph::Queue, VkCommandBuffer&, std::string&)>;
    using ReadbackRecord = std::function<bool(VkCommandBuffer, std::string&)>;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, std::string& error);
    bool record(VkCommandBuffer command, GpuParticleCompute& compute, const GpuParticleBuffers& buffers,
        GpuParticleUniforms& uniforms, uint32_t base, const fg::GpuPapiConstants& capacities,
        uint32_t pending_roots, uint32_t visible_roots, const float* planes,
        bool initialize_buffers, std::string& error,
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
        RequestSubmissionCommandBuffer request_submission_command_buffer = {},
        ReadbackRecord readback_record = {},
        GpuParticleUpload* upload = nullptr,
        const std::vector<GpuParticleUpload::Update>* upload_updates = nullptr,
        bool use_copy_queue = false) const;
private:
    PFN_vkCmdFillBuffer fill_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
