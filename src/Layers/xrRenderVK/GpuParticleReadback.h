#pragma once
#include "FrameSubmissionGraph.h"
#include "GpuParticleBuffers.h"
#include "gpu_particles/gpu_particle_types.h"
#include <vector>

namespace xray::render::vulkan
{
// Per-frame-slot readback. Wait for the submission fence before read or reuse.
class GpuParticleReadback
{
public:
    struct Result { std::vector<fg::GpuPapiStatus> statuses; uint32_t errors{}; };
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t roots,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error);
    bool record(VkCommandBuffer command, const GpuParticleBuffers& buffers,
        VkQueueFlags queue_flags, std::string& error,
        FrameSubmissionGraph::Queue recording_queue = FrameSubmissionGraph::Queue::Graphics) const;
    bool read(Result& result, std::string& error) const;
    void destroy();
private:
    BufferResource buffer_;
    uint32_t roots_{};
    PFN_vkCmdCopyBuffer copy_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
