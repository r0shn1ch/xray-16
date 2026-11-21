#pragma once
#include "GpuParticleBuffers.h"
#include "GpuParticleCollisionData.h"
#include "GpuParticleCommands.h"
#include "GpuParticlePrograms.h"
#include "GpuParticleUpload.h"

namespace xray::render::vulkan
{
struct GpuParticleFrameInputResult
{
    uint32_t pending_roots{}, visible_roots{}, command_count{}, program_count{};
    bool uploaded_programs{};
    bool uploads_recorded{};
    std::vector<GpuParticleUpload::Update> uploads;
};
// Validates and stages the frame's upload batch. In deferred mode the caller
// places these copies in its frame graph; otherwise this records the legacy
// inline graph immediately. Programs remain dirty until submission succeeds.
bool upload_gpu_particle_frame(VkCommandBuffer command, const GpuParticleBuffers& buffers,
    GpuParticleUpload& upload, const GpuParticlePrograms& programs, const GpuParticleCommands::Batch& batch,
    const std::vector<uint32_t>& visible_roots, const GpuParticleCollisionInput* collision,
    bool upload_noise, bool force_programs, VkQueueFlags queue_flags,
    GpuParticleFrameInputResult& result, std::string& error,
    VkCommandBuffer upload_command = VK_NULL_HANDLE,
    VkQueueFlags upload_queue_flags = 0,
    FrameSubmissionGraph::Queue command_queue = FrameSubmissionGraph::Queue::Graphics,
    FrameSubmissionGraph::Queue copy_queue = FrameSubmissionGraph::Queue::Copy,
    bool defer_record = false);
}
