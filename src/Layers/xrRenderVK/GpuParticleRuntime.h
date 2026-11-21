#pragma once
#include "GpuParticleFrame.h"
#include "GpuParticleFrameInput.h"
#include "GpuParticleReadback.h"
#include <memory>

namespace xray::render::vulkan
{
// GPU resources and submission bookkeeping for the upstream PAPI pipeline.
// FrameImageGraph schedules uploads before simulation and tracks queue waits.
// Its frame fence must cover all generated submissions; failed recordings are
// discarded and completion is required before reuse.
class GpuParticleRuntime
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        const GpuParticleBuffers::Capacity& capacity, uint32_t frames, VkDeviceSize staging_bytes,
        VkPipelineCache cache, const GpuParticleCompute::Source& source, std::string& error,
        FrameBufferPool* buffer_pool = nullptr);
    bool record(uint32_t slot, VkCommandBuffer command, const std::vector<uint32_t>& visible_roots,
        const GpuParticleCollisionInput* collision, const float* planes, std::string& error,
        VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
        bool semaphore_handoff = false,
        FrameSubmissionGraph* submission_graph = nullptr,
        FrameSubmissionGraph::Pass compute_submission_pass = FrameSubmissionGraph::InvalidPass,
        FrameSubmissionGraph::Pass graphics_prepass_pass = FrameSubmissionGraph::InvalidPass,
        FrameSubmissionGraph::Pass graphics_continuation_pass = FrameSubmissionGraph::InvalidPass,
        uint64_t previous_graphics_submission = 0,
        bool copy_queue_available = false,
        uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED,
        uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED,
        uint32_t copy_queue_family = VK_QUEUE_FAMILY_IGNORED,
        FrameSubmissionGraph::Queue recording_queue = FrameSubmissionGraph::Queue::Graphics,
        VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT,
        GpuParticleFrame::RequestSubmissionCommandBuffer request_submission_command_buffer = {});
    // Call submitted only after vkQueueSubmit succeeds; otherwise cancel.
    bool submitted(uint32_t slot, std::string& error);
    bool cancel(uint32_t slot, std::string& error);
    bool complete(uint32_t slot, GpuParticleReadback::Result& result, std::string& error);
    // Resolve the GPU compact bucket against the exact root/program table
    // recorded for this frame, even if the program table subsequently grows.
    bool draw_bucket(uint32_t slot, uint32_t root, uint32_t material, uint32_t blend,
        uint32_t variant, bool hud, uint32_t& bucket, bool child = false) const;
    bool collision_resize_needed(VkDeviceSize bytes) const;
    // Requires device idle and completion of all recorded slots.
    bool reserve_collision(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        VkDeviceSize bytes, std::string& error);
    bool draw_resize_needed(uint32_t visible_roots) const;
    bool reserve_draw_buckets(VkDevice device, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        uint32_t visible_roots, std::string& error);
    void destroy();
    GpuParticlePrograms& programs() { return programs_; }
    GpuParticleCommands& commands() { return *commands_; }
    const GpuParticleBuffers& buffers() const { return buffers_; }
private:
    struct Slot
    {
        GpuParticleUniforms uniforms;
        GpuParticleUpload upload;
        GpuParticleReadback readback;
        bool recorded{}, inflight{}, initializes{}, uploaded_programs{};
        uint64_t program_revision{};
        uint32_t draw_program_count{};
        std::vector<uint32_t> visible_roots;
    };
    GpuParticleCompute compute_;
    GpuParticleBuffers buffers_;
    GpuParticleFrame frame_;
    GpuParticlePrograms programs_;
    std::unique_ptr<GpuParticleCommands> commands_;
    std::vector<std::unique_ptr<Slot>> slots_;
    GpuParticleBuffers::Capacity capacity_{};
    bool initialized_{}, initial_recorded_{};
};
}
