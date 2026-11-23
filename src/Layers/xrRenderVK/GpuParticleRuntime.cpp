#include "xrEngine/stdafx.h"
#include "GpuParticleRuntime.h"
#include "GpuParticleDrawBuckets.h"
#include <algorithm>
#include <utility>

namespace xray::render::vulkan
{
bool GpuParticleRuntime::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch, const GpuParticleBuffers::Capacity& capacity,
    uint32_t frames, VkDeviceSize staging_bytes, VkPipelineCache cache,
    const GpuParticleCompute::Source& source, std::string& error, FrameBufferPool* buffer_pool)
{
    destroy();
    if (!frames || frames > 5) { error = "invalid GPU particle frame count"; return false; }
    if (capacity.commands < capacity.roots)
    { error = "GPU particle command capacity must allow one command per root"; return false; }
    if (!compute_.initialize_from_source(device, get, limits, frames * 3, cache, source, error,
            dispatch.nvrhi_synchronization2_enabled ? dispatch.nvrhi_device : nullptr) ||
        !buffers_.initialize(device, limits, memory, dispatch, capacity, error, buffer_pool) ||
        !frame_.initialize(device, get, error))
    { destroy(); return false; }
    for (uint32_t i = 0; i < frames; ++i)
    {
        auto slot = std::make_unique<Slot>();
        if (!slot->uniforms.initialize(device, limits, memory, dispatch, 8, error) ||
            !slot->upload.initialize(device, get, staging_bytes, memory, dispatch, error) ||
            !slot->readback.initialize(device, get, capacity.roots, memory, dispatch, error))
        { destroy(); return false; }
        slots_.push_back(std::move(slot));
    }
    commands_ = std::make_unique<GpuParticleCommands>(capacity.roots);
    programs_ = GpuParticlePrograms({capacity.particles, capacity.action_states, capacity.programs, capacity.actions});
    capacity_ = capacity; error.clear(); return true;
}
bool GpuParticleRuntime::record(uint32_t index, VkCommandBuffer command, const std::vector<uint32_t>& visible,
    const GpuParticleCollisionInput* collision, const float* planes, std::string& error,
    VkQueueFlags queue_flags, bool semaphore_handoff, FrameSubmissionGraph* submission_graph,
    FrameSubmissionGraph::Pass compute_submission_pass,
    FrameSubmissionGraph::Pass graphics_prepass_pass,
    FrameSubmissionGraph::Pass graphics_continuation_pass,
    uint64_t previous_graphics_submission, bool copy_queue_available,
    uint32_t graphics_queue_family,
    uint32_t compute_queue_family, uint32_t copy_queue_family,
    FrameSubmissionGraph::Queue recording_queue, VkQueueFlags graphics_queue_flags,
    GpuParticleFrame::RequestSubmissionCommandBuffer request_submission_command_buffer,
    GpuParticleFrame::ResolveNvrhiCommandList resolve_nvrhi_command_list)
{
    if (!commands_ || index >= slots_.size() || slots_[index]->recorded || slots_[index]->inflight ||
        (!initialized_ && (initial_recorded_ || !collision)))
    { error = "GPU particle frame slot unavailable or initial collision data missing"; return false; }
    auto& slot = *slots_[index];
    slot.uniforms.reset();
    GpuParticleCommands::Batch batch;
    if (!commands_->snapshot(batch, error, capacity_.commands)) return false;
    GpuParticleFrameInputResult inputs{};
    const uint64_t revision = programs_.revision();
    if (!upload_gpu_particle_frame(command, buffers_, slot.upload, programs_, batch, visible, collision,
            !initialized_, !initialized_, queue_flags, inputs, error, VK_NULL_HANDLE, 0,
            recording_queue, FrameSubmissionGraph::Queue::Copy, true))
        return false;
    fg::GpuPapiConstants params{};
    params.particleCapacity = capacity_.particles; params.emitterCapacity = capacity_.emitters;
    params.rootCapacity = capacity_.roots;
    params.actionStateCapacity = capacity_.action_states; params.commandCount = inputs.command_count;
    params.programCount = inputs.program_count;
    const auto readback_record = [&](VkCommandBuffer readback_command, std::string& readback_error) {
        return slot.readback.record(readback_command, buffers_, queue_flags, readback_error, recording_queue);
    };
    if (!frame_.record(command, compute_, buffers_, slot.uniforms, index * 3, params, inputs.pending_roots,
            inputs.visible_roots, planes, !initialized_, error, queue_flags, semaphore_handoff,
            submission_graph, compute_submission_pass, graphics_continuation_pass,
            previous_graphics_submission, graphics_queue_family,
            compute_queue_family, copy_queue_family, recording_queue, graphics_queue_flags,
            std::move(request_submission_command_buffer), readback_record,
            &slot.upload, &inputs.uploads, copy_queue_available, std::move(resolve_nvrhi_command_list)))
        return false;
    slot.recorded = true; slot.initializes = !initialized_; slot.uploaded_programs = inputs.uploaded_programs;
    slot.visible_roots = visible; slot.draw_program_count = inputs.program_count;
    slot.program_revision = revision;
    if (slot.initializes) initial_recorded_ = true;
    error.clear(); return true;
}
bool GpuParticleRuntime::submitted(uint32_t index, std::string& error)
{
    if (index >= slots_.size() || !slots_[index]->recorded || slots_[index]->inflight)
    { error = "GPU particle frame was not recorded"; return false; }
    auto& slot = *slots_[index];
    if (slot.initializes) { initialized_ = true; initial_recorded_ = false; }
    if (slot.uploaded_programs) programs_.mark_uploaded(slot.program_revision);
    slot.recorded = false; slot.inflight = true; error.clear(); return true;
}
bool GpuParticleRuntime::cancel(uint32_t index, std::string& error)
{
    if (index >= slots_.size() || !slots_[index]->recorded || slots_[index]->inflight)
    { error = "GPU particle frame cannot be cancelled"; return false; }
    auto& slot = *slots_[index];
    if (slot.initializes) initial_recorded_ = false;
    slot.recorded = false; slot.visible_roots.clear(); slot.draw_program_count = 0; error.clear(); return true;
}
bool GpuParticleRuntime::complete(uint32_t index, GpuParticleReadback::Result& result, std::string& error)
{
    if (index >= slots_.size() || !slots_[index]->inflight)
    { error = "GPU particle frame is not in flight"; return false; }
    GpuParticleReadback::Result readback;
    if (!slots_[index]->readback.read(readback, error) || !commands_->acknowledge(readback.statuses, error))
        return false;
    slots_[index]->inflight = false; slots_[index]->visible_roots.clear(); slots_[index]->draw_program_count = 0;
    result = std::move(readback); error.clear(); return true;
}
bool GpuParticleRuntime::draw_bucket(uint32_t index, uint32_t root, uint32_t material, uint32_t blend,
    uint32_t variant, bool hud, uint32_t& bucket, bool child) const
{
    if (index >= slots_.size()) return false;
    const auto& slot = *slots_[index];
    if (!slot.recorded && !slot.inflight) return false;
    const auto found = std::find(slot.visible_roots.begin(), slot.visible_roots.end(), root);
    if (found == slot.visible_roots.end()) return false;
    return gpu_particle_draw_bucket(static_cast<uint32_t>(found - slot.visible_roots.begin()),
        static_cast<uint32_t>(slot.visible_roots.size()), material, slot.draw_program_count, blend, variant, hud, bucket, child);
}
bool GpuParticleRuntime::collision_resize_needed(VkDeviceSize bytes) const
{
    const VkDeviceSize staging = bytes + VkDeviceSize(capacity_.programs) * sizeof(fg::GpuPapiProgram) +
        VkDeviceSize(capacity_.actions) * sizeof(fg::GpuPapiAction) + VkDeviceSize(capacity_.commands) * sizeof(fg::GpuPapiCommand) +
        VkDeviceSize(capacity_.roots) * 12 + 514 * 16;
    if (bytes > buffers_.get(GpuParticleBuffers::Collision).size()) return true;
    for (const auto& slot : slots_) if (staging > slot->upload.capacity()) return true;
    return false;
}
bool GpuParticleRuntime::draw_resize_needed(uint32_t roots) const
{
    const uint64_t pairs = uint64_t(roots) * programs_.programs().size();
    return pairs > UINT32_MAX / 28 || pairs * 28 > buffers_.get(GpuParticleBuffers::DrawArguments).size() / 16;
}
bool GpuParticleRuntime::reserve_draw_buckets(VkDevice device, const VkPhysicalDeviceLimits& limits,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
    uint32_t visible_roots, std::string& error)
{
    const uint64_t pairs = uint64_t(visible_roots) * programs_.programs().size();
    if (!commands_ || !pairs || pairs > UINT32_MAX / 28 || pairs * 28 * 16 > limits.maxStorageBufferRange)
    { error = "GPU particle ordered draw buckets exceed device limits"; return false; }
    const uint64_t required = pairs * 28;
    for (const auto& slot : slots_) if (slot->recorded || slot->inflight)
    { error = "GPU particle draw bucket resize requires completed frames"; return false; }
    const uint64_t current = buffers_.get(GpuParticleBuffers::DrawArguments).size() / 16;
    const uint64_t limit = std::min<uint64_t>(UINT32_MAX, limits.maxStorageBufferRange / 16);
    const uint64_t grown = required <= current ? current : current <= limit / 2 ? std::max(required, current * 2) : required;
    return buffers_.resize_draw_buckets(device, static_cast<uint32_t>(grown), limits.maxStorageBufferRange, memory, dispatch, error);
}
bool GpuParticleRuntime::reserve_collision(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch, VkDeviceSize bytes, std::string& error)
{
    if (!commands_ || bytes < 48 || bytes % 4 || bytes > UINT32_MAX || bytes > limits.maxStorageBufferRange)
    { error = "GPU particle collision geometry exceeds device limits"; return false; }
    for (const auto& slot : slots_) if (slot->recorded || slot->inflight)
    { error = "GPU particle collision resize requires completed frames"; return false; }
    const VkDeviceSize current = buffers_.get(GpuParticleBuffers::Collision).size();
    const VkDeviceSize ceiling = std::min<VkDeviceSize>(limits.maxStorageBufferRange, UINT32_MAX & ~VkDeviceSize(3));
    const VkDeviceSize grown = bytes <= current ? current : current <= ceiling / 2 ? std::max(bytes, current * 2) : bytes;
    if (!buffers_.resize_collision(device, grown, limits.maxStorageBufferRange, memory, dispatch, error)) return false;
    const VkDeviceSize staging = bytes + VkDeviceSize(capacity_.programs) * sizeof(fg::GpuPapiProgram) +
        VkDeviceSize(capacity_.actions) * sizeof(fg::GpuPapiAction) + VkDeviceSize(capacity_.commands) * sizeof(fg::GpuPapiCommand) +
        VkDeviceSize(capacity_.roots) * 12 + 514 * 16;
    for (auto& slot : slots_)
        if (staging > slot->upload.capacity() && !slot->upload.initialize(device, get, staging, memory, dispatch, error)) return false;
    capacity_.collision_bytes = grown; error.clear(); return true;
}
void GpuParticleRuntime::destroy()
{
    slots_.clear(); commands_.reset(); programs_.clear(); buffers_.destroy(); compute_.destroy();
    capacity_ = {}; initialized_ = initial_recorded_ = false;
}
}
