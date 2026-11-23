#include "xrEngine/stdafx.h"
#include "SmokeTrailCompute.h"
#include "FrameImageGraph.h"
#include "ShaderModule.h"
#include <cmath>
#include <cstring>

namespace xray::render::vulkan
{
bool SmokeTrailCompute::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch, uint32_t frames, VkPipelineCache cache,
    const GpuParticleCompute::Source& source, std::string& error, FrameBufferPool* buffer_pool)
{
    destroy();
    if (!device || !get || !source || !frames || frames > 5 || limits.maxComputeWorkGroupInvocations < 64 ||
        limits.maxComputeWorkGroupSize[0] < 64 || limits.maxComputeWorkGroupCount[0] < 4 ||
        limits.maxPerStageDescriptorStorageBuffers < 4 || limits.maxDescriptorSetStorageBuffers < 4 ||
        limits.maxPerStageResources < 5 || limits.maxComputeSharedMemorySize < 8192)
    { error = "device limits do not support smoke trail compute"; return false; }
    fill_ = reinterpret_cast<PFN_vkCmdFillBuffer>(get(device, "vkCmdFillBuffer"));
    barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    if (!fill_ || !barrier_) { error = "smoke trail command procedures unavailable"; destroy(); return false; }
    buffer_pool_ = buffer_pool;
    const VkDeviceSize sizes[]{MaxPoints * 48, MaxPoints * 32, 16, 16};
    for (uint32_t i = 0; i < ResourceCount; ++i)
    {
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (i == DrawArguments) usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (sizes[i] > limits.maxStorageBufferRange)
        { error = "smoke trail buffer exceeds device limit"; destroy(); return false; }
        if (buffer_pool_)
        {
            if (!buffer_pool_->acquire(sizes[i], usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    !dispatch.nvrhi_device, buffers_[i], error,
                    VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            { destroy(); return false; }
        }
        else
        {
            owned_buffers_[i] = std::make_unique<BufferResource>();
            if (!owned_buffers_[i]->initialize(device, sizes[i], usage,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, dispatch, error, !dispatch.nvrhi_device,
                    VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            { destroy(); return false; }
            buffers_[i] = owned_buffers_[i].get();
        }
        if (!buffers_[i])
        { if (error.empty()) error = "smoke trail buffer exceeds device limit"; destroy(); return false; }
    }
    const char* names[]{"vk/smoke_trail_emit.cs.spv", "vk/smoke_trail_simulate.cs.spv", "vk/smoke_trail_compact.cs.spv"};
    const uint32_t binding_counts[]{3, 2, 5};
    for (uint32_t i = 0; i < 3; ++i)
    {
        std::vector<uint8_t> bytes;
        if (!source(names[i], bytes, error) || !has_spirv_entry(bytes.data(), bytes.size(), 5, "main"))
        { if (error.empty()) error = "invalid smoke trail shader"; destroy(); return false; }
        std::vector<uint32_t> words(bytes.size() / 4); std::memcpy(words.data(), bytes.data(), bytes.size());
        std::vector<StorageComputePass::Binding> bindings;
        for (uint32_t n = 0; n < binding_counts[i]; ++n)
            bindings.push_back({n, n ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER});
        if (!passes_[i].initialize(device, get, frames, bindings, 0, limits.maxStorageBufferRange,
                limits.maxUniformBufferRange, cache, words.data(), bytes.size(), error))
        { destroy(); return false; }
    }
    for (uint32_t i = 0; i < frames; ++i)
    {
        auto slot = std::make_unique<GpuParticleUniforms>();
        if (!slot->initialize(device, limits, memory, dispatch, 3, error)) { destroy(); return false; }
        uniforms_.push_back(std::move(slot));
    }
    error.clear(); return true;
}
bool SmokeTrailCompute::record(VkCommandBuffer command, uint32_t slot, const SmokeTrailEmitParams& emit,
    const SmokeTrailSimParams& simulation, const SmokeTrailCompactParams& compact, bool clear,
    std::string& error, VkQueueFlags queue_flags, bool semaphore_handoff,
    FrameSubmissionGraph* submission_graph,
    FrameSubmissionGraph::Pass compute_submission_pass,
    FrameSubmissionGraph::Pass graphics_continuation_pass,
    uint64_t previous_graphics_submission, uint32_t graphics_queue_family,
    uint32_t compute_queue_family, uint32_t copy_queue_family,
    FrameSubmissionGraph::Queue recording_queue, VkQueueFlags graphics_queue_flags,
    RequestSubmissionCommandBuffer request_submission_command_buffer)
{
    if (!command || slot >= uniforms_.size() || !fill_ || !barrier_ || emit.max_points != MaxPoints ||
        simulation.max_points != MaxPoints || compact.max_points != MaxPoints || emit.count > MaxPoints ||
        !std::isfinite(simulation.dt) || simulation.dt < 0.f || !std::isfinite(emit.lifetime) || emit.lifetime < 0.f)
    { error = "invalid smoke trail frame"; return false; }
    auto& uniforms = *uniforms_[slot]; uniforms.reset();
    VkDescriptorBufferInfo e{}, s{}, c{};
    if (!uniforms.append(&emit, sizeof(emit), e, error) || !uniforms.append(&simulation, sizeof(simulation), s, error) ||
        !uniforms.append(&compact, sizeof(compact), c, error)) return false;
    const auto info = [&](Resource r) { return VkDescriptorBufferInfo{buffers_[r]->handle(), 0, buffers_[r]->size()}; };

    FrameImageGraph resource_graph(recording_queue);
    std::array<FrameImageGraph::BufferResource, ResourceCount> current{};
    for (uint32_t i = 0; i < ResourceCount; ++i)
        if (!resource_graph.import_buffer(buffers_[i]->handle(), 0, buffers_[i]->size(),
                BufferUse::AllCommandsReadWrite, current[i], error,
                FrameSubmissionGraph::Queue::Graphics, previous_graphics_submission,
                buffers_[i]->sharing_mode(), buffers_[i]->sharing_queue_families()))
            return false;
    const auto write = [&](FrameImageGraph::Pass pass, Resource resource, BufferUse use) {
        FrameImageGraph::BufferResource next;
        if (!resource_graph.write_buffer(pass, current[resource], use, next, error)) return false;
        current[resource] = next;
        return true;
    };
    const auto read = [&](FrameImageGraph::Pass pass, Resource resource, BufferUse use) {
        return resource_graph.read_buffer(pass, current[resource], use, error);
    };
    const auto read_write = [&](FrameImageGraph::Pass pass, Resource resource) {
        FrameImageGraph::BufferResource next;
        if (!resource_graph.read_write_buffer(pass, current[resource], BufferUse::ComputeReadWrite,
                next, error)) return false;
        current[resource] = next;
        return true;
    };
    if (clear)
    {
        const auto clear_pass = resource_graph.add_command_pass("clear smoke trail buffers",
            recording_queue, [&](VkCommandBuffer recording_command) {
                for (const auto& buffer : buffers_)
                    fill_(recording_command, buffer->handle(), 0, buffer->size(), 0);
            });
        for (uint32_t i = 0; i < ResourceCount; ++i)
            if (!write(clear_pass, static_cast<Resource>(i), BufferUse::TransferDestination)) return false;
    }
    if (emit.count)
    {
        const auto emit_pass = resource_graph.add_checked_command_pass("emit smoke trail points",
            recording_queue, [&](VkCommandBuffer recording_command, std::string& pass_error) {
                return passes_[0].record(recording_command, slot,
                    {e, info(Simulation), info(State)}, nullptr, 1, 1, pass_error);
            });
        if (!read_write(emit_pass, Simulation) || !read_write(emit_pass, State)) return false;
    }
    const auto simulate_pass = resource_graph.add_checked_command_pass("simulate smoke trail points",
        recording_queue, [&](VkCommandBuffer recording_command, std::string& pass_error) {
            return passes_[1].record(recording_command, slot,
                {s, info(Simulation)}, nullptr, 4, 1, pass_error);
        });
    if (!read_write(simulate_pass, Simulation)) return false;
    const auto compact_pass = resource_graph.add_checked_command_pass("compact smoke trail draw data",
        recording_queue, [&](VkCommandBuffer recording_command, std::string& pass_error) {
            return passes_[2].record(recording_command, slot,
                {c, info(ControlPoints), info(State), info(DrawArguments), info(Simulation)},
                nullptr, 1, 1, pass_error);
        });
    // HLSL exposes these outputs as readable RWStructuredBuffer bindings.
    if (!read(compact_pass, Simulation, BufferUse::ComputeRead) || !read_write(compact_pass, State) ||
        !read_write(compact_pass, ControlPoints) ||
        !read_write(compact_pass, DrawArguments)) return false;
    // Keep the consumer in the same graph queue when compute and draw share a
    // command buffer so execution emits the in-list compute-to-vertex barrier.
    // A split frame instead records the consumer on its graphics continuation.
    const bool has_graphics_continuation = submission_graph &&
        compute_submission_pass != FrameSubmissionGraph::InvalidPass &&
        graphics_continuation_pass != FrameSubmissionGraph::InvalidPass;
    const FrameSubmissionGraph::Queue draw_queue = has_graphics_continuation ?
        FrameSubmissionGraph::Queue::Graphics : recording_queue;
    const auto draw_consumer = resource_graph.add_pass("draw compacted smoke trail", draw_queue, {});
    if (!read(draw_consumer, ControlPoints, BufferUse::VertexRead) ||
        !read(draw_consumer, State, BufferUse::VertexRead) ||
        !read(draw_consumer, DrawArguments, BufferUse::IndirectRead)) return false;
    ImageStateTracker image_states;
    const ImageStateDispatch state_dispatch{barrier_};
    if (submission_graph && compute_submission_pass != FrameSubmissionGraph::InvalidPass &&
        graphics_continuation_pass != FrameSubmissionGraph::InvalidPass &&
        request_submission_command_buffer)
    {
        std::vector<FrameImageGraph::QueueGroup> groups;
        std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
        if (!resource_graph.generate_submission_groups(*submission_graph,
                [graphics_queue_family, compute_queue_family, copy_queue_family](FrameSubmissionGraph::Queue queue) {
                    if (queue == FrameSubmissionGraph::Queue::Compute) return compute_queue_family;
                    if (queue == FrameSubmissionGraph::Queue::Graphics) return graphics_queue_family;
                    return copy_queue_family;
                }, groups, submission_for_pass, error))
            return false;

        FrameImageGraph::ExecutionState execution;
        if (!resource_graph.begin_execution(execution, error))
            return false;
        for (const FrameImageGraph::QueueGroup& group : groups)
        {
            const FrameSubmissionGraph::Pass submission = group.submission;
            VkCommandBuffer group_command = VK_NULL_HANDLE;
            if (submission == FrameSubmissionGraph::InvalidPass ||
                !request_submission_command_buffer(submission, group.queue, group_command, error))
                return false;
            const VkQueueFlags group_queue_flags = group.queue == FrameSubmissionGraph::Queue::Compute ?
                queue_flags : group.queue == FrameSubmissionGraph::Queue::Copy ?
                    VK_QUEUE_TRANSFER_BIT : graphics_queue_flags;
            if (!resource_graph.execute_group(group, group_command, image_states, execution,
                    state_dispatch, error, group_queue_flags, semaphore_handoff))
                return false;
        }
    }
    else
    {
        if (submission_graph && compute_submission_pass != FrameSubmissionGraph::InvalidPass &&
            graphics_continuation_pass != FrameSubmissionGraph::InvalidPass &&
            !resource_graph.connect_submission_dependencies(*submission_graph,
                [compute_submission_pass, graphics_continuation_pass](FrameSubmissionGraph::Queue queue) {
                    if (queue == FrameSubmissionGraph::Queue::Compute) return compute_submission_pass;
                    if (queue == FrameSubmissionGraph::Queue::Graphics) return graphics_continuation_pass;
                    return FrameSubmissionGraph::InvalidPass;
                },
                [graphics_queue_family, compute_queue_family, copy_queue_family](FrameSubmissionGraph::Queue queue) {
                    if (queue == FrameSubmissionGraph::Queue::Compute) return compute_queue_family;
                    if (queue == FrameSubmissionGraph::Queue::Graphics) return graphics_queue_family;
                    return copy_queue_family;
                }, error))
            return false;
        if (!resource_graph.execute(command, image_states, state_dispatch, error,
                queue_flags, semaphore_handoff, recording_queue))
            return false;
    }
    error.clear(); return true;
}
void SmokeTrailCompute::destroy()
{
    uniforms_.clear();
    for (auto& pass : passes_) pass.destroy();
    for (size_t i = 0; i < ResourceCount; ++i)
    {
        if (buffer_pool_ && buffers_[i] && !owned_buffers_[i])
            buffer_pool_->release(buffers_[i]);
        owned_buffers_[i].reset();
        buffers_[i] = nullptr;
    }
    buffer_pool_ = nullptr;
    fill_ = nullptr; barrier_ = nullptr;
}
}
