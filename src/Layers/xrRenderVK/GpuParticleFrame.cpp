#include "xrEngine/stdafx.h"
#include "GpuParticleFrame.h"
#include "FrameImageGraph.h"
#include <algorithm>
#include <array>
#include <utility>

namespace xray::render::vulkan
{
bool GpuParticleFrame::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, std::string& error)
{
    fill_ = device && get ? reinterpret_cast<PFN_vkCmdFillBuffer>(get(device, "vkCmdFillBuffer")) : nullptr;
    barrier_ = device && get ? reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier")) : nullptr;
    if (!fill_ || !barrier_) { error = "GPU particle frame procedures unavailable"; return false; }
    error.clear(); return true;
}
bool GpuParticleFrame::record(VkCommandBuffer command, GpuParticleCompute& compute,
    const GpuParticleBuffers& buffers, GpuParticleUniforms& uniforms, uint32_t base,
    const fg::GpuPapiConstants& capacities, uint32_t pending_roots, uint32_t visible_roots,
    const float* planes, bool initialize_buffers, std::string& error,
    VkQueueFlags queue_flags, bool semaphore_handoff, FrameSubmissionGraph* submission_graph,
    FrameSubmissionGraph::Pass compute_submission_pass,
    FrameSubmissionGraph::Pass graphics_continuation_pass,
    uint64_t previous_graphics_submission, uint32_t graphics_queue_family,
    uint32_t compute_queue_family, uint32_t copy_queue_family,
    FrameSubmissionGraph::Queue recording_queue, VkQueueFlags graphics_queue_flags,
    RequestSubmissionCommandBuffer request_submission_command_buffer,
    ReadbackRecord readback_record, GpuParticleUpload* upload,
    const std::vector<GpuParticleUpload::Update>* upload_updates,
    bool use_copy_queue) const
{
    using B = GpuParticleBuffers;
    using P = GpuParticleCompute::Pass;
    const uint64_t pairs = uint64_t(visible_roots) * capacities.programCount;
    const uint64_t bucket_count = pairs > UINT32_MAX / 28 ? uint64_t(UINT32_MAX) + 1 : pairs * 28;
    const auto fits = [&](B::Resource r, uint64_t bytes) { return buffers.get(r).handle() && bytes <= buffers.get(r).size(); };
    if (!command || !fill_ || !barrier_ || !planes || base > UINT32_MAX - 2 ||
        !capacities.particleCapacity || !capacities.emitterCapacity || !capacities.programCount ||
        bucket_count > UINT32_MAX ||
        pending_roots > 65535 || visible_roots > 65535 ||
        !capacities.rootCapacity || capacities.rootCapacity != buffers.root_slots() ||
        capacities.rootCapacity > capacities.emitterCapacity / 2 ||
        pending_roots > capacities.rootCapacity || visible_roots > capacities.rootCapacity ||
        !fits(B::Particles, uint64_t(capacities.particleCapacity) * sizeof(fg::GpuPapiParticle)) ||
        !fits(B::Emitters, uint64_t(capacities.emitterCapacity) * sizeof(fg::GpuPapiEmitter)) ||
        !fits(B::ActionState, uint64_t(capacities.actionStateCapacity) * 16) ||
        !fits(B::ParticleAllocation, uint64_t(capacities.particleCapacity) * 4) ||
        !fits(B::StateAllocation, uint64_t(capacities.actionStateCapacity) * 4) ||
        !fits(B::Programs, uint64_t(capacities.programCount) * sizeof(fg::GpuPapiProgram)) ||
        !fits(B::Commands, uint64_t(capacities.commandCount) * sizeof(fg::GpuPapiCommand)) ||
        !fits(B::Status, uint64_t(capacities.rootCapacity) * sizeof(fg::GpuPapiStatus)) ||
        !fits(B::RootCommands, uint64_t(pending_roots) * 8) || !fits(B::VisibleRoots, uint64_t(visible_roots) * 4) ||
        !fits(B::DrawArguments, bucket_count * 16) ||
        !fits(B::BucketOffsets, bucket_count * 4) ||
        !fits(B::BucketCursors, bucket_count * 4))
    { error = "invalid GPU particle frame capacities"; return false; }
    const uint64_t largest = std::max({uint64_t(capacities.particleCapacity), uint64_t(capacities.emitterCapacity),
        uint64_t(capacities.actionStateCapacity)});
    const uint64_t groups = (largest + 63) / 64;
    if (initialize_buffers && groups > 65535)
    { error = "GPU particle initialization exceeds dispatch limit"; return false; }
    auto dispatch = [&](VkCommandBuffer recording_command, P pass, uint32_t slot,
                        fg::GpuPapiConstants params,
                        VkDescriptorBufferInfo visibility, uint32_t count, std::string& dispatch_error) {
        VkDescriptorBufferInfo constants{};
        std::vector<VkDescriptorBufferInfo> descriptors;
        if (!uniforms.append(&params, sizeof(params), constants, dispatch_error) ||
            !buffers.descriptors(pass, constants, visibility, descriptors, dispatch_error))
            return false;
        if (compute.record(pass, recording_command, slot, descriptors, count, dispatch_error))
            return true;
        const char* pass_name = pass == P::Initialize ? "initialize" :
            pass == P::Simulate ? "simulate" : pass == P::Compact ? "compact" : "motion capture";
        dispatch_error = std::string("GPU particle ") + pass_name + " dispatch: " + dispatch_error;
        return false;
    };
    fg::GpuPapiConstants params = capacities;
    FrameImageGraph graph(recording_queue);
    std::array<FrameImageGraph::BufferResource, B::Count> resources{};
    std::array<FrameImageGraph::BufferResource, B::Count> current{};
    FrameImageGraph::Pass draw_data_consumer = ~0u;
    for (size_t i = 0; i < B::Count; ++i)
    {
        const auto resource = static_cast<B::Resource>(i);
        const auto& buffer = buffers.get(resource);
        if (!buffer.handle() || !buffer.size() ||
            !graph.import_buffer(buffer.handle(), 0, buffer.size(), BufferUse::AllCommandsReadWrite,
                resources[i], error, FrameSubmissionGraph::Queue::Graphics,
                previous_graphics_submission, buffer.sharing_mode(),
                buffer.sharing_queue_families(), FrameSubmissionGraph::InvalidPass, buffer.nvrhi_buffer()))
        {
            if (error.empty()) error = "GPU particle frame buffer unavailable";
            return false;
        }
        current[i] = resources[i];
    }

    const auto read = [&](FrameImageGraph::Pass pass, B::Resource resource, BufferUse use) {
        return graph.read_buffer(pass, current[resource], use, error);
    };
    const auto write = [&](FrameImageGraph::Pass pass, B::Resource resource, BufferUse use) {
        FrameImageGraph::BufferResource next;
        if (!graph.write_buffer(pass, current[resource], use, next, error)) return false;
        current[resource] = next;
        return true;
    };
    const auto read_write = [&](FrameImageGraph::Pass pass, B::Resource resource) {
        FrameImageGraph::BufferResource next;
        if (!graph.read_write_buffer(pass, current[resource], BufferUse::ComputeReadWrite, next, error)) return false;
        current[resource] = next;
        return true;
    };

    if (upload_updates && !upload_updates->empty())
    {
        if (!upload || !upload->staging_buffer())
        { error = "GPU particle frame graph has no prepared upload staging buffer"; return false; }
        VkDeviceSize staging_bytes = 0;
        for (const auto& update : *upload_updates)
        {
            if (!update.destination || !update.size || staging_bytes > upload->capacity() ||
                update.size > upload->capacity() - staging_bytes)
            { error = "GPU particle frame graph upload exceeds staging capacity"; return false; }
            staging_bytes += update.size;
        }
        const FrameSubmissionGraph::Queue upload_queue = use_copy_queue ?
            FrameSubmissionGraph::Queue::Copy : recording_queue;
        FrameImageGraph::BufferResource staging_resource;
        if (!graph.import_buffer(upload->staging_buffer(), 0, staging_bytes, BufferUse::HostWrite,
                staging_resource, error, upload_queue))
            return false;
        const auto uploads = *upload_updates;
        const auto upload_pass = graph.add_checked_command_pass("upload GPU particle inputs",
            upload_queue, [upload, uploads](VkCommandBuffer recording_command,
                std::string& pass_error) {
                return upload->record_copy(recording_command, uploads, pass_error);
            });
        if (!graph.read_buffer(upload_pass, staging_resource, BufferUse::TransferSource, error))
            return false;
        for (const auto& update : *upload_updates)
        {
            B::Resource destination = B::Count;
            for (size_t i = 0; i < B::Count; ++i)
                if (&buffers.get(static_cast<B::Resource>(i)) == update.destination)
                { destination = static_cast<B::Resource>(i); break; }
            if (destination == B::Count)
            { error = "GPU particle frame upload targets an unregistered buffer"; return false; }
            FrameImageGraph::BufferResource next;
            if (!graph.write_buffer(upload_pass, current[destination],
                    BufferUse::TransferDestination, next, error))
                return false;
            current[destination] = next;
        }
    }

    if (initialize_buffers)
    {
        if (!buffers.get(B::Status).handle()) { error = "GPU particle status buffer unavailable"; return false; }
        const auto clear_status = graph.add_command_pass("clear GPU particle status", [&](VkCommandBuffer recording_command) {
            fill_(recording_command, buffers.get(B::Status).handle(), 0, buffers.get(B::Status).size(), 0);
        });
        if (!write(clear_status, B::Status, BufferUse::TransferDestination)) return false;

        params.phase = 0;
        const auto initialize = graph.add_checked_command_pass("initialize GPU particle pools",
            [&, dispatch_params = params](VkCommandBuffer recording_command, std::string& pass_error) {
            return dispatch(recording_command, P::Initialize, base, dispatch_params, {},
                static_cast<uint32_t>(groups), pass_error);
        });
        if (!write(initialize, B::Emitters, BufferUse::ComputeWrite) ||
            !write(initialize, B::ParticleAllocation, BufferUse::ComputeWrite) ||
            !write(initialize, B::StateAllocation, BufferUse::ComputeWrite) ||
            !write(initialize, B::Errors, BufferUse::ComputeWrite)) return false;
    }
    else if (pending_roots || readback_record)
    {
        const auto clear_errors = graph.add_command_pass("clear GPU particle frame errors",
            [&](VkCommandBuffer recording_command) {
                fill_(recording_command, buffers.get(B::Errors).handle(), 0, sizeof(uint32_t), 0);
            });
        if (!write(clear_errors, B::Errors, BufferUse::TransferDestination)) return false;
    }

    if (pending_roots || visible_roots)
    {
        params.phase = 0;
        const uint32_t emitter_groups = std::min(capacities.emitterCapacity, 65535u);
        const auto capture_motion = graph.add_checked_command_pass("capture GPU particle motion history",
            [&, dispatch_params = params, emitter_groups](VkCommandBuffer recording_command, std::string& pass_error) {
                return dispatch(recording_command, P::CaptureMotion, base, dispatch_params, {},
                    emitter_groups, pass_error);
            });
        if (!read(capture_motion, B::Emitters, BufferUse::ComputeRead) ||
            !read_write(capture_motion, B::Particles)) return false;
    }

    if (pending_roots)
    {
        // Simulation uses phase for the total root pool size, while rootCapacity
        // is the number of command groups dispatched for this frame.
        params.phase = capacities.rootCapacity;
        params.rootCapacity = pending_roots;
        const auto simulate = graph.add_checked_command_pass("simulate GPU particles",
            [&, dispatch_params = params](VkCommandBuffer recording_command, std::string& pass_error) {
            return dispatch(recording_command, P::Simulate, base, dispatch_params, {}, pending_roots, pass_error);
        });
        for (B::Resource resource : {B::Programs, B::Actions, B::Commands, B::RootCommands,
                B::Noise, B::Collision})
            if (!read(simulate, resource, BufferUse::ComputeRead)) return false;
        for (B::Resource resource : {B::Emitters, B::Particles, B::ActionState,
                B::ParticleAllocation, B::StateAllocation, B::Errors})
            if (!read_write(simulate, resource)) return false;
        // The compiled RWStructuredBuffer status descriptor permits reads as
        // well as writes. Include both accesses after the initial clear.
        if (!read_write(simulate, B::Status)) return false;
    }

    if (visible_roots)
    {
        const auto clear_arguments = graph.add_command_pass("clear GPU particle draw arguments",
            [&](VkCommandBuffer recording_command) {
            fill_(recording_command, buffers.get(B::DrawArguments).handle(), 0, bucket_count * 16, 0);
        });
        if (!write(clear_arguments, B::DrawArguments, BufferUse::TransferDestination)) return false;

        VkDescriptorBufferInfo visibility{};
        if (!uniforms.append(planes, 6 * 16, visibility, error)) return false;
        params.rootCapacity = visible_roots; params.pad = 0;
        params.phase = 0;
        const auto count_visible = graph.add_checked_command_pass("count visible GPU particles",
            [&, dispatch_params = params, visibility](VkCommandBuffer recording_command, std::string& pass_error) {
            return dispatch(recording_command, P::Compact, base, dispatch_params, visibility,
                visible_roots, pass_error);
        });
        for (B::Resource resource : {B::VisibleRoots, B::Particles, B::Emitters, B::Programs})
            if (!read(count_visible, resource, BufferUse::ComputeRead)) return false;
        if (!read_write(count_visible, B::DrawArguments)) return false;

        params.phase = 1;
        const auto allocate_buckets = graph.add_checked_command_pass("allocate GPU particle draw buckets",
            [&, dispatch_params = params, visibility](VkCommandBuffer recording_command, std::string& pass_error) {
            return dispatch(recording_command, P::Compact, base + 1, dispatch_params, visibility, 1, pass_error);
        });
        if (!read_write(allocate_buckets, B::DrawArguments) ||
            !write(allocate_buckets, B::BucketOffsets, BufferUse::ComputeWrite) ||
            !write(allocate_buckets, B::BucketCursors, BufferUse::ComputeWrite)) return false;

        params.phase = 2;
        const auto compact = graph.add_checked_command_pass("compact GPU particle draw indices",
            [&, dispatch_params = params, visibility](VkCommandBuffer recording_command, std::string& pass_error) {
            return dispatch(recording_command, P::Compact, base + 2, dispatch_params, visibility,
                visible_roots, pass_error);
        });
        for (B::Resource resource : {B::VisibleRoots, B::Particles, B::Emitters, B::Programs,
                B::DrawArguments, B::BucketOffsets})
            if (!read(compact, resource, BufferUse::ComputeRead)) return false;
        if (!read_write(compact, B::BucketCursors) ||
            !write(compact, B::Indices, BufferUse::ComputeWrite)) return false;

        draw_data_consumer = graph.add_pass("publish GPU particle draw data",
            FrameSubmissionGraph::Queue::Graphics, {});
        if (!read(draw_data_consumer, B::Particles, BufferUse::VertexRead) ||
            !read(draw_data_consumer, B::Emitters, BufferUse::VertexRead) ||
            !read(draw_data_consumer, B::Programs, BufferUse::VertexRead) ||
            !read(draw_data_consumer, B::Indices, BufferUse::VertexRead) ||
            !read(draw_data_consumer, B::BucketOffsets, BufferUse::VertexRead) ||
            !read(draw_data_consumer, B::DrawArguments, BufferUse::IndirectRead)) return false;
    }

    if (readback_record)
    {
        const auto readback = graph.add_checked_command_pass("stage GPU particle status readback",
            recording_queue, std::move(readback_record));
        if (!read(readback, B::Status, BufferUse::TransferSource) ||
            !read(readback, B::Errors, BufferUse::TransferSource))
            return false;
    }

    const bool use_generated_submissions = submission_graph &&
        graphics_continuation_pass != FrameSubmissionGraph::InvalidPass &&
        request_submission_command_buffer;
    if (use_generated_submissions)
    {
        std::vector<FrameImageGraph::QueueGroup> queue_groups;
        std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
        if (!graph.generate_submission_groups(*submission_graph,
                [graphics_queue_family, compute_queue_family, copy_queue_family](FrameSubmissionGraph::Queue queue) {
                    if (queue == FrameSubmissionGraph::Queue::Compute) return compute_queue_family;
                    if (queue == FrameSubmissionGraph::Queue::Graphics) return graphics_queue_family;
                    return copy_queue_family;
                }, queue_groups, submission_for_pass, error))
            return false;

        FrameImageGraph::ExecutionState execution;
        if (!graph.begin_execution(execution, error))
            return false;
        ImageStateTracker image_states;
        const ImageStateDispatch state_dispatch{barrier_};
        for (const FrameImageGraph::QueueGroup& group : queue_groups)
        {
            const FrameSubmissionGraph::Pass submission = group.submission;
            VkCommandBuffer group_command = VK_NULL_HANDLE;
            if (submission == FrameSubmissionGraph::InvalidPass ||
                !request_submission_command_buffer(submission, group.queue, group_command, error))
                return false;
            const VkQueueFlags group_queue_flags = group.queue == FrameSubmissionGraph::Queue::Compute ?
                queue_flags : group.queue == FrameSubmissionGraph::Queue::Copy ?
                    VK_QUEUE_TRANSFER_BIT : graphics_queue_flags;
            if (!graph.execute_group(group, group_command, image_states, execution,
                    state_dispatch, error, group_queue_flags, semaphore_handoff))
                return false;
        }
    }
    else if (submission_graph && compute_submission_pass != FrameSubmissionGraph::InvalidPass &&
        graphics_continuation_pass != FrameSubmissionGraph::InvalidPass &&
        !graph.connect_submission_dependencies(*submission_graph,
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

    if (!use_generated_submissions)
    {
        ImageStateTracker image_states;
        const ImageStateDispatch state_dispatch{barrier_};
        if (!graph.execute(command, image_states, state_dispatch, error, queue_flags, semaphore_handoff,
                recording_queue)) return false;
    }
    error.clear(); return true;
}
}
