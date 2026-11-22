#include "xrEngine/stdafx.h"
#include "GpuParticleFrameInput.h"
#include "GpuParticleNoise.h"
#include <unordered_set>

namespace xray::render::vulkan
{
bool upload_gpu_particle_frame(VkCommandBuffer command, const GpuParticleBuffers& buffers,
    GpuParticleUpload& upload, const GpuParticlePrograms& programs, const GpuParticleCommands::Batch& batch,
    const std::vector<uint32_t>& visible_roots, const GpuParticleCollisionInput* collision,
    bool upload_noise, bool force_programs, VkQueueFlags queue_flags,
    GpuParticleFrameInputResult& result, std::string& error,
    VkCommandBuffer upload_command, VkQueueFlags upload_queue_flags,
    FrameSubmissionGraph::Queue command_queue, FrameSubmissionGraph::Queue copy_queue,
    bool defer_record)
{
    using B = GpuParticleBuffers;
    if (!command || programs.programs().empty() || programs.programs().size() > UINT32_MAX ||
        batch.commands.size() > UINT32_MAX || batch.ranges.size() > UINT32_MAX || visible_roots.size() > UINT32_MAX)
    { error = "invalid GPU particle frame input counts"; return false; }
    for (const auto& program : programs.programs())
        if (program.materialID != GPU_PAPI_INVALID && program.materialID >= programs.programs().size())
        { error = "GPU particle material index exceeds the material table"; return false; }
    const auto is_root = [&](uint32_t index) { return index % 2 == 0 && index / 2 < buffers.root_slots(); };
    size_t next = 0;
    std::unordered_set<uint32_t> command_roots;
    for (const auto& range : batch.ranges)
    {
        if (!range[1] || range[0] != next || range[0] > batch.commands.size() ||
            range[1] > batch.commands.size() - range[0])
        { error = "invalid GPU particle command range"; return false; }
        const auto& first = batch.commands[range[0]];
        if (!is_root(first.emitter) || !command_roots.insert(first.emitter).second)
        { error = "invalid GPU particle command root"; return false; }
        uint32_t last_serial = 0;
        for (size_t i = range[0]; i < size_t(range[0]) + range[1]; ++i)
        {
            const auto& item = batch.commands[i];
            const auto valid_program = [&](uint32_t id) { return id == GPU_PAPI_INVALID || id < programs.programs().size(); };
            if (item.emitter != first.emitter || item.generation != first.generation || !item.generation ||
                item.serial <= last_serial || item.type > 7 ||
                (item.type == 0 && item.p0 >= programs.programs().size()) ||
                (item.type == 7 && (!valid_program(item.p0) || !valid_program(item.p1) || !valid_program(item.p2))))
            { error = "invalid GPU particle command sequence"; return false; }
            last_serial = item.serial;
        }
        next += range[1];
    }
    if (next != batch.commands.size()) { error = "unreferenced GPU particle commands"; return false; }
    std::unordered_set<uint32_t> visible;
    for (uint32_t index : visible_roots)
        if (!is_root(index) || !visible.insert(index).second)
        { error = "invalid visible GPU particle root"; return false; }
    std::vector<uint8_t> collision_bytes;
    if (collision && !pack_gpu_particle_collision(*collision, buffers.get(B::Collision).size(), collision_bytes, error))
        return false;
    std::array<Fvector4, 514> noise{};
    if (upload_noise) noise = gpu_particle_noise_table();
    std::vector<GpuParticleUpload::Update> updates;
    const bool update_programs = force_programs || programs.dirty();
    const auto add = [&](B::Resource id, const void* data, size_t bytes) {
        if (bytes) updates.push_back({&buffers.get(id), 0, data, bytes});
    };
    if (update_programs)
    {
        add(B::Programs, programs.programs().data(), programs.programs().size() * sizeof(fg::GpuPapiProgram));
        add(B::Actions, programs.actions().data(), programs.actions().size() * sizeof(fg::GpuPapiAction));
    }
    add(B::Commands, batch.commands.data(), batch.commands.size() * sizeof(fg::GpuPapiCommand));
    add(B::RootCommands, batch.ranges.data(), batch.ranges.size() * sizeof(batch.ranges[0]));
    add(B::VisibleRoots, visible_roots.data(), visible_roots.size() * sizeof(uint32_t));
    add(B::Collision, collision_bytes.data(), collision_bytes.size());
    if (upload_noise) add(B::Noise, noise.data(), sizeof(noise));
    const bool uploads_recorded = !updates.empty();
    if (uploads_recorded)
    {
        if (defer_record)
        {
            if (!upload.prepare(updates, error)) return false;
        }
        else if (!upload.record(upload_command ? upload_command : command, updates,
                upload_command && upload_queue_flags ? upload_queue_flags : queue_flags, error,
                upload_command && upload_queue_flags ? copy_queue : command_queue))
            return false;
    }
    result = {static_cast<uint32_t>(batch.ranges.size()), static_cast<uint32_t>(visible_roots.size()),
        static_cast<uint32_t>(batch.commands.size()), static_cast<uint32_t>(programs.programs().size()),
        update_programs, uploads_recorded};
    if (defer_record) result.uploads = std::move(updates);
    error.clear(); return true;
}
}
