#include "GpuParticleUpload.h"
#include "FrameImageGraph.h"

namespace xray::render::vulkan
{
bool GpuParticleUpload::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkDeviceSize bytes,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error)
{
    destroy();
    if (!device || !get || !bytes || bytes % 4)
    { error = "invalid GPU particle staging configuration"; return false; }
    copy_ = reinterpret_cast<PFN_vkCmdCopyBuffer>(get(device, "vkCmdCopyBuffer"));
    barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    if (!copy_ || !barrier_)
    { error = "GPU particle upload procedures unavailable"; destroy(); return false; }
    if (!staging_.initialize(device, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, dispatch, error))
    { destroy(); return false; }
    error.clear(); return true;
}
bool GpuParticleUpload::prepare(const std::vector<Update>& updates, std::string& error)
{
    if (!staging_.handle() || !copy_ || !barrier_ || updates.empty())
    { error = "invalid GPU particle upload"; return false; }
    VkDeviceSize total = 0;
    for (size_t i = 0; i < updates.size(); ++i)
    {
        const auto& item = updates[i];
        if (!item.destination || !item.destination->handle() || !item.data || !item.size ||
            item.offset % 4 || item.size % 4 || item.offset > item.destination->size() ||
            item.size > item.destination->size() - item.offset || total > staging_.size() ||
            item.size > staging_.size() - total)
        { error = "invalid GPU particle upload range"; return false; }
        for (size_t j = 0; j < i; ++j)
            if (updates[j].destination->handle() == item.destination->handle() &&
                updates[j].offset < item.offset + item.size && item.offset < updates[j].offset + updates[j].size)
            { error = "overlapping GPU particle uploads"; return false; }
        total += item.size;
    }
    total = 0;
    for (const auto& item : updates)
    {
        if (!staging_.write(total, item.data, item.size, error)) return false;
        total += item.size;
    }
    error.clear(); return true;
}

bool GpuParticleUpload::record_copy(VkCommandBuffer command,
    const std::vector<Update>& updates, std::string& error) const
{
    if (!command || !staging_.handle() || !copy_ || updates.empty())
    { error = "invalid GPU particle copy recording"; return false; }
    VkDeviceSize total = 0;
    for (const auto& item : updates)
    {
        if (!item.destination || !item.destination->handle() || !item.size ||
            item.offset > item.destination->size() || item.size > item.destination->size() - item.offset ||
            total > staging_.size() || item.size > staging_.size() - total)
        { error = "invalid GPU particle upload range during copy recording"; return false; }
        const VkBufferCopy region{total, item.offset, item.size};
        copy_(command, staging_.handle(), item.destination->handle(), 1, &region);
        total += item.size;
    }
    error.clear(); return true;
}

bool GpuParticleUpload::record(VkCommandBuffer command, const std::vector<Update>& updates,
    VkQueueFlags queue_flags, std::string& error,
    FrameSubmissionGraph::Queue recording_queue)
{
    if (!command || !barrier_ || !prepare(updates, error))
        return false;
    VkDeviceSize total = 0;
    for (const auto& item : updates) total += item.size;
    FrameImageGraph graph(recording_queue);
    FrameImageGraph::BufferResource staging_resource;
    if (!graph.import_buffer(staging_.handle(), 0, total, BufferUse::HostWrite,
            staging_resource, error)) return false;
    std::vector<FrameImageGraph::BufferResource> destinations(updates.size());
    std::vector<FrameImageGraph::BufferResource> uploaded(updates.size());
    for (size_t i = 0; i < updates.size(); ++i)
    {
        const auto& item = updates[i];
        if (!graph.import_buffer(item.destination->handle(), item.offset, item.size,
                BufferUse::AllCommandsReadWrite, destinations[i], error)) return false;
    }
    const auto upload = graph.add_checked_command_pass("upload GPU particle inputs", recording_queue,
        [this, &updates](VkCommandBuffer recording_command, std::string& pass_error) {
            return record_copy(recording_command, updates, pass_error);
        });
    if (!graph.read_buffer(upload, staging_resource, BufferUse::TransferSource, error)) return false;
    for (size_t i = 0; i < updates.size(); ++i)
        if (!graph.write_buffer(upload, destinations[i], BufferUse::TransferDestination,
                uploaded[i], error)) return false;
    ImageStateTracker image_states;
    const ImageStateDispatch state_dispatch{barrier_};
    if (!graph.execute(command, image_states, state_dispatch, error, queue_flags, false,
            recording_queue)) return false;
    error.clear(); return true;
}
void GpuParticleUpload::destroy()
{
    staging_.destroy(); copy_ = nullptr; barrier_ = nullptr;
}
}
