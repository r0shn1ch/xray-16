#include "xrEngine/stdafx.h"
#include "GpuParticleReadback.h"
#include "FrameImageGraph.h"

namespace xray::render::vulkan
{
bool GpuParticleReadback::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t roots,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error)
{
    destroy();
    if (!device || !get || !roots) { error = "invalid GPU particle readback configuration"; return false; }
    copy_ = reinterpret_cast<PFN_vkCmdCopyBuffer>(get(device, "vkCmdCopyBuffer"));
    barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    if (!copy_ || !barrier_) { error = "GPU particle readback procedures unavailable"; destroy(); return false; }
    const VkDeviceSize bytes = VkDeviceSize(roots) * sizeof(fg::GpuPapiStatus) + sizeof(uint32_t);
    if (!buffer_.initialize(device, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, dispatch, error))
    { destroy(); return false; }
    roots_ = roots; error.clear(); return true;
}
bool GpuParticleReadback::record(VkCommandBuffer command, const GpuParticleBuffers& buffers,
    VkQueueFlags queue_flags, std::string& error,
    FrameSubmissionGraph::Queue recording_queue) const
{
    const auto& statuses = buffers.get(GpuParticleBuffers::Status);
    const auto& errors = buffers.get(GpuParticleBuffers::Errors);
    const VkDeviceSize bytes = VkDeviceSize(roots_) * sizeof(fg::GpuPapiStatus);
    if (!command || !buffer_.handle() || !copy_ || !barrier_ || !statuses.handle() || statuses.size() < bytes ||
        !errors.handle() || errors.size() < sizeof(uint32_t))
    { error = "invalid GPU particle readback buffers"; return false; }
    FrameImageGraph graph(recording_queue);
    FrameImageGraph::BufferResource status_resource, error_resource, readback_resource, copied_resource;
    if (!graph.import_buffer(statuses.handle(), 0, bytes, BufferUse::AllCommandsReadWrite,
            status_resource, error) ||
        !graph.import_buffer(errors.handle(), 0, sizeof(uint32_t), BufferUse::AllCommandsReadWrite,
            error_resource, error) ||
        !graph.import_buffer(buffer_.handle(), 0, bytes + sizeof(uint32_t), BufferUse::Undefined,
            readback_resource, error)) return false;
    const auto copy_pass = graph.add_command_pass("copy GPU particle readback", recording_queue,
        [&](VkCommandBuffer recording_command) {
        const VkBufferCopy status_copy{0, 0, bytes}, error_copy{0, bytes, sizeof(uint32_t)};
        copy_(recording_command, statuses.handle(), buffer_.handle(), 1, &status_copy);
        copy_(recording_command, errors.handle(), buffer_.handle(), 1, &error_copy);
    });
    if (!graph.read_buffer(copy_pass, status_resource, BufferUse::TransferSource, error) ||
        !graph.read_buffer(copy_pass, error_resource, BufferUse::TransferSource, error) ||
        !graph.write_buffer(copy_pass, readback_resource, BufferUse::TransferDestination,
            copied_resource, error)) return false;
    const auto host_read = graph.add_pass("particle readback ready for host", {});
    if (!graph.read_buffer(host_read, copied_resource, BufferUse::HostRead, error)) return false;
    ImageStateTracker image_states;
    const ImageStateDispatch state_dispatch{barrier_};
    if (!graph.execute(command, image_states, state_dispatch, error, queue_flags, false,
            recording_queue)) return false;
    error.clear(); return true;
}
bool GpuParticleReadback::read(Result& result, std::string& error) const
{
    if (!roots_ || !buffer_.handle()) { error = "GPU particle readback not initialized"; return false; }
    Result output; output.statuses.resize(roots_);
    const size_t bytes = size_t(roots_) * sizeof(fg::GpuPapiStatus);
    if (!buffer_.read(0, output.statuses.data(), bytes, error) ||
        !buffer_.read(bytes, &output.errors, sizeof(output.errors), error)) return false;
    result = std::move(output); error.clear(); return true;
}
void GpuParticleReadback::destroy()
{
    buffer_.destroy(); roots_ = 0; copy_ = nullptr; barrier_ = nullptr;
}
}
