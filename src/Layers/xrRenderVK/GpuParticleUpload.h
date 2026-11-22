#pragma once
#include "BufferResource.h"
#include "FrameSubmissionGraph.h"
#include <nvrhi/nvrhi.h>
#include <vector>

namespace xray::render::vulkan
{
// One instance belongs to one fence-protected frame slot. Reuse or destroy it
// only after all submissions referencing its staging buffer have completed;
// record at most one batch per slot before that completion.
class GpuParticleUpload
{
public:
    struct Update { const BufferResource* destination; VkDeviceSize offset; const void* data; size_t size; };
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkDeviceSize bytes,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error);
    bool prepare(const std::vector<Update>& updates, std::string& error);
    bool record_copy(VkCommandBuffer command, const std::vector<Update>& updates,
        std::string& error, nvrhi::ICommandList* nvrhi_command = nullptr) const;
    bool record(VkCommandBuffer command, const std::vector<Update>& updates,
        VkQueueFlags queue_flags, std::string& error,
        FrameSubmissionGraph::Queue recording_queue = FrameSubmissionGraph::Queue::Copy);
    VkDeviceSize capacity() const { return staging_.size(); }
    VkBuffer staging_buffer() const { return staging_.handle(); }
    nvrhi::IBuffer* nvrhi_staging_buffer() const { return nvrhi_staging_.Get(); }
    void destroy();
private:
    BufferResource staging_;
    nvrhi::BufferHandle nvrhi_staging_;
    PFN_vkCmdCopyBuffer copy_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
