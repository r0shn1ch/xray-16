#pragma once

#include "StorageComputePass.h"

namespace xray::render::vulkan
{
// GPU allocation for dynamic virtual-shadow pages. All buffers and descriptors
// belong to the caller and are reused only after the corresponding frame fence.
// Counters need TRANSFER_DST usage; counts[0] includes overflow, counts[1+L]
// counts accepted pages at each level. Unneeded/overflow pages are UINT32_MAX.
class VsmPageAllocator
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
        VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error);
    bool record(VkCommandBuffer command, uint32_t frame_slot, uint32_t axis, uint32_t levels,
        uint32_t physical_pages, const BufferResource& needed, const BufferResource& table,
        const BufferResource& pages, const BufferResource& counters, std::string& error) const;
    void destroy();
private:
    StorageComputePass pass_;
    PFN_vkCmdFillBuffer fill_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
