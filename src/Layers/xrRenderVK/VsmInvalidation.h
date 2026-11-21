#pragma once
#include "VsmCasterBinning.h"

namespace xray::render::vulkan
{
struct VsmInvalidationParameters
{
    int32_t base[6][4]{};
    float level[6][4]{};
    float zrange[4]{};
    uint32_t axis{}, levels{}, reserved[2]{};
};
static_assert(sizeof(VsmInvalidationParameters) == 224);

// Invalidates cached pages intersecting changed light-space geometry bounds.
// Record before VsmStaticCache::prepare, with persistent slots synchronized;
// the caller must include both old and new bounds for moved/removed casters.
// Page scale or light-view rotation changes require full-cache invalidation.
class VsmInvalidation
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
        VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error);
    bool record(VkCommandBuffer command, uint32_t frame_slot, uint32_t change_count,
        const VsmInvalidationParameters& params, const BufferResource& changes,
        BufferResource& params_buffer, const BufferResource& slots, std::string& error) const;
    void destroy() { pass_.destroy(); }
private:
    StorageComputePass pass_;
};
}
