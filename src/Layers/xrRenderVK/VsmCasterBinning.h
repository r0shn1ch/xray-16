#pragma once
#include "StorageComputePass.h"

namespace xray::render::vulkan
{
struct VsmCasterBounds { float minimum[4]{}, maximum[4]{}; };
static_assert(sizeof(VsmCasterBounds) == 32);
struct VsmCasterBinningParameters
{
    float level[6][4]{};
    float zrange[4]{};
    uint32_t axis{}, levels{}, reserved[2]{};
};
static_assert(sizeof(VsmCasterBinningParameters) == 128);

// Bins light-space caster bounds against published static-cache jobs. The
// output begins with a 16-byte header (match count, overflow count), followed
// by {caster index, physical page} pairs. Caller owns and synchronizes buffers;
// pairs and jobs require TRANSFER_DST usage. Overflow marks the job header so
// VsmStaticCache::publish automatically withholds incomplete pages.
class VsmCasterBinning
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
        VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error);
    bool record(VkCommandBuffer command, uint32_t frame_slot, uint32_t caster_count,
        uint32_t job_budget, uint32_t pair_capacity, const VsmCasterBinningParameters& parameters,
        const BufferResource& casters, const BufferResource& jobs, BufferResource& params_buffer,
        const BufferResource& pairs, std::string& error) const;
    void destroy();
private:
    StorageComputePass pass_;
    PFN_vkCmdFillBuffer fill_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
