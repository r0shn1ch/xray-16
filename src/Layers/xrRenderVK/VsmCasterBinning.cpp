#include "VsmCasterBinning.h"
#include "VsmBinShaders.h"
#include <cmath>

namespace xray::render::vulkan
{
bool VsmCasterBinning::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
    VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error)
{
    destroy();
    if (!device || !get) { error = "VSM caster binning needs a Vulkan device"; return false; }
    fill_ = reinterpret_cast<PFN_vkCmdFillBuffer>(get(device, "vkCmdFillBuffer"));
    barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    if (!fill_ || !barrier_ || !pass_.initialize(device, get, frame_slots, 4, 16, max_storage_range,
            cache, vsm_bin_shaders::Casters, sizeof(vsm_bin_shaders::Casters), error))
    { if (error.empty()) error = "VSM caster binning procedures unavailable"; destroy(); return false; }
    return true;
}

bool VsmCasterBinning::record(VkCommandBuffer command, uint32_t frame_slot, uint32_t caster_count,
    uint32_t job_budget, uint32_t pair_capacity, const VsmCasterBinningParameters& params,
    const BufferResource& casters, const BufferResource& jobs, BufferResource& params_buffer,
    const BufferResource& pairs, std::string& error) const
{
    if (!command || !fill_ || !barrier_ || !params.axis || params.axis > 32768 || !params.levels || params.levels > 6 ||
        uint64_t(params.axis) * params.axis * params.levels > UINT32_MAX ||
        job_budget > 65535 || job_budget > uint64_t(params.axis) * params.axis * params.levels ||
        (uint64_t(caster_count) + 63) / 64 > 65535 ||
        uint64_t(caster_count) * job_budget > UINT32_MAX || !casters.handle() || !jobs.handle() ||
        !params_buffer.handle() || !pairs.handle() || casters.size() < uint64_t(caster_count) * sizeof(VsmCasterBounds) ||
        jobs.size() < 16 + uint64_t(job_budget) * 16 || params_buffer.size() < sizeof(params) ||
        pairs.size() < 16 + uint64_t(pair_capacity) * 8 || !std::isfinite(params.zrange[0]) ||
        !std::isfinite(params.zrange[1]) || params.zrange[0] > params.zrange[1])
    { error = "invalid VSM caster binning request"; return false; }
    for (uint32_t i = 0; i < params.levels; ++i)
        if (!std::isfinite(params.level[i][0]) || !std::isfinite(params.level[i][1]) ||
            !std::isfinite(params.level[i][2]) || params.level[i][2] <= 0.f)
        { error = "invalid VSM caster binning clipmap"; return false; }
    if (!params_buffer.write(0, &params, sizeof(params), error)) return false;
    VkMemoryBarrier before_fill{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before_fill.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    before_fill.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier_(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &before_fill, 0, nullptr, 0, nullptr);
    fill_(command, pairs.handle(), 0, 16, 0);
    fill_(command, jobs.handle(), 4, 4, 0);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier_(command, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    if (!caster_count || !job_budget) { error.clear(); return true; }
    const uint32_t limits[]{caster_count, job_budget, pair_capacity, 0};
    const std::vector<VkDescriptorBufferInfo> buffers{{casters.handle(), 0, casters.size()},
        {jobs.handle(), 0, jobs.size()}, {params_buffer.handle(), 0, sizeof(params)}, {pairs.handle(), 0, pairs.size()}};
    return pass_.record(command, frame_slot, buffers, limits, (caster_count + 63) / 64, job_budget, error);
}
void VsmCasterBinning::destroy()
{ pass_.destroy(); fill_ = nullptr; barrier_ = nullptr; }
}
