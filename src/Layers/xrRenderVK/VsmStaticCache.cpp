#include "VsmStaticCache.h"
#include "VsmCacheShaders.h"
#include <algorithm>
#include <limits>

namespace xray::render::vulkan
{
bool VsmStaticCache::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
    VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error)
{
    destroy();
    if (!device || !get) { error = "VSM static cache needs a Vulkan device"; return false; }
    fill_ = reinterpret_cast<PFN_vkCmdFillBuffer>(get(device, "vkCmdFillBuffer"));
    barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    if (!fill_ || !barrier_ ||
        !prepare_.initialize(device, get, frame_slots, 4, sizeof(VsmStaticCacheParameters), max_storage_range,
            cache, vsm_cache_shaders::Prepare, sizeof(vsm_cache_shaders::Prepare), error) ||
        !publish_.initialize(device, get, frame_slots, 4, sizeof(VsmStaticCacheParameters), max_storage_range,
            cache, vsm_cache_shaders::Publish, sizeof(vsm_cache_shaders::Publish), error))
    { if (error.empty()) error = "VSM static cache procedures unavailable"; destroy(); return false; }
    return true;
}

bool VsmStaticCache::reset(VkCommandBuffer command, const BufferResource& slots, std::string& error) const
{
    if (!command || !fill_ || !barrier_ || !slots.handle() || !slots.size() || slots.size() % 16)
    { error = "invalid VSM static cache reset"; return false; }
    fill_(command, slots.handle(), 0, slots.size(), 0);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier_(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    error.clear(); return true;
}
bool VsmStaticCache::prepare(VkCommandBuffer command, uint32_t slot, VsmStaticCacheParameters params,
    const BufferResource& needed, const BufferResource& table, const BufferResource& slots,
    const BufferResource& work, std::string& error) const
{ return record(false, command, slot, params, needed, table, slots, work, error); }
bool VsmStaticCache::publish(VkCommandBuffer command, uint32_t slot, VsmStaticCacheParameters params,
    const BufferResource& needed, const BufferResource& table, const BufferResource& slots,
    const BufferResource& work, std::string& error) const
{ return record(true, command, slot, params, needed, table, slots, work, error); }

bool VsmStaticCache::render_and_publish(const VsmAtlas& atlas, const FrameRecordingContext& frame,
    const VsmPageProjection& projection, VsmStaticCacheParameters params,
    const BufferResource& needed, const BufferResource& table, const BufferResource& slots,
    const BufferResource& work, VsmPageDraw draw, void* user, std::string& error,
    VsmPagePrepare prepare) const
{
    if (projection.axis != params.axis || projection.levels != params.levels || atlas.page_count() > UINT32_MAX)
    { error = "VSM static page projection does not match its cache"; return false; }
    std::vector<VsmPageJob> jobs;
    if (!read_vsm_static_jobs(work, params.axis, params.levels, params.budget,
            static_cast<uint32_t>(atlas.page_count()), jobs, error) ||
        !record_vsm_pages(atlas, frame, projection, jobs, draw, user, error, prepare)) return false;
    return publish(frame.command_buffer, frame.frame_index, params, needed, table, slots, work, error);
}

bool VsmStaticCache::record(bool publication, VkCommandBuffer command, uint32_t frame_slot,
    VsmStaticCacheParameters params, const BufferResource& needed, const BufferResource& table,
    const BufferResource& slots, const BufferResource& work, std::string& error) const
{
    if (!command || !fill_ || !barrier_ || !params.axis || params.axis > 32768 ||
        (params.axis & (params.axis - 1)) || !params.levels || params.levels > 6)
    { error = "invalid VSM static cache dimensions"; return false; }
    const uint64_t count = uint64_t(params.axis) * params.axis * params.levels;
    if (count > UINT32_MAX || params.budget > count || !needed.handle() || !table.handle() ||
        !slots.handle() || !work.handle() || needed.size() < count * 4 || table.size() < count * 4 ||
        slots.size() < count * 16 || work.size() < 16 + uint64_t(params.budget) * 16)
    { error = "invalid VSM static cache buffers or budget"; return false; }
    for (uint32_t i = 0; i < params.levels; ++i)
        for (unsigned axis = 0; axis < 2; ++axis)
            if (int64_t(params.base[i][axis]) + params.axis - 1 > std::numeric_limits<int32_t>::max())
            { error = "VSM static cache page coordinates overflow"; return false; }
    if (!publication)
    {
        fill_(command, work.handle(), 0, 16, 0);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier_(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    const uint64_t items = publication ? std::max<uint64_t>(1, params.budget) : count;
    const uint32_t groups = static_cast<uint32_t>((items + 63) / 64);
    params.groups_x = std::min(groups, 65535u);
    const std::vector<VkDescriptorBufferInfo> buffers{{needed.handle(), 0, needed.size()},
        {table.handle(), 0, table.size()}, {slots.handle(), 0, slots.size()}, {work.handle(), 0, work.size()}};
    return (publication ? publish_ : prepare_).record(command, frame_slot, buffers, &params,
        params.groups_x, (groups + params.groups_x - 1) / params.groups_x, error);
}

void VsmStaticCache::destroy()
{
    prepare_.destroy(); publish_.destroy(); fill_ = nullptr; barrier_ = nullptr;
}
}
