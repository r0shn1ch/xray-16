#include "VsmPageAllocator.h"
#include "VsmAllocateShaders.h"
#include <algorithm>
#include <limits>

namespace xray::render::vulkan
{
namespace
{
constexpr uint32_t allocation_phases = 3;
}

bool VsmPageAllocator::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
    VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error)
{
    destroy();
    if (!device || !get || frame_slots > UINT32_MAX / allocation_phases)
    { error = "VSM page allocator needs a Vulkan device and valid frame slots"; return false; }
    fill_ = reinterpret_cast<PFN_vkCmdFillBuffer>(get(device, "vkCmdFillBuffer"));
    barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    if (!fill_ || !barrier_ || !pass_.initialize(device, get, frame_slots * allocation_phases, 4, 20, max_storage_range,
            cache, vsm_allocate_shaders::Dynamic, sizeof(vsm_allocate_shaders::Dynamic), error))
    { if (error.empty()) error = "VSM page allocator procedures unavailable"; destroy(); return false; }
    return true;
}

bool VsmPageAllocator::record(VkCommandBuffer command, uint32_t frame_slot, uint32_t axis, uint32_t levels,
    uint32_t physical_pages, const BufferResource& needed, const BufferResource& table,
    const BufferResource& pages, const BufferResource& counters, std::string& error) const
{
    if (!axis || axis > 65535u || !levels || levels > 65535u)
    { error = "VSM page allocator dimensions are invalid"; return false; }
    const uint64_t count64 = uint64_t(axis) * axis * levels;
    if (!command || !fill_ || !barrier_ || frame_slot >= UINT32_MAX / allocation_phases || !count64 || count64 > UINT32_MAX ||
        count64 > uint64_t(65535) * 65535 * 64 || !needed.handle() || !table.handle() ||
        !pages.handle() || !counters.handle() || needed.size() < count64 * 4 || table.size() < count64 * 4 ||
        pages.size() < std::max<uint64_t>(1, physical_pages) * 16 || counters.size() < (uint64_t(levels) + 1) * 4)
    { error = "VSM page allocator buffer sizes or dimensions are invalid"; return false; }
    const uint32_t count = static_cast<uint32_t>(count64);
    const uint32_t groups = static_cast<uint32_t>((count64 + 63) / 64);
    const uint32_t groups_x = std::min(groups, 65535u);
    uint32_t constants[]{count, axis, physical_pages, groups_x, 0};
    fill_(command, counters.handle(), 0, (VkDeviceSize(levels) + 1) * 4, 0);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier_(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    const std::vector<VkDescriptorBufferInfo> buffers{
        {needed.handle(), 0, needed.size()}, {table.handle(), 0, table.size()},
        {pages.handle(), 0, pages.size()}, {counters.handle(), 0, counters.size()}};
    const uint32_t groups_y = (groups + groups_x - 1) / groups_x;
    for (uint32_t phase = 0; phase < allocation_phases; ++phase)
    {
        constants[4] = phase;
        if (!pass_.record(command, frame_slot * allocation_phases + phase, buffers, constants, groups_x, groups_y, error))
            return false;
    }
    return true;
}

void VsmPageAllocator::destroy()
{
    pass_.destroy();
    fill_ = nullptr;
    barrier_ = nullptr;
}
}
