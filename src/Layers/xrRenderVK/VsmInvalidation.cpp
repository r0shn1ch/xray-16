#include "VsmInvalidation.h"
#include "VsmInvalidateShaders.h"
#include <cmath>

namespace xray::render::vulkan
{
bool VsmInvalidation::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
    VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error)
{
    return pass_.initialize(device, get, frame_slots, 3, 16, max_storage_range, cache,
        vsm_invalidate_shaders::Invalidate, sizeof(vsm_invalidate_shaders::Invalidate), error);
}
bool VsmInvalidation::record(VkCommandBuffer command, uint32_t frame_slot, uint32_t change_count,
    const VsmInvalidationParameters& params, const BufferResource& changes,
    BufferResource& params_buffer, const BufferResource& slots, std::string& error) const
{
    if (!command || !params.axis || params.axis > 32768 || !params.levels || params.levels > 6 ||
        change_count > 65535 || !changes.handle() || !params_buffer.handle() || !slots.handle() ||
        changes.size() < uint64_t(change_count) * sizeof(VsmCasterBounds) || params_buffer.size() < sizeof(params) ||
        !std::isfinite(params.zrange[0]) || !std::isfinite(params.zrange[1]) || params.zrange[0] > params.zrange[1])
    { error = "invalid VSM spatial invalidation request"; return false; }
    const uint64_t count = uint64_t(params.axis) * params.axis * params.levels;
    if (count > UINT32_MAX || (count + 63) / 64 > 65535 || slots.size() < count * 16)
    { error = "VSM spatial invalidation slot count exceeds limits"; return false; }
    for (uint32_t i = 0; i < params.levels; ++i)
        if (!std::isfinite(params.level[i][0]) || !std::isfinite(params.level[i][1]) ||
            !std::isfinite(params.level[i][2]) || params.level[i][2] <= 0.f)
        { error = "invalid VSM spatial invalidation clipmap"; return false; }
    if (!change_count) { error.clear(); return true; }
    if (!params_buffer.write(0, &params, sizeof(params), error)) return false;
    const uint32_t limits[]{static_cast<uint32_t>(count), change_count, 0, 0};
    const std::vector<VkDescriptorBufferInfo> buffers{{changes.handle(), 0, changes.size()},
        {params_buffer.handle(), 0, sizeof(params)}, {slots.handle(), 0, slots.size()}};
    return pass_.record(command, frame_slot, buffers, limits, static_cast<uint32_t>((count + 63) / 64), change_count, error);
}
}
