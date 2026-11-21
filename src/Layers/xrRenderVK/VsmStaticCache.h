#pragma once
#include "StorageComputePass.h"
#include "VsmPageRender.h"

namespace xray::render::vulkan
{
struct VsmStaticCacheParameters
{
    uint32_t axis{}, levels{}, frame{}, refresh_interval{};
    uint32_t budget{}, force_dirty{}, groups_x{}, reserved{};
    int32_t base[6][4]{}; // absolute page origin per level, xy
};
static_assert(sizeof(VsmStaticCacheParameters) == 128);

// Toroidal static shadow page cache. Persistent slots must be serialized
// across frames, cleared with reset() initially, and outlive both passes.
// Only call publish() after successfully rendering every listed job into the
// corresponding atlas page and making those writes visible to the resolver.
// The caller synchronizes buffer reuse and atlas rendering between frames.
class VsmStaticCache
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t frame_slots,
        VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error);
    bool reset(VkCommandBuffer command, const BufferResource& slots, std::string& error) const;
    bool prepare(VkCommandBuffer command, uint32_t frame_slot, VsmStaticCacheParameters params,
        const BufferResource& needed, const BufferResource& table, const BufferResource& slots,
        const BufferResource& work, std::string& error) const;
    bool publish(VkCommandBuffer command, uint32_t frame_slot, VsmStaticCacheParameters params,
        const BufferResource& needed, const BufferResource& table, const BufferResource& slots,
        const BufferResource& work, std::string& error) const;
    // work must be host-visible and its producing submission must have
    // completed. Keep the buffers, projection and atlas stable until this
    // command buffer completes; discard the command buffer on failure.
    bool render_and_publish(const VsmAtlas& atlas, const FrameRecordingContext& frame,
        const VsmPageProjection& projection, VsmStaticCacheParameters params,
        const BufferResource& needed, const BufferResource& table, const BufferResource& slots,
        const BufferResource& work, VsmPageDraw draw, void* user, std::string& error,
        VsmPagePrepare prepare = nullptr) const;
    void destroy();
private:
    bool record(bool publish, VkCommandBuffer command, uint32_t frame_slot, VsmStaticCacheParameters params,
        const BufferResource& needed, const BufferResource& table, const BufferResource& slots,
        const BufferResource& work, std::string& error) const;
    StorageComputePass prepare_, publish_;
    PFN_vkCmdFillBuffer fill_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
