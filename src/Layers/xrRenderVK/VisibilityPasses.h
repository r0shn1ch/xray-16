#pragma once

#include "StorageComputePass.h"
#include "BindlessTextureTable.h"
#include "FrameContext.h"
#include "Layers/xrRenderGraph/VisibilityId.h"

#include <functional>

namespace xray::render::vulkan
{
class GpuLevel;
class VisibilityPasses
{
public:
    using Source = std::function<bool(const char*, std::vector<uint8_t>&, std::string&)>;

    struct Geometry
    {
        VkDescriptorBufferInfo entries{}, indices{}, vertices{};
    };
    enum class RasterPhase : uint32_t { Depth, Id };
    struct Target { VkImage image{}; VkImageView view{}; };
    struct Targets
    {
        Target ids, depth, base_color, normals;
        VkExtent2D extent{};
        VkSampler id_sampler{};
    };

    // Caller owns one target set per in-flight slot and supplies images with
    // STORAGE, TRANSFER_DST and SAMPLED usage. Previous contents are discarded.
    bool record(const FrameRecordingContext& frame, const GpuLevel& level, const Targets& targets,
        const visibility::Camera& camera, const BindlessTextureTable& materials,
        VkBufferView draw_mask, std::string& error) const;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, uint32_t batch_count, VkPipelineCache cache, const BindlessTextureTable& materials, const Source& source,
        std::string& error);

    // Record all depth batches before any ID batch. Each batch and phase has
    // its own descriptor set in the frame slot.
    // The caller supplies equally sized R32_UINT images in GENERAL layout,
    // cleared to ID 0 and floatBitsToUint(1.0), with clears visible to compute.
    // Draw entries and the camera commands share an index. Unselected commands
    // have no indices or instances, and only entries in the draw range are read.
    bool rasterize(VkCommandBuffer command, uint32_t slot, uint32_t batch, RasterPhase phase, const Geometry& geometry,
        const VkDescriptorBufferInfo& draw_commands, VkBufferView draw_mask,
        visibility::SoftwareRasterParameters parameters,
        VkImageView ids, VkImageView depth, std::string& error) const;

    // The RGBA16F outputs and ID image cover the same extent. Background
    // pixels retain their previous output values.
    bool resolve(VkCommandBuffer command, uint32_t slot, uint32_t batch, const Geometry& geometry,
        const VkDescriptorImageInfo& ids, const BindlessTextureTable& materials,
        VkImageView base_color, VkImageView normals, VkExtent2D extent,
        visibility::MaterialResolveParameters parameters, std::string& error) const;

private:
    StorageComputePass raster_, resolve_;
    uint32_t slots_{};
    uint32_t batch_count_{};
    VkDescriptorSetLayout material_layout_{};
    VkExtent2D max_dispatch_{};
    PFN_vkCmdClearColorImage clear_{};
    PFN_vkCmdPipelineBarrier barrier_{};
};
}
