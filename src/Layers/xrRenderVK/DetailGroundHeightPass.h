#pragma once
#include "DetailGeneratePass.h"
#include "DetailGroundHeightShaders.h"

namespace xray::render::vulkan
{
struct DetailGroundHeightParameters
{
    float origin[2]{}, texel_size{.5f};
    uint32_t size{130};
    uint32_t level_size[2]{};
    int32_t offset[2]{};
    uint32_t cell_count{};
    float minimum_normal_y{};
    uint32_t select_lowest{}, padding{};
};
static_assert(sizeof(DetailGroundHeightParameters)==48);

// The source cells are ordered by their world cell ID, as in GpuLevel's
// resident ground stream. A height texel needs one binary search, followed by
// ray/triangle tests only within that cell. No CPU readback is involved.
class DetailGroundHeightPass
{
public:
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, std::string& error)
    {
        if (limits.maxPushConstantsSize<sizeof(DetailGroundHeightParameters))
        { error="GPU detail heightmap push constants unavailable"; return false; }
        max_groups_x_=limits.maxComputeWorkGroupCount[0]; max_groups_y_=limits.maxComputeWorkGroupCount[1];
        return pass_.initialize(device,get,slots,{{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
            {1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},{2,VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}},
            sizeof(DetailGroundHeightParameters),limits.maxStorageBufferRange,limits.maxUniformBufferRange,
            cache,detail_ground_height_shaders::Compute,sizeof(detail_ground_height_shaders::Compute),error);
    }
    bool record(VkCommandBuffer command, uint32_t slot, const BufferResource& cells, const BufferResource& triangles,
        VkImageView destination, const DetailGroundHeightParameters& params, std::string& error) const
    {
        const uint32_t groups=(params.size+7)/8;
        if (!params.size || params.size>16384 || groups>max_groups_x_ || groups>max_groups_y_ ||
            !params.level_size[0] || !params.level_size[1] ||
            params.level_size[0]>INT32_MAX || params.level_size[1]>INT32_MAX ||
            uint64_t(params.level_size[0])*params.level_size[1]>UINT32_MAX ||
            uint64_t(params.cell_count)*sizeof(DetailGenerateSlot)>cells.size() ||
            cells.size()<sizeof(DetailGenerateSlot) || triangles.size()<sizeof(DetailGroundTriangle) || !destination ||
            !std::isfinite(params.texel_size) || params.texel_size<=0.f ||
            !std::isfinite(params.origin[0]) || !std::isfinite(params.origin[1]) ||
            !std::isfinite(params.minimum_normal_y) || params.minimum_normal_y<0.f || params.minimum_normal_y>1.f)
        { error="invalid GPU detail heightmap inputs"; return false; }
        return pass_.record_mixed(command,slot,{{cells.handle(),0,cells.size()},{triangles.handle(),0,triangles.size()}},
            {{VK_NULL_HANDLE,destination,VK_IMAGE_LAYOUT_GENERAL}},&params,groups,groups,error);
    }
private:
    StorageComputePass pass_;
    uint32_t max_groups_x_{},max_groups_y_{};
};
}
