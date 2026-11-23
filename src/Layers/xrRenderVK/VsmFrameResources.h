#pragma once
#include "FrameImagePool.h"
#include "VsmDemandPass.h"
#include "VsmPageAllocator.h"
#include "VsmStaticCache.h"
#include "VsmResolvePass.h"

namespace xray::render::vulkan
{
// Owns the swapchain-sized VSM resources. The caller serializes use of the
// shared atlases and cache, and waits for GPU idle before destruction.
class VsmFrameResources
{
public:
    struct Image
    {
        BufferResource demand_parameters, needed, static_table, dynamic_table;
        BufferResource static_work, dynamic_pages, dynamic_counts, resolve_parameters;
        VkImage output{};
        VkImageView output_view{};
        VkImageLayout output_layout{VK_IMAGE_LAYOUT_UNDEFINED};
    };
    ~VsmFrameResources() { destroy(); }
    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        VkExtent2D extent, const std::vector<VkImageView>& depth, VkSampler depth_sampler, VkFormat shadow_format,
        uint32_t axis, uint32_t levels, uint32_t page_size, uint32_t dynamic_columns,
        uint32_t dynamic_rows, const VkPhysicalDeviceProperties& properties,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& buffers,
        const FrameDispatch& images, FrameImagePool& pool, VkPipelineCache pipeline_cache, std::string& error);
    void destroy();
    Image& image(uint32_t index) { return images_.at(index); }
    uint32_t count() const { return static_cast<uint32_t>(images_.size()); }
    VsmAtlas static_atlas, dynamic_atlas;
    BufferResource static_slots, visible_static_table, visible_dynamic_table;
    VsmDemandPass demand;
    VsmPageAllocator allocator;
    VsmStaticCache cache;
    VsmResolvePass resolve;
private:
    VkDevice device_{};
    FrameDispatch dispatch_{};
    FrameImagePool* pool_{};
    std::vector<Image> images_;
};
}
