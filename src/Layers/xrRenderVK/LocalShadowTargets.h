#pragma once

#include "BufferResource.h"
#include "FrameContext.h"
#include "VulkanGameLightingMath.h"
#include "LightingLimits.h"

#include <array>
#include <vector>

namespace xray::render::vulkan
{
constexpr uint32_t LocalShadowSlots = 4;
constexpr uint32_t LocalShadowFaces = 6;
constexpr uint32_t LocalShadowSize = 512;
constexpr float LocalShadowReceiverOffset = .005f; // metres toward the light
constexpr uint32_t LocalLightCapacity = XRAY_VK_MAX_LOCAL_LIGHTS;

struct alignas(16) LocalLightUniform
{
    float inverse_view_projection[16]{};
    float shadow_matrices[LocalShadowFaces][16]{};
    float position_range[4]{};
    float direction_cone[4]{};
    float color_type[4]{};
    float shadow_params[4]{}; // first face, world-space bias, enabled, projection enabled
    float spot_right[4]{};    // right.xyz, tan(cone/2)
    float debug_mode[4]{};    // debug pane, volumetric intensity, quality, range multiplier
    int32_t shadow_tiles[LocalShadowFaces][4]{}; // x, y, size, layer
};
static_assert(sizeof(LocalLightUniform) % 16 == 0);

// A bounded, frame-image-owned array avoids reusing a depth layer or uniform
// while that swapchain image is in flight. Slots are reassigned after its fence.
class LocalShadowTargets
{
  public:
    ~LocalShadowTargets()
    {
        destroy();
    }
    bool initialize(VkDevice device, VkFormat format, uint32_t image_count,
                    const VkPhysicalDeviceMemoryProperties &memory, const FrameDispatch &dispatch,
                    const BufferResourceDispatch &buffers, PFN_vkCreateSampler create_sampler,
                    PFN_vkDestroySampler destroy_sampler, std::string &error, VkDeviceSize uniform_alignment = 256,
                    FrameImagePool* image_pool = nullptr);
    bool reserve_lights(uint32_t image, size_t count, std::string &error);
    bool write_light(uint32_t image, uint32_t index, const LocalLightUniform &value, std::string &error);
    struct TileRequest { uint32_t resolution{}, faces{}; };
    void layout_tiles(const std::array<TileRequest, LocalShadowSlots>& requests);
    VkExtent2D face_extent(uint32_t slot, uint32_t face) const;
    bool initialize_layers(const FrameRecordingContext &frame, bool graph_managed_layouts = false);
    bool begin_face(const FrameRecordingContext &frame, uint32_t slot, uint32_t face,
                    FrameRecordingContext &shadow_frame, bool cached = false, bool preserve = false,
                    bool graph_managed_layouts = false) const;
    bool cached_face(uint32_t slot, uint32_t face, const float *matrix) const;
    void remember_face(uint32_t slot, uint32_t face, const float *matrix);
    void commit_cache();
    bool copy_cached_face(const FrameRecordingContext &frame, uint32_t slot, uint32_t face) const;
    bool record_cached_face_copy(VkCommandBuffer command, uint32_t image, uint32_t slot, uint32_t face) const;
    bool caching() const
    {
        return cache_index_ != UINT32_MAX;
    }
    void invalidate_cache()
    {
        cache_signatures_.fill(0);
        pending_cache_signatures_.fill(0);
    }
    void end(VkCommandBuffer command) const;
    VkImageView view(uint32_t image) const;
    VkImage image(uint32_t index) const
    {
        return index < image_count_ ? targets_[index].image : VK_NULL_HANDLE;
    }
    VkImage cache_image() const
    {
        return caching() && cache_index_ < targets_.size() ? targets_[cache_index_].image : VK_NULL_HANDLE;
    }
    VkImageAspectFlags image_aspects() const { return barrier_aspects_; }
    bool layers_initialized(uint32_t image) const
    {
        return image < image_count_ && targets_[image].initialized;
    }
    bool cache_layers_initialized() const
    {
        return caching() && cache_index_ < targets_.size() && targets_[cache_index_].initialized;
    }
    VkBuffer uniform(uint32_t image) const;
    VkSampler sampler() const
    {
        return sampler_;
    }
    VkRenderPass render_pass() const
    {
        return tile_pass_;
    }
    VkRenderPass load_render_pass() const { return load_pass_; }
    VkRenderPass graph_render_pass() const { return graph_pass_; }
    VkRenderPass graph_load_render_pass() const { return graph_load_pass_; }
    VkDeviceSize uniform_stride() const
    {
        return uniform_stride_;
    }
    void destroy();

  private:
    struct Target
    {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView array_view{};
        VkFramebuffer framebuffer{};
        BufferResource constants;
        bool initialized{};
    };
    VkExtent2D atlas_extent_{};
    std::array<VkRect2D, LocalShadowSlots * LocalShadowFaces> tiles_{}, fixed_tiles_{};
    std::array<TileRequest, LocalShadowSlots> last_tile_requests_{};
    bool tile_layout_valid_{};
    VkImageAspectFlags barrier_aspects_{VK_IMAGE_ASPECT_DEPTH_BIT};
    VkDevice device_{};
    VkRenderPass pass_{}, tile_pass_{}, load_pass_{}, graph_pass_{}, graph_load_pass_{};
    uint32_t image_count_{}, cache_index_{UINT32_MAX};
    std::array<uint64_t, LocalShadowSlots * LocalShadowFaces> cache_signatures_{};
    std::array<uint64_t, LocalShadowSlots * LocalShadowFaces> pending_cache_signatures_{};
    VkSampler sampler_{};
    PFN_vkDestroySampler destroy_sampler_{};
    FrameDispatch vk_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    BufferResourceDispatch buffers_{};
    FrameImagePool* image_pool_{};
    VkDeviceSize uniform_stride_{sizeof(LocalLightUniform)};
    std::vector<Target> targets_;
};
} // namespace xray::render::vulkan
