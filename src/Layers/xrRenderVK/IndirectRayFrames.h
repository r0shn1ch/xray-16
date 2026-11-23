#pragma once

#include "Common/Common.hpp"
#include "RayGeneratePass.h"
#include "RaySurface.h"
#include "IndirectSunPass.h"
#include "StorageComputePass.h"
#include "RayVisibilityPass.h"
#include "ReferencePathTracePass.h"
#include "xrCore/xrCore.h"
#include "xrCommon/xr_vector.h"
#include <memory>

namespace xray::render::vulkan
{
constexpr float IndirectLocalLightSpotFlag = 1.f;
constexpr float IndirectLocalLightShadowFlag = 2.f;

struct IndirectLocalLight
{
    float position_range[4]{};
    float direction_cone[4]{};
    float color_type[4]{};
    float spot_right[4]{}; // normalized basis and tan(half cone)
    uint32_t projection_slot{UINT32_MAX}, padding[3]{};
};
static_assert(sizeof(IndirectLocalLight) == 80);

struct RestirGIReservoir
{
    float radiance[4]{};
    uint32_t packed_direction[2]{};
    float weight_sum{};
    uint32_t sample_count{};
    float target{};
    float proposal_pdf{};
    uint32_t packed_surface[2]{};
};
static_assert(sizeof(RestirGIReservoir) == 48);
// Per-swapchain-image indirect-ray resources. Input views and the alpha texture
// table outlive this object. Destroy or rebuild only after GPU work is idle.
class IndirectRayFrames
{
public:
    ~IndirectRayFrames() { destroy(); }
    IndirectRayFrames() = default;
    IndirectRayFrames(const IndirectRayFrames&) = delete;
    IndirectRayFrames& operator=(const IndirectRayFrames&) = delete;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
        const std::vector<VkImageView>& depth, const std::vector<VkImageView>& normal, VkSampler sampler,
        const VkPhysicalDeviceProperties& properties, const VkPhysicalDeviceMemoryProperties& memory,
        const BufferResourceDispatch& buffers, const BindlessTextureTable& alpha_textures,
        VkPipelineCache cache, std::string& error,
        const std::vector<VkImage>& depth_images = {}, const std::vector<VkImage>& normal_images = {},
        const std::vector<VkImageView>& albedo_views = {}, const std::vector<VkImageView>& surface_views = {},
        const std::vector<VkImage>& albedo_images = {}, const std::vector<VkImage>& surface_images = {},
        const FrameImageSharing* sharing = nullptr);
    // Images must be RGBA16F with storage and sampled usage and outlive this object.
    bool attach_output(VkDevice device, PFN_vkGetDeviceProcAddr get, const std::vector<VkImage>& images,
        const std::vector<VkImageView>& views, VkPipelineCache cache, std::string& error);
    bool prepare_output(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        const FrameDispatch& dispatch, FrameImagePool& pool, VkPipelineCache cache, std::string& error);
    bool record(const FrameRecordingContext& frame, const LevelRayScene& scene, RaySurfacePass& surface_pass,
        const float* inverse_view_projection, float max_distance, uint32_t seed,
        const float* sun_direction_distance, const float* sky_color, const float* sun_color,
        const std::vector<IndirectLocalLight>& local_lights, std::string& error,
        const std::vector<VkImageView>& projections = {}, VkSampler projection_sampler = VK_NULL_HANDLE,
        VkImageView motion_view = VK_NULL_HANDLE, bool motion_history_valid = false,
        bool temporal_history_valid = false,
        VkImage motion_image = VK_NULL_HANDLE, bool motion_has_previous_depth = false);
    void reset_reservoir_history()
    {
        reservoir_history_valid_ = false;
        reservoir_history_index_ = 0;
        world_cache_clear_pending_ = true;
    }
    void commit_reservoir_history()
    {
        reservoir_history_index_ = reservoir_output_index_;
        reservoir_history_valid_ = true;
        world_cache_clear_pending_ = false;
    }
    void destroy();
    const BufferResource& surfaces(uint32_t index) const { return surfaces_.at(index); }
    const BufferResource& sun_hits(uint32_t index) const { return sun_hits_.at(index); }
    const BufferResource& sun_rays(uint32_t index) const { return sun_rays_.at(index); }
    const BufferResource& radiance(uint32_t index) const { return radiance_.at(index); }
    VkExtent2D extent() const { return extent_; }
    VkImageView output_view(uint32_t index) const { return index < output_views_.size() ? output_views_[index] : VK_NULL_HANDLE; }
    uint32_t projection_capacity() const { return projection_tables_.empty() ? 0 : projection_tables_[0]->capacity(); }
    bool reference_path_trace_enabled() const { return path_trace_enabled_; }
    bool reference_path_trace_history_reused() const { return path_trace_history_reused_; }
    void mark_output_recorded(uint32_t index) { output_layouts_.at(index) = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }

private:
    VkExtent2D extent_{};
    bool local_shadows_{};
    xr_vector<VkImage> depth_images_, normal_images_;
    xr_vector<VkImage> albedo_images_, surface_images_;
    xr_vector<VkImageView> normal_views_, albedo_views_, surface_views_;
    xr_vector<BufferResource> rays_, hits_, surfaces_, primary_surfaces_, sun_rays_, sun_hits_;
    xr_vector<BufferResource> local_rays_, local_hits_, radiance_, restir_radiance_;
    xr_vector<BufferResource> reservoir_history_;
    xr_vector<BufferResource> path_trace_history_;
    BufferResource world_radiance_cache_;
    StorageComputePass world_cache_update_pass_;
    xr_vector<BufferResource> local_lights_;
    RayGeneratePass generator_;
    RayQueryBatch queries_;
    IndirectSunPass sun_generator_;
    RayQueryBatch sun_queries_;
    StorageComputePass local_generator_;
    RayQueryBatch local_queries_;
    StorageComputePass radiance_pass_;
    ReferencePathTracePass reference_path_trace_pass_;
    StorageComputePass restir_spatial_pass_;
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier_{};
    PFN_vkCmdFillBuffer cmd_fill_buffer_{};
    bool world_cache_clear_pending_{true};
    uint32_t world_cache_signature_{};
    bool path_trace_enabled_{};
    bool path_trace_history_reused_{};
    uint32_t reservoir_history_index_{};
    uint32_t reservoir_output_index_{};
    uint32_t alpha_texture_capacity_{};
    bool reservoir_history_valid_{};
    xr_vector<std::unique_ptr<BindlessTextureTable>> projection_tables_;
    RayVisibilityPass image_resolve_;
    xr_vector<VkImageLayout> output_layouts_;
    VkDevice output_device_{};
    PFN_vkDestroyImageView destroy_output_view_{};
    FrameImagePool* output_pool_{};
    FrameImageSharing sharing_{};
    xr_vector<VkImage> owned_images_;
    xr_vector<VkImageView> output_views_;
    void release_output();
};
}
