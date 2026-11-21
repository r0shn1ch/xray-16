#pragma once

#include "BufferUpload.h"
#include "LevelRayScene.h"
#include "DeferredPass.h"
#include "EngineLevelModels.h"
#include "GameTextureFactory.h"
#include "LevelPassGraph.h"
#include "LevelHierarchyBounds.h"
#include "ClusterLodBuilder.h"
#include "ClusterVisibility.h"
#include "Layers/xrRenderGraph/VisibilityId.h"
#include "DetailGeneratePass.h"
#include "ResourceIssueCounter.h"
#include "xrCore/_matrix.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <unordered_map>

class IRenderVisual;

namespace xray::render::vulkan
{
class VulkanVisual;
class ObjectVelocityPass;
// GPU ownership of the supported static level subset. A failed load leaves
// the previous level intact; no partially uploaded scene is exposed.
class GpuLevel
{
  public:
    using RayGeometryBatch = xray::render::vulkan::RayGeometryBatch;
    struct VisibilityDrawRange
    {
        uint32_t geometry_batch;
        uint32_t first_draw;
        uint32_t draw_count;
    };
    const std::vector<VisibilityDrawRange>& visibility_draw_ranges() const { return visibility_draw_ranges_; }
    struct VisibilityTexture { std::string name; bool lightmap; };
    const std::vector<VisibilityTexture>& visibility_textures() const { return visibility_textures_; }
    VkDescriptorBufferInfo visibility_draw_entries() const
    { return {visibility_draw_entries_.handle(), 0, visibility_draw_entries_.size()}; }
    VkDescriptorBufferInfo visibility_camera_commands(uint32_t frame) const
    { return cluster_visibility_.camera_commands(frame); }
    bool visibility_geometry(uint32_t batch, VkDescriptorBufferInfo& vertices, VkDescriptorBufferInfo& indices) const;
    void mark_visibility_draws(uint32_t visual, float lod, uint32_t* words, size_t word_count) const;
    GpuLevel() = default;
    ~GpuLevel();
    GpuLevel(const GpuLevel &) = delete;
    GpuLevel &operator=(const GpuLevel &) = delete;

    bool load(IReader &level, VkDevice device, VkQueue queue, VkCommandPool pool,
              const VkPhysicalDeviceMemoryProperties &memory, const BufferUploadDispatch &upload,
              GameTextureFactory &textures, DeferredPass &pass, std::string &error,
              PFN_vkGetBufferDeviceAddress get_buffer_address = nullptr,
              PFN_vkGetDeviceProcAddr get_device_proc = nullptr,
              VkDeviceSize max_storage_buffer_range = 0);
    const std::vector<RayGeometryBatch>& ray_geometry_batches() const { return ray_batches_; }
    // Entries follow the same batch/geometry ordering as the RT metadata.
    const std::vector<std::vector<TextureMatrixSet>>& ray_texture_matrices() const { return ray_texture_matrices_; }
    const std::vector<std::string>& ray_texture_names() const { return ray_texture_names_; }
    const std::vector<std::string>& ray_diffuse_texture_names() const { return ray_diffuse_texture_names_; }
    const std::vector<uint8_t>& ray_diffuse_texture_linear() const { return ray_diffuse_texture_linear_; }
    // Call after load, while the graphics queue is idle with respect to this level.
    // A failed optional RT build leaves the raster level intact.
    bool build_ray_scene(VkQueue queue, const AccelerationStructureDispatch& dispatch,
        const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error);
    VkDeviceAddress ray_scene_address() const { return ray_scene_.address(); }
    const LevelRayScene& ray_scene() const { return ray_scene_; }
    bool record(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16]) const;
    bool record_cluster_visibility(const FrameRecordingContext& frame, const float (&camera_mvp)[16],
        const float (&sun_mvp)[16], uint32_t shadow_height, std::string& error,
        const float* previous_mvp = nullptr, VkImage previous_hiz_image = VK_NULL_HANDLE,
        VkImageView previous_hiz = VK_NULL_HANDLE, VkSampler hiz_sampler = VK_NULL_HANDLE,
        const FrameImageSharing* previous_hiz_sharing = nullptr, uint32_t previous_hiz_levels = 0,
        bool hiz_valid = false) const;
    bool record_cluster_shadow_visibility(const FrameRecordingContext& frame,
        const float (&shadow_mvp)[16], std::string& error) const;
    bool has_cluster_visibility() const { return cluster_visibility_.ready(); }
    bool record(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16],
                GeometryPhase phase) const;
    bool record_visual(size_t index, const FrameRecordingContext &frame, const DeferredPass &pass,
                       const float (&mvp)[16], GeometryPhase phase = GeometryPhase::OpaqueAndAlphaTest, float lod = 1.f,
                       bool cull_static = true, bool secondary_only = false, int priority_filter = -1,
                       bool visibility_rasterized = false) const;
    bool record_hud_visual(size_t index, const FrameRecordingContext &frame, const DeferredPass &pass,
                           const float (&mvp)[16], float lod = 1.f) const;
    bool record_object_velocity_visual(size_t index, const FrameRecordingContext& frame,
        const ObjectVelocityPass& pass, const float (&current_mvp)[16],
        const float (&previous_mvp)[16], float lod = 1.f, bool mark_base_geometry = true) const;
    bool supports_dynamic_occluder_mask(size_t visual) const;
    bool record_tree_velocity(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
        const float (&current_mvp)[16], const float (&previous_mvp)[16],
        const TreeWindParameters& current_wind, const TreeWindParameters& previous_wind) const;
    size_t model_count() const
    {
        return meshes_.size();
    }
    bool retire_completed_uploads();
    uint64_t geometry_bytes() const
    {
        const auto mesh_bytes = [](const Mesh &mesh) {
            uint64_t bytes = mesh.vertex_bytes + mesh.index_bytes;
            bytes += mesh.cluster_index_bytes;
            if (mesh.fast)
                bytes += mesh.fast->vertex_bytes + mesh.fast->index_bytes + mesh.fast->cluster_index_bytes;
            return bytes;
        };
        uint64_t bytes = 0;
        for (const auto &mesh : meshes_)
            bytes += mesh_bytes(mesh);
        for (const auto &mesh : detail_meshes_)
            bytes += mesh_bytes(mesh);
        bytes += cluster_visibility_.bytes();
        for (const BufferResource& commands : cluster_draw_commands_)
            bytes += commands.size();
        return bytes;
    }
    size_t visual_count() const
    {
        return visuals_.size();
    }
    bool has_secondary_priority(size_t index) const
    {
        return index < visual_phases_.size() && (visual_phases_[index] & 4u) != 0;
    }
    bool has_transparent_priority(size_t index, uint8_t priority) const
    {
        return index < visual_priorities_.size() && priority < 8 && (visual_priorities_[index] & (1u << priority)) != 0;
    }
    bool has_water() const
    {
        return has_water_;
    }
    uint64_t revision() const
    {
        return revision_;
    }
    IRenderVisual *get_visual(size_t index) const;
    int find_visual_index(const IRenderVisual *visual) const;
    const LevelVisual *visual_node(size_t index) const;
    size_t camera_sector(const Fvector &position, const Fvector &direction, float geometry_distance,
                         size_t geometry_sector) const;
    bool visible_sector_roots(size_t camera_sector, const Fmatrix &view_projection, const Fvector &camera_position,
                              std::vector<uint32_t> &roots) const;
    void all_level_roots(std::vector<uint32_t> &roots) const;
    bool enable_gpu_details(PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits, std::string& error);
    bool record_detail_cull(const FrameRecordingContext& frame, const float (&mvp)[16], std::string& error) const;
    void prepare_details(const Fvector &camera_position, float max_distance = 40.f);
    bool record_details(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16]) const;
    bool prepare_transparent_details(const FrameRecordingContext &frame, const float (&mvp)[16],
                                     std::vector<float> &sort_distances) const;
    bool record_transparent_detail(size_t draw_index, const FrameRecordingContext &frame, const DeferredPass &pass,
                                   const float (&mvp)[16]) const;
    bool record_sun_shadow(const FrameRecordingContext &frame, const DeferredPass &pass,
                           const float (&sun_mvp)[16], bool include_static = true, bool include_trees = true,
                           bool use_cluster_visibility = false) const;
    void destroy();

  private:
    void report_visual_record_failure(size_t visual, int32_t mesh, GeometryPhase phase, bool water, uint32_t indices,
                                      float lod, const char *reason) const;
    struct Mesh
    {
        struct TerrainRayLayer
        {
            std::string diffuse, pbr, metallic, roughness, ao, bump, bump_aux, parallax;
        };
        BufferResource vertices, indices;
        VkBuffer packed_vertices{}, packed_indices{};
        uint32_t geometry_batch{UINT32_MAX};
        bool visibility_rasterizable{};
        uint32_t first_vertex{}, first_index{};
        uint64_t vertex_bytes{}, index_bytes{};
        uint64_t cluster_index_bytes{};
        VkBuffer vertex_buffer() const
        {
            return packed_vertices ? packed_vertices : vertices.handle();
        }
        VkBuffer index_buffer() const
        {
            return packed_indices ? packed_indices : indices.handle();
        }
        uint32_t index_count{};
        uint32_t cluster_first_index{};
        uint32_t cluster_first_draw_command{};
        uint32_t cluster_cull_group_first{}, cluster_cull_group_count{};
        uint32_t cluster_cull_cluster_first{}, cluster_cull_cluster_count{};
        VkBuffer packed_cluster_indices{};
        ClusterLodData cluster_lod;
        struct ClusterWindow
        {
            ClusterLodData data;
            uint32_t first_index{};
            uint32_t index_count{};
            uint32_t first_draw_command{};
            uint32_t cull_group_first{}, cull_group_count{};
            uint32_t cull_cluster_first{}, cull_cluster_count{};
            bool visibility_rasterizable{};
        };
        std::vector<ClusterWindow> cluster_lod_windows;
        size_t cluster_index_count() const
        {
            size_t count = cluster_lod.indices.size();
            for (const ClusterWindow& window : cluster_lod_windows)
                count += window.data.indices.size();
            return count;
        }
        bool has_cluster_lod() const
        {
            if (!cluster_lod.clusters.empty()) return true;
            return std::any_of(cluster_lod_windows.begin(), cluster_lod_windows.end(),
                [](const ClusterWindow& window) { return !window.data.clusters.empty(); });
        }
        std::vector<SlideWindow> windows;
        VkDescriptorSet material{};
        bool owns_material{};
        uint32_t material_id{};
        std::string shader;
        SurfaceMode mode{SurfaceMode::Opaque};
        bool secondary_opaque{};
        uint8_t priority{1};
        int alpha_ref{128};
        bool wallmark_multiply{};
        bool wallmark_blend{};
        bool glow_alpha_add{};
        bool fog{};
        bool lightmapped{};
        std::string lightmap_texture_name;
        bool lod_atlas{};
        bool terrain{};
        bool surface{};
        bool single_detail{};
        bool environment{};
        bool laemb{};
        bool environment_constant{};
        LevelPassGraph pass_graph;
        std::array<WaveForm, 4> constant_waveforms{};
        TextureMatrixSet matrices;
        uint8_t tessellation{};
        std::string bump_texture;
        std::string pbr_texture, metallic_texture, roughness_texture, ao_texture;
        std::string parallax_texture;
        std::string ray_normal_texture;
        std::array<TerrainRayLayer, 4> terrain_ray_layers;
        uint32_t bump_mode{1};
        float bump_height{.05f};
        float terrain_scale{1.f};
        bool water{};
        bool glass{};
        bool tree{};
        bool tree_wave{};
        bool emissive{};
        float tree_wind_extent{};
        bool depth_proxy{};
        std::unique_ptr<Mesh> fast;
    };
    struct DetailBatch
    {
        BufferResource vertices, indices;
        uint32_t index_count{};
        uint32_t instance_count{};
    };
    VkDevice device_{};
    LevelGameProfile profile_{LevelGameProfile::Unknown};
    VkPhysicalDeviceMemoryProperties memory_{};
    VkCommandPool pool_{};
    BufferUploadDispatch upload_{};
    GameTextureFactory *textures_{};
    DeferredPass *pass_{};
    std::vector<PendingBufferUpload> pending_;
    struct GeometryBatch
    {
        BufferResource vertices, indices, cluster_indices;
    };
    std::vector<GeometryBatch> geometry_batches_;
    BufferResource visibility_draw_entries_;
    std::vector<VisibilityDrawRange> visibility_draw_ranges_;
    std::vector<VisibilityTexture> visibility_textures_;
    // CPU-authored indirect arguments use one fenced buffer per frame slot.
    // The command array is also storage-capable for the planned GPU culler.
    mutable std::array<BufferResource, FrameContext::FramesInFlight> cluster_draw_commands_;
    uint32_t cluster_draw_command_count_{};
    ClusterVisibility cluster_visibility_;
    std::vector<RayGeometryBatch> ray_batches_;
    std::vector<std::vector<TextureMatrixSet>> ray_texture_matrices_;
    std::vector<std::string> ray_texture_names_;
    std::vector<std::string> ray_diffuse_texture_names_;
    std::vector<uint8_t> ray_diffuse_texture_linear_;
    LevelRayScene ray_scene_;
    std::vector<Mesh> meshes_;
    struct DetailPlacement
    {
        uint8_t model{};
        float x{}, y{}, z{}, yaw{}, scale{};
    };
    DetailAssets details_;
    std::vector<Mesh> detail_meshes_;
    std::unordered_map<size_t, std::vector<DetailPlacement>> detail_cache_;
    float detail_spacing_{};
    std::vector<DetailPlacement> visible_details_;
    // Each frame slot is reused only after FrameContext has waited on its fence.
    mutable std::array<std::vector<DetailBatch>, FrameContext::FramesInFlight> detail_batches_;
    struct GpuDetailBatch { BufferResource placements, instances, arguments; uint32_t count{}; };
    std::unique_ptr<DetailCullPass> detail_cull_;
    std::unique_ptr<DetailGeneratePass> detail_generate_;
    mutable bool detail_gpu_generation_failed_{};
    bool detail_cache_gpu_only_{};
    struct DetailGenerationFrame { BufferResource cells, triangles, prototypes, placed; };
    mutable std::array<DetailGenerationFrame, FrameContext::FramesInFlight> detail_generation_frames_;
    std::unordered_map<size_t, std::vector<DetailGroundTriangle>> detail_ground_cache_;
    std::vector<DetailGenerateSlot> detail_generation_cells_;
    std::vector<DetailGroundTriangle> detail_generation_triangles_;
    DetailGeneratePass::Parameters detail_generation_params_;
    bool record_generated_details(const FrameRecordingContext& frame, const float (&mvp)[16], std::string& error) const;

    mutable std::array<std::vector<GpuDetailBatch>, FrameContext::FramesInFlight> gpu_detail_batches_;
    mutable std::array<bool, FrameContext::FramesInFlight> gpu_detail_recorded_{};

    mutable std::array<uint64_t, FrameContext::FramesInFlight> detail_signatures_{};
    mutable std::array<std::vector<DetailBatch>, FrameContext::FramesInFlight> transparent_detail_batches_;
    struct TransparentDetailDraw
    {
        uint8_t model;
        uint32_t first, count;
    };
    mutable std::array<std::vector<TransparentDetailDraw>, FrameContext::FramesInFlight> transparent_detail_draws_;
    std::vector<LevelVisual> visuals_;
    std::vector<LevelCullSphere> cull_bounds_;
    std::vector<float> cull_wind_extents_;
    std::vector<uint8_t> visual_phases_;     // opaque bit 1, forward bit 2, secondary opaque bit 4
    std::vector<uint8_t> visual_priorities_; // ordinary transparent priority bits 1..3
    std::vector<uint32_t> roots_;
    std::vector<LevelSector> sectors_;
    std::vector<LevelPortal> portals_;
    std::vector<std::unique_ptr<VulkanVisual>> visual_objects_;
    std::unordered_map<const IRenderVisual *, uint32_t> visual_indices_;
    uint64_t revision_{};
    bool has_water_{};
    mutable ResourceIssueCounter draw_failures_;
};
} // namespace xray::render::vulkan
