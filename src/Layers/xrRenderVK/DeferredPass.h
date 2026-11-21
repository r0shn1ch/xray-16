#pragma once

#include "FrameContext.h"
#include "LevelModels.h"
#include "ScenePass.h"
#include "ForwardLighting.h"
#include "GeometryCommandCache.h"
#include "ShaderPairCache.h"
#include "ShaderReflection.h"
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <array>
#include <cmath>
#include <vector>

namespace xray::render::vulkan
{
struct TextureMatrixSet;
struct TreeWindParameters { float wave[4]{}, direction[2]{}; };
struct DeferredLight
{
    float direction_ambient[4];
    float color[4];
    float grade[4]{1.f, 1.f, 1.f, 0.f}; // gamma, brightness, contrast, hemisphere intensity
};

struct WeatherLighting
{
    float ray_base[4]{}, ray_dx[4]{}, ray_dy[4]{};
    DeferredLight light{};
    float sky_color[4]{};
    float clouds_color[4]{};
};
static_assert(sizeof(WeatherLighting) == 128);

struct DeferredEnvironment
{
    float sun_direction[3]{0.f, -1.f, 0.f};
    float sun_color[3]{1.f, 1.f, 1.f};
    float ambient_color[3]{};
    float hemi_color[3]{};
    float fog_color[3]{};
    float fog_near{};
    float fog_far{500.f};
};

// Per-draw surface contract. The fragment push constants occupy bytes 80..111;
// vertex normal rows use the same offsets in the independent vertex stage.
struct SurfaceParameters
{
    // external_normal packs the surface map flags below.
    float bump_mode{}, detail_normal{}, external_normal{}, material{1.f};
    float material_weight{}, bump_height{.05f}, emissive{}, terrain_bump{};
};

inline constexpr uint32_t SurfaceMapExternalNormal = 1u << 0;
inline constexpr uint32_t SurfaceMapPackedPbr = 1u << 1;
inline constexpr uint32_t SurfaceMapMetallic = 1u << 2;
inline constexpr uint32_t SurfaceMapRoughness = 1u << 3;
inline constexpr uint32_t SurfaceMapAo = 1u << 4;
inline constexpr uint32_t SurfaceMapParallax = 1u << 5;
inline constexpr uint32_t SurfaceMapSteepParallax = 1u << 6;
inline constexpr uint32_t SurfaceMapPbrMask = SurfaceMapPackedPbr | SurfaceMapMetallic |
    SurfaceMapRoughness | SurfaceMapAo;
inline constexpr uint32_t SurfaceMaterialBaseTextureCount = 7;
inline constexpr uint32_t SurfaceMaterialTextureCount = 41;
inline constexpr uint32_t SurfaceMaterialExtraTextureCount =
    SurfaceMaterialTextureCount - SurfaceMaterialBaseTextureCount;

enum class GeometryPhase
{
    OpaqueAndAlphaTest,
    SecondaryOpaque,
    Transparent,
    Hud,
    Distortion
};

DeferredLight make_environment_deferred_light(const DeferredEnvironment &environment);

// Framebuffer attachment order: albedo, normal, depth. Color images must
// support color attachment and sampling; depth must support attachment and
// sampling because the render pass preserves it in a read-only layout.
bool create_gbuffer_render_pass(VkDevice device, VkFormat albedo_format, VkFormat normal_format,
                                VkFormat surface_format, VkFormat depth_format, const FrameDispatch &vk,
                                VkRenderPass &result, std::string &error);

// Attachment order: multisampled albedo, normal, surface, depth followed by
// single-sample resolved albedo, normal, surface and depth. The caller checks
// image-format sample counts and depth/stencil resolve modes on the device.
bool create_msaa_gbuffer_render_pass(VkDevice device, VkFormat albedo_format, VkFormat normal_format,
    VkFormat surface_format, VkFormat depth_format, VkSampleCountFlagBits samples,
    VkResolveModeFlagBits depth_resolve, VkResolveModeFlagBits stencil_resolve,
    PFN_vkCreateRenderPass2 create_render_pass, VkRenderPass& result, std::string& error);

// Geometry pass expects two color attachments (RGBA albedo, encoded normal)
// and depth. Transparent geometry blends into the lit swapchain color, using
// the opaque scene depth; HUD follows with a fresh depth buffer.
// Attachments, render-pass transitions and per-frame synchronization belong to
// the caller.
class DeferredPass
{
  public:
    const float *model_hemi_negative() const
    {
        return model_hemi_negative_;
    }
    ~DeferredPass()
    {
        destroy();
    }
    DeferredPass(const DeferredPass &) = delete;
    DeferredPass &operator=(const DeferredPass &) = delete;
    DeferredPass() = default;
    GeometryCommandCache::Batch geometry_batch() const
    {
        return GeometryCommandCache::Batch(command_cache_);
    }
    void invalidate_geometry_cache() const { command_cache_.invalidate(); }

    bool initialize(VkDevice device, VkRenderPass geometry_pass, VkRenderPass light_pass,
                    VkShaderModule geometry_vertex, VkShaderModule geometry_fragment,
                    VkShaderModule alpha_test_fragment, VkShaderModule transparent_fragment,
                    VkShaderModule light_vertex, VkShaderModule light_fragment, VkShaderModule weather_fragment,
                    const ScenePassDispatch &dispatch, std::string &error, VkRenderPass shadow_pass = VK_NULL_HANDLE,
                    VkShaderModule shadow_vertex = VK_NULL_HANDLE, VkShaderModule shadow_opaque = VK_NULL_HANDLE,
                    VkShaderModule shadow_cutout = VK_NULL_HANDLE, VkRenderPass local_shadow_pass = VK_NULL_HANDLE,
                    VkShaderModule local_light_fragment = VK_NULL_HANDLE,
                    VkShaderModule water_fragment = VK_NULL_HANDLE, VkShaderModule shadow_skin = VK_NULL_HANDLE,
                    VkShaderModule multisampled_light_fragment = VK_NULL_HANDLE,
                    VkShaderModule multisampled_weather_fragment = VK_NULL_HANDLE,
                    VkShaderModule multisampled_local_light_fragment = VK_NULL_HANDLE);
    bool material(VkImageView albedo, VkSampler sampler, VkDescriptorSet &set, std::string &error);
    bool lightmapped_material(VkImageView albedo, VkImageView lightmap, VkSampler sampler, VkDescriptorSet &set,
                              std::string &error, VkSampler lightmap_sampler = VK_NULL_HANDLE);
    bool terrain_material(const std::array<VkImageView, SurfaceMaterialBaseTextureCount> &views, VkSampler sampler,
                          VkDescriptorSet &set,
                          std::string &error, VkSampler lightmap_sampler = VK_NULL_HANDLE);
    void update_surface_textures(VkDescriptorSet set,
        const std::array<VkImageView, SurfaceMaterialExtraTextureCount> &views, VkSampler sampler);
    void set_material_image_shapes(VkDescriptorSet set,
        const std::array<ShaderImageShape, SurfaceMaterialTextureCount>& shapes);
    void set_surface_parameters(VkDescriptorSet set, const SurfaceParameters &parameters)
    {
        surface_parameters_[set] = parameters;
    }
    struct GamePipelineStages
    {
        VkShaderModule hull{}, domain{}, geometry{};
        std::string hull_name, domain_name, geometry_name;
        uint64_t specialization_hash{}, shader_fingerprint{}, shader_variant_hash{};
        const VkSpecializationInfo *vertex_specialization{}, *hull_specialization{},
            *domain_specialization{}, *geometry_specialization{}, *fragment_specialization{};
        std::array<ShaderImageShape, SurfaceMaterialTextureCount> material_image_shapes{};
        std::array<uint32_t, SurfaceMaterialTextureCount> material_descriptor_counts = [] {
            std::array<uint32_t, SurfaceMaterialTextureCount> counts{};
            counts.fill(1);
            return counts;
        }();
        std::array<ShaderImageShape, 6> weather_image_shapes{};
        std::array<uint32_t, 6> weather_descriptor_counts = [] {
            std::array<uint32_t, 6> counts{};
            counts.fill(1);
            return counts;
        }();
    };
    // A named SVS/SPS pair from the game's precompiled Vulkan shader set.
    // Unknown names and incompatible pass state fail before a draw is recorded.
    bool create_game_pipeline(const std::string &vertex_name, const std::string &fragment_name, VkShaderModule vertex,
                              VkShaderModule fragment, SurfaceMode mode, bool hud, std::string &error,
                              bool skinned = false);
    bool create_game_pipeline(const std::string &vertex_name, const std::string &fragment_name, VkShaderModule vertex,
                              VkShaderModule fragment, SurfaceMode mode, bool hud, std::string &error,
                              bool skinned, const GamePipelineStages &stages);
    bool has_game_pipeline(const std::string &vertex_name, const std::string &fragment_name,
        uint64_t variant_hash = 0) const;
    using GamePipelineRequest = std::function<bool(const std::string &, const std::string &, SurfaceMode, bool, bool,
        const std::vector<std::string> &, const VkSpecializationInfo *, const VkSpecializationInfo *,
        const VkSpecializationInfo *, const VkSpecializationInfo *, const VkSpecializationInfo *,
        uint64_t &, std::string &)>;
    using LegacyGamePipelineRequest =
        std::function<bool(const std::string &, const std::string &, SurfaceMode, bool, bool, std::string &)>;
    void set_game_pipeline_request(GamePipelineRequest request)
    {
        game_pipeline_request_ = std::move(request);
    }
    void set_game_pipeline_request(LegacyGamePipelineRequest request)
    {
        game_pipeline_request_ = [request = std::move(request)](const std::string &vertex,
            const std::string &fragment, SurfaceMode mode, bool hud, bool skinned,
            const std::vector<std::string> &defines, const VkSpecializationInfo *vertex_specialization,
            const VkSpecializationInfo *hull_specialization, const VkSpecializationInfo *domain_specialization,
            const VkSpecializationInfo *geometry_specialization, const VkSpecializationInfo *pixel_specialization,
            uint64_t &variant_hash, std::string &error) {
            variant_hash = 0;
            if ((vertex_specialization && vertex_specialization->mapEntryCount) ||
                (hull_specialization && hull_specialization->mapEntryCount) ||
                (domain_specialization && domain_specialization->mapEntryCount) ||
                (geometry_specialization && geometry_specialization->mapEntryCount) ||
                (pixel_specialization && pixel_specialization->mapEntryCount))
            {
                error = "specialized game shaders require a specialization-aware pipeline loader";
                return false;
            }
            if (!defines.empty())
            {
                error = "generated shader defines require a define-aware game pipeline loader";
                return false;
            }
            return request(vertex, fragment, mode, hud, skinned, error);
        };
    }
    bool request_game_pipeline(const std::string &vertex_name, const std::string &fragment_name, SurfaceMode mode,
                               bool hud, bool skinned, std::string &error,
                               const std::vector<std::string> &defines = {}, uint64_t *variant_hash = nullptr,
                               const VkSpecializationInfo *vertex_specialization = nullptr,
                               const VkSpecializationInfo *hull_specialization = nullptr,
                               const VkSpecializationInfo *domain_specialization = nullptr,
                               const VkSpecializationInfo *geometry_specialization = nullptr,
                               const VkSpecializationInfo *pixel_specialization = nullptr)
    {
        uint64_t local_variant_hash{};
        uint64_t &resolved_variant_hash = variant_hash ? *variant_hash : local_variant_hash;
        resolved_variant_hash = 0;
        if (game_pipeline_request_)
            return game_pipeline_request_(vertex_name, fragment_name, mode, hud, skinned,
                defines, vertex_specialization, hull_specialization, domain_specialization,
                geometry_specialization, pixel_specialization, resolved_variant_hash, error);
        if ((vertex_specialization && vertex_specialization->mapEntryCount) ||
            (hull_specialization && hull_specialization->mapEntryCount) ||
            (domain_specialization && domain_specialization->mapEntryCount) ||
            (geometry_specialization && geometry_specialization->mapEntryCount) ||
            (pixel_specialization && pixel_specialization->mapEntryCount))
        {
            error = "specialized game shaders require a configured game shader pipeline loader";
            return false;
        }
        if (!defines.empty())
        {
            error = "generated shader defines require a configured game shader pipeline loader";
            return false;
        }
        return require_game_pipeline(vertex_name, fragment_name, mode, hud, skinned,
            error, resolved_variant_hash);
    }
    bool require_game_pipeline(const std::string &vertex_name, const std::string &fragment_name, SurfaceMode mode,
                               bool hud, bool skinned, std::string &error, uint64_t variant_hash = 0) const
    {
        return require_game_pipeline(vertex_name.c_str(), fragment_name.c_str(), mode, hud, skinned, error,
            variant_hash);
    }
    bool require_game_pipeline(const char *vertex_name, const char *fragment_name, SurfaceMode mode, bool hud,
                               bool skinned, std::string &error, uint64_t variant_hash = 0) const;
    void begin_game_pipeline_reload();
    void commit_game_pipeline_reload();
    void abort_game_pipeline_reload();
    bool pose_descriptor(VkBuffer pose, VkDeviceSize bytes, VkDescriptorSet &result, std::string &error);
    void release_pose_descriptor(VkDescriptorSet &set);
    bool record_skinned(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices, uint32_t index_count,
                        const float (&mvp)[16], VkDescriptorSet material_set, VkDescriptorSet pose_set,
                        SurfaceMode mode, bool hud, uint32_t first_index, const char *vertex_name,
                        const char *fragment_name, float alpha_ref = 128.f / 255.f, const float *normal_rows = nullptr,
                        float detail_scale = 1.f, const float *environment_camera = nullptr,
                        const TextureMatrixSet *matrices = nullptr, uint64_t variant_hash = 0) const;
    void update_material(VkDescriptorSet set, VkImageView view, VkSampler sampler);
    void update_lightmapped_material(VkDescriptorSet set, VkImageView albedo, VkImageView lightmap, VkSampler sampler,
                                     VkSampler lightmap_sampler = VK_NULL_HANDLE);
    void update_terrain_material(VkDescriptorSet set,
        const std::array<VkImageView, SurfaceMaterialBaseTextureCount> &views, VkSampler sampler,
        VkSampler lightmap_sampler = VK_NULL_HANDLE);
    bool gbuffer(VkImageView albedo, VkImageView normal, VkImageView depth, VkImageView surface, VkSampler sampler,
                 VkDescriptorSet &set, std::string &error);
    bool gbuffer_multisampled(VkImageView albedo, VkImageView normal, VkImageView depth, VkImageView surface,
                              VkImageView resolved_depth, VkImageView resolved_normal, VkSampler sampler,
                              VkDescriptorSet &set, std::string &error);
    void pair_gbuffer_sets(VkDescriptorSet resolved_set, VkDescriptorSet multisampled_set);
    void bind_sun_shadow(VkDescriptorSet set, VkImageView view, VkSampler sampler, VkBuffer uniform);
    void bind_rain_shadow(VkDescriptorSet set, VkImageView view, VkSampler sampler);
    void bind_ray_visibility(VkDescriptorSet set, VkImageView view, VkSampler sampler);
    void bind_indirect_light(VkDescriptorSet set, VkImageView view, VkSampler sampler);
    void set_frame_ready(std::function<void(uint32_t)> callback)
    {
        frame_ready_ = std::move(callback);
    }
    void notify_frame_ready(uint32_t slot) const
    {
        if (frame_ready_)
            frame_ready_(slot);
    }
    bool local_light_set(VkImageView shadow_array, VkSampler sampler, VkBuffer uniform, VkDeviceSize offset,
                         VkDeviceSize range, VkDescriptorSet &set, std::string &error,
                         VkImageView projection = VK_NULL_HANDLE);
    void update_light_projection(VkDescriptorSet set, VkImageView projection, VkSampler sampler);
    bool water_set(uint32_t image_index, VkImageView refraction, VkImageView distortion_field, VkImageView depth,
                   VkSampler sampler, VkSampler depth_sampler, VkBuffer scene_uniform, std::string &error);
    void release_water_sets();
    bool weather_set(VkImageView sky_a, VkImageView sky_b, VkImageView clouds_a, VkImageView clouds_b,
                     VkImageView environment_a, VkImageView environment_b, VkSampler sampler, VkDescriptorSet &set,
                     const std::array<ShaderImageShape, 6>& image_shapes, std::string &error);
    void release_gbuffer(VkDescriptorSet &set);
    bool forward_set(uint32_t image, VkBuffer buffer, std::string &error);
    void set_forward_weather(VkDescriptorSet set)
    {
        forward_weather_ = set;
    }
    void set_model_hemi_negative(const float *faces)
    {
        for (unsigned i = 0; i < 3; ++i)
            model_hemi_negative_[i] = faces ? faces[i] : 0.f;
    }
    void release_forward_sets();
    VkDescriptorSetLayout gbuffer_layout() const
    {
        return gbuffer_layout_;
    }
    VkDescriptorSetLayout material_layout() const { return material_layout_; }
    VkDescriptorSetLayout water_scene_layout() const { return water_set_layout_; }
    VkDescriptorSet water_scene_set(uint32_t image) const
    { return image < water_sets_.size() ? water_sets_[image] : VK_NULL_HANDLE; }
    SurfaceParameters material_surface(VkDescriptorSet material) const
    {
        const auto found = surface_parameters_.find(material);
        return found == surface_parameters_.end() ? SurfaceParameters{} : found->second;
    }
    VkDescriptorSetLayout pose_layout() const { return pose_layout_; }
    void account_light_grid_draw() const
    {
        ++draw_calls_;
        ++triangles_;
    }
    void rebind_compatible_render_passes(VkRenderPass geometry_pass, VkRenderPass light_pass,
                                         VkRenderPass shadow_pass = VK_NULL_HANDLE,
                                         VkRenderPass local_shadow_pass = VK_NULL_HANDLE,
                                         VkRenderPass overlay_pass = VK_NULL_HANDLE,
                                         VkRenderPass rain_shadow_pass = VK_NULL_HANDLE)
    {
        geometry_pass_ = geometry_pass;
        light_pass_ = light_pass;
        overlay_pass_ = overlay_pass;
        if (shadow_pass)
            shadow_pass_ = shadow_pass;
        if (local_shadow_pass)
            local_shadow_pass_ = local_shadow_pass;
        if (rain_shadow_pass)
            rain_shadow_pass_ = rain_shadow_pass;
    }
    void set_rain_shadow_pass(VkRenderPass pass) { rain_shadow_pass_ = pass; }
    void set_local_shadow_graph_passes(VkRenderPass clear_pass, VkRenderPass load_pass, VkRenderPass legacy_load_pass)
    {
        local_shadow_graph_pass_ = clear_pass;
        local_shadow_graph_load_pass_ = load_pass;
        local_shadow_load_pass_ = legacy_load_pass;
    }
    // Caller guarantees compatibility with the depth-only shadow render pass
    // (same depth format and sample count) and clears this before destroying it.
    void set_vsm_shadow_pass(VkRenderPass pass, VkRenderPass dynamic_pass = VK_NULL_HANDLE)
    { vsm_shadow_pass_ = pass; vsm_dynamic_shadow_pass_ = dynamic_pass; }
    void set_tree_wind(const TreeWindParameters& wind) { tree_wind_ = wind; }
    float tree_wind_amplitude() const { return std::hypot(tree_wind_.direction[0], tree_wind_.direction[1]); }
    bool record_geometry(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices, uint32_t index_count,
                         const float (&mvp)[16], VkDescriptorSet material_set, SurfaceMode mode = SurfaceMode::Opaque,
                         uint32_t first_index = 0, const char *vertex_name = nullptr,
                         const char *fragment_name = nullptr, float alpha_ref = 128.f / 255.f,
                         const float *normal_rows = nullptr, float terrain_scale = 1.f,
                         const float *environment_camera = nullptr, const float *environment_factor = nullptr,
                         int32_t vertex_base = 0, const TextureMatrixSet *matrices = nullptr,
                         VkBuffer instances = VK_NULL_HANDLE, uint32_t instance_count = 1,
                         VkBuffer indirect_commands = VK_NULL_HANDLE, VkDeviceSize indirect_offset = 0,
                         uint32_t indirect_draw_count = 0, uint64_t variant_hash = 0) const;
    bool record_sun_shadow(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                           uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material, bool alpha_test,
                           uint32_t first_index = 0, float alpha_ref = 128.f / 255.f, int32_t vertex_base = 0,
                           VkDescriptorSet pose = VK_NULL_HANDLE, bool tree_wave = false,
                           const TextureMatrixSet* matrices = nullptr,
                           VkBuffer indirect_commands = VK_NULL_HANDLE, VkDeviceSize indirect_offset = 0,
                           uint32_t indirect_draw_count = 0) const;
    bool supports_multi_draw_indirect() const { return vk_.multi_draw_indirect; }
    bool is_sun_shadow_pass(VkRenderPass render_pass) const { return render_pass && render_pass == shadow_pass_; }
    VkPipelineCache pipeline_cache() const { return vk_.pipeline_cache; }
    bool create_visibility_gbuffer_pipeline(VkRenderPass render_pass, VkPipelineLayout layout,
        VkShaderModule vertex, VkShaderModule fragment, VkPipeline& pipeline, std::string& error) const;
    void draw_visibility_gbuffer(const FrameRecordingContext& frame, VkPipeline pipeline,
        VkPipelineLayout layout, VkDescriptorSet sources) const;
    bool record_hud(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices, uint32_t index_count,
                    const float (&mvp)[16], VkDescriptorSet material_set, uint32_t first_index = 0,
                    const char *vertex_name = nullptr, const char *fragment_name = nullptr, float alpha_ref = 0.f,
                    const float *normal_rows = nullptr, int32_t vertex_base = 0,
                    const TextureMatrixSet *matrices = nullptr, uint64_t variant_hash = 0) const;
    bool record_transparent(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                            uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material_set,
                            uint32_t first_index = 0, const char *vertex_name = nullptr,
                            const char *fragment_name = nullptr, const char **failure = nullptr, float alpha_ref = 0.f,
                            const float *normal_rows = nullptr, float detail_scale = 1.f, int32_t vertex_base = 0,
                            const TextureMatrixSet *matrices = nullptr, uint64_t variant_hash = 0) const;
    bool record_lighting(const FrameRecordingContext &frame, VkDescriptorSet gbuffer_set, const DeferredLight &light,
                         VkDescriptorSet weather_set = VK_NULL_HANDLE, const WeatherLighting *weather = nullptr) const;
    bool record_local_light(const FrameRecordingContext &frame, VkDescriptorSet gbuffer_set, VkDescriptorSet local_set,
                            const VkRect2D &scissor) const;
    bool initialize_distortion(VkShaderModule vertex, VkShaderModule fragment, VkShaderModule screen_vertex,
                               VkShaderModule composite, std::string &error);
    bool composite_distortion(const FrameRecordingContext &frame) const;
    bool record_distortion(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                           uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material, bool field = false) const;
    bool record_water(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices, uint32_t index_count,
                      const float (&mvp)[16], VkDescriptorSet material, uint32_t first_index, float time,
                      float opacity = .72f, int32_t vertex_base = 0) const;
    void destroy();
    void reset_draw_statistics()
    {
        draw_calls_ = triangles_ = 0;
    }
    uint32_t draw_calls() const
    {
        return draw_calls_;
    }
    uint32_t triangles() const
    {
        return triangles_;
    }

  private:
    std::function<void(uint32_t)> frame_ready_;
    mutable GeometryCommandCache command_cache_;
    void push_texture_matrices(const FrameRecordingContext &frame, VkPipelineLayout layout,
                               const TextureMatrixSet *matrices, const float *normal_rows = nullptr,
                               bool tree_wave = false) const;
    TreeWindParameters tree_wind_{};
    struct GamePipeline
    {
        VkPipeline handle{};
        VkPipelineLayout layout{};
        VkDescriptorSetLayout material_layout{};
        SurfaceMode mode{};
        bool hud{}, skinned{};
        std::string hull_name, domain_name, geometry_name;
        uint64_t specialization_hash{}, shader_fingerprint{}, shader_variant_hash{};
        std::array<ShaderImageShape, SurfaceMaterialTextureCount> material_image_shapes{};
        std::array<uint32_t, SurfaceMaterialTextureCount> material_descriptor_counts{};
        std::array<ShaderImageShape, 6> weather_image_shapes{};
        std::array<uint32_t, 6> weather_descriptor_counts{};
        VkDescriptorSetLayout weather_layout{};
    };
    mutable ShaderPairCache<GamePipeline> pipeline_lookup_cache_;
    const GamePipeline *game_pipeline(const char *vertex_name, const char *fragment_name,
        uint64_t variant_hash = 0) const;
    bool allocate(VkDescriptorSetLayout layout, VkImageView first, VkImageView second, VkSampler sampler,
                  VkDescriptorSet &set, std::string &error);
    bool material_image_shapes_match(VkDescriptorSet set,
        const std::array<ShaderImageShape, SurfaceMaterialTextureCount>& expected,
        const std::array<uint32_t, SurfaceMaterialTextureCount>& counts) const;
    VkDescriptorSet material_descriptor_set(VkDescriptorSet set, const GamePipeline* pipeline) const;
    bool weather_image_shapes_match(VkDescriptorSet set,
        const std::array<ShaderImageShape, 6>& expected,
        const std::array<uint32_t, 6>& counts) const;
    VkDescriptorSet weather_descriptor_set(VkDescriptorSet set, const GamePipeline* pipeline) const;
    bool write_material_descriptor_set(VkDescriptorSet target,
        const std::array<uint32_t, SurfaceMaterialTextureCount>& counts,
        const std::array<VkDescriptorImageInfo, SurfaceMaterialTextureCount>& images) const;
    bool write_weather_descriptor_set(VkDescriptorSet target, const std::array<uint32_t, 6>& counts,
        const std::array<VkDescriptorImageInfo, 6>& images) const;
    void refresh_material_descriptor_sets(VkDescriptorSet set);
    void release_material_descriptor_sets(VkDescriptorSet set);
    void release_all_material_descriptor_sets();
    void destroy_game_pipeline(GamePipeline& pipeline);
    void push_surface(const FrameRecordingContext &frame, VkPipelineLayout layout, VkDescriptorSet material,
                      uint32_t offset = 80) const;
    VkDevice device_{};
    VkRenderPass geometry_pass_{}, light_pass_{}, shadow_pass_{}, local_shadow_pass_{},
        local_shadow_load_pass_{}, local_shadow_graph_pass_{}, local_shadow_graph_load_pass_{}, vsm_shadow_pass_{},
        vsm_dynamic_shadow_pass_{}, rain_shadow_pass_{}, overlay_pass_{};
    bool lighting_compatible(const FrameRecordingContext &frame) const
    {
        return frame.render_pass == light_pass_ || (overlay_pass_ && frame.render_pass == overlay_pass_);
    }
    VkPipeline shadow_skin_opaque_{}, shadow_skin_cutout_{};
    VkPipeline geometry_{}, alpha_test_{}, transparent_{}, hud_{}, lighting_{}, weather_pipeline_{}, shadow_opaque_{},
        shadow_cutout_{}, local_light_pipeline_{}, water_pipeline_{}, distortion_pipeline_{}, distortion_composite_{},
        multisampled_lighting_{}, multisampled_weather_pipeline_{}, multisampled_local_light_pipeline_{};
    std::unordered_set<VkDescriptorSet> multisampled_gbuffer_sets_;
    std::unordered_map<VkDescriptorSet, VkDescriptorSet> gbuffer_msaa_pairs_;
    std::unordered_map<std::string, GamePipeline> game_pipelines_;
    std::unordered_map<VkDescriptorSet, SurfaceParameters> surface_parameters_;
    std::unordered_map<std::string, GamePipeline> pending_game_pipelines_;
    bool reloading_game_pipelines_{};
    std::unordered_map<VkDescriptorSet,
        std::array<ShaderImageShape, SurfaceMaterialTextureCount>> material_image_shapes_;
    std::unordered_map<VkDescriptorSet,
        std::array<VkDescriptorImageInfo, SurfaceMaterialTextureCount>> material_image_descriptors_;
    struct MaterialDescriptorVariant
    {
        VkDescriptorSet set{};
        std::array<uint32_t, SurfaceMaterialTextureCount> counts{};
    };
    mutable std::unordered_map<VkDescriptorSet,
        std::unordered_map<VkDescriptorSetLayout, MaterialDescriptorVariant>> material_descriptor_variants_;
    std::unordered_map<VkDescriptorSet, std::array<VkDescriptorImageInfo, 6>> weather_image_descriptors_;
    std::unordered_map<VkDescriptorSet, std::array<ShaderImageShape, 6>> weather_image_shapes_;
    mutable std::unordered_map<VkDescriptorSet,
        std::unordered_map<VkDescriptorSetLayout, VkDescriptorSet>> weather_descriptor_variants_;
    VkPipelineLayout geometry_layout_{}, skinned_layout_{}, light_layout_{}, weather_layout_{}, local_light_layout_{},
        water_layout_{};
    VkDescriptorSetLayout material_layout_{}, pose_layout_{}, gbuffer_layout_{}, weather_set_layout_{},
        local_set_layout_{}, water_set_layout_{}, forward_layout_{};
    std::vector<VkDescriptorSet> water_sets_;
    std::vector<VkDescriptorSet> forward_sets_;
    VkDescriptorSet forward_weather_{};
    float model_hemi_negative_[3]{};
    VkDescriptorPool pool_{};
    ScenePassDispatch vk_{};
    VkShaderStageFlags game_stage_flags_{VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT};
    GamePipelineRequest game_pipeline_request_;
    mutable uint32_t draw_calls_{}, triangles_{};
};
} // namespace xray::render::vulkan
