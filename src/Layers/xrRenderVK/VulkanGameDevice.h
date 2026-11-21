#pragma once

#include "VisibilityPasses.h"

#include "DeferredFrame.h"
#include "DeferredShaderFactory.h"
#include "GameTextureFactory.h"
#include "GpuModel.h"
#include "LightGrid.h"
#include "HiZPyramid.h"
#include "MotionVectors.h"
#include "RayFrameResources.h"
#include "VsmFrameResources.h"
#include "VsmClipmap.h"
#include "IndirectRayFrames.h"
#include "ObjectMotionHistory.h"
#include "ObjectVelocityPass.h"
#include "TemporalExposure.h"
#include "TemporalResolve.h"
#include "Fsr3Upscaler.h"
#include "LocalShadowTargets.h"
#include "PipelineCache.h"
#include "ScreenCopyPass.h"
#include "VulkanFontRender.h"
#include "VulkanUIRender.h"
#include "VulkanUIShader.h"
#include "VulkanWindowDevice.h"
#include "WaterTargets.h"
#include "GpuParticleRuntime.h"
#include "FrameBufferPool.h"
#include "GpuParticleEmitters.h"
#include "GpuParticleLevelCollision.h"
#include "GpuParticleMaterials.h"
#include "GpuParticleGraphics.h"
#include "SmokeTrailController.h"
#include "SmokeTrailNoise.h"
#include "TrailManager.h"

#include <array>
#include "SkyProbeBakePass.h"
#include "ResourceRetirement.h"
#include <unordered_set>
#include <memory>
#include <unordered_map>
#include <vector>
#include <functional>

namespace xray::render::vulkan
{
class VulkanParticleEffect;
class VulkanParticleGroup;
class VulkanRainRender;
class VulkanThunderboltRender;
struct VsmPageJob;
// Owns the Vulkan gameplay frame resources. Destruction waits for submitted
// work before releasing descriptors, pipelines, images and the device.
class VulkanGameDevice
{
  public:
    bool initialize(SDL_Window *window, VkExtent2D extent, std::string &error);
    void begin_frame();
    void queue_weather(VkDescriptorSet set, const WeatherLighting &lighting, const float (&fog_color)[3],
                       float fog_near, float fog_far);
    void queue_lights(const std::vector<VulkanLightSnapshot> &lights)
    {
        light_snapshots_ = lights;
    }
    VkDescriptorSet glow_texture(const std::string &name);
    void release_level_glows();
    VkImageView light_projection(const std::string &name);
    void queue_rain(VulkanRainRender &rain);
    void queue_thunderbolt(VulkanThunderboltRender &bolt);
    void discard_rain(const VulkanRainRender *rain);
    void discard_thunderbolt(const VulkanThunderboltRender *bolt);
    void request_screenshot()
    {
        screenshot_requested_ = !window_.frame().hdr10_active();
    }
    bool take_screenshot(std::vector<uint8_t> &pixels, VkExtent2D &extent, VkFormat &format);
    void set_postprocess(const PostProcessConstants &params, std::string first, std::string second);
    void detach_color_maps_for_asset_reload();
    bool refresh_color_maps_after_asset_reload(std::string &error);
    bool render(const GpuLevel &level, const float (&mvp)[16], const DeferredLight &light, FrameStatus &status,
                std::string &error, bool render_world = true, bool clear_target = false,
                const DeferredEnvironment *environment = nullptr);
    bool recreate_swapchain(VkExtent2D extent, std::string &error, bool recreate_surface = false);
    bool prepare_for_reset(std::string &error);
    void set_level_visibility(const VisibilityPasses* passes, const BindlessTextureTable* materials);
    bool reload_game_shaders(std::string &error);
    bool wait_idle();
    GpuParticleEmitters* gpu_particle_emitters() const { return gpu_particle_failed_ ? nullptr : gpu_particle_emitters_.get(); }
    void update_smoke_trail(const Fvector& position, const Fvector& direction, float dt, uint32_t frame)
    { if (smoke_compute_) smoke_controller_.update(&position.x, &direction.x, dt, frame); }
    bool update_projectile_trail(uint64_t id, const Fvector& start, const Fvector& end, float width, uint32_t frame, float dt, uint32_t color)
    { return trail_enabled_ && !trail_failed_ && trail_manager_.observe(id, &start.x, &end.x, width, frame, dt, color); }
    void release_trail_texture();
    void set_gpu_particle_catalog(std::shared_ptr<const ParticleCatalog> catalog) { gpu_particle_catalog_ = std::move(catalog); }
    bool prepare_ray_frames(std::string& error);
    bool prepare_vsm_frames(std::string& error);
    void release_vsm_frames();
    bool prepare_indirect_frames(RaySurfacePass& surfaces, BindlessTextureTable& alpha_textures,
        std::string& error);
    void release_indirect_frames();
    void release_ray_frames();
    bool sample_sky(const std::array<float, 3>& position, std::array<float, 6>& faces) const
    { return sample_sky_probes(sky_probe_layout_, sky_probe_cpu_, position, faces); }
    void set_ray_queries(RayQueryBatch* queries, const BindlessTextureTable* sky_probe_textures)
    {
        ray_queries_ = queries;
        sky_probe_textures_ = sky_probe_textures;
        if (sky_probe_textures_)
            window_.frame().set_frame_observer(gpu_particles_submitted, gpu_particles_completed, this);
    }
    using ModelRayPreparer = std::function<bool(GpuModel&, IKinematics*, uint32_t, VkCommandBuffer,
        const VkTransformMatrixKHR&, RayGeometryBatch&, std::vector<TextureMatrixSet>&, bool&, std::string&)>;
    void set_model_ray_preparer(ModelRayPreparer prepare) { model_ray_preparer_ = std::move(prepare); }
    template <class T> void retire(std::unique_ptr<T> resource)
    {
        retired_.retire(std::move(resource));
    }
    void collect_retired(uint32_t slot);

    void queue_model(GpuModel &model, IKinematics *skeleton, const float (&mvp)[16], bool hud = false,
                     float sort_distance = 0.f, const void *instance = nullptr, float lod = 1.f,
                     const Fmatrix *world = nullptr, const float *hemi_faces = nullptr, bool shadow_only = false,
                     const Fsphere *shadow_bounds = nullptr, float opacity = 1.f);
    void queue_level_visual(uint32_t index, const float (&mvp)[16], bool hud = false, float sort_distance = 0.f,
                            const void *instance = nullptr, float lod = 1.f, const Fmatrix* world = nullptr);
    void queue_particle(IRenderVisual *visual, const float (&mvp)[16], const Fvector &right, const Fvector &up,
                        bool hud, float distance);
    void discard_scene_draws();
    void discard_model_draws(const void *instance);
    void use_scene_visibility(bool enabled)
    {
        scene_visibility_ = enabled;
    }
    void destroy();
    void save_pipeline_cache();
    bool reset_required() const
    {
        return reset_required_;
    }
    bool device_lost() const
    {
        return window_.frame().device_lost();
    }
    bool surface_lost() const
    {
        return window_.frame().surface_lost();
    }
    void clear_reset_required()
    {
        reset_required_ = false;
    }
    void request_reset()
    {
        if (window_.device())
            reset_required_ = true;
    }
    ~VulkanGameDevice()
    {
        destroy();
    }

    VulkanUIRender &ui()
    {
        return ui_;
    }
    uint32_t last_ui_draw_calls() const
    {
        return last_ui_draw_calls_;
    }
    uint32_t last_ui_triangles() const
    {
        return last_ui_triangles_;
    }
    std::unique_ptr<VulkanUIShader> create_ui_shader()
    {
        return std::make_unique<VulkanUIShader>(textures_, ui_pass_);
    }
    std::unique_ptr<VulkanFontRender> create_font_render()
    {
        return std::make_unique<VulkanFontRender>(*this);
    }
    GameTextureFactory &textures()
    {
        return textures_;
    }
    DeferredPass &deferred()
    {
        return deferred_;
    }
    GameShaderResources &shader_resources()
    {
        return shader_resources_;
    }
    bool request_shader_pair(const std::string &svs, const std::string &sps, SurfaceMode mode, bool hud, bool skinned,
                             std::string &error, const std::vector<std::string> &defines = {},
                             uint64_t *variant_hash = nullptr,
                             const VkSpecializationInfo *vertex_specialization = nullptr,
                             const VkSpecializationInfo *hull_specialization = nullptr,
                             const VkSpecializationInfo *domain_specialization = nullptr,
                             const VkSpecializationInfo *geometry_specialization = nullptr,
                             const VkSpecializationInfo *pixel_specialization = nullptr)
    {
        return shader_resources_.pipeline(deferred_, svs, sps, mode, hud, skinned, error,
            {}, {}, {}, vertex_specialization, hull_specialization, domain_specialization,
            geometry_specialization, pixel_specialization, defines, variant_hash);
    }
    ScenePass &ui_pass()
    {
        return ui_pass_;
    }
    VulkanWindowDevice &window()
    {
        return window_;
    }
    const BufferUploadDispatch &buffer_upload() const
    {
        return buffer_upload_;
    }
    PFN_vkDeviceWaitIdle device_wait_idle_proc() const
    {
        return frame_dispatch_.device_wait_idle;
    }

  private:
    void render_lights(const float (&mvp)[16], bool render_world, const DeferredLight &light,
                       const SunShadowUniform &sun_uniform);
    static bool validate_frame_recording(void *user, std::string& error);
    static void record_ui(const FrameRecordingContext &frame, void *user);
    static bool capture_ui(VkCommandBuffer command, VkImage image, uint32_t index, void *user,
        std::string& error);
    static void record_ui_distortion(const FrameRecordingContext &frame, void *user);
    bool create_ui_scene_sets(std::string &error);
    void release_ui_scene_sets();
    static void record_hud(const FrameRecordingContext &frame, void *user);
    static void record_shadow_models(const FrameRecordingContext &frame, const float (&vp)[16], void *user);
    static bool record_vsm_shadow_models(const FrameRecordingContext &frame, const float *vp,
                                         const VsmPageJob &job, void *user, std::string &error);
    static void record_models(const FrameRecordingContext &frame, void *user);
    static void record_transparent(const FrameRecordingContext &frame, void *user);
    static void record_distortion(const FrameRecordingContext &frame, void *user);
    static void prepare_light_grid(const FrameRecordingContext &frame, void *user);
    static void record_local_lights(const FrameRecordingContext &frame, void *user);
    static void record_level_visuals(const FrameRecordingContext &frame, void *user);
    static bool record_object_velocity(const FrameRecordingContext &frame, void *user, std::string& error);
    static void record_readback(VkCommandBuffer command, VkImage image, VkExtent2D extent, void *user);
    static void record_postprocess(const FrameRecordingContext &frame, void *user);
    static void record_display_grade(const FrameRecordingContext &frame, void *user);
    static void record_hdr_output(const FrameRecordingContext& frame, void* user);
    bool create_postprocess(std::string &error);
    bool refresh_color_maps(std::string &error);
    bool update_rain_shadow_resolution(std::string &error);
    bool create_forward_targets(std::string &error);
    bool initialize_gbuffer_targets(std::string &error);
    void release_forward_targets();
    struct LevelDraw
    {
        uint32_t index{};
        std::array<float, 16> mvp{};
        std::array<float, 16> previous_mvp{};
        bool has_previous_mvp{};
        bool moved{};
        float sort_distance{};
        float lod{1.f};
        bool hud{};
        const void *instance{};
    };
    struct ModelDraw
    {
        GpuModel *model{};
        const void *instance{};
        IKinematics *skeleton{};
        std::array<float, 16> mvp{};
        std::array<float, 16> previous_mvp{};
        bool has_previous_mvp{};
        bool moved{};
        std::array<float, 12> normal_rows{};
        std::array<float, 16> world{};
        std::array<float, 3> hemi_negative{};
        std::array<float, 4> shadow_sphere{};
        bool has_shadow_sphere{};
        float sort_distance{};
        float lod{1.f};
        float opacity{1.f};
        bool hud{};
    };
    struct ParticleDraw
    {
        IRenderVisual *visual{};
        std::array<float, 16> mvp{};
        Fvector right{}, up{};
        float sort_distance{};
        bool hud{};
    };
    bool record_gpu_particle_effect(const FrameRecordingContext& frame, VulkanParticleEffect& effect,
        const ParticleDraw& draw, GeometryPhase phase, VulkanParticleEffect* root_effect = nullptr);
    bool record_gpu_particle_motion(const FrameRecordingContext& frame, VulkanParticleEffect& effect,
        const ParticleDraw& draw, const float* current_mvp, const float* previous_mvp,
        bool transparent, VulkanParticleEffect* root_effect = nullptr);
    VulkanWindowDevice window_;
    FrameBufferPool frame_buffer_pool_;
    std::unique_ptr<GpuParticleRuntime> gpu_particles_;
    std::unique_ptr<GpuParticleEmitters> gpu_particle_emitters_;
    std::shared_ptr<const ParticleCatalog> gpu_particle_catalog_;
    GpuParticleMaterials gpu_particle_materials_;
    std::unique_ptr<GpuParticleGraphics> gpu_particle_graphics_;
    bool gpu_particle_graphics_failed_{};
    std::unique_ptr<SmokeTrailCompute> smoke_compute_;
    std::unique_ptr<SmokeTrailNoise> smoke_noise_;
    std::unique_ptr<SmokeTrailDrawPass> smoke_draw_;
    std::unique_ptr<SmokeTrailDrawPass> trail_draw_;
    TrailManager trail_manager_;
    VkImageView trail_texture_{};
    bool trail_enabled_{}, trail_failed_{};
    SmokeTrailController smoke_controller_;
    std::array<SmokeTrailController::Frame, FrameContext::FramesInFlight> smoke_frames_{};
    std::array<bool, FrameContext::FramesInFlight> smoke_recorded_{};
    bool smoke_initialized_{}, smoke_failed_{};
    std::array<bool, FrameContext::FramesInFlight> gpu_particle_recorded_{}, gpu_particle_pending_{};
    std::array<bool, FrameContext::FramesInFlight> gpu_collision_recorded_{};
    GpuParticleLevelCollision gpu_collision_;
    bool gpu_collision_uploaded_{};
    bool gpu_particle_failed_{};
    static bool record_gpu_particles(const FrameRecordingContext& frame, void* user, std::string& error);
    static bool record_pre_geometry(const FrameRecordingContext& frame, void* user, std::string& error);
    static void gpu_particles_submitted(uint32_t slot, uint32_t image, void* user);
    static bool gpu_particles_completed(uint32_t slot, uint32_t image, void* user, std::string& error);
#if defined(XR_PLATFORM_ANDROID)
    SDL_Window *scaled_android_window_{};
    VkExtent2D requested_android_extent_{};
#endif
    FrameDispatch frame_dispatch_{};
    TextureUploadDispatch texture_dispatch_{};
    BufferUploadDispatch buffer_upload_{};
    GBufferTargets targets_;
    GBufferMsaaConfig gbuffer_msaa_config_{};
    bool gbuffer_msaa_configured_{};
    RayFrameResources ray_frames_;
    VsmFrameResources vsm_frames_;
    TreeWindParameters current_tree_wind_{}, previous_tree_wind_{};
    VsmClipmap vsm_clipmap_{};
    bool vsm_cache_initialized_{};
    enum class VsmWorkState { Idle, Waiting, Ready };
    VsmWorkState vsm_work_state_{VsmWorkState::Idle};
    VsmClipmap vsm_pending_clipmap_{}, vsm_previous_clipmap_{};
    VsmClipmap vsm_visible_clipmap_{};
    VsmStaticCacheParameters vsm_pending_cache_{};
    uint32_t vsm_pending_frame_{}, vsm_pending_image_{};
    const GpuLevel* vsm_pending_level_{};
    const GpuLevel* vsm_previous_level_{};
    const GpuLevel* vsm_visible_level_{};
    bool vsm_previous_valid_{};
    bool vsm_visible_valid_{};
    bool vsm_ray_visibility_{};
    std::vector<VsmPageJob> vsm_dynamic_jobs_;
    static bool vsm_frame_completed(uint32_t frame, uint32_t image, void* user, std::string& error);
    static bool record_vsm_requests(const FrameRecordingContext& frame, void* user);
    static bool record_vsm_dynamic_page(const FrameRecordingContext& frame, const float* vp,
        const VsmPageJob& job, void* user, std::string& error);
    IndirectRayFrames indirect_frames_;
    TemporalResolve indirect_temporal_;
    RaySurfacePass* indirect_surfaces_{};
    BindlessTextureTable* indirect_alpha_textures_{};
    uint32_t indirect_recorded_image_{UINT32_MAX};
    RayQueryBatch* ray_queries_{};
    const BindlessTextureTable* sky_probe_textures_{};
    std::unique_ptr<SkyProbeBakePass> sky_probe_bake_;
    framegraph::SkyProbeLayout sky_probe_layout_;
    BufferResource sky_probe_results_;
    uint32_t sky_probe_next_{};
    uint32_t sky_probe_last_slot_{UINT32_MAX};
    std::vector<SkyProbeVisibility> sky_probe_cpu_;
    bool sky_probe_failed_{};
    std::array<uint32_t, FrameContext::FramesInFlight> sky_probe_recorded_{};
    void record_sky_probe_slice(const FrameRecordingContext& frame, const LevelRayScene& scene);
    FrameRayScenes model_ray_scenes_;
    ModelRayPreparer model_ray_preparer_;
    static bool record_ray_visibility(const FrameRecordingContext& frame, const GpuLevel& level,
        const SunShadowUniform& shadow, const DeferredLight& light, void* user);
    HiZPyramid hiz_;
    MotionVectors motion_;
    ObjectMotionHistory object_motion_;
    ObjectVelocityPass object_velocity_;
    TemporalExposure exposure_;
    TemporalResolve temporal_;
    Fsr3Upscaler fsr3_upscaler_;
    std::array<float, 2> temporal_jitter_ndc_{};
    uint32_t temporal_sample_index_{};
    bool temporal_jitter_active_{};
    bool hiz_unsupported_dynamic_previous_{};
    WaterTargets water_targets_;
    WaterTargets ui_snapshot_targets_;
    SunShadowTargets sun_shadows_;
    SunShadowTargets rain_shadows_;
    LocalShadowTargets local_shadows_;
    std::vector<std::vector<VkDescriptorSet>> local_sets_;
    std::vector<VkBuffer> local_descriptor_buffers_;
    std::vector<VulkanLightSnapshot> light_snapshots_;
    struct LightCandidate
    {
        const VulkanLightSnapshot *light;
        float distance;
    };
    std::vector<LightCandidate> light_candidates_;
    ResourceRetirement retired_{FrameContext::FramesInFlight};
    std::unordered_set<std::string> failed_light_projections_;
    std::unordered_map<std::string, VkImageView> light_projections_;
    std::vector<std::pair<std::string, VkImageView>> indirect_projections_;
    std::vector<VkImageView> local_projection_views_;
    std::vector<std::vector<VkImageView>> local_bound_projections_;
    std::vector<LocalLightUniform> local_uniforms_;
    std::vector<VkRect2D> local_scissors_;
    std::vector<BufferResource> forward_buffers_;
    ForwardLightUniform forward_lighting_{};
    bool local_lights_recorded_{true};
    DeferredPass deferred_;
    GameShaderResources shader_resources_;
    ScenePass ui_pass_;
    std::vector<VkDescriptorSet> ui_scene_sets_;
    std::vector<VkDescriptorSet> ui_depth_sets_;
    GameTextureFactory textures_;
    std::unordered_map<std::string, VkDescriptorSet> glow_textures_;
    VulkanUIRender ui_;
    DeferredFrame frame_;
    static void record_level_visibility(const FrameRecordingContext&, const GpuLevel&, const float (&)[16], void*);
    static void draw_level_visibility(const FrameRecordingContext&, void*);
    bool initialize_visibility_gbuffer(std::string& error);
    void release_visibility_gbuffer();
    bool ensure_visibility_targets(uint32_t slot, VkExtent2D extent, std::string& error);
    void release_visibility_targets(VisibilityPasses::Targets& targets);
    const VisibilityPasses* visibility_passes_{};
    const BindlessTextureTable* visibility_materials_{};
    std::array<VisibilityPasses::Targets, FrameContext::FramesInFlight> visibility_targets_{};
    VkSampler visibility_sampler_{};
    ScreenCopyDispatch visibility_graphics_{};
    VkPipeline visibility_gbuffer_pipeline_{};
    VkPipelineLayout visibility_gbuffer_layout_{};
    VkDescriptorSetLayout visibility_descriptor_layout_{};
    VkDescriptorPool visibility_descriptor_pool_{};
    std::array<VkDescriptorSet, FrameContext::FramesInFlight> visibility_descriptors_{};
    std::array<bool, FrameContext::FramesInFlight> visibility_recorded_{};
    std::array<BufferResource, FrameContext::FramesInFlight> visibility_draw_masks_;
    std::array<VkBufferView, FrameContext::FramesInFlight> visibility_mask_views_{};
    PFN_vkDestroyBufferView destroy_visibility_mask_view_{};
    std::vector<std::unique_ptr<ScreenCopyPass>> postprocess_passes_;
    std::vector<std::array<std::unique_ptr<ScreenCopyPass>, 2>> temporal_upscale_passes_;
    std::vector<std::unique_ptr<ScreenCopyPass>> fsr3_upscale_passes_;
    bool temporal_upscale_{};
    std::vector<std::unique_ptr<ScreenCopyPass>> sun_shaft_passes_;
    bool sun_shafts_active_{};
    std::vector<std::unique_ptr<ScreenCopyPass>> bloom_bright_passes_;
    std::vector<std::vector<std::unique_ptr<ScreenCopyPass>>> bloom_upsample_passes_;
    float legacy_bloom_factor_{0.5f};
    std::vector<std::unique_ptr<ScreenCopyPass>> display_grade_passes_;
    std::vector<std::unique_ptr<ScreenCopyPass>> hdr_output_passes_;
    PostProcessConstants postprocess_params_{};
    std::string color_map_names_[2];
    std::string active_map_names_[2];
    VkImageView color_map_views_[2]{};
    bool ui_recorded_{true};
    std::string ui_error_;
    LightGrid light_grid_;
    PipelineCache pipeline_cache_;
    std::vector<ModelDraw> model_draws_;
    std::vector<ModelDraw> shadow_model_draws_;
    std::unordered_multimap<const GpuModel *, size_t> shadow_draw_lookup_;
    std::vector<ParticleDraw> particle_draws_;
    std::vector<VulkanRainRender *> rain_draws_;
    std::vector<VulkanThunderboltRender *> thunderbolt_draws_;
    std::vector<LevelDraw> level_draws_;
    // A visual can be submitted through several sector roots. Look up only
    // submissions for the same visual instead of scanning every draw.
    std::unordered_multimap<uint32_t, size_t> level_draw_lookup_;
    const GpuLevel *current_level_{};
    std::array<float, 16> scene_mvp_{};
    bool scene_visibility_{};
    bool hiz_dynamic_mask_recorded_{true};
    bool level_recorded_{true};
    std::string level_error_;
    bool models_recorded_{true};
    std::string model_error_;
    PFN_vkCreateSampler create_sampler_{};
    PFN_vkDestroySampler destroy_sampler_{};
    bool reset_required_{};
    VkDescriptorSet weather_set_{};
    WeatherLighting weather_lighting_{};
    BufferResource screenshot_buffer_;
    bool screenshot_requested_{}, screenshot_ready_{}, readback_enabled_{};
    uint32_t last_ui_draw_calls_{}, last_ui_triangles_{};
    VkExtent2D screenshot_extent_{};
};
} // namespace xray::render::vulkan
