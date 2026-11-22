#pragma once

#include "GpuLevel.h"
#include "BindlessTextureTable.h"
#include "VisibilityPasses.h"
#include "RayQueryBatch.h"
#include "RaySurface.h"
#include "ShaderModule.h"
#include "VulkanCameraState.h"
#include "VulkanDeviceResourceState.h"
#include "VulkanFramePhaseState.h"
#include "VulkanGameLighting.h"
#include "Light_DB.h"
#include "VulkanRenderContextState.h"
#include "VulkanSurfaceLifecycle.h"
#include "VulkanFrameMetrics.h"
#include "VulkanFrameBenchmark.h"
#include "ScreenCopyPass.h"
#include "ResourceIssueCounter.h"
#include "xrEngine/Render.h"
#include "xrEngine/pure.h"
#include "xrCommon/xr_vector.h"

#include <chrono>
#include <memory>
#include <unordered_map>
#include <vector>

namespace xray::render::vulkan
{
class VulkanGameDevice;
class VulkanWindowDevice;
class VulkanFontRender;
class VulkanUIShader;
class VulkanModelVisual;
class GpuModel;
struct RayModelGeometry;
struct RayModelTextureSlots;
class VulkanParticleEffect;
class VulkanParticleGroup;
class GpuParticleEmitters;
struct ParticleCatalog;
// IRender adapter for the Vulkan device, level geometry and model resources.
class VulkanLevelRender : public IRender, public pureAppDeactivate
{
public:
    VulkanLevelRender();
    explicit VulkanLevelRender(VulkanGameDevice& device);
    ~VulkanLevelRender() override;
    void bind_level_device(VulkanGameDevice& resources);
    // The manager must outlive attached visuals; detach before its teardown.
    void set_gpu_particle_emitters(GpuParticleEmitters* emitters);
    std::unique_ptr<VulkanUIShader> create_ui_shader();
    std::unique_ptr<VulkanFontRender> create_font_render();
    void bind_level_device(VkDevice device, VkQueue queue, VkCommandPool pool,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferUploadDispatch& upload,
        GameTextureFactory& textures, DeferredPass& pass, PFN_vkDeviceWaitIdle wait_idle);
    void level_Load(IReader* reader) override;
    void level_Unload() override;
    void reset_begin() override;
    void reset_end() override;
    IRenderVisual* getVisual(int index) override;
    HRESULT shader_compile(pcstr name, IReader* source, pcstr entry, pcstr target, u32 flags, void*& result) override;
    IRenderVisual* model_Create(pcstr name, IReader* data = nullptr) override;
    IRenderVisual* model_CreateChild(pcstr name, IReader* data) override;
    IRenderVisual* model_Duplicate(IRenderVisual* visual) override;
    IRenderVisual* model_CreateParticles(pcstr name) override;
    void model_Delete(IRenderVisual*& visual, bool discard = false) override;
    void models_Clear(bool complete) override;
    void Create(SDL_Window* window, u32& width, u32& height,
        float& half_width, float& half_height) override;
    void Destroy() override;
    bool start_benchmark(uint32_t frames);
    void interrupt_benchmark(const char* reason);
    void request_sampler_settings_reset();
    void OnAppDeactivate() override { interrupt_benchmark("application deactivated"); }
    void Reset(SDL_Window* window, u32& width, u32& height,
        float& half_width, float& half_height) override;
    void SetupStates() override;
    void OnDeviceCreate(pcstr shader_archive) override;
    void OnDeviceDestroy(bool keep_textures) override;
    void Begin() override;
    void Calculate() override;
    void Render() override;
    void UpdateSmokeTrail(const Fvector& position, const Fvector& direction, float dt, bool hud) override;
    bool UpdateProjectileTrail(u64 id, const Fvector& start, const Fvector& end, float width, u32 frame, float dt, u32 color) override;
    void RenderMenu() override;
    void Clear() override;
    void End() override;
    void ClearTarget() override;
    void OnCameraUpdated() override;
    void SetCacheXform(Fmatrix& view, Fmatrix& project) override;
    RenderContext GetCurrentContext() const override;
    void MakeContextCurrent(RenderContext context) override;
    void add_Visual(u32 context_id, IRenderable* root, IRenderVisual* visual,
        Fmatrix& world) override;
    void queue_visual(IRenderable *root, IRenderVisual *visual, Fmatrix &world, bool shadow_only);
    IRender_ObjectSpecific* ros_create(IRenderable* parent) override;
    void ros_destroy(IRender_ObjectSpecific*& object) override;
    IRender_Light* light_create() override;
    void light_destroy(IRender_Light* light) override;
    IRender_Glow* glow_create() override;
    void glow_destroy(IRender_Glow* glow) override;
    void add_StaticWallmark(const wm_shader& shader, const Fvector& point,
        float size, CDB::TRI* triangle, Fvector* vertices) override;
    void add_StaticWallmark(IWallMarkArray* array, const Fvector& point,
        float size, CDB::TRI* triangle, Fvector* vertices) override;
    void add_SkeletonWallmark(const Fmatrix* transform, IKinematics* skeleton,
        IWallMarkArray* array, const Fvector& start, const Fvector& direction,
        float size) override;
    void clear_static_wallmarks() override;
    DeviceState GetDeviceState() override;
    void OnAppLifecycleChanged(bool active) override;
    GenerationLevel GetGeneration() const override { return GENERATION_R2; }
    BackendAPI GetBackendAPI() const override { return BackendAPI::Vulkan; }
    bool is_sun_static() override { return false; }
    u32 get_dx_level() override { return 110; }
    void create() override;
    void destroy() override;
    void DumpStatistics(IGameFont& font, IPerformanceAlert* alert) override;
    pcstr getShaderPath() override { return "vk\\"; }
    xrImTextureData GetImGuiTextureId(pcstr texture_name) override;
    void model_Logging(bool enabled) override { model_logging_ = enabled; }
    void models_Prefetch() override;
    bool occ_visible(vis_data& visual) override;
    bool occ_visible(Fbox& bounds) override;
    bool occ_visible(sPoly& polygon) override;
    void BeforeWorldRender() override;
    void AfterWorldRender() override;
    void Screenshot(ScreenshotMode mode = SM_NORMAL, pcstr name = nullptr) override;
    bool ScreenshotPending() const override { return screenshot_ != nullptr; }
    void SetPostProcessParams(const SPPInfo& ppi) override;
    void setGamma(float value) override;
    void setBrightness(float value) override;
    void setContrast(float value) override;
    void updateGamma() override;
    void ObtainRequiredWindowFlags(u32& flags) override;
    void overdrawBegin() override;
    void overdrawEnd() override;
    void DeferredLoad(bool enabled) override;
    void ResourcesDeferredUpload() override;
    void ResourcesDeferredUnload() override;
    void ResourcesGetMemoryUsage(u32& m_base, u32& c_base, u32& m_lmaps, u32& c_lmaps) override;
    void ResourcesDestroyNecessaryTextures() override;
    void ResourcesStoreNecessaryTextures() override;
    void ResourcesDumpMemoryUsage() override;
    bool HWSupportsShaderYUV2RGB() override { return false; }
    bool GetForceGPU_REF() override { return false; }
    u32 GetCacheStatCalls() override { return frame_draw_calls_; }
    u32 GetCacheStatPolys() override { return frame_triangles_; }
    void OnAssetsChanged() override;

protected:
    GpuLevel& gpu_level() { return level_; }
    const GpuLevel& gpu_level() const { return level_; }

private:
  void refresh_engine_shader_defines();
  void release_ray_materials();
  void release_ray_diffuse_materials();
  bool prepare_visibility_materials(const VulkanWindowDevice& window, std::string& error);
  void release_visibility_materials();
  bool acquire_model_ray_materials(const std::vector<RayModelGeometry>& geometry,
      std::vector<RayModelTextureSlots>& slots, std::string& error);
  bool prepare_model_ray_batch(GpuModel& model, IKinematics* skeleton, uint32_t slot, VkCommandBuffer command,
      const VkTransformMatrixKHR& transform, RayGeometryBatch& batch, std::vector<TextureMatrixSet>& matrices,
      bool& recorded_update, std::string& error);
  void flush_render_failures();
  void flush_resource_failures();
  void retire_model_uploads();
  std::unique_ptr<VulkanModelVisual> create_model_tree(const VisualRecord &record, const std::string &inherited_texture, const std::string &cache_name,
                                                       std::string &error, bool skeletal_child = false,
                                                       unsigned skeleton_lod_depth = 0);
  std::unique_ptr<VulkanModelVisual> load_model_base(pcstr name, IReader *data, const std::string &cache_name, std::string &error, bool skeletal_child = false);
  VulkanModelVisual *get_model_base(pcstr name, IReader *data, const std::string &cache_name, std::string &error, bool skeletal_child = false);
  void destroy_all_models();
  void create_wallmark(ModelGeometry &&geometry, const Fvector &center, IKinematics *skeleton = nullptr, const Fmatrix *transform = nullptr);
  Fmatrix current_view_projection() const;
  void update_view_frustum();
  GpuLevel level_;
  BindlessTextureTable ray_materials_;
  RayQueryBatch ray_queries_;
  BindlessTextureTable ray_diffuse_materials_;
  BindlessTextureTable visibility_materials_;
  std::unique_ptr<VisibilityPasses> visibility_passes_;
  std::vector<VkImageView> visibility_material_views_;
  RaySurfacePass ray_surfaces_;
  std::vector<VkImageView> ray_material_views_;
  std::vector<VkImageView> ray_diffuse_material_views_;
  std::vector<std::string> ray_material_names_, ray_diffuse_material_names_;
  std::vector<uint8_t> ray_diffuse_material_linear_;
  std::string last_render_error_;
  uint32_t repeated_render_errors_{};
  ResourceIssueCounter resource_failures_;
  std::unordered_map<IRenderVisual *, std::unique_ptr<VulkanModelVisual>> models_;
  std::unordered_map<IRenderVisual *, std::unique_ptr<VulkanParticleEffect>> particle_effects_;
  std::unordered_map<IRenderVisual *, std::unique_ptr<VulkanParticleGroup>> particle_groups_;
  std::shared_ptr<const ParticleCatalog> particle_catalog_;
  GpuParticleEmitters* gpu_particle_emitters_{};
  std::unordered_multimap<std::string, std::unique_ptr<VulkanModelVisual>> model_pool_;
  std::unordered_map<std::string, std::unique_ptr<VulkanModelVisual>> model_bases_;
  std::unordered_map<std::string, std::weak_ptr<GpuModel>> model_gpu_cache_;
  xr_vector<std::weak_ptr<GpuModel>> pending_model_uploads_;
  VulkanCameraState camera_state_;
  VulkanRenderContextState context_state_;
  VkDevice device_{};
  VkQueue queue_{};
  VkCommandPool pool_{};
  VkPhysicalDeviceMemoryProperties memory_{};
  BufferUploadDispatch upload_{};
  GameTextureFactory *textures_{};
  DeferredPass *pass_{};
  PFN_vkDeviceWaitIdle wait_idle_{};
  VulkanGameDevice *game_device_{};
  SDL_Window *window_{};
  VulkanSurfaceLifecycle surface_state_;
  std::unique_ptr<VulkanGameDevice> owned_game_device_;
  VulkanGameDevice *external_game_device_{};
  std::vector<VulkanLight *> lights_;
  std::unordered_map<IRenderVisual*, std::pair<const VulkanModelVisual*, const void*>> model_visual_index_;
  CLight_DB Lights;
  std::vector<VulkanGlow *> glows_;
  IRender_Sector::sector_id_t last_camera_sector_{IRender_Sector::INVALID_SECTOR_ID};
  // World glows use screen-space UI quads. Cache static-geometry visibility
  // briefly so they do not shine through walls without a ray test per frame.
  std::unordered_map<const VulkanGlow*, std::pair<u32, bool>> glow_visibility_;
  std::vector<VulkanObjectSpecific *> object_specifics_;
  struct Wallmark
  {
      std::unique_ptr<GpuModel> model;
      IKinematics *skeleton{};
      const Fmatrix *transform{};
      Fvector center{};
      u16 bone{UINT16_MAX};
      float remaining_ttl{}, lifetime{}, radius{};
      std::string texture;
  };
    std::vector<Wallmark> wallmarks_;
    VulkanDeviceResourceState device_resource_state_;
    bool reset_in_progress_{};
    bool device_loss_reported_{};
    VulkanFrameMetrics metrics_;
    VulkanFrameMetrics log_metrics_;
    VulkanFrameBenchmark benchmark_;
    std::string benchmark_context_;
    void save_benchmark();
    std::chrono::steady_clock::time_point last_metric_frame_{};
    VulkanFramePhaseState frame_phase_;
    bool clear_target_pending_{};
    bool frame_clear_target_{};
    bool deferred_load_{}, model_logging_{};
    bool assets_dirty_{};
    u32 frame_draw_calls_{}, frame_triangles_{};
    float gamma_{1.f}, brightness_{1.f}, contrast_{1.f}, gray_{};
    PostProcessConstants postprocess_{};
    std::string color_map_a_, color_map_b_;
    struct ImGuiTexture { VkDescriptorSet descriptor{}; VkExtent2D extent{}; };
    std::unordered_map<std::string, ImGuiTexture> imgui_textures_;
    struct ScreenshotRequest { ScreenshotMode mode; std::string name; };
    std::unique_ptr<ScreenshotRequest> screenshot_;
    void save_screenshot();
};
}
