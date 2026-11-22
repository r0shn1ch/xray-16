#pragma once

#include "ParticleCatalog.h"
#include "ParticleGroupTiming.h"
#include "BufferResource.h"
#include "DeferredPass.h"
#include "GameTextureFactory.h"
#include "Include/xrRender/ParticleCustom.h"
#include "Include/xrRender/RenderVisual.h"
#include "xrEngine/vis_common.h"

#include <array>
#include <functional>
#include <memory>
#include <unordered_map>

namespace PAPI { struct Particle; }

namespace xray::render::vulkan
{
class GpuParticleEmitters;
class ObjectVelocityPass;
// particles_xadd.s replaces the serialized S_SET program with particle.ps.
inline bool particle_uses_screen_program(const std::string& shader, const std::string& cls)
{
    return cls == "S_SET   " && shader != "particles\\xadd";
}

inline bool particle_records_in_phase(bool hud, bool screen_set, int blend_mode,
    GeometryPhase phase)
{
    const bool deferred_set = !hud && !screen_set && blend_mode == 0;
    return (phase == GeometryPhase::OpaqueAndAlphaTest) == deferred_set;
}

class VulkanParticleEffect final : public IRenderVisual, public IParticleCustom
{
public:
    VulkanParticleEffect(std::shared_ptr<const ParticleCatalog> catalog, const ParticleEffectDef& def);
    ~VulkanParticleEffect() override;
    VulkanParticleEffect(const VulkanParticleEffect&) = delete;
    bool initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
        const BufferResourceDispatch& dispatch, GameTextureFactory& textures, DeferredPass& pass,
        std::string& error);
    bool record(const FrameRecordingContext& frame, const DeferredPass& pass, const float (&mvp)[16],
        const Fvector& right, const Fvector& up, bool hud, std::string& error,
        GeometryPhase phase = GeometryPhase::Transparent, bool prepare_only = false);
    bool prepare_motion_geometry(const FrameRecordingContext& frame, const DeferredPass& pass,
        const float (&mvp)[16], const Fvector& right, const Fvector& up, bool hud,
        bool gpu_draw_active, std::string& error);
    bool record_particle_motion(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
        const float (&mvp)[16], const float (&previous_mvp)[16], bool history_valid,
        bool hud, bool gpu_draw_active);
    bool needs_dynamic_occluder_mask(bool hud, bool gpu_draw_active) const;
    bool supports_dynamic_occluder_mask(bool hud, bool gpu_draw_active) const;
    void invalidate_dynamic_occluder_history(uint32_t frame_index);
    bool record_dynamic_occluder_mask(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
        const float (&mvp)[16], const float (&previous_mvp)[16], bool history_valid,
        bool hud, bool gpu_draw_active);
    void release_gpu(); // The owner waits for all submitted frames first.
    uint8_t priority() const { return priority_; }
    bool has_active_distortion() const;
    void set_events(std::function<void(bool, PAPI::Particle&, u32)> events);
    void particles(PAPI::Particle*& data, u32& count) const;
    const ParticleEffectDef& definition() const { return *def_; }
    Fvector particle_world_position(const PAPI::Particle& particle) const;
    int particle_handle() const { return effect_; }
    // Attach before Play; CPU simulation remains active until GPU rendering
    // and event handling are wired into the game frame.
    bool attach_gpu_emitter(GpuParticleEmitters& emitters, std::string& error);
    void detach_gpu_emitter();
    uint64_t gpu_emitter_handle() const { return gpu_emitter_; }
    bool gpu_visible_index(uint32_t& index) const;
    const std::vector<uint32_t>* gpu_draw_programs() const;
    bool gpu_draw_compatible() const;
    bool gpu_water_contact() const { return water_contact_; }
    const Fmatrix& gpu_water_plane() const { return water_plane_; }
    void append_gpu_visible(std::vector<uint32_t>& roots) const;
    u32 getType() const override { return 8; }
    vis_data& getVisData() override { return visibility_; }
    IParticleCustom* dcast_ParticleCustom() override { return this; }
    void OnDeviceCreate() override {}
    void OnDeviceDestroy() override { detach_gpu_emitter(); release_gpu(); }
    void set_water_plane(const Fmatrix& plane) { water_plane_.set(plane); }
    void UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL transform) override;
    void OnFrame(u32 dt) override;
    void Play() override;
    void Stop(BOOL deferred = TRUE) override;
    BOOL IsPlaying() override { return playing_; }
    u32 ParticlesCount() override;
    float GetTimeLimit() override { return def_->time_limit; }
    const shared_str Name() override { return def_->name.c_str(); }
    void SetHudMode(BOOL value) override;
    BOOL GetHudMode() override { return hud_; }
#ifdef DEBUG
    shared_str getDebugName() override { return def_->name.c_str(); }
#endif

private:
    static void on_birth(void* owner, u32, PAPI::Particle& particle, u32);
    static void on_dead(void* owner, u32, PAPI::Particle& particle, u32);
    std::shared_ptr<const ParticleCatalog> catalog_;
    const ParticleEffectDef* def_{};
    vis_data visibility_;
    Fmatrix water_plane_;
    bool water_contact_{};
    void project_water_center(Fvector& center) const;
    Fvector initial_position_;
    Fmatrix transform_;
    Fmatrix gpu_parent_;
    Fvector gpu_parent_velocity_{};
    bool gpu_parent_valid_{}, gpu_parent_local_{};
    GpuParticleEmitters* gpu_emitters_{};
    uint64_t gpu_emitter_{};
    void gpu_emitter_failed(const std::string& error);
    bool playing_{}, deferred_{}, hud_{};
    float elapsed_{};
    u32 remainder_{};
    int effect_{-1}, actions_{-1};
    VkDevice device_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    BufferResourceDispatch dispatch_{};
    GameTextureFactory* textures_{};
    DeferredPass* pass_{};
    VkDescriptorSet material_{};
    std::string material_name_;
    int blend_mode_{-1};
    int alpha_ref_{128};
    bool screen_set_{}, distortion_{};
    uint8_t priority_{1};
    TextureMatrixSet matrices_;
    std::function<void(bool, PAPI::Particle&, u32)> events_;
    struct Buffers
    {
        BufferResource vertices, indices;
        BufferResource previous_vertices;
        size_t capacity{};
        uint32_t dynamic_occluder_index_count{};
        uint32_t particle_frame{};
        bool dynamic_occluder_ready{};
        bool particle_motion_ready{};
        uint32_t particle_motion_index_count{};
        std::vector<LevelVertex> scratch;
        std::vector<LevelVertex> previous_scratch;
        std::vector<uint64_t> rendered_particle_ids;
        std::unordered_map<uint64_t, std::array<std::array<float, 3>, 4>> particle_vertices;
    };
    std::array<Buffers, FrameContext::FramesInFlight> buffers_;
    std::vector<uint64_t> particle_ids_;
    uint64_t next_particle_id_{1};
};

class VulkanParticleGroup final : public IRenderVisual, public IParticleCustom
{
public:
    VulkanParticleGroup(std::shared_ptr<const ParticleCatalog> catalog, const ParticleGroupDef& def);
    ~VulkanParticleGroup() override;
    bool initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
        const BufferResourceDispatch& dispatch, GameTextureFactory& textures, DeferredPass& pass,
        std::string& error);
    bool attach_gpu_emitters(GpuParticleEmitters& emitters, std::string& error);
    void detach_gpu_emitters();
    void append_gpu_visible(std::vector<uint32_t>& roots) const;
    const auto& effects() const { return effects_; }
    void visit_effects(const std::function<void(const VulkanParticleEffect&)>& visit) const;
    bool has_priority(uint8_t priority) const;
    bool has_distortion() const;
    size_t triggered_count() const { return triggered_.size(); }
    size_t related_count(size_t root) const { return root < related_.size() ? related_[root].size() : 0; }
    bool record(const FrameRecordingContext& frame, const DeferredPass& pass, const float (&mvp)[16],
        const Fvector& right, const Fvector& up, bool hud, std::string& error,
        GeometryPhase phase = GeometryPhase::Transparent, int priority_filter = -1,
        const std::function<bool(VulkanParticleEffect&)>& gpu_record = {},
        const std::function<bool(VulkanParticleEffect&, VulkanParticleEffect&)>& gpu_child_record = {});
    bool prepare_motion_geometry(const FrameRecordingContext& frame, const DeferredPass& pass,
        const float (&mvp)[16], const Fvector& right, const Fvector& up, bool hud,
        bool gpu_draw_active, std::string& error);
    bool record_particle_motion(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
        const float (&mvp)[16], const float (&previous_mvp)[16], bool history_valid,
        bool hud, bool gpu_draw_active, bool transparent_only, int priority_filter = -1,
        const std::function<bool(VulkanParticleEffect&)>& gpu_record = {},
        const std::function<bool(VulkanParticleEffect&, VulkanParticleEffect&)>& gpu_child_record = {});
    bool supports_dynamic_occluder_mask(bool hud, bool gpu_draw_active) const;
    bool record_dynamic_occluder_mask(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
        const float (&mvp)[16], const float (&previous_mvp)[16], bool history_valid,
        bool hud, bool gpu_draw_active);
    void invalidate_dynamic_occluder_history(uint32_t frame_index);
    u32 getType() const override { return 9; }
    vis_data& getVisData() override { return visibility_; }
    IRenderVisual* getSubModel(u8 index) override
    { return index < effects_.size() ? effects_[index].get() : nullptr; }
    IParticleCustom* dcast_ParticleCustom() override { return this; }
    void OnDeviceCreate() override {}
    void OnDeviceDestroy() override;
    void UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL transform) override;
    void OnFrame(u32 dt) override;
    void Play() override;
    void Stop(BOOL deferred = TRUE) override;
    BOOL IsPlaying() override { return playing_; }
    u32 ParticlesCount() override;
    float GetTimeLimit() override { return def_->time_limit; }
    const shared_str Name() override { return def_->name.c_str(); }
    void SetHudMode(BOOL value) override;
    BOOL GetHudMode() override { return hud_; }
#ifdef DEBUG
    shared_str getDebugName() override { return def_->name.c_str(); }
#endif

private:
    size_t spawn(const std::string& name, const Fvector& position, const Fvector& velocity, bool related, size_t root);
    void on_particle(size_t index, bool birth, PAPI::Particle& particle, u32 particle_index);
    std::shared_ptr<const ParticleCatalog> catalog_;
    const ParticleGroupDef* def_{};
    std::vector<std::unique_ptr<VulkanParticleEffect>> effects_;
    std::vector<ParticleGroupEntryPhase> entry_phases_;
    struct Triggered
    {
        std::string name;
        std::unique_ptr<VulkanParticleEffect> effect;
        bool related{};
        size_t root{};
    };
    struct Pending
    {
        std::string name;
        Fvector position, velocity;
        size_t root{};
        u32 particle_index{};
        enum class Kind { Free, RelatedBirth, RelatedDeath } kind{Kind::Free};
    };
    std::vector<Triggered> triggered_;
    std::vector<Pending> pending_;
    std::vector<std::vector<size_t>> related_;
    Fmatrix parent_;
    bool transform_parent_{};
    VkDevice device_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    BufferResourceDispatch dispatch_{};
    GameTextureFactory* textures_{};
    DeferredPass* pass_{};
    vis_data visibility_;
    Fvector initial_position_;
    float time_{};
    bool playing_{}, deferred_{}, hud_{};
};
} // namespace xray::render::vulkan
