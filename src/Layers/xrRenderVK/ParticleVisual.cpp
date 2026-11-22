#include "xrEngine/stdafx.h"
#include "ParticleVisual.h"
#include "ParticleSprite.h"
#include "Include/xrRender/WaterContactEffect.h"
#include "LevelPassGraph.h"
#include "GpuParticleEmitters.h"
#include "ObjectVelocityPass.h"

#include "xrParticles/psystem.h"
#include "xrEngine/device.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>

namespace xray::render::vulkan
{
namespace
{
constexpr u32 step_ms = 33;
constexpr float step_seconds = .033f;
constexpr u32 enabled = 1u << 2;
constexpr u32 deferred_stop = 1u;

void bounds(vis_data& vis, const Fvector& position)
{
    vis.box.set(position, position);
    vis.box.grow(EPS_L);
    vis.box.getsphere(vis.sphere.P, vis.sphere.R);
}

std::array<float, 3> water_plane_normal(const Fmatrix& plane)
{
    return particle_water_plane_normal({
        {plane.i.x, plane.i.y, plane.i.z},
        {plane.j.x, plane.j.y, plane.j.z},
        {plane.k.x, plane.k.y, plane.k.z}});
}

}

void VulkanParticleEffect::on_birth(void* owner, u32, PAPI::Particle& particle, u32 index)
{
    auto& effect = *static_cast<VulkanParticleEffect*>(owner);
    const auto& def = *effect.def_;
    if (index >= effect.particle_ids_.size()) effect.particle_ids_.resize(size_t(index) + 1);
    if (!effect.next_particle_id_) ++effect.next_particle_id_;
    effect.particle_ids_[index] = effect.next_particle_id_++;
    if (def.flags & (1u << 12))
        particle.frame = static_cast<u16>(Random.randI(def.frame_count) * 255);
    if ((def.flags & (1u << 11)) && (def.flags & (1u << 13)) && Random.randI(2))
        particle.flags.set(PAPI::Particle::ANIMATE_CCW, TRUE);
    if (effect.events_) effect.events_(true, particle, index);
}

void VulkanParticleEffect::on_dead(void* owner, u32, PAPI::Particle& particle, u32 index)
{
    auto& effect = *static_cast<VulkanParticleEffect*>(owner);
    if (effect.effect_ >= 0)
    {
        const u32 count = PAPI::ParticleManager()->GetParticlesCount(effect.effect_);
        if (count && index < count && index < effect.particle_ids_.size())
        {
            const size_t last = size_t(count) - 1;
            if (index != last && last < effect.particle_ids_.size())
                effect.particle_ids_[index] = effect.particle_ids_[last];
            if (last < effect.particle_ids_.size()) effect.particle_ids_[last] = 0;
        }
    }
    if (effect.events_) effect.events_(false, particle, index);
}

void VulkanParticleEffect::set_events(std::function<void(bool, PAPI::Particle&, u32)> events)
{
    events_ = std::move(events);
}
void VulkanParticleEffect::particles(PAPI::Particle*& data, u32& count) const
{
    data = nullptr; count = 0;
    if (effect_ >= 0) PAPI::ParticleManager()->GetParticles(effect_, data, count);
}

Fvector VulkanParticleEffect::particle_world_position(const PAPI::Particle& particle) const
{
    Fvector center;
    transform_.transform_tiny(center, particle.pos);
    return center;
}

VulkanParticleEffect::VulkanParticleEffect(std::shared_ptr<const ParticleCatalog> catalog,
    const ParticleEffectDef& def) : catalog_(std::move(catalog)), def_(&def)
{
    water_plane_.identity();
    transform_.identity();
    gpu_parent_.identity();
    initial_position_.set(0, 0, 0);
    bounds(visibility_, initial_position_);
    auto* manager = PAPI::ParticleManager();
    effect_ = manager->CreateEffect(def.max_particles);
    actions_ = manager->CreateActionList();
    if (effect_ >= 0 && actions_ >= 0)
    {
        IReader reader(const_cast<uint8_t*>(def.actions.data()), def.actions.size());
        manager->LoadActions(actions_, reader);
        manager->SetMaxParticles(effect_, def.max_particles);
        manager->SetCallback(effect_, on_birth, on_dead, this, 0);
    }
}

VulkanParticleEffect::~VulkanParticleEffect()
{
    detach_gpu_emitter();
    release_gpu();
    auto* manager = PAPI::ParticleManager();
    if (effect_ >= 0) manager->DestroyEffect(effect_);
    if (actions_ >= 0) manager->DestroyActionList(actions_);
}

bool VulkanParticleEffect::attach_gpu_emitter(GpuParticleEmitters& emitters, std::string& error)
{
    if (gpu_emitters_ || playing_ || ParticlesCount())
    { error = "GPU emitter must be attached before the particle effect starts"; return false; }
    uint64_t handle{};
    if (!emitters.create(*def_, handle, error)) return false;
    if (gpu_parent_valid_ && !emitters.set_parent(handle, gpu_parent_, gpu_parent_velocity_, gpu_parent_local_, error))
    {
        std::string ignored; emitters.destroy(handle, ignored); return false;
    }
    if (hud_ && !emitters.set_hud(handle, true, error))
    {
        std::string ignored; emitters.destroy(handle, ignored); return false;
    }
    gpu_emitters_ = &emitters; gpu_emitter_ = handle; error.clear(); return true;
}
void VulkanParticleEffect::detach_gpu_emitter()
{
    if (gpu_emitters_ && gpu_emitters_->draw_programs(gpu_emitter_))
    {
        std::string error;
        if (!gpu_emitters_->destroy(gpu_emitter_, error))
            Msg("! [renderer-vulkan] GPU particle emitter '%s' release: %s", def_->name.c_str(), error.c_str());
    }
    gpu_emitters_ = nullptr; gpu_emitter_ = 0;
}
bool VulkanParticleEffect::gpu_visible_index(uint32_t& index) const
{
    return gpu_emitters_ && gpu_emitters_->visible_index(gpu_emitter_, index);
}
const std::vector<uint32_t>* VulkanParticleEffect::gpu_draw_programs() const
{
    return gpu_emitters_ ? gpu_emitters_->draw_programs(gpu_emitter_) : nullptr;
}
bool VulkanParticleEffect::gpu_draw_compatible() const
{
    uint32_t root_index{};
    return gpu_visible_index(root_index);
}

bool VulkanParticleEffect::needs_dynamic_occluder_mask(bool hud, bool gpu_draw_active) const
{
    // Screen-set programs are recorded in the forward transparent phase, after
    // this frame's Hi-Z pyramid has been built. Their depth writes therefore
    // cannot become occluders in that pyramid.
    const bool depth_writing_world = !hud && !screen_set_ && material_ && !distortion_ && blend_mode_ == 0;
    if (!depth_writing_world) return false;
    if (gpu_draw_active && gpu_draw_compatible()) return true;
    return effect_ >= 0 && PAPI::ParticleManager()->GetParticlesCount(effect_) != 0;
}

bool VulkanParticleEffect::supports_dynamic_occluder_mask(bool hud, bool gpu_draw_active) const
{
    return !needs_dynamic_occluder_mask(hud, gpu_draw_active) ||
        (!gpu_draw_active || !gpu_draw_compatible());
}

void VulkanParticleEffect::invalidate_dynamic_occluder_history(uint32_t frame_index)
{
    if (frame_index >= buffers_.size()) return;
    auto& buffers = buffers_[frame_index];
    buffers.dynamic_occluder_ready = false;
    buffers.dynamic_occluder_index_count = 0;
    buffers.particle_motion_ready = false;
    buffers.particle_motion_index_count = 0;
    buffers.particle_frame = 0;
    buffers.rendered_particle_ids.clear();
    buffers.particle_vertices.clear();
}

bool VulkanParticleEffect::record_dynamic_occluder_mask(const FrameRecordingContext& frame,
    const ObjectVelocityPass& pass, const float (&mvp)[16], const float (&previous_mvp)[16],
    bool history_valid, bool hud, bool gpu_draw_active)
{
    if (!needs_dynamic_occluder_mask(hud, gpu_draw_active) ||
        (gpu_draw_active && gpu_draw_compatible())) return true;
    if (frame.frame_index >= buffers_.size()) return false;
    Buffers& buffers = buffers_[frame.frame_index];
    if (!buffers.dynamic_occluder_ready || !buffers.dynamic_occluder_index_count ||
        !buffers.vertices.handle() || !buffers.previous_vertices.handle() || !buffers.indices.handle())
        return false;
    const Buffers* previous = history_valid ? &buffers_[frame.frame_index ^ 1u] : nullptr;
    if (previous && previous->particle_frame + 1 != Device.dwFrame) previous = nullptr;
    auto& previous_vertices = buffers.previous_scratch;
    previous_vertices.resize(buffers.scratch.size());
    for (size_t particle = 0; particle < buffers.rendered_particle_ids.size(); ++particle)
    {
        const uint64_t id = buffers.rendered_particle_ids[particle];
        const std::array<std::array<float, 3>, 4>* previous_particle = nullptr;
        if (previous && id)
        {
            const auto found = previous->particle_vertices.find(id);
            if (found != previous->particle_vertices.end()) previous_particle = &found->second;
        }
        const bool matched = previous_particle != nullptr;
        for (size_t corner = 0; corner < 4; ++corner)
        {
            LevelVertex vertex = buffers.scratch[particle * 4 + corner];
            if (matched)
                std::copy((*previous_particle)[corner].begin(), (*previous_particle)[corner].end(), vertex.position);
            // This otherwise unused component tells the particle shader whether the
            // previous PAPI slot belonged to the same particle.
            vertex.normal[0] = matched ? 1.f : -1.f;
            previous_vertices[particle * 4 + corner] = vertex;
        }
    }
    std::string error;
    if (!buffers.previous_vertices.write(0, previous_vertices.data(),
            previous_vertices.size() * sizeof(LevelVertex), error))
        return false;
    float base_rows[8]{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
    if (matrices_.enabled[0])
    {
        const auto matrix = matrices_.values[0].evaluate(Device.fTimeGlobal,
            reinterpret_cast<const float*>(&Device.mView));
        for (size_t row = 0; row < 2; ++row)
            std::copy(matrix[row].begin(), matrix[row].end(), base_rows + row * 4);
    }
    pass.draw_particle_velocity(frame, buffers.vertices.handle(), buffers.previous_vertices.handle(),
        buffers.indices.handle(), buffers.dynamic_occluder_index_count, mvp, previous_mvp, material_,
        float(alpha_ref_) / 255.f, base_rows);
    return true;
}

bool VulkanParticleEffect::prepare_motion_geometry(const FrameRecordingContext& frame, const DeferredPass& pass,
    const float (&mvp)[16], const Fvector& right, const Fvector& up, bool hud,
    bool gpu_draw_active, std::string& error)
{
    if (hud || (!gpu_draw_active && !material_) || (gpu_draw_active && gpu_draw_compatible())) return true;
    if (!screen_set_ && !distortion_ && blend_mode_ == 0) return true;
    const GeometryPhase phase = distortion_ ? GeometryPhase::Distortion : GeometryPhase::Transparent;
    return record(frame, pass, mvp, right, up, hud, error, phase, true);
}

bool VulkanParticleEffect::record_particle_motion(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
    const float (&mvp)[16], const float (&previous_mvp)[16], bool history_valid, bool hud, bool gpu_draw_active)
{
    if (hud || !material_ || (!screen_set_ && !distortion_ && blend_mode_ == 0) ||
        (gpu_draw_active && gpu_draw_compatible()) || effect_ < 0 ||
        PAPI::ParticleManager()->GetParticlesCount(effect_) == 0) return true;
    if (frame.frame_index >= buffers_.size()) return false;
    Buffers& buffers = buffers_[frame.frame_index];
    if (!buffers.particle_motion_ready || !buffers.particle_motion_index_count ||
        !buffers.vertices.handle() || !buffers.previous_vertices.handle() || !buffers.indices.handle()) return false;
    const Buffers* previous = history_valid ? &buffers_[frame.frame_index ^ 1u] : nullptr;
    if (previous && (!previous->particle_motion_ready || previous->particle_frame + 1 != Device.dwFrame))
        previous = nullptr;
    auto& previous_vertices = buffers.previous_scratch;
    previous_vertices.resize(buffers.scratch.size());
    for (size_t particle = 0; particle < buffers.rendered_particle_ids.size(); ++particle)
    {
        const uint64_t id = buffers.rendered_particle_ids[particle];
        const std::array<std::array<float, 3>, 4>* previous_particle = nullptr;
        if (previous && id)
        {
            const auto found = previous->particle_vertices.find(id);
            if (found != previous->particle_vertices.end()) previous_particle = &found->second;
        }
        const bool matched = previous_particle != nullptr;
        for (size_t corner = 0; corner < 4; ++corner)
        {
            LevelVertex vertex = buffers.scratch[particle * 4 + corner];
            if (matched)
                std::copy((*previous_particle)[corner].begin(), (*previous_particle)[corner].end(), vertex.position);
            vertex.normal[0] = matched ? 1.f : -1.f;
            previous_vertices[particle * 4 + corner] = vertex;
        }
    }
    std::string error;
    if (!buffers.previous_vertices.write(0, previous_vertices.data(),
            previous_vertices.size() * sizeof(LevelVertex), error)) return false;
    float base_rows[8]{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
    if (matrices_.enabled[0])
    {
        const auto matrix = matrices_.values[0].evaluate(Device.fTimeGlobal,
            reinterpret_cast<const float*>(&Device.mView));
        for (size_t row = 0; row < 2; ++row)
            std::copy(matrix[row].begin(), matrix[row].end(), base_rows + row * 4);
    }
    pass.draw_particle_velocity(frame, buffers.vertices.handle(), buffers.previous_vertices.handle(),
        buffers.indices.handle(), buffers.particle_motion_index_count, mvp, previous_mvp, material_,
        0.01f, base_rows, true);
    return true;
}
void VulkanParticleEffect::append_gpu_visible(std::vector<uint32_t>& roots) const
{
    uint32_t index{};
    if (gpu_visible_index(index) &&
        std::find(roots.begin(), roots.end(), index) == roots.end()) roots.push_back(index);
}
void VulkanParticleEffect::gpu_emitter_failed(const std::string& error)
{
    Msg("! [renderer-vulkan] GPU particle emitter '%s': %s", def_->name.c_str(), error.c_str());
    detach_gpu_emitter();
}
void VulkanParticleEffect::SetHudMode(BOOL value)
{
    hud_ = value;
    if (gpu_emitters_)
    {
        std::string error;
        if (!gpu_emitters_->set_hud(gpu_emitter_, value != FALSE, error)) gpu_emitter_failed(error);
    }
}

bool VulkanParticleEffect::initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch, GameTextureFactory& textures, DeferredPass& pass,
    std::string& error)
{
    if (effect_ < 0 || actions_ < 0)
    { error = "particle simulator could not allocate effect/actions"; return false; }
    device_ = device;
    memory_ = memory;
    dispatch_ = dispatch;
    textures_ = &textures;
    pass_ = &pass;
    if ((def_->flags & 1u) == 0) return true; // Non-sprite actions still simulate.
    SurfaceMode surface{};
    MaterialDefinition definition;
    if (!textures.surface_mode(def_->shader, def_->texture, surface, error, &alpha_ref_, &blend_mode_,
            true, false, false, false, nullptr, nullptr, &definition))
    {
        error = "particle '" + def_->name + "': " + error;
        release_gpu();
        return false;
    }
    priority_ = definition.priority;
    matrices_ = definition.matrices;
    screen_set_ = particle_uses_screen_program(def_->shader, definition.class_name);
    // This game's Lua special pass overrides its serialized S_SET additive
    // state (gl/particles_xdistort.s). Keep its offset map out of color ADD.
    distortion_ = def_->shader == "particles\\xdistort";
    water_contact_ = distortion_ && is_water_contact_effect(def_->name);
    if (distortion_)
        Msg("[renderer-vulkan] particle.distortion definition='%s' route=scene-refraction", def_->name.c_str());
    for (bool hud : {false, true})
    {
        if (distortion_) break;
        RenderPassGraph graph;
        const bool deferred_set = !hud && !screen_set_ && blend_mode_ == 0;
        if (!build_render_pass_graph({RenderConsumer::Particle,
                deferred_set ? SurfaceMode::AlphaTest : SurfaceMode::Transparent,
                hud, false, false, false, 0, blend_mode_, 0, false, -1, screen_set_}, graph, error) ||
            !pass.request_game_pipeline(graph.passes[0].vertex, graph.passes[0].fragment,
                hud ? SurfaceMode::Opaque : deferred_set ? SurfaceMode::AlphaTest :
                    SurfaceMode::Transparent, hud, false, error))
        {
            error = "particle '" + def_->name + "' pass=" + (hud ? "HUD" : "world") + ": " + error;
            release_gpu();
            return false;
        }
    }
    if (!textures.material(def_->texture, pass, material_, error,
            (definition.class_name == "S_SET   " || definition.class_name == "PARTICLE") &&
            definition.pass.clamp))
    {
        release_gpu();
        return false;
    }
    material_name_ = def_->texture;
    return true;
}

void VulkanParticleEffect::release_gpu()
{
    for (auto& buffers : buffers_)
    {
        buffers.vertices.destroy();
        buffers.indices.destroy();
        buffers.capacity = 0;
    }
    if (material_ && textures_ && pass_)
        textures_->release_material(material_, *pass_);
    material_ = VK_NULL_HANDLE;
    textures_ = nullptr;
    pass_ = nullptr;
    device_ = VK_NULL_HANDLE;
}

void VulkanParticleEffect::UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL transform)
{
    gpu_parent_ = m; gpu_parent_velocity_ = velocity; gpu_parent_local_ = transform != FALSE; gpu_parent_valid_ = true;
    if (gpu_emitters_)
    {
        std::string error;
        if (!gpu_emitters_->set_parent(gpu_emitter_, m, velocity, transform != FALSE, error)) gpu_emitter_failed(error);
    }
    set_water_plane(m);
    initial_position_.set(m.c);
    if (transform) transform_.set(m);
    else
    {
        transform_.identity();
        if (actions_ >= 0) PAPI::ParticleManager()->Transform(actions_, m, velocity);
    }
}

void VulkanParticleEffect::project_water_center(Fvector& center) const
{
    const auto& origin = water_plane_.c;
    const auto normal = water_plane_normal(water_plane_);
    const auto projected = project_particle_to_water({center.x, center.y, center.z},
        {origin.x, origin.y, origin.z}, normal);
    center.set(projected[0], projected[1], projected[2]);
}

void VulkanParticleEffect::Play()
{
    if (effect_ < 0 || actions_ < 0) return;
    elapsed_ = 0.f;
    remainder_ = 0;
    deferred_ = false;
    playing_ = true;
    PAPI::ParticleManager()->PlayEffect(effect_, actions_);
    if (gpu_emitters_)
    {
        std::string error;
        if (!gpu_emitters_->play(gpu_emitter_, error)) gpu_emitter_failed(error);
    }
}

void VulkanParticleEffect::Stop(BOOL deferred)
{
    if (effect_ < 0 || actions_ < 0) return;
    PAPI::ParticleManager()->StopEffect(effect_, actions_, deferred != FALSE);
    if (gpu_emitters_)
    {
        std::string error;
        if (!gpu_emitters_->stop(gpu_emitter_, deferred != FALSE, error)) gpu_emitter_failed(error);
    }
    deferred_ = deferred != FALSE;
    if (!deferred_) playing_ = false;
}

u32 VulkanParticleEffect::ParticlesCount()
{
    return effect_ < 0 ? 0 : PAPI::ParticleManager()->GetParticlesCount(effect_);
}

void VulkanParticleEffect::OnFrame(u32 dt)
{
    if (!playing_) { bounds(visibility_, initial_position_); return; }
    remainder_ = std::min<u32>(remainder_ + std::min<u32>(dt, 1000), 1000);
    const u32 steps = std::min<u32>(remainder_ / step_ms, 3);
    remainder_ %= step_ms;
    for (u32 i = 0; i < steps; ++i)
    {
        if (def_->time_limit >= 0.f && !deferred_ && (elapsed_ += step_seconds) > def_->time_limit)
            Stop(TRUE);
        PAPI::ParticleManager()->Update(effect_, actions_, step_seconds);
        if (gpu_emitters_)
        {
            std::string error;
            if (!gpu_emitters_->tick(gpu_emitter_, step_ms, error)) gpu_emitter_failed(error);
        }
        if (def_->flags & (1u << 11))
        {
            PAPI::Particle* particles = nullptr;
            u32 count = 0;
            PAPI::ParticleManager()->GetParticles(effect_, particles, count);
            for (u32 index = 0; index < count; ++index)
            {
                float frame = particles[index].frame / 255.f +
                    (particles[index].flags.is(PAPI::Particle::ANIMATE_CCW) ? -1.f : 1.f) *
                    def_->frame_rate * step_seconds;
                frame = std::fmod(frame, float(def_->frame_count));
                if (frame < 0.f) frame += def_->frame_count;
                particles[index].frame = static_cast<u16>(frame * 255.f);
            }
        }
    }
    PAPI::Particle* particles = nullptr;
    u32 count = 0;
    PAPI::ParticleManager()->GetParticles(effect_, particles, count);
    if (deferred_ && count == 0 && (!gpu_emitters_ || !gpu_emitters_->draining(gpu_emitter_)))
    {
        deferred_ = false;
        playing_ = false;
    }
    if (!count || !particles) { bounds(visibility_, initial_position_); return; }
    visibility_.box.invalidate();
    float radius = 0.f;
    for (u32 i = 0; i < count; ++i)
    {
        Fvector position;
        transform_.transform_tiny(position, particles[i].pos);
        if (water_contact_)
            project_water_center(position);
        visibility_.box.modify(position);
        radius = std::max(radius, std::max(particles[i].size.x, particles[i].size.y));
    }
    visibility_.box.grow(radius);
    visibility_.box.getsphere(visibility_.sphere.P, visibility_.sphere.R);
}

bool VulkanParticleEffect::record(const FrameRecordingContext& frame, const DeferredPass& pass,
    const float (&mvp)[16], const Fvector& right, const Fvector& up, bool hud, std::string& error,
    GeometryPhase phase, bool prepare_only)
{
    if (frame.frame_index < buffers_.size())
    {
        buffers_[frame.frame_index].dynamic_occluder_ready = false;
        buffers_[frame.frame_index].dynamic_occluder_index_count = 0;
        buffers_[frame.frame_index].particle_motion_ready = false;
        buffers_[frame.frame_index].particle_motion_index_count = 0;
    }
    if (!material_ || !device_) return true;
    // Refraction samples a snapshot of the completed transparent world.
    // Ordinary geometry phases must not consume the earlier opaque snapshot.
    if (distortion_ ? (hud ? phase != GeometryPhase::Transparent && phase != GeometryPhase::Hud
                          : phase != GeometryPhase::Distortion)
                    : phase == GeometryPhase::Distortion) return true;
    const bool deferred_set = !hud && !screen_set_ && blend_mode_ == 0;
    if (!distortion_ && !particle_records_in_phase(hud, screen_set_, blend_mode_, phase)) return true;
    PAPI::Particle* particles = nullptr;
    u32 count = 0;
    PAPI::ParticleManager()->GetParticles(effect_, particles, count);
    if (!count) return true;
    if (!particles || frame.frame_index >= buffers_.size() || count > def_->max_particles)
    { error = "invalid particle frame/count"; return false; }
    auto& buffers = buffers_[frame.frame_index];
    if (count > buffers.capacity)
    {
        // FrameContext waited for this frame slot's fence before invoking the recorder.
        const size_t capacity = std::min<size_t>(def_->max_particles,
            std::max<size_t>(count, buffers.capacity ? buffers.capacity * 2 : 16));
        BufferResource vertices, previous_vertices, indices;
        if (!vertices.initialize(device_, capacity * 4 * sizeof(LevelVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory_, dispatch_, error) ||
            !previous_vertices.initialize(device_, capacity * 4 * sizeof(LevelVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory_, dispatch_, error) ||
            !indices.initialize(device_, capacity * 6 * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory_, dispatch_, error)) return false;
        std::vector<uint32_t> quad_indices(capacity * 6);
        constexpr uint32_t corners[]{0, 1, 2, 0, 2, 3};
        for (size_t particle = 0; particle < capacity; ++particle)
            for (size_t corner = 0; corner < 6; ++corner)
                quad_indices[particle * 6 + corner] = uint32_t(particle * 4) + corners[corner];
        if (!indices.write(0, quad_indices.data(), quad_indices.size() * sizeof(uint32_t), error))
            return false;
        buffers.vertices = std::move(vertices);
        buffers.previous_vertices = std::move(previous_vertices);
        buffers.indices = std::move(indices);
        buffers.capacity = capacity;
    }
    auto &vertices = buffers.scratch;
    vertices.resize(count * 4);
    for (u32 i = 0; i < count; ++i)
    {
        const auto& p = particles[i];
        Fvector center = particle_world_position(p);
        float sx = p.size.x * .5f, sy = p.size.y * .5f;
        Fvector sprite_right=right,sprite_up=up;
        const Fvector velocity{p.vel.x,p.vel.y,p.vel.z};
        const float speed=velocity.magnitude();
        if(def_->flags & (1u<<18))
        { sx+=speed*def_->velocity_scale[0];sy+=speed*def_->velocity_scale[1]; }
        if (water_contact_ && !hud)
        {
            project_water_center(center);
            const auto normal = water_plane_normal(water_plane_);
            const auto basis = particle_face_basis(normal);
            sprite_right.set(basis.right[0], basis.right[1], basis.right[2]);
            sprite_up.set(basis.up[0], basis.up[1], basis.up[2]);
        }
        else if (def_->flags & (1u << 15))
        {
            if (speed < EPS_S && (def_->flags & (1u << 20)))
            {
                Fmatrix basis;
                basis.setXYZ(def_->align_rotation[0], def_->align_rotation[1], def_->align_rotation[2]);
                basis.mulA_43(transform_);
                sprite_right = basis.i;
                sprite_up = basis.k;
            }
            else if (speed >= EPS_S && (def_->flags & (1u << 21)))
            {
                const auto basis = particle_face_basis({velocity.x / speed, velocity.y / speed, velocity.z / speed});
                const Fvector local_right{basis.right[0], basis.right[1], basis.right[2]};
                const Fvector local_up{basis.up[0], basis.up[1], basis.up[2]};
                transform_.transform_dir(sprite_right, local_right);
                transform_.transform_dir(sprite_up, local_up);
            }
            else
            {
                Fvector direction, transformed;
                if (speed >= EPS_S)
                    direction.div(velocity, speed);
                else
                    direction.setHP(-def_->align_rotation[1], -def_->align_rotation[0]);
                transform_.transform_dir(transformed, direction);
                sprite_up = transformed;
                sprite_right.crossproduct(sprite_up, Device.vCameraDirection).normalize_safe();
            }
        }
        const auto [u0, v0] = particle_sprite_uv_origin(p.frame, def_->frame_count,
            def_->frame_columns, def_->frame_size);
        const float du = def_->frame_size[0], dv = def_->frame_size[1];

        for (u32 corner = 0; corner < 4; ++corner)
        {
            const auto sprite = particle_sprite_corner(corner, sx, sy, p.rot.x);
            auto& v = vertices[i * 4 + corner];
            const float rotated_x = sprite.right, rotated_y = sprite.up;
            v.position[0] = center.x + sprite_right.x * rotated_x + sprite_up.x * rotated_y;
            v.position[1] = center.y + sprite_right.y * rotated_x + sprite_up.y * rotated_y;
            v.position[2] = center.z + sprite_right.z * rotated_x + sprite_up.z * rotated_y;
            v.normal[0] = 0.f; v.normal[1] = 1.f; v.normal[2] = 0.f;
            v.uv[0] = u0 + sprite.u * du;
            v.uv[1] = v0 + sprite.v * dv;
            // FVF::LIT sends the simulation's ARGB tint to the particle
            // shader. Ignoring it turns a translucent muzzle sprite into a
            // large opaque patch and loses its animated fade-out.
            v.baked[0] = float((p.color >> 16) & 255) / 255.f;
            v.baked[1] = float((p.color >> 8) & 255) / 255.f;
            v.baked[2] = float(p.color & 255) / 255.f;
            v.baked[3] = float((p.color >> 24) & 255) / 255.f;
        }
    }
    if (!buffers.vertices.write(0, vertices.data(), vertices.size() * sizeof(LevelVertex), error))
        return false;
    buffers.rendered_particle_ids.resize(count);
    buffers.particle_vertices.clear();
    for (u32 i = 0; i < count; ++i)
    {
        const uint64_t id = i < particle_ids_.size() ? particle_ids_[i] : 0;
        buffers.rendered_particle_ids[i] = id;
        if (!id) continue;
        std::array<std::array<float, 3>, 4> snapshot{};
        for (size_t corner = 0; corner < 4; ++corner)
            std::copy_n(vertices[i * 4 + corner].position, 3, snapshot[corner].data());
        buffers.particle_vertices.emplace(id, std::move(snapshot));
    }
    buffers.particle_frame = Device.dwFrame;
    if (deferred_set)
    {
        buffers.dynamic_occluder_ready = true;
        buffers.dynamic_occluder_index_count = count * 6;
    }
    buffers.particle_motion_ready = true;
    buffers.particle_motion_index_count = count * 6;
    if (prepare_only) return true;
    if (distortion_)
        return pass.record_distortion(frame, buffers.vertices.handle(), buffers.indices.handle(),
            count * 6, mvp, material_, phase == GeometryPhase::Distortion);
    RenderPassGraph graph;
    if (!build_render_pass_graph({RenderConsumer::Particle,
            deferred_set ? SurfaceMode::AlphaTest : SurfaceMode::Transparent,
            hud, false, false, false, 0, blend_mode_, 0, false, -1, screen_set_}, graph, error)) return false;
    const auto& node = graph.passes[0];
    const char* vertex = node.vertex.c_str();
    const char* fragment = node.fragment.c_str();
    return deferred_set ? pass.record_geometry(frame, buffers.vertices.handle(), buffers.indices.handle(),
            count * 6, mvp, material_, SurfaceMode::AlphaTest, 0, vertex, fragment,
            float(alpha_ref_) / 255.f, nullptr, 1.f, nullptr, nullptr, 0, &matrices_) :
        hud ? pass.record_hud(frame, buffers.vertices.handle(), buffers.indices.handle(),
            count * 6, mvp, material_, 0, vertex, fragment, float(alpha_ref_) / 255.f,
            nullptr, 0, &matrices_) :
        pass.record_transparent(frame, buffers.vertices.handle(), buffers.indices.handle(),
            count * 6, mvp, material_, 0, vertex, fragment, nullptr, float(alpha_ref_) / 255.f,
            nullptr, 1.f, 0, &matrices_);
}

VulkanParticleGroup::VulkanParticleGroup(std::shared_ptr<const ParticleCatalog> catalog,
    const ParticleGroupDef& def) : catalog_(std::move(catalog)), def_(&def)
{
    initial_position_.set(0, 0, 0);
    parent_.identity();
    bounds(visibility_, initial_position_);
    for (const auto& item : def.effects)
        effects_.push_back(std::make_unique<VulkanParticleEffect>(catalog_, *catalog_->effect(item.effect)));
    entry_phases_.resize(effects_.size(), ParticleGroupEntryPhase::Waiting);
    related_.resize(effects_.size());
    for (size_t i = 0; i < effects_.size(); ++i)
        effects_[i]->set_events([this, i](bool birth, PAPI::Particle& particle, u32 index)
        { on_particle(i, birth, particle, index); });
}

VulkanParticleGroup::~VulkanParticleGroup()
{
    detach_gpu_emitters();
    for (auto& effect : effects_) effect->set_events({});
}

void VulkanParticleGroup::visit_effects(const std::function<void(const VulkanParticleEffect&)>& visit) const
{
    for (const auto& effect : effects_)
        visit(*effect);
    for (const auto& child : triggered_)
        visit(*child.effect);
}

bool VulkanParticleGroup::attach_gpu_emitters(GpuParticleEmitters& emitters, std::string& error)
{
    if (playing_ || !triggered_.empty()) { error = "GPU emitters must be attached before the group starts"; return false; }
    for (const auto& effect : effects_)
        if (effect->gpu_emitter_handle()) { error = "particle group already has GPU emitters"; return false; }
    struct Children { uint32_t birth{GPU_PAPI_INVALID}, play{GPU_PAPI_INVALID}, death{GPU_PAPI_INVALID}; };
    std::vector<Children> children(def_->effects.size());
    for (size_t i = 0; i < def_->effects.size(); ++i)
    {
        const auto& item = def_->effects[i];
        const auto resolve = [&](uint32_t bit, const std::string& name, uint32_t& result) {
            if (!(item.flags & bit)) return true;
            const auto* child = catalog_->effect(name);
            if (!child) { error = "missing GPU particle group child: " + name; return false; }
            if ((bit == 32u || bit == 64u) && !(child->flags & (1u << 14)))
            { error = "unlimited GPU particle group child: " + name; return false; }
            return emitters.register_program(*child, result, error);
        };
        if (!resolve(32u, item.on_birth, children[i].birth) || !resolve(2u, item.on_play, children[i].play) ||
            !resolve(64u, item.on_dead, children[i].death)) return false;
    }
    for (size_t i = 0; i < effects_.size(); ++i)
    {
        if (!effects_[i]->attach_gpu_emitter(emitters, error) ||
            !emitters.children(effects_[i]->gpu_emitter_handle(), children[i].birth, children[i].play,
                children[i].death, def_->effects[i].flags, error))
        { detach_gpu_emitters(); return false; }
    }
    error.clear(); return true;
}
void VulkanParticleGroup::detach_gpu_emitters()
{
    for (auto& effect : effects_) effect->detach_gpu_emitter();
}
void VulkanParticleGroup::append_gpu_visible(std::vector<uint32_t>& roots) const
{
    for (const auto& effect : effects_) effect->append_gpu_visible(roots);
}

bool VulkanParticleGroup::initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch, GameTextureFactory& textures, DeferredPass& pass,
    std::string& error)
{
    device_ = device;
    memory_ = memory;
    dispatch_ = dispatch;
    textures_ = &textures;
    pass_ = &pass;
    for (auto& effect : effects_)
        if (!effect->initialize(device, memory, dispatch, textures, pass, error))
        {
            OnDeviceDestroy();
            return false;
        }
    return true;
}

void VulkanParticleGroup::OnDeviceDestroy()
{
    detach_gpu_emitters();
    // Triggered children may have been created after initialize(). Release
    // them before the catalog/pass leases are detached; a later initialize()
    // must start with only the definition's persistent effects.
    for (auto& item : triggered_) item.effect->set_events({});
    triggered_.clear();
    pending_.clear();
    for (auto& roots : related_) roots.clear();
    for (auto& effect : effects_) effect->OnDeviceDestroy();
    textures_ = nullptr;
    pass_ = nullptr;
    device_ = VK_NULL_HANDLE;
}

void VulkanParticleGroup::UpdateParent(const Fmatrix& m, const Fvector& velocity, BOOL transform)
{
    initial_position_.set(m.c);
    parent_.set(m);
    transform_parent_ = transform != FALSE;
    for (auto& effect : effects_) effect->UpdateParent(m, velocity, transform);
}

void VulkanParticleGroup::Play()
{
    time_ = 0.f;
    deferred_ = false;
    playing_ = true;
    std::fill(entry_phases_.begin(), entry_phases_.end(), ParticleGroupEntryPhase::Waiting);
    for (auto& effect : effects_) effect->Stop(FALSE);
    for (auto& item : triggered_) item.effect->Stop(FALSE);
    pending_.clear();
    for (auto& child : triggered_) child.related = false;
    for (auto& roots : related_) roots.clear();
}

void VulkanParticleGroup::Stop(BOOL deferred)
{
    deferred_ = deferred != FALSE;
    if (!deferred_) playing_ = false;
    for (auto& effect : effects_) effect->Stop(deferred);
    for (auto& item : triggered_) item.effect->Stop(deferred);
    if (!deferred)
    {
        pending_.clear();
        for (auto& child : triggered_) child.related = false;
        for (auto& roots : related_) roots.clear();
    }
}

void VulkanParticleGroup::OnFrame(u32 dt)
{
    if (!playing_) { bounds(visibility_, initial_position_); return; }
    const float next = time_ + dt / 1000.f;
    for (size_t i = 0; i < effects_.size(); ++i)
    {
        auto& effect = *effects_[i];
        const auto& item = def_->effects[i];
        if ((item.flags & enabled) && !deferred_)
        {
            auto& phase = entry_phases_[i];
            const auto next_phase = advance_particle_group_entry(phase, next, item.begin, item.end);
            if (phase != next_phase)
            {
                phase = next_phase;
                if (phase == ParticleGroupEntryPhase::Emitting)
                    effect.Play();
                else if (effect.IsPlaying())
                    effect.Stop(item.flags & deferred_stop);
            }
        }
        effect.OnFrame(dt);
    }
    for (const auto& event : pending_)
    {
        auto& roots = related_[event.root];
        if (event.kind == Pending::Kind::RelatedDeath)
        {
            if (event.particle_index < roots.size())
            {
                const size_t child = roots[event.particle_index];
                if (child < triggered_.size()) { triggered_[child].effect->Stop(TRUE); triggered_[child].related = false; }
                roots[event.particle_index] = roots.back(); roots.pop_back();
            }
        }
        else
        {
            const bool is_related = event.kind == Pending::Kind::RelatedBirth;
            const size_t child = spawn(event.name, event.position, event.velocity, is_related, event.root);
            if (is_related) roots.push_back(child);
        }
    }
    pending_.clear();
    for (size_t root = 0; root < effects_.size(); ++root)
    {
        PAPI::Particle* particles = nullptr; u32 count = 0;
        effects_[root]->particles(particles, count);
        for (size_t i = 0; particles && i < count && i < related_[root].size(); ++i)
        {
            const size_t child_index = related_[root][i];
            if (child_index >= triggered_.size()) continue;
            auto& child = *triggered_[child_index].effect;
            Fvector position, velocity;
            if (transform_parent_) parent_.transform_tiny(position, particles[i].pos);
            else position.set(particles[i].pos);
            velocity.sub(particles[i].pos, particles[i].posB); velocity.div(step_seconds);
            if (transform_parent_) parent_.transform_dir(velocity);
            Fmatrix frame = transform_parent_ ? parent_ : Fidentity; frame.c.set(position);
            child.UpdateParent(frame, velocity, FALSE);
            if (!deferred_ && !child.IsPlaying() && (def_->effects[root].flags & (1u << 4))) child.Play();
        }
    }
    for (auto& child : triggered_) if (child.effect->IsPlaying()) child.effect->OnFrame(dt);
    time_ = next;
    if (!deferred_ && def_->time_limit > 0.f && time_ > def_->time_limit) Stop(TRUE);
    if (deferred_)
    {
        // A newly started child can be active before its first fixed simulation
        // step creates particles. Keep advancing the group until it drains.
        bool active = false;
        visit_effects([&](const VulkanParticleEffect& effect)
        {
            active = active || effect.simulation_active();
        });
        if (!active)
        {
            playing_ = false;
            deferred_ = false;
        }
    }
    visibility_.box.invalidate();
    for (const auto& effect : effects_)
    {
        visibility_.box.modify(effect->getVisData().box.vMin);
        visibility_.box.modify(effect->getVisData().box.vMax);
    }
    for (const auto& child : triggered_)
    {
        visibility_.box.modify(child.effect->getVisData().box.vMin);
        visibility_.box.modify(child.effect->getVisData().box.vMax);
    }
    if (effects_.empty() && triggered_.empty()) bounds(visibility_, initial_position_);
    else visibility_.box.getsphere(visibility_.sphere.P, visibility_.sphere.R);
}

u32 VulkanParticleGroup::ParticlesCount()
{
    u32 count = 0;
    for (auto& effect : effects_) count += effect->ParticlesCount();
    for (auto& child : triggered_) count += child.effect->ParticlesCount();
    return count;
}

void VulkanParticleGroup::SetHudMode(BOOL value)
{
    hud_ = value != FALSE;
    for (auto& effect : effects_) effect->SetHudMode(value);
    for (auto& child : triggered_) child.effect->SetHudMode(value);
}

bool VulkanParticleGroup::record(const FrameRecordingContext& frame, const DeferredPass& pass,
    const float (&mvp)[16], const Fvector& right, const Fvector& up, bool hud, std::string& error,
    GeometryPhase phase, int priority_filter, const std::function<bool(VulkanParticleEffect&)>& gpu_record,
    const std::function<bool(VulkanParticleEffect&, VulkanParticleEffect&)>& gpu_child_record)
{
    for (auto& effect : effects_)
        if ((priority_filter < 0 || effect->priority() == priority_filter) &&
            !(gpu_record && gpu_record(*effect)) &&
            !effect->record(frame, pass, mvp, right, up, hud, error, phase)) return false;
    struct DrawnChild { size_t root; std::string name; bool handled; };
    std::vector<DrawnChild> drawn;
    for (auto& child : triggered_)
    {
        if (priority_filter >= 0 && child.effect->priority() != priority_filter) continue;
        bool handled = false;
        if (gpu_child_record && child.root < effects_.size())
        {
            auto found = std::find_if(drawn.begin(), drawn.end(), [&](const DrawnChild& item) {
                return item.root == child.root && item.name == child.name;
            });
            if (found == drawn.end())
            {
                handled = gpu_child_record(*effects_[child.root], *child.effect);
                drawn.push_back({child.root, child.name, handled});
            }
            else handled = found->handled;
        }
        if (!handled && !child.effect->record(frame, pass, mvp, right, up, hud, error, phase)) return false;
    }
    return true;
}

bool VulkanParticleGroup::prepare_motion_geometry(const FrameRecordingContext& frame, const DeferredPass& pass,
    const float (&mvp)[16], const Fvector& right, const Fvector& up, bool hud,
    bool gpu_draw_active, std::string& error)
{
    if (hud) return true;
    for (auto& effect : effects_)
        if (!effect->prepare_motion_geometry(frame, pass, mvp, right, up, hud, gpu_draw_active, error))
            return false;
    for (auto& child : triggered_)
    {
        const bool gpu_child = gpu_draw_active && child.root < effects_.size() &&
            effects_[child.root]->gpu_draw_compatible();
        if (!gpu_child && !child.effect->prepare_motion_geometry(frame, pass, mvp, right, up,
                hud, gpu_draw_active, error))
            return false;
    }
    return true;
}

bool VulkanParticleGroup::record_particle_motion(const FrameRecordingContext& frame,
    const ObjectVelocityPass& pass, const float (&mvp)[16], const float (&previous_mvp)[16],
    bool history_valid, bool hud, bool gpu_draw_active, bool transparent_only, int priority_filter,
    const std::function<bool(VulkanParticleEffect&)>& gpu_record,
    const std::function<bool(VulkanParticleEffect&, VulkanParticleEffect&)>& gpu_child_record)
{
    if (hud) return true;
    for (const auto& effect : effects_)
        if (priority_filter < 0 || effect->priority() == priority_filter)
        {
            const bool gpu_handled = gpu_record && gpu_record(*effect);
            if (!gpu_handled && transparent_only &&
                !effect->record_particle_motion(frame, pass, mvp, previous_mvp, history_valid, hud,
                    gpu_record ? false : gpu_draw_active)) return false;
        }
    struct DrawnChild { size_t root; std::string name; bool handled; };
    std::vector<DrawnChild> drawn;
    for (const auto& child : triggered_)
    {
        if (priority_filter >= 0 && child.effect->priority() != priority_filter) continue;
        const bool gpu_child = gpu_draw_active && child.root < effects_.size() &&
            effects_[child.root]->gpu_draw_compatible();
        bool gpu_handled = false;
        if (gpu_child_record && child.root < effects_.size())
        {
            auto found = std::find_if(drawn.begin(), drawn.end(), [&](const DrawnChild& item) {
                return item.root == child.root && item.name == child.name;
            });
            if (found == drawn.end())
            {
                gpu_handled = gpu_child_record(*effects_[child.root], *child.effect);
                drawn.push_back({child.root, child.name, gpu_handled});
            }
            else gpu_handled = found->handled;
        }
        if (!gpu_handled && transparent_only &&
            !child.effect->record_particle_motion(frame, pass, mvp, previous_mvp, history_valid, hud,
                gpu_child_record ? false : gpu_child ? true : gpu_draw_active)) return false;
    }
    return true;
}

bool VulkanParticleGroup::supports_dynamic_occluder_mask(bool hud, bool gpu_draw_active) const
{
    for (const auto& effect : effects_)
        if (!effect->supports_dynamic_occluder_mask(hud, gpu_draw_active)) return false;
    for (const auto& child : triggered_)
        if (!child.effect->supports_dynamic_occluder_mask(hud, gpu_draw_active)) return false;
    return true;
}

bool VulkanParticleGroup::record_dynamic_occluder_mask(const FrameRecordingContext& frame,
    const ObjectVelocityPass& pass, const float (&mvp)[16], const float (&previous_mvp)[16],
    bool history_valid, bool hud, bool gpu_draw_active)
{
    for (const auto& effect : effects_)
        if (!effect->record_dynamic_occluder_mask(frame, pass, mvp, previous_mvp, history_valid,
                hud, gpu_draw_active)) return false;
    for (const auto& child : triggered_)
        if (!child.effect->record_dynamic_occluder_mask(frame, pass, mvp, previous_mvp, history_valid,
                hud, gpu_draw_active)) return false;
    return true;
}

void VulkanParticleGroup::invalidate_dynamic_occluder_history(uint32_t frame_index)
{
    for (auto& effect : effects_) effect->invalidate_dynamic_occluder_history(frame_index);
    for (auto& child : triggered_) child.effect->invalidate_dynamic_occluder_history(frame_index);
}

bool VulkanParticleEffect::has_active_distortion() const
{
    return distortion_ && material_ && effect_ >= 0 &&
        PAPI::ParticleManager()->GetParticlesCount(effect_) != 0;
}

bool VulkanParticleGroup::has_distortion() const
{
    for (const auto& effect : effects_) if (effect->has_active_distortion()) return true;
    for (const auto& child : triggered_) if (child.effect->has_active_distortion()) return true;
    return false;
}

bool VulkanParticleGroup::has_priority(uint8_t priority) const
{
    for (const auto& effect : effects_)
        if (effect->priority() == priority) return true;
    for (const auto& child : triggered_)
        if (child.effect->priority() == priority) return true;
    return false;
}

void VulkanParticleGroup::on_particle(size_t index, bool birth, PAPI::Particle& particle, u32 particle_index)
{
    const auto& item = def_->effects[index];
    Fvector position;
    if (transform_parent_) parent_.transform_tiny(position, particle.pos);
    else position.set(particle.pos);
    Fvector velocity;
    velocity.sub(particle.pos, particle.posB);
    velocity.div(step_seconds);
    if (transform_parent_)
        parent_.transform_dir(velocity);
    if (birth && (item.flags & (1u << 5)) && !item.on_birth.empty())
        pending_.push_back({item.on_birth, position, velocity, index, particle_index, Pending::Kind::Free});
    if ((item.flags & (1u << 1)) && !item.on_play.empty())
        pending_.push_back({item.on_play, position, velocity, index, particle_index,
            birth ? Pending::Kind::RelatedBirth : Pending::Kind::RelatedDeath});
    if (!birth && (item.flags & (1u << 6)) && !item.on_dead.empty())
        pending_.push_back({item.on_dead, position, velocity, index, particle_index, Pending::Kind::Free});
}

size_t VulkanParticleGroup::spawn(const std::string& name, const Fvector& position,
    const Fvector& velocity, bool related, size_t root)
{
    const auto* def = catalog_->effect(name);
    if (!def) return SIZE_MAX;
    VulkanParticleEffect* effect = nullptr;
    size_t child_index = 0;
    for (; child_index < triggered_.size(); ++child_index)
        if (auto& child = triggered_[child_index]; child.name == name && !child.related &&
            !child.effect->IsPlaying() && !child.effect->ParticlesCount())
        { effect = child.effect.get(); child.related = related; child.root = root; break; }
    if (!effect)
    {
        auto child = std::make_unique<VulkanParticleEffect>(catalog_, *def);
        if (device_)
        {
            std::string error;
            if (!child->initialize(device_, memory_, dispatch_, *textures_, *pass_, error))
            {
                Msg("! [renderer-vulkan] particle child '%s': %s", name.c_str(), error.c_str());
                return SIZE_MAX;
            }
        }
        effect = child.get();
        child_index = triggered_.size();
        triggered_.push_back({name, std::move(child), related, root});
    }
    // Child actions inherit the emitter frame, as in ParticleGroup::StartFreeChild.
    // Translation alone loses the surface orientation of water impact emitters.
    Fmatrix parent = transform_parent_ ? parent_ : Fidentity;
    parent.c.set(position);
    effect->SetHudMode(hud_);
    effect->UpdateParent(parent, velocity, FALSE);
    effect->set_water_plane(parent_);
    effect->Play();
    return child_index;
}
} // namespace xray::render::vulkan
