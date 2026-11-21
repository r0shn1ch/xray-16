#include "xrEngine/stdafx.h"
#include "VulkanGameDevice.h"
#include "ShadowStability.h"
#include "ParticleVisual.h"
#include "VulkanRainRender.h"
#include "VulkanThunderboltRender.h"
#include "SceneShaders.h"
#include "ShaderModule.h"
#include "PostProcessShaders.h"
#include "TransparentDrawOrder.h"
#include "VulkanGameLightingMath.h"
#include "VulkanSettings.h"
#include "Include/xrRender/Kinematics.h"
#include "Include/xrRender/RenderVisual.h"
#include "xrCDB/Frustum.h"
#include "xrEngine/x_ray.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#if defined(XR_PLATFORM_ANDROID)
#include <SDL_syswm.h>
#include <android/native_window.h>
#endif

namespace xray::render::vulkan
{
namespace
{
// Attribute CPU recording time to actual render passes in the normal device
// log. Aggregate CPU p95 also includes acquisition/fence waits.
struct PassCpuSample
{
    const char *name;
    const DeferredPass &pass;
    uint32_t draws;
    std::chrono::steady_clock::time_point begin{std::chrono::steady_clock::now()};
    ~PassCpuSample()
    {
        if (Device.dwFrame % 300 == 0)
            Msg("[renderer-vulkan] pass.cpu frame=%u pass=%s ms=%.3f draws=%u", Device.dwFrame, name,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count(),
                pass.draw_calls() - draws);
    }
};

} // namespace

bool VulkanGameDevice::record_gpu_particle_effect(const FrameRecordingContext& frame, VulkanParticleEffect& effect,
    const ParticleDraw& draw, GeometryPhase phase, VulkanParticleEffect* root_effect)
{
    VulkanParticleEffect& root_visual = root_effect ? *root_effect : effect;
    if (!gpu_particle_graphics_ || gpu_particle_graphics_failed_ || gpu_particle_failed_ || !gpu_particles_ ||
        frame.frame_index >= gpu_particle_recorded_.size() || !gpu_particle_recorded_[frame.frame_index] ||
        !root_visual.gpu_draw_compatible()) return false;
    const auto* programs = root_visual.gpu_draw_programs();
    uint32_t root{};
    if (!programs || programs->empty() || !root_visual.gpu_visible_index(root)) return false;
    uint32_t program = programs->front();
    if (root_effect)
    {
        const auto& names = gpu_particles_->programs().names();
        const auto effect_name = effect.Name();
        const char* name = effect_name.c_str();
        const auto found = std::find_if(programs->begin(), programs->end(), [&](uint32_t id) {
            return name && id < names.size() && names[id] == name;
        });
        if (found == programs->end()) return false;
        program = *found;
    }
    const auto* material = gpu_particle_materials_.get(program);
    if (!material || !material->resolved || !material->set) return false;
    if (material->distortion ? (draw.hud ? phase != GeometryPhase::Transparent && phase != GeometryPhase::Hud :
        phase != GeometryPhase::Distortion) :
        phase == GeometryPhase::Distortion || !particle_records_in_phase(draw.hud, material->screen, material->blend, phase))
        return true;
    uint32_t bucket{};
    const uint32_t variant = material->distortion ? 2u : material->screen ? 1u : 0u;
    if (!gpu_particles_->draw_bucket(frame.frame_index, root, program, material->blend, variant, draw.hud, bucket,
            root_effect != nullptr))
        return false;
    std::string fragment, error; bool gbuffer{};
    std::array<float, 64> constants{};
    if (!gpu_particle_materials_.fragment(program, draw.hud, fragment, gbuffer, error) ||
        !gpu_particle_materials_.constants(program, Device.fTimeGlobal, &Device.mView.m[0][0], deferred_, constants, error))
        return false;
    GpuParticleDrawParams camera{};
    camera.camera_top = {draw.up.x, draw.up.y, draw.up.z, 0.f};
    camera.camera_right = {draw.right.x, draw.right.y, draw.right.z, 0.f};
    camera.camera_direction = {Device.vCameraDirection.x, Device.vCameraDirection.y, Device.vCameraDirection.z, 0.f};
    camera.eye_position = {Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z, 1.f};
    camera.screen = {1.f / frame.extent.width, 1.f / frame.extent.height, 0.f, 0.f};
    camera.bucket = bucket;
    const VkDescriptorSetLayout scene_layout = draw.hud && material->distortion ? deferred_.water_scene_layout() : VK_NULL_HANDLE;
    const VkDescriptorSet scene = scene_layout ? deferred_.water_scene_set(frame.image_index) : VK_NULL_HANDLE;
    if (scene_layout && !scene) return false;
    if (!draw.hud && effect.gpu_water_contact())
    {
        const auto& plane = effect.gpu_water_plane();
        camera.water_origin = {plane.c.x, plane.c.y, plane.c.z, 1.f};
        camera.water_normal = {plane.k.x, plane.k.y, plane.k.z, 0.f};
    }
    // The particle pipeline changes bindings behind DeferredPass's cache.
    deferred_.invalidate_geometry_cache();
    if (!gpu_particle_graphics_->record(frame, gpu_particles_->buffers(), material->set, fragment,
            material->distortion ? 1u : static_cast<uint32_t>(material->blend), draw.hud && !material->distortion, gbuffer,
            camera, constants.data(), draw.mvp.data(), error, scene_layout, scene))
    {
        Msg("! [renderer-vulkan] GPU particle graphics disabled: %s", error.c_str());
        gpu_particle_graphics_failed_ = true;
        return false;
    }
    if (phase == GeometryPhase::OpaqueAndAlphaTest && effect.needs_dynamic_occluder_mask(draw.hud, true))
        effect.invalidate_dynamic_occluder_history(frame.frame_index);
    return true;
}

bool VulkanGameDevice::record_gpu_particle_motion(const FrameRecordingContext& frame, VulkanParticleEffect& effect,
    const ParticleDraw& draw, const float* current_mvp, const float* previous_mvp, bool transparent,
    VulkanParticleEffect* root_effect)
{
    VulkanParticleEffect& root_visual = root_effect ? *root_effect : effect;
    if (!gpu_particle_graphics_ || gpu_particle_graphics_failed_ || gpu_particle_failed_ || !gpu_particles_ ||
        frame.frame_index >= gpu_particle_recorded_.size() || !gpu_particle_recorded_[frame.frame_index] ||
        !root_visual.gpu_draw_compatible()) return false;
    const auto* programs = root_visual.gpu_draw_programs();
    uint32_t root{};
    if (!programs || programs->empty() || !root_visual.gpu_visible_index(root)) return false;
    uint32_t program = programs->front();
    const bool child = root_effect != nullptr;
    if (child)
    {
        const auto& names = gpu_particles_->programs().names();
        const auto effect_name = effect.Name();
        const char* name = effect_name.c_str();
        const auto found = std::find_if(programs->begin(), programs->end(), [&](uint32_t id) {
            return name && id < names.size() && names[id] == name;
        });
        if (found == programs->end()) return false;
        program = *found;
    }
    const auto* material = gpu_particle_materials_.get(program);
    if (!material || !material->resolved || !material->set) return false;
    const bool gpu_transparent = material->distortion || material->screen || material->blend != 0;
    if (gpu_transparent != transparent) return true;
    const uint32_t variant = material->distortion ? 2u : material->screen ? 1u : 0u;
    uint32_t bucket{};
    if (!gpu_particles_->draw_bucket(frame.frame_index, root, program, material->blend, variant, false,
            bucket, child)) return false;

    GpuParticleDrawParams camera{};
    camera.camera_top = {draw.up.x, draw.up.y, draw.up.z, 0.f};
    camera.camera_right = {draw.right.x, draw.right.y, draw.right.z, 0.f};
    camera.camera_direction = {Device.vCameraDirection.x, Device.vCameraDirection.y,
        Device.vCameraDirection.z, 0.f};
    camera.eye_position = {Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z, 1.f};
    camera.screen = {1.f / frame.extent.width, 1.f / frame.extent.height, 0.f, 0.f};
    camera.bucket = bucket;
    camera.padding[0] = gpu_transparent ? 1.f : 0.f;
    if (effect.gpu_water_contact())
    {
        const auto& plane = effect.gpu_water_plane();
        camera.water_origin = {plane.c.x, plane.c.y, plane.c.z, 1.f};
        camera.water_normal = {plane.k.x, plane.k.y, plane.k.z, 0.f};
    }
    std::array<float, 64> constants{};
    std::string error;
    if (!gpu_particle_materials_.constants(program, Device.fTimeGlobal, &Device.mView.m[0][0], deferred_,
            constants, error)) return false;
    if (gpu_transparent) constants[16] = 0.01f;
    if (!gpu_particle_graphics_->record_velocity(frame, gpu_particles_->buffers(), material->set, camera,
            constants.data(), current_mvp, previous_mvp, gpu_transparent, error))
    {
        Msg("! [renderer-vulkan] GPU particle motion vectors skipped: %s", error.c_str());
        return true;
    }
    return true;
}

void VulkanGameDevice::record_distortion(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    auto command_batch = owner.deferred_.geometry_batch();
    std::vector<const ParticleDraw *> sorted;
    for (const auto &draw : owner.particle_draws_)
        if (!draw.hud && (draw.visual->getType() == 8
                ? static_cast<VulkanParticleEffect *>(draw.visual)->has_active_distortion()
                : static_cast<VulkanParticleGroup *>(draw.visual)->has_distortion()))
            sorted.push_back(&draw);
    std::stable_sort(sorted.begin(), sorted.end(), [](const ParticleDraw *a, const ParticleDraw *b) {
        return a->sort_distance > b->sort_distance;
    });
    for (const auto *entry : sorted)
    {
        const auto &draw = *entry;
        if (draw.visual->getType() == 8 && owner.record_gpu_particle_effect(frame,
                *static_cast<VulkanParticleEffect *>(draw.visual), draw, GeometryPhase::Distortion))
            continue;
        float mvp[16];
        std::copy(draw.mvp.begin(), draw.mvp.end(), mvp);
        const bool ok = draw.visual->getType() == 8
            ? static_cast<VulkanParticleEffect *>(draw.visual)->record(
                  frame, owner.deferred_, mvp, draw.right, draw.up, false, owner.model_error_, GeometryPhase::Distortion)
            : static_cast<VulkanParticleGroup *>(draw.visual)->record(
                  frame, owner.deferred_, mvp, draw.right, draw.up, false, owner.model_error_, GeometryPhase::Distortion, -1,
                  [&](VulkanParticleEffect& effect) {
                      return owner.record_gpu_particle_effect(frame, effect, draw, GeometryPhase::Distortion);
                  }, [&](VulkanParticleEffect& root, VulkanParticleEffect& child) {
                      return owner.record_gpu_particle_effect(frame, child, draw, GeometryPhase::Distortion, &root);
                  });
        owner.models_recorded_ &= ok;
    }
}

bool VulkanGameDevice::capture_ui(VkCommandBuffer command, VkImage image, uint32_t index, void *user,
    std::string& error)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    auto &snapshot = owner.window_.frame().presentation_render_pass() != VK_NULL_HANDLE
        ? owner.ui_snapshot_targets_ : owner.water_targets_;
    if (!snapshot.capture(command, image, index))
    {
        owner.ui_recorded_ = false;
        error = "UI image capture failed";
        return false;
    }
    error.clear();
    return true;
}

void VulkanGameDevice::record_ui_distortion(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    owner.ui_recorded_ &= owner.ui_.record(frame, owner.ui_error_, true, owner.ui_scene_sets_.at(frame.image_index));
}

void VulkanGameDevice::record_ui(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    owner.ui_recorded_ &= owner.ui_.record(
        frame, owner.ui_error_, false, VK_NULL_HANDLE,
        frame.image_index < owner.ui_depth_sets_.size() ? owner.ui_depth_sets_[frame.image_index] : VK_NULL_HANDLE);
    if (!owner.ui_recorded_)
        Msg("! [renderer-vulkan] ui.record failed image=%u frame=%u overlay=%d draws=%u reason=%s", frame.image_index,
            frame.frame_index, frame.render_pass == owner.window_.frame().overlay_render_pass() ? 1 : 0,
            owner.last_ui_draw_calls_, owner.ui_error_.c_str());
}

void VulkanGameDevice::record_postprocess(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    if (frame.render_pass == owner.window_.frame().bloom_render_pass())
        owner.bloom_bright_passes_[frame.image_index]->record(frame);
    else if (frame.render_pass == owner.window_.frame().bloom_upsample_render_pass())
        owner.bloom_upsample_passes_[frame.image_index][frame.mip_level]->record(frame);
    else
    {
        if (owner.fsr3_upscaler_.active() && frame.image_index < owner.fsr3_upscale_passes_.size())
            owner.fsr3_upscale_passes_[frame.image_index]->record(frame);
        else if (owner.temporal_upscale_ && owner.temporal_.active())
            owner.temporal_upscale_passes_[frame.image_index][owner.temporal_.output_slot()]->record(frame);
        else
            owner.postprocess_passes_[frame.image_index]->record(frame);
        if (owner.sun_shafts_active_)
            owner.sun_shaft_passes_[frame.image_index]->record(frame);
    }
}

void VulkanGameDevice::record_display_grade(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    owner.display_grade_passes_[frame.image_index]->record(frame);
}

void VulkanGameDevice::record_level_visuals(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    PassCpuSample timing{"opaque-level", owner.deferred_, owner.deferred_.draw_calls()};
    auto command_batch = owner.deferred_.geometry_batch();
    const bool visibility_ready = frame.frame_index < owner.visibility_recorded_.size() &&
        owner.visibility_recorded_[frame.frame_index] &&
        owner.visibility_targets_[frame.frame_index].extent.width == frame.extent.width &&
        owner.visibility_targets_[frame.frame_index].extent.height == frame.extent.height;
    for (const auto &draw : owner.level_draws_)
    {
        if (draw.hud)
            continue;
        float mvp[16];
        std::copy(draw.mvp.begin(), draw.mvp.end(), mvp);
        const bool use_visibility = visibility_ready && !draw.instance && draw.mvp == owner.scene_mvp_;
        if (!owner.current_level_->record_visual(draw.index, frame, owner.deferred_, mvp,
                GeometryPhase::OpaqueAndAlphaTest, draw.lod, draw.instance == nullptr, false, -1, use_visibility))
        {
            owner.level_recorded_ = false;
            owner.level_error_ = "Vulkan opaque level visual " + std::to_string(draw.index) +
                                 " failed (lod=" + std::to_string(draw.lod) + ")";
            return;
        }
    }
    float mvp[16];
    std::copy(owner.scene_mvp_.begin(), owner.scene_mvp_.end(), mvp);
    if (!owner.current_level_->record_details(frame, owner.deferred_, mvp))
    {
        owner.level_recorded_ = false;
        owner.level_error_ = "Vulkan level detail recording failed";
    }
}

void VulkanGameDevice::record_transparent(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    PassCpuSample timing{"transparent", owner.deferred_, owner.deferred_.draw_calls()};
    auto command_batch = owner.deferred_.geometry_batch();
    if (frame.image_index >= owner.forward_buffers_.size() ||
        !owner.forward_buffers_[frame.image_index].write(0, &owner.forward_lighting_, sizeof(owner.forward_lighting_),
                                                         owner.model_error_))
    {
        owner.models_recorded_ = false;
        return;
    }
    // R2 draws secondary priority ordinary geometry after combine and before
    // the sorted forward list. This pass replaces color and writes depth.
    if (owner.current_level_)
        for (const auto &level : owner.level_draws_)
        {
            if (level.hud || !owner.current_level_->has_secondary_priority(level.index))
                continue;
            float mvp[16];
            std::copy(level.mvp.begin(), level.mvp.end(), mvp);
            auto command_batch = owner.deferred_.geometry_batch();
            if (!owner.current_level_->record_visual(level.index, frame, owner.deferred_, mvp,
                                                     GeometryPhase::Transparent, level.lod, level.instance == nullptr,
                                                     true))
            {
                owner.level_recorded_ = false;
                owner.level_error_ =
                    "Vulkan secondary-priority level visual " + std::to_string(level.index) + " failed";
                return;
            }
        }
    for (auto &model : owner.model_draws_)
    {
        if (model.hud || !model.model->has_secondary_priority())
            continue;
        float mvp[16];
        owner.deferred_.set_model_hemi_negative(model.hemi_negative.data());
        std::copy(model.mvp.begin(), model.mvp.end(), mvp);
        const bool ok =
            model.skeleton
                ? model.model->record_animated(frame, owner.deferred_, mvp, *model.skeleton, owner.model_error_,
                                               GeometryPhase::SecondaryOpaque, model.lod, model.normal_rows.data())
                : model.model->record(frame, owner.deferred_, mvp, nullptr, 0, owner.model_error_,
                                      GeometryPhase::SecondaryOpaque, model.lod, nullptr, model.normal_rows.data());
        if (!ok)
        {
            owner.models_recorded_ = false;
            return;
        }
    }
    std::vector<TransparentDrawRef> draws;
    for (size_t i = 0; i < owner.level_draws_.size(); ++i)
        if (!owner.level_draws_[i].hud && owner.current_level_)
            for (uint8_t priority = 1; priority <= 3; ++priority)
                if (owner.current_level_->has_transparent_priority(owner.level_draws_[i].index, priority))
                    draws.push_back({owner.level_draws_[i].sort_distance, false, i, false, false, priority});
    for (size_t i = 0; i < owner.model_draws_.size(); ++i)
        if (!owner.model_draws_[i].hud)
            for (uint8_t priority = 1; priority <= 3; ++priority)
                if (owner.model_draws_[i].model->has_transparent_priority(priority))
                    draws.push_back({owner.model_draws_[i].sort_distance, true, i, false, false, priority});
    for (size_t i = 0; i < owner.particle_draws_.size(); ++i)
    {
        const auto &particle = owner.particle_draws_[i];
        if (particle.hud)
            continue;
        if (particle.visual->getType() == 8)
            draws.push_back({particle.sort_distance, false, i, true, false,
                             static_cast<VulkanParticleEffect *>(particle.visual)->priority()});
        else
        {
            const auto *group = static_cast<VulkanParticleGroup *>(particle.visual);
            for (uint8_t priority = 1; priority <= 3; ++priority)
                if (group->has_priority(priority))
                    draws.push_back({particle.sort_distance, false, i, true, false, priority});
        }
    }
    if (owner.current_level_)
    {
        float mvp[16];
        std::copy(owner.scene_mvp_.begin(), owner.scene_mvp_.end(), mvp);
        std::vector<float> detail_distances;
        if (!owner.current_level_->prepare_transparent_details(frame, mvp, detail_distances))
        {
            owner.level_recorded_ = false;
            owner.level_error_ = "Vulkan transparent level detail preparation failed";
            return;
        }
        for (size_t i = 0; i < detail_distances.size(); ++i)
            draws.push_back({detail_distances[i], false, i, false, true});
    }
    sort_transparent_draws(draws);
    for (const TransparentDrawRef &draw : draws)
    {
        float mvp[16];
        if (draw.detail)
        {
            std::copy(owner.scene_mvp_.begin(), owner.scene_mvp_.end(), mvp);
            if (!owner.current_level_->record_transparent_detail(draw.index, frame, owner.deferred_, mvp))
            {
                owner.level_recorded_ = false;
                owner.level_error_ = "Vulkan transparent level detail recording failed";
                return;
            }
        }
        else if (draw.particle)
        {
            auto &particle = owner.particle_draws_[draw.index];
            if (particle.visual->getType() == 8 && owner.record_gpu_particle_effect(frame,
                    *static_cast<VulkanParticleEffect *>(particle.visual), particle, GeometryPhase::Transparent))
                continue;
            std::copy(particle.mvp.begin(), particle.mvp.end(), mvp);
            bool ok =
                particle.visual->getType() == 8
                    ? static_cast<VulkanParticleEffect *>(particle.visual)
                          ->record(frame, owner.deferred_, mvp, particle.right, particle.up, false, owner.model_error_)
                    : static_cast<VulkanParticleGroup *>(particle.visual)
                          ->record(frame, owner.deferred_, mvp, particle.right, particle.up, false, owner.model_error_,
                                   GeometryPhase::Transparent, draw.priority, [&](VulkanParticleEffect& effect) {
                                       return owner.record_gpu_particle_effect(frame, effect, particle, GeometryPhase::Transparent);
                                   }, [&](VulkanParticleEffect& root, VulkanParticleEffect& child) {
                                       return owner.record_gpu_particle_effect(frame, child, particle, GeometryPhase::Transparent, &root);
                                   });
            if (!ok)
            {
                owner.models_recorded_ = false;
                return;
            }
        }
        else if (draw.model)
        {
            ModelDraw &model = owner.model_draws_[draw.index];
            owner.deferred_.set_model_hemi_negative(model.hemi_negative.data());
            std::copy(model.mvp.begin(), model.mvp.end(), mvp);
            if (model.skeleton ? !model.model->record_animated(frame, owner.deferred_, mvp, *model.skeleton,
                                                               owner.model_error_, GeometryPhase::Transparent,
                                                               model.lod, model.normal_rows.data(), draw.priority,
                                                               model.opacity)
                               : !model.model->record(frame, owner.deferred_, mvp, nullptr, 0, owner.model_error_,
                                                      GeometryPhase::Transparent, model.lod, nullptr,
                                                      model.normal_rows.data(), draw.priority, model.opacity))
            {
                owner.models_recorded_ = false;
                return;
            }
        }
        else
        {
            const LevelDraw &level = owner.level_draws_[draw.index];
            std::copy(level.mvp.begin(), level.mvp.end(), mvp);
            auto command_batch = owner.deferred_.geometry_batch();
            if (!owner.current_level_->record_visual(level.index, frame, owner.deferred_, mvp,
                                                     GeometryPhase::Transparent, level.lod, level.instance == nullptr,
                                                     false, draw.priority))
            {
                owner.level_recorded_ = false;
                owner.level_error_ = "Vulkan transparent level visual " + std::to_string(level.index) +
                                     " failed (lod=" + std::to_string(level.lod) + ")";
                return;
            }
        }
    }
    if (owner.smoke_compute_ && owner.smoke_noise_ && !owner.smoke_failed_ &&
        frame.frame_index < owner.smoke_recorded_.size() && owner.smoke_recorded_[frame.frame_index])
    {
        std::string error;
        if (!owner.smoke_draw_)
        {
            auto pass = std::make_unique<SmokeTrailDrawPass>();
            const auto& physical = owner.window_.physical();
            if (pass->initialize(owner.window_.device(), owner.window_.device_proc(), physical.properties.limits,
                    physical.memory, owner.buffer_upload_.buffer, frame.render_pass, owner.pipeline_cache_.handle(),
                    FrameContext::FramesInFlight, read_engine_shader, error, false,
                    owner.targets_.samples()))
                owner.smoke_draw_ = std::move(pass);
        }
        owner.deferred_.invalidate_geometry_cache();
        if (!owner.smoke_draw_ || !owner.smoke_draw_->record(frame, *owner.smoke_compute_, owner.smoke_noise_->view(),
                owner.smoke_noise_->sampler(), owner.smoke_frames_[frame.frame_index].draw, owner.scene_mvp_.data(), error))
        {
            owner.smoke_failed_ = true;
            Msg("! [renderer-vulkan] smoke trail drawing disabled: %s", error.c_str());
        }
    }
    if (owner.trail_enabled_ && !owner.trail_failed_ && owner.trail_texture_)
    {
        owner.trail_manager_.advance(Device.dwFrame, Device.fTimeDelta);
        auto groups = owner.trail_manager_.groups();
        if (!groups.empty())
        {
            std::string error;
            if (!owner.trail_draw_)
            {
                auto pass = std::make_unique<SmokeTrailDrawPass>();
                const auto& physical = owner.window_.physical();
                if (pass->initialize(owner.window_.device(), owner.window_.device_proc(), physical.properties.limits,
                        physical.memory, owner.buffer_upload_.buffer, frame.render_pass, owner.pipeline_cache_.handle(),
                        FrameContext::FramesInFlight, read_engine_shader, error, true,
                        owner.targets_.samples()))
                    owner.trail_draw_ = std::move(pass);
            }
            SmokeTrailDrawParams params{};
            params.subdivisions = 12; params.texcoord_factor = 1.f; params.smoothing = 1;
            params.edge_policy = 1; params.tail_fade = 1; params.use_gpu_state = 0;
            params.camera_x = Device.vCameraPosition.x; params.camera_y = Device.vCameraPosition.y;
            params.camera_z = Device.vCameraPosition.z; params.camera_facing = 1;
            owner.deferred_.invalidate_geometry_cache();
            if (!owner.trail_draw_ || !owner.trail_draw_->record_trail(frame, groups, params, owner.scene_mvp_.data(),
                    owner.trail_texture_, owner.textures_.sampler(), error))
            {
                owner.trail_failed_ = true;
                Msg("! [renderer-vulkan] projectile trails disabled: %s", error.c_str());
            }
        }
    }
    for (VulkanRainRender *rain : owner.rain_draws_)
        if (rain && !rain->record(frame, owner.model_error_))
        {
            owner.models_recorded_ = false;
            return;
        }
    for (VulkanThunderboltRender *bolt : owner.thunderbolt_draws_)
        if (bolt && !bolt->record(frame, owner.model_error_))
        {
            owner.models_recorded_ = false;
            return;
        }
    // World-space IUIRender effects belong before the HUD clears depth and
    // paints the weapon. Overlay UI is recorded separately after the HUD.
    owner.ui_recorded_ &= owner.ui_.record(
        frame, owner.ui_error_, false, VK_NULL_HANDLE,
        frame.image_index < owner.ui_depth_sets_.size() ? owner.ui_depth_sets_[frame.image_index] : VK_NULL_HANDLE,
        true);
}

void VulkanGameDevice::prepare_light_grid(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    if (!use_light_grid || owner.targets_.samples() != VK_SAMPLE_COUNT_1_BIT)
    {
        owner.light_grid_.discard_frame(frame.image_index);
        return;
    }
    if (!owner.light_grid_.prepare(frame, owner.local_uniforms_, owner.local_scissors_, owner.model_error_))
    {
        owner.light_grid_.discard_frame(frame.image_index);
        if (Device.dwFrame % 300 == 0)
            Msg("! [renderer-vulkan] light grid frame fallback: %s", owner.model_error_.c_str());
    }
}

void VulkanGameDevice::record_local_lights(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    PassCpuSample timing{"local-lights", owner.deferred_, owner.deferred_.draw_calls()};
    if (frame.image_index >= owner.local_sets_.size())
    {
        owner.local_lights_recorded_ = false;
        return;
    }
    if (owner.local_bound_projections_.size() != owner.local_sets_.size())
        owner.local_bound_projections_.resize(owner.local_sets_.size());
    const bool use_grid_for_frame = use_light_grid && owner.targets_.samples() == VK_SAMPLE_COUNT_1_BIT;
    if (!owner.local_lights_recorded_ ||
        (use_grid_for_frame && !owner.light_grid_.record(frame, owner.targets_.lighting_set(frame.image_index))))
    {
        owner.local_lights_recorded_ = false;
        return;
    }
    if (use_grid_for_frame && owner.light_grid_.count(frame.image_index))
        owner.deferred_.account_light_grid_draw();
    if (Device.dwFrame % 300 == 0)
        Msg("[renderer-vulkan] light.grid frame=%u point-lights=%u total=%zu", Device.dwFrame,
            owner.light_grid_.count(frame.image_index), owner.local_uniforms_.size());
    auto &sets = owner.local_sets_[frame.image_index];
    auto &projections = owner.local_bound_projections_[frame.image_index];
    owner.local_descriptor_buffers_.resize(owner.local_sets_.size());
    auto &descriptor_buffer = owner.local_descriptor_buffers_[frame.image_index];
    const auto uniform_buffer = owner.local_shadows_.uniform(frame.image_index);
    if (descriptor_buffer != uniform_buffer)
    {
        for (auto &set : sets)
            owner.deferred_.release_gbuffer(set);
        sets.clear();
        projections.clear();
        descriptor_buffer = uniform_buffer;
    }
    if (sets.size() < owner.local_uniforms_.size())
        sets.resize(owner.local_uniforms_.size());
    projections.resize(sets.size());
    for (uint32_t i = 0; i < owner.local_uniforms_.size(); ++i)
    {
        if (use_grid_for_frame && owner.light_grid_.contains(frame.image_index, i))
            continue;
        VkDescriptorSet &set = owner.local_sets_[frame.image_index][i];
        auto &bound = owner.local_bound_projections_[frame.image_index][i];
        if (!set)
            bound = VK_NULL_HANDLE;
        if (!set && !owner.deferred_.local_light_set(
                        owner.local_shadows_.view(frame.image_index), owner.local_shadows_.sampler(),
                        owner.local_shadows_.uniform(frame.image_index),
                        VkDeviceSize(i) * owner.local_shadows_.uniform_stride(), sizeof(LocalLightUniform), set,
                        owner.model_error_, owner.local_projection_views_[i]))
        {
            owner.local_lights_recorded_ = false;
            return;
        }
        if (bound != owner.local_projection_views_[i])
        {
            owner.deferred_.update_light_projection(set, owner.local_projection_views_[i], owner.textures_.sampler());
            bound = owner.local_projection_views_[i];
        }
        VkRect2D scissor = owner.local_scissors_[i];
        if (owner.local_uniforms_[i].debug_mode[0] == 1.f)
        {
            // Project each light's scissor into the upper-left base panel.
            // Round the far edge up so edge pixels cannot lose their light.
            const uint32_t right = uint32_t(scissor.offset.x) + scissor.extent.width;
            const uint32_t bottom = uint32_t(scissor.offset.y) + scissor.extent.height;
            scissor.offset.x /= 2;
            scissor.offset.y /= 2;
            scissor.extent.width = (right + 1) / 2 - uint32_t(scissor.offset.x);
            scissor.extent.height = (bottom + 1) / 2 - uint32_t(scissor.offset.y);
        }
        if (!owner.deferred_.record_local_light(frame,
                owner.targets_.multisampled_lighting_set(frame.image_index), set, scissor))
        {
            owner.local_lights_recorded_ = false;
            return;
        }
    }
}

void VulkanGameDevice::record_hud(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    struct DrawRef
    {
        float distance{};
        bool model{};
        size_t index{};
        bool particle{};
    };
    std::vector<DrawRef> hud;
    for (size_t i = 0; i < owner.level_draws_.size(); ++i)
        if (owner.level_draws_[i].hud)
            hud.push_back({owner.level_draws_[i].sort_distance, false, i});
    for (size_t i = 0; i < owner.model_draws_.size(); ++i)
        if (owner.model_draws_[i].hud)
            hud.push_back({owner.model_draws_[i].sort_distance, true, i});
    for (size_t i = 0; i < owner.particle_draws_.size(); ++i)
        if (owner.particle_draws_[i].hud)
            hud.push_back({owner.particle_draws_[i].sort_distance, false, i, true});
    std::stable_sort(hud.begin(), hud.end(),
                     [](const DrawRef &left, const DrawRef &right) { return left.distance > right.distance; });
    for (const DrawRef &draw : hud)
    {
        float mvp[16];
        if (draw.particle)
        {
            auto &particle = owner.particle_draws_[draw.index];
            if (particle.visual->getType() == 8 && owner.record_gpu_particle_effect(frame,
                    *static_cast<VulkanParticleEffect *>(particle.visual), particle, GeometryPhase::Transparent))
                continue;
            std::copy(particle.mvp.begin(), particle.mvp.end(), mvp);
            bool ok =
                particle.visual->getType() == 8
                    ? static_cast<VulkanParticleEffect *>(particle.visual)
                          ->record(frame, owner.deferred_, mvp, particle.right, particle.up, true, owner.model_error_)
                    : static_cast<VulkanParticleGroup *>(particle.visual)
                          ->record(frame, owner.deferred_, mvp, particle.right, particle.up, true, owner.model_error_,
                                   GeometryPhase::Transparent, -1, [&](VulkanParticleEffect& effect) {
                                       return owner.record_gpu_particle_effect(frame, effect, particle, GeometryPhase::Transparent);
                                   }, [&](VulkanParticleEffect& root, VulkanParticleEffect& child) {
                                       return owner.record_gpu_particle_effect(frame, child, particle, GeometryPhase::Transparent, &root);
                                   });
            if (!ok)
            {
                owner.models_recorded_ = false;
                return;
            }
        }
        else if (draw.model)
        {
            ModelDraw &model = owner.model_draws_[draw.index];
            owner.deferred_.set_model_hemi_negative(model.hemi_negative.data());
            std::copy(model.mvp.begin(), model.mvp.end(), mvp);
            const bool recorded =
                model.skeleton
                    ? model.model->record_animated(frame, owner.deferred_, mvp, *model.skeleton, owner.model_error_,
                                                   GeometryPhase::Hud, model.lod, model.normal_rows.data())
                    : model.model->record(frame, owner.deferred_, mvp, nullptr, 0, owner.model_error_,
                                          GeometryPhase::Hud, model.lod, nullptr, model.normal_rows.data());
            if (!recorded)
            {
                owner.models_recorded_ = false;
                return;
            }
        }
        else
        {
            const LevelDraw &level = owner.level_draws_[draw.index];
            std::copy(level.mvp.begin(), level.mvp.end(), mvp);
            if (!owner.current_level_->record_hud_visual(level.index, frame, owner.deferred_, mvp, level.lod))
            {
                owner.level_recorded_ = false;
                owner.level_error_ = "Vulkan HUD level visual " + std::to_string(level.index) +
                                     " failed (lod=" + std::to_string(level.lod) + ")";
                return;
            }
        }
    }
}

void VulkanGameDevice::record_shadow_models(const FrameRecordingContext &frame, const float (&vp)[16], void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    PassCpuSample timing{"shadow-models", owner.deferred_, owner.deferred_.draw_calls()};
    Fmatrix light;
    std::memcpy(&light, vp, sizeof(light));
    for (auto &draw : owner.shadow_model_draws_)
    {
        if (draw.hud)
            continue;
        Fmatrix world, mvp;
        std::memcpy(&world, draw.world.data(), sizeof(world));
        mvp.mul(light, world);
        if (draw.skeleton)
        {
            draw.skeleton->CalculateBones();
            const auto &sphere = draw.skeleton->dcast_RenderVisual()->getVisData().sphere;
            if (std::isfinite(sphere.R) && sphere.R > 0)
            {
                CFrustum frustum;
                frustum.CreateFromMatrix(mvp, FRUSTUM_P_LRTB | FRUSTUM_P_NEAR | FRUSTUM_P_FAR);
                if (!frustum.testSphere_dirty(sphere.P, sphere.R))
                    continue;
            }
        }
        else if (draw.has_shadow_sphere)
        {
            Fvector center;
            center.set(draw.shadow_sphere[0], draw.shadow_sphere[1], draw.shadow_sphere[2]);
            CFrustum frustum;
            frustum.CreateFromMatrix(mvp, FRUSTUM_P_LRTB | FRUSTUM_P_NEAR | FRUSTUM_P_FAR);
            if (!frustum.testSphere_dirty(center, draw.shadow_sphere[3]))
                continue;
        }
        float transform[16];
        std::memcpy(transform, &mvp, sizeof(mvp));
        if (!draw.model->record_shadow(frame, owner.deferred_, transform, draw.skeleton, owner.model_error_,
                draw.lod, draw.world.data()))
        {
            owner.models_recorded_ = false;
            return;
        }
    }
}

bool VulkanGameDevice::record_vsm_shadow_models(const FrameRecordingContext &frame, const float *vp,
                                                const VsmPageJob &, void *user, std::string &error)
{
    if (!user || !vp)
    {
        error = "VSM model page needs a device and a light projection";
        return false;
    }
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    if (!owner.models_recorded_)
    {
        error = owner.model_error_;
        return false;
    }
    float matrix[16];
    std::copy_n(vp, 16, matrix);
    record_shadow_models(frame, matrix, user);
    if (!owner.models_recorded_)
    {
        error = owner.model_error_;
        return false;
    }
    error.clear();
    return true;
}

void VulkanGameDevice::record_models(const FrameRecordingContext &frame, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    PassCpuSample timing{"opaque-models", owner.deferred_, owner.deferred_.draw_calls()};
    auto command_batch = owner.deferred_.geometry_batch();
    for (auto &draw : owner.model_draws_)
    {
        if (draw.hud)
            continue;
        owner.deferred_.set_model_hemi_negative(draw.hemi_negative.data());
        float mvp[16];
        std::copy(draw.mvp.begin(), draw.mvp.end(), mvp);
        if (draw.skeleton
                ? !draw.model->record_animated(frame, owner.deferred_, mvp, *draw.skeleton, owner.model_error_,
                                               GeometryPhase::OpaqueAndAlphaTest, draw.lod, draw.normal_rows.data(),
                                               -1, 1.f, draw.world.data())
                : !draw.model->record(frame, owner.deferred_, mvp, nullptr, 0, owner.model_error_,
                                      GeometryPhase::OpaqueAndAlphaTest, draw.lod, nullptr, draw.normal_rows.data(),
                                      -1, 1.f, draw.world.data()))
        {
            owner.models_recorded_ = false;
            return;
        }
    }
    // R2's PARTICLE SET is a cutout G-buffer draw. Particle groups can mix
    // this mode with forward effects; each child filters itself by phase.
    for (auto &particle : owner.particle_draws_)
    {
        if (particle.hud)
            continue;
        if (particle.visual->getType() == 8 && owner.record_gpu_particle_effect(frame,
                *static_cast<VulkanParticleEffect *>(particle.visual), particle, GeometryPhase::OpaqueAndAlphaTest))
            continue;
        float mvp[16];
        std::copy(particle.mvp.begin(), particle.mvp.end(), mvp);
        const bool ok = particle.visual->getType() == 8
                            ? static_cast<VulkanParticleEffect *>(particle.visual)
                                  ->record(frame, owner.deferred_, mvp, particle.right, particle.up, false,
                                           owner.model_error_, GeometryPhase::OpaqueAndAlphaTest)
                            : static_cast<VulkanParticleGroup *>(particle.visual)
                                  ->record(frame, owner.deferred_, mvp, particle.right, particle.up, false,
                                           owner.model_error_, GeometryPhase::OpaqueAndAlphaTest, -1,
                                           [&](VulkanParticleEffect& effect) {
                                               return owner.record_gpu_particle_effect(frame, effect, particle,
                                                   GeometryPhase::OpaqueAndAlphaTest);
                                           }, [&](VulkanParticleEffect& root, VulkanParticleEffect& child) {
                                               return owner.record_gpu_particle_effect(frame, child, particle,
                                                   GeometryPhase::OpaqueAndAlphaTest, &root);
                                           });
        if (!ok)
        {
            owner.models_recorded_ = false;
            return;
        }
    }
}

bool VulkanGameDevice::record_object_velocity(const FrameRecordingContext &frame, void *user,
    std::string& error)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    const bool gpu_draw_active = owner.gpu_particle_graphics_ && !owner.gpu_particle_graphics_failed_ &&
        !owner.gpu_particle_failed_ && frame.frame_index < owner.gpu_particle_recorded_.size() &&
        owner.gpu_particle_recorded_[frame.frame_index];
    if (!owner.object_velocity_.begin(frame))
    {
        owner.models_recorded_ = false;
        owner.model_error_ = "object velocity render pass could not begin";
        error = owner.model_error_;
        return false;
    }
    for (const ParticleDraw& draw : owner.particle_draws_)
    {
        if (draw.hud) continue;
        float particle_mvp[16];
        std::copy(draw.mvp.begin(), draw.mvp.end(), particle_mvp);
        bool ok = false;
        if (draw.visual->getType() == 8)
            ok = static_cast<VulkanParticleEffect *>(draw.visual)->prepare_motion_geometry(frame, owner.deferred_,
                particle_mvp, draw.right, draw.up, false, gpu_draw_active, owner.model_error_);
        else if (draw.visual->getType() == 9)
            ok = static_cast<VulkanParticleGroup *>(draw.visual)->prepare_motion_geometry(frame, owner.deferred_,
                particle_mvp, draw.right, draw.up, false, gpu_draw_active, owner.model_error_);
        if (!ok)
        {
            owner.models_recorded_ = false;
            owner.model_error_ = "Vulkan transparent particle motion geometry preparation failed";
            break;
        }
    }
    for (const ModelDraw& draw : owner.model_draws_)
    {
        if (draw.hud)
            continue;
        const float* previous_mvp = draw.has_previous_mvp ? draw.previous_mvp.data() : draw.mvp.data();
        if (!draw.model->record_object_velocity(frame, owner.object_velocity_, draw.mvp.data(),
                previous_mvp, draw.skeleton, draw.lod, owner.model_error_, draw.world.data()))
        {
            owner.models_recorded_ = false;
            break;
        }
    }
    if (owner.models_recorded_ && owner.scene_visibility_ && owner.current_level_)
        for (const LevelDraw& draw : owner.level_draws_)
        {
            if (draw.hud)
                continue;
            float current[16], previous[16];
            std::copy(draw.mvp.begin(), draw.mvp.end(), current);
            std::copy(draw.has_previous_mvp ? draw.previous_mvp.begin() : draw.mvp.begin(),
                draw.has_previous_mvp ? draw.previous_mvp.end() : draw.mvp.end(), previous);
            if (!owner.current_level_->record_object_velocity_visual(draw.index, frame,
                    owner.object_velocity_, current, previous, draw.lod, draw.instance != nullptr))
            {
                owner.level_recorded_ = false;
                owner.level_error_ = "Vulkan level visual object velocity failed (index=" +
                    std::to_string(draw.index) + ")";
                break;
            }
        }
    if (owner.models_recorded_)
    {
        const bool particle_history_valid = owner.motion_.history_valid();
        for (const ParticleDraw& draw : owner.particle_draws_)
        {
            if (draw.hud) continue;
            float particle_mvp[16];
            std::copy(draw.mvp.begin(), draw.mvp.end(), particle_mvp);
            float previous_particle_mvp[16];
            std::copy_n(particle_history_valid ? owner.motion_.previous_view_projection() : particle_mvp,
                16, previous_particle_mvp);
            bool ok = false;
            if (draw.visual->getType() == 8)
                ok = static_cast<VulkanParticleEffect *>(draw.visual)->record_dynamic_occluder_mask(
                    frame, owner.object_velocity_, particle_mvp, previous_particle_mvp, particle_history_valid, false,
                    owner.gpu_particle_graphics_ && !owner.gpu_particle_graphics_failed_);
            else if (draw.visual->getType() == 9)
                ok = static_cast<VulkanParticleGroup *>(draw.visual)->record_dynamic_occluder_mask(
                    frame, owner.object_velocity_, particle_mvp, previous_particle_mvp, particle_history_valid, false,
                    owner.gpu_particle_graphics_ && !owner.gpu_particle_graphics_failed_);
            if (!ok)
            {
                owner.hiz_dynamic_mask_recorded_ = false;
                owner.frame_.set_hiz_dynamic_mask_enabled(false);
                owner.hiz_.reset_history();
                Msg("! [renderer-vulkan] PAPI particle depth mask unavailable; Hi-Z culling disabled for this frame");
                break;
            }
        }
    }
    if (owner.models_recorded_)
    {
        const bool particle_history_valid = owner.motion_.history_valid();
        for (const ParticleDraw& draw : owner.particle_draws_)
        {
            if (draw.hud) continue;
            float current_mvp[16], previous_mvp[16];
            std::copy(draw.mvp.begin(), draw.mvp.end(), current_mvp);
            std::copy_n(particle_history_valid ? owner.motion_.previous_view_projection() : current_mvp,
                16, previous_mvp);
            if (draw.visual->getType() == 8)
                owner.record_gpu_particle_motion(frame, *static_cast<VulkanParticleEffect *>(draw.visual), draw,
                    current_mvp, previous_mvp, false);
            else if (draw.visual->getType() == 9)
                static_cast<VulkanParticleGroup *>(draw.visual)->record_particle_motion(frame, owner.object_velocity_,
                    current_mvp, previous_mvp, particle_history_valid, false, gpu_draw_active, false, -1,
                    [&](VulkanParticleEffect& effect) {
                        return owner.record_gpu_particle_motion(frame, effect, draw, current_mvp, previous_mvp, false);
                    }, [&](VulkanParticleEffect& root, VulkanParticleEffect& child) {
                        return owner.record_gpu_particle_motion(frame, child, draw, current_mvp, previous_mvp,
                            false, &root);
                    });
        }
    }
    if (owner.models_recorded_ && owner.level_recorded_ && owner.current_level_)
    {
        const bool have_previous_view = owner.motion_.history_valid();
        float current[16], previous[16];
        std::copy(owner.scene_mvp_.begin(), owner.scene_mvp_.end(), current);
        std::copy_n(have_previous_view ? owner.motion_.previous_view_projection() : current, 16, previous);
        if (!owner.current_level_->record_tree_velocity(frame, owner.object_velocity_, current, previous,
                owner.current_tree_wind_, have_previous_view ? owner.previous_tree_wind_ : owner.current_tree_wind_))
        {
            owner.level_recorded_ = false;
            owner.level_error_ = "Vulkan tree wind velocity recording failed";
        }
    }
    if (owner.models_recorded_ && owner.level_recorded_)
    {
        std::vector<TransparentDrawRef> particle_draws;
        for (size_t i = 0; i < owner.particle_draws_.size(); ++i)
        {
            const ParticleDraw& particle = owner.particle_draws_[i];
            if (particle.hud) continue;
            if (particle.visual->getType() == 8)
                particle_draws.push_back({particle.sort_distance, false, i, true, false,
                    static_cast<VulkanParticleEffect *>(particle.visual)->priority()});
            else if (particle.visual->getType() == 9)
            {
                const auto *group = static_cast<VulkanParticleGroup *>(particle.visual);
                for (uint8_t priority = 1; priority <= 3; ++priority)
                    if (group->has_priority(priority))
                        particle_draws.push_back({particle.sort_distance, false, i, true, false, priority});
            }
        }
        sort_transparent_draws(particle_draws);
        const bool particle_history_valid = owner.motion_.history_valid();
        for (const TransparentDrawRef& item : particle_draws)
        {
            const ParticleDraw& particle = owner.particle_draws_[item.index];
            float current_particle_mvp[16];
            std::copy(particle.mvp.begin(), particle.mvp.end(), current_particle_mvp);
            float previous_mvp[16];
            std::copy_n(particle_history_valid ? owner.motion_.previous_view_projection() : current_particle_mvp,
                16, previous_mvp);
            bool ok = true;
            if (particle.visual->getType() == 8)
            {
                auto& effect = *static_cast<VulkanParticleEffect *>(particle.visual);
                const bool gpu_handled = owner.record_gpu_particle_motion(frame, effect, particle,
                    current_particle_mvp, previous_mvp, true);
                ok = gpu_handled || effect.record_particle_motion(frame, owner.object_velocity_,
                    current_particle_mvp, previous_mvp, particle_history_valid, false, false);
            }
            else
                ok = static_cast<VulkanParticleGroup *>(particle.visual)->record_particle_motion(frame,
                    owner.object_velocity_, current_particle_mvp, previous_mvp, particle_history_valid, false,
                    gpu_draw_active, true, item.priority,
                    [&](VulkanParticleEffect& effect) {
                        return owner.record_gpu_particle_motion(frame, effect, particle, current_particle_mvp,
                            previous_mvp, true);
                    }, [&](VulkanParticleEffect& root, VulkanParticleEffect& child) {
                        return owner.record_gpu_particle_motion(frame, child, particle, current_particle_mvp,
                            previous_mvp, true, &root);
                    });
            if (!ok)
            {
                owner.models_recorded_ = false;
                owner.model_error_ = "Vulkan transparent particle object velocity recording failed";
                break;
            }
        }
    }
    owner.object_velocity_.end(frame);
    if (!owner.models_recorded_)
    {
        error = owner.model_error_.empty() ? "object velocity model recording failed" : owner.model_error_;
        return false;
    }
    if (!owner.level_recorded_)
    {
        error = owner.level_error_.empty() ? "object velocity level recording failed" : owner.level_error_;
        return false;
    }
    error.clear();
    return true;
}

void VulkanGameDevice::record_readback(VkCommandBuffer command, VkImage image, VkExtent2D extent, void *user)
{
    auto &owner = *static_cast<VulkanGameDevice *>(user);
    VkBufferImageCopy region{};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {extent.width, extent.height, 1};
    owner.frame_dispatch_.cmd_copy_image_to_buffer(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                                   owner.screenshot_buffer_.handle(), 1, &region);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = owner.screenshot_buffer_.handle();
    barrier.size = VK_WHOLE_SIZE;
    owner.frame_dispatch_.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0,
                                               0, nullptr, 1, &barrier, 0, nullptr);
}
} // namespace xray::render::vulkan
