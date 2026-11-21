#include "xrEngine/stdafx.h"
#include "src/Layers/xrRenderVK/ParticleVisual.h"
#include "src/Layers/xrRenderVK/LevelPassGraph.h"
#include "src/Layers/xrRenderVK/GpuParticleTranslate.h"
#include "src/Layers/xrRenderVK/GpuParticleNoise.h"
#include "src/Layers/xrRenderVK/GpuParticleCommands.h"
#include "src/Layers/xrRenderVK/GpuParticlePrograms.h"
#include "src/Layers/xrRenderVK/GpuParticleDrawBuckets.h"
#include "src/Layers/xrRenderVK/GpuParticleEmitters.h"
#include "src/Layers/xrRenderVK/GpuParticleMaterials.h"
#include "src/Layers/xrRenderVK/SmokeTrailController.h"
#include "src/Layers/xrRenderVK/TrailState.h"
#include "src/Layers/xrRenderVK/TrailManager.h"
#include "xrParticles/noise.h"
#include "xrParticles/psystem.h"
#include "xrParticles/particle_actions_collection.h"

#undef NDEBUG
#include <cassert>

using namespace xray::render::vulkan;

int main()
{
    {
        TrailManager trails;
        const float a[]{0.f, 0.f, 0.f}, b[]{1.f, 0.f, 0.f}, c[]{2.f, 0.f, 0.f};
        trails.observe(1, a, b, .1f, 10, .1f, 0xffff0000u);
        trails.observe(2, a, b, .1f, 10, .1f, 0xff00ff00u);
        assert(trails.groups().size() == 2);
        auto colored = trails.groups();
        assert(colored[0].color != colored[1].color);
        trails.observe(1, b, c, .1f, 11, .1f);
        auto groups = trails.groups();
        assert(groups.size() == 2 && groups[0].points.size() == 3);
        trails.observe(1, a, b, .1f, 13, .1f);
        assert(trails.groups().size() == 3);
        trails.advance(14, 3.1f);
        assert(trails.groups().empty());
    }
    {
        TrailState trail;
        const float start[]{0.f, 0.f, 0.f}, too_close[]{.01f, 0.f, 0.f}, next[]{.1f, 0.f, 0.f};
        assert(trail.update(0.f, start, .15f) && trail.update(.1f, too_close, .15f) && trail.size() == 1);
        assert(trail.update(.1f, next, .15f) && trail.size() == 2);
        auto groups = trail.groups();
        assert(groups.size() == 1 && groups[0].points.size() == 2 &&
            std::abs(groups[0].points[1].distance - .1f) < .0001f &&
            std::abs(groups[0].points[0].direction[2] - .15f) < .0001f);
        trail.break_group();
        assert(trail.update(0.f, next, .15f) && trail.groups().size() == 2);
        assert(!trail.update(-1.f, start, .15f) && trail.size() == 3);
        assert(trail.update(3.f, nullptr, .15f) && trail.size() == 0);
    }
    {
        SmokeTrailController controller;
        const float first[]{1.f, 2.f, 3.f}, next[]{2.f, 2.f, 3.f}, direction[]{0.f, 0.f, 1.f};
        controller.update(first, direction, .05f, 10);
        const auto a = controller.prepare(10, .05f, 1.f);
        const auto retry = controller.prepare(10, .05f, 1.f);
        assert(a.active && a.emit.count == 2 && retry.emit.count == a.emit.count &&
            a.emit.previous_x == 1.f && std::abs(a.accumulator - .25f) < .001f);
        controller.commit(a);
        assert(!controller.prepare(10, .05f, 1.f).active);
        controller.update(next, direction, .05f, 11);
        const auto b = controller.prepare(11, .05f, 1.05f);
        assert(b.active && b.emit.count == 2 && b.emit.previous_x == 1.f && b.emit.current_x == 2.f);
        controller.commit(b);
        controller.commit(controller.prepare(12, .05f, 1.1f));
        const float moved[]{20.f, 2.f, 3.f};
        controller.update(moved, direction, .05f, 20);
        const auto resumed = controller.prepare(20, .05f, 2.f);
        assert(resumed.emit.previous_x == 20.f && resumed.emit.current_x == 20.f);
    }
    // A mixed group visits both record callbacks; every child must choose
    // exactly one world phase, including R2 PARTICLE SET versus S_SET SET.
    for (bool screen_set : {false, true})
        for (int mode = 0; mode < 6; ++mode)
            for (bool hud : {false, true})
            {
                const bool deferred = particle_records_in_phase(hud, screen_set, mode,
                    GeometryPhase::OpaqueAndAlphaTest);
                const bool forward = particle_records_in_phase(hud, screen_set, mode,
                    GeometryPhase::Transparent);
                assert(deferred != forward && deferred == (!hud && !screen_set && mode == 0));
            }
    assert(!particle_uses_screen_program("particles\\xadd", "S_SET   "));
    assert(particle_uses_screen_program("effects\\screen_add", "S_SET   "));
    assert(!particle_uses_screen_program("particles\\add", "PARTICLE"));
    RenderPassGraph graph;
    std::string error;
    RenderPassSelection xadd{RenderConsumer::Particle, SurfaceMode::Transparent};
    xadd.blend_mode = 2;
    xadd.particle_screen_set = particle_uses_screen_program("particles\\xadd", "S_SET   ");
    assert(build_render_pass_graph(xadd, graph, error));
    assert(graph.passes[0].fragment == "vk\\particle_world_additive.ps");
    Core.Initialize("vulkan_particle_visual_test", nullptr, false);
    {
    uint32_t bucket{};
    assert(gpu_particle_draw_bucket(1, 2, 1, 2, 0, 0, true, bucket) && bucket == 90);
    assert(gpu_particle_draw_bucket(0, 2, 1, 2, 0, 2, false, bucket) && bucket == 40);
    assert(gpu_particle_draw_bucket(0, 2, 0, 2, 0, 0, true, bucket, true) && bucket == 20);
    assert(!gpu_particle_draw_bucket(0, 2, 2, 2, 0, 0, false, bucket));
    const auto noise = gpu_particle_noise_table();
    int permutation[514]; float gradients[514 * 3];
    PAPI::GetNoise3Tables(permutation, gradients);
    for (size_t i = 0; i < noise.size(); ++i)
    {
        u32 index; std::memcpy(&index, &noise[i].w, sizeof(index));
        assert(index == static_cast<u32>(permutation[i]) && index < 256 &&
            noise[i].x == gradients[3 * i] && noise[i].y == gradients[3 * i + 1] &&
            noise[i].z == gradients[3 * i + 2]);
    }
    GpuParticleCommands command_queue(2);
    GpuParticleCommands::Batch batch;
    std::string queue_error;
    assert(command_queue.begin_generation(1, queue_error));
    xray::render::fg::GpuPapiCommand queued{};
    assert(command_queue.enqueue(1, queued, queue_error));
    queued.type = 2;
    assert(command_queue.enqueue(1, queued, queue_error));
    assert(command_queue.snapshot(batch, queue_error));
    assert(batch.ranges.size() == 1 && batch.ranges[0][0] == 0 && batch.ranges[0][1] == 2 &&
        batch.commands[0].emitter == 2 && batch.commands[0].generation == 1 && batch.commands[1].serial == 2);
    assert(!command_queue.acknowledge(1, 1, 3, queue_error));
    assert(command_queue.acknowledge(1, 1, 1, queue_error));
    assert(command_queue.snapshot(batch, queue_error) && batch.commands.size() == 1 && batch.commands[0].serial == 2);
    assert(!command_queue.begin_generation(1, queue_error));
    assert(command_queue.acknowledge(1, 1, 2, queue_error) && command_queue.begin_generation(1, queue_error));
    assert(command_queue.acknowledge(1, 1, 2, queue_error)); // stale generation is harmless
    assert(command_queue.enqueue(1, queued, queue_error) && command_queue.snapshot(batch, queue_error) &&
        batch.commands[0].generation == 2 && batch.commands[0].serial == 1);
    assert(command_queue.begin_generation(0, queue_error) && command_queue.enqueue(0, queued, queue_error) &&
        command_queue.snapshot(batch, queue_error));
    std::vector<xray::render::fg::GpuPapiStatus> statuses(2);
    statuses[0].generation = 1; statuses[0].serial = 1;
    statuses[1].generation = 2; statuses[1].serial = 2;
    assert(!command_queue.acknowledge(statuses, queue_error));
    assert(command_queue.snapshot(batch, queue_error) && batch.commands.size() == 2);
    statuses[1].serial = 1;
    assert(command_queue.acknowledge(statuses, queue_error) && command_queue.snapshot(batch, queue_error) && batch.commands.empty());
    ParticleEffectDef translated_def;
    translated_def.name = "gravity";
    translated_def.max_particles = 128;
    translated_def.collision_friction = .75f;
    auto append_u32 = [&](u32 value) {
        for (unsigned i = 0; i < 4; ++i) translated_def.actions.push_back(static_cast<u8>(value >> (i * 8)));
    };
    auto append_float = [&](float value) { u32 bits; std::memcpy(&bits, &value, sizeof(bits)); append_u32(bits); };
    append_u32(2);
    append_u32(PAPI::PAGravityID); append_u32(0); append_u32(PAPI::PAGravityID);
    append_float(1.f); append_float(-2.f); append_float(3.f);
    append_u32(PAPI::PAMoveID); append_u32(0); append_u32(PAPI::PAMoveID);
    xray::render::fg::GpuPapiProgram program{};
    xr_vector<xray::render::fg::GpuPapiAction> actions;
    xr_string translate_error;
    assert(TranslateGpuParticleDefinition(translated_def, program, actions, translate_error));
    assert(program.maxParticles == 128 && program.actionCount == 2 && program.collisionFriction == .75f);
    assert(actions.size() == 2 && actions[0].type == PAPI::PAGravityID &&
        actions[0].p0.x == 1.f && actions[0].p0.y == -2.f && actions[0].p0.z == 3.f &&
        actions[1].type == PAPI::PAMoveID);
    GpuParticlePrograms programs;
    uint32_t program_id = UINT32_MAX;
    std::string program_error;
    assert(programs.intern(translated_def, program_id, program_error) && program_id == 0 && programs.dirty());
    programs.mark_uploaded();
    assert(programs.intern(translated_def, program_id, program_error) && program_id == 0 && !programs.dirty());
    auto second_def = translated_def; second_def.name = "gravity_second";
    assert(programs.intern(second_def, program_id, program_error) && program_id == 1 &&
        programs.programs()[1].actionFirst == 2 && programs.actions().size() == 4 &&
        programs.names()[0] == "gravity" && programs.names()[1] == "gravity_second");
    assert(programs.set_material(1, 3, 2, 1, program_error) && programs.programs()[1].materialID == 3);
    assert(!programs.set_material(1, 3, 6, 1, program_error));
    auto invalid_def = translated_def; invalid_def.name = "invalid"; invalid_def.actions.pop_back();
    assert(!programs.intern(invalid_def, program_id, program_error) &&
        programs.programs().size() == 2 && programs.actions().size() == 4);
    {
        ParticleCatalog material_catalog; material_catalog.effects.push_back(translated_def);
        GpuParticlePrograms material_programs;
        assert(material_programs.intern(material_catalog.effects[0], program_id, program_error));
        GameTextureFactory unused_textures; DeferredPass unused_pass; GpuParticleMaterials material_table;
        assert(material_table.sync(material_programs, material_catalog, unused_textures, unused_pass, program_error) &&
            material_table.get(0) && material_table.get(0)->resolved && !material_table.get(0)->set);
        material_programs.clear();
        assert(!material_table.sync(material_programs, material_catalog, unused_textures, unused_pass, program_error));
        material_table.clear(unused_textures, unused_pass);
    }
    GpuParticleCommands emitter_commands(1);
    GpuParticleEmitters emitter_manager(programs, emitter_commands, 1);
    GpuParticleEmitters::Handle emitter_handle{};
    assert(emitter_manager.create(translated_def, emitter_handle, program_error));
    assert(!emitter_manager.has_child_effects(emitter_handle));
    assert(emitter_manager.play(emitter_handle, program_error));
    assert(!emitter_manager.children(emitter_handle, GPU_PAPI_INVALID, GPU_PAPI_INVALID,
        GPU_PAPI_INVALID, 32, program_error));
    assert(emitter_manager.children(emitter_handle, GPU_PAPI_INVALID, 0, GPU_PAPI_INVALID, 2, program_error));
    // A child using the root's own program still represents separate GPU
    // particles and must not be drawn together with CPU-triggered children.
    assert(emitter_manager.has_child_effects(emitter_handle) && emitter_manager.draw_programs(emitter_handle)->size() == 1);
    assert(emitter_manager.set_hud(emitter_handle, true, program_error) && emitter_manager.tick(emitter_handle, 33, program_error));
    GpuParticleCommands::Batch emitter_batch;
    assert(emitter_commands.snapshot(emitter_batch, program_error) && emitter_batch.commands.size() == 5);
    assert(emitter_batch.commands[0].type == 0 && emitter_batch.commands[1].type == 2 &&
        emitter_batch.commands[2].type == 7 && emitter_batch.commands[3].type == 5 && emitter_batch.commands[4].type == 6);
    std::vector<xray::render::fg::GpuPapiStatus> emitter_status(1);
    emitter_status[0].generation = 1; emitter_status[0].serial = 1;
    xray::render::fg::GpuPapiStatus read_status{};
    assert(emitter_manager.apply(emitter_status, program_error) && !emitter_manager.snapshot(emitter_handle, read_status));
    emitter_status[0].serial = 5;
    assert(emitter_commands.acknowledge(emitter_status, program_error) && emitter_manager.apply(emitter_status, program_error) &&
        emitter_manager.snapshot(emitter_handle, read_status));
    {
        GpuParticleCommands draw_commands(1);
        GpuParticleEmitters draw_emitters(programs, draw_commands, 1);
        GpuParticleEmitters::Handle root{};
        assert(draw_emitters.create(translated_def, root, program_error));
        assert(!draw_emitters.has_child_effects(root));
        assert(draw_emitters.draw_programs(root) && *draw_emitters.draw_programs(root) == std::vector<uint32_t>{0});
        assert(draw_emitters.children(root, GPU_PAPI_INVALID, 1, GPU_PAPI_INVALID, 2, program_error));
        assert(draw_emitters.has_child_effects(root));
        assert(draw_emitters.children(root, GPU_PAPI_INVALID, 0, GPU_PAPI_INVALID, 2, program_error));
        assert(draw_emitters.draw_programs(root) && *draw_emitters.draw_programs(root) == (std::vector<uint32_t>{0, 1}));
        assert(!draw_emitters.children(root, GPU_PAPI_INVALID, 99, GPU_PAPI_INVALID, 2, program_error));
        assert(draw_emitters.draw_programs(root)->size() == 2);
        assert(draw_emitters.destroy(root, program_error) && !draw_emitters.draw_programs(root));
    }
    translated_def.actions.pop_back();
    assert(!TranslateGpuParticleDefinition(translated_def, program, actions, translate_error));
    assert(program.actionCount == 2 && actions.size() == 2 && !translate_error.empty());
    auto catalog = std::make_shared<ParticleCatalog>();
    auto& def = catalog->effects.emplace_back();
    def.name = "smoke";
    def.max_particles = 8;
    def.actions = {0, 0, 0, 0};
    def.time_limit = 1.f;
    auto& group = catalog->groups.emplace_back();
    group.name = "smoke_group";
    group.time_limit = .5f;
    group.effects.push_back({"smoke", "smoke", "", "", 0.f, .1f, 7u});
    group.effects.push_back({"smoke", "", "", "", .2f, .4f, 5u});
    GpuParticlePrograms mirror_programs;
    GpuParticleCommands mirror_commands(2);
    GpuParticleEmitters mirror_emitters(mirror_programs, mirror_commands, 2);
    GpuParticlePrograms group_gpu_programs;
    GpuParticleCommands group_gpu_commands(4);
    GpuParticleEmitters group_gpu_emitters(group_gpu_programs, group_gpu_commands, 4);
    {
        VulkanParticleEffect first(catalog, def);
        VulkanParticleEffect second(catalog, def);
        Fmatrix parent; parent.identity(); parent.c.set(3.f, 4.f, 5.f);
        Fvector parent_velocity; parent_velocity.set(1.f, 2.f, 3.f);
        first.UpdateParent(parent, parent_velocity, TRUE);
        assert(first.attach_gpu_emitter(mirror_emitters, program_error));
        first.SetHudMode(TRUE);
        first.Play();
        first.OnFrame(40);
        assert(first.IsPlaying() && !second.IsPlaying());
        first.Stop(FALSE);
        assert(!first.IsPlaying() && first.ParticlesCount() == 0);
        GpuParticleCommands::Batch mirror_batch;
        assert(mirror_commands.snapshot(mirror_batch, program_error) && mirror_batch.commands.size() == 6);
        assert(mirror_batch.commands[0].type == 0 && mirror_batch.commands[1].type == 4 &&
            mirror_batch.commands[1].origin.x == 3.f && mirror_batch.commands[2].type == 5 &&
            mirror_batch.commands[3].type == 2 && mirror_batch.commands[4].type == 6 &&
            mirror_batch.commands[4].p0 == 33 && mirror_batch.commands[5].type == 3);
        VulkanParticleGroup instance(catalog, group);
        assert(instance.getType() == 9 && instance.getSubModel(0) && instance.getSubModel(1));
        assert(instance.getSubModel(2) == nullptr);
        assert(instance.attach_gpu_emitters(group_gpu_emitters, program_error));
        assert(!instance.attach_gpu_emitters(group_gpu_emitters, program_error));
        GpuParticleCommands::Batch group_batch;
        assert(group_gpu_commands.snapshot(group_batch, program_error) && group_batch.commands.size() == 4);
        assert(group_batch.commands[0].type == 0 && group_batch.commands[1].type == 7 &&
            group_batch.commands[1].p1 == 0 && group_batch.commands[2].type == 0 && group_batch.commands[3].type == 7);
        assert(instance.effects()[0]->gpu_draw_compatible() && instance.effects()[1]->gpu_draw_compatible());
        FrameRecordingContext empty_frame{}; DeferredPass empty_pass;
        const float identity[]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        Fvector right{1.f, 0.f, 0.f}, up{0.f, 1.f, 0.f};
        std::vector<VulkanParticleEffect*> visited;
        const auto try_gpu = [&](VulkanParticleEffect& effect) {
            visited.push_back(&effect); return effect.gpu_draw_compatible();
        };
        assert(instance.record(empty_frame, empty_pass, identity, right, up, false, program_error,
            GeometryPhase::Transparent, 1, try_gpu));
        assert(visited.size() == 2 && visited[0] == instance.effects()[0].get() && visited[1] == instance.effects()[1].get());
        visited.clear();
        assert(instance.record(empty_frame, empty_pass, identity, right, up, false, program_error,
            GeometryPhase::Transparent, 2, try_gpu) && visited.empty());
        instance.Play();
        assert(!instance.attach_gpu_emitters(group_gpu_emitters, program_error));
        instance.OnFrame(40);
        assert(instance.effects()[0]->IsPlaying() && !instance.effects()[1]->IsPlaying());
        assert(instance.triggered_count() == 0 && instance.related_count(0) == 0);
        instance.OnFrame(80);
        assert(!instance.effects()[0]->IsPlaying());
        instance.OnFrame(100);
        assert(instance.effects()[1]->IsPlaying());
        instance.Stop(FALSE);
        assert(!instance.IsPlaying() && !instance.effects()[0]->IsPlaying() &&
            !instance.effects()[1]->IsPlaying());
    }
    {
        auto* source = static_cast<PAPI::PASource*>(PAPI::ParticleManager()->CreateAction(PAPI::PASourceID));
        source->m_Flags.zero(); source->m_Flags.set(PAPI::PASource::flVertexB_tracks, true);
        source->position = PAPI::pDomain(PAPI::PDPoint, 1.f, 2.f, 3.f);
        source->velocity = PAPI::pDomain(PAPI::PDPoint);
        source->rot = PAPI::pDomain(PAPI::PDPoint);
        source->size = PAPI::pDomain(PAPI::PDPoint, 1.f, 1.f, 1.f);
        source->color = PAPI::pDomain(PAPI::PDPoint, 1.f, 1.f, 1.f);
        source->alpha = 1.f; source->particle_rate = 100.f; source->age = source->age_sigma = 0.f;
        source->parent_vel.set(0.f, 0.f, 0.f); source->parent_motion = 0.f;
        CMemoryWriter writer; writer.w_u32(1); writer.w_u32(PAPI::PASourceID); source->Save(writer);
        xr_delete(source);
        auto& emitting = catalog->effects.emplace_back();
        emitting.name = "emitting_parent"; emitting.max_particles = 2; emitting.time_limit = 1.f;
        emitting.actions.assign(writer.pointer(), writer.pointer() + writer.size());
        ParticleGroupDef related_group; related_group.name = "related_group";
        related_group.effects.push_back({"emitting_parent", "smoke", "", "", 0.f, 1.f, 6u});
        VulkanParticleGroup related(catalog, related_group);
        related.Play(); related.OnFrame(40);
        assert(related.effects()[0]->ParticlesCount() == 2 && related.related_count(0) == 2);
        FrameRecordingContext empty_frame{}; DeferredPass empty_pass;
        const float identity[]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        Fvector right{1.f, 0.f, 0.f}, up{0.f, 1.f, 0.f};
        unsigned child_draws = 0;
        const auto try_child = [&](VulkanParticleEffect& root, VulkanParticleEffect& child) {
            assert(&root == related.effects()[0].get() && !std::strcmp(child.Name().c_str(), "smoke"));
            ++child_draws; return true;
        };
        assert(related.record(empty_frame, empty_pass, identity, right, up, false, program_error,
            GeometryPhase::Transparent, 1, {}, try_child) && child_draws == 1);
        child_draws = 0;
        const auto unavailable_child = [&](VulkanParticleEffect&, VulkanParticleEffect&) {
            ++child_draws; return false;
        };
        assert(related.record(empty_frame, empty_pass, identity, right, up, false, program_error,
            GeometryPhase::Transparent, 1, {}, unavailable_child) && child_draws == 1);
        PAPI::ParticleManager()->RemoveParticle(related.effects()[0]->particle_handle(), 0);
        related.OnFrame(1);
        assert(related.related_count(0) == 1);
        PAPI::ParticleManager()->RemoveParticle(related.effects()[0]->particle_handle(), 0);
        related.OnFrame(1);
        assert(related.related_count(0) == 0);
    }
    // A group whose start and end both lie in one update must emit once,
    // matching the legacy particle group's state transition ordering.
    ParticleGroupDef short_group;
    short_group.name = "short_flame";
    short_group.effects.push_back({"smoke", "", "", "", 0.f, .01f, 4u});
    VulkanParticleGroup short_effect(catalog, short_group);
    short_effect.Play();
    short_effect.OnFrame(40);
    assert(short_effect.effects()[0]->IsPlaying());
    }
    Core._destroy();
}
