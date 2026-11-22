#include "xrEngine/stdafx.h"
#include "src/Layers/xrRenderVK/ParticleCatalog.h"
#include "src/Layers/xrRenderVK/EngineParticleSource.h"

#include <cassert>
#include <cstring>

using namespace xray::render::vulkan;
using Bytes = std::vector<uint8_t>;

static void put_u16(Bytes& b, uint16_t n) { b.push_back(uint8_t(n)); b.push_back(uint8_t(n >> 8)); }
static void put_u32(Bytes& b, uint32_t n)
{
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(n >> (8 * i)));
}
static void real(Bytes& b, float n)
{
    uint32_t bits;
    std::memcpy(&bits, &n, 4);
    put_u32(b, bits);
}
static void name(Bytes& b, const char* value)
{
    while (*value) b.push_back(uint8_t(*value++));
    b.push_back(0);
}
static void chunk(Bytes& b, uint32_t id, const Bytes& contents)
{
    put_u32(b, id); put_u32(b, static_cast<uint32_t>(contents.size()));
    b.insert(b.end(), contents.begin(), contents.end());
}

int main()
{
    Bytes effect, group, effects, groups, library, data;
    Bytes version; put_u16(version, 1);
    chunk(effect, 1, version);
    name(data, "campfire"); chunk(effect, 2, data); data.clear();
    put_u32(data, 64); chunk(effect, 3, data); data.clear();
    chunk(effect, 4, Bytes{0, 0, 0, 0});
    put_u32(data, 1); chunk(effect, 5, data); data.clear();
    name(data, "particle"); name(data, "fire"); chunk(effect, 7, data); data.clear();
    chunk(effect, 0x7ff0, Bytes{0xde, 0xad, 0xbe, 0xef});
    chunk(effects, 0, effect);
    Bytes group_version; put_u16(group_version, 3);
    chunk(group, 1, group_version);
    name(data, "campfire_group"); chunk(group, 2, data); data.clear();
    put_u32(data, 0); chunk(group, 3, data); data.clear();
    const Bytes group_header = group;
    put_u32(data, 1);
    name(data, "campfire"); name(data, ""); name(data, ""); name(data, "");
    real(data, 0.f); real(data, 2.f); put_u32(data, 5);
    chunk(group, 4, data);
    chunk(groups, 0, group);
    chunk(library, 1, version);
    chunk(library, 3, effects);
    chunk(library, 4, groups);
    ParticleCatalog catalog;
    std::string error;
    assert(parse_particle_catalog({library.data(), library.size()}, catalog, error));
    assert(catalog.effect("campfire") && catalog.effect("campfire")->max_particles == 64);
    assert(catalog.effect("campfire")->collision_friction == 1.f);
    assert(catalog.group("campfire_group") && catalog.group("campfire_group")->effects.size() == 1);
    Bytes timing_items, timing_groups, timing_library;
    put_u32(timing_items, 2);
    name(timing_items, "unavailable");
    for (int i = 0; i < 3; ++i) name(timing_items, "");
    real(timing_items, 0.f); real(timing_items, 1.f); put_u32(timing_items, 4);
    timing_items.insert(timing_items.end(), data.begin() + 4, data.end());
    Bytes timing_group = group_header;
    chunk(timing_group, 4, timing_items);
    chunk(timing_groups, 0, timing_group);
    chunk(timing_library, 1, version);
    Bytes replacement_def, replacement_name, timing_effects = effects;
    chunk(replacement_def, 1, version);
    name(replacement_name, "campfire_replacement");
    chunk(replacement_def, 2, replacement_name);
    chunk(replacement_def, 3, Bytes{64, 0, 0, 0});
    chunk(replacement_def, 4, Bytes{0, 0, 0, 0});
    chunk(timing_effects, 1, replacement_def);
    chunk(timing_library, 3, timing_effects);
    chunk(timing_library, 4, timing_groups);
    Bytes retimed = timing_library;
    assert(!edit_particle_group_item(retimed, "campfire_group", 0, 3.f, 2.f, error) && retimed == timing_library);
    const std::string replacement = "campfire_replacement";
    assert(edit_particle_group_item(retimed, "campfire_group", 0, .5f, 4.f, error, &replacement));
    assert(parse_particle_catalog({retimed.data(), retimed.size()}, catalog, error));
    const auto* retimed_group = catalog.group("campfire_group");
    assert(retimed_group && retimed_group->effects.size() == 1 && retimed_group->effects[0].begin == .5f &&
        retimed_group->effects[0].end == 4.f && retimed_group->time_limit == 4.f &&
        retimed_group->effects[0].effect == replacement && retimed_group->effects[0].flags == 5);
    const std::string original = "campfire";
    assert(edit_particle_group_item(retimed, "campfire_group", 0, 0.f, 2.f, error, &original) &&
        retimed == timing_library);
    ParticleGroupChildEdits children{replacement, original, replacement};
    assert(edit_particle_group_item(retimed, "campfire_group", 0, 0.f, 2.f, error, nullptr, children));
    assert(parse_particle_catalog({retimed.data(), retimed.size()}, catalog, error));
    const auto& child_entry = catalog.group("campfire_group")->effects.front();
    assert(child_entry.on_play == replacement && child_entry.on_birth == original &&
        child_entry.on_dead == replacement && child_entry.flags == (5u | 2u | 32u | 64u));
    children.fill(std::string{});
    assert(edit_particle_group_item(retimed, "campfire_group", 0, 0.f, 2.f, error, nullptr, children) &&
        retimed == timing_library);
    assert(change_particle_group_items(retimed, "campfire_group", 0, ParticleGroupItemOperation::Duplicate, error));
    assert(parse_particle_catalog({retimed.data(), retimed.size()}, catalog, error));
    assert(catalog.group("campfire_group")->effects.size() == 2);
    assert(change_particle_group_items(retimed, "campfire_group", 1, ParticleGroupItemOperation::Remove, error) &&
        retimed == timing_library);
    assert(change_particle_group_items(retimed, "campfire_group", 0, ParticleGroupItemOperation::Remove, error));
    assert(parse_particle_catalog({retimed.data(), retimed.size()}, catalog, error));
    assert(catalog.group("campfire_group")->effects.empty());
    assert(add_particle_group_item(retimed, "campfire_group", original, .25f, 1.5f, error));
    assert(parse_particle_catalog({retimed.data(), retimed.size()}, catalog, error));
    const auto& appended = catalog.group("campfire_group")->effects;
    assert(appended.size() == 1 && appended[0].effect == original && appended[0].begin == .25f &&
        appended[0].end == 1.5f && appended[0].flags == 4);
    Bytes no_items_groups, no_items_library;
    chunk(no_items_groups, 0, group_header);
    chunk(no_items_library, 1, version);
    chunk(no_items_library, 3, effects);
    chunk(no_items_library, 4, no_items_groups);
    assert(add_particle_group_item(no_items_library, "campfire_group", original, 0.f, 1.f, error));
    assert(parse_particle_catalog({no_items_library.data(), no_items_library.size()}, catalog, error));
    assert(catalog.group("campfire_group")->effects.size() == 1);
    Bytes edited = library;
    IReader editor_input(library.data(), library.size());
    assert(editable_particle_library_bytes(editor_input, edited, error) && edited == library);
    CMemoryWriter compressed;
    compressed.w_chunk(1, version.data(), version.size());
    compressed.w_chunk(3 | CFS_CompressMark, effects.data(), effects.size());
    compressed.w_chunk(4 | CFS_CompressMark, groups.data(), groups.size());
    IReader compressed_input(compressed.pointer(), compressed.size());
    assert(editable_particle_library_bytes(compressed_input, edited, error) && edited == library);
    assert(edit_particle_effect_limit(edited, "campfire", 512, error));
    assert(parse_particle_catalog({edited.data(), edited.size()}, catalog, error));
    assert(catalog.effect("campfire")->max_particles == 512);
    const Bytes before_invalid_edit = edited;
    assert(!edit_particle_effect_limit(edited, "campfire", 65536, error) && edited == before_invalid_edit);
    assert(!edit_particle_effect_limit(edited, "missing", 128, error) && edited == before_invalid_edit);
    assert(edit_particle_effect_limit(edited, "campfire", 64, error) && edited == library);
    assert(edit_particle_effect_limit(edited, "campfire", 512, error, 2.5f));
    assert(edited.size() == library.size() + 12);
    assert(parse_particle_catalog({edited.data(), edited.size()}, catalog, error));
    assert(catalog.effect("campfire")->time_limit == 2.5f &&
        catalog.effect("campfire")->max_particles == 512 && catalog.effect("campfire")->texture == "fire");
    assert(catalog.group("campfire_group")->effects.size() == 1);
    const Bytes timed = edited;
    assert(!edit_particle_effect_limit(edited, "campfire", 512, error, -2.f) && edited == timed);
    assert(edit_particle_effect_limit(edited, "campfire", 512, error, -1.f));
    assert(parse_particle_catalog({edited.data(), edited.size()}, catalog, error));
    assert(catalog.effect("campfire")->time_limit == -1.f);
    assert(edit_particle_effect_limit(edited, "campfire", 512, error, 2.5f) && edited == timed);
    const ParticleSpriteMaterial smoke{"particles\\blend", "fx\\campfire_smoke"};
    assert(edit_particle_effect_limit(edited, "campfire", 128, error, 3.f, &smoke));
    assert(parse_particle_catalog({edited.data(), edited.size()}, catalog, error));
    const auto* smoke_effect = catalog.effect("campfire");
    assert(smoke_effect && smoke_effect->shader == smoke.shader && smoke_effect->texture == smoke.texture &&
        smoke_effect->max_particles == 128 && smoke_effect->time_limit == 3.f);
    assert(catalog.group("campfire_group")->effects.size() == 1);
    const ParticleSpriteMaterial fire{"particle", "fire"};
    assert(edit_particle_effect_limit(edited, "campfire", 512, error, 2.5f, &fire) && edited == timed);
    const auto collision_library = [&](bool include, bool truncated) {
        Bytes def, values, table, lib;
        chunk(def, 1, version); name(values, "collision"); chunk(def, 2, values); values.clear();
        put_u32(values, 64); chunk(def, 3, values); values.clear(); chunk(def, 4, Bytes{0, 0, 0, 0});
        put_u32(values, 1u << 16); chunk(def, 5, values); values.clear();
        real(values, .75f); real(values, .5f); real(values, .25f);
        if (truncated) values.pop_back();
        if (include) chunk(def, 0x21, values);
        chunk(table, 0, def); chunk(lib, 1, version); chunk(lib, 3, table);
        return lib;
    };
    auto collision = collision_library(true, false);
    assert(parse_particle_catalog({collision.data(), collision.size()}, catalog, error));
    assert(catalog.effect("collision") && catalog.effect("collision")->collision_friction == .75f &&
        catalog.effect("collision")->collision_resilience == .5f && catalog.effect("collision")->collision_cutoff == .25f);
    for (const auto& bad : {collision_library(false, false), collision_library(true, true)})
    {
        assert(!parse_particle_catalog({bad.data(), bad.size()}, catalog, error));
        assert(error.find("collision chunk") != std::string::npos);
    }
    // Exercise the production IReader adapter as well as the portable parser:
    // PED extended chunk ids must survive normalization or valid effects are
    // silently dropped before catalog parsing.
    Bytes source_effect, source_effects, source_library, source_field;
    chunk(source_effect, 1, version);
    name(source_field, "source_collision"); chunk(source_effect, 2, source_field); source_field.clear();
    put_u32(source_field, 32); chunk(source_effect, 3, source_field); source_field.clear();
    chunk(source_effect, 4, Bytes{0, 0, 0, 0});
    put_u32(source_field, (1u << 16) | (1u << 18) | (1u << 15));
    chunk(source_effect, 5, source_field); source_field.clear();
    real(source_field, .75f); real(source_field, .5f); real(source_field, .25f);
    chunk(source_effect, 0x21, source_field); source_field.clear();
    real(source_field, 1.f); real(source_field, 2.f); real(source_field, 3.f);
    chunk(source_effect, 0x22, source_field); source_field.clear();
    real(source_field, .1f); real(source_field, .2f); real(source_field, .3f);
    chunk(source_effect, 0x25, source_field); source_field.clear();
    chunk(source_effects, 0, source_effect);
    chunk(source_library, 1, version); chunk(source_library, 3, source_effects);
    IReader source_reader(source_library.data(), source_library.size());
    const auto normalized_source = particle_library_bytes(source_reader);
    assert(parse_particle_catalog({normalized_source.data(), normalized_source.size()}, catalog, error));
    const auto* normalized_effect = catalog.effect("source_collision");
    assert(normalized_effect && normalized_effect->collision_friction == .75f &&
        normalized_effect->collision_resilience == .5f && normalized_effect->collision_cutoff == .25f &&
        normalized_effect->velocity_scale[0] == 1.f && normalized_effect->velocity_scale[2] == 3.f &&
        normalized_effect->align_rotation[0] == .1f && normalized_effect->align_rotation[2] == .3f);
    Bytes legacy_effect, legacy_effects, legacy_effect_library;
    chunk(legacy_effect, 1, version);
    data.clear();
    name(data, "legacy_campfire"); chunk(legacy_effect, 2, data); data.clear();
    put_u32(data, 64); chunk(legacy_effect, 3, data); data.clear();
    chunk(legacy_effect, 4, Bytes{0, 0, 0, 0});
    name(data, "particle"); name(data, "fire"); chunk(legacy_effect, 7, data); data.clear();
    chunk(legacy_effects, 0, legacy_effect);
    chunk(legacy_effect_library, 1, version);
    chunk(legacy_effect_library, 3, legacy_effects);
    assert(parse_particle_catalog({legacy_effect_library.data(), legacy_effect_library.size()}, catalog, error));
    assert(catalog.effect("legacy_campfire") && catalog.effect("legacy_campfire")->flags == 0);
    Bytes trailing_effect, trailing_effects, trailing_library;
    chunk(trailing_effect, 1, version);
    data.clear(); name(data, "legacy_trailing"); chunk(trailing_effect, 2, data); data.clear();
    put_u32(data, 64); chunk(trailing_effect, 3, data); data.clear();
    chunk(trailing_effect, 4, Bytes{0, 0, 0, 0});
    put_u32(data, 1 | (1u << 10) | (1u << 14) | (1u << 15) | (1u << 18));
    chunk(trailing_effect, 5, data); data.clear();
    name(data, "particle"); name(data, "fire"); data.push_back(0xaa);
    chunk(trailing_effect, 7, data); data.clear();
    real(data, 1.f); real(data, 1.f); real(data, 0.f); real(data, 0.f);
    put_u32(data, 8); put_u32(data, 16); real(data, 24.f); put_u32(data, 0xdeadbeef);
    chunk(trailing_effect, 6, data); data.clear();
    real(data, 5.f); put_u32(data, 0xdeadbeef); chunk(trailing_effect, 8, data); data.clear();
    real(data, 1.f); real(data, 2.f); real(data, 3.f); put_u32(data, 0xdeadbeef);
    chunk(trailing_effect, 0x22, data); data.clear();
    real(data, .1f); real(data, .2f); real(data, .3f); put_u32(data, 0xdeadbeef);
    chunk(trailing_effect, 0x25, data); data.clear();
    chunk(trailing_effects, 0, trailing_effect);
    chunk(trailing_library, 1, version); chunk(trailing_library, 3, trailing_effects);
    assert(parse_particle_catalog({trailing_library.data(), trailing_library.size()}, catalog, error));
    assert(catalog.effect("legacy_trailing") && catalog.effect("legacy_trailing")->velocity_scale[2] == 3.f &&
        catalog.effect("legacy_trailing")->align_rotation[1] == .2f);
    Bytes missing_scale_effect, missing_scale_effects, missing_scale_library;
    chunk(missing_scale_effect, 1, version);
    data.clear(); name(data, "legacy_missing_scale"); chunk(missing_scale_effect, 2, data); data.clear();
    put_u32(data, 64); chunk(missing_scale_effect, 3, data); data.clear();
    chunk(missing_scale_effect, 4, Bytes{0, 0, 0, 0});
    put_u32(data, 1 | (1u << 18)); chunk(missing_scale_effect, 5, data); data.clear();
    name(data, "particle"); name(data, "smoke"); chunk(missing_scale_effect, 7, data); data.clear();
    chunk(missing_scale_effects, 0, missing_scale_effect);
    chunk(missing_scale_library, 1, version); chunk(missing_scale_library, 3, missing_scale_effects);
    assert(parse_particle_catalog({missing_scale_library.data(), missing_scale_library.size()}, catalog, error));
    assert(catalog.effect("legacy_missing_scale") &&
        catalog.effect("legacy_missing_scale")->velocity_scale[0] == 0.f &&
        error.find("velocity_scale=1") != std::string::npos);
    const auto aligned_library=[&](bool scale_chunk,bool align_chunk,bool truncated) {
        Bytes def,values,table,lib;
        chunk(def,1,version);name(values,"campfire");chunk(def,2,values);values.clear();
        put_u32(values,64);chunk(def,3,values);values.clear();chunk(def,4,Bytes{0,0,0,0});
        put_u32(values,1|(1u<<15)|(1u<<18));chunk(def,5,values);values.clear();
        name(values,"particle");name(values,"fire");chunk(def,7,values);values.clear();
        real(values,1);real(values,2);real(values,3);if(scale_chunk)chunk(def,0x22,values);values.clear();
        real(values,.1f);real(values,.2f);real(values,.3f);if(truncated)values.pop_back();if(align_chunk)chunk(def,0x25,values);
        chunk(table,0,def);chunk(lib,1,version);chunk(lib,3,table);chunk(lib,4,groups);return lib;
    };
    auto aligned=aligned_library(true,true,false);
    assert(parse_particle_catalog({aligned.data(),aligned.size()},catalog,error));
    assert(catalog.effect("campfire")->velocity_scale[1]==2 && catalog.effect("campfire")->align_rotation[2]==.3f);
    aligned=aligned_library(true,false,false);
    assert(parse_particle_catalog({aligned.data(),aligned.size()},catalog,error));
    assert(catalog.effect("campfire")->align_rotation[0]<-1.57f);
    aligned=aligned_library(false,true,false);
    assert(parse_particle_catalog({aligned.data(),aligned.size()},catalog,error));
    assert(catalog.effect("campfire") && catalog.effect("campfire")->velocity_scale[0] == 0.f &&
        error.find("velocity_scale=1") != std::string::npos);
    aligned=aligned_library(true,true,true);
    assert(parse_particle_catalog({aligned.data(),aligned.size()},catalog,error));
    assert(!catalog.effect("campfire") && error.find("effects=1")!=std::string::npos);
    // Older libraries can omit the optional flags chunk; a missing child
    // effect must not make the entire group disappear.
    Bytes legacy_group, legacy_groups, legacy_library;
    chunk(legacy_group, 1, group_version);
    data.clear(); name(data, "legacy_group"); chunk(legacy_group, 2, data);
    data.clear(); put_u32(data, 1); name(data, "campfire"); name(data, "missing_child");
    name(data, ""); name(data, ""); real(data, 0.f); real(data, 2.f);
    put_u32(data, 5 | (1u << 1)); chunk(legacy_group, 4, data);
    chunk(legacy_groups, 0, legacy_group);
    chunk(legacy_library, 1, version); chunk(legacy_library, 3, effects);
    chunk(legacy_library, 4, legacy_groups);
    assert(parse_particle_catalog({legacy_library.data(), legacy_library.size()}, catalog, error));
    assert(catalog.group("legacy_group") && catalog.group("legacy_group")->effects.size() == 1);
    assert(!(catalog.group("legacy_group")->effects[0].flags & (1u << 1)));
    Bytes wrong_group = group;
    wrong_group[8] = 1;
    Bytes wrong_groups, wrong_library;
    chunk(wrong_groups, 0, wrong_group);
    chunk(wrong_library, 1, version);
    chunk(wrong_library, 3, effects);
    chunk(wrong_library, 4, wrong_groups);
    assert(parse_particle_catalog({wrong_library.data(), wrong_library.size()}, catalog, error));
    assert(catalog.effect("campfire") && catalog.groups.empty());
    assert(error.find("groups=1") != std::string::npos);
    const auto previous = catalog;
    library.pop_back();
    assert(!parse_particle_catalog({library.data(), library.size()}, catalog, error));
    assert(catalog.effects.size() == previous.effects.size() && catalog.groups.size() == previous.groups.size());
}
