#include "src/Layers/xrRenderVK/ParticleCatalog.h"

#include <cassert>
#include <cstring>

using namespace xray::render::vulkan;
using Bytes = std::vector<uint8_t>;

static void u16(Bytes& b, uint16_t n) { b.push_back(uint8_t(n)); b.push_back(uint8_t(n >> 8)); }
static void u32(Bytes& b, uint32_t n)
{
    for (int i = 0; i < 4; ++i) b.push_back(uint8_t(n >> (8 * i)));
}
static void real(Bytes& b, float n)
{
    uint32_t bits;
    std::memcpy(&bits, &n, 4);
    u32(b, bits);
}
static void name(Bytes& b, const char* value)
{
    while (*value) b.push_back(uint8_t(*value++));
    b.push_back(0);
}
static void chunk(Bytes& b, uint32_t id, const Bytes& contents)
{
    u32(b, id); u32(b, static_cast<uint32_t>(contents.size()));
    b.insert(b.end(), contents.begin(), contents.end());
}

int main()
{
    Bytes effect, group, effects, groups, library, data;
    Bytes version; u16(version, 1);
    chunk(effect, 1, version);
    name(data, "campfire"); chunk(effect, 2, data); data.clear();
    u32(data, 64); chunk(effect, 3, data); data.clear();
    chunk(effect, 4, Bytes{0, 0, 0, 0});
    u32(data, 1); chunk(effect, 5, data); data.clear();
    name(data, "particle"); name(data, "fire"); chunk(effect, 7, data); data.clear();
    chunk(effects, 0, effect);
    Bytes group_version; u16(group_version, 3);
    chunk(group, 1, group_version);
    name(data, "campfire_group"); chunk(group, 2, data); data.clear();
    u32(data, 0); chunk(group, 3, data); data.clear();
    u32(data, 1);
    name(data, "campfire"); name(data, ""); name(data, ""); name(data, "");
    real(data, 0.f); real(data, 2.f); u32(data, 5);
    chunk(group, 4, data);
    chunk(groups, 0, group);
    chunk(library, 1, version);
    chunk(library, 3, effects);
    chunk(library, 4, groups);
    ParticleCatalog catalog;
    std::string error;
    assert(parse_particle_catalog({library.data(), library.size()}, catalog, error));
    assert(catalog.effect("campfire") && catalog.effect("campfire")->max_particles == 64);
    assert(catalog.group("campfire_group") && catalog.group("campfire_group")->effects.size() == 1);
    Bytes legacy_effect, legacy_effects, legacy_effect_library;
    chunk(legacy_effect, 1, version);
    data.clear();
    name(data, "legacy_campfire"); chunk(legacy_effect, 2, data); data.clear();
    u32(data, 64); chunk(legacy_effect, 3, data); data.clear();
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
    u32(data, 64); chunk(trailing_effect, 3, data); data.clear();
    chunk(trailing_effect, 4, Bytes{0, 0, 0, 0});
    u32(data, 1 | (1u << 10) | (1u << 14) | (1u << 15) | (1u << 18));
    chunk(trailing_effect, 5, data); data.clear();
    name(data, "particle"); name(data, "fire"); data.push_back(0xaa);
    chunk(trailing_effect, 7, data); data.clear();
    real(data, 1.f); real(data, 1.f); real(data, 0.f); real(data, 0.f);
    u32(data, 8); u32(data, 16); real(data, 24.f); u32(data, 0xdeadbeef);
    chunk(trailing_effect, 6, data); data.clear();
    real(data, 5.f); u32(data, 0xdeadbeef); chunk(trailing_effect, 8, data); data.clear();
    real(data, 1.f); real(data, 2.f); real(data, 3.f); u32(data, 0xdeadbeef);
    chunk(trailing_effect, 0x22, data); data.clear();
    real(data, .1f); real(data, .2f); real(data, .3f); u32(data, 0xdeadbeef);
    chunk(trailing_effect, 0x25, data); data.clear();
    chunk(trailing_effects, 0, trailing_effect);
    chunk(trailing_library, 1, version); chunk(trailing_library, 3, trailing_effects);
    assert(parse_particle_catalog({trailing_library.data(), trailing_library.size()}, catalog, error));
    assert(catalog.effect("legacy_trailing") && catalog.effect("legacy_trailing")->velocity_scale[2] == 3.f &&
        catalog.effect("legacy_trailing")->align_rotation[1] == .2f);
    Bytes missing_scale_effect, missing_scale_effects, missing_scale_library;
    chunk(missing_scale_effect, 1, version);
    data.clear(); name(data, "legacy_missing_scale"); chunk(missing_scale_effect, 2, data); data.clear();
    u32(data, 64); chunk(missing_scale_effect, 3, data); data.clear();
    chunk(missing_scale_effect, 4, Bytes{0, 0, 0, 0});
    u32(data, 1 | (1u << 18)); chunk(missing_scale_effect, 5, data); data.clear();
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
        u32(values,64);chunk(def,3,values);values.clear();chunk(def,4,Bytes{0,0,0,0});
        u32(values,1|(1u<<15)|(1u<<18));chunk(def,5,values);values.clear();
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
    data.clear(); u32(data, 1); name(data, "campfire"); name(data, "missing_child");
    name(data, ""); name(data, ""); real(data, 0.f); real(data, 2.f);
    u32(data, 5 | (1u << 1)); chunk(legacy_group, 4, data);
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
