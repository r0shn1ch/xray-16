#pragma once

#include "LevelModels.h"

#include <array>
#include <optional>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct ParticleEffectDef
{
    std::string name, shader, texture;
    uint32_t flags{}, max_particles{};
    float time_limit{-1.f};
    float collision_friction{1.f}, collision_resilience{}, collision_cutoff{};
    std::array<float, 2> frame_size{1.f, 1.f};
    uint32_t frame_columns{1}, frame_count{1};
    float frame_rate{24.f};
    std::array<float,3> velocity_scale{};
    std::array<float,3> align_rotation{-1.57079632679f,0.f,0.f};
    std::vector<uint8_t> actions;
};

struct ParticleGroupItem
{
    std::string effect, on_play, on_birth, on_dead;
    float begin{}, end{};
    uint32_t flags{};
};

struct ParticleGroupDef
{
    std::string name;
    uint32_t flags{};
    float time_limit{};
    float authored_time_limit{};
    std::vector<ParticleGroupItem> effects;
};

struct ParticleCatalog
{
    std::vector<ParticleEffectDef> effects;
    std::vector<ParticleGroupDef> groups;
    const ParticleEffectDef* effect(const std::string& name) const;
    const ParticleGroupDef* group(const std::string& name) const;
};

bool parse_particle_catalog(LevelBytes bytes, ParticleCatalog& result, std::string& error);
bool duplicate_particle_effect(std::vector<uint8_t>& library, const std::string& source,
    const std::string& name, std::string& error);
struct ParticleSpriteMaterial
{
    std::string shader, texture;
};

bool edit_particle_effect_limit(std::vector<uint8_t>& library, const std::string& name,
    uint32_t max_particles, std::string& error, std::optional<float> time_limit = {},
    const ParticleSpriteMaterial* material = nullptr);
using ParticleGroupChildEdits = std::array<std::optional<std::string>, 3>; // Attached, birth, death
inline constexpr std::array<uint32_t, 3> particle_group_child_flags{1u << 1, 1u << 5, 1u << 6};
bool edit_particle_group_item(std::vector<uint8_t>& library, const std::string& name,
    size_t item_index, float begin, float end, std::string& error, const std::string* replacement = nullptr,
    const ParticleGroupChildEdits& children = {});
enum class ParticleGroupItemOperation { Duplicate, Remove };
bool change_particle_group_items(std::vector<uint8_t>& library, const std::string& name,
    size_t item_index, ParticleGroupItemOperation operation, std::string& error);
bool add_particle_group_item(std::vector<uint8_t>& library, const std::string& group_name,
    const std::string& effect_name, float begin, float end, std::string& error);
bool edit_particle_group_duration(std::vector<uint8_t>& library, const std::string& group_name,
    float seconds, std::string& error);
} // namespace xray::render::vulkan
