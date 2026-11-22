#include "ParticleCatalog.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace xray::render::vulkan
{
namespace
{
constexpr uint32_t effect_time_limited = 1u << 14;
uint16_t get16(const uint8_t* p) { return uint16_t(p[0]) | (uint16_t(p[1]) << 8); }
uint32_t get32(const uint8_t* p)
{
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) |
        (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
void put32(uint8_t* dest, uint32_t value)
{
    for (unsigned i = 0; i < sizeof(value); ++i)
        dest[i] = static_cast<uint8_t>(value >> (i * 8));
}

void put_real(uint8_t* dest, float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    put32(dest, bits);
}

struct Cursor
{
    LevelBytes bytes;
    size_t at{};
    bool take(size_t size, const uint8_t*& data)
    {
        if (!bytes.data || at > bytes.size || size > bytes.size - at) return false;
        data = bytes.data + at;
        at += size;
        return true;
    }
    bool word(uint16_t& value)
    {
        const uint8_t* p;
        if (!take(2, p)) return false;
        value = get16(p);
        return true;
    }
    bool integer(uint32_t& value)
    {
        const uint8_t* p;
        if (!take(4, p)) return false;
        value = get32(p);
        return true;
    }
    bool real(float& value)
    {
        uint32_t bits;
        if (!integer(bits)) return false;
        std::memcpy(&value, &bits, 4);
        return std::isfinite(value);
    }
    bool name(std::string& value)
    {
        size_t end = at;
        while (end < bytes.size && bytes.data[end] && end - at < 256) ++end;
        if (end >= bytes.size || end - at >= 256) return false;
        value.assign(reinterpret_cast<const char*>(bytes.data + at), end - at);
        at = end + 1;
        return true;
    }
    bool done() const { return at == bytes.size; }
};

bool chunks(LevelBytes bytes, std::vector<std::pair<uint32_t, LevelBytes>>& result)
{
    Cursor r{bytes};
    while (!r.done())
    {
        uint32_t id, size;
        const uint8_t* p;
        if (!r.integer(id) || !r.integer(size) || (id & 0x80000000u) || !r.take(size, p)) return false;
        result.emplace_back(id, LevelBytes{p, size});
    }
    return true;
}
LevelBytes find(const std::vector<std::pair<uint32_t, LevelBytes>>& chunks, uint32_t id)
{
    for (const auto& chunk : chunks) if (chunk.first == id) return chunk.second;
    return {};
}
bool unique_chunks(const std::vector<std::pair<uint32_t, LevelBytes>>& chunks)
{
    for (size_t i = 0; i < chunks.size(); ++i)
        for (size_t j = i + 1; j < chunks.size(); ++j)
            if (chunks[i].first == chunks[j].first) return false;
    return true;
}

bool effect(LevelBytes bytes, ParticleEffectDef& def, std::string& reason)
{
    const auto reject = [&](const char* message) { reason = message; return false; };
    std::vector<std::pair<uint32_t, LevelBytes>> parts;
    if (!chunks(bytes, parts) || !unique_chunks(parts)) return reject("chunk table");
    Cursor version{find(parts, 1)}, name{find(parts, 2)}, maximum{find(parts, 3)}, flags{find(parts, 5)};
    uint16_t number;
    if (!version.word(number) || number != 1 || !version.done()) return reject("version");
    if (!name.name(def.name) || def.name.empty() || !name.done()) return reject("name");
    if (!maximum.integer(def.max_particles) || !maximum.done()) return reject("particle limit chunk");
    if (!def.max_particles || def.max_particles > 65535) return reject("particle limit range");
    // CPEDef::Load treats PED_CHUNK_FLAGS as optional; the constructor leaves
    // flags at zero when older particle libraries omit it.
    if (flags.bytes.data && (!flags.integer(def.flags) || !flags.done())) return reject("flags chunk");
    auto actions = find(parts, 4);
    if (!actions.data || actions.size < 4 || actions.size > 8 * 1024 * 1024 ||
        get32(actions.data) > 4096 || get32(actions.data) > (actions.size - 4) / 4)
        return reject("action list");
    def.actions.assign(actions.data, actions.data + actions.size);
    if (def.flags & 1u)
    {
        Cursor sprite{find(parts, 7)};
        if (!sprite.name(def.shader) || !sprite.name(def.texture) || def.texture.empty())
            return reject("sprite chunk");
    }
    if (def.flags & (1u << 10))
    {
        Cursor frame{find(parts, 6)};
        float unused[2];
        int32_t columns, count;
        uint32_t bits;
        if (!frame.real(def.frame_size[0]) || !frame.real(def.frame_size[1]) ||
            !frame.real(unused[0]) || !frame.real(unused[1]) ||
            !frame.integer(bits)) return reject("frame chunk");
        columns = static_cast<int32_t>(bits);
        if (!frame.integer(bits)) return reject("frame chunk");
        count = static_cast<int32_t>(bits);
        if (!frame.real(def.frame_rate) || columns <= 0 || count <= 0 ||
            count > 256 || def.frame_size[0] <= 0.f || def.frame_size[1] <= 0.f ||
            def.frame_rate < 0.f) return reject("frame values");
        def.frame_columns = columns;
        def.frame_count = count;
    }
    if (def.flags & effect_time_limited)
    {
        Cursor limit{find(parts, 8)};
        if (!limit.real(def.time_limit) || def.time_limit < 0.f)
            return reject("time limit chunk");
    }
    if (def.flags & (1u << 16))
    {
        Cursor collision{find(parts, 0x21)};
        if (!collision.real(def.collision_friction) || !collision.real(def.collision_resilience) ||
            !collision.real(def.collision_cutoff))
            return reject("collision chunk");
    }
    if(def.flags & (1u<<18))
    {
        const auto scale_bytes = find(parts, 0x22);
        if (!scale_bytes.data)
        {
            // CPEDef starts with a zero velocity scale. Release builds tolerate
            // legacy particle records that set the flag but omit this chunk.
            def.velocity_scale.fill(0.f);
            reason = "defaulted velocity scale to zero";
        }
        else
        {
            Cursor scale{scale_bytes};
            for (size_t i = 0; i < def.velocity_scale.size(); ++i)
            {
                const size_t offset = scale.at;
                if (!scale.real(def.velocity_scale[i]))
                {
                    if (offset + sizeof(uint32_t) > scale_bytes.size)
                        reason = "velocity scale truncated: bytes=" + std::to_string(scale_bytes.size) +
                            " index=" + std::to_string(i);
                    else
                        reason = "velocity scale nonfinite: index=" + std::to_string(i) +
                            " bits=" + std::to_string(get32(scale_bytes.data + offset));
                    return false;
                }
            }
        }
    }
    if(def.flags & (1u<<15))
    {
        const auto bytes=find(parts,0x25);
        if(bytes.data)
        {
            Cursor align{bytes};
            for(float& value:def.align_rotation) if(!align.real(value)) return reject("alignment chunk");
        }
    }
    return true;
}

bool read_group_item(Cursor& r, ParticleGroupItem& item, size_t& timing_offset)
{
    if (!r.name(item.effect) || !r.name(item.on_play) || !r.name(item.on_birth) || !r.name(item.on_dead))
        return false;
    timing_offset = r.at;
    return r.real(item.begin) && r.real(item.end) && r.integer(item.flags);
}

bool group(LevelBytes bytes, ParticleGroupDef& def, std::string& reason)
{
    std::vector<std::pair<uint32_t, LevelBytes>> parts;
    if (!chunks(bytes, parts) || !unique_chunks(parts)) { reason = "chunk table"; return false; }
    Cursor version{find(parts, 1)}, name{find(parts, 2)}, flags{find(parts, 3)};
    uint16_t number;
    if (!version.word(number) || number != 3) { reason = "version"; return false; }
    if (!name.name(def.name) || def.name.empty()) { reason = "name"; return false; }
    // CPGDef::Load reads this optional field with r_chunk, leaving the
    // default Flags32 value when old particle libraries omit it.
    if (flags.bytes.data && !flags.integer(def.flags)) { reason = "flags"; return false; }
    const auto limit = find(parts, 5);
    if (limit.data)
    {
        Cursor r{limit};
        if (!r.real(def.authored_time_limit)) { reason = "time limit"; return false; }
        def.time_limit = def.authored_time_limit;
    }
    const auto effects = find(parts, 4);
    if (effects.data)
    {
        Cursor r{effects};
        uint32_t count;
        if (!r.integer(count) || count > 1024) { reason = "effect count"; return false; }
        for (uint32_t i = 0; i < count; ++i)
        {
            ParticleGroupItem item;
            size_t timing_offset;
            if (!read_group_item(r, item, timing_offset))
            { reason = "effect entry " + std::to_string(i); return false; }
            def.effects.push_back(std::move(item));
        }
    }
    if (def.time_limit <= 0.f)
        for (const auto& item : def.effects) def.time_limit = std::max(def.time_limit, item.end);
    return true;
}
}

const ParticleEffectDef* ParticleCatalog::effect(const std::string& name) const
{
    for (const auto& def : effects) if (def.name == name) return &def;
    return nullptr;
}
const ParticleGroupDef* ParticleCatalog::group(const std::string& name) const
{
    for (const auto& def : groups) if (def.name == name) return &def;
    return nullptr;
}

bool parse_particle_catalog(LevelBytes bytes, ParticleCatalog& result, std::string& error)
{
    if (!bytes.data || bytes.size > 64 * 1024 * 1024)
    { error = "particle catalog is missing or too large"; return false; }
    std::vector<std::pair<uint32_t, LevelBytes>> root;
    Cursor version;
    if (!chunks(bytes, root) || !unique_chunks(root) || !(version.bytes = find(root, 1)).data)
    { error = "invalid particle library chunk table"; return false; }
    uint16_t number;
    if (!version.word(number) || number != 1 || !version.done())
    { error = "unsupported particle library version"; return false; }
    ParticleCatalog parsed;
    size_t skipped_effects = 0, skipped_groups = 0, defaulted_velocity_scales = 0;
    std::string first_skipped;
    for (uint32_t kind : {3u, 4u})
    {
        const auto container = find(root, kind);
        if (!container.data) continue;
        std::vector<std::pair<uint32_t, LevelBytes>> entries;
        if (!chunks(container, entries) || entries.size() > 100000)
        { error = "invalid particle library entries"; return false; }
        for (size_t i = 0; i < entries.size(); ++i)
        {
            if (entries[i].first != i)
            { error = "invalid particle definition index kind=" + std::to_string(kind) +
                " entry=" + std::to_string(i); return false; }
            if (kind == 3)
            {
                ParticleEffectDef def;
                std::string reason;
                if (effect(entries[i].second, def, reason))
                {
                    if (reason == "defaulted velocity scale to zero") ++defaulted_velocity_scales;
                    parsed.effects.push_back(std::move(def));
                }
                else
                {
                    ++skipped_effects;
                    if (first_skipped.empty()) first_skipped = "effect=" + std::to_string(i) + " reason=" + reason;
                }
            }
            else
            {
                ParticleGroupDef def;
                std::string reason;
                if (group(entries[i].second, def, reason)) parsed.groups.push_back(std::move(def));
                else
                {
                    ++skipped_groups;
                    if (first_skipped.empty()) first_skipped = "group=" + std::to_string(i) + " reason=" + reason;
                }
            }
        }
    }
    size_t skipped_items = 0;
    for (auto& def : parsed.groups)
    {
        const auto before = def.effects.size();
        def.effects.erase(std::remove_if(def.effects.begin(), def.effects.end(),
            [&](const ParticleGroupItem& item) { return !parsed.effect(item.effect); }), def.effects.end());
        skipped_items += before - def.effects.size();
        for (auto& item : def.effects)
        {
            if (!item.on_play.empty() && !parsed.effect(item.on_play)) item.flags &= ~(1u << 1);
            if (!item.on_birth.empty() && !parsed.effect(item.on_birth)) item.flags &= ~(1u << 5);
            if (!item.on_dead.empty() && !parsed.effect(item.on_dead)) item.flags &= ~(1u << 6);
        }
    }
    if (parsed.effects.empty() && parsed.groups.empty())
    { error = "no supported particle definitions (" + first_skipped + ")"; return false; }
    result = std::move(parsed);
    error = skipped_effects || skipped_groups || skipped_items ?
        "skipped unsupported definitions: effects=" + std::to_string(skipped_effects) +
        " groups=" + std::to_string(skipped_groups) + " group-items=" + std::to_string(skipped_items) +
        " first=" + first_skipped : defaulted_velocity_scales ?
        "legacy defaults applied: velocity_scale=" + std::to_string(defaulted_velocity_scales) : std::string{};
    return true;
}
bool edit_particle_effect_limit(std::vector<uint8_t>& library, const std::string& name,
    uint32_t max_particles, std::string& error, std::optional<float> time_limit,
    const ParticleSpriteMaterial* material)
{
    if (!max_particles || max_particles > 65535 || name.empty())
    {
        error = "invalid effect name or particle limit";
        return false;
    }
    if (time_limit && (!std::isfinite(*time_limit) || (*time_limit < 0.f && *time_limit != -1.f)))
    {
        error = "effect duration must be nonnegative, or -1 to disable it";
        return false;
    }
    const LevelBytes input{library.data(), library.size()};
    ParticleCatalog validated;
    if (!parse_particle_catalog(input, validated, error))
        return false;
    const auto* definition = validated.effect(name);
    if (!definition)
    {
        error = "particle effect is missing or unsupported: " + name;
        return false;
    }
    std::vector<std::pair<uint32_t, LevelBytes>> root, entries, parts;
    if (!chunks(input, root) || !chunks(find(root, 3), entries))
    {
        error = "invalid particle effect table";
        return false;
    }
    LevelBytes target{};
    LevelBytes record{}, flags{}, duration{}, sprite{};
    for (const auto& entry : entries)
    {
        parts.clear();
        if (!chunks(entry.second, parts))
            continue;
        Cursor field{find(parts, 2)};
        std::string candidate;
        if (!field.name(candidate) || candidate != name)
            continue;
        if (target.data || !unique_chunks(parts) || (target = find(parts, 3)).size != sizeof(uint32_t))
        {
            error = "ambiguous or invalid particle effect record: " + name;
            return false;
        }
        record = entry.second;
        flags = find(parts, 5);
        duration = find(parts, 8);
        sprite = find(parts, 7);
    }
    if (!target.data)
    {
        error = "particle effect record not found: " + name;
        return false;
    }

    // Updating the original chunk preserves fields and action opcodes that
    // the runtime does not interpret, including unknown legacy chunks.
    const auto offset = [&](const uint8_t* field) { return static_cast<size_t>(field - input.data); };
    const auto write = put32;
    std::vector<uint8_t> updated(record.data, record.data + record.size);
    const auto local_offset = [&](LevelBytes field) { return static_cast<size_t>(field.data - record.data); };
    write(updated.data() + local_offset(target), max_particles);
    if (time_limit)
    {
        const bool enabled = *time_limit >= 0.f;
        if ((flags.data && flags.size != 4) || (enabled && duration.data && duration.size < 4))
        {
            error = "invalid effect flags or duration chunk: " + name;
            return false;
        }
        uint32_t bits{};
        std::memcpy(&bits, &*time_limit, sizeof(bits));
        const uint32_t old_flags = flags.data ? get32(flags.data) : 0;
        const uint32_t new_flags = enabled ? old_flags | effect_time_limited : old_flags & ~effect_time_limited;
        const auto append = [&](uint32_t id, uint32_t value)
        {
            const auto start = updated.size();
            updated.resize(start + 12);
            write(updated.data() + start, id);
            write(updated.data() + start + 4, 4);
            write(updated.data() + start + 8, value);
        };
        if (flags.data) write(updated.data() + local_offset(flags), new_flags);
        if (enabled && duration.data) write(updated.data() + local_offset(duration), bits);
        if (enabled && !flags.data) append(5, new_flags);
        if (enabled && !duration.data) append(8, bits);
    }
    if (material)
    {
        const auto valid_name = [](const std::string& value)
        {
            return value.size() < 256 && value.find('\0') == std::string::npos;
        };
        if (!(definition->flags & 1u) || !valid_name(material->shader) ||
            material->texture.empty() || !valid_name(material->texture))
        {
            error = "invalid sprite material for effect: " + name;
            return false;
        }
        const auto start = local_offset(sprite);
        const auto old_names_size = definition->shader.size() + definition->texture.size() + 2;
        std::vector<uint8_t> names(material->shader.c_str(), material->shader.c_str() + material->shader.size() + 1);
        names.insert(names.end(), material->texture.c_str(), material->texture.c_str() + material->texture.size() + 1);
        updated.erase(updated.begin() + start, updated.begin() + start + old_names_size);
        updated.insert(updated.begin() + start, names.begin(), names.end());
        write(updated.data() + start - 4, static_cast<uint32_t>(sprite.size - old_names_size + names.size()));
    }
    const auto table = find(root, 3);
    const auto remaining_table_bytes = table.size - record.size;
    if (updated.size() > UINT32_MAX - remaining_table_bytes)
    {
        error = "effect table exceeds the particle library format limit";
        return false;
    }
    const auto table_offset = offset(table.data);
    const auto record_offset = offset(record.data);
    if (updated.size() > record.size)
        library.insert(library.begin() + record_offset + record.size, updated.size() - record.size, 0);
    else if (updated.size() < record.size)
        library.erase(library.begin() + record_offset + updated.size(), library.begin() + record_offset + record.size);
    std::copy(updated.begin(), updated.end(), library.begin() + record_offset);
    write(library.data() + table_offset - 4, static_cast<uint32_t>(remaining_table_bytes + updated.size()));
    write(library.data() + record_offset - 4, static_cast<uint32_t>(updated.size()));
    error.clear();
    return true;
}

static bool update_particle_group_item(std::vector<uint8_t>& library, const std::string& name,
    size_t item_index, float begin, float end, std::string& error, const std::string* replacement,
    std::optional<ParticleGroupItemOperation> operation, bool add = false,
    const ParticleGroupChildEdits& children = {}, std::optional<float> group_duration = {})
{
    if (!operation && !group_duration &&
        (!std::isfinite(begin) || !std::isfinite(end) || begin < 0.f || end <= begin))
    {
        error = "Group effect end must be greater than its nonnegative start.";
        return false;
    }
    const LevelBytes input{library.data(), library.size()};
    ParticleCatalog catalog;
    if (!parse_particle_catalog(input, catalog, error))
        return false;
    if (replacement && !catalog.effect(*replacement))
    {
        error = "Replacement effect not found: " + *replacement;
        return false;
    }
    for (const auto& child : children)
    {
        if (child && !child->empty() && !catalog.effect(*child))
        {
            error = "Child effect not found: " + *child;
            return false;
        }
    }
    const auto* definition = catalog.group(name);
    if (!definition || (!add && !group_duration && item_index >= definition->effects.size()))
    {
        error = "Group effect entry not found: " + name;
        return false;
    }
    std::vector<std::pair<uint32_t, LevelBytes>> root, groups, parts;
    chunks(input, root);
    const auto table = find(root, 4);
    chunks(table, groups);
    LevelBytes record{}, effects{}, duration{};
    for (const auto& entry : groups)
    {
        parts.clear();
        if (!chunks(entry.second, parts))
            continue;
        Cursor group_name{find(parts, 2)};
        std::string candidate;
        if (!group_name.name(candidate) || candidate != name)
            continue;
        if (record.data || !unique_chunks(parts))
        {
            error = "Ambiguous or invalid group record: " + name;
            return false;
        }
        record = entry.second;
        effects = find(parts, 4);
        duration = find(parts, 5);
    }
    constexpr size_t max_catalog_bytes = 64 * 1024 * 1024;
    if (group_duration && record.data)
    {
        if (duration.data)
        {
            auto* field = library.data() + (duration.data - input.data);
            put_real(field, *group_duration);
        }
        else if (*group_duration > 0.f)
        {
            uint8_t field[12];
            if (sizeof(field) > max_catalog_bytes - library.size())
            {
                error = "Particle catalog exceeds the size limit.";
                return false;
            }
            put32(field, 5);
            put32(field + 4, 4);
            put_real(field + 8, *group_duration);
            const size_t table_offset = table.data - input.data;
            const size_t record_offset = record.data - input.data;
            library.insert(library.begin() + record_offset + record.size, field, field + sizeof(field));
            put32(library.data() + table_offset - 4, static_cast<uint32_t>(table.size + sizeof(field)));
            put32(library.data() + record_offset - 4, static_cast<uint32_t>(record.size + sizeof(field)));
        }
        error.clear();
        return true;
    }
    if (add && record.data)
    {
        Cursor cursor{effects};
        uint32_t count = 0;
        if (effects.data && (!cursor.integer(count) || count >= 1024))
        {
            error = "The group has reached the particle effect limit.";
            return false;
        }
        for (uint32_t i = 0; i < count; ++i)
        {
            ParticleGroupItem item;
            size_t timing_offset;
            if (!read_group_item(cursor, item, timing_offset))
            {
                error = "Invalid group effect entry: " + name;
                return false;
            }
        }
        const auto prefix = effects.data ? 0u : 12u;
        std::vector<uint8_t> bytes(prefix + replacement->size() + 4 + 12, 0);
        if (bytes.size() > max_catalog_bytes - library.size())
        {
            error = "Particle catalog exceeds the size limit.";
            return false;
        }
        std::copy(replacement->begin(), replacement->end(), bytes.begin() + prefix);
        auto* timing = bytes.data() + prefix + replacement->size() + 4;
        put_real(timing, begin);
        put_real(timing + 4, end);
        put32(timing + 8, 1u << 2); // Enabled
        if (!effects.data)
        {
            put32(bytes.data(), 4);
            put32(bytes.data() + 4, static_cast<uint32_t>(bytes.size() - 8));
            put32(bytes.data() + 8, 1);
        }
        const size_t table_offset = table.data - input.data;
        const size_t record_offset = record.data - input.data;
        const size_t effects_offset = effects.data ? effects.data - input.data : record_offset + record.size;
        library.insert(library.begin() + effects_offset + cursor.at, bytes.begin(), bytes.end());
        put32(library.data() + table_offset - 4, static_cast<uint32_t>(table.size + bytes.size()));
        put32(library.data() + record_offset - 4, static_cast<uint32_t>(record.size + bytes.size()));
        if (effects.data)
        {
            put32(library.data() + effects_offset - 4, static_cast<uint32_t>(effects.size + bytes.size()));
            put32(library.data() + effects_offset, count + 1);
        }
        error.clear();
        return true;
    }
    Cursor cursor{effects};
    uint32_t count;
    if (cursor.integer(count))
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            const auto item_start = cursor.at;
            ParticleGroupItem item;
            size_t timing_offset;
            if (!read_group_item(cursor, item, timing_offset))
                break;
            // Unsupported effects are hidden in the editor, but remain in the file.
            if (!catalog.effect(item.effect) || item_index-- != 0)
                continue;
            const auto effects_offset = static_cast<size_t>(effects.data - input.data);
            const auto table_offset = static_cast<size_t>(table.data - input.data);
            const auto record_offset = static_cast<size_t>(record.data - input.data);
            const auto resize_chunks = [&](size_t removed, size_t added)
            {
                put32(library.data() + table_offset - 4, static_cast<uint32_t>(table.size - removed + added));
                put32(library.data() + record_offset - 4, static_cast<uint32_t>(record.size - removed + added));
                put32(library.data() + effects_offset - 4, static_cast<uint32_t>(effects.size - removed + added));
            };
            if (operation)
            {
                const auto start = effects_offset + item_start;
                const auto finish = effects_offset + cursor.at;
                const auto size = finish - start;
                if (*operation == ParticleGroupItemOperation::Duplicate)
                {
                    if (count >= 1024 || size > max_catalog_bytes - library.size())
                    {
                        error = "Duplicating this entry would exceed the particle library limits.";
                        return false;
                    }
                    const std::vector<uint8_t> copy(library.begin() + start, library.begin() + finish);
                    library.insert(library.begin() + finish, copy.begin(), copy.end());
                    resize_chunks(0, size);
                    put32(library.data() + effects_offset, count + 1);
                }
                else
                {
                    library.erase(library.begin() + start, library.begin() + finish);
                    resize_chunks(size, 0);
                    put32(library.data() + effects_offset, count - 1);
                }
                error.clear();
                return true;
            }
            if (replacement || std::any_of(children.begin(), children.end(), [](const auto& child)
                { return child.has_value(); }))
            {
                std::array<const std::string*, 4> names{
                    replacement ? replacement : &item.effect, &item.on_play, &item.on_birth, &item.on_dead};
                for (size_t child = 0; child < children.size(); ++child)
                {
                    if (!children[child])
                        continue;
                    names[child + 1] = &*children[child];
                    const auto flag = particle_group_child_flags[child];
                    if (children[child]->empty())
                        item.flags &= ~flag;
                    else
                        item.flags |= flag;
                }
                std::vector<uint8_t> serialized;
                for (const auto* field : names)
                {
                    serialized.insert(serialized.end(), field->begin(), field->end());
                    serialized.push_back(0);
                }
                const auto old_size = timing_offset - item_start;
                const auto new_size = serialized.size();
                if (new_size > max_catalog_bytes - library.size() + old_size)
                {
                    error = "Particle catalog exceeds the size limit.";
                    return false;
                }
                const auto start = effects_offset + item_start;
                if (new_size > old_size)
                    library.insert(library.begin() + start + old_size, new_size - old_size, 0);
                else if (new_size < old_size)
                    library.erase(library.begin() + start + new_size, library.begin() + start + old_size);
                std::copy(serialized.begin(), serialized.end(), library.begin() + start);
                resize_chunks(old_size, new_size);
                timing_offset = timing_offset - old_size + new_size;
            }
            auto* dest = library.data() + effects_offset + timing_offset;
            put_real(dest, begin);
            put_real(dest + 4, end);
            put32(dest + 8, item.flags);
            error.clear();
            return true;
        }
    }
    error = "Group effect record not found: " + name;
    return false;
}

bool edit_particle_group_item(std::vector<uint8_t>& library, const std::string& name,
    size_t item_index, float begin, float end, std::string& error, const std::string* replacement,
    const ParticleGroupChildEdits& children)
{
    return update_particle_group_item(library, name, item_index, begin, end, error, replacement, {}, false, children);
}

bool change_particle_group_items(std::vector<uint8_t>& library, const std::string& name,
    size_t item_index, ParticleGroupItemOperation operation, std::string& error)
{
    return update_particle_group_item(library, name, item_index, 0.f, 0.f, error, nullptr, operation);
}

bool add_particle_group_item(std::vector<uint8_t>& library, const std::string& group_name,
    const std::string& effect_name, float begin, float end, std::string& error)
{
    return update_particle_group_item(library, group_name, 0, begin, end, error, &effect_name, {}, true);
}

bool edit_particle_group_duration(std::vector<uint8_t>& library, const std::string& group_name,
    float seconds, std::string& error)
{
    if (!std::isfinite(seconds) || seconds < 0.f)
    {
        error = "Group duration must be nonnegative.";
        return false;
    }
    return update_particle_group_item(library, group_name, 0, 0.f, 0.f, error, nullptr, {}, false, {}, seconds);
}
} // namespace xray::render::vulkan
