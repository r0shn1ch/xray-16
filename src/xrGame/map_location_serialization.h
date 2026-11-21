#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace map_location_serialization
{
struct StringView
{
    const char* data{};
    size_t size{};
};

struct StateRecord
{
    StringView hint;
    uint32_t flags{};
    StringView owner_task_id;
    size_t consumed{};
};

struct KeyRecord
{
    uint16_t object_id{};
    StringView spot_type;
    size_t header_size{};
    size_t consumed{};
};

inline bool read_string_z(const char* data, size_t size, StringView& value, size_t& consumed) noexcept
{
    if (!data || !size)
        return false;

    const auto* end = static_cast<const char*>(std::memchr(data, '\0', size));
    if (!end)
        return false;

    value = {data, static_cast<size_t>(end - data)};
    consumed = value.size + 1;
    return true;
}

// Parses one map marker's saved state without consuming bytes from a following record.
inline bool parse_state(const char* data, size_t size, StateRecord& result) noexcept
{
    StateRecord parsed{};
    size_t cursor = 0;
    if (!read_string_z(data, size, parsed.hint, cursor) || cursor > size || size - cursor < sizeof(parsed.flags))
        return false;

    std::memcpy(&parsed.flags, data + cursor, sizeof(parsed.flags));
    cursor += sizeof(parsed.flags);
    size_t owner_bytes = 0;
    if (!read_string_z(data + cursor, size - cursor, parsed.owner_task_id, owner_bytes))
        return false;

    parsed.consumed = cursor + owner_bytes;
    result = parsed;
    return true;
}

// Parses and validates the fixed header plus marker state before callers create UI objects.
inline bool parse_key(const char* data, size_t size, KeyRecord& result) noexcept
{
    if (!data || size < sizeof(uint16_t))
        return false;

    KeyRecord parsed{};
    std::memcpy(&parsed.object_id, data, sizeof(parsed.object_id));
    size_t cursor = sizeof(parsed.object_id);
    size_t spot_type_bytes = 0;
    if (!read_string_z(data + cursor, size - cursor, parsed.spot_type, spot_type_bytes) ||
        parsed.spot_type.size >= 512)
        return false;
    cursor += spot_type_bytes;
    if (cursor >= size) // Reserved byte in the existing save format.
        return false;
    ++cursor;

    StateRecord state{};
    if (!parse_state(data + cursor, size - cursor, state))
        return false;

    parsed.header_size = cursor;
    parsed.consumed = cursor + state.consumed;
    result = parsed;
    return true;
}
} // namespace map_location_serialization
