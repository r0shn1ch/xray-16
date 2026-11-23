#include "xrGame/map_location_serialization.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
void append(std::vector<char>& bytes, const void* data, size_t size)
{
    const auto* source = static_cast<const char*>(data);
    bytes.insert(bytes.end(), source, source + size);
}

void append_string(std::vector<char>& bytes, const char* value)
{
    append(bytes, value, std::strlen(value) + 1);
}

std::vector<char> make_record()
{
    std::vector<char> bytes;
    const uint16_t object_id = 0x1234;
    const uint32_t flags = 0x89abcdef;
    const uint8_t reserved = 0;
    append(bytes, &object_id, sizeof(object_id));
    append_string(bytes, "test_spot");
    append(bytes, &reserved, sizeof(reserved));
    append_string(bytes, "test_hint");
    append(bytes, &flags, sizeof(flags));
    append_string(bytes, "task_42");
    return bytes;
}

void test_valid_record_preserves_next_record()
{
    std::vector<char> bytes = make_record();
    const std::array<char, 4> sentinel{'n', 'e', 'x', 't'};
    append(bytes, sentinel.data(), sentinel.size());

    map_location_serialization::KeyRecord key{};
    assert(map_location_serialization::parse_key(bytes.data(), bytes.size(), key));
    assert(key.object_id == 0x1234);
    assert(key.spot_type.size == std::strlen("test_spot"));
    assert(std::memcmp(key.spot_type.data, "test_spot", key.spot_type.size) == 0);
    assert(key.consumed == bytes.size() - sentinel.size());

    map_location_serialization::StateRecord state{};
    assert(map_location_serialization::parse_state(bytes.data() + key.header_size,
        bytes.size() - key.header_size, state));
    assert(state.flags == 0x89abcdef);
    assert(state.hint.size == std::strlen("test_hint"));
    assert(state.owner_task_id.size == std::strlen("task_42"));
    assert(state.consumed + key.header_size == key.consumed);
}

void test_truncated_key_and_missing_terminators_are_rejected()
{
    const std::array<char, 3> no_spot_terminator{char(1), char(0), 'x'};
    map_location_serialization::KeyRecord key{};
    assert(!map_location_serialization::parse_key(no_spot_terminator.data(), no_spot_terminator.size(), key));

    std::vector<char> bytes;
    const uint16_t object_id = 1;
    const uint8_t reserved = 0;
    append(bytes, &object_id, sizeof(object_id));
    append_string(bytes, "spot");
    append(bytes, &reserved, sizeof(reserved));
    append(bytes, "hint", 4);
    assert(!map_location_serialization::parse_key(bytes.data(), bytes.size(), key));
}

void test_oversized_spot_and_truncated_state_are_rejected()
{
    std::vector<char> oversized;
    const uint16_t object_id = 1;
    const uint8_t reserved = 0;
    append(oversized, &object_id, sizeof(object_id));
    oversized.insert(oversized.end(), 512, 's');
    oversized.push_back('\0');
    append(oversized, &reserved, sizeof(reserved));
    map_location_serialization::KeyRecord key{};
    assert(!map_location_serialization::parse_key(oversized.data(), oversized.size(), key));

    const std::array<char, 4> truncated_flags{'h', '\0', char(0x12), char(0x34)};
    map_location_serialization::StateRecord state{};
    assert(!map_location_serialization::parse_state(truncated_flags.data(), truncated_flags.size(), state));
}

void test_missing_owner_terminator_is_rejected()
{
    std::vector<char> bytes;
    append_string(bytes, "hint");
    const uint32_t flags = 7;
    append(bytes, &flags, sizeof(flags));
    append(bytes, "owner", 5);

    map_location_serialization::StateRecord state{};
    assert(!map_location_serialization::parse_state(bytes.data(), bytes.size(), state));
}
} // namespace

int main()
{
    test_valid_record_preserves_next_record();
    test_truncated_key_and_missing_terminators_are_rejected();
    test_oversized_spot_and_truncated_state_are_rejected();
    test_missing_owner_terminator_is_rejected();
    return 0;
}
