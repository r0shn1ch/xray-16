#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <vector>
class IReader;
namespace xray::render::vulkan
{
inline constexpr std::array<uint32_t, 14> particle_effect_chunk_ids{{
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0x21, 0x22, 0x25}};
inline constexpr std::array<uint32_t, 5> particle_group_chunk_ids{{1, 2, 3, 4, 5}};
std::vector<uint8_t> particle_library_bytes(IReader& source);
inline constexpr size_t editor_library_limit = 64 * 1024 * 1024;
bool editable_particle_library_bytes(IReader& source, std::vector<uint8_t>& bytes, std::string& error);
}
