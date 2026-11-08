#pragma once

#include <cstddef>
#include <cstdint>

namespace xray::render::vulkan
{

// Expands one 16-byte BC7 block into a 4x4 RGBA8 block. The destination
// pitch is expressed in bytes and must accommodate four RGBA pixels.
void decode_bc7_block(const uint8_t *source, uint8_t *destination, size_t destination_pitch);

}
