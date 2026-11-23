#pragma once

#include <cstdint>

namespace xray::render::vulkan
{
// Decodes one BC6H block into sixteen RGB half-float texels. destination_pitch
// is the number of uint16_t elements between rows and must be at least 12.
void decode_bc6h_block(const uint8_t* source, uint16_t* destination,
    int destination_pitch, bool is_signed);
}
