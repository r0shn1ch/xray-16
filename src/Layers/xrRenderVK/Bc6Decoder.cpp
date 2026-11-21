#include "Bc6Decoder.h"

// BC6H decoding is provided by bcdec 0.985, commit
// 80859ed3b7afb1c527a2a99d70c61457bea72d0c, by Sergii Kudlai:
// https://github.com/iOrange/bcdec
// Used under the MIT license; the full license text is retained in bcdec.h.
#define BCDEC_STATIC
#define BCDEC_IMPLEMENTATION
#include <bcdec.h>

namespace xray::render::vulkan
{
void decode_bc6h_block(const uint8_t* source, uint16_t* destination,
    int destination_pitch, bool is_signed)
{
    bcdec_bc6h_half(source, destination, destination_pitch, is_signed ? 1 : 0);
}
}
