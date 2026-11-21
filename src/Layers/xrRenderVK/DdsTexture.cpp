#include "DdsTexture.h"
#include "Bc6Decoder.h"
#include "Bc7Decoder.h"

#include <gli/convert.hpp>
#include <gli/load_dds.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>
#include <sstream>
#include <utility>

namespace xray::render::vulkan
{
namespace
{
uint32_t read32(const std::uint8_t* bytes, size_t offset)
{
    return uint32_t(bytes[offset]) | (uint32_t(bytes[offset + 1]) << 8) |
        (uint32_t(bytes[offset + 2]) << 16) | (uint32_t(bytes[offset + 3]) << 24);
}

// Decode each compressed block once, directly into the final upload payload.
// gli::convert fetches/decompresses the same block separately for every texel
// and builds an additional complete RGBA texture before we copy it again.
void expand_bc(const uint8_t* source, uint8_t* target, size_t width, size_t height, VkFormat format)
{
    const bool bc1=format>=VK_FORMAT_BC1_RGB_UNORM_BLOCK && format<=VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    const bool rgb=format==VK_FORMAT_BC1_RGB_UNORM_BLOCK || format==VK_FORMAT_BC1_RGB_SRGB_BLOCK;
    const bool bc2=format==VK_FORMAT_BC2_UNORM_BLOCK || format==VK_FORMAT_BC2_SRGB_BLOCK;
    const bool bc4=format==VK_FORMAT_BC4_UNORM_BLOCK || format==VK_FORMAT_BC4_SNORM_BLOCK;
    const bool bc5=format==VK_FORMAT_BC5_UNORM_BLOCK || format==VK_FORMAT_BC5_SNORM_BLOCK;
    const bool signed_bc4=format==VK_FORMAT_BC4_SNORM_BLOCK;
    const bool signed_bc5=format==VK_FORMAT_BC5_SNORM_BLOCK;
    const bool bc7=format==VK_FORMAT_BC7_UNORM_BLOCK || format==VK_FORMAT_BC7_SRGB_BLOCK;
    const size_t block_size=(bc1 || bc4)?8:16, columns=(width+3)/4;
    for(size_t y=0;y<height;y+=4) for(size_t x=0;x<width;x+=4)
    {
        const uint8_t* data=source+((y/4)*columns+x/4)*block_size;
        if(bc7)
        {
            uint8_t pixels[4*4*4]{};
            decode_bc7_block(data,pixels,4*4);
            for(size_t row=0;row<std::min(size_t(4),height-y);++row)
                std::memcpy(target+((y+row)*width+x)*4,pixels+row*4*4,
                    std::min(size_t(4),width-x)*4);
            continue;
        }
        if (bc4 || bc5)
        {
            gli::detail::bc4_block block4{};
            gli::detail::bc5_block block5{};
            if (bc4)
                std::memcpy(&block4, data, sizeof(block4));
            else
                std::memcpy(&block5, data, sizeof(block5));
            gli::detail::texel_block4x4 pixels;
            if (bc4)
                pixels = signed_bc4 ? gli::detail::decompress_bc4snorm_block(block4) :
                    gli::detail::decompress_bc4unorm_block(block4);
            else if (!signed_bc5)
                pixels = gli::detail::decompress_bc5unorm_block(block5);
            else
            {
                // The pinned GLI BC5-SNORM block helper selects the green
                // channel's interpolation mode from the red endpoints. Use
                // its scalar endpoint decoders to build both channels correctly.
                float red_lut[8], green_lut[8];
                glm::uint64 red_bitmap = 0, green_bitmap = 0;
                gli::detail::single_channel_bitmap_data_snorm(block5.Red0, block5.Red1,
                    block5.RedBitmap, red_lut, red_bitmap);
                gli::detail::single_channel_bitmap_data_snorm(block5.Green0, block5.Green1,
                    block5.GreenBitmap, green_lut, green_bitmap);
                for (size_t row = 0; row < 4; ++row)
                for (size_t col = 0; col < 4; ++col)
                {
                    const unsigned bit = static_cast<unsigned>((row * 4 + col) * 3);
                    const size_t red_index = static_cast<size_t>((red_bitmap >> bit) & 7u);
                    const size_t green_index = static_cast<size_t>((green_bitmap >> bit) & 7u);
                    pixels.Texel[row][col] = glm::vec4(red_lut[red_index], green_lut[green_index], 0.f, 1.f);
                }
            }
            for (size_t row = 0; row < std::min(size_t(4), height - y); ++row)
            for (size_t col = 0; col < std::min(size_t(4), width - x); ++col)
            {
                const glm::vec4 pixel = pixels.Texel[row][col];
                uint8_t* output = target + ((y + row) * width + x + col) * 4;
                if (signed_bc4 || signed_bc5)
                {
                    int8_t channels[4]{};
                    for (size_t channel = 0; channel < 4; ++channel)
                        channels[channel] = static_cast<int8_t>(std::clamp(std::lround(pixel[static_cast<glm::length_t>(channel)] * 127.f), -127l, 127l));
                    std::memcpy(output, channels, sizeof(channels));
                }
                else
                {
                    const auto packed = gli::compScale<uint8_t>(pixel);
                    std::memcpy(output, &packed[0], 4);
                }
            }
            continue;
        }
        gli::detail::texel_block4x4 pixels;
        if(bc1) { gli::detail::dxt1_block block;std::memcpy(&block,data,sizeof(block));pixels=gli::detail::decompress_dxt1_block(block); }
        else if(bc2) { gli::detail::dxt3_block block;std::memcpy(&block,data,sizeof(block));pixels=gli::detail::decompress_dxt3_block(block); }
        else { gli::detail::dxt5_block block;std::memcpy(&block,data,sizeof(block));pixels=gli::detail::decompress_dxt5_block(block); }
        for(size_t row=0;row<std::min(size_t(4),height-y);++row)
            for(size_t col=0;col<std::min(size_t(4),width-x);++col)
            {
                // sRGB DDS already contains encoded bytes. Preserve them and
                // let the Vulkan sRGB image perform the only decode at sampling.
                auto packed=gli::compScale<uint8_t>(pixels.Texel[row][col]);
                if(rgb) packed.a=255;
                std::memcpy(target+((y+row)*width+x+col)*4,&packed[0],4);
            }
    }
}

void expand_bc6h(const uint8_t* source, uint8_t* target, size_t width, size_t height, bool is_signed)
{
    const size_t columns = (width + 3) / 4;
    constexpr uint16_t half_one = 0x3c00;
    for (size_t y = 0; y < height; y += 4)
    for (size_t x = 0; x < width; x += 4)
    {
        const uint8_t* block = source + ((y / 4) * columns + x / 4) * 16;
        uint16_t rgb[4 * 4 * 3]{};
        decode_bc6h_block(block, rgb, 4 * 3, is_signed);
        for (size_t row = 0; row < std::min(size_t(4), height - y); ++row)
        for (size_t col = 0; col < std::min(size_t(4), width - x); ++col)
        {
            const size_t source_pixel = (row * 4 + col) * 3;
            const uint16_t rgba[4]{rgb[source_pixel], rgb[source_pixel + 1],
                rgb[source_pixel + 2], half_one};
            std::memcpy(target + ((y + row) * width + x + col) * sizeof(rgba), rgba, sizeof(rgba));
        }
    }
}

VkFormat vk_format(gli::format format)
{
    switch (format)
    {
    case gli::FORMAT_RGBA8_UNORM_PACK8: return VK_FORMAT_R8G8B8A8_UNORM;
    case gli::FORMAT_BGRA8_UNORM_PACK8: return VK_FORMAT_B8G8R8A8_UNORM;
    case gli::FORMAT_RGBA8_SRGB_PACK8: return VK_FORMAT_R8G8B8A8_SRGB;
    case gli::FORMAT_BGRA8_SRGB_PACK8: return VK_FORMAT_B8G8R8A8_SRGB;
    case gli::FORMAT_R16_SFLOAT_PACK16: return VK_FORMAT_R16_SFLOAT;
    case gli::FORMAT_RG16_SFLOAT_PACK16: return VK_FORMAT_R16G16_SFLOAT;
    case gli::FORMAT_RGBA16_SFLOAT_PACK16: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case gli::FORMAT_R32_SFLOAT_PACK32: return VK_FORMAT_R32_SFLOAT;
    case gli::FORMAT_RG32_SFLOAT_PACK32: return VK_FORMAT_R32G32_SFLOAT;
    case gli::FORMAT_RGBA32_SFLOAT_PACK32: return VK_FORMAT_R32G32B32A32_SFLOAT;
    case gli::FORMAT_RGB_BP_UFLOAT_BLOCK16: return VK_FORMAT_BC6H_UFLOAT_BLOCK;
    case gli::FORMAT_RGB_BP_SFLOAT_BLOCK16: return VK_FORMAT_BC6H_SFLOAT_BLOCK;
    case gli::FORMAT_RGB_DXT1_UNORM_BLOCK8: return VK_FORMAT_BC1_RGB_UNORM_BLOCK;
    case gli::FORMAT_RGB_DXT1_SRGB_BLOCK8: return VK_FORMAT_BC1_RGB_SRGB_BLOCK;
    case gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8: return VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case gli::FORMAT_RGBA_DXT1_SRGB_BLOCK8: return VK_FORMAT_BC1_RGBA_SRGB_BLOCK;
    case gli::FORMAT_RGBA_DXT3_UNORM_BLOCK16: return VK_FORMAT_BC2_UNORM_BLOCK;
    case gli::FORMAT_RGBA_DXT3_SRGB_BLOCK16: return VK_FORMAT_BC2_SRGB_BLOCK;
    case gli::FORMAT_RGBA_DXT5_UNORM_BLOCK16: return VK_FORMAT_BC3_UNORM_BLOCK;
    case gli::FORMAT_RGBA_DXT5_SRGB_BLOCK16: return VK_FORMAT_BC3_SRGB_BLOCK;
    case gli::FORMAT_R_ATI1N_UNORM_BLOCK8: return VK_FORMAT_BC4_UNORM_BLOCK;
    case gli::FORMAT_R_ATI1N_SNORM_BLOCK8: return VK_FORMAT_BC4_SNORM_BLOCK;
    case gli::FORMAT_RG_ATI2N_UNORM_BLOCK16: return VK_FORMAT_BC5_UNORM_BLOCK;
    case gli::FORMAT_RG_ATI2N_SNORM_BLOCK16: return VK_FORMAT_BC5_SNORM_BLOCK;
    case gli::FORMAT_RGBA_BP_UNORM_BLOCK16: return VK_FORMAT_BC7_UNORM_BLOCK;
    case gli::FORMAT_RGBA_BP_SRGB_BLOCK16: return VK_FORMAT_BC7_SRGB_BLOCK;
    default: return VK_FORMAT_UNDEFINED;
    }
}

bool compressed(VkFormat format)
{
    return (format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && format <= VK_FORMAT_BC3_SRGB_BLOCK) ||
        (format >= VK_FORMAT_BC4_UNORM_BLOCK && format <= VK_FORMAT_BC5_SNORM_BLOCK) ||
        format == VK_FORMAT_BC6H_UFLOAT_BLOCK || format == VK_FORMAT_BC6H_SFLOAT_BLOCK ||
        format == VK_FORMAT_BC7_UNORM_BLOCK || format == VK_FORMAT_BC7_SRGB_BLOCK;
}

constexpr uint32_t make_fourcc(char a, char b, char c, char d)
{
    return uint32_t(static_cast<uint8_t>(a)) | (uint32_t(static_cast<uint8_t>(b)) << 8) |
        (uint32_t(static_cast<uint8_t>(c)) << 16) | (uint32_t(static_cast<uint8_t>(d)) << 24);
}

uint8_t channel(uint32_t pixel, uint32_t mask)
{
    if (!mask) return 0;
    unsigned shift = 0;
    while (!(mask & 1u)) { mask >>= 1; ++shift; }
    const uint64_t value = (pixel >> shift) & mask;
    return static_cast<uint8_t>((value * 255u + mask / 2u) / mask);
}

uint32_t numeric_float_fourcc_bytes(uint32_t fourcc)
{
    // Legacy D3DFORMAT values stored in dwFourCC by several DDS writers.
    switch (fourcc)
    {
    case 111: return 2;  // D3DFMT_R16F
    case 112: return 4;  // D3DFMT_G16R16F
    case 113: return 8;  // D3DFMT_A16B16G16R16F
    case 114: return 4;  // D3DFMT_R32F
    case 115: return 8;  // D3DFMT_G32R32F
    case 116: return 16; // D3DFMT_A32B32G32R32F
    default: return 0;
    }
}

uint32_t dxgi_float_bytes(uint32_t format)
{
    // DXGI_FORMAT_R32G32B32A32_FLOAT, R16G16B16A16_FLOAT,
    // R32G32_FLOAT, R16G16_FLOAT, R32_FLOAT and R16_FLOAT.
    switch (format)
    {
    case 2: return 16;
    case 10: return 8;
    case 16: return 8;
    case 34: return 4;
    case 41: return 4;
    case 54: return 2;
    default: return 0;
    }
}

bool contiguous(uint32_t mask)
{
    if (!mask) return true;
    while (!(mask & 1u)) mask >>= 1;
    return (mask & (mask + 1u)) == 0;
}

bool decode_legacy_pixels(const uint8_t* bytes, size_t size, uint32_t width,
    uint32_t height, uint32_t depth, uint32_t mip_count, uint32_t faces, bool volume, DdsTexture& result,
    std::string& error, uint32_t maximum_dimension, uint32_t mip_skip)
{
    const uint32_t flags = read32(bytes, 80);
    const uint32_t bits = read32(bytes, 88);
    const bool rgb = (flags & 0x40u) != 0;
    const bool luminance = (flags & 0x20000u) != 0;
    const bool alpha_only = (flags & 0x2u) != 0 && !rgb && !luminance;
    uint32_t red = read32(bytes, 92), green = read32(bytes, 96);
    uint32_t blue = read32(bytes, 100), alpha = read32(bytes, 104);
    if (bits != 8 && bits != 16 && bits != 24 && bits != 32)
    {
        error = "unsupported DDS pixel bit count: " + std::to_string(bits);
        return false;
    }
    const uint32_t valid_bits = bits == 32 ? ~0u : (1u << bits) - 1u;
    if ((luminance || alpha_only) && !red && !alpha)
        (alpha_only ? alpha : red) = valid_bits;
    const bool valid_masks = rgb ? red && green && blue : (luminance ? red != 0 : alpha_only && alpha != 0);
    if (!valid_masks || ((red | green | blue | alpha) & ~valid_bits) ||
        (red & green) || (red & blue) || (red & alpha) ||
        (green & blue) || (green & alpha) || (blue & alpha) ||
        !contiguous(red) || !contiguous(green) || !contiguous(blue) || !contiguous(alpha))
    {
        std::ostringstream detail;
        detail << "unsupported DDS pixel masks flags=" << flags << " bits=" << bits <<
            " masks=" << red << "/" << green << "/" << blue << "/" << alpha;
        error = detail.str();
        return false;
    }

    // Writers disagree about whether uncompressed mip rows are byte- or
    // DWORD-aligned. Select the layout matching the actual payload length.
    unsigned alignment = 0;
    for (const unsigned candidate : {1u, 4u})
    {
        size_t expected = 128;
        bool valid_layout = true;
        for (uint32_t level = 0; level < mip_count; ++level)
        {
            const size_t row = (size_t(std::max(1u, width >> level)) * bits + 7u) / 8u;
            const size_t stride = ((row + candidate - 1u) / candidate) * candidate;
            const size_t h = std::max(1u, height >> level);
            const size_t d = volume ? std::max(1u, depth >> level) : 1u;
            constexpr size_t max_payload = 256u * 1024u * 1024u;
            if (stride > max_payload / h || stride * h > max_payload / d ||
                stride * h * d > max_payload / faces)
            {
                valid_layout = false;
                break;
            }
            const size_t mip_bytes = stride * h * d * faces;
            if (mip_bytes > max_payload - expected)
            {
                valid_layout = false;
                break;
            }
            expected += mip_bytes;
        }
        if (valid_layout && expected == size)
        { alignment = candidate; break; }
    }
    if (!alignment)
    {
        error = "DDS mip payload has an invalid size";
        return false;
    }

    uint32_t first_mip = faces == 1 ? std::min(mip_skip, mip_count - 1) : 0;
    while(maximum_dimension && faces==1 && first_mip+1<mip_count &&
        std::max(std::max(width>>first_mip,height>>first_mip), volume ? depth>>first_mip : 1u) > maximum_dimension)
        ++first_mip;
    DdsTexture decoded;
    decoded.format = VK_FORMAT_R8G8B8A8_UNORM;
    decoded.extent = {std::max(1u,width>>first_mip),std::max(1u,height>>first_mip),
        volume ? std::max(1u, depth >> first_mip) : 1u};
    decoded.mip_levels = mip_count-first_mip;
    decoded.layers = faces;
    decoded.cube = faces == 6;
    decoded.volume = volume;
    size_t source = 128;
    for (uint32_t face = 0; face < faces; ++face)
    for (uint32_t level = 0; level < mip_count; ++level)
    {
        const uint32_t w = std::max(1u, width >> level), h = std::max(1u, height >> level);
        const uint32_t d = volume ? std::max(1u, depth >> level) : 1u;
        const size_t row = (size_t(w) * bits + 7u) / 8u;
        const size_t stride = (row + alignment - 1u) / alignment * alignment;
        if(level<first_mip) { source += stride*h*d; continue; }
        VkBufferImageCopy copy{};
        copy.bufferOffset = decoded.pixels.size();
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level-first_mip, face, 1};
        copy.imageExtent = {w, h, d};
        decoded.copies.push_back(copy);
        decoded.pixels.reserve(decoded.pixels.size() + size_t(w) * h * d * 4u);
        for (uint32_t z = 0; z < d; ++z)
        for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x)
        {
            const size_t index = source + (size_t(z) * h + y) * stride + size_t(x) * (bits / 8u);
            uint32_t pixel = 0;
            for (uint32_t i = 0; i < bits / 8u; ++i)
                pixel |= uint32_t(bytes[index + i]) << (8u * i);
            const uint8_t color = channel(pixel, red);
            decoded.pixels.push_back(alpha_only ? 255 : color);
            decoded.pixels.push_back(luminance ? color : alpha_only ? 255 : channel(pixel, green));
            decoded.pixels.push_back(luminance ? color : alpha_only ? 255 : channel(pixel, blue));
            decoded.pixels.push_back(alpha ? channel(pixel, alpha) : 255);
        }
        source += stride * h * d;
    }
    result = std::move(decoded);
    error.clear();
    return true;
}
}

bool decode_dds(const void* data, size_t size, bool bc_supported, DdsTexture& result, std::string& error, uint32_t maximum_dimension, uint32_t mip_skip) try
{
    result = {};
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    if (!bytes || size < 128 || std::memcmp(bytes, "DDS ", 4) != 0 || read32(bytes, 4) != 124 ||
        read32(bytes, 76) != 32)
    {
        error = "invalid DDS header";
        return false;
    }

    const uint32_t width = read32(bytes, 16), height = read32(bytes, 12);
    const uint32_t header_flags = read32(bytes, 8);
    const uint32_t mip_count = (header_flags & 0x20000u) ? read32(bytes, 28) : 1;
    if (!width || !height || width > 16384 || height > 16384 || !mip_count || mip_count > 15)
    {
        error = "unsupported DDS dimensions or target";
        return false;
    }

    const uint32_t fourcc = (read32(bytes, 80) & 0x4u) ? read32(bytes, 84) : 0;
    const bool extended = fourcc == 0x30315844u; // DX10
    const size_t header_size = extended ? 148 : 128;
    if (size < header_size)
    {
        error = "unsupported DDS extension";
        return false;
    }
    const uint32_t resource_dimension = extended ? read32(bytes, 132) : 3;
    const uint32_t array_size = extended ? read32(bytes, 140) : 1;
    if (extended && ((resource_dimension != 3 && resource_dimension != 4) || !array_size || array_size > 2048))
    {
        error = "unsupported DDS extension";
        return false;
    }

    const uint32_t caps2 = read32(bytes, 112);
    const uint32_t cube_caps = caps2 & 0xfe00u;
    const bool cube = extended ? (read32(bytes, 136) & 4u) != 0 : (cube_caps & 0x200u) != 0;
    const bool volume = extended ? resource_dimension == 4 : (caps2 & 0x200000u) != 0;
    const bool has_depth_flag = (header_flags & 0x800000u) != 0;
    const uint32_t depth = volume ? read32(bytes, 24) : 1u;
    if ((cube && width != height) || (cube && volume) ||
        (!extended && cube_caps && cube_caps != 0xfe00u) ||
        has_depth_flag != volume || (volume && (array_size != 1 || !depth || depth > 16384 || cube_caps)))
    {
        error = cube && width != height ? "DDS cubemap requires six square faces" :
            "DDS volume target has inconsistent dimensions, depth, or array metadata";
        return false;
    }
    const uint32_t faces = cube ? 6 : 1;
    const uint32_t layers = array_size * faces;

    // GLI's pinned loader asserts on truncated payloads. Validate its expected layout first.
    const uint32_t dxgi = extended ? read32(bytes, 128) : 0;
    uint32_t block_bytes = 0;
    const bool legacy_bc4 = fourcc == make_fourcc('A', 'T', 'I', '1') || fourcc == make_fourcc('A', 'T', '1', 'N') ||
        fourcc == make_fourcc('B', 'C', '4', 'U') || fourcc == make_fourcc('B', 'C', '4', 'S');
    const bool legacy_bc5 = fourcc == make_fourcc('A', 'T', 'I', '2') || fourcc == make_fourcc('A', 'T', '2', 'N') ||
        fourcc == make_fourcc('B', 'C', '5', 'U') || fourcc == make_fourcc('B', 'C', '5', 'S');
    if (fourcc == 0x31545844u || dxgi == 71 || dxgi == 72) block_bytes = 8; // BC1
    else if (legacy_bc4 || dxgi == 80 || dxgi == 81) block_bytes = 8; // BC4
    else if (legacy_bc5 || dxgi == 83 || dxgi == 84) block_bytes = 16; // BC5
    else if (fourcc == 0x33545844u || fourcc == 0x35545844u ||
        dxgi == 74 || dxgi == 75 || dxgi == 77 || dxgi == 78 ||
        dxgi == 95 || dxgi == 96 || dxgi == 98 || dxgi == 99) block_bytes = 16; // BC2/BC3/BC6H/BC7
    const uint32_t float_pixel_bytes = extended ? dxgi_float_bytes(dxgi) : numeric_float_fourcc_bytes(fourcc);
    const bool known_rgba8_dxgi = extended &&
        (dxgi == 28 || dxgi == 29 || dxgi == 87 || dxgi == 91);
    if (fourcc != 0 && !block_bytes && !known_rgba8_dxgi && !float_pixel_bytes)
    {
        error = "unsupported DDS FourCC";
        return false;
    }
    const uint32_t bits = read32(bytes, 88);
    if (!block_bytes && !extended && !float_pixel_bytes && (bits != 32 ||
        !(read32(bytes, 92) == 0x000000ffu && read32(bytes, 96) == 0x0000ff00u &&
          read32(bytes, 100) == 0x00ff0000u && read32(bytes, 104) == 0xff000000u) &&
        !(read32(bytes, 92) == 0x00ff0000u && read32(bytes, 96) == 0x0000ff00u &&
          read32(bytes, 100) == 0x000000ffu && read32(bytes, 104) == 0xff000000u)))
    {
        return decode_legacy_pixels(bytes, size, width, height, depth, mip_count, faces, volume,
            result, error, maximum_dimension, mip_skip);
    }
    const uint32_t pixel_bytes = float_pixel_bytes ? float_pixel_bytes : (known_rgba8_dxgi ? 4u : bits / 8u);

    constexpr size_t max_payload_bytes = 256u * 1024u * 1024u;
    size_t bytes_per_layer = 0;
    for (uint32_t level = 0; level < mip_count; ++level)
    {
        const size_t w = std::max(1u, width >> level), h = std::max(1u, height >> level);
        const size_t d = volume ? std::max(1u, depth >> level) : 1u;
        size_t mip_bytes = 0;
        if (block_bytes)
        {
            const size_t block_width = (w + 3) / 4, block_height = (h + 3) / 4;
            if (block_width > max_payload_bytes / block_height ||
                block_width * block_height > max_payload_bytes / d ||
                block_width * block_height * d > max_payload_bytes / block_bytes)
            {
                error = "DDS mip payload is too large";
                return false;
            }
            mip_bytes = block_width * block_height * d * block_bytes;
        }
        else
        {
            if (w > max_payload_bytes / h || w * h > max_payload_bytes / d ||
                w * h * d > max_payload_bytes / pixel_bytes)
            {
                error = "DDS mip payload is too large";
                return false;
            }
            mip_bytes = w * h * d * pixel_bytes;
        }
        if (mip_bytes > max_payload_bytes - bytes_per_layer)
        {
            error = "DDS mip payload is too large";
            return false;
        }
        bytes_per_layer += mip_bytes;
    }
    if (bytes_per_layer > max_payload_bytes / layers ||
        header_size + bytes_per_layer * layers != size)
    {
        error = "DDS mip payload has an invalid size";
        return false;
    }

    const auto* characters = reinterpret_cast<const char*>(bytes);
    gli::texture loaded = gli::load_dds(characters, size);
    const gli::target expected_target = volume ? gli::TARGET_3D :
        cube ? (array_size > 1 ? gli::TARGET_CUBE_ARRAY : gli::TARGET_CUBE) :
        array_size > 1 ? gli::TARGET_2D_ARRAY : gli::TARGET_2D;
    if (loaded.empty() || loaded.target() != expected_target ||
        loaded.layers() != (volume ? 1u : array_size) || loaded.faces() != faces ||
        loaded.levels() != mip_count || static_cast<uint32_t>(loaded.extent().x) != width ||
        static_cast<uint32_t>(loaded.extent().y) != height ||
        static_cast<uint32_t>(loaded.extent().z) != depth)
    {
        error = "DDS loader rejected texture";
        return false;
    }
    VkFormat format = vk_format(loaded.format());
    if (format == VK_FORMAT_UNDEFINED)
    {
        error = "DDS texture format has no Vulkan mapping";
        return false;
    }

    uint32_t first_mip = cube ? 0 : std::min(mip_skip, mip_count - 1);
    while (maximum_dimension && !cube && first_mip + 1 < mip_count &&
        std::max(std::max(width >> first_mip, height >> first_mip), volume ? depth >> first_mip : 1u) > maximum_dimension)
        ++first_mip;
    if (first_mip)
        loaded = gli::texture(loaded, loaded.target(), loaded.format(), 0, loaded.layers() - 1,
            0, loaded.faces() - 1,
            first_mip, mip_count - 1);
    const uint32_t output_mips = mip_count - first_mip;
    const uint32_t output_width = std::max(1u, width >> first_mip);
    const uint32_t output_height = std::max(1u, height >> first_mip);
    const uint32_t output_depth = volume ? std::max(1u, depth >> first_mip) : 1u;

    const VkFormat source_format=format;
    const bool expand=!bc_supported && compressed(format);
    const bool bc6h = source_format == VK_FORMAT_BC6H_UFLOAT_BLOCK ||
        source_format == VK_FORMAT_BC6H_SFLOAT_BLOCK;
    const size_t expanded_pixel_bytes = bc6h ? 8u : 4u;
    if (expand)
    {
        const bool signed_bc = source_format == VK_FORMAT_BC4_SNORM_BLOCK ||
            source_format == VK_FORMAT_BC5_SNORM_BLOCK;
        const bool srgb = gli::is_srgb(loaded.format());
        format = bc6h ? VK_FORMAT_R16G16B16A16_SFLOAT :
            signed_bc ? VK_FORMAT_R8G8B8A8_SNORM :
            srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    }

    DdsTexture decoded;
    decoded.format = format;
    decoded.extent = { output_width, output_height, output_depth };
    decoded.mip_levels = output_mips;
    decoded.layers = layers;
    decoded.cube = cube;
    decoded.volume = volume;
    // Copy each mip into one allocation. Repeated vector growth temporarily
    // keeps both the old and new buffers resident on 32-bit Android.
    const size_t alignment_bytes = size_t(layers) * output_mips * 3u;
    size_t payload_bytes=loaded.size();
    if(expand)
    {
        payload_bytes=0;
        for(uint32_t level=0;level<output_mips;++level)
        {
            const size_t layer_bytes=size_t(std::max(1u,output_width>>level))*
                std::max(1u,output_height>>level)*expanded_pixel_bytes;
            const size_t mip_depth = volume ? std::max(1u, output_depth >> level) : 1u;
            if (layer_bytes > decoded.pixels.max_size() / mip_depth)
            { error="DDS decoded payload is too large";return false; }
            const size_t volume_bytes = layer_bytes * mip_depth;
            if(volume_bytes>(decoded.pixels.max_size()-payload_bytes-alignment_bytes)/layers)
            { error="DDS decoded payload is too large";return false; }
            payload_bytes+=volume_bytes*layers;
        }
    }
    if (payload_bytes > decoded.pixels.max_size() - alignment_bytes)
    {
        error = "DDS decoded payload is too large";
        return false;
    }
    decoded.pixels.reserve(payload_bytes + alignment_bytes);
    decoded.copies.reserve(size_t(layers) * output_mips);
    for (uint32_t layer = 0; layer < array_size; ++layer)
    for (uint32_t face = 0; face < faces; ++face)
    for (uint32_t level = 0; level < output_mips; ++level)
    {
        const size_t offset = (decoded.pixels.size() + 3u) & ~size_t(3u);
        decoded.pixels.resize(offset, 0);
        VkBufferImageCopy copy{};
        copy.bufferOffset = offset;
        copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, level, layer * faces + face, 1 };
        copy.imageExtent = { std::max(1u, output_width >> level), std::max(1u, output_height >> level),
            volume ? std::max(1u, output_depth >> level) : 1u };
        decoded.copies.push_back(copy);
        const size_t bytes = loaded.size(level);
        const size_t w = copy.imageExtent.width, h = copy.imageExtent.height;
        const size_t d = copy.imageExtent.depth;
        const size_t expected_mip = compressed(source_format) ?
            ((w + 3) / 4) * ((h + 3) / 4) * d * block_bytes : w * h * d * pixel_bytes;
        const auto extent = loaded.extent(level);
        if (!bytes || bytes != expected_mip || static_cast<size_t>(extent.x) != w ||
            static_cast<size_t>(extent.y) != h || static_cast<size_t>(extent.z) != d)
        {
            error = "DDS array-layer/mip layout is inconsistent: layer=" +
                std::to_string(layer * faces + face) +
                " level=" + std::to_string(level);
            return false;
        }
        const auto* src = static_cast<const std::uint8_t*>(loaded.data(layer, face, level));
        if (!src)
        {
            error = "DDS array-layer/mip data is missing: layer=" +
                std::to_string(layer * faces + face) +
                " level=" + std::to_string(level);
            return false;
        }
        if(expand)
        {
            const size_t slice_size = compressed(source_format) ? ((w + 3) / 4) * ((h + 3) / 4) * block_bytes :
                w * h * pixel_bytes;
            decoded.pixels.resize(offset + w * h * d * expanded_pixel_bytes);
            for (size_t slice = 0; slice < d; ++slice)
            {
                uint8_t* destination = decoded.pixels.data() + offset +
                    slice * w * h * expanded_pixel_bytes;
                if (bc6h)
                    expand_bc6h(src + slice * slice_size, destination, w, h,
                        source_format == VK_FORMAT_BC6H_SFLOAT_BLOCK);
                else
                    expand_bc(src + slice * slice_size, destination, w, h, source_format);
            }
        }
        else decoded.pixels.insert(decoded.pixels.end(), src, src + bytes);
    }
    result = std::move(decoded);
    error.clear();
    return true;
}
catch (const std::bad_alloc&)
{
    result = {};
    error = "DDS OOM";
    return false;
}
}
