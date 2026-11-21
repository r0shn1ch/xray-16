#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <cstdint>

namespace xray::render::vulkan
{
inline float temporal_halton(uint32_t index, uint32_t base)
{
    float fraction = 1.f;
    float result = 0.f;
    while (index)
    {
        fraction /= float(base);
        result += fraction * float(index % base);
        index /= base;
    }
    return result;
}

// One of eight subpixel offsets, expressed in clip-space NDC units.
inline std::array<float, 2> temporal_jitter_ndc(uint32_t sample, VkExtent2D extent)
{
    if (!extent.width || !extent.height) return {};
    const uint32_t index = sample % 8 + 1;
    return {2.f * (temporal_halton(index, 2) - .5f) / float(extent.width),
        2.f * (temporal_halton(index, 3) - .5f) / float(extent.height)};
}

// Vulkan shaders read the engine's raw Fmatrix bytes as a column-major mat4.
// Adding jitter * clip.w to clip.xy works for view-projection and model MVPs.
inline void apply_temporal_jitter(float* matrix, const std::array<float, 2>& ndc)
{
    if (!matrix) return;
    for (uint32_t column = 0; column < 4; ++column)
    {
        const float w = matrix[column * 4 + 3];
        matrix[column * 4] += ndc[0] * w;
        matrix[column * 4 + 1] += ndc[1] * w;
    }
}
}
