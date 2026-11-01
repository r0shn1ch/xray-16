#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
namespace xray::render::vulkan
{
struct RenderQuality
{
    unsigned texture_limit;
    int detail_distance, detail_updates, point_shadows, spot_shadows;
    float sun_span;
};
inline constexpr std::array<RenderQuality, 5> RenderQualities{{{512, 0, 8, 0, 1, 64},
                                                               {1024, 20, 12, 0, 1, 96},
                                                               {1024, 40, 16, 1, 1, 120},
                                                               {0, 60, 24, 1, 2, 160},
                                                               {0, 80, 32, 2, 2, 180}}};
inline const RenderQuality &render_quality(unsigned preset)
{
    return RenderQualities[std::min<unsigned>(preset, RenderQualities.size() - 1)];
}
inline float glow_facing(const std::array<float, 3> &position, const std::array<float, 3> &direction,
                         const std::array<float, 3> &eye, float radius)
{
    float distance = 0, cosine = 0, length = 0;
    for (unsigned i = 0; i < 3; ++i)
    {
        const float d = eye[i] - position[i];
        distance += d * d;
        cosine += d * direction[i];
        length += direction[i] * direction[i];
    }
    if (!std::isfinite(distance) || !std::isfinite(length) || distance <= radius * radius)
        return 0;
    if (length < 1e-8f)
        return 1;
    return std::max(cosine / std::sqrt(distance * length), 0.f);
}
} // namespace xray::render::vulkan
