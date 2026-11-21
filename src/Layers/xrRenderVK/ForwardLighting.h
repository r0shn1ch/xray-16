#pragma once

#include <cstdint>
#include "LightingLimits.h"

namespace xray::render::vulkan
{
constexpr uint32_t ForwardLightCapacity = XRAY_VK_MAX_LOCAL_LIGHTS;
struct alignas(16) ForwardLightUniform
{
    float inverse_view_projection[16]{};
    float sun_direction_ambient[4]{};
    float sun_color_count[4]{};
    float viewport[4]{};
    float fog_color_start[4]{};
    float fog_end_camera[4]{};
    float fog_plane[4]{0.f, 0.f, 0.f, 1.f};
    float ambient_color[4]{};
    float hemi_color[4]{};
    struct alignas(16) Local
    {
        float position_range[4]{};
        float direction_cone[4]{};
        float color_type[4]{};
    } local[ForwardLightCapacity]{};
};
static_assert(sizeof(ForwardLightUniform) == 64 + 128 + 48 * ForwardLightCapacity);
}
