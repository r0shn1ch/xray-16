#pragma once
#include "VsmDemandPass.h"
#include "VsmResolvePass.h"
#include "VsmStaticCache.h"

namespace xray::render::vulkan
{
struct VsmClipmap
{
    VsmDemandParameters demand{};
    VsmResolveParameters resolve{};
    VsmPageProjection projection{};
    VsmStaticCacheParameters cache{};
};

// Quantization keeps small sun-direction changes from invalidating every
// static page each frame. The output is a normalized light direction.
bool stabilize_vsm_direction(const float (&direction)[3], float (&stable)[3]);

// Builds all page-space consumers from the same light-space origin. The
// supplied view must remain stable as the camera moves; changing the view or
// depth range requires invalidating the static atlas contents.
bool build_vsm_clipmap(const float (&inverse_view_projection)[16], const float (&light_view)[16],
    const float (&camera_world)[3], uint32_t axis, uint32_t levels, float first_width,
    float near_depth, float far_depth, VsmClipmap& result, std::string& error);
}
