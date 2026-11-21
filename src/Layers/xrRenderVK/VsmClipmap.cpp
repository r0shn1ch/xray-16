#include "VsmClipmap.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace xray::render::vulkan
{
bool stabilize_vsm_direction(const float (&direction)[3], float (&stable)[3])
{
    double length = 0.0;
    for (float value : direction)
    {
        if (!std::isfinite(value)) return false;
        length += double(value) * value;
    }
    if (!std::isfinite(length) || length < 1e-12) return false;
    length = std::sqrt(length);
    double quantized[3]{}, quantized_length = 0.0;
    for (unsigned i = 0; i < 3; ++i)
    {
        quantized[i] = std::round(double(direction[i]) / length * 1024.0) / 1024.0;
        quantized_length += quantized[i] * quantized[i];
    }
    if (quantized_length < 1e-12) return false;
    quantized_length = std::sqrt(quantized_length);
    for (unsigned i = 0; i < 3; ++i) stable[i] = float(quantized[i] / quantized_length);
    return true;
}

bool build_vsm_clipmap(const float (&inverse_view_projection)[16], const float (&light_view)[16],
    const float (&camera_world)[3], uint32_t axis, uint32_t levels, float first_width,
    float near_depth, float far_depth, VsmClipmap& result, std::string& error)
{
    if (!axis || axis > 32768 || (axis & (axis - 1)) || !levels || levels > 6 ||
        uint64_t(axis) * axis * levels > UINT32_MAX ||
        !std::isfinite(first_width) || first_width <= 0.f || !std::isfinite(near_depth) ||
        !std::isfinite(far_depth) || far_depth <= near_depth || !std::isfinite(far_depth - near_depth) ||
        !std::all_of(inverse_view_projection, inverse_view_projection + 16, [](float x) { return std::isfinite(x); }) ||
        !std::all_of(light_view, light_view + 16, [](float x) { return std::isfinite(x); }) ||
        !std::all_of(camera_world, camera_world + 3, [](float x) { return std::isfinite(x); }))
    { error = "invalid VSM clipmap inputs"; return false; }
    // A perspective light transform would make a fixed world-space page grid invalid.
    if (std::abs(light_view[3]) > 1e-6f || std::abs(light_view[7]) > 1e-6f ||
        std::abs(light_view[11]) > 1e-6f || std::abs(light_view[15] - 1.f) > 1e-6f)
    { error = "VSM clipmaps require an affine light view"; return false; }
    VsmClipmap next{};
    next.demand.axis = next.resolve.axis = next.projection.axis = next.cache.axis = axis;
    next.demand.levels = next.resolve.levels = next.projection.levels = next.cache.levels = levels;
    next.demand.step = 1;
    std::copy_n(inverse_view_projection, 16, next.demand.inverse_view_projection);
    std::copy_n(inverse_view_projection, 16, next.resolve.inverse_view_projection);
    std::copy_n(light_view, 16, next.demand.light_view);
    std::copy_n(light_view, 16, next.resolve.light_view);
    std::copy_n(light_view, 16, next.projection.light_view);
    next.projection.near_depth = next.resolve.near_depth = near_depth;
    next.projection.far_depth = far_depth;
    next.resolve.inverse_depth_range = 1.f / (far_depth - near_depth);
    const double x = double(light_view[0]) * camera_world[0] + double(light_view[4]) * camera_world[1] +
        double(light_view[8]) * camera_world[2] + light_view[12];
    const double y = double(light_view[1]) * camera_world[0] + double(light_view[5]) * camera_world[1] +
        double(light_view[9]) * camera_world[2] + light_view[13];
    for (uint32_t level = 0; level < levels; ++level)
    {
        const double width = std::ldexp(double(first_width), int(level));
        const double step = width / axis;
        const double base_x = std::floor(x / step) - axis / 2;
        const double base_y = std::floor(y / step) - axis / 2;
        const double maximum = double(std::numeric_limits<int32_t>::max()) - axis + 1;
        if (!std::isfinite(width) || width > std::numeric_limits<float>::max() || float(step) <= 0.f ||
            !std::isfinite(base_x) || !std::isfinite(base_y) ||
            base_x < std::numeric_limits<int32_t>::min() || base_y < std::numeric_limits<int32_t>::min() ||
            base_x > maximum || base_y > maximum ||
            std::abs(base_x * step) > std::numeric_limits<float>::max() ||
            std::abs(base_y * step) > std::numeric_limits<float>::max())
        { error = "VSM clipmap coordinates overflow"; return false; }
        next.cache.base[level][0] = static_cast<int32_t>(base_x);
        next.cache.base[level][1] = static_cast<int32_t>(base_y);
        for (float* values : {next.demand.level[level], next.resolve.level[level], next.projection.level[level]})
        {
            values[0] = float(base_x * step); values[1] = float(base_y * step); values[2] = float(width);
        }
    }
    result = next;
    error.clear(); return true;
}
}
