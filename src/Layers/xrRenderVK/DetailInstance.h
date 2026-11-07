#pragma once
#include <array>
#include <cmath>
namespace xray::render::vulkan
{
// Separate instance data keeps a prototype's vertex/index buffers immutable.
struct DetailInstance
{
    float position_scale[4];
    float rotation[2];
};
static_assert(sizeof(DetailInstance) == 24);
inline DetailInstance detail_instance(float x, float y, float z, float yaw, float scale)
{
    return {{x, y, z, scale}, {std::cos(yaw), std::sin(yaw)}};
}
inline std::array<float, 3> detail_transform(const DetailInstance &p, const float *v)
{
    return {p.position_scale[0] + p.position_scale[3] * (p.rotation[0] * v[0] + p.rotation[1] * v[2]),
            p.position_scale[1] + p.position_scale[3] * v[1],
            p.position_scale[2] + p.position_scale[3] * (-p.rotation[1] * v[0] + p.rotation[0] * v[2])};
}
} // namespace xray::render::vulkan
