#pragma once
#include <vulkan/vulkan.h>
namespace xray::render::vulkan
{
struct GeometryDepthBias
{
    bool enabled;
    float constant, slope;
};
inline GeometryDepthBias geometry_depth_bias(bool shadow, bool wallmark, bool cutout)
{
    return shadow ? GeometryDepthBias{true, 1.25f, 1.5f} :
        wallmark || cutout ? GeometryDepthBias{true, -1.f, -1.f} : GeometryDepthBias{false, 0.f, 0.f};
}
inline VkCompareOp geometry_depth_compare(bool shadow)
{
    return shadow ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_LESS_OR_EQUAL;
}
}
