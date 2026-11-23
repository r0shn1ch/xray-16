#include "GpuParticleStaticCollision.h"
#include <cmath>

namespace xray::render::vulkan
{
namespace
{
uint32_t build(std::vector<GpuParticleCollisionNode>& nodes,
    std::vector<GpuParticleCollisionTriangle>& triangles, uint32_t first, uint32_t count)
{
    GpuParticleCollisionNode node{};
    for (unsigned axis = 0; axis < 3; ++axis)
    { node.minimum[axis] = std::numeric_limits<float>::max(); node.maximum[axis] = -node.minimum[axis]; }
    for (uint32_t i = first; i < first + count; ++i)
        for (const float* point : {triangles[i].a, triangles[i].b, triangles[i].c})
            for (unsigned axis = 0; axis < 3; ++axis)
            { node.minimum[axis] = std::min(node.minimum[axis], point[axis]); node.maximum[axis] = std::max(node.maximum[axis], point[axis]); }
    const uint32_t index = static_cast<uint32_t>(nodes.size());
    nodes.push_back(node);
    if (count <= 8) { nodes[index].first = first; nodes[index].count = count; }
    else
    {
        const double x = double(node.maximum[0]) - node.minimum[0], y = double(node.maximum[1]) - node.minimum[1];
        const double z = double(node.maximum[2]) - node.minimum[2];
        const unsigned axis = x >= y && x >= z ? 0 : y >= z ? 1 : 2;
        const uint32_t middle = first + count / 2;
        const auto coordinate = [axis](const GpuParticleCollisionTriangle& triangle) {
            return double(triangle.a[axis]) + triangle.b[axis] + triangle.c[axis];
        };
        std::nth_element(triangles.begin() + first, triangles.begin() + middle, triangles.begin() + first + count,
            [&](const auto& a, const auto& b) { return coordinate(a) < coordinate(b); });
        build(nodes, triangles, first, middle - first);
        build(nodes, triangles, middle, first + count - middle);
    }
    nodes[index].escape = static_cast<uint32_t>(nodes.size());
    return index;
}
}
bool build_gpu_particle_static_collision(const std::vector<GpuParticleCollisionTriangle>& triangles,
    GpuParticleCollisionInput& output, std::string& error)
{
    if (triangles.size() > UINT32_MAX / 2)
    { error = "GPU particle static collision tree is too large"; return false; }
    for (const auto& triangle : triangles)
        for (const float* point : {triangle.a, triangle.b, triangle.c})
            for (unsigned axis = 0; axis < 3; ++axis)
                if (!std::isfinite(point[axis]))
                { error = "nonfinite GPU particle collision vertex"; return false; }
    std::vector<GpuParticleCollisionTriangle> sorted = triangles;
    std::vector<GpuParticleCollisionNode> nodes;
    if (!sorted.empty())
    {
        nodes.reserve(sorted.size() / 2 + 1);
        build(nodes, sorted, 0, static_cast<uint32_t>(sorted.size()));
    }
    output.nodes = std::move(nodes); output.static_triangles = std::move(sorted);
    error.clear(); return true;
}
}
