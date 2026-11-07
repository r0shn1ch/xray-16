#pragma once
#include "ModelGeometry.h"
#include <array>
#include <algorithm>
#include <cmath>
namespace xray::render::vulkan
{
inline float tangent_value(float value)
{
    return value;
}
inline float tangent_value(int8_t value)
{
    return float(value) / 127.f;
}
inline void tangent_store(float &target, float value)
{
    target = value;
}
inline void tangent_store(int8_t &target, float value)
{
    target = int8_t(std::round(std::clamp(value, -1.f, 1.f) * 127.f));
}
// Construct missing frames once from object-space triangles, not from view-
// dependent pixel derivatives. Authored OGF frames take precedence.
template <class Vertex> void prepare_tangent_basis(std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
{
    std::vector<std::array<float, 6>> frames(vertices.size());
    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        if (a >= vertices.size() || b >= vertices.size() || c >= vertices.size())
            continue;
        const auto &A = vertices[a];
        const auto &B = vertices[b];
        const auto &C = vertices[c];
        const float u = B.uv[0] - A.uv[0], w = B.uv[1] - A.uv[1], x = C.uv[0] - A.uv[0], y = C.uv[1] - A.uv[1];
        const float det = u * y - w * x;
        if (!std::isfinite(det) || std::abs(det) < 1e-12f)
            continue;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const float e = B.position[axis] - A.position[axis], f = C.position[axis] - A.position[axis];
            const float t = (e * y - f * w) / det, n = (f * u - e * x) / det;
            for (auto id : {a, b, c})
            {
                frames[id][axis] += t;
                frames[id][axis + 3] += n;
            }
        }
    }
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        auto &vertex = vertices[i];
        float authored = 0, authored_b = 0;
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            authored += std::abs(tangent_value(vertex.tangent[axis]));
            authored_b += std::abs(tangent_value(vertex.bitangent[axis]));
        }
        if (authored > .01f && authored_b > .01f)
            continue;
        for (unsigned frame = 0; frame < 2; ++frame)
        {
            float length = 0;
            for (unsigned axis = 0; axis < 3; ++axis)
                length += frames[i][frame * 3 + axis] * frames[i][frame * 3 + axis];
            if (!std::isfinite(length) || length < 1e-12f)
                continue;
            for (unsigned axis = 0; axis < 3; ++axis)
                tangent_store(frame ? vertex.bitangent[axis] : vertex.tangent[axis],
                              frames[i][frame * 3 + axis] / std::sqrt(length));
        }
    }
}
} // namespace xray::render::vulkan
