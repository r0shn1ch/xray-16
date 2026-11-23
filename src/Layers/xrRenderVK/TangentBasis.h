#pragma once
#include "ModelGeometry.h"
#include <algorithm>
#include <array>
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
inline float basis_dot(const std::array<float, 3> &a, const std::array<float, 3> &b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
inline std::array<float, 3> basis_cross(const std::array<float, 3> &a, const std::array<float, 3> &b)
{
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}
template <class Vertex> bool valid_tangent_basis(const Vertex &v)
{
    std::array<float, 3> n{v.normal[0], v.normal[1], v.normal[2]}, t{}, b{};
    const float n2 = basis_dot(n, n);
    if (!(n2 > 1e-12f && n2 < 1e20f))
        return false;
    for (unsigned i = 0; i < 3; ++i)
    {
        n[i] /= std::sqrt(n2);
        t[i] = tangent_value(v.tangent[i]);
        b[i] = tangent_value(v.bitangent[i]);
    }
    const float nt = basis_dot(n, t), nb = basis_dot(n, b);
    for (unsigned i = 0; i < 3; ++i)
    {
        t[i] -= n[i] * nt;
        b[i] -= n[i] * nb;
    }
    const float t2 = basis_dot(t, t), b2 = basis_dot(b, b), det = basis_dot(basis_cross(n, t), b);
    return t2 > 1e-12f && b2 > 1e-12f && t2 < 1e20f && b2 < 1e20f && det * det > t2 * b2 * 1e-6f;
}
// Valid authored data needs no triangle pass or temporary allocation. Repair
// missing/collapsed frames in object space, before view-dependent derivatives.
template <class Vertex>
void prepare_tangent_basis(std::vector<Vertex> &vertices, const std::vector<uint32_t> &indices)
{
    if (std::all_of(vertices.begin(), vertices.end(), [](const Vertex &v) { return valid_tangent_basis(v); }))
        return;
    std::vector<uint8_t> repair(vertices.size());
    for (size_t i = 0; i < vertices.size(); ++i)
        repair[i] = !valid_tangent_basis(vertices[i]);
    std::vector<std::array<float, 9>> frames(vertices.size());
    for (size_t i = 0; i + 2 < indices.size(); i += 3)
    {
        const uint32_t a = indices[i], b = indices[i + 1], c = indices[i + 2];
        if (a >= vertices.size() || b >= vertices.size() || c >= vertices.size() ||
            !(repair[a] || repair[b] || repair[c]))
            continue;
        const auto &A = vertices[a], &B = vertices[b], &C = vertices[c];
        std::array<float, 3> e{}, f{};
        for (unsigned k = 0; k < 3; ++k)
        {
            e[k] = B.position[k] - A.position[k];
            f[k] = C.position[k] - A.position[k];
        }
        const auto face = basis_cross(e, f);
        for (auto id : {a, b, c})
            if (repair[id])
                for (unsigned k = 0; k < 3; ++k)
                    frames[id][6 + k] += face[k];
        const float u = B.uv[0] - A.uv[0], w = B.uv[1] - A.uv[1], x = C.uv[0] - A.uv[0], y = C.uv[1] - A.uv[1];
        const float det = u * y - w * x;
        if (!std::isfinite(det) || std::abs(det) < 1e-12f)
            continue;
        for (unsigned k = 0; k < 3; ++k)
            for (auto id : {a, b, c})
                if (repair[id])
                {
                    frames[id][k] += (e[k] * y - f[k] * w) / det;
                    frames[id][3 + k] += (f[k] * u - e[k] * x) / det;
                }
    }
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        if (!repair[i])
            continue;
        auto &v = vertices[i];
        std::array<float, 3> n{v.normal[0], v.normal[1], v.normal[2]}, t{}, b{};
        float n2 = basis_dot(n, n);
        if (!(n2 > 1e-12f && n2 < 1e20f))
        {
            n = {frames[i][6], frames[i][7], frames[i][8]};
            n2 = basis_dot(n, n);
        }
        if (!(n2 > 1e-12f && n2 < 1e20f))
        {
            n = {0, 1, 0};
            n2 = 1;
        }
        for (unsigned k = 0; k < 3; ++k)
        {
            n[k] /= std::sqrt(n2);
            v.normal[k] = n[k];
            t[k] = frames[i][k];
            b[k] = frames[i][3 + k];
        }
        const float nt = basis_dot(n, t);
        for (unsigned k = 0; k < 3; ++k)
            t[k] -= n[k] * nt;
        const float t2 = basis_dot(t, t);
        if (!(t2 > 1e-12f && t2 < 1e20f))
        {
            for (unsigned k = 0; k < 3; ++k)
                v.tangent[k] = v.bitangent[k] = 0;
            continue;
        }
        for (float &value : t)
            value /= std::sqrt(t2);
        auto crossed = basis_cross(n, t);
        const float sign = basis_dot(crossed, b) < 0 ? -1.f : 1.f;
        for (unsigned k = 0; k < 3; ++k)
        {
            tangent_store(v.tangent[k], t[k]);
            tangent_store(v.bitangent[k], crossed[k] * sign);
        }
    }
}
} // namespace xray::render::vulkan
