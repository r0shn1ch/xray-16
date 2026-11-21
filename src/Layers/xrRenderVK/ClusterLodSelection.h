#pragma once

#include "ClusterLodBuilder.h"
#include "LevelPassGraph.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace xray::render::vulkan
{
struct LevelClusterLodEligibility
{
    bool lod_pairs{};
    bool sliding_windows{};
    bool tree{};
    // Static tree/object-LOD meshes keep the regular indexed vertex layout.
    // Wind-deformed tree geometry needs a separate conservative-motion path.
    bool static_tree{};
    bool water{};
    bool glass{};
    bool terrain{}; // Tessellation is baked before cluster construction, so terrain can use the same indexed path.
    bool lod_atlas{};
    bool depth_proxy{};
    bool wallmark_multiply{};
    bool wallmark_blend{};
    SurfaceMode mode{SurfaceMode::Opaque};
    const LevelPassGraph* pass_graph{};
};

inline bool supports_level_cluster_cpu_lod(const LevelClusterLodEligibility& input)
{
    return !input.lod_pairs && (!input.tree || input.static_tree) && !input.water && !input.glass &&
        !input.lod_atlas && !input.depth_proxy && !input.wallmark_multiply && !input.wallmark_blend &&
        input.mode != SurfaceMode::Transparent && input.pass_graph &&
        supports_cluster_lod_pass_graph(*input.pass_graph);
}

inline bool supports_level_cluster_lod(const LevelClusterLodEligibility& input)
{
    return !input.sliding_windows && supports_level_cluster_cpu_lod(input);
}

inline bool cluster_lod_sphere_visible(const float* mvp, const float* center, float radius)
{
    if (!mvp || !center || !std::isfinite(radius) || radius <= 0.f ||
        !std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(center[2]))
        return true;
    const auto component = [&](int column) {
        return mvp[column] * center[0] + mvp[column + 4] * center[1] +
            mvp[column + 8] * center[2] + mvp[column + 12];
    };
    const float cx = component(0), cy = component(1), cz = component(2), cw = component(3);
    const auto outside = [&](float distance, float a, float b, float c) {
        return distance < -radius * std::sqrt(a * a + b * b + c * c);
    };
    return !outside(cx + cw, mvp[0] + mvp[3], mvp[4] + mvp[7], mvp[8] + mvp[11]) &&
        !outside(cw - cx, mvp[3] - mvp[0], mvp[7] - mvp[4], mvp[11] - mvp[8]) &&
        !outside(cy + cw, mvp[1] + mvp[3], mvp[5] + mvp[7], mvp[9] + mvp[11]) &&
        !outside(cw - cy, mvp[3] - mvp[1], mvp[7] - mvp[5], mvp[11] - mvp[9]) &&
        !outside(cz, mvp[2], mvp[6], mvp[10]) &&
        !outside(cw - cz, mvp[3] - mvp[2], mvp[7] - mvp[6], mvp[11] - mvp[10]);
}

enum class ClusterLodDisposition : uint8_t { Invalid, UseOriginalMesh, FullyCulled, UseClusters };

struct ClusterLodSelection
{
    ClusterLodDisposition disposition{ClusterLodDisposition::Invalid};
    std::vector<uint32_t> clusters;
};

inline ClusterLodSelection select_cluster_lod(const ClusterLodData& lod_data, VkExtent2D extent,
    const float* mvp, const float* world_transform, const std::array<float, 3>& camera, float fov)
{
    if (!mvp || lod_data.groups.empty() || lod_data.clusters.empty() || !extent.height)
        return {};

    float world_scale = 1.f;
    if (world_transform)
    {
        float maximum_scale_squared = 0.f;
        for (unsigned basis = 0; basis < 3; ++basis)
        {
            const float x = world_transform[basis * 4];
            const float y = world_transform[basis * 4 + 1];
            const float z = world_transform[basis * 4 + 2];
            maximum_scale_squared = std::max(maximum_scale_squared, x * x + y * y + z * z);
        }
        if (maximum_scale_squared > 0.f && std::isfinite(maximum_scale_squared))
            world_scale = std::sqrt(maximum_scale_squared);
    }
    const auto world_position = [&](const float* local, float* result) {
        if (!world_transform)
        {
            std::copy_n(local, 3, result);
            return;
        }
        result[0] = world_transform[0] * local[0] + world_transform[4] * local[1] +
            world_transform[8] * local[2] + world_transform[12];
        result[1] = world_transform[1] * local[0] + world_transform[5] * local[1] +
            world_transform[9] * local[2] + world_transform[13];
        result[2] = world_transform[2] * local[0] + world_transform[6] * local[1] +
            world_transform[10] * local[2] + world_transform[14];
    };

    constexpr float pi = 3.14159265358979323846f;
    const float half_fov = std::clamp(fov, 1.f, 179.f) * (pi / 360.f);
    const float projection_scale = float(extent.height) * 0.5f / std::tan(half_fov);
    constexpr float target_error_pixels = 1.f;
    const auto projected_error = [&](const float* local_center, float radius, float error) {
        if (!(error >= 0.f) || !std::isfinite(error))
            return std::numeric_limits<float>::infinity();
        float center[3]{};
        world_position(local_center, center);
        const float dx = center[0] - camera[0];
        const float dy = center[1] - camera[1];
        const float dz = center[2] - camera[2];
        const float distance = std::max(std::sqrt(dx * dx + dy * dy + dz * dz) - radius * world_scale, 0.05f);
        return error * world_scale * projection_scale / distance;
    };

    std::vector<float> group_errors(lod_data.groups.size());
    for (size_t i = 0; i < lod_data.groups.size(); ++i)
    {
        const ClusterLodGroup& group = lod_data.groups[i];
        if (group.first_cluster > lod_data.clusters.size() ||
            group.cluster_count > lod_data.clusters.size() - group.first_cluster)
            return {};
        group_errors[i] = projected_error(group.center, group.radius, group.error);
    }
    bool use_cluster_geometry = false;
    std::vector<uint32_t> selected;
    for (size_t group_index = 0; group_index < lod_data.groups.size(); ++group_index)
    {
        const ClusterLodGroup& group = lod_data.groups[group_index];
        if (group_errors[group_index] <= target_error_pixels)
            continue;
        use_cluster_geometry = true;
        for (uint32_t i = 0; i < group.cluster_count; ++i)
        {
            const uint32_t cluster_index = group.first_cluster + i;
            const ClusterLodCluster& cluster = lod_data.clusters[cluster_index];
            if (cluster.refined_group < -1 ||
                (cluster.refined_group >= 0 && size_t(cluster.refined_group) >= group_errors.size()))
                return {};
            if (cluster.refined_group >= 0 &&
                group_errors[size_t(cluster.refined_group)] > target_error_pixels)
                continue;
            if (cluster_lod_sphere_visible(mvp, cluster.center, cluster.radius))
                selected.push_back(cluster_index);
        }
    }
    if (!use_cluster_geometry)
        return {ClusterLodDisposition::UseOriginalMesh, {}};
    if (selected.empty())
        return {ClusterLodDisposition::FullyCulled, {}};
    return {ClusterLodDisposition::UseClusters, std::move(selected)};
}

inline std::vector<uint32_t> select_visible_cluster_lod(const ClusterLodData& lod_data, VkExtent2D extent,
    const float* mvp, const float* world_transform, const std::array<float, 3>& camera, float fov)
{
    ClusterLodSelection selection = select_cluster_lod(lod_data, extent, mvp, world_transform, camera, fov);
    return selection.disposition == ClusterLodDisposition::UseClusters ? std::move(selection.clusters) :
        std::vector<uint32_t>{};
}
} // namespace xray::render::vulkan
