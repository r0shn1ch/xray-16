#include <vulkan/vulkan.h>

#include "Layers/xrRenderVK/ClusterLodBuilder.h"
#include "Layers/xrRenderVK/ClusterLodSelection.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_set>

using namespace xray::render::vulkan;

namespace
{
LevelModel make_wavy_grid(uint32_t cells)
{
    LevelModel mesh;
    const uint32_t side = cells + 1;
    mesh.vertices.resize(size_t(side) * side);
    for (uint32_t y = 0; y < side; ++y)
    {
        for (uint32_t x = 0; x < side; ++x)
        {
            const float u = float(x) / float(cells);
            const float v = float(y) / float(cells);
            LevelVertex& vertex = mesh.vertices[size_t(y) * side + x];
            vertex.position[0] = -0.8f + 1.6f * u;
            vertex.position[1] = -0.8f + 1.6f * v;
            vertex.position[2] = 0.5f + 0.15f * std::sin(u * 6.2831853f) * std::cos(v * 6.2831853f);
            vertex.normal[2] = 1.f;
            vertex.uv[0] = u;
            vertex.uv[1] = v;
            vertex.lightmap_uv[0] = u;
            vertex.lightmap_uv[1] = v;
        }
    }

    mesh.indices.reserve(size_t(cells) * cells * 6);
    for (uint32_t y = 0; y < cells; ++y)
    {
        for (uint32_t x = 0; x < cells; ++x)
        {
            const uint32_t a = y * side + x;
            const uint32_t b = a + 1;
            const uint32_t c = a + side;
            const uint32_t d = c + 1;
            mesh.indices.insert(mesh.indices.end(), {a, b, c, b, d, c});
        }
    }
    return mesh;
}

void verify_selection(const std::vector<uint32_t>& selected, const ClusterLodData& data)
{
    assert(!selected.empty());
    std::unordered_set<uint32_t> seen;
    for (uint32_t index : selected)
    {
        assert(index < data.clusters.size());
        assert(seen.insert(index).second);
    }
}
}

int main()
{
    LevelModel mesh = make_wavy_grid(24);
    ClusterLodData lod;
    std::string error;
    assert(build_cluster_lod(mesh, lod, error));
    assert(!lod.groups.empty() && !lod.clusters.empty());
    assert(lod.groups.size() > 1);
    for (const ClusterLodGroup& group : lod.groups)
    {
        assert(group.first_cluster <= lod.clusters.size());
        assert(group.cluster_count <= lod.clusters.size() - group.first_cluster);
        assert(std::isfinite(group.center[0]) && std::isfinite(group.center[1]) &&
            std::isfinite(group.center[2]));
        assert(group.radius > 0.f);
    }
    for (const ClusterLodCluster& cluster : lod.clusters)
    {
        assert(cluster.first_index <= lod.indices.size());
        assert(cluster.index_count <= lod.indices.size() - cluster.first_index);
        assert(cluster.index_count % 3 == 0);
        assert(cluster.refined_group < 0 || size_t(cluster.refined_group) < lod.groups.size());
        for (uint32_t i = 0; i < cluster.index_count; ++i)
            assert(lod.indices[cluster.first_index + i] < mesh.vertices.size());
    }

    // Progressive OGF windows need independent hierarchies. Their packed GPU
    // index ranges are distinct, and each hierarchy must retain its own
    // original vertex block.
    ModelGeometry ogf;
    const size_t vertex_count = mesh.vertices.size();
    ogf.vertices.resize(vertex_count * 2);
    for (size_t block = 0; block < 2; ++block)
    {
        for (size_t i = 0; i < vertex_count; ++i)
        {
            const LevelVertex& source = mesh.vertices[i];
            ModelVertex& target = ogf.vertices[block * vertex_count + i];
            std::copy_n(source.position, 3, target.position);
            std::copy_n(source.normal, 3, target.normal);
            std::copy_n(source.uv, 2, target.uv);
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                target.tangent[axis] = float(source.tangent[axis]) / 127.f;
                target.bitangent[axis] = float(source.bitangent[axis]) / 127.f;
            }
            target.position[0] += float(block) * 4.f;
        }
    }
    const uint32_t split = static_cast<uint32_t>((mesh.indices.size() / 6) * 3);
    ogf.indices.insert(ogf.indices.end(), mesh.indices.begin(), mesh.indices.begin() + split);
    ogf.windows.push_back({0, split, static_cast<uint16_t>(vertex_count)});
    const uint32_t second_offset = static_cast<uint32_t>(ogf.indices.size());
    for (size_t i = split; i < mesh.indices.size(); ++i)
        ogf.indices.push_back(static_cast<uint32_t>(vertex_count) + mesh.indices[i]);
    ogf.windows.push_back({second_offset, static_cast<uint32_t>(ogf.indices.size()) - second_offset,
        static_cast<uint16_t>(vertex_count * 2)});
    ClusterLodData first_window_lod, second_window_lod;
    assert(build_cluster_lod(ogf, ogf.windows[0], first_window_lod, error));
    assert(build_cluster_lod(ogf, ogf.windows[1], second_window_lod, error));
    assert(!first_window_lod.clusters.empty() && !second_window_lod.clusters.empty());
    assert(first_window_lod.indices.size() <= ogf.indices.size());
    assert(second_window_lod.indices.size() <= ogf.indices.size());
    for (uint32_t index : first_window_lod.indices)
        assert(index < vertex_count);
    for (uint32_t index : second_window_lod.indices)
        assert(index >= vertex_count && index < vertex_count * 2);

    LevelModel progressive;
    progressive.vertices = mesh.vertices;
    progressive.vertices.insert(progressive.vertices.end(), mesh.vertices.begin(), mesh.vertices.end());
    for (size_t i = 0; i < vertex_count; ++i)
        progressive.vertices[vertex_count + i].position[0] += 4.f;
    progressive.indices = ogf.indices;
    progressive.windows = ogf.windows;
    ClusterLodData level_first_window, level_second_window;
    assert(build_cluster_lod(progressive, progressive.windows[0], level_first_window, error));
    assert(build_cluster_lod(progressive, progressive.windows[1], level_second_window, error));
    for (uint32_t index : level_first_window.indices)
        assert(index < vertex_count);
    for (uint32_t index : level_second_window.indices)
        assert(index >= vertex_count && index < vertex_count * 2);
    std::vector<uint32_t> packed_window_indices{UINT32_MAX};
    uint32_t first_window_base = 0, first_window_count = 0;
    uint32_t second_window_base = 0, second_window_count = 0;
    assert(append_packed_cluster_indices(packed_window_indices, level_first_window,
        first_window_base, first_window_count, error));
    assert(append_packed_cluster_indices(packed_window_indices, level_second_window,
        second_window_base, second_window_count, error));
    assert(first_window_base == 1 && first_window_count > 0);
    assert(second_window_base == first_window_base + first_window_count && second_window_count > 0);
    assert(level_first_window.indices.empty() && level_second_window.indices.empty());
    for (const ClusterLodCluster& cluster : level_second_window.clusters)
    {
        assert(cluster.first_index <= second_window_count);
        assert(cluster.index_count <= second_window_count - cluster.first_index);
        for (uint32_t i = 0; i < cluster.index_count; ++i)
        {
            const uint32_t index = packed_window_indices[second_window_base + cluster.first_index + i];
            assert(index >= vertex_count && index < vertex_count * 2);
        }
    }
    SlideWindow invalid_window{static_cast<uint32_t>(progressive.indices.size()), 3, 0};
    ClusterLodData invalid_window_lod;
    assert(!build_cluster_lod(progressive, invalid_window, invalid_window_lod, error));
    assert(invalid_window_lod.groups.empty() && invalid_window_lod.clusters.empty() &&
        invalid_window_lod.indices.empty());

    LevelModel invalid_index = mesh;
    invalid_index.indices[0] = static_cast<uint32_t>(invalid_index.vertices.size());
    ClusterLodData rejected;
    assert(!build_cluster_lod(invalid_index, rejected, error));
    assert(rejected.groups.empty() && rejected.clusters.empty() && rejected.indices.empty());
    LevelModel invalid_position = mesh;
    invalid_position.vertices[0].position[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!build_cluster_lod(invalid_position, rejected, error));
    assert(rejected.groups.empty() && rejected.clusters.empty() && rejected.indices.empty());

    const float identity[16]{
        1.f, 0.f, 0.f, 0.f,
        0.f, 1.f, 0.f, 0.f,
        0.f, 0.f, 1.f, 0.f,
        0.f, 0.f, 0.f, 1.f};
    const VkExtent2D extent{1920, 1080};
    const std::array<float, 3> near_camera{0.f, 0.f, -2.f};
    const std::array<float, 3> far_camera{0.f, 0.f, -100000.f};
    const auto near_result = select_cluster_lod(lod, extent, identity, nullptr, near_camera, 60.f);
    const auto far_result = select_cluster_lod(lod, extent, identity, nullptr, far_camera, 60.f);
    assert(near_result.disposition == ClusterLodDisposition::UseClusters && !near_result.clusters.empty());
    ClusterLodData low_error_lod = lod;
    for (ClusterLodGroup& group : low_error_lod.groups)
        group.error = 0.f;
    const auto original_mesh_result = select_cluster_lod(low_error_lod, extent, identity, nullptr,
        near_camera, 60.f);
    assert(original_mesh_result.disposition == ClusterLodDisposition::UseOriginalMesh &&
        original_mesh_result.clusters.empty());
    verify_selection(near_result.clusters, lod);
    verify_selection(far_result.clusters, lod);

    LevelPassGraph opaque_pass_graph{};
    opaque_pass_graph.count = 1;
    opaque_pass_graph.passes[0].fragment = "vk\\level_lightmap.ps";
    LevelClusterLodEligibility eligibility{};
    eligibility.pass_graph = &opaque_pass_graph;
    assert(supports_level_cluster_lod(eligibility));
    eligibility.tree = true;
    assert(!supports_level_cluster_lod(eligibility));
    eligibility.static_tree = true;
    assert(supports_level_cluster_lod(eligibility));
    eligibility.tree = false;
    eligibility.mode = SurfaceMode::Transparent;
    assert(!supports_level_cluster_lod(eligibility));
    assert(near_result.clusters.size() >= far_result.clusters.size());

    float outside_frustum[16];
    std::copy(std::begin(identity), std::end(identity), std::begin(outside_frustum));
    outside_frustum[12] = 4.f;
    const auto culled_result = select_cluster_lod(lod, extent, outside_frustum, nullptr,
        near_camera, 60.f);
    assert(culled_result.disposition == ClusterLodDisposition::FullyCulled && culled_result.clusters.empty());
    assert(select_cluster_lod(lod, {1920, 0}, identity, nullptr,
        near_camera, 60.f).disposition == ClusterLodDisposition::Invalid);
    assert(select_cluster_lod(lod, extent, nullptr, nullptr,
        near_camera, 60.f).disposition == ClusterLodDisposition::Invalid);

    ClusterLodData malformed = lod;
    malformed.groups[0].first_cluster = std::numeric_limits<uint32_t>::max();
    malformed.groups[0].cluster_count = 1;
    malformed.groups[0].error = std::numeric_limits<float>::max();
    const auto malformed_result = select_cluster_lod(malformed, extent, identity,
        nullptr, near_camera, 60.f);
    assert(malformed_result.disposition == ClusterLodDisposition::Invalid);
    assert(malformed_result.clusters.empty());
}
