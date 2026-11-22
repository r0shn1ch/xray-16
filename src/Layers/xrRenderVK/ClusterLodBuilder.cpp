#include "xrEngine/stdafx.h"

#include "ClusterBasis.h"
#include "ClusterDeviation.h"
#include "ClusterLodBuilder.h"
#include "clusterlod.h"
#include "TangentBasis.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace xray::render::vulkan
{
namespace
{
constexpr uint32_t kAttributeDirections = 9;
constexpr uint32_t kAttributeWeighted = kAttributeDirections + 2;
constexpr uint32_t kAttributeStride = kAttributeWeighted + 2;
constexpr uint32_t kAttributeProtectMask = ((1u << (kAttributeStride - kAttributeDirections)) - 1u)
    << kAttributeDirections;
constexpr uint32_t kClusterMaxTriangles = 128;
constexpr float kDirectionWeight = 0.5f;
constexpr float kMinDirectionLengthSquared = 1e-8f;

struct BuildContext
{
    ClusterLodData* output{};
    bool failed{};
};

float length_squared(const float* v)
{
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
}

void normalize_or(float* v, const float* fallback)
{
    const float len_sq = length_squared(v);
    if (!(len_sq > kMinDirectionLengthSquared) || !std::isfinite(len_sq))
    {
        std::copy(fallback, fallback + 3, v);
        return;
    }
    const float inv_len = 1.f / std::sqrt(len_sq);
    for (unsigned i = 0; i < 3; ++i)
        v[i] *= inv_len;
}

void remove_normal_component(float* tangent, const float* normal)
{
    const float dot = tangent[0] * normal[0] + tangent[1] * normal[1] + tangent[2] * normal[2];
    for (unsigned i = 0; i < 3; ++i)
        tangent[i] -= dot * normal[i];
}

float estimate_uv_world_scale(const LevelModel& mesh, const uint32_t* indices, size_t index_count)
{
    std::vector<float> ratios;
    ratios.reserve(index_count);
    for (size_t i = 0; i + 2 < index_count; i += 3)
    {
        const uint32_t tri[3]{indices[i], indices[i + 1], indices[i + 2]};
        for (unsigned edge = 0; edge < 3; ++edge)
        {
            const LevelVertex& a = mesh.vertices[tri[edge]];
            const LevelVertex& b = mesh.vertices[tri[(edge + 1) % 3]];
            const float dx = a.position[0] - b.position[0];
            const float dy = a.position[1] - b.position[1];
            const float dz = a.position[2] - b.position[2];
            const float du = a.uv[0] - b.uv[0];
            const float dv = a.uv[1] - b.uv[1];
            const float position_distance = std::sqrt(dx * dx + dy * dy + dz * dz);
            const float uv_distance = std::sqrt(du * du + dv * dv);
            if (uv_distance > 1e-5f && position_distance > 1e-5f)
                ratios.push_back(position_distance / uv_distance);
        }
    }
    if (ratios.empty())
        return 1.f;
    auto middle = ratios.begin() + ratios.size() / 2;
    std::nth_element(ratios.begin(), middle, ratios.end());
    return std::clamp(*middle, 0.01f, 1000.f);
}

void fill_attributes(const LevelModel& mesh, std::vector<float>& attributes)
{
    attributes.resize(mesh.vertices.size() * kAttributeStride);
    constexpr float default_normal[3]{0.f, 0.f, 1.f};
    constexpr float default_tangent[3]{1.f, 0.f, 0.f};
    constexpr float default_binormal[3]{0.f, 1.f, 0.f};

    for (size_t i = 0; i < mesh.vertices.size(); ++i)
    {
        const LevelVertex& vertex = mesh.vertices[i];
        float* a = attributes.data() + i * kAttributeStride;
        float normal[3]{vertex.normal[0], vertex.normal[1], vertex.normal[2]};
        normalize_or(normal, default_normal);

        float tangent[3]{float(vertex.tangent[0]) / 127.f, float(vertex.tangent[1]) / 127.f,
                         float(vertex.tangent[2]) / 127.f};
        remove_normal_component(tangent, normal);
        if (!(length_squared(tangent) > kMinDirectionLengthSquared))
        {
            Fvector3 n, t, b;
            n.set(normal[0], normal[1], normal[2]);
            ClusterDeriveBasis(n, t, b);
            tangent[0] = t.x;
            tangent[1] = t.y;
            tangent[2] = t.z;
        }
        normalize_or(tangent, default_tangent);

        float binormal[3]{float(vertex.bitangent[0]) / 127.f, float(vertex.bitangent[1]) / 127.f,
                          float(vertex.bitangent[2]) / 127.f};
        remove_normal_component(binormal, normal);
        remove_normal_component(binormal, tangent);
        if (!(length_squared(binormal) > kMinDirectionLengthSquared))
        {
            binormal[0] = normal[1] * tangent[2] - normal[2] * tangent[1];
            binormal[1] = normal[2] * tangent[0] - normal[0] * tangent[2];
            binormal[2] = normal[0] * tangent[1] - normal[1] * tangent[0];
        }
        normalize_or(binormal, default_binormal);

        std::copy(normal, normal + 3, a + 0);
        std::copy(tangent, tangent + 3, a + 3);
        std::copy(binormal, binormal + 3, a + 6);
        a[9] = vertex.uv[0];
        a[10] = vertex.uv[1];
        a[11] = vertex.lightmap_uv[0];
        a[12] = vertex.lightmap_uv[1];
    }
}

float deviation_error_hook(void* context, const clodMesh* mesh, const unsigned int* source, size_t source_count,
                           const unsigned int* result, size_t result_count, unsigned int method, float error)
{
    auto* build = static_cast<BuildContext*>(context);
    if (method & 4u)
        return error;

    float deviation = 0.f;
    if (!ClusterMeshDeviation(mesh->vertex_positions, mesh->vertex_positions_stride / sizeof(float),
                              mesh->vertex_attributes, mesh->vertex_attributes_stride / sizeof(float),
                              mesh->attribute_weights, static_cast<uint32_t>(mesh->attribute_count), source,
                              source_count, result, result_count, deviation))
    {
        build->failed = true;
        return error;
    }
    return std::max(error, deviation);
}

int append_group(void* context, clodGroup group, const clodCluster* clusters, size_t cluster_count)
{
    auto* build = static_cast<BuildContext*>(context);
    if (!build || !build->output || (cluster_count && !clusters))
        return -1;
    try
    {
        ClusterLodData& out = *build->output;
        if (out.groups.size() >= static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
            cluster_count > std::numeric_limits<uint32_t>::max() - out.clusters.size())
        {
            build->failed = true;
            return -1;
        }

        ClusterLodGroup output_group{};
        output_group.depth = group.depth;
        std::copy(group.simplified.center, group.simplified.center + 3, output_group.center);
        output_group.radius = group.simplified.radius;
        output_group.error = group.simplified.error;
        output_group.first_cluster = static_cast<uint32_t>(out.clusters.size());
        output_group.cluster_count = static_cast<uint32_t>(cluster_count);
        const int group_id = static_cast<int>(out.groups.size());
        out.groups.push_back(output_group);

        for (size_t i = 0; i < cluster_count; ++i)
        {
            if (out.indices.size() > std::numeric_limits<uint32_t>::max() ||
                clusters[i].index_count > std::numeric_limits<uint32_t>::max() - out.indices.size())
            {
                build->failed = true;
                return -1;
            }
            ClusterLodCluster cluster{};
            cluster.refined_group = clusters[i].refined;
            std::copy(clusters[i].bounds.center, clusters[i].bounds.center + 3, cluster.center);
            cluster.radius = clusters[i].bounds.radius;
            cluster.error = clusters[i].bounds.error;
            cluster.first_index = static_cast<uint32_t>(out.indices.size());
            cluster.index_count = static_cast<uint32_t>(clusters[i].index_count);
            cluster.vertex_count = static_cast<uint32_t>(clusters[i].vertex_count);
            out.indices.insert(out.indices.end(), clusters[i].indices, clusters[i].indices + clusters[i].index_count);
            out.clusters.push_back(cluster);
        }
        return group_id;
    }
    catch (...)
    {
        build->failed = true;
        return -1;
    }
}
}

bool build_cluster_lod(const LevelModel& mesh, ClusterLodData& result, std::string& error)
{
    if (mesh.indices.size() > std::numeric_limits<uint32_t>::max())
    {
        result = {};
        error = "cluster LOD input exceeds the indexed range";
        return false;
    }
    return build_cluster_lod(mesh,
        SlideWindow{0, static_cast<uint32_t>(mesh.indices.size()), 0}, result, error);
}

bool build_cluster_lod(const LevelModel& mesh, const SlideWindow& window,
    ClusterLodData& result, std::string& error)
{
    result = {};
    if (mesh.vertices.empty() || mesh.indices.empty() || window.index_count % 3 != 0 ||
        window.offset > mesh.indices.size() || window.index_count > mesh.indices.size() - window.offset ||
        mesh.vertices.size() > std::numeric_limits<unsigned int>::max() ||
        window.index_count > std::numeric_limits<unsigned int>::max())
    {
        error = "cluster LOD input has an invalid vertex or triangle index range";
        return false;
    }
    // A terminal progressive LOD may have vertices but no remaining faces.
    if (!window.index_count)
    {
        error.clear();
        return true;
    }
    const uint32_t* indices = mesh.indices.data() + window.offset;
    for (size_t i = 0; i < window.index_count; ++i)
    {
        const uint32_t index = indices[i];
        if (index >= mesh.vertices.size())
        {
            error = "cluster LOD input index is outside the vertex array";
            return false;
        }
        if (window.vertex_count && index >= window.vertex_count)
        {
            error = "cluster LOD window references an inactive vertex";
            return false;
        }
    }

    std::vector<float> positions(mesh.vertices.size() * 3);
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
    {
        const LevelVertex& vertex = mesh.vertices[i];
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            const float value = vertex.position[axis];
            if (!std::isfinite(value))
            {
                error = "cluster LOD input contains a non-finite vertex position";
                return false;
            }
            positions[i * 3 + axis] = value;
        }
        for (unsigned axis = 0; axis < 2; ++axis)
        {
            if (!std::isfinite(vertex.uv[axis]) || !std::isfinite(vertex.lightmap_uv[axis]))
            {
                error = "cluster LOD input contains a non-finite texture coordinate";
                return false;
            }
        }
    }

    std::vector<float> attributes;
    fill_attributes(mesh, attributes);
    const float uv_scale = estimate_uv_world_scale(mesh, indices, window.index_count);
    float attribute_weights[kAttributeWeighted];
    for (uint32_t i = 0; i < kAttributeDirections; ++i)
        attribute_weights[i] = kDirectionWeight;
    attribute_weights[kAttributeDirections] = uv_scale;
    attribute_weights[kAttributeDirections + 1] = uv_scale;

    clodMesh input{};
    input.indices = indices;
    input.index_count = window.index_count;
    input.vertex_count = mesh.vertices.size();
    input.vertex_positions = positions.data();
    input.vertex_positions_stride = sizeof(float) * 3;
    input.vertex_attributes = attributes.data();
    input.vertex_attributes_stride = sizeof(float) * kAttributeStride;
    input.attribute_weights = attribute_weights;
    input.attribute_count = kAttributeWeighted;
    input.attribute_protect_mask = kAttributeProtectMask;

    clodConfig config = clodDefaultConfig(kClusterMaxTriangles);
    config.optimize_bounds = true;
    config.optimize_clusters = true;
    config.simplify_permissive = false;
    config.simplify_fallback_permissive = false;
    config.simplify_fallback_sloppy = false;
    config.simplify_prune = false;

    BuildContext context{&result, false};
    config.simplify_error_hook = deviation_error_hook;
    config.simplify_error_context = &context;
    const size_t built_clusters = clodBuild(config, input, &context, append_group);
    if (context.failed || result.groups.empty() || result.clusters.empty() || built_clusters != result.clusters.size())
    {
        result = {};
        error = "meshoptimizer failed to bake the cluster LOD hierarchy";
        return false;
    }
    return true;
}

bool build_cluster_lod(const ModelGeometry& mesh, ClusterLodData& result, std::string& error)
{
    if (mesh.indices.size() > std::numeric_limits<uint32_t>::max())
    {
        result = {};
        error = "cluster LOD model input exceeds the indexed range";
        return false;
    }
    return build_cluster_lod(mesh,
        SlideWindow{0, static_cast<uint32_t>(mesh.indices.size()), 0}, result, error);
}

bool build_cluster_lod(const ModelGeometry& mesh, const SlideWindow& window,
    ClusterLodData& result, std::string& error)
{
    result = {};
    if (mesh.vertices.empty() || mesh.indices.empty() || mesh.type == 4 || mesh.type == 5 ||
        window.index_count % 3 || window.offset > mesh.indices.size() ||
        window.index_count > mesh.indices.size() - window.offset)
    {
        error = "cluster LOD model window is skinned or outside its index buffer";
        return false;
    }
    if (!window.index_count)
    {
        error.clear();
        return true;
    }
    for (size_t i = window.offset; i < size_t(window.offset) + window.index_count; ++i)
        if (mesh.indices[i] >= mesh.vertices.size() ||
            (window.vertex_count && mesh.indices[i] >= window.vertex_count))
        {
            error = "cluster LOD model window references an inactive vertex";
            return false;
        }

    LevelModel level;
    level.indices.assign(mesh.indices.begin() + window.offset,
        mesh.indices.begin() + window.offset + window.index_count);
    level.vertices.resize(mesh.vertices.size());
    for (size_t i = 0; i < mesh.vertices.size(); ++i)
    {
        const ModelVertex& source = mesh.vertices[i];
        LevelVertex& target = level.vertices[i];
        std::copy_n(source.position, 3, target.position);
        std::copy_n(source.normal, 3, target.normal);
        std::copy_n(source.uv, 2, target.uv);
        for (unsigned axis = 0; axis < 3; ++axis)
        {
            tangent_store(target.tangent[axis], source.tangent[axis]);
            tangent_store(target.bitangent[axis], source.bitangent[axis]);
        }
    }
    return build_cluster_lod(level, result, error);
}

bool append_packed_cluster_indices(std::vector<uint32_t>& packed, ClusterLodData& source,
    uint32_t& first_index, uint32_t& index_count, std::string& error)
{
    if (packed.size() > std::numeric_limits<uint32_t>::max() ||
        source.indices.size() > std::numeric_limits<uint32_t>::max() - packed.size())
    {
        error = "cluster DAG packed index range exceeds 32-bit draw offsets";
        return false;
    }
    first_index = static_cast<uint32_t>(packed.size());
    index_count = static_cast<uint32_t>(source.indices.size());
    packed.insert(packed.end(), source.indices.begin(), source.indices.end());
    std::vector<uint32_t>().swap(source.indices);
    error.clear();
    return true;
}
}
