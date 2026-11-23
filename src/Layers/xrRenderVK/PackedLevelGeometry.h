#pragma once

#include "LevelModels.h"

#include <cstdint>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// One pair of Vulkan buffers holds many complete meshes. Each index remains
// local to its mesh; vkCmdDrawIndexed adds first_vertex as vertexOffset.
// Keeping index values local preserves OGF sliding-window offsets and bounds.
struct PackedLevelGeometry
{
    static constexpr size_t TargetBytes = 16u * 1024u * 1024u;
    std::vector<LevelVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<uint32_t> cluster_indices;

    bool would_flush(const LevelModel& model, size_t cluster_index_count = 0) const
    {
        return !vertices.empty() &&
            (vertices.size() * sizeof(LevelVertex) + indices.size() * sizeof(uint32_t) +
             cluster_indices.size() * sizeof(uint32_t) + cluster_index_count * sizeof(uint32_t) +
             model.vertices.size() * sizeof(LevelVertex) +
             model.indices.size() * sizeof(uint32_t) > TargetBytes);
    }

    bool append(const LevelModel& model, uint32_t& first_vertex,
        uint32_t& first_index, std::string& error)
    {
        if (model.vertices.empty() || model.indices.empty() ||
            model.vertices.size() > size_t(INT32_MAX) ||
            vertices.size() > size_t(INT32_MAX) - model.vertices.size() ||
            model.indices.size() > size_t(UINT32_MAX) ||
            indices.size() > size_t(UINT32_MAX) - model.indices.size())
        {
            error = "level geometry batch has empty mesh or index/vertex offset overflow";
            return false;
        }
        first_vertex = static_cast<uint32_t>(vertices.size());
        first_index = static_cast<uint32_t>(indices.size());
        vertices.insert(vertices.end(), model.vertices.begin(), model.vertices.end());
        indices.insert(indices.end(), model.indices.begin(), model.indices.end());
        return true;
    }
    bool append_cluster_indices(const std::vector<uint32_t>& source, uint32_t& first_index, std::string& error)
    {
        if (source.size() > size_t(UINT32_MAX) || cluster_indices.size() > size_t(UINT32_MAX) - source.size())
        {
            error = "level cluster index batch exceeds Vulkan draw range";
            return false;
        }
        first_index = static_cast<uint32_t>(cluster_indices.size());
        cluster_indices.insert(cluster_indices.end(), source.begin(), source.end());
        return true;
    }
    void reset()
    {
        std::vector<LevelVertex>().swap(vertices);
        std::vector<uint32_t>().swap(indices);
        std::vector<uint32_t>().swap(cluster_indices);
    }
};
}
