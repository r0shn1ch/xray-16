#pragma once

#include "LevelModels.h"
#include "ModelGeometry.h"

#include <cstdint>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct ClusterLodGroup
{
    int32_t depth{};
    float center[3]{};
    float radius{};
    float error{};
    uint32_t first_cluster{};
    uint32_t cluster_count{};
};

struct ClusterLodCluster
{
    int32_t refined_group{-1};
    float center[3]{};
    float radius{};
    float error{};
    uint32_t first_index{};
    uint32_t index_count{};
    uint32_t vertex_count{};
};

struct ClusterLodData
{
    std::vector<ClusterLodGroup> groups;
    std::vector<ClusterLodCluster> clusters;
    std::vector<uint32_t> indices;
};

// Builds the upstream meshoptimizer cluster LOD hierarchy for one decoded
// level mesh. Cluster indices refer to the original LevelModel vertex array.
bool build_cluster_lod(const LevelModel& mesh, ClusterLodData& result, std::string& error);
bool build_cluster_lod(const LevelModel& mesh, const SlideWindow& window,
    ClusterLodData& result, std::string& error);
bool build_cluster_lod(const ModelGeometry& mesh, ClusterLodData& result, std::string& error);
bool build_cluster_lod(const ModelGeometry& mesh, const SlideWindow& window,
    ClusterLodData& result, std::string& error);
bool append_packed_cluster_indices(std::vector<uint32_t>& packed, ClusterLodData& source,
    uint32_t& first_index, uint32_t& index_count, std::string& error);
}
