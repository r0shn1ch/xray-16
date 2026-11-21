#pragma once

#include "FGTypes.h"
#include <cstddef>
#include <set>
#include <utility>
#include <vector>

namespace xray::render::framegraph
{
struct ResourceVersion
{
    uint32_t producer = INVALID_INDEX;
    std::vector<uint32_t> readers;
};

struct ResourceLifetime
{
    uint32_t first_use = INVALID_INDEX;
    uint32_t last_use = INVALID_INDEX;

    void record_use(uint32_t execution_index)
    {
        if (first_use == INVALID_INDEX)
            first_use = execution_index;
        last_use = execution_index;
    }
};

template <typename AddEdge>
void add_resource_dependencies(const std::vector<ResourceVersion>& versions, AddEdge&& add_edge)
{
    for (size_t i = 0; i < versions.size(); ++i)
    {
        const auto& version = versions[i];
        for (uint32_t reader : version.readers)
            add_edge(version.producer, reader);
        if (i + 1 == versions.size())
            continue;
        const uint32_t writer = versions[i + 1].producer;
        add_edge(version.producer, writer);
        for (uint32_t reader : version.readers)
            add_edge(reader, writer);
    }
}

inline bool sort_passes(const std::vector<std::set<uint32_t>>& edges,
    std::vector<uint32_t> indegree, std::vector<uint32_t>& order)
{
    std::set<uint32_t> ready;
    for (uint32_t pass = 0; pass < edges.size(); ++pass)
        if (!indegree[pass]) ready.insert(pass);
    std::vector<uint32_t> sorted;
    sorted.reserve(edges.size());
    while (!ready.empty())
    {
        const uint32_t pass = *ready.begin();
        ready.erase(ready.begin());
        sorted.push_back(pass);
        for (uint32_t consumer : edges[pass])
            if (!--indegree[consumer]) ready.insert(consumer);
    }
    if (sorted.size() != edges.size())
        return false;
    order = std::move(sorted);
    return true;
}
}
