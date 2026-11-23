#pragma once

#include "FGTypes.h"
#include <cstddef>
#include <limits>
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

// Shared SSA-style version tracking for imported and transient resources.
// Backends validate physical ranges and access states before calling this.
class ResourceVersionSequence
{
public:
    explicit ResourceVersionSequence(bool imported = true) : imported_(imported) {}

    bool read(uint32_t version, uint32_t pass)
    {
        if (!readable(version, pass)) return false;
        versions_[version].readers.push_back(pass);
        return true;
    }

    bool write(uint32_t version, uint32_t pass, bool preserve, uint32_t& output)
    {
        if (pass == INVALID_INDEX || version >= versions_.size() || version + 1 != versions_.size() ||
            versions_.size() >= INVALID_INDEX || (preserve && !readable(version, pass)))
            return false;
        if (preserve) versions_[version].readers.push_back(pass);
        output = static_cast<uint32_t>(versions_.size());
        versions_.push_back({pass, {}});
        return true;
    }

    const std::vector<ResourceVersion>& versions() const { return versions_; }
    operator const std::vector<ResourceVersion>&() const { return versions_; }
    size_t size() const { return versions_.size(); }
    const ResourceVersion& operator[](size_t index) const { return versions_[index]; }
    const ResourceVersion& front() const { return versions_.front(); }
    auto begin() const { return versions_.begin(); }
    auto end() const { return versions_.end(); }
    bool imported() const { return imported_; }

private:
    bool readable(uint32_t version, uint32_t pass) const
    {
        return pass != INVALID_INDEX && version < versions_.size() &&
            (imported_ || version != 0) && versions_[version].producer != pass;
    }
    std::vector<ResourceVersion> versions_{{}};
    bool imported_;
};

struct ResourceLifetime
{
    uint32_t first_use = INVALID_INDEX;
    uint32_t last_use = INVALID_INDEX;

    void record_use(uint32_t execution_index)
    {
        if (first_use == INVALID_INDEX || execution_index < first_use)
            first_use = execution_index;
        if (last_use == INVALID_INDEX || execution_index > last_use)
            last_use = execution_index;
    }
};

inline bool resource_lifetime(const std::vector<ResourceVersion>& versions,
    const std::vector<uint32_t>& execution_indices, ResourceLifetime& lifetime)
{
    ResourceLifetime compiled;
    for (const auto& version : versions)
    {
        if (version.producer != INVALID_INDEX)
        {
            if (version.producer >= execution_indices.size() ||
                execution_indices[version.producer] == INVALID_INDEX)
                return false;
            compiled.record_use(execution_indices[version.producer]);
        }
        for (uint32_t reader : version.readers)
        {
            if (reader >= execution_indices.size() || execution_indices[reader] == INVALID_INDEX)
                return false;
            compiled.record_use(execution_indices[reader]);
        }
    }
    lifetime = compiled;
    return true;
}

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
    if (edges.size() != indegree.size() || edges.size() > std::numeric_limits<uint32_t>::max())
        return false;
    std::vector<uint32_t> actual_indegree(edges.size());
    for (const auto& consumers : edges)
        for (uint32_t consumer : consumers)
        {
            if (consumer >= edges.size() || actual_indegree[consumer] == UINT32_MAX)
                return false;
            ++actual_indegree[consumer];
        }
    if (actual_indegree != indegree)
        return false;
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

// Backend-independent dependency compiler for versioned virtual resources.
// Backends may add alias hazards to the same edge set before compiling.
class ResourceDependencyGraph
{
public:
    explicit ResourceDependencyGraph(size_t passes) : edges_(passes), indegree_(passes) {}

    bool add_edge(uint32_t producer, uint32_t consumer)
    {
        if (producer == INVALID_INDEX) return consumer < edges_.size();
        if (producer >= edges_.size() || consumer >= edges_.size()) return false;
        if (producer != consumer && edges_[producer].insert(consumer).second)
            ++indegree_[consumer];
        return true;
    }

    bool add_versions(const std::vector<ResourceVersion>& versions)
    {
        // Validate before inserting, so malformed chains cannot partially
        // mutate the dependency graph.
        for (const auto& version : versions)
        {
            if (version.producer != INVALID_INDEX && version.producer >= edges_.size()) return false;
            for (uint32_t reader : version.readers)
                if (reader >= edges_.size()) return false;
        }
        for (size_t i = 1; i < versions.size(); ++i)
            if (versions[i].producer == INVALID_INDEX) return false;
        add_resource_dependencies(versions, [this](uint32_t producer, uint32_t consumer) {
            add_edge(producer, consumer);
        });
        return true;
    }

    std::vector<std::set<uint32_t>>& edges() { return edges_; }
    std::vector<uint32_t>& indegree() { return indegree_; }
    bool compile(std::vector<uint32_t>& order) const { return sort_passes(edges_, indegree_, order); }

private:
    std::vector<std::set<uint32_t>> edges_;
    std::vector<uint32_t> indegree_;
};
}
