#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace xray::render::framegraph
{
template <typename Descriptor, typename Resource>
class ResourcePool
{
public:
    template <typename Compatible, typename Create>
    Resource* acquire(const Descriptor& descriptor, Compatible&& compatible, Create&& create)
    {
        Entry* fallback = nullptr;
        for (auto& entry : entries_)
        {
            if (entry.in_use)
                continue;
            if (entry.descriptor == descriptor)
            {
                entry.in_use = true;
                return entry.resource.get();
            }
            if (!fallback && compatible(entry.descriptor, descriptor))
                fallback = &entry;
        }
        if (fallback)
        {
            fallback->in_use = true;
            return fallback->resource.get();
        }

        auto resource = create();
        if (!resource)
            return nullptr;
        entries_.push_back({descriptor, std::move(resource), frame_, true});
        return entries_.back().resource.get();
    }

    // The backend must establish GPU completion for every queue using the resource.
    bool release(Resource* resource)
    {
        for (auto& entry : entries_)
        {
            if (entry.resource.get() == resource && entry.in_use)
            {
                entry.in_use = false;
                entry.last_used = frame_;
                return true;
            }
        }
        return false;
    }

    void tick()
    {
        ++frame_;
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [this](const Entry& entry) {
            return !entry.in_use && frame_ - entry.last_used >= EvictAfterFrames;
        }), entries_.end());
    }

    void reset()
    {
        entries_.clear();
        frame_ = 0;
    }

private:
    struct Entry
    {
        Descriptor descriptor;
        std::unique_ptr<Resource> resource;
        uint64_t last_used;
        bool in_use;
    };
    static constexpr uint64_t EvictAfterFrames = 4;
    std::vector<Entry> entries_;
    uint64_t frame_{};
};
}
