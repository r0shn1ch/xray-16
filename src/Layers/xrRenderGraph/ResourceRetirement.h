#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
namespace xray::render::framegraph
{
// A resource is released only after every flight slot that could reference it
// has completed its fence. Image acquisition alone is not sufficient.
class ResourceRetirement
{
    struct Entry
    {
        uint32_t pending;
        std::shared_ptr<void> resource;
    };
    std::vector<Entry> entries_;
    uint32_t slots_;

  public:
    explicit ResourceRetirement(uint32_t slots) : slots_(std::min(slots, 32u))
    {
    }
    ResourceRetirement(const ResourceRetirement&) = delete;
    ResourceRetirement& operator=(const ResourceRetirement&) = delete;
    template <class T> void retire(std::unique_ptr<T> resource)
    {
        if (resource && slots_)
            entries_.push_back({all_slots(), std::shared_ptr<T>(std::move(resource))});
    }
    template <class Function> void defer(Function function)
    {
        defer_for_slots(all_slots(), std::move(function));
    }
    template <class Function> void defer_for_slots(uint32_t pending, Function function)
    {
        pending &= all_slots();
        if (!pending)
        {
            function();
            return;
        }
        struct Action
        {
            Function function;
            explicit Action(Function f) : function(std::move(f)) {}
            ~Action() { function(); }
        };
        entries_.push_back({pending, std::make_shared<Action>(std::move(function))});
    }
    void complete(uint32_t slot)
    {
        if (slot >= slots_)
            return;
        const uint32_t completed_slot = 1u << slot;
        std::vector<std::shared_ptr<void>> completed;
        // Allocate before changing any pending masks. Release callbacks can
        // schedule new work, so run them after removing the completed entries.
        completed.reserve(std::count_if(entries_.begin(), entries_.end(),
            [completed_slot](const Entry& entry) { return entry.pending == completed_slot; }));
        for (auto& entry : entries_)
        {
            entry.pending &= ~completed_slot;
            if (!entry.pending)
                completed.push_back(std::move(entry.resource));
        }
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
            [](const Entry& entry) { return !entry.pending; }), entries_.end());
    }
    void drain()
    {
        // Destructors can retire additional resources. The caller has waited
        // for every relevant queue, so drain those follow-up entries too.
        while (!entries_.empty())
        {
            std::vector<Entry> pending;
            pending.swap(entries_);
        }
    }
    size_t size() const
    {
        return entries_.size();
    }
private:
    uint32_t all_slots() const { return slots_ == 32 ? UINT32_MAX : (1u << slots_) - 1; }
};
} // namespace xray::render::framegraph
