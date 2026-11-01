#pragma once
#include <cstdint>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
namespace xray::render::vulkan
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
    explicit ResourceRetirement(uint32_t slots) : slots_(slots < 32 ? slots : 31)
    {
    }
    template <class T> void retire(std::unique_ptr<T> resource)
    {
        if (resource)
            entries_.push_back({(1u << slots_) - 1, std::shared_ptr<T>(std::move(resource))});
    }
    template <class Function> void defer(Function function)
    {
        struct Action
        {
            Function function;
            explicit Action(Function f) : function(std::move(f))
            {
            }
            ~Action()
            {
                function();
            }
        };
        retire(std::make_unique<Action>(std::move(function)));
    }
    void complete(uint32_t slot)
    {
        if (slot >= slots_)
            return;
        for (auto &e : entries_)
            e.pending &= ~(1u << slot);
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [](const auto &e) { return !e.pending; }),
                       entries_.end());
    }
    void drain()
    {
        entries_.clear();
    }
    size_t size() const
    {
        return entries_.size();
    }
};
} // namespace xray::render::vulkan
