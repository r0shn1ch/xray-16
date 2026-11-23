#pragma once
#include "TrailState.h"
#include <unordered_map>

namespace xray::render::vulkan
{
// A projectile owns its ribbon even when the gameplay bullet array is moved.
class TrailManager
{
public:
    void advance(uint32_t frame, float dt)
    {
        if (has_frame_ && frame == frame_) return;
        has_frame_ = true; frame_ = frame;
        for (auto it = trails_.begin(); it != trails_.end();)
        {
            it->second.state.update(dt, nullptr, 0.f);
            if (!it->second.state.size()) it = trails_.erase(it); else ++it;
        }
    }
    bool observe(uint64_t id, const float* start, const float* end, float half_width, uint32_t frame, float dt, uint32_t color = 0xffffffffu)
    {
        if (!start || !end || !std::isfinite(half_width) || half_width < 0.f)
            return false;
        advance(frame, dt);
        auto found = trails_.find(id);
        if (found == trails_.end())
        {
            if (trails_.size() >= 256) return false;
            found = trails_.emplace(id, Entry{}).first;
            if (!found->second.state.update(0.f, start, half_width))
            {
                trails_.erase(found);
                return false;
            }
        }
        else if (found->second.last_frame != frame && found->second.last_frame + 1 != frame)
        {
            found->second.state.break_group();
            if (!found->second.state.update(0.f, start, half_width))
                return false;
        }
        found->second.last_frame = frame;
        found->second.color = color;
        return found->second.state.update(0.f, end, half_width);
    }
    std::vector<TrailState::Group> groups() const
    {
        std::vector<const Entry*> ordered;
        for (const auto& item : trails_) ordered.push_back(&item.second);
        std::sort(ordered.begin(), ordered.end(), [](const Entry* a, const Entry* b) { return a->last_frame > b->last_frame; });
        std::vector<TrailState::Group> result;
        size_t remaining = TrailState::MaxPoints;
        for (const auto* entry : ordered)
            for (auto& group : entry->state.groups())
            {
                if (remaining < 2 || result.size() == 128) return result;
                if (group.points.size() < 2) continue;
                if (group.points.size() > remaining) group.points.resize(remaining);
                group.color = entry->color;
                remaining -= group.points.size();
                result.push_back(std::move(group));
            }
        return result;
    }
    void clear() { trails_.clear(); has_frame_ = false; }
private:
    struct Entry { TrailState state; uint32_t last_frame{}, color{0xffffffffu}; };
    std::unordered_map<uint64_t, Entry> trails_;
    uint32_t frame_{}; bool has_frame_{};
};
}
