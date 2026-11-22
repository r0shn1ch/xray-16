#pragma once
#include "GrassInteractionTypes.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <mutex>

namespace xray::grass
{
// The update thread stages samples, then atomically publishes a complete
// snapshot. Readers retain the preceding snapshot during an unfinished update.
class State
{
public:
    static constexpr size_t Capacity = 256;
    bool begin(uint32_t frame, uint32_t time_ms, const float* focus)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        pending_count_ = 0;
        frame_ = frame; time_ms_ = time_ms; open_ = true;
        valid_ = focus != nullptr;
        for (unsigned axis=0; axis<3 && valid_; ++axis)
        {
            valid_ = std::isfinite(focus[axis]);
            focus_[axis] = focus[axis];
        }
        return valid_;
    }
    bool submit(const Sample& sample)
    {
        if (!sample.instance || !std::isfinite(sample.radius) || sample.radius < 0.f ||
            !std::isfinite(sample.mass) || sample.mass < 0.f)
            return false;
        for (float value : sample.position) if (!std::isfinite(value)) return false;
        std::lock_guard<std::mutex> guard(mutex_);
        if (!open_ || !valid_) return false;
        Entry entry{sample};
        for (unsigned axis=0; axis<3; ++axis)
        {
            const double distance = double(sample.position[axis])-focus_[axis];
            entry.distance += distance*distance;
        }
        for (size_t i=0; i<pending_count_; ++i)
            if (pending_[i].sample.instance == sample.instance)
            { pending_[i] = entry; return true; }
        if (pending_count_ < Capacity) pending_[pending_count_++] = entry;
        else
        {
            const auto furthest = std::max_element(pending_.begin(),pending_.end(),less);
            if (!less(entry,*furthest)) return false;
            *furthest = entry;
        }
        return true;
    }
    void publish()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!open_) return;
        std::sort(pending_.begin(),pending_.begin()+pending_count_,less);
        const uint32_t elapsed = time_ms_-published_.time_ms;
        const bool continuous = frame_-published_.frame == 1u && elapsed > 0u && elapsed <= 250u;
        Frame next;
        next.frame = frame_; next.time_ms = time_ms_; next.revision = ++revision_;
        next.entities.reserve(pending_count_);
        for (size_t i=0; i<pending_count_; ++i)
        {
            const auto& sample = pending_[i].sample;
            Entity entity;
            std::copy_n(sample.position,3,entity.position);
            entity.radius = std::clamp(sample.radius,.2f,3.f);
            entity.weight = sample.mass > 0.f ? std::clamp(sample.mass/100.f,.3f,2.f) : 1.f;
            if (continuous)
                for (size_t previous=0; previous<previous_count_; ++previous)
                {
                    const auto& old = previous_[previous].sample;
                    if (old.instance != sample.instance || old.object_id != sample.object_id) continue;
                    double distance = 0;
                    for (unsigned axis=0; axis<3; ++axis)
                    {
                        const double delta = double(sample.position[axis])-old.position[axis];
                        distance += delta*delta;
                    }
                    // Teleports and pooled identities must not manufacture a
                    // movement direction for the contact solver.
                    const double maximum_step = (std::max)(1.0,double(elapsed)*.1); // 100 metres/second
                    if (distance <= maximum_step*maximum_step)
                        for (unsigned axis=0; axis<3; ++axis)
                            entity.velocity[axis] = (sample.position[axis]-old.position[axis]) * (1000.f/elapsed);
                    break;
                }
            next.entities.push_back(entity);
        }
        published_ = std::move(next);
        std::copy_n(pending_.begin(),pending_count_,previous_.begin());
        previous_count_ = pending_count_;
        pending_count_ = 0; open_ = false;
    }
    Frame snapshot() const
    {
        std::lock_guard<std::mutex> guard(mutex_);
        return published_;
    }
    void forget(const void* instance)
    {
        std::lock_guard<std::mutex> guard(mutex_);
        for (size_t i=0; i<pending_count_;)
            if (pending_[i].sample.instance == instance) pending_[i] = pending_[--pending_count_];
            else ++i;
        for (size_t i=0; i<previous_count_;)
            if (previous_[i].sample.instance == instance)
            {
                std::move(previous_.begin()+i+1,previous_.begin()+previous_count_,previous_.begin()+i);
                --previous_count_;
                published_.entities.erase(published_.entities.begin()+i);
                published_.revision = ++revision_;
            }
            else ++i;
    }
    void reset()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        pending_count_ = previous_count_ = 0; open_ = false;
        published_ = {}; published_.revision = ++revision_;
    }
private:
    struct Entry { Sample sample; double distance{}; };
    static bool less(const Entry& a, const Entry& b)
    {
        if (a.distance != b.distance) return a.distance < b.distance;
        if (a.sample.object_id != b.sample.object_id) return a.sample.object_id < b.sample.object_id;
        return std::less<const void*>{}(a.sample.instance,b.sample.instance);
    }
    mutable std::mutex mutex_;
    std::array<Entry,Capacity> pending_{}, previous_{};
    size_t pending_count_{}, previous_count_{};
    float focus_[3]{};
    uint32_t frame_{}, time_ms_{};
    uint64_t revision_{};
    bool open_{}, valid_{};
    Frame published_;
};
}
