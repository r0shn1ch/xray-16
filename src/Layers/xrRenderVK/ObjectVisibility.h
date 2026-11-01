#pragma once
#include <array>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <algorithm>
namespace xray::render::vulkan
{
inline constexpr float SkyDirections[26][3] = {
    {-0.26287, 0.52573, 0.80902},  {0.27639, 0.44721, 0.85065},   {-0.95106, 0.00000, 0.30902},
    {-0.95106, 0.00000, -0.30902}, {0.58779, 0.00000, -0.80902},  {0.58779, 0.00000, 0.80902},

    {-0.00000, 0.00000, 1.00000},  {0.52573, 0.85065, 0.00000},   {-0.26287, 0.52573, -0.80902},
    {-0.42533, 0.85065, 0.30902},  {0.95106, 0.00000, 0.30902},   {0.95106, 0.00000, -0.30902},

    {0.00000, 1.00000, 0.00000},   {-0.58779, 0.00000, 0.80902},  {-0.72361, 0.44721, 0.52573},
    {-0.72361, 0.44721, -0.52573}, {-0.58779, 0.00000, -0.80902}, {0.16246, 0.85065, -0.50000},

    {0.89443, 0.44721, 0.00000},   {-0.85065, 0.52573, -0.00000}, {0.16246, 0.85065, 0.50000},
    {0.68819, 0.52573, -0.50000},  {0.27639, 0.44721, -0.85065},  {0.00000, 0.00000, -1.00000},

    {-0.42533, 0.85065, -0.30902}, {0.68819, 0.52573, 0.50000},
};
class ObjectVisibility
{
    struct Entry
    {
        std::array<float, 3> target{};
        uint32_t frame{};
        float value{1};
    };
    std::unordered_map<uint64_t, Entry> local_;
    std::array<float, 3> position_{};
    std::array<float, 26> sky_{};
    unsigned sky_index_{};
    bool positioned_{};

  public:
    ObjectVisibility()
    {
        sky_.fill(.5f);
    }
    void position(const std::array<float, 3> &p)
    {
        float distance = 0;
        for (unsigned i = 0; i < 3; ++i)
            distance += (p[i] - position_[i]) * (p[i] - position_[i]);
        if (!positioned_ || distance > .01f)
        {
            position_ = p;
            positioned_ = true;
            for (auto &e : local_)
                e.second.frame = 0;
        }
    }
    template <class Trace>
    float light(uint64_t id, const std::array<float, 3> &target, uint32_t frame, unsigned &budget, Trace trace)
    {
        auto &e = local_[id];
        float moved = 0;
        for (unsigned i = 0; i < 3; ++i)
            moved += std::abs(e.target[i] - target[i]);
        if (budget && (!e.frame || frame - e.frame >= 8 || moved > .1f))
        {
            --budget;
            e.value = trace(position_, target) ? 0.f : 1.f;
            e.frame = frame ? frame : 1;
            e.target = target;
        }
        const float value = e.value;
        if (local_.size() > 256)
            for (auto it = local_.begin(); it != local_.end();)
                if (frame - it->second.frame > 60)
                    it = local_.erase(it);
                else
                    ++it;
        return value;
    }
    template <class Trace> void sky(unsigned &budget, Trace trace)
    {
        if (!budget)
            return;
        --budget;
        std::array<float, 3> end{};
        for (unsigned i = 0; i < 3; ++i)
            end[i] = position_[i] + SkyDirections[sky_index_][i] * 50.f;
        sky_[sky_index_] = trace(position_, end) ? 0.f : 1.f;
        sky_index_ = (sky_index_ + 1) % sky_.size();
    }
    std::array<float, 6> faces(float scale) const
    {
        std::array<float, 6> result{};
        for (unsigned s = 0; s < sky_.size(); ++s)
            for (unsigned axis = 0; axis < 3; ++axis)
                result[axis + (SkyDirections[s][axis] < 0 ? 3 : 0)] +=
                    std::abs(SkyDirections[s][axis]) * sky_[s] * scale;
        for (float &f : result)
            f = std::clamp(f, 1.f / 255.f, 1.f);
        return result;
    }
};
} // namespace xray::render::vulkan
