#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace xray::render::vulkan
{
struct TrailControlPoint
{
    float position[3]{}, direction[3]{};
    float age{}, distance{};
};
static_assert(sizeof(TrailControlPoint) == 32);

// Newest point first, as in PR #2187. A group break prevents a teleport or
// restarted emitter from connecting to the previous ribbon.
class TrailState
{
public:
    static constexpr size_t MaxPoints = 256;
    struct Group { uint32_t id{}; std::vector<TrailControlPoint> points; uint32_t color{0xffffffffu}; };
    void break_group() { ++group_; }
    bool update(float dt, const float* position, float half_width, float max_age = 3.f)
    {
        if (!std::isfinite(dt) || dt < 0 || !std::isfinite(max_age) || max_age <= 0 ||
            !std::isfinite(half_width) || half_width < 0) return false;
        if (position && (!std::isfinite(position[0]) || !std::isfinite(position[1]) || !std::isfinite(position[2])))
            return false;
        for (auto& point : points_) point.age += dt;
        points_.erase(std::remove_if(points_.begin(), points_.end(),
            [&](const Point& point) { return point.age >= max_age; }), points_.end());
        max_age_ = max_age;
        if (!position) return true;
        if (!points_.empty() && points_[0].group == group_)
        {
            const float dx = position[0] - points_[0].position[0], dy = position[1] - points_[0].position[1];
            const float dz = position[2] - points_[0].position[2];
            if (dx * dx + dy * dy + dz * dz < .025f * .025f) return true;
        }
        Point point{}; point.group = group_; std::copy_n(position, 3, point.position.begin());
        point.direction = {half_width, 0.f, 0.f};
        if (!points_.empty() && points_[0].group == group_)
        {
            const float dx = position[0] - points_[0].position[0], dz = position[2] - points_[0].position[2];
            const float length = std::sqrt(dx * dx + dz * dz);
            if (length > .0001f) point.direction = {-dz * half_width / length, 0.f, dx * half_width / length};
        }
        if (points_.size() == MaxPoints) points_.pop_back();
        points_.insert(points_.begin(), point);
        return true;
    }
    std::vector<Group> groups() const
    {
        std::vector<Group> result;
        for (const auto& point : points_)
        {
            if (result.empty() || result.back().id != point.group) result.push_back({point.group, {}});
            auto& group = result.back();
            TrailControlPoint packed{};
            std::copy(point.position.begin(), point.position.end(), packed.position);
            std::copy(point.direction.begin(), point.direction.end(), packed.direction);
            packed.age = point.age / max_age_;
            if (!group.points.empty())
            {
                const auto& previous = group.points.back();
                const float dx = packed.position[0] - previous.position[0], dy = packed.position[1] - previous.position[1];
                const float dz = packed.position[2] - previous.position[2];
                packed.distance = previous.distance + std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            group.points.push_back(packed);
        }
        return result;
    }
    size_t size() const { return points_.size(); }
    void clear() { points_.clear(); group_ = 0; max_age_ = 3.f; }
private:
    struct Point { std::array<float, 3> position{}, direction{}; float age{}; uint32_t group{}; };
    std::vector<Point> points_;
    uint32_t group_{}; float max_age_{3.f};
};
}
