#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <unordered_map>
#include <utility>

namespace xray::render::vulkan
{
// CPU-side previous-MVP history for render instances. Only submitted world
// frames advance it; a missed or failed frame leaves the last visible pose.
class ObjectMotionHistory
{
public:
    enum class Kind : uint8_t { Model, LevelVisual };
    struct Key
    {
        Kind kind{};
        const void* instance{};
        const void* resource{};
        uint32_t visual{};

        bool operator==(const Key& other) const
        {
            return kind == other.kind && instance == other.instance &&
                resource == other.resource && visual == other.visual;
        }
    };
    using Matrix = std::array<float, 16>;

    void begin_frame() { pending_.clear(); }
    bool stage(Key key, const Matrix& current, Matrix& previous,
        const Matrix* world = nullptr, bool* moved = nullptr)
    {
        if (!key.instance || !valid(current)) return false;
        const auto found = submitted_.find(key);
        if (found != submitted_.end()) previous = found->second.mvp;
        if (moved) *moved = found != submitted_.end() &&
            (!world || found->second.world != *world);
        // A visual may be submitted through more than one sector. Keep its
        // first transform for the next frame and never use this frame's pose
        // as its own previous pose.
        pending_.emplace(key, Entry{current, world ? *world : Matrix{}});
        return found != submitted_.end();
    }
    void commit_frame() { submitted_ = std::move(pending_); pending_.clear(); }
    void discard_frame() { pending_.clear(); }
    void reset() { pending_.clear(); submitted_.clear(); }
    void forget_instance(const void* instance)
    {
        if (!instance) return;
        for (auto* entries : {&pending_, &submitted_})
            for (auto it = entries->begin(); it != entries->end();)
                it = it->first.instance == instance ? entries->erase(it) : std::next(it);
    }

private:
    struct Entry { Matrix mvp{}, world{}; };
    struct Hash
    {
        size_t operator()(const Key& key) const
        {
            size_t value = std::hash<const void*>{}(key.instance);
            value ^= std::hash<const void*>{}(key.resource) + size_t{0x9e3779b9} +
                (value << 6) + (value >> 2);
            value ^= std::hash<uint32_t>{}(key.visual) + size_t{0x9e3779b9} +
                (value << 6) + (value >> 2);
            return value ^ static_cast<size_t>(key.kind);
        }
    };
    static bool valid(const Matrix& matrix)
    {
        for (float value : matrix)
            if (!std::isfinite(value)) return false;
        return true;
    }
    std::unordered_map<Key, Entry, Hash> submitted_;
    std::unordered_map<Key, Entry, Hash> pending_;
};
}
