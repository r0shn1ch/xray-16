#pragma once
#include "SmokeTrailDrawPass.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace xray::render::vulkan
{
// Frame preparation is transactional: simulation state advances only after
// the command buffer is submitted, so an aborted frame can be retried.
class SmokeTrailController
{
public:
    struct Frame
    {
        SmokeTrailEmitParams emit{}; SmokeTrailSimParams simulation{};
        SmokeTrailCompactParams compact{}; SmokeTrailDrawParams draw{};
        std::array<float, 3> position{}; float accumulator{};
        uint32_t sample_frame{}, seed{}; bool active{}, restart_pending{};
    };
    void update(const float* position, const float* direction, float dt, uint32_t frame)
    {
        if (!position || !direction || !std::isfinite(dt) || dt < 0.f) return;
        for (unsigned i = 0; i < 3; ++i)
            if (!std::isfinite(position[i]) || !std::isfinite(direction[i])) return;
        std::copy_n(position, 3, position_.begin()); std::copy_n(direction, 3, direction_.begin());
        sample_dt_ = std::min(dt, .1f); sample_frame_ = frame; has_sample_ = true;
    }
    Frame prepare(uint32_t frame, float dt, float time) const
    {
        Frame out{};
        out.active = has_sample_ && uint32_t(frame - sample_frame_) <= 1 && (!has_committed_ || sample_frame_ != committed_frame_);
        out.position = position_; out.sample_frame = sample_frame_; out.seed = seed_ * 1664525u + 1013904223u;
        const float frame_dt = std::isfinite(dt) ? std::clamp(dt, 0.f, .1f) : 0.f;
        const bool continuous = has_previous_ && uint32_t(sample_frame_ - committed_frame_) <= 2;
        out.restart_pending = restart_pending_ || (out.active && has_committed_ && !continuous);
        const auto previous = continuous ? previous_ : position_;
        out.accumulator = accumulator_;
        if (out.active) out.accumulator += 45.f * sample_dt_;
        const uint32_t count = std::min<uint32_t>(static_cast<uint32_t>(out.accumulator), SmokeTrailCompute::MaxPoints);
        out.accumulator -= count;
        auto& e = out.emit;
        e.previous_x = previous[0]; e.previous_y = previous[1]; e.previous_z = previous[2];
        e.current_x = position_[0]; e.current_y = position_[1]; e.current_z = position_[2];
        e.direction_x = direction_[0]; e.direction_y = direction_[1]; e.direction_z = direction_[2];
        e.lifetime = 2.5f; e.max_width = .04f; e.count = out.active ? count : 0;
        e.max_points = SmokeTrailCompute::MaxPoints; e.seed = float(out.seed) / 4294967296.f;
        e.restart = out.restart_pending && e.count != 0;
        out.restart_pending = out.restart_pending && !e.restart;
        auto& s = out.simulation;
        s.dt = frame_dt; s.gravity = .15f; s.buoyancy = 1.f; s.turbulence = .8f; s.drag = .7f;
        s.time = time; s.max_points = SmokeTrailCompute::MaxPoints; s.heat = 1.f;
        auto& c = out.compact;
        c.max_points = SmokeTrailCompute::MaxPoints; c.subdivisions = 12; c.max_width = .04f;
        c.turbulence = .8f; c.frequency = 2.f; c.evolution = time * .5f; c.sphere_radius = 3.f;
        c.center_x = position_[0]; c.center_y = position_[1]; c.center_z = position_[2];
        auto& d = out.draw;
        d.subdivisions = 12; d.texcoord_factor = 1.f; d.tail_fade = 1; d.edge_policy = 1;
        d.turbulence = c.turbulence; d.frequency = c.frequency; d.evolution = c.evolution;
        d.center_x = c.center_x; d.center_y = c.center_y; d.center_z = c.center_z; d.sphere_radius = c.sphere_radius;
        return out;
    }
    void commit(const Frame& frame)
    {
        accumulator_ = frame.accumulator; seed_ = frame.seed; restart_pending_ = frame.restart_pending;
        if (frame.active) { previous_ = frame.position; committed_frame_ = frame.sample_frame; has_previous_ = has_committed_ = true; }
        else has_previous_ = false;
    }
    void reset() { *this = {}; }
private:
    std::array<float, 3> position_{}, direction_{}, previous_{};
    float sample_dt_{}, accumulator_{};
    uint32_t sample_frame_{}, committed_frame_{}, seed_{};
    bool has_sample_{}, has_previous_{}, has_committed_{}, restart_pending_{};
};
}
