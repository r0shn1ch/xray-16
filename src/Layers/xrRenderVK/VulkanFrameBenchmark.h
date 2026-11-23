#pragma once

#include "VulkanFrameMetrics.h"
#include <iomanip>
#include <string_view>

namespace xray::render::vulkan
{
// A bounded, contiguous game interval. Session metrics intentionally remain
// separate: loading, menus and lifecycle gaps are useful diagnostics, but are
// not a reproducible gameplay benchmark.
class VulkanFrameBenchmark
{
public:
    static constexpr uint32_t MaximumFrames = 36000;
    static bool parse_frame_count(std::string_view text, uint32_t& frames)
    {
        uint32_t value = 0;
        for (char digit : text)
        {
            if (digit < '0' || digit > '9')
                return false;
            value = value * 10 + static_cast<uint32_t>(digit - '0');
            if (value > MaximumFrames)
                return false;
        }
        if (!value)
            return false;
        frames = value;
        return true;
    }

    bool arm(uint32_t frames, uint32_t gpu_latency_frames)
    {
        if (active() || !frames || frames > MaximumFrames)
            return false;
        metrics_ = {};
        requested_ = frames;
        gpu_latency_ = gpu_latency_frames;
        state_ = State::Armed;
        reason_.clear();
        return true;
    }

    bool active() const { return state_ == State::Armed || state_ == State::Recording; }
    bool recording() const { return state_ == State::Recording; }
    bool complete() const { return state_ == State::Complete; }
    uint32_t requested_frames() const { return requested_; }
    const VulkanFrameMetrics& metrics() const { return metrics_; }

    // now_ms must be a monotonic wall clock, independent of game time_factor.
    // GPU results describe submissions gpu_latency_ frames earlier. Omit the
    // first results so work submitted before the interval cannot contaminate
    // its GPU distribution; the final pending results are not awaited.
    bool frame(double now_ms, VulkanFrameSample sample, bool eligible, bool presented)
    {
        if (!active())
            return false;
        if (!eligible || !presented)
            return recording() ? interrupt(!eligible ? "gameplay interrupted" : "frame not presented") : false;
        if (state_ == State::Armed)
        {
            previous_ms_ = now_ms;
            state_ = State::Recording;
            return false; // Prime the interval at a completed game frame.
        }
        sample.frame_ms = now_ms - previous_ms_;
        previous_ms_ = now_ms;
        if (!std::isfinite(sample.frame_ms) || sample.frame_ms <= 0. ||
            !std::isfinite(sample.cpu_ms) || sample.cpu_ms < 0.)
            return interrupt("invalid frame timing");
        if (metrics_.frame_count() < gpu_latency_)
            sample.gpu_ms.reset();
        metrics_.record(sample);
        if (metrics_.frame_count() == requested_)
        {
            state_ = State::Complete;
            return true;
        }
        return false;
    }

    bool interrupt(const char* reason)
    {
        if (!active())
            return false;
        state_ = State::Interrupted;
        reason_ = reason;
        return true;
    }

    bool write_json(std::ostream& out) const
    {
        out << "{\n  \"schema_version\": 1,\n  \"complete\": " << (complete() ? "true" : "false")
            << ",\n  \"requested_frames\": " << requested_
            << ",\n  \"gpu_latency_frames\": " << gpu_latency_
            << ",\n  \"interruption\": " << std::quoted(reason_) << ",\n  \"metrics\": ";
        if (!metrics_.write_json(out))
            return false;
        out << "}\n";
        return out.good();
    }

private:
    enum class State { Idle, Armed, Recording, Complete, Interrupted };
    State state_{State::Idle};
    uint32_t requested_{}, gpu_latency_{};
    double previous_ms_{};
    std::string reason_;
    VulkanFrameMetrics metrics_;
};
}
