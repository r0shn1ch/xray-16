#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace xray::render::vulkan
{
struct DetailWindParameters
{
    // Two XZ directions and the shared cyclic phase (turns).
    float directions[4]{};
    float phase{};
    float amplitude() const
    {
        return (std::max)(std::hypot(directions[0], directions[1]), std::hypot(directions[2], directions[3]));
    }
};

struct DetailWindSettings
{
    float rotation[2]{}, amplitude[2]{}, speed{};
};

class DetailWindAnimation
{
public:
    DetailWindSettings normal{}, fast{};
    bool configured{};
    DetailWindParameters advance(float time, float strength)
    {
        if (!configured || !std::isfinite(time) || !std::isfinite(strength)) return {};
        constexpr float turn = 6.283185307179586f;
        float delta = time - previous_time_;
        if (delta < 0.f || delta > 1.f) delta = .03f;
        previous_time_ = time;
        const float factor = std::clamp(strength, 0.f, 1.f);
        const auto lerp = [factor](float a, float b) { return a + (b - a) * factor; };
        const float speed = lerp(normal.speed, fast.speed);
        if (!std::isfinite(speed)) return {};
        DetailWindParameters result;
        for (unsigned i = 0; i < 2; ++i)
        {
            const float rotation = lerp(normal.rotation[i], fast.rotation[i]);
            const float amplitude = lerp(normal.amplitude[i], fast.amplitude[i]);
            if (!std::isfinite(rotation) || rotation <= 1e-5f || !std::isfinite(amplitude)) return {};
            angles_[i] = std::fmod(angles_[i] + turn * delta / rotation, turn);
            result.directions[i * 2] = std::sin(angles_[i]) * amplitude;
            result.directions[i * 2 + 1] = std::cos(angles_[i]) * amplitude;
        }
        phase_ = std::fmod(phase_ + delta * speed / turn, 1.f);
        result.phase = phase_;
        return result;
    }
private:
    float previous_time_{}, angles_[2]{}, phase_{};
};

// Match DetailManager_VS::QC, including authored vertices below the origin.
inline float detail_wind_weight(float y, float height, uint32_t flags)
{
    if ((flags & 1u) || !(height > 1e-5f)) return 0.f; // DO_NO_WAVING
    return std::clamp(std::floor((y / height) * 16384.f), -32768.f, 32767.f) / 16384.f;
}

inline bool detail_second_wave(float x, float z)
{
    uint32_t a, b;
    // Treat +0 and -0 as the same position, on CPU and in the vertex shader.
    if (x == 0.f) x = 0.f;
    if (z == 0.f) z = 0.f;
    std::memcpy(&a, &x, 4); std::memcpy(&b, &z, 4);
    uint32_t hash = a * 0x9e3779b9u ^ b * 0x85ebca6bu;
    hash ^= hash >> 16;
    return hash % 3u == 0u;
}

inline std::array<float, 3> detail_wind_displacement(const float* p, const float* root, float weight,
    const DetailWindParameters& wind)
{
    constexpr float inverse_turn = 1.f / 6.283185307179586f;
    const bool second = detail_second_wave(root[0], root[2]);
    const float phase = wind.phase + (p[0] / (second ? 3.f : 5.f) + p[1] / 7.f +
        p[2] / (second ? 5.f : 3.f)) * inverse_turn;
    const float cyclic = 2.8284271f * (phase - std::floor(phase)) - 1.4142136f;
    const float bend = (p[1] - root[1]) * (cyclic * cyclic - 1.f) * weight;
    const unsigned direction = second ? 2 : 0;
    return {wind.directions[direction] * bend, 0.f, wind.directions[direction + 1] * bend};
}

inline float detail_wind_radius(float radius, float extent, float amplitude)
{
    // Account for the polynomial's floating-point approximation at phase 0/1.
    return radius + extent * amplitude * 1.001f;
}
}
