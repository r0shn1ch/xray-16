#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace xray::render::vulkan
{
struct DetailInteractionEntity
{
    float position[3]{}, radius{};
    float velocity[3]{}, weight{};
};
static_assert(sizeof(DetailInteractionEntity) == 32);

struct DetailInteractionParameters
{
    float origin[2]{}, previous_origin[2]{};
    int32_t shift[2]{};
    float texel_size{};
    uint32_t size{};
    float spring[4]{};
    float contact_rate{}, delta{}, max_velocity{12.f};
    uint32_t entity_count{};
    float height_origin[2]{}, height_texel_size{};
    uint32_t previous_valid{};
};
static_assert(sizeof(DetailInteractionParameters) == 80);

// Match the moving window and analytic damped spring in OpenXRay #2187.
// The production window is 1024^2 at 0.0625 metres; smaller windows also allow
// isolated tests without allocating a full game frame.
inline bool detail_interaction_parameters(DetailInteractionParameters& output, const float* camera_xz,
    const float* previous_origin, bool history_valid, float delta, float frequency, float damping,
    uint32_t entity_count, const float* height_origin, float height_texel_size,
    uint32_t size = 1024, float texel_size = .0625f)
{
    if (!camera_xz || !previous_origin || !height_origin || !size || size > 1024 || entity_count > 64 ||
        !std::isfinite(delta) || !std::isfinite(frequency) || !std::isfinite(damping) ||
        !std::isfinite(texel_size) || texel_size <= 0.f ||
        !std::isfinite(height_texel_size) || height_texel_size <= 0.f) return false;
    DetailInteractionParameters params{};
    params.size = size; params.texel_size = texel_size;
    params.previous_valid = history_valid;
    params.entity_count = entity_count;
    params.height_texel_size = height_texel_size;
    for (unsigned axis = 0; axis < 2; ++axis)
    {
        if (!std::isfinite(camera_xz[axis]) || !std::isfinite(previous_origin[axis]) ||
            !std::isfinite(height_origin[axis])) return false;
        params.height_origin[axis] = height_origin[axis];
        const double cell = std::floor((double(camera_xz[axis]) - double(size) * texel_size * .5) / texel_size);
        params.origin[axis] = static_cast<float>(cell * texel_size);
        if (!std::isfinite(params.origin[axis])) return false;
        params.previous_origin[axis] = previous_origin[axis];
        const double shift = std::round((double(params.origin[axis]) - previous_origin[axis]) / texel_size);
        if (std::abs(shift) >= size) params.previous_valid = 0;
        else params.shift[axis] = static_cast<int32_t>(shift);
    }
    params.delta = std::clamp(delta, 0.f, .1f);
    const float omega = 6.283185307179586f * (std::max)(frequency, .05f);
    if (!std::isfinite(omega)) return false;
    const float zeta = std::clamp(damping, .01f, .98f);
    const float damped = omega * std::sqrt(1.f - zeta * zeta);
    const float decay = std::exp(-zeta * omega * params.delta);
    const float cosine = std::cos(damped * params.delta), sine = std::sin(damped * params.delta);
    params.spring[0] = decay * (cosine + zeta * omega / damped * sine);
    params.spring[1] = decay * sine / damped;
    params.spring[2] = -decay * omega * (omega / damped) * sine;
    params.spring[3] = decay * (cosine - zeta * omega / damped * sine);
    params.contact_rate = 1.f - std::exp(-params.delta / .04f);
    for (float value : params.spring) if (!std::isfinite(value)) return false;
    output = params;
    return true;
}
}
