#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace xray::render::framegraph
{
// Shared layout for baking and sampling a level's regular sky visibility grid.
struct SkyProbeLayout
{
    std::array<float, 3> origin{};
    std::array<uint32_t, 3> dimensions{};
    float spacing{};
    uint32_t count{};

    bool initialize(const std::array<float, 3>& minimum, const std::array<float, 3>& maximum,
        float requested_spacing, uint32_t probe_budget)
    {
        *this = {};
        if (!std::isfinite(requested_spacing) || requested_spacing <= 0.f || probe_budget < 8)
            return false;
        std::array<double, 3> spans{};
        for (unsigned axis = 0; axis < spans.size(); ++axis)
        {
            if (!std::isfinite(minimum[axis]) || !std::isfinite(maximum[axis]) || maximum[axis] < minimum[axis])
                return false;
            spans[axis] = double(maximum[axis]) - minimum[axis];
        }

        // Increase spacing uniformly until all three axes fit the same budget.
        double step = requested_spacing;
        std::array<uint32_t, 3> dims{};
        uint32_t total = 0;
        for (;; step *= 2.)
        {
            if (step > std::numeric_limits<float>::max())
                return false;
            total = 1;
            for (unsigned axis = 0; axis < spans.size(); ++axis)
            {
                const double samples = std::max(2., std::ceil(spans[axis] / step) + 1.);
                if (samples > probe_budget / total)
                {
                    total = 0;
                    break;
                }
                dims[axis] = static_cast<uint32_t>(samples);
                total *= dims[axis];
            }
            if (total)
                break;
        }
        origin = minimum;
        for (unsigned axis = 0; axis < spans.size(); ++axis)
            if (double(minimum[axis]) + step * (dims[axis] - 1) > std::numeric_limits<float>::max())
            {
                *this = {};
                return false;
            }
        dimensions = dims;
        spacing = static_cast<float>(step);
        count = total;
        return true;
    }

    bool position(uint32_t index, std::array<float, 3>& result) const
    {
        if (index >= count)
            return false;
        for (unsigned axis = 0; axis < dimensions.size(); ++axis)
        {
            result[axis] = static_cast<float>(double(origin[axis]) + double(spacing) * (index % dimensions[axis]));
            index /= dimensions[axis];
        }
        return true;
    }
};
}
