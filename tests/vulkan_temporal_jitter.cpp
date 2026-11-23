#include "src/Layers/xrRenderVK/TemporalJitter.h"
#include <array>
#include <cassert>
#include <cmath>
#include <set>

int main()
{
    using namespace xray::render::vulkan;
    const VkExtent2D extent{1280, 720};
    std::set<std::pair<float, float>> samples;
    for (uint32_t frame = 0; frame < 8; ++frame)
    {
        const auto ndc = temporal_jitter_ndc(frame, extent);
        assert(std::abs(ndc[0] * extent.width / 2.f) <= .5f);
        assert(std::abs(ndc[1] * extent.height / 2.f) <= .5f);
        samples.emplace(ndc[0], ndc[1]);
        assert(temporal_jitter_ndc(frame + 8, extent) == ndc);
    }
    assert(samples.size() == 8);
    assert((temporal_jitter_ndc(0, {0, 0}) == std::array<float, 2>{0.f, 0.f}));
    std::array<float, 16> matrix{};
    matrix[0] = matrix[5] = matrix[10] = 1.f;
    matrix[11] = 1.f; // clip.w = input.z
    const auto before = matrix;
    const auto offset = temporal_jitter_ndc(3, extent);
    apply_temporal_jitter(matrix.data(), offset);
    const std::array<float, 4> point{.3f, -.2f, 2.f, 1.f};
    const auto clip = [&](const std::array<float, 16>& m, uint32_t row) {
        float result = 0.f;
        for (uint32_t column = 0; column < 4; ++column)
            result += m[column * 4 + row] * point[column];
        return result;
    };
    assert(std::abs(clip(matrix, 0) - clip(before, 0) - offset[0] * clip(before, 3)) < 1e-6f);
    assert(std::abs(clip(matrix, 1) - clip(before, 1) - offset[1] * clip(before, 3)) < 1e-6f);
    assert(clip(matrix, 2) == clip(before, 2) && clip(matrix, 3) == clip(before, 3));
}
