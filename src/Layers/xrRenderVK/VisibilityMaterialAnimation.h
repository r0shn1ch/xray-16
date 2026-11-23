#pragma once

#include "ShaderMaterialLibrary.h"
#include "Layers/xrRenderGraph/VisibilityId.h"
#include <cmath>
#include <vector>

namespace xray::render::vulkan
{
struct VisibilityMatrixRange
{
    uint32_t first, count;
    TextureMatrixSet matrices;
};

inline bool update_visibility_material_matrices(std::vector<visibility::DrawEntry>& entries,
    const std::vector<VisibilityMatrixRange>& ranges, float time, const float* view)
{
    if (!std::isfinite(time)) return false;
    for (const auto& range : ranges)
        if (range.first > entries.size() || range.count > entries.size() - range.first) return false;
    for (const auto& range : ranges)
    {
        std::array<std::array<float, 4>, 3> base{{{{1, 0, 0, 0}}, {{0, 1, 0, 0}}, {{0, 0, 1, 0}}}};
        auto detail = base;
        auto environment = base;
        if (range.matrices.enabled[0]) base = range.matrices.values[0].evaluate(time, view);
        if (range.matrices.enabled[1]) detail = range.matrices.values[1].evaluate(time, view);
        if (range.matrices.enabled[2]) environment = range.matrices.values[2].evaluate(time, view);
        for (const auto* matrix : {&base, &detail, &environment})
            for (size_t row = 0; row < 3; ++row)
                for (size_t column : {size_t(0), size_t(1), size_t(2), size_t(3)})
                    if (!std::isfinite((*matrix)[row][column])) return false;
        for (uint32_t i = 0; i < range.count; ++i)
        {
            auto& entry = entries[range.first + i];
            entry.base_u_x = base[0][0]; entry.base_u_y = base[0][1]; entry.base_u_offset = base[0][3];
            entry.base_v_x = base[1][0]; entry.base_v_y = base[1][1]; entry.base_v_offset = base[1][3];
            entry.detail_u_x = detail[0][0]; entry.detail_u_y = detail[0][1]; entry.detail_u_offset = detail[0][3];
            entry.detail_v_x = detail[1][0]; entry.detail_v_y = detail[1][1]; entry.detail_v_offset = detail[1][3];
            entry.environment_xx = environment[0][0]; entry.environment_xy = environment[0][1]; entry.environment_xz = environment[0][2];
            entry.environment_yx = environment[1][0]; entry.environment_yy = environment[1][1]; entry.environment_yz = environment[1][2];
            entry.environment_zx = environment[2][0]; entry.environment_zy = environment[2][1]; entry.environment_zz = environment[2][2];
        }
    }
    return true;
}
}
