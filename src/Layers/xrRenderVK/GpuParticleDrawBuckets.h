#pragma once
#include <cstdint>
#include <limits>

namespace xray::render::vulkan
{
// Must match GpuDrawBucket in gpu_particle_compact.cs. Material IDs are
// dense and identify the entire compatible graphics state, including shader
// variant and texture. Unsupported blend states must use another draw path.
inline bool gpu_particle_draw_bucket(uint32_t root, uint32_t root_count, uint32_t material, uint32_t material_capacity, uint32_t blend,
    uint32_t shader_variant, bool hud, uint32_t& result, bool child = false)
{
    if (root >= root_count || material >= material_capacity || !material_capacity ||
        uint64_t(root_count) * material_capacity > UINT32_MAX / 28 || blend > 5 || shader_variant > 2)
        return false;
    const uint32_t mode = shader_variant == 2 ? (hud ? 13u : 12u) : blend + (hud ? 6u : 0u);
    result = (root * material_capacity + material) * 28 + (child ? 14 : 0) + mode;
    return true;
}
}
