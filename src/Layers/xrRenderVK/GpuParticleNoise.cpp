#include "xrEngine/stdafx.h"
#include "GpuParticleNoise.h"
#include "xrParticles/noise.h"
#include <cstring>

namespace xray::render::vulkan
{
std::array<Fvector4, 514> gpu_particle_noise_table()
{
    int permutation[514];
    float gradients[514 * 3];
    PAPI::GetNoise3Tables(permutation, gradients);
    std::array<Fvector4, 514> result{};
    static_assert(sizeof(permutation[0]) == sizeof(result[0].w));
    for (size_t i = 0; i < result.size(); ++i)
    {
        result[i].set(gradients[3 * i], gradients[3 * i + 1], gradients[3 * i + 2], 0.f);
        std::memcpy(&result[i].w, &permutation[i], sizeof(permutation[i]));
    }
    return result;
}
}
