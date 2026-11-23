#pragma once
#include "xrCore/_vector4.h"
#include <array>

namespace xray::render::vulkan
{
// Bits of w contain the CPU PAPI permutation; xyz are its gradients.
std::array<Fvector4, 514> gpu_particle_noise_table();
}
