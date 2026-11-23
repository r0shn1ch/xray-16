#pragma once
#include "psystem.h"

namespace PAPI
{
// 514 permutation entries and 514 packed float3 gradients, matching noise3.
PARTICLES_API void GetNoise3Tables(int* permutation, float* gradients);
}

float noise3(const Fvector& vec);
float fractalsum3(const Fvector& v, float freq, int octaves);
float turbulence3(const Fvector& v, float freq, int octaves);
