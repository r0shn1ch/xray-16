#include "gpu_particle_types.h"
[[vk::binding(0, 0)]] cbuffer GpuPapiParams { GpuPapiConstants g_Papi; }
[[vk::binding(1, 0)]] RWStructuredBuffer<GpuPapiEmitter> g_Emitters;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> g_ParticleAllocation;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> g_StateAllocation;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> g_Errors;
// 64 threads also fits devices with Vulkan's minimum compute limits.
[numthreads(64,1,1)]
void main(uint3 dispatchID : SV_DispatchThreadID) {
    uint index = dispatchID.x;
    if (index < g_Papi.emitterCapacity) g_Emitters[index] = (GpuPapiEmitter)0;
    if (index < g_Papi.particleCapacity) g_ParticleAllocation[index] = 0;
    if (index < g_Papi.actionStateCapacity) g_StateAllocation[index] = 0;
    if (index == 0) g_Errors[0] = 0;
}
