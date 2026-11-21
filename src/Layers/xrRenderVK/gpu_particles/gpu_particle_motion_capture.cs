#include "gpu_particle_types.h"

[[vk::binding(0, 0)]] cbuffer GpuPapiParams { GpuPapiConstants g_Papi; }
[[vk::binding(1, 0)]] RWStructuredBuffer<GpuPapiParticle> g_Particles;
[[vk::binding(2, 0)]] StructuredBuffer<GpuPapiEmitter> g_Emitters;

[numthreads(64, 1, 1)]
void main(uint3 dispatchID : SV_DispatchThreadID)
{
    uint index = dispatchID.x;
    if (index >= g_Papi.particleCapacity) return;
    GpuPapiParticle particle = g_Particles[index];
    if (particle.emitter >= g_Papi.emitterCapacity) return;
    GpuPapiEmitter emitter = g_Emitters[particle.emitter];
    if (emitter.active != 1 || emitter.particleFirst == GPU_PAPI_INVALID ||
        index < emitter.particleFirst || index - emitter.particleFirst >= emitter.count) return;
    float3 worldPosition = particle.position;
    if ((emitter.flags & GPU_PAPI_LOCAL) != 0u)
        worldPosition = emitter.origin.xyz + emitter.basisX.xyz * particle.position.x +
            emitter.basisY.xyz * particle.position.y + emitter.basisZ.xyz * particle.position.z;
    particle.renderPreviousWorldPosition = worldPosition;
    particle.renderMotionValid = 1u;
    g_Particles[index] = particle;
}
