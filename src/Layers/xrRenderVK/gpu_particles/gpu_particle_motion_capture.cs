#include "gpu_particle_types.h"

[[vk::binding(0, 0)]] cbuffer GpuPapiParams { GpuPapiConstants g_Papi; }
[[vk::binding(1, 0)]] RWStructuredBuffer<GpuPapiParticle> g_Particles;
[[vk::binding(2, 0)]] StructuredBuffer<GpuPapiEmitter> g_Emitters;

[numthreads(64, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint lane : SV_GroupIndex)
{
    uint stride = min(g_Papi.emitterCapacity, 65535u);
    for (uint emitterIndex = groupID.x; emitterIndex < g_Papi.emitterCapacity; emitterIndex += stride)
    {
        GpuPapiEmitter emitter = g_Emitters[emitterIndex];
        if (emitter.active != 1 || emitter.particleFirst >= g_Papi.particleCapacity) continue;
        uint count = min(emitter.count, g_Papi.particleCapacity - emitter.particleFirst);
        for (uint i = lane; i < count; i += 64u)
        {
            uint index = emitter.particleFirst + i;
            GpuPapiParticle particle = g_Particles[index];
            float3 worldPosition = particle.position;
            if ((emitter.flags & GPU_PAPI_LOCAL) != 0u)
                worldPosition = emitter.origin.xyz + emitter.basisX.xyz * particle.position.x +
                    emitter.basisY.xyz * particle.position.y + emitter.basisZ.xyz * particle.position.z;
            particle.renderPreviousWorldPosition = worldPosition;
            particle.renderMotionValid = all(isfinite(worldPosition)) ? 1u : 0u;
            g_Particles[index] = particle;
        }
    }
}
