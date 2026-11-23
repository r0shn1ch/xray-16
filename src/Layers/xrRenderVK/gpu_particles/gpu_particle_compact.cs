#include "gpu_particle_types.h"
[[vk::binding(0, 0)]] cbuffer GpuPapiParams { GpuPapiConstants g_Papi; }
[[vk::binding(1, 0)]] cbuffer GpuPapiVisibility { float4 g_FrustumPlanes[6]; }
[[vk::binding(2, 0)]] StructuredBuffer<uint> g_VisibleRoots;
[[vk::binding(3, 0)]] StructuredBuffer<GpuPapiParticle> g_Particles;
[[vk::binding(4, 0)]] StructuredBuffer<GpuPapiEmitter> g_Emitters;
[[vk::binding(5, 0)]] StructuredBuffer<GpuPapiProgram> g_Programs;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> g_Indices;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint4> g_DrawArgs;
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> g_BucketOffsets;
[[vk::binding(9, 0)]] RWStructuredBuffer<uint> g_BucketCursors;
groupshared uint g_LaneOffsets[64];
groupshared uint g_EmitterFirst;
uint GpuDrawBucket(GpuPapiEmitter emitter,GpuPapiProgram program,uint rootIndex,bool child) {
    bool hud = (emitter.flags & GPU_PAPI_HUD) != 0;
    uint blendBucket = program.shaderVariant == 2 ? (hud ? 13u : 12u) : min(program.blendMode,5u) + (hud ? 6u : 0u);
    return (rootIndex * g_Papi.programCount + program.materialID) * 28u + (child ? 14u : 0u) + blendBucket;
}
float3 GpuCompactTransformDirection(GpuPapiEmitter emitter,float3 direction) {
    return emitter.basisX.xyz * direction.x + emitter.basisY.xyz * direction.y + emitter.basisZ.xyz * direction.z;
}
bool GpuParticleVisible(GpuPapiParticle particle,GpuPapiEmitter emitter,GpuPapiProgram program) {
    if ((emitter.flags & GPU_PAPI_HUD) != 0) return true;
    bool localSpace = (emitter.flags & GPU_PAPI_LOCAL) != 0;
    float3 position = particle.position;
    if (localSpace) position = GpuCompactTransformDirection(emitter,position) + emitter.origin.xyz;
    float2 halfSize = particle.size.xy * 0.5;
    float speed = 0.0;
    bool aligned = (program.flags & (1u << 15u)) != 0;
    if ((program.flags & ((1u << 18u) | (1u << 15u))) != 0) speed = length(particle.velocity);
    if ((program.flags & (1u << 18u)) != 0) halfSize += speed * program.velocityScale.xy;
    float radius = length(halfSize);
    bool transformedPlane = aligned && ((speed < 0.0000001 && (program.flags & (1u << 20u)) != 0) ||
        (speed >= 0.0000001 && (program.flags & (1u << 21u)) != 0));
    float3 top = 0.0;
    if (aligned && !transformedPlane) {
        if (speed >= 0.0000001) top = particle.velocity / speed;
        else {
            float2 sine,cosine;
            sincos(program.alignRotation.xy,sine,cosine);
            top = float3(cosine.x * sine.y,-sine.x,cosine.x * cosine.y);
        }
        if (localSpace) top = GpuCompactTransformDirection(emitter,top);
    }
    for (uint plane = 0; plane < 6; ++plane) {
        float4 p = g_FrustumPlanes[plane];
        float projectedScaleSquared = dot(p.xyz,p.xyz);
        if (transformedPlane && localSpace) {
            float3 projectedBasis = float3(dot(p.xyz,emitter.basisX.xyz),dot(p.xyz,emitter.basisY.xyz),
                dot(p.xyz,emitter.basisZ.xyz));
            projectedScaleSquared = dot(projectedBasis,projectedBasis);
        }
        else if (aligned && !transformedPlane) {
            float projectedTop = dot(p.xyz,top);
            projectedScaleSquared += projectedTop * projectedTop;
        }
        float extent = radius * sqrt(projectedScaleSquared);
        float distance = dot(p.xyz,position) + p.w;
        float tolerance = 0.00001 * (dot(abs(p.xyz),abs(position)) + abs(p.w) + extent + 1.0);
        if (distance > extent + tolerance) return false;
    }
    return true;
}
void GpuCompactEmitter(GpuPapiEmitter emitter,uint lane,uint rootIndex,bool child) {
    if (emitter.active != 1 || emitter.count == 0) return;
    GpuPapiProgram program = g_Programs[emitter.program];
    if ((program.flags & 1u) == 0 || program.materialID >= g_Papi.programCount) return;
    uint bucket = GpuDrawBucket(emitter,program,rootIndex,child);
    uint laneSize = emitter.count / 64;
    uint remainder = emitter.count % 64;
    uint begin = lane * laneSize + min(lane,remainder);
    uint end = begin + laneSize + (lane < remainder ? 1u : 0u);
    bool hud = (emitter.flags & GPU_PAPI_HUD) != 0;
    uint visibleCount = end - begin;
    if (!hud) {
        visibleCount = 0;
        for (uint ordinal = begin; ordinal < end; ++ordinal)
            if (GpuParticleVisible(g_Particles[emitter.particleFirst + ordinal],emitter,program)) ++visibleCount;
    }
    g_LaneOffsets[lane] = visibleCount;
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0) {
        uint total = 0;
        for (uint i = 0; i < 64; ++i) {
            uint count = g_LaneOffsets[i];
            if (g_Papi.phase != 0) g_LaneOffsets[i] = total;
            total += count;
        }
        g_EmitterFirst = 0;
        if (total != 0) {
            if (g_Papi.phase == 0) InterlockedAdd(g_DrawArgs[bucket].y,total);
            else InterlockedAdd(g_BucketCursors[bucket],total,g_EmitterFirst);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (g_Papi.phase != 0 && visibleCount != 0) {
        uint first = g_EmitterFirst + g_LaneOffsets[lane];
        uint bucketOffset = g_BucketOffsets[bucket];
        uint bucketCount = g_DrawArgs[bucket].y;
        for (uint ordinal = begin; ordinal < end; ++ordinal) {
            uint particleIndex = emitter.particleFirst + ordinal;
            bool visible = true;
            if (!hud) visible = GpuParticleVisible(g_Particles[particleIndex],emitter,program);
            if (visible) {
                if (first < bucketCount && bucketOffset + first < g_Papi.particleCapacity)
                    g_Indices[bucketOffset + first] = particleIndex;
                ++first;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
}
[numthreads(64,1,1)]
void main(uint3 groupID : SV_GroupID,uint lane : SV_GroupIndex) {
    if (g_Papi.phase == 1) {
        if (groupID.x != 0) return;
        if (lane == 0) g_EmitterFirst = 0;
        GroupMemoryBarrierWithGroupSync();
        uint bucketCount = g_Papi.rootCapacity * g_Papi.programCount * 28u;
        for (uint first = 0; first < bucketCount; first += 64u) {
            uint bucket = first + lane;
            uint count = bucket < bucketCount ? min(g_DrawArgs[bucket].y,g_Papi.particleCapacity) : 0u;
            g_LaneOffsets[lane] = count;
            GroupMemoryBarrierWithGroupSync();
            // Saturating inclusive scan prevents integer overflow even if the
            // input counts exceed the available particle-index buffer.
            for (uint stride = 1; stride < 64u; stride <<= 1u) {
                uint add = lane >= stride ? g_LaneOffsets[lane - stride] : 0u;
                GroupMemoryBarrierWithGroupSync();
                uint value = g_LaneOffsets[lane];
                g_LaneOffsets[lane] = value + min(add,g_Papi.particleCapacity - value);
                GroupMemoryBarrierWithGroupSync();
            }
            uint base = g_EmitterFirst;
            uint before = lane == 0 ? 0u : g_LaneOffsets[lane - 1u];
            uint offset = base + min(before,g_Papi.particleCapacity - base);
            if (bucket < bucketCount) {
                g_BucketOffsets[bucket] = offset;
                g_BucketCursors[bucket] = 0;
                g_DrawArgs[bucket] = uint4(6,min(count,g_Papi.particleCapacity - offset),0,0);
            }
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0)
                g_EmitterFirst = base + min(g_LaneOffsets[63],g_Papi.particleCapacity - base);
            GroupMemoryBarrierWithGroupSync();
        }
        return;
    }
    uint index = g_Papi.pad + groupID.x;
    if (index >= g_Papi.rootCapacity) return;
    GpuPapiEmitter emitter = g_Emitters[g_VisibleRoots[index]];
    if (emitter.active != 1) return;
    GpuCompactEmitter(emitter,lane,index,false);
    uint child = emitter.pad0;
    while (child != GPU_PAPI_INVALID) {
        GpuPapiEmitter childEmitter = g_Emitters[child];
        GpuCompactEmitter(childEmitter,lane,index,true);
        child = childEmitter.pad0;
    }
}
