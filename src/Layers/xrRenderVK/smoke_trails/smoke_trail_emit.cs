#include "smoke_common.h"
[[vk::binding(0, 0)]] cbuffer SmokeEmitCB {
    float g_PrevPosX, g_PrevPosY, g_PrevPosZ, g_CurrPosX;
    float g_CurrPosY, g_CurrPosZ, g_EmitDirX, g_EmitDirY;
    float g_EmitDirZ, g_BaseLifetime, g_LifetimeVariance, g_MaxWidth;
    uint g_EmitCount, g_MaxPoints; float g_FrameSeed; uint g_Restart;
};
[[vk::binding(1, 0)]] RWStructuredBuffer<SmokeSimPoint> g_SimBuffer;
[[vk::binding(2, 0)]] RWByteAddressBuffer g_StateBuffer;
groupshared uint g_First;
[numthreads(64, 1, 1)]
void main(uint lane : SV_GroupIndex) {
    uint count = min(g_EmitCount, g_MaxPoints);
    if (lane == 0) {
        g_First = 0;
        if (count != 0 && g_Restart != 0) {
            g_StateBuffer.Store(0, count);
            g_StateBuffer.Store(4, count);
        } else if (count != 0) {
            g_StateBuffer.InterlockedAdd(0, count, g_First);
            uint ignored; g_StateBuffer.InterlockedAdd(4, count, ignored);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (uint i = lane; i < count; i += 64) {
        float t = count > 1 ? (float(i) + 0.5) / float(count) : 0.5;
        float3 pos = lerp(float3(g_PrevPosX,g_PrevPosY,g_PrevPosZ), float3(g_CurrPosX,g_CurrPosY,g_CurrPosZ), t);
        SmokeSimPoint p = (SmokeSimPoint)0;
        p.px = pos.x; p.py = pos.y; p.pz = pos.z; p.lifetime = g_BaseLifetime;
        p.vy = 0.18; p.width = 0.001;
        g_SimBuffer[(g_First + i) % g_MaxPoints] = p;
    }
}
