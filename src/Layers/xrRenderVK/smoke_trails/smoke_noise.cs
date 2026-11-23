#include "noise4d.h"
[[vk::binding(0, 0)]] cbuffer Perlin4DGenParams {
    float g_time, g_tileScale; uint g_textureSize; float g_pad0;
};
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> g_output;
[numthreads(4, 4, 4)]
void main(uint3 group : SV_GroupID, uint3 lane : SV_GroupThreadID)
{
    uint groups = (g_textureSize + 3u) / 4u;
    uint3 id = uint3(group.x * 4u + lane.x, (group.y % groups) * 4u + lane.y,
        (group.y / groups) * 4u + lane.z);
    if (any(id >= g_textureSize)) return;
    float3 uvw = float3(id) / float(g_textureSize);
    float4 p = float4(uvw * g_tileScale, g_time);
    float3 value = float3(fbm4D(p), fbm4D(p + float4(31.416, -47.853, 12.679, 0.0)),
        fbm4D(p + float4(-17.236, 53.147, -28.912, 0.0)));
    uint3 rgb = uint3(round(saturate(value * 0.5 + 0.5) * 255.0));
    g_output[id.x + id.y * g_textureSize + id.z * g_textureSize * g_textureSize] =
        rgb.x | (rgb.y << 8u) | (rgb.z << 16u) | 0xff000000u;
}
