#include "gpu_particle_types.h"

[[vk::binding(0, 1)]] StructuredBuffer<GpuPapiParticle> g_Particles;
[[vk::binding(1, 1)]] StructuredBuffer<GpuPapiEmitter> g_Emitters;
[[vk::binding(2, 1)]] StructuredBuffer<GpuPapiProgram> g_Programs;
[[vk::binding(3, 1)]] StructuredBuffer<uint> g_ParticleIndices;
[[vk::binding(4, 1)]] StructuredBuffer<uint> g_BucketOffsets;

[[vk::push_constant]] cbuffer Camera { float4x4 currentMvp; float4x4 previousMvp; };
[[vk::binding(5, 1)]] cbuffer GpuParticleDrawParams
{
    float4 g_CameraTop;
    float4 g_CameraRight;
    float4 g_CameraDirection;
    float4 g_EyePosition;
    uint g_DrawBucket;
    float3 g_DrawPadding;
    float4 g_WaterOrigin;
    float4 g_WaterNormal;
};

struct VS_OUTPUT
{
    float4 hpos : SV_Position;
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float2 texcoord : TEXCOORD1;
    [[vk::location(2)]] float2 lightmap : TEXCOORD2;
    [[vk::location(3)]] float4 color : COLOR0;
    [[vk::location(4)]] float3 worldPos : TEXCOORD4;
    [[vk::location(5)]] float hemi : TEXCOORD5;
    [[vk::location(6)]] float3 tangent : TEXCOORD6;
    [[vk::location(7)]] float3 bitangent : TEXCOORD7;
    [[vk::location(8)]] float4 previousClip : TEXCOORD8;
    [[vk::location(9)]] float4 currentClip : TEXCOORD9;
    [[vk::location(10)]] nointerpolation uint motionValid : TEXCOORD10;
};

float3 GpuParticleTransformDirection(GpuPapiEmitter emitter, float3 direction)
{
    return emitter.basisX.xyz * direction.x + emitter.basisY.xyz * direction.y + emitter.basisZ.xyz * direction.z;
}

VS_OUTPUT main(uint vertexID : SV_VertexID, uint instanceID : SV_InstanceID)
{
    uint drawBucket = g_DrawBucket;
    GpuPapiParticle particle = g_Particles[g_ParticleIndices[g_BucketOffsets[drawBucket] + instanceID]];
    GpuPapiEmitter emitter = g_Emitters[particle.emitter];
    GpuPapiProgram program = g_Programs[particle.program];
    bool localSpace = (emitter.flags & GPU_PAPI_LOCAL) != 0u;
    float3 position = particle.position;
    if (localSpace)
        position = GpuParticleTransformDirection(emitter, position) + emitter.origin.xyz;
    float3 previousCenter = particle.renderPreviousWorldPosition;

    float2 radius = particle.size.xy * 0.5;
    float speed = 0.0;
    if ((program.flags & ((1u << 18u) | (1u << 15u))) != 0u)
        speed = length(particle.velocity);
    if ((program.flags & (1u << 18u)) != 0u)
        radius += speed * program.velocityScale.xy;

    float3 top = g_CameraTop.xyz;
    float3 right = g_CameraRight.xyz;
    if (g_WaterOrigin.w > 0.5)
    {
        float3 normal = g_WaterNormal.xyz;
        float normalLength = length(normal);
        normal = normalLength > 0.000001 && isfinite(normalLength) ? normal / normalLength : float3(0, 1, 0);
        float3 projectionNormal = normal.y < 0.0 ? -normal : normal;
        position -= (dot(position - g_WaterOrigin.xyz, projectionNormal) - 0.015) * projectionNormal;
        previousCenter -= (dot(previousCenter - g_WaterOrigin.xyz, projectionNormal) - 0.015) * projectionNormal;
        float3 axis = abs(normal.y) > 0.99 ? float3(0, 0, 1) : float3(0, 1, 0);
        right = normalize(cross(axis, normal));
        top = cross(normal, right);
    }
    else if ((program.flags & (1u << 15u)) != 0u)
    {
        if (speed < 0.0000001 && (program.flags & (1u << 20u)) != 0u)
        {
            float3 sine;
            float3 cosine;
            sincos(program.alignRotation, sine, cosine);
            top = float3(-cosine.x * sine.y, sine.x, cosine.x * cosine.y);
            right = float3(cosine.y * cosine.z - sine.x * sine.y * sine.z,
                -cosine.x * sine.z, sine.x * cosine.y * sine.z + sine.y * cosine.z);
            if (localSpace)
            {
                top = GpuParticleTransformDirection(emitter, top);
                right = GpuParticleTransformDirection(emitter, right);
            }
        }
        else if (speed >= 0.0000001 && (program.flags & (1u << 21u)) != 0u)
        {
            float3 direction = particle.velocity / speed;
            float3 up = abs(direction.y) > 0.99 ? float3(0, 0, 1) : float3(0, 1, 0);
            right = normalize(cross(up, direction));
            top = normalize(cross(direction, right));
            if (localSpace)
            {
                top = GpuParticleTransformDirection(emitter, top);
                right = GpuParticleTransformDirection(emitter, right);
            }
        }
        else
        {
            if (speed >= 0.0000001)
                top = particle.velocity / speed;
            else
            {
                float2 sine;
                float2 cosine;
                sincos(program.alignRotation.xy, sine, cosine);
                top = float3(cosine.x * sine.y, -sine.x, cosine.x * cosine.y);
            }
            if (localSpace)
                top = GpuParticleTransformDirection(emitter, top);
            right = cross(top, g_CameraDirection.xyz);
            float magnitude = length(right);
            right = magnitude > 0.0000001 ? right / magnitude : g_CameraRight.xyz;
        }
    }

    float sine;
    float cosine;
    sincos(particle.rotation.x, sine, cosine);
    float3 horizontal = radius.x * (top * sine + right * cosine);
    float3 vertical = radius.y * (top * cosine - right * sine);
    static const uint corners[6] = { 0u, 1u, 2u, 2u, 1u, 3u };
    uint corner = corners[vertexID % 6u];
    float2 uv = float2((corner & 2u) != 0u ? 1.0 : 0.0, (corner & 1u) != 0u ? 0.0 : 1.0);
    float3 billboardOffset = horizontal * (uv.x * 2.0 - 1.0) + vertical * (1.0 - uv.y * 2.0);
    position += billboardOffset;
    float4 shownPos = float4(position, 1.0);

    if ((program.flags & (1u << 10u)) != 0u)
    {
        uint frame = (particle.frame / 255u) % max(program.frameCount, 1u);
        uint columns = max(program.frameDimX, 1u);
        uv = (float2(frame % columns, frame / columns) + uv) * program.frameSize;
    }

    VS_OUTPUT output;
    output.hpos = mul(currentMvp, shownPos);
    output.hpos.y = -output.hpos.y;
    output.currentClip = output.hpos;
    output.previousClip = mul(previousMvp, float4(previousCenter + billboardOffset, 1.0));
    output.previousClip.y = -output.previousClip.y;
    output.motionValid = particle.renderMotionValid;
    output.texcoord = uv;
    output.color = float4((particle.color >> 16u) & 255u, (particle.color >> 8u) & 255u,
        particle.color & 255u, (particle.color >> 24u) & 255u) / 255.0;
    output.worldPos = shownPos.xyz;
    float3 view_vector = g_EyePosition.xyz - position;
    float view_length = length(view_vector);
    output.normal = view_length > 0.0000001 ? view_vector / view_length : -g_CameraDirection.xyz;
    output.hemi = saturate(0.5 + 0.5 * output.normal.y);
    output.lightmap = float2(0.0, 0.0);
    output.tangent = right;
    output.bitangent = top;
    return output;
}
