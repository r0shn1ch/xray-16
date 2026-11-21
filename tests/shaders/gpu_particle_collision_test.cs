#include "gpu_particle_collision.h"
#include "gpu_particle_animation.h"
[[vk::binding(0, 0)]] RWStructuredBuffer<float4> output;

[numthreads(1, 1, 1)]
void main()
{
    float3 normal; uint source;
    bool hit = GpuParticleRayPick(float3(0,0,0), float3(0,0,1), 4, false, false, normal, source);
    output[0] = float4(hit ? 1 : 0, source, normal.z, normal.y);
    hit = GpuParticleRayPick(float3(0,0,0), float3(0,0,1), 4, true, false, normal, source);
    output[1] = float4(hit ? 1 : 0, source, normal.z, normal.y);
    hit = GpuParticleRayPick(float3(0,0,0), float3(0,0,1), 4, true, true, normal, source);
    output[2] = float4(hit ? 1 : 0, source, normal.z, normal.y);
    hit = GpuParticleRayPick(float3(2,0,0), float3(0,0,1), 4, true, false, normal, source);
    output[3] = float4(hit ? 1 : 0, source, normal.z, normal.y);
    hit = GpuParticleRayPick(float3(5,0,0), float3(0,0,1), 4, true, false, normal, source);
    output[4] = float4(hit ? 1 : 0, source, normal.z, normal.y);
    float3 position = float3(0,0,3), velocity = float3(0,0,3);
    bool dead = false;
    uint response = GpuParticleCollision(position, float3(0,0,0), velocity, 1, 0, 1, 1, 0, dead);
    output[5] = float4(response, dead ? 1 : 0, position.z, velocity.z);
    position = float3(0,0,3); velocity = float3(0,0,3); dead = false;
    response = GpuParticleCollision(position, float3(0,0,0), velocity, 1, 1u << 17, 1, 1, 0, dead);
    output[6] = float4(response, dead ? 1 : 0, position.z, velocity.z);
    GpuPapiParticle p = (GpuPapiParticle)0;
    GpuPapiProgram program = (GpuPapiProgram)0;
    program.frameCount = 2; program.frameSpeed = 1000.0;
    GpuAnimateParticle(p, program);
    float forward = p.frame;
    p.frame = 0; p.flags = 1;
    GpuAnimateParticle(p, program);
    float reverse = p.frame;
    p.frame = 255; p.flags = 0;
    GpuAnimateParticle(p, program);
    float wrapped = p.frame;
    p.frame = 123; program.frameCount = 0;
    GpuAnimateParticle(p, program);
    output[7] = float4(forward, reverse, wrapped, p.frame);
}
