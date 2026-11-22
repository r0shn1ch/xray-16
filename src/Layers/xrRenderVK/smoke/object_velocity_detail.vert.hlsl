#include "../../../../res/gamedata/shaders/vk/detail_wind.hlsli"
[[vk::push_constant]] cbuffer Matrices {
    float4x4 current_mvp;
    float4x4 previous_mvp;
    float4 material_padding[3];
    float4 current_directions;
    float4 previous_directions;
    float4 phases;
};
struct Vertex {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(3)]] float2 wind : TEXCOORD1;
    [[vk::location(4)]] float4 position_scale : TEXCOORD2;
    [[vk::location(5)]] float2 rotation : TEXCOORD3;
};
struct Fragment {
    float4 position : SV_Position;
    [[vk::location(0)]] float4 previous_clip : TEXCOORD0;
    [[vk::location(1)]] float4 current_clip : TEXCOORD1;
    [[vk::location(2)]] float2 uv : TEXCOORD2;
};
Fragment main(Vertex vertex) {
    Fragment result;
    float3 current = detail_world_position(vertex.position, vertex.position_scale, vertex.rotation,
        vertex.wind.x, current_directions, phases.x);
    float3 previous = detail_world_position(vertex.position, vertex.position_scale, vertex.rotation,
        vertex.wind.x, previous_directions, phases.y);
    result.position = mul(current_mvp, float4(current, 1));
    result.position.y = -result.position.y;
    result.current_clip = result.position;
    result.previous_clip = mul(previous_mvp, float4(previous, 1));
    result.previous_clip.y = -result.previous_clip.y;
    result.uv = vertex.uv;
    return result;
}
