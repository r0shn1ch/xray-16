// The prototype is immutable; both opaque and transparent details share this
// transform and the two legacy detail waves. No per-vertex CPU expansion.
[[vk::push_constant]] cbuffer Camera {
    float4x4 mvp;
    float4 material_padding[4];
    float4 base_matrix_u;
    float4 base_matrix_v;
    float4 detail_matrix_u;
    float4 detail_matrix_v;
    float4 environment_matrix_u;
    float4 environment_matrix_v;
    float4 environment_matrix_z;
    float4 object_rotation; // two XZ wind directions for details
};

struct Vertex {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float2 uv : TEXCOORD0;
    [[vk::location(5)]] float2 lightmap_uv : TEXCOORD1;
    [[vk::location(6)]] float4 baked : COLOR0;
    [[vk::location(7)]] float3 tangent : TANGENT;
    [[vk::location(8)]] float3 bitangent : BINORMAL;
    [[vk::location(9)]] float4 instance_position_scale : TEXCOORD9;
    [[vk::location(10)]] float2 instance_rotation : TEXCOORD10;
};
struct Fragment {
    float4 position : SV_Position;
    [[vk::location(0)]] float3 normal : TEXCOORD0;
    [[vk::location(1)]] float2 uv : TEXCOORD1;
    [[vk::location(2)]] float2 lightmap_uv : TEXCOORD2;
    [[vk::location(3)]] float4 baked : COLOR0;
    [[vk::location(4)]] float3 world_position : TEXCOORD4;
    [[vk::location(5)]] float hemi : TEXCOORD5;
    [[vk::location(6)]] float3 tangent : TEXCOORD6;
    [[vk::location(7)]] float3 bitangent : TEXCOORD7;
};
float3 rotate_detail(float3 p, float2 rotation) {
    return float3(rotation.x * p.x + rotation.y * p.z, p.y, -rotation.y * p.x + rotation.x * p.z);
}
Fragment main(Vertex vertex) {
    Fragment result;
    float3 root = vertex.instance_position_scale.xyz;
    float3 p = rotate_detail(vertex.position * vertex.instance_position_scale.w, vertex.instance_rotation) + root;
    uint hash = asuint(root.x == 0.0 ? 0.0 : root.x) * 0x9e3779b9u ^
        asuint(root.z == 0.0 ? 0.0 : root.z) * 0x85ebca6bu;
    hash ^= hash >> 16;
    bool second = hash % 3u == 0u;
    float3 wave = second ? float3(1.0 / 3.0, 1.0 / 7.0, 1.0 / 5.0) : float3(1.0 / 5.0, 1.0 / 7.0, 1.0 / 3.0);
    float phase = frac(base_matrix_u.z + dot(p, wave) / 6.283185307179586);
    float cyclic = 2.8284271 * phase - 1.4142136;
    float bend = (p.y - root.y) * (cyclic * cyclic - 1.0) * vertex.lightmap_uv.x;
    p.xz += (second ? object_rotation.zw : object_rotation.xy) * bend;
    result.position = mul(mvp, float4(p, 1.0));
    result.position.y = -result.position.y;
    result.normal = rotate_detail(vertex.normal, vertex.instance_rotation);
    float length_squared = dot(result.normal, result.normal);
    result.normal = length_squared > 1e-12 && length_squared < 1e20 ?
        result.normal * rsqrt(length_squared) : float3(0, 1, 0);
    result.tangent = rotate_detail(vertex.tangent, vertex.instance_rotation);
    result.bitangent = rotate_detail(vertex.bitangent, vertex.instance_rotation);
    result.uv = vertex.uv;
    result.lightmap_uv = vertex.lightmap_uv;
    result.baked = vertex.baked;
    result.world_position = p;
    result.hemi = saturate(.5 + .5 * result.normal.y);
    return result;
}
