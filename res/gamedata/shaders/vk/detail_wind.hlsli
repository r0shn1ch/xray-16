float3 rotate_detail(float3 p, float2 rotation) {
    return float3(rotation.x * p.x + rotation.y * p.z, p.y, -rotation.y * p.x + rotation.x * p.z);
}
float3 detail_world_position(float3 position, float4 position_scale, float2 rotation,
    float weight, float4 directions, float wind_phase) {
    float3 root = position_scale.xyz;
    float3 p = rotate_detail(position * position_scale.w, rotation) + root;
    uint hash = asuint(root.x == 0.0 ? 0.0 : root.x) * 0x9e3779b9u ^
        asuint(root.z == 0.0 ? 0.0 : root.z) * 0x85ebca6bu;
    hash ^= hash >> 16;
    bool second = hash % 3u == 0u;
    float3 wave = second ? float3(1.0 / 3.0, 1.0 / 7.0, 1.0 / 5.0) : float3(1.0 / 5.0, 1.0 / 7.0, 1.0 / 3.0);
    float phase = frac(wind_phase + dot(p, wave) / 6.283185307179586);
    float cyclic = 2.8284271 * phase - 1.4142136;
    float bend = (p.y - root.y) * (cyclic * cyclic - 1.0) * weight;
    p.xz += (second ? directions.zw : directions.xy) * bend;
    return p;
}
