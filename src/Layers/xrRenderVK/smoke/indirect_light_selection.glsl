// Resampled importance sampling with a bounded number of uniform candidates.
// Small light sets are enumerated, avoiding random omissions on mobile scenes
// with only a few lamps. Large sets use uniform draws with replacement.
// Both ray generation and evaluation include this file so they select the
// same source and the visibility query always belongs to the evaluated light.
uint light_hash(uint x)
{
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; return x ^ (x >> 16);
}
float light_random(uint x)
{
    return float(light_hash(x) >> 8) * (1.0 / 16777216.0);
}
float light_importance(Surface surface, LocalLight light)
{
    vec3 delta = surface.position_valid.xyz - light.position_range.xyz;
    float d2 = dot(delta, delta);
    float r2 = light.position_range.w * light.position_range.w;
    if (d2 <= 1e-10 || d2 >= r2) return 0.0;
    vec3 from_light = delta * inversesqrt(d2);
    if (dot(surface.geometric_normal.xyz, -from_light) <= 0.0) return 0.0;
    float attenuation = max(1.0 - d2 / r2, 0.0);
    if (mod(light.color_type.w, 2.0) > 0.5)
        attenuation *= smoothstep(light.direction_cone.w, min(1.0, light.direction_cone.w + 0.05),
            dot(from_light, light.direction_cone.xyz));
    float cosine = max(dot(surface.normal.xyz, -from_light), 0.0);
    vec3 color = max(light.color_type.rgb, vec3(0.0));
    return max(max(color.r, color.g), color.b) * attenuation * cosine;
}
bool select_indirect_light(Surface surface, uint index, uint seed, uint light_count,
    out uint selected, out float multiplier)
{
    selected = 0u;
    multiplier = 0.0;
    if (light_count == 0u) return false;
    uint candidates = min(light_count, 8u);
    uint random_seed = light_hash(index ^ seed ^ 0x9e3779b9u);
    float sum = 0.0, selected_weight = 0.0;
    for (uint i = 0u; i < candidates; ++i)
    {
        uint candidate = light_count <= 8u ? (random_seed % light_count + i) % light_count :
            light_hash(random_seed ^ (i * 0x85ebca6bu + 0xc2b2ae35u)) % light_count;
        float weight = light_importance(surface, local_lights[candidate]);
        sum += weight;
        if (weight > 0.0 && light_random(random_seed ^ (i * 0x27d4eb2du + 0x165667b1u)) * sum < weight)
        {
            selected = candidate;
            selected_weight = weight;
        }
    }
    if (selected_weight <= 0.0) return false;
    multiplier = (float(light_count) / float(candidates)) * sum / selected_weight;
    return true;
}
