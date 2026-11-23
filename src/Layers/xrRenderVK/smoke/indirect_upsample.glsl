// Depth- and normal-aware reconstruction of the reduced-resolution GI image.
// The source pixel is selected in full-resolution G-buffer coordinates in the
// same way as the indirect ray generator.
#include "indirect_weight.glsl"
vec3 sample_indirect_radiance(vec2 uv, float center_depth, vec3 center_normal)
{
    ivec2 gi_size = textureSize(indirect_radiance, 0);
    ivec2 depth_size = textureSize(resolved_depth_buffer, 0);
    ivec2 normal_size = textureSize(resolved_normal_buffer, 0);
    vec2 position = uv * vec2(gi_size) - 0.5;
    ivec2 base = ivec2(floor(position));
    vec2 fraction = fract(position);
    vec3 total = vec3(0.0);
    float total_weight = 0.0;
    vec3 center_position = ao_world_position(uv, center_depth);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x)
        {
            ivec2 gi_pixel = clamp(base + ivec2(x, y), ivec2(0), gi_size - 1);
            vec2 sample_uv = (vec2(gi_pixel) + 0.5) / vec2(gi_size);
            ivec2 depth_pixel = clamp(ivec2(sample_uv * vec2(depth_size)), ivec2(0), depth_size - 1);
            ivec2 normal_pixel = clamp(ivec2(sample_uv * vec2(normal_size)), ivec2(0), normal_size - 1);
            float sample_depth = texelFetch(resolved_depth_buffer, depth_pixel, 0).r;
            if (sample_depth >= 0.99999) continue;
            vec3 sample_normal = texelFetch(resolved_normal_buffer, normal_pixel, 0).rgb * 2.0 - 1.0;
            float normal_length = length(sample_normal);
            float normal_weight = normal_length > 1e-6 ?
                pow(max(dot(center_normal, sample_normal / normal_length), 0.0), 16.0) : 0.0;
            vec2 depth_uv = (vec2(depth_pixel) + 0.5) / vec2(depth_size);
            vec3 sample_position = ao_world_position(depth_uv, sample_depth);
            float depth_weight = indirect_plane_weight(center_position, sample_position, center_normal);
            float bilinear = (x == 0 ? 1.0 - fraction.x : fraction.x) *
                (y == 0 ? 1.0 - fraction.y : fraction.y);
            float weight = bilinear * depth_weight * normal_weight;
            total += texelFetch(indirect_radiance, gi_pixel, 0).rgb * weight;
            total_weight += weight;
        }
    // Reject unrelated surfaces instead of allowing a bright neighbor to bleed
    // across a silhouette when no compatible low-resolution sample exists.
    return total_weight > 1e-5 ? total / total_weight : vec3(0.0);
}
