// Screen-space ambient occlusion shared by the plain and weather combine paths.
// The inverse view-projection reconstructs positions in world units, so the
// distance rejection remains stable as the camera moves toward an object.
const int ao_ring_count = 2;
const int maximum_ao_directions = 16;
const int ao_ring_radii[ao_ring_count] = int[ao_ring_count](4, 12);
// Every quality tier uses complete rotations around the center.
const ivec2 ao_directions[maximum_ao_directions] = ivec2[maximum_ao_directions](
    ivec2(1, 0), ivec2(0, 1), ivec2(-1, 0), ivec2(0, -1),
    ivec2(1, 1), ivec2(-1, 1), ivec2(-1, -1), ivec2(1, -1),
    ivec2(2, 1), ivec2(-1, 2), ivec2(-2, -1), ivec2(1, -2),
    ivec2(1, 2), ivec2(-2, 1), ivec2(-1, -2), ivec2(2, -1));

vec3 ao_world_position(vec2 uv, float depth)
{
    vec4 clip = shadow.inverse_view_projection *
        vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return clip.xyz / clip.w;
}

float ambient_occlusion(float depth, vec2 uv, vec3 normal)
{
    if (!(shadow.options.w > 0.0 && depth >= 0.0 && depth < 0.99999) ||
        any(isnan(normal)) || any(isinf(normal)))
        return 1.0;
    ivec2 size = textureSize(resolved_depth_buffer, 0);
    ivec2 center = clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
    vec3 position = ao_world_position(uv, depth);
    if (any(isnan(position)) || any(isinf(position))) return 1.0;
    int direction_count = clamp(int(shadow.ao_settings.x + 0.5), 4, maximum_ao_directions);
    // The rings are specified in presentation pixels.
    vec2 scene_scale = shadow.ao_settings.yz;
    if (!all(greaterThan(scene_scale, vec2(0.0))) || any(isinf(scene_scale)))
        scene_scale = vec2(1.0);
    float occlusion = 0.0;
    for (int ring = 0; ring < ao_ring_count; ++ring)
    {
        for (int i = 0; i < maximum_ao_directions; ++i)
        {
            if (i >= direction_count)
                break;
            vec2 direction = normalize(vec2(ao_directions[i]));
            ivec2 offset = ivec2(round(direction * float(ao_ring_radii[ring]) * scene_scale));
            ivec2 pixel = center + offset;
            if (any(lessThan(pixel, ivec2(0))) || any(greaterThanEqual(pixel, size)))
                continue;
            float sample_depth = texelFetch(resolved_depth_buffer, pixel, 0).r;
            if (!(sample_depth >= 0.0 && sample_depth < 0.99999))
                continue;
            vec2 sample_uv = (vec2(pixel) + 0.5) / vec2(size);
            vec3 displacement = ao_world_position(sample_uv, sample_depth) - position;
            float distance_to_sample = length(displacement);
            if (!(distance_to_sample >= 0.01 && distance_to_sample < 1.25))
                continue;
            float facing = dot(normal, displacement) / distance_to_sample;
            float range_weight = 1.0 - smoothstep(0.15, 1.25, distance_to_sample);
            occlusion += smoothstep(0.08, 0.35, facing) * range_weight;
        }
    }
    float sample_count = float(ao_ring_count * direction_count);
    return clamp(1.0 - shadow.options.w * occlusion / sample_count, 0.2, 1.0);
}
