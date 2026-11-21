float rain_wetness(vec3 world_position, vec3 normal, vec3 camera_position)
{
    float density = shadow.rain_params.x;
    if (density <= 1e-4)
        return 0.0;

    // Offset outward by about a map texel so downward projection does not
    // self-occlude rain-facing wall pixels.
    float texel_world = 2.0 / max(length(shadow.rain_view_projection[0].xyz) *
        float(textureSize(rain_shadow, 0).x), 1e-4);
    vec3 sample_position = world_position + vec3(normal.x, 0.0, normal.z) *
        max(0.035, 1.25 * texel_world);
    vec4 clip = shadow.rain_view_projection * vec4(sample_position, 1.0);
    if (clip.w <= 0.0)
        return 0.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        projected.z <= 0.0 || projected.z >= 1.0)
        return 0.0;

    ivec2 size = textureSize(rain_shadow, 0);
    vec2 grid = uv * vec2(size) - 0.5;
    ivec2 base = ivec2(floor(grid));
    vec2 fraction = fract(grid);
    float visible = 0.0;
    for (int y = 0; y <= 1; ++y)
        for (int x = 0; x <= 1; ++x)
        {
            ivec2 pixel = clamp(base + ivec2(x, y), ivec2(0), size - 1);
            float tap = projected.z - 0.00025 <= texelFetch(rain_shadow, pixel, 0).r ? 1.0 : 0.0;
            vec2 weight = mix(1.0 - fraction, fraction, vec2(x, y));
            visible += tap * weight.x * weight.y;
        }

    float near_distance = max(shadow.rain_params.y, 0.0);
    float far_distance = max(shadow.rain_params.z, near_distance + 0.01);
    float fade = 1.0 - smoothstep(near_distance, far_distance,
        length(world_position - camera_position));
    float rain_facing = clamp(0.55 + 0.45 * normal.y, 0.0, 1.0);
    return clamp(visible * density * fade * fade * rain_facing, 0.0, 1.0);
}

vec3 rain_perturb_normal(vec3 world_position, vec3 normal, float wetness)
{
    if (wetness <= 1e-4)
        return normal;
    float time = shadow.rain_params.w;
    vec3 wave_x = vec3(0.20, 1.0, 0.07);
    vec3 wave_z = vec3(-0.06, 1.0, 0.22);
    float phase_x = dot(world_position, wave_x) * 13.0 + time * 3.1;
    float phase_z = dot(world_position, wave_z) * 19.0 - time * 2.3;
    vec3 gradient = wave_x * cos(phase_x) * 0.035 + wave_z * cos(phase_z) * 0.022;
    gradient -= normal * dot(normal, gradient);
    return normalize(normal - gradient * wetness);
}
