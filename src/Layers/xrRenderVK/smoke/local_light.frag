#version 450
#extension GL_GOOGLE_include_directive : require
#include "pbr_material.glsl"
#ifdef GBUFFER_MSAA
layout(set = 0, binding = 0) uniform sampler2DMS albedo_buffer;
layout(set = 0, binding = 1) uniform sampler2DMS normal_buffer;
layout(set = 0, binding = 2) uniform sampler2DMS depth_buffer;
layout(set = 0, binding = 5) uniform sampler2DMS surface_buffer;
#else
layout(set = 0, binding = 0) uniform sampler2D albedo_buffer;
layout(set = 0, binding = 1) uniform sampler2D normal_buffer;
layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
layout(set = 0, binding = 5) uniform sampler2D surface_buffer;
#endif
layout(set = 0, binding = 4, std140) uniform ShadowConstants {
    mat4 inverse_view_projection;
    mat4 sun_view_projection;
    vec4 options;
    vec4 ambient_color;
    vec4 hemi_color;
    mat4 rain_view_projection;
    vec4 rain_params;
    vec4 fog_plane;
    vec4 fog_color;
    vec4 ao_settings;
} shadow;
layout(set = 0, binding = 10) uniform sampler2D rain_shadow;
layout(set = 1, binding = 0) uniform sampler2DArray shadow_array;
layout(set = 1, binding = 1, std140) uniform LocalLight {
    mat4 inverse_view_projection;
    mat4 shadow_matrices[6];
    vec4 position_range;
    vec4 direction_cone;
    vec4 color_type;
    vec4 shadow_params;
    vec4 spot_right;
    vec4 debug_mode;
} light;
layout(set=1,binding=2) uniform sampler2D spot_projection;
layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 pixel_color;
#include "rain_wetness.glsl"

ivec2 gbuffer_pixel(vec2 uv, ivec2 size)
{
    return clamp(ivec2(uv * vec2(size)), ivec2(0), size - 1);
}

float gbuffer_depth(vec2 uv, int sample_index)
{
#ifdef GBUFFER_MSAA
    return texelFetch(depth_buffer, gbuffer_pixel(uv, textureSize(depth_buffer)), sample_index).r;
#else
    return texture(depth_buffer, uv).r;
#endif
}

vec4 gbuffer_albedo(vec2 uv, int sample_index)
{
#ifdef GBUFFER_MSAA
    return texelFetch(albedo_buffer, gbuffer_pixel(uv, textureSize(albedo_buffer)), sample_index);
#else
    return texture(albedo_buffer, uv);
#endif
}

vec4 gbuffer_normal(vec2 uv, int sample_index)
{
#ifdef GBUFFER_MSAA
    return texelFetch(normal_buffer, gbuffer_pixel(uv, textureSize(normal_buffer)), sample_index);
#else
    return texture(normal_buffer, uv);
#endif
}

vec4 gbuffer_surface(vec2 uv, int sample_index)
{
#ifdef GBUFFER_MSAA
    return texelFetch(surface_buffer, gbuffer_pixel(uv, textureSize(surface_buffer)), sample_index);
#else
    return texture(surface_buffer, uv);
#endif
}

float visibility(vec3 world, vec3 delta, vec3 normal)
{
    if (light.shadow_params.z < 0.5) return 1.0;
    int face = 0;
    if (light.color_type.w < 0.5)
    {
        vec3 a = abs(delta);
        face = a.x >= a.y && a.x >= a.z ? (delta.x >= 0.0 ? 0 : 1) :
            a.y >= a.z ? (delta.y >= 0.0 ? 2 : 3) : (delta.z >= 0.0 ? 4 : 5);
    }
    vec4 clip = light.shadow_matrices[face] * vec4(world, 1.0);
    if (clip.w <= 0.0) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        projected.z <= 0.0 || projected.z >= 1.0) return 1.0;
    // Bias in world units: a fixed 0..1 perspective-depth offset erases
    // shadows at distance because depth precision scales with 1/z squared.
    vec3 receiver = world - normalize(delta) * light.shadow_params.y;
    vec4 receiver_clip = light.shadow_matrices[face] * vec4(receiver, 1.0);
    if (receiver_clip.w <= 0.0) return 1.0;
    float reference_depth = receiver_clip.z / receiver_clip.w;
    // Correct each PCF tap to the receiver plane. A single centre depth
    // self-shadows sloped faces when the light texel covers several pixels.
    // Derive the receiver plane from its world normal. Screen derivatives
    // cross silhouettes and are undefined after the range/cone branches.
    vec3 tangent = normalize(cross(normal, abs(normal.y) < .9 ? vec3(0,1,0) : vec3(1,0,0)));
    vec3 bitangent = cross(normal, tangent);
    vec4 tx = light.shadow_matrices[face] * vec4(tangent, 0.0);
    vec4 ty = light.shadow_matrices[face] * vec4(bitangent, 0.0);
    vec3 dx = (tx.xyz - projected * tx.w) / clip.w;
    vec3 dy = (ty.xyz - projected * ty.w) / clip.w;
    vec2 ux = dx.xy * .5, uy = dy.xy * .5;
    float determinant = ux.x * uy.y - ux.y * uy.x;
    vec2 gradient = abs(determinant) > 1e-12 ?
        vec2(uy.y * dx.z - ux.y * dy.z, ux.x * dy.z - uy.x * dx.z) / determinant : vec2(0);
    float layer = light.shadow_params.x + float(face);
    ivec2 size = textureSize(shadow_array, 0).xy;
    vec2 grid = uv * vec2(size) - 0.5;
    ivec2 center = ivec2(floor(grid));
    vec2 fraction = grid - vec2(center);
    float lit = 0.0, total = 0.0;
    for (int y = 0; y <= 1; ++y)
        for (int x = 0; x <= 1; ++x)
        {
            vec2 weight = vec2(x == 0 ? 1.0-fraction.x : fraction.x, y == 0 ? 1.0-fraction.y : fraction.y);
            float w = weight.x * weight.y;
            ivec2 pixel = clamp(center + ivec2(x,y), ivec2(0), size - 1);
            float tap_depth = reference_depth + clamp(dot(gradient, (vec2(pixel) + .5) / vec2(size) - uv), -.01, .01);
            lit += (tap_depth <=
                texelFetch(shadow_array, ivec3(pixel, int(layer)), 0).r ? 1.0 : 0.0) * w;
            total += w;
        }
    return lit / max(total, 1e-6);
}

float volume_shadow(vec3 world, vec3 delta)
{
    if (light.shadow_params.z < 0.5) return 1.0;
    int face = 0;
    if (light.color_type.w < 0.5)
    {
        vec3 a = abs(delta);
        face = a.x >= a.y && a.x >= a.z ? (delta.x >= 0.0 ? 0 : 1) :
            a.y >= a.z ? (delta.y >= 0.0 ? 2 : 3) : (delta.z >= 0.0 ? 4 : 5);
    }
    vec4 clip = light.shadow_matrices[face] * vec4(world, 1.0);
    if (clip.w <= 0.0) return 1.0;
    vec3 projected = clip.xyz / clip.w;
    vec2 uv = projected.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        projected.z <= 0.0 || projected.z >= 1.0) return 1.0;
    float nearest_depth = textureLod(shadow_array, vec3(uv, light.shadow_params.x + float(face)), 0.0).r;
    return projected.z <= nearest_depth + 0.0001 ? 1.0 : 0.0;
}

vec3 volumetric_light(vec2 uv, float depth)
{
    if (light.debug_mode.y <= 0.0 || light.debug_mode.z <= 0.0 || light.debug_mode.w <= 0.0)
        return vec3(0.0);
    vec2 clip_xy = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    vec4 near_clip = light.inverse_view_projection * vec4(clip_xy, 0.0, 1.0);
    vec4 far_clip = light.inverse_view_projection * vec4(clip_xy, min(depth, 0.99999), 1.0);
    if (abs(near_clip.w) < 1e-6 || abs(far_clip.w) < 1e-6) return vec3(0.0);
    vec3 origin = near_clip.xyz / near_clip.w;
    vec3 segment = far_clip.xyz / far_clip.w - origin;
    float segment_length = length(segment);
    if (segment_length < 1e-5) return vec3(0.0);
    vec3 ray = segment / segment_length;
    float radius = max(light.position_range.w * light.debug_mode.w, 0.01);
    vec3 from_center = origin - light.position_range.xyz;
    float b = dot(from_center, ray);
    float discriminant = b * b - dot(from_center, from_center) + radius * radius;
    if (discriminant <= 0.0) return vec3(0.0);
    float root = sqrt(discriminant);
    float start = max(-b - root, 0.0);
    float end = min(-b + root, segment_length);
    if (end <= start) return vec3(0.0);
    int steps = int(clamp(ceil(8.0 * light.debug_mode.z), 4.0, 32.0));
    float step_length = (end - start) / float(steps);
    float sum = 0.0;
    for (int i = 0; i < steps; ++i)
    {
        vec3 world = origin + ray * (start + (float(i) + 0.5) * step_length);
        vec3 delta = world - light.position_range.xyz;
        float distance_squared = dot(delta, delta);
        if (distance_squared < 1e-10) continue;
        float attenuation = max(1.0 - distance_squared / (radius * radius), 0.0);
        if (light.color_type.w >= 0.5)
            attenuation *= smoothstep(light.direction_cone.w, min(1.0, light.direction_cone.w + 0.05),
                dot(normalize(delta), normalize(light.direction_cone.xyz)));
        if (light.shadow_params.w > 0.5)
        {
            vec3 direction = normalize(light.direction_cone.xyz);
            vec3 right = normalize(light.spot_right.xyz - direction * dot(light.spot_right.xyz, direction));
            vec3 up = cross(direction, right);
            float z = dot(delta, direction);
            vec2 projection_uv = vec2(dot(delta, right), -dot(delta, up)) /
                max(z * light.spot_right.w, 1e-6) * .5 + .5;
            if (z <= 0.0 || any(lessThan(projection_uv, vec2(0.0))) || any(greaterThan(projection_uv, vec2(1.0))))
                attenuation = 0.0;
            else
                attenuation *= textureLod(spot_projection, projection_uv, 0.0).r;
        }
        sum += attenuation * volume_shadow(world, delta);
    }
    return light.color_type.rgb * (sum * step_length / radius) * light.debug_mode.y;
}

void shade_local_light_sample(int sample_index, out vec4 output_color)
{
    // Use the base panel's G-buffer coordinates in the four-panel view.
    vec2 screen_uv = light.debug_mode.x > 0.5 && light.debug_mode.x < 1.5 ?
        texcoord * 2.0 : texcoord;
    if (any(greaterThanEqual(screen_uv, vec2(1.0))))
    { output_color = vec4(0.0); return; }
    float depth = gbuffer_depth(screen_uv, sample_index);
    vec3 volume = volumetric_light(screen_uv, depth);
    if (depth >= 0.99999) { output_color = vec4(volume, 0.0); return; }
    vec4 world_clip = light.inverse_view_projection *
        vec4(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0, depth, 1.0);
    vec3 world = world_clip.xyz / max(abs(world_clip.w), 1e-6) * sign(world_clip.w);
    float fog_visibility = clamp(dot(vec4(world, 1.0), shadow.fog_plane), 0.0, 1.0);
    vec3 delta = world - light.position_range.xyz;
    float distance = length(delta);
    float range = max(light.position_range.w, 0.01);
    if (distance >= range || distance <= 1e-5) { output_color = vec4(volume, 0.0); return; }
    vec4 encoded_normal = gbuffer_normal(screen_uv, sample_index);
    vec3 normal = normalize(encoded_normal.rgb * 2.0 - 1.0);
    float wetness = 0.0;
    if (shadow.rain_params.x > 1e-4)
    {
        vec4 eye_clip = light.inverse_view_projection * vec4(0.0, 0.0, 1.0, 0.0);
        if (abs(eye_clip.w) < 1e-6)
            eye_clip = light.inverse_view_projection * vec4(0.0, 0.0, 0.0, 1.0);
        vec3 camera_position = eye_clip.xyz / max(abs(eye_clip.w), 1e-6) * sign(eye_clip.w);
        wetness = rain_wetness(world, normal, camera_position);
        normal = rain_perturb_normal(world, normal, wetness);
    }
    vec4 surface = gbuffer_surface(screen_uv, sample_index);
    float diffuse = max(dot(normal, -delta / distance), 0.0);
    float normalized_distance = distance / range;
    float attenuation = max(1.0 - normalized_distance * normalized_distance, 0.0);
    if (light.color_type.w >= 0.5)
        attenuation *= smoothstep(light.direction_cone.w, min(1.0, light.direction_cone.w + 0.05),
            dot(normalize(delta), normalize(light.direction_cone.xyz)));
    if(light.shadow_params.w>0.5)
    {
        vec3 direction=normalize(light.direction_cone.xyz);
        vec3 right=normalize(light.spot_right.xyz-direction*dot(light.spot_right.xyz,direction));
        vec3 up=cross(direction,right);
        float z=dot(delta,direction);
        vec2 projection_uv=vec2(dot(delta,right),-dot(delta,up))/max(z*light.spot_right.w,1e-6)*.5+.5;
        if(z<=0.0 || any(lessThan(projection_uv,vec2(0))) || any(greaterThan(projection_uv,vec2(1))))
        {output_color=vec4(volume,0);return;}
        attenuation*=texture(spot_projection,projection_uv).r;
    }
    vec3 albedo = gbuffer_albedo(screen_uv, sample_index).rgb * (1.0 - 0.16 * wetness);
    vec3 color = albedo * light.color_type.rgb * diffuse;
    bool is_pbr = pbr_material(encoded_normal.a);
    if (is_pbr || surface.r > 0.0)
    {
        vec4 near_clip = light.inverse_view_projection *
            vec4(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0, 0.0, 1.0);
        vec3 view_ray = normalize(near_clip.xyz / near_clip.w - world);
        if (is_pbr)
            color = pbr_direct(albedo, surface.g, mix(surface.r, 0.08, wetness * 0.8),
                normal, view_ray, -delta / distance) * light.color_type.rgb;
        else
        {
            float specular = pow(max(dot(normal, normalize(view_ray - delta / distance)), 0.0),
                mix(4.0, 64.0, clamp(surface.r + wetness * 0.35, 0.0, 1.0)));
            color += specular * clamp(surface.r + wetness * 0.2, 0.0, 1.0) *
                mix(0.04, 0.25, surface.g) * light.color_type.rgb;
        }
    }
    color *= attenuation * visibility(world, delta, normal) * fog_visibility;
    output_color = vec4(color + volume * fog_visibility, 0.0);
}

void main()
{
#ifdef GBUFFER_MSAA
    int samples = textureSamples(albedo_buffer);
    vec4 resolved = vec4(0.0);
    for (int sample_index = 0; sample_index < samples; ++sample_index)
    {
        vec4 sample_color;
        shade_local_light_sample(sample_index, sample_color);
        resolved += sample_color;
    }
    pixel_color = resolved / float(max(samples, 1));
#else
    shade_local_light_sample(0, pixel_color);
#endif
}
