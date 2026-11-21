#version 450
#ifdef GBUFFER_MSAA
layout(set = 0, binding = 0) uniform sampler2DMS albedo_buffer;
layout(set = 0, binding = 1) uniform sampler2DMS normal_buffer;
layout(set = 0, binding = 2) uniform sampler2DMS depth_buffer;
#else
layout(set = 0, binding = 0) uniform sampler2D albedo_buffer;
layout(set = 0, binding = 1) uniform sampler2D normal_buffer;
layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
#endif
layout(set = 0, binding = 3) uniform sampler2D sun_shadow;
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
#ifdef GBUFFER_MSAA
layout(set = 0, binding = 5) uniform sampler2DMS surface_buffer;
#else
layout(set = 0, binding = 5) uniform sampler2D surface_buffer;
#endif
layout(set = 0, binding = 6) uniform sampler2D ray_visibility;
layout(set = 0, binding = 7) uniform sampler2D indirect_radiance;
layout(set = 0, binding = 8) uniform sampler2D resolved_depth_buffer;
layout(set = 0, binding = 9) uniform sampler2D resolved_normal_buffer;
layout(set = 0, binding = 10) uniform sampler2D rain_shadow;
layout(push_constant) uniform Light {
    vec4 direction_ambient;
    vec4 color;
    vec4 grade;
} light;
layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 pixel_color;
#extension GL_GOOGLE_include_directive : require
#include "ambient_occlusion.glsl"
#include "indirect_upsample.glsl"
#include "pbr_material.glsl"
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
float shadow_map_visibility(float depth, vec2 screen_uv)
{
    vec4 world = shadow.inverse_view_projection *
        vec4(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0, depth, 1.0);
    if (abs(world.w) < 1e-6) return 1.0;
    vec4 projected = shadow.sun_view_projection * (world / world.w);
    if (projected.w <= 0.0) return 1.0;
    vec3 coord = projected.xyz / projected.w;
    vec2 uv = coord.xy * 0.5 + 0.5;
    if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0))) ||
        coord.z <= 0.0 || coord.z >= 1.0) return 1.0;
    // Interpolate comparisons across the four neighbouring shadow texels.
    // Sampling at +/- half a texel with nearest filtering produced 1024px
    // blocks on low sun angles as the camera moved.
    ivec2 size = textureSize(sun_shadow, 0);
    vec2 grid = uv * vec2(size) - 0.5;
    ivec2 center = ivec2(floor(grid + 0.5));
    vec2 fraction = grid - vec2(center);
    float lit = 0.0, total = 0.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
        {
            vec2 weight = max(vec2(1.5) - abs(vec2(x, y) - fraction), vec2(0.0));
            float w = weight.x * weight.y;
            ivec2 pixel = clamp(center + ivec2(x, y), ivec2(0), size - 1);
            float visibility = coord.z - shadow.options.x <= texelFetch(sun_shadow, pixel, 0).r ? 1.0 : 0.0;
            lit += visibility * w;
            total += w;
        }
    float visibility = lit / max(total, 1e-6);
    // Fade the short mobile shadow range at its rim instead of exposing
    // the square border of the orthographic map while walking.
    float edge = min(min(uv.x, uv.y), min(1.0-uv.x, 1.0-uv.y));
    float coverage = shadow.options.z > 0.0 ? smoothstep(0.0, shadow.options.z, edge) : 1.0;
    return mix(1.0, visibility, coverage);
}
float sun_visibility(float depth, vec2 screen_uv)
{
    float visibility = shadow_map_visibility(depth, screen_uv);
    if (shadow.ambient_color.a > 0.5)
        visibility = min(visibility, texture(ray_visibility, screen_uv).r);
    return visibility;
}
void shade_deferred_sample(int sample_index, out vec4 output_color)
{
    bool diagnostic = shadow.options.y > 0.5;
    bool four_panes = diagnostic && shadow.options.y < 1.5;
    vec2 pane = four_panes ? floor(min(texcoord * 2.0, vec2(1.9999))) : vec2(0.0);
    int panel = four_panes ? int(pane.x) + 2 * int(pane.y) :
        diagnostic ? clamp(int(shadow.options.y) - 2, 0, 3) : 0;
    vec2 screen_uv = four_panes ? fract(texcoord * 2.0) : texcoord;
    float depth = gbuffer_depth(screen_uv, sample_index);
    if (diagnostic && depth >= 0.99999 && panel != 0)
    {
        output_color = vec4(0.04, 0.08, 0.18, 1.0);
        return;
    }
    vec4 albedo = gbuffer_albedo(screen_uv, sample_index);
    vec4 encoded_normal = gbuffer_normal(screen_uv, sample_index);
    vec4 surface = gbuffer_surface(screen_uv, sample_index);
    vec3 normal = normalize(encoded_normal.rgb * 2.0 - 1.0);
    float wetness = 0.0;
    if (shadow.rain_params.x > 1e-4)
    {
        vec4 rain_world_clip = shadow.inverse_view_projection *
            vec4(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0, depth, 1.0);
        vec3 rain_world = rain_world_clip.xyz / max(abs(rain_world_clip.w), 1e-6) *
            sign(rain_world_clip.w);
        vec4 eye_clip = shadow.inverse_view_projection * vec4(0.0, 0.0, 1.0, 0.0);
        if (abs(eye_clip.w) < 1e-6)
            eye_clip = shadow.inverse_view_projection * vec4(0.0, 0.0, 0.0, 1.0);
        vec3 camera_position = eye_clip.xyz / max(abs(eye_clip.w), 1e-6) * sign(eye_clip.w);
        wetness = rain_wetness(rain_world, normal, camera_position);
        normal = rain_perturb_normal(rain_world, normal, wetness);
    }
    vec3 wet_albedo = albedo.rgb * (1.0 - 0.16 * wetness);
    float diffuse = max(dot(normal, normalize(-light.direction_ambient.xyz)), 0.0);
    bool is_pbr = pbr_material(encoded_normal.a);
    bool static_lightmap = is_pbr ? pbr_static_lightmap(encoded_normal.a) : encoded_normal.a < 0.8;
    float hemisphere = is_pbr ? pbr_hemisphere(encoded_normal.a) : static_lightmap ? clamp(encoded_normal.a / 0.75, 0.0, 1.0) :
        clamp((encoded_normal.a - 0.875) * 8.0, 0.0, 1.0);
    if (diagnostic && panel == 1)
    {
        output_color = vec4(vec3(sun_visibility(depth, screen_uv)), 1.0);
        return;
    }
    if (diagnostic && panel == 2)
    {
        output_color = vec4(static_lightmap ? albedo.a : 1.0,
            hemisphere, static_lightmap ? 0.0 : 1.0, 1.0);
        return;
    }
    if (diagnostic && panel == 3)
    {
        output_color = vec4(albedo.rgb, 1.0);
        return;
    }
    float sun_scale = 0.0;
    vec3 sunlight = vec3(0.0);
    if (max(max(light.color.r, light.color.g), light.color.b) > 0.0001)
    {
        sun_scale = sun_visibility(depth, screen_uv);
        sunlight = diffuse * sun_scale * light.color.rgb;
    }
    float ao = diagnostic ? 1.0 : ambient_occlusion(depth, screen_uv, normal);
    vec3 lit = wet_albedo *
        ((shadow.ambient_color.rgb + shadow.hemi_color.rgb * hemisphere) * ao + sunlight);
    if (is_pbr)
    {
        float wet_roughness = mix(surface.r, 0.08, wetness * 0.8);
        vec4 world = shadow.inverse_view_projection * vec4(screen_uv.x * 2.0 - 1.0,
            1.0 - screen_uv.y * 2.0, depth, 1.0);
        vec4 near_point = shadow.inverse_view_projection * vec4(screen_uv.x * 2.0 - 1.0,
            1.0 - screen_uv.y * 2.0, 0.0, 1.0);
        vec3 view = normalize(near_point.xyz / near_point.w - world.xyz / world.w);
        lit = wet_albedo * (1.0 - surface.g) * surface.b *
            (shadow.ambient_color.rgb + shadow.hemi_color.rgb * hemisphere) * ao;
        if (sun_scale > 0.0 && dot(light.direction_ambient.xyz, light.direction_ambient.xyz) > 1e-8)
            lit += pbr_direct(wet_albedo, surface.g, wet_roughness, normal, view,
                normalize(-light.direction_ambient.xyz)) * light.color.rgb * sun_scale;
    }
    if (shadow.hemi_color.a > 0.0)
        lit += albedo.rgb * (is_pbr ? (1.0 - surface.g) : 1.0) *
            sample_indirect_radiance(screen_uv, depth, normal) * shadow.hemi_color.a;
    if (!is_pbr && surface.r > 0.0 && sun_scale > 0.0)
    {
        vec4 world_clip = shadow.inverse_view_projection *
            vec4(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0,
                depth, 1.0);
        vec4 near_clip = shadow.inverse_view_projection *
            vec4(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0, 0.0, 1.0);
        vec3 view_ray = normalize(near_clip.xyz / near_clip.w - world_clip.xyz / world_clip.w);
        vec3 half_dir = normalize(view_ray - normalize(light.direction_ambient.xyz));
        float wet_gloss = clamp(surface.r + wetness * 0.35, 0.0, 1.0);
        float specular = pow(max(dot(normal, half_dir), 0.0), mix(4.0, 64.0, wet_gloss));
        lit += specular * clamp(surface.r + wetness * 0.2, 0.0, 1.0) * mix(0.04, 0.25, surface.g) *
            light.color.rgb * sun_scale;
    }
    lit += albedo.rgb * (is_pbr ? surface.a : surface.b);
    vec4 fog_world_clip = shadow.inverse_view_projection *
        vec4(screen_uv.x * 2.0 - 1.0, 1.0 - screen_uv.y * 2.0, depth, 1.0);
    vec3 fog_world = fog_world_clip.xyz / max(abs(fog_world_clip.w), 1e-6) * sign(fog_world_clip.w);
    float fog_visibility = clamp(dot(vec4(fog_world, 1.0), shadow.fog_plane), 0.0, 1.0);
    lit = mix(shadow.fog_color.rgb, lit, fog_visibility);
    lit = (lit - 0.5) * light.grade.z + 0.5;
    lit = pow(max(lit * light.grade.y, 0.0), vec3(1.0 / max(light.grade.x, 0.01)));
    output_color = vec4(lit, albedo.a);
}

void main()
{
#ifdef GBUFFER_MSAA
    int samples = textureSamples(albedo_buffer);
    vec4 resolved = vec4(0.0);
    for (int sample_index = 0; sample_index < samples; ++sample_index)
    {
        vec4 sample_color;
        shade_deferred_sample(sample_index, sample_color);
        resolved += sample_color;
    }
    pixel_color = resolved / float(max(samples, 1));
#else
    shade_deferred_sample(0, pixel_color);
#endif
}
