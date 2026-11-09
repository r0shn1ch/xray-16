#version 450
#extension GL_GOOGLE_include_directive : require
#include "../LightingLimits.h"
layout(set = 0, binding = 0) uniform sampler2D albedo_buffer;
layout(set = 0, binding = 1) uniform sampler2D normal_buffer;
layout(set = 0, binding = 2) uniform sampler2D depth_buffer;
layout(set = 0, binding = 5) uniform sampler2D surface_buffer;
struct PointLight
{
    vec4 position_range;
    vec4 color;
    ivec4 rectangle;
};
layout(set = 1, binding = 0, std430) readonly buffer Lights
{
    mat4 inverse_vp;
    uvec4 shape;
    PointLight lights[];
}
data;
layout(set = 1, binding = 1, std430) readonly buffer Tiles
{
    uint values[];
}
tiles;
layout(location = 0) out vec4 color;
void main()
{
    uvec2 pixel = uvec2(gl_FragCoord.xy), tile = pixel / XRAY_VK_LIGHT_TILE_SIZE;
    uint base = (tile.y * data.shape.z + tile.x) * (XRAY_VK_MAX_LOCAL_LIGHTS + 1), count = tiles.values[base];
    if (count == 0)
        discard;
    ivec2 p = ivec2(pixel);
    float depth = texelFetch(depth_buffer, p, 0).r;
    if (depth >= .99999)
        discard;
    vec2 uv = (vec2(pixel) + .5) / vec2(data.shape.xy);
    vec4 clip = vec4(uv.x * 2 - 1, 1 - uv.y * 2, depth, 1);
    vec4 wp = data.inverse_vp * clip;
    vec3 world = wp.xyz / wp.w;
    vec3 normal = normalize(texelFetch(normal_buffer, p, 0).rgb * 2 - 1);
    vec3 albedo = texelFetch(albedo_buffer, p, 0).rgb;
    vec4 surface = texelFetch(surface_buffer, p, 0);
    vec3 view_ray = vec3(0);
    if (surface.r > 0)
    {
        vec4 near_clip = data.inverse_vp * vec4(clip.xy, 0, 1);
        view_ray = normalize(near_clip.xyz / near_clip.w - world);
    }
    vec3 sum = vec3(0);
    for (uint j = 0; j < count; ++j)
    {
        PointLight light = data.lights[tiles.values[base + 1 + j]];
        vec3 delta = world - light.position_range.xyz;
        float d2 = dot(delta, delta), range2 = light.position_range.w * light.position_range.w;
        if (d2 >= range2 || d2 <= 1e-10)
            continue;
        vec3 ray = delta * inversesqrt(d2);
        vec3 value = albedo * max(dot(normal, -ray), 0);
        if (surface.r > 0)
        {
            float spec = pow(max(dot(normal, normalize(view_ray - ray)), 0), mix(4, 64, surface.r));
            value += spec * surface.r * mix(.04, .2, surface.g);
        }
        sum += value * light.color.rgb * max(1 - d2 / range2, 0);
    }
    color = vec4(sum, 0);
}
