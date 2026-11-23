#version 450
#extension GL_GOOGLE_include_directive : require
layout(set = 0, binding = 0) uniform texture2D frame_image;
layout(set = 0, binding = 1) uniform sampler frame_sampler;
layout(push_constant) uniform Settings {
    vec4 grade;
    vec4 tint;
    vec4 add;
    vec4 gray_weights;
    vec4 effect;
    vec4 noise;
    vec4 color_map;
    vec4 bloom;
} settings;
layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 pixel_color;

#include "bloom.glsl"

vec3 sample_source(vec2 uv)
{
    return textureLod(sampler2D(frame_image, frame_sampler), uv, 0.0).rgb;
}

void main()
{
    vec2 step_uv = bloom_upsample_radius(settings.bloom.z) /
        vec2(textureSize(sampler2D(frame_image, frame_sampler), 0));
    vec3 color = sample_source(texcoord) * 4.0;
    color += sample_source(texcoord + vec2(step_uv.x, 0.0)) * 2.0;
    color += sample_source(texcoord - vec2(step_uv.x, 0.0)) * 2.0;
    color += sample_source(texcoord + vec2(0.0, step_uv.y)) * 2.0;
    color += sample_source(texcoord - vec2(0.0, step_uv.y)) * 2.0;
    if (settings.bloom.w < 0.0)
    {
        pixel_color = vec4(color * (1.0 / 12.0), 0.0);
        return;
    }
    color += sample_source(texcoord + step_uv);
    color += sample_source(texcoord - step_uv);
    color += sample_source(texcoord + vec2(step_uv.x, -step_uv.y));
    color += sample_source(texcoord + vec2(-step_uv.x, step_uv.y));
    pixel_color = vec4(color * (1.0 / 16.0), 0.0);
}
