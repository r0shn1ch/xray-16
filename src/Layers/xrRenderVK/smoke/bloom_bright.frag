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

void main()
{
    vec3 scene = texture(sampler2D(frame_image, frame_sampler), texcoord).rgb;
    pixel_color = vec4(bloom_highlights(scene, settings.bloom.y, settings.bloom.w), 1.0);
}
