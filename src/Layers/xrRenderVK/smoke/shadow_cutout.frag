#version 450
layout(set = 0, binding = 0) uniform sampler2D diffuse;
layout(push_constant) uniform MaterialCutout {
    mat4 mvp;
    float alpha_ref;
    layout(offset = 128) vec4 base_u;
    vec4 base_v;
} material;
layout(location = 0) in vec2 texcoord;
void main()
{
    vec4 coordinate = vec4(texcoord, 0.0, 1.0);
    vec2 uv = vec2(dot(coordinate, material.base_u), dot(coordinate, material.base_v));
    if (texture(diffuse, uv).a < material.alpha_ref) discard;
}
