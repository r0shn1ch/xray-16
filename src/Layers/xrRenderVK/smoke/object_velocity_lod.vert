#version 450
layout(location = 0) in vec3 first_position;
layout(location = 1) in vec3 second_position;
layout(location = 2) in vec2 uv0;
layout(location = 3) in vec2 uv1;
layout(location = 0) out vec4 previous_clip;
layout(location = 1) out vec4 current_clip;
layout(location = 2) out vec2 out_uv0;
layout(location = 3) out vec2 out_uv1;
layout(push_constant) uniform Matrices
{
    mat4 current_mvp;
    mat4 previous_mvp;
    layout(offset = 128) float alpha_ref;
    float blend;
} transforms;

void main()
{
    vec4 position = vec4(mix(second_position, first_position, transforms.blend), 1.0);
    gl_Position = transforms.current_mvp * position;
    gl_Position.y = -gl_Position.y;
    current_clip = gl_Position;
    previous_clip = transforms.previous_mvp * position;
    previous_clip.y = -previous_clip.y;
    out_uv0 = uv0;
    out_uv1 = uv1;
}
