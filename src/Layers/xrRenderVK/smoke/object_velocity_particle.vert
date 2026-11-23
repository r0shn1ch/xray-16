#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 previous_position;
layout(location = 2) in vec2 texcoord;
layout(location = 4) in vec4 baked;
layout(location = 5) in float previous_valid;
layout(location = 0) out vec4 previous_clip;
layout(location = 1) out vec4 current_clip;
layout(location = 2) out vec2 uv;
layout(location = 3) out float particle_alpha;
layout(location = 4) out float has_previous;
layout(push_constant) uniform Matrices
{
    mat4 current_mvp;
    mat4 previous_mvp;
} transforms;

void main()
{
    gl_Position = transforms.current_mvp * vec4(position, 1.0);
    gl_Position.y = -gl_Position.y;
    current_clip = gl_Position;
    previous_clip = transforms.previous_mvp * vec4(previous_position, 1.0);
    previous_clip.y = -previous_clip.y;
    uv = texcoord;
    particle_alpha = baked.a;
    has_previous = previous_valid;
}
