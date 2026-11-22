#version 450
layout(push_constant) uniform Camera { mat4 mvp; } camera;
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec2 uv;
layout(location = 7) in vec3 tangent;
layout(location = 8) in vec3 bitangent;
layout(location = 0) out vec3 world_normal;
layout(location = 1) out vec2 texcoord;
// The same baseline vertex stage feeds the initial forward transparent/HUD
// pipelines. Their material shader also consumes position and the TBN basis.
layout(location = 4) out vec3 world_position;
layout(location = 6) out vec3 world_tangent;
layout(location = 7) out vec3 world_bitangent;
void main()
{
    gl_Position = camera.mvp * vec4(position, 1.0);
    gl_Position.y = -gl_Position.y;
    world_normal = normal;
    texcoord = uv;
    world_position = position;
    world_tangent = tangent;
    world_bitangent = bitangent;
}
