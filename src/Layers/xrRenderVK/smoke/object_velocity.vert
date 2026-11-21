#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 previous_position;
layout(location = 2) in vec2 uv;
layout(location = 3) in vec2 tree_wind;
layout(location = 0) out vec4 previous_clip;
layout(location = 1) out vec4 current_clip;
layout(location = 2) out vec2 texcoord;
layout(push_constant) uniform Matrices
{
    mat4 current_mvp;
    mat4 previous_mvp;
    layout(offset = 176) vec4 current_wave;
    vec4 previous_wave;
    vec4 directions;
} transforms;
vec3 displace(vec3 p, vec4 wave, vec2 direction)
{
    float phase = fract(wave.w + dot(p, wave.xyz));
    float cyclic = 2.8284271 * phase - 1.4142136;
    float bend = (p.y - tree_wind.y) * (cyclic * cyclic - 1.0) * tree_wind.x;
    return p + vec3(direction.x * bend, 0.0, direction.y * bend);
}
void main()
{
    gl_Position = transforms.current_mvp * vec4(displace(position, transforms.current_wave, transforms.directions.xy), 1.0);
    gl_Position.y = -gl_Position.y;
    current_clip = gl_Position;
    texcoord = uv;
    previous_clip = transforms.previous_mvp * vec4(displace(previous_position, transforms.previous_wave, transforms.directions.zw), 1.0);
    previous_clip.y = -previous_clip.y;
}
