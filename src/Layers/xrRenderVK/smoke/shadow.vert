#version 450
layout(push_constant) uniform Camera {
    mat4 mvp;
    layout(offset = 136) float wind_x;
    layout(offset = 152) float wind_z;
    layout(offset = 240) vec4 wave;
} camera;
layout(location = 0) in vec3 position;
layout(location = 2) in vec2 uv;
layout(location = 11) in vec2 tree_wind;
layout(location = 0) out vec2 texcoord;
void main()
{
    float phase = fract(camera.wave.w + dot(position, camera.wave.xyz));
    float cyclic = 2.8284271 * phase - 1.4142136;
    float bend = (position.y - tree_wind.y) * (cyclic * cyclic - 1.0) * tree_wind.x;
    vec3 displaced = position + vec3(camera.wind_x * bend, 0.0, camera.wind_z * bend);
    gl_Position = camera.mvp * vec4(displaced, 1.0);
    texcoord = uv;
}
