#version 450
layout(location = 0) in vec4 previous_clip;
layout(location = 1) in vec4 current_clip;
layout(location = 2) in vec2 uv;
layout(location = 3) in float particle_alpha;
layout(location = 4) in float has_previous;
layout(location = 0) out vec4 velocity;
layout(set = 0, binding = 0) uniform sampler2D diffuse;
layout(push_constant) uniform Cutout
{
    layout(offset = 128) float alpha_ref;
    layout(offset = 132) float transparent_mode;
    layout(offset = 144) vec4 base_u;
    layout(offset = 160) vec4 base_v;
} cutout;

void main()
{
    vec4 coordinate = vec4(uv, 0.0, 1.0);
    vec2 base_uv = vec2(dot(coordinate, cutout.base_u), dot(coordinate, cutout.base_v));
    if (texture(diffuse, base_uv).a * particle_alpha < cutout.alpha_ref) discard;
    if (has_previous < 0.0 || previous_clip.w <= 1e-6)
    {
        velocity = vec4(0.0, 0.0, -1.0,
            cutout.transparent_mode > 0.5 ? 2.0 + gl_FragCoord.z : 1.0);
        return;
    }
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    velocity = vec4(previous_uv - current_uv, previous_clip.z / previous_clip.w,
        cutout.transparent_mode > 0.5 ? 2.0 + gl_FragCoord.z : 1.0);
}
