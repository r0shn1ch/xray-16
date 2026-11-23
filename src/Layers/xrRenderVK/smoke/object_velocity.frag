#version 450
layout(location = 0) in vec4 previous_clip;
layout(location = 1) in vec4 current_clip;
layout(location = 2) in vec2 texcoord;
layout(location = 0) out vec4 velocity;
#ifdef XRAY_OBJECT_VELOCITY_CUTOUT
layout(set = 0, binding = 0) uniform sampler2D diffuse;
layout(push_constant) uniform Cutout
{
    layout(offset = 128) float alpha_ref;
    vec4 base_u;
    vec4 base_v;
} cutout;
#endif
void main()
{
    float dynamic_marker = 1.0;
#ifdef XRAY_OBJECT_VELOCITY_TRANSPARENT
    dynamic_marker = 2.0 + gl_FragCoord.z;
#endif
#ifdef XRAY_OBJECT_VELOCITY_CUTOUT
    vec4 coordinate = vec4(texcoord, 0.0, 1.0);
    vec2 base_uv = vec2(dot(coordinate, cutout.base_u), dot(coordinate, cutout.base_v));
    if (texture(diffuse, base_uv).a < cutout.alpha_ref) discard;
#endif
    if (previous_clip.w <= 1e-6)
    {
        velocity = vec4(0.0, 0.0, -1.0, dynamic_marker);
        return;
    }
    vec2 current_uv = current_clip.xy / current_clip.w * 0.5 + 0.5;
    vec2 previous_uv = previous_clip.xy / previous_clip.w * 0.5 + 0.5;
    // Alpha is a conservative dynamic-depth marker consumed when building Hi-Z.
    velocity = vec4(previous_uv - current_uv, previous_clip.z / previous_clip.w, dynamic_marker);
}
