#version 450
layout(location = 0) in vec4 previous_clip;
layout(location = 1) in vec4 current_clip;
layout(location = 2) in vec2 uv0;
layout(location = 3) in vec2 uv1;
layout(location = 0) out vec4 velocity;
layout(set = 0, binding = 0) uniform sampler2D atlas;
layout(push_constant) uniform LODCutout
{
    layout(offset = 128) float alpha_ref;
    float blend;
} cutout;

void main()
{
    float alpha = mix(texture(atlas, uv1).a, texture(atlas, uv0).a, cutout.blend);
    if (alpha < abs(cutout.alpha_ref)) discard;
    // LOD billboard facets can change as the camera moves. Their previous
    // frame geometry is not retained, so reject temporal color history rather
    // than reprojecting this facet as if it were the same surface. Alpha 1
    // still marks the depth as dynamic for conservative Hi-Z construction.
    if (previous_clip.w <= 1e-6)
    {
        velocity = vec4(0.0, 0.0, -1.0, 1.0);
        return;
    }
    velocity = vec4(0.0, 0.0, -1.0, 1.0);
}
