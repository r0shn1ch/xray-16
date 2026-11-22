#version 450
layout(set = 0, binding = 6) uniform sampler2D depth_image;
layout(push_constant) uniform Settings {
    vec4 grade;
    vec4 tint; // screen-space sun position xy, intensity z
    vec4 add;  // sun color rgb
    vec4 gray_weights;
    vec4 effect;
    vec4 noise;
    vec4 color_map;
    vec4 bloom;
} settings;
layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 pixel_color;

const int maximum_samples = 32;

void main()
{
    vec2 sun = settings.tint.xy;
    vec2 delta = sun - texcoord;
    vec2 size = vec2(textureSize(depth_image, 0));
    float distance = length(delta * vec2(size.x / size.y, 1.0));
    vec2 offscreen = max(-sun, sun - 1.0);
    float fade = 1.0 - smoothstep(0.0, 0.5, max(offscreen.x, offscreen.y));
    float falloff = exp(-3.0 * distance) * fade;
    float ray_end = 1.0;
    if (sun.x < 0.0 || sun.x > 1.0)
        ray_end = min(ray_end, (clamp(sun.x, 0.0, 1.0) - texcoord.x) / delta.x);
    if (sun.y < 0.0 || sun.y > 1.0)
        ray_end = min(ray_end, (clamp(sun.y, 0.0, 1.0) - texcoord.y) / delta.y);
    int sample_count = clamp(int(settings.effect.x + 0.5), 1, maximum_samples);
    float visibility = 0.0;
    float weight_sum = 0.0;
    for (int i = 0; i < maximum_samples; ++i)
    {
        if (i >= sample_count)
            break;
        float t = (float(i) + 0.5) / float(sample_count);
        vec2 uv = texcoord + delta * (t * ray_end);
        if (all(greaterThanEqual(uv, vec2(0.0))) && all(lessThanEqual(uv, vec2(1.0))))
        {
            float weight = t * t;
            ivec2 pixel = min(ivec2(uv * size), ivec2(size) - 1);
            // The depth attachment is cleared to 1. Far geometry can be very
            // close to that value and must still occlude the sky.
            float sky = texelFetch(depth_image, pixel, 0).r >= 1.0 ? 1.0 : 0.0;
            visibility += sky * weight;
            weight_sum += weight;
        }
    }
    float shafts = settings.tint.z * falloff * visibility / max(weight_sum, 0.001);
    pixel_color = vec4(settings.add.rgb * shafts, 0.0);
}
