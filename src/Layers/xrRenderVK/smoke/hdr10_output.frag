#version 450

layout(set = 0, binding = 0) uniform texture2D composed_image;
layout(set = 0, binding = 1) uniform sampler composed_sampler;
layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 pixel_color;

layout(push_constant) uniform Calibration
{
    float paper_white_nits;
    float peak_nits;
    float black_nits;
    float test_pattern;
} calibration;

vec3 encode_pq(vec3 nits)
{
    // ITU-R BT.2100 reference PQ inverse EOTF, normalized to 10000 cd/m².
    const float m1 = 2610.0 / 16384.0;
    const float m2 = 2523.0 / 32.0;
    const float c1 = 3424.0 / 4096.0;
    const float c2 = 2413.0 / 128.0;
    const float c3 = 2392.0 / 128.0;
    vec3 power = pow(clamp(nits * 0.0001, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * power) / (1.0 + c3 * power), vec3(m2));
}

void main()
{
    float peak = clamp(calibration.peak_nits, 80.0, 10000.0);
    float white = clamp(calibration.paper_white_nits, 80.0, peak);
    float black = clamp(calibration.black_nits, 0.0, white - 1.0);
    if (calibration.test_pattern > 0.5)
    {
        // Keep the three reference patches above near-black and highlight steps.
        float x = clamp(texcoord.x, 0.0, 0.999999);
        float level;
        if (texcoord.y < 0.5)
            level = x < 1.0 / 3.0 ? black : x < 2.0 / 3.0 ? white : peak;
        else if (texcoord.y < 0.75)
        {
            float step_index = floor(x * 8.0);
            level = min(white, black + step_index * 0.1);
        }
        else
        {
            float step_fraction = floor(x * 8.0) / 7.0;
            level = mix(max(white, peak * 0.8), peak, step_fraction);
        }
        pixel_color = vec4(encode_pq(vec3(level)), 1.0);
        return;
    }
    vec3 composed = texture(sampler2D(composed_image, composed_sampler), texcoord).rgb;
    // Keep invalid HDR highlights from contaminating every channel in PQ.
    vec3 rgb = clamp(mix(composed, vec3(0.0), isnan(composed)), 0.0, 65504.0);
    // Convert the renderer's linear Rec.709 primaries to linear BT.2020.
    const mat3 to_bt2020 = mat3(
        vec3(0.627404, 0.069097, 0.016391),
        vec3(0.329283, 0.919540, 0.088013),
        vec3(0.043313, 0.011363, 0.895596));
    rgb = to_bt2020 * rgb;
    float maximum = max(rgb.r, max(rgb.g, rgb.b));
    // Match reference white at input 1 with a continuous highlight shoulder.
    float headroom = peak - white;
    float white_range = white - black;
    float scaled_max = white_range * maximum;
    if (maximum > 1.0)
    {
        float excess = scaled_max - white_range;
        scaled_max = white_range + headroom * (excess / (headroom + excess));
    }
    vec3 nits = vec3(black) + rgb * (maximum > 0.0 ? scaled_max / maximum : 0.0);
    pixel_color = vec4(encode_pq(nits), 1.0);
}
