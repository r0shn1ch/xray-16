#version 450
#extension GL_GOOGLE_include_directive : require
layout(set = 0, binding = 0) uniform texture2D frame_image;
layout(set = 0, binding = 1) uniform sampler frame_sampler;
layout(set = 0, binding = 2) uniform texture2D color_map_a;
layout(set = 0, binding = 3) uniform texture2D color_map_b;
layout(set = 0, binding = 4) uniform texture2D bloom_image;
layout(set = 0, binding = 5) uniform sampler2D exposure_image;
layout(set = 0, binding = 6) uniform sampler2D depth_image;
layout(set = 0, binding = 7) uniform texture2D motion_image;

layout(push_constant) uniform Settings {
    vec4 grade;
    vec4 tint;
    vec4 add;
    vec4 gray_weights;
    vec4 effect;
    vec4 noise;
    vec4 color_map;
    vec4 bloom;
} settings;
layout(location = 0) in vec2 texcoord;
layout(location = 0) out vec4 pixel_color;

float random_noise(vec2 p)
{
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 sample_source(vec2 uv)
{
    vec2 border = 0.5 / vec2(textureSize(sampler2D(frame_image, frame_sampler), 0));
    return texture(sampler2D(frame_image, frame_sampler), clamp(uv, border, 1.0 - border)).rgb;
}

vec3 aa_sample(vec2 uv)
{
    vec2 border = 0.5 / vec2(textureSize(sampler2D(frame_image, frame_sampler), 0));
    return textureLod(sampler2D(frame_image, frame_sampler), clamp(uv, border, 1.0 - border), 0.0).rgb;
}

#include "fxaa.glsl"

float cubic_weight(float x)
{
    x = abs(x);
    if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
    if (x < 2.0) return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
    return 0.0;
}

vec3 cubic_upscale(vec2 uv)
{
    ivec2 size = textureSize(sampler2D(frame_image, frame_sampler), 0);
    vec2 position = uv * vec2(size) - 0.5;
    ivec2 base = ivec2(floor(position));
    vec2 fraction = fract(position);
    vec3 result = vec3(0.0), minimum_color = vec3(1e20), maximum_color = vec3(-1e20);
    for (int y = -1; y <= 2; ++y)
        for (int x = -1; x <= 2; ++x)
        {
            ivec2 at = clamp(base + ivec2(x, y), ivec2(0), size - 1);
            vec3 value = texelFetch(sampler2D(frame_image, frame_sampler), at, 0).rgb;
            result += value * cubic_weight(float(x) - fraction.x) * cubic_weight(float(y) - fraction.y);
            if (x >= 0 && x <= 1 && y >= 0 && y <= 1)
            {
                minimum_color = min(minimum_color, value);
                maximum_color = max(maximum_color, value);
            }
        }
    return clamp(result, minimum_color, maximum_color);
}

vec3 reconstructed(vec2 uv, vec2 pixel)
{
    float upscale_mode = floor(settings.color_map.w + 0.0001);
    vec3 filtered = fxaa(uv, pixel, mod(upscale_mode, 2.0));
    if (upscale_mode >= 2.0)
        filtered += cubic_upscale(uv) - aa_sample(uv);
    return max(filtered, vec3(0.0));
}

void main()
{
    vec2 pixel = 1.0 / vec2(textureSize(sampler2D(frame_image, frame_sampler), 0));
    // The legacy duality values are normalized screen offsets.
    vec2 dual = settings.effect.yz * 0.5;
    vec3 image = reconstructed(texcoord, pixel);
    if (any(notEqual(dual, vec2(0.0))))
        image = 0.5 * (reconstructed(texcoord + dual, pixel) + reconstructed(texcoord - dual, pixel));
    float motion_blur = clamp(fract(settings.color_map.w) * 2.0, 0.0, 1.0);
    if (motion_blur > 0.001)
    {
        vec2 velocity = texture(sampler2D(motion_image, frame_sampler), texcoord).rg * motion_blur;
        vec2 motion_pixels = velocity / pixel;
        float pixel_length = length(motion_pixels);
        if (pixel_length > 24.0)
        {
            velocity *= 24.0 / pixel_length;
            pixel_length = 24.0;
        }
        if (pixel_length > 0.5)
        {
            float depth = texture(depth_image, texcoord).r;
            vec3 blurred = vec3(0.0);
            float total_weight = 0.0;
            for (int tap = 0; tap < 5; ++tap)
            {
                float position = float(tap) * 0.25 - 0.5;
                vec2 sample_uv = clamp(texcoord + velocity * position,
                    pixel * 0.5, 1.0 - pixel * 0.5);
                float sample_depth = texture(depth_image, sample_uv).r;
                float weight = 1.0 - smoothstep(0.002, 0.03, abs(sample_depth - depth));
                blurred += aa_sample(sample_uv) * weight;
                total_weight += weight;
            }
            if (total_weight > 0.0)
                image = mix(image, blurred / total_weight, motion_blur);
        }
    }
    if (settings.add.w > 0.0)
    {
        float depth = texture(depth_image, texcoord).r;
        // Perspective depth packs most of the playable distance range near 1.
        // Normalize by the focus disparity so far focus distances retain blur.
        float focus_disparity = max(1.0 - settings.tint.w, 0.00001);
        float defocus = abs(depth - settings.tint.w) / focus_disparity;
        float amount = smoothstep(0.03, 0.7, defocus);
        if (settings.bloom.w == 1.0)
        {
            float limit = depth < settings.tint.w ? settings.bloom.y : settings.bloom.z;
            float width = max(abs(limit - settings.tint.w), 0.000001);
            amount = smoothstep(0.0, 1.0, abs(depth - settings.tint.w) / width);
        }
        amount *= settings.add.w;
        if (amount > 0.001)
        {
            const vec2 taps[8] = vec2[8](
                vec2(1.0, 0.0), vec2(0.7071, 0.7071), vec2(0.0, 1.0),
                vec2(-0.7071, 0.7071), vec2(-1.0, 0.0), vec2(-0.7071, -0.7071),
                vec2(0.0, -1.0), vec2(0.7071, -0.7071));
            vec3 blurred = image * 2.0;
            float total = 2.0;
            for (int ring = 0; ring < 2; ++ring)
            {
                float radius = (ring == 0 ? 0.25 : 1.0) * (2.0 + 6.0 * amount);
                float ring_weight = ring == 0 ? 1.0 : 0.5;
                for (int i = 0; i < 8; ++i)
                {
                    vec2 neighbor = clamp(texcoord + taps[i] * pixel * radius,
                        pixel * 0.5, 1.0 - pixel * 0.5);
                    float neighbor_depth = texture(depth_image, neighbor).r;
                    float disparity = max(1.0 - depth, focus_disparity);
                    float weight = ring_weight * (1.0 - smoothstep(0.025, 0.35,
                        abs(neighbor_depth - depth) / disparity));
                    blurred += aa_sample(neighbor) * weight;
                    total += weight;
                }
            }
            image = mix(image, blurred / total, amount);
        }
    }
    if (settings.grade.w > 0.0 && (settings.gray_weights.w > 0.5 || textureQueryLevels(sampler2D(frame_image, frame_sampler)) > 1))
    {
        // The source image's last mip is the measured scene-wide average.
        // Restrict the gain so a bright sky or dark interior cannot turn a
        // single frame into clipped white or amplify dark UI excessively.
        int last_mip = textureQueryLevels(sampler2D(frame_image, frame_sampler)) - 1;
        vec3 average = texelFetch(sampler2D(frame_image, frame_sampler), ivec2(0), last_mip).rgb;
        float luminance = dot(average, vec3(0.2126, 0.7152, 0.0722));
        float exposure = settings.gray_weights.w > 0.5
            ? texelFetch(exposure_image, ivec2(0), 0).r
            : clamp(settings.color_map.z / max(luminance, 0.1), 0.65, 2.0);
        vec3 exposed = max(image * exposure, 0.0);
        const float white_squared = 1.7 * 1.7;
        vec3 mapped = exposed * (1.0 + exposed / white_squared) / (1.0 + exposed);
        image = mix(image, mapped, clamp(settings.grade.w, 0.0, 1.0));
    }
    int mip_levels = textureQueryLevels(sampler2D(bloom_image, frame_sampler));
    if (settings.bloom.x > 0.0 && mip_levels > 0)
    {
        image += textureLod(sampler2D(bloom_image, frame_sampler), texcoord, 0.0).rgb * settings.bloom.x;
    }
    if (settings.effect.x > 0.0)
    {
        vec2 radius = pixel * settings.effect.x * 0.5;
        vec3 blurred = image * 4.0;
        blurred += sample_source(texcoord + vec2(radius.x, 0.0));
        blurred += sample_source(texcoord - vec2(radius.x, 0.0));
        blurred += sample_source(texcoord + vec2(0.0, radius.y));
        blurred += sample_source(texcoord - vec2(0.0, radius.y));
        image = blurred * 0.125;
    }
    if (settings.color_map.x > 0.0)
    {
        float luminance = clamp(dot(image, vec3(0.3333)), 0.001, 0.999);
        vec2 uv = vec2(luminance, 0.5);
        vec3 mapped = mix(texture(sampler2D(color_map_a, frame_sampler), uv).rgb,
                          texture(sampler2D(color_map_b, frame_sampler), uv).rgb,
                          settings.color_map.y);
        image = mix(image, mapped, settings.color_map.x);
    }
    float gray = dot(image, settings.gray_weights.rgb);
    image = mix(image, vec3(gray), settings.effect.w);
    if (settings.noise.x > 0.0)
    {
        float tick = floor(settings.noise.w * max(settings.noise.z, 0.01));
        vec2 cell = floor(texcoord * vec2(textureSize(sampler2D(frame_image, frame_sampler), 0)) /
                          max(settings.noise.y, 1.0));
        float n = random_noise(cell + tick * vec2(13.0, 37.0));
        image *= mix(1.0, n * 2.0, settings.noise.x);
    }
    image = image * settings.tint.rgb + settings.add.rgb;
    image = (image - 0.5) * settings.grade.z + 0.5;
    image = pow(max(image * settings.grade.y, 0.0),
                vec3(1.0 / max(settings.grade.x, 0.01)));
    pixel_color = vec4(image, 1.0);
}
