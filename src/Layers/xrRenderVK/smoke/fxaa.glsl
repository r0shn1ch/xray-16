// Compact GLSL quality-path port of the FXAA edge/span and subpixel logic
// already shipped in res/gamedata/shaders/gl/fxaa.ps. Use RGB luminance
// instead of its green-only mode. aa_sample supplies bilinear mip-zero reads.
// This file is also executed by the software scene regression test.
float aa_luma(vec3 color)
{
    // Edge thresholds operate on perceptual brightness. The source is
    // linear, and may contain highlights above reference white.
    return sqrt(max(dot(color, vec3(0.2126f, 0.7152f, 0.0722f)), 0.0f));
}

vec3 fxaa(vec2 uv, vec2 pixel, float enabled)
{
    vec3 center = aa_sample(uv);
    if (enabled < 0.5f)
        return center;
    float lm = aa_luma(center);
    float ln = aa_luma(aa_sample(uv + vec2(0.0f, -pixel.y)));
    float ls = aa_luma(aa_sample(uv + vec2(0.0f, pixel.y)));
    float lw = aa_luma(aa_sample(uv + vec2(-pixel.x, 0.0f)));
    float le = aa_luma(aa_sample(uv + vec2(pixel.x, 0.0f)));
    float low = min(lm, min(min(ln, ls), min(lw, le)));
    float high = max(lm, max(max(ln, ls), max(lw, le)));
    float range = high - low;
    if (range < max(0.0312f, high * 0.125f))
        return center;
    float lnw = aa_luma(aa_sample(uv - pixel));
    float lne = aa_luma(aa_sample(uv + vec2(pixel.x, -pixel.y)));
    float lsw = aa_luma(aa_sample(uv + vec2(-pixel.x, pixel.y)));
    float lse = aa_luma(aa_sample(uv + pixel));
    float horizontal = 2.0f * abs(ln + ls - 2.0f * lm) +
        abs(lne + lse - 2.0f * le) + abs(lnw + lsw - 2.0f * lw);
    float vertical = 2.0f * abs(lw + le - 2.0f * lm) +
        abs(lnw + lne - 2.0f * ln) + abs(lsw + lse - 2.0f * ls);
    bool along_x = horizontal >= vertical;
    float negative = along_x ? ln : lw;
    float positive = along_x ? ls : le;
    bool pair_negative = abs(negative - lm) >= abs(positive - lm);
    float neighbor = pair_negative ? negative : positive;
    float gradient = abs(neighbor - lm) * 0.25f;
    float average = 0.5f * (neighbor + lm);
    vec2 across = along_x ? vec2(0.0f, pixel.y) : vec2(pixel.x, 0.0f);
    if (pair_negative) across = -across;
    vec2 along = along_x ? vec2(pixel.x, 0.0f) : vec2(0.0f, pixel.y);
    vec2 lower = pixel * 0.5f;
    vec2 upper = vec2(1.0f) - lower;
    vec2 origin = clamp(uv + across * 0.5f, lower, upper);
    vec2 end_negative = max(origin - along, lower);
    vec2 end_positive = min(origin + along, upper);
    float delta_negative = aa_luma(aa_sample(end_negative)) - average;
    float delta_positive = aa_luma(aa_sample(end_positive)) - average;
    int axis = along_x ? 0 : 1;
    bool done_negative = abs(delta_negative) >= gradient || end_negative[axis] <= lower[axis];
    bool done_positive = abs(delta_positive) >= gradient || end_positive[axis] >= upper[axis];
    // Bounded span search: no dynamic allocation or shader variants.
    for (int step = 0; step < 6; ++step)
    {
        if (done_negative && done_positive) break;
        float distance_step = step == 0 ? 1.0f : step == 1 ? 1.5f :
            step < 4 ? 2.0f : step == 4 ? 4.0f : 8.0f;
        if (!done_negative)
        {
            end_negative = max(end_negative - along * distance_step, lower);
            delta_negative = aa_luma(aa_sample(end_negative)) - average;
            done_negative = abs(delta_negative) >= gradient || end_negative[axis] <= lower[axis];
        }
        if (!done_positive)
        {
            end_positive = min(end_positive + along * distance_step, upper);
            delta_positive = aa_luma(aa_sample(end_positive)) - average;
            done_positive = abs(delta_positive) >= gradient || end_positive[axis] >= upper[axis];
        }
    }
    float dn = origin[axis] - end_negative[axis];
    float dp = end_positive[axis] - origin[axis];
    float closest_delta = dn < dp ? delta_negative : delta_positive;
    bool valid_span = abs(closest_delta) >= gradient &&
        ((closest_delta < 0.0f) != (lm - average < 0.0f));
    float edge_offset = valid_span ? 0.5f - min(dn, dp) / max(dn + dp, 1e-6f) : 0.0f;
    float subpixel = abs((2.0f * (ln + ls + lw + le) + lnw + lne + lsw + lse) / 12.0f - lm) / range;
    subpixel = clamp(subpixel, 0.0f, 1.0f);
    subpixel = subpixel * subpixel * (3.0f - 2.0f * subpixel);
    subpixel = subpixel * subpixel * 0.75f;
    return aa_sample(uv + across * max(edge_offset, subpixel));
}
