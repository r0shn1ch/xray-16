// Packed PBR materials reserve a narrow gap between legacy lightmap and
// dynamic hemisphere codes in the nearest-filtered G-buffer normal alpha.
bool pbr_material(float encoded_hemi)
{
    return encoded_hemi > 0.755 && encoded_hemi < 0.874;
}
bool pbr_static_lightmap(float encoded_hemi)
{
    return encoded_hemi < 0.815;
}
float pbr_hemisphere(float encoded_hemi)
{
    return clamp((encoded_hemi - (pbr_static_lightmap(encoded_hemi) ? 0.76 : 0.82)) / 0.05, 0.0, 1.0);
}
vec3 pbr_environment_brdf(vec3 albedo, float metallic, float roughness, float ndotv)
{
    vec3 f0 = mix(vec3(0.04), albedo, clamp(metallic, 0.0, 1.0));
    vec4 r = clamp(roughness, 0.04, 1.0) * vec4(-1.0, -0.0275, -0.572, 0.022) +
        vec4(1.0, 0.0425, 1.04, -0.04);
    float a004 = min(r.x * r.x, exp2(-9.28 * clamp(ndotv, 0.0, 1.0))) * r.x + r.y;
    vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
    return max(f0 * ab.x + ab.y, vec3(0.0));
}
vec3 pbr_direct(vec3 albedo, float metallic, float roughness, vec3 normal, vec3 view, vec3 light)
{
    const float pi = 3.14159265359;
    metallic = clamp(metallic, 0.0, 1.0);
    roughness = clamp(roughness, 0.04, 1.0);
    float nl = max(dot(normal, light), 0.0);
    float nv = max(dot(normal, view), 0.0);
    vec3 half_vector = view + light;
    float half_length = length(half_vector);
    if (nl <= 0.0 || nv <= 0.0 || half_length < 1e-6) return vec3(0.0);
    vec3 h = half_vector / half_length;
    float nh = max(dot(normal, h), 0.0);
    float vh = max(dot(view, h), 0.0);
    float a = roughness * roughness;
    float a2 = a * a;
    float denominator = max(1.0 - nh * nh, 0.0) + nh * nh * a2;
    float distribution = a2 / max(pi * denominator * denominator, 1e-12);
    float k = (roughness + 1.0) * (roughness + 1.0) * 0.125;
    float geometry = nv / max(nv * (1.0 - k) + k, 1e-6) *
        nl / max(nl * (1.0 - k) + k, 1e-6);
    vec3 f0 = mix(vec3(0.04), albedo, metallic);
    vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - vh, 5.0);
    vec3 specular = distribution * geometry * fresnel / (4.0 * max(nv, 0.001) * max(nl, 0.001));
    vec3 diffuse = (1.0 - fresnel) * (1.0 - metallic) * albedo / pi;
    return (diffuse + specular) * nl;
}
