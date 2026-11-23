#ifndef XRAY_VISIBILITY_PARALLAX_GLSL
#define XRAY_VISIBILITY_PARALLAX_GLSL

vec3 finiteDirection(vec3 value, vec3 fallback)
{
    float squaredLength = dot(value, value);
    return squaredLength > 1e-12 && squaredLength < 1e20 ? value * inversesqrt(squaredLength) : fallback;
}

vec2 parallaxUv(vec2 uv, vec2 dx, vec2 dy, uint slot, float heightScale,
    vec3 position, vec3 geometric, vec3 inputTangent, vec3 inputBitangent, vec4 eyeAndSteep)
{
    vec3 tangent = finiteDirection(inputTangent - geometric * dot(geometric, inputTangent), vec3(1, 0, 0));
    vec3 bitangent = finiteDirection(inputBitangent - geometric * dot(geometric, inputBitangent) -
        tangent * dot(tangent, inputBitangent), vec3(0, 0, 1));
    vec3 view = finiteDirection(eyeAndSteep.xyz - position, geometric);
    vec3 tangentView = vec3(dot(view, tangent), dot(view, bitangent), dot(view, geometric));
    float scale = max(heightScale, 0.0) * (1.0 - smoothstep(32.0, 96.0, length(eyeAndSteep.xyz - position)));
    if (eyeAndSteep.w < 0.5)
    {
        float height = textureGrad(baseColorTextures[nonuniformEXT(slot)], uv, dx, dy).r;
        return uv + finiteDirection(tangentView, vec3(0, 0, 1)).xy * (height - 0.5) * scale;
    }
    if (scale <= 1e-5) return uv;
    int count = int(mix(16.0, 8.0, clamp(abs(tangentView.z), 0.0, 1.0)));
    float layer = 1.0 / float(count);
    vec2 stepUv = tangentView.xy / max(abs(tangentView.z), 0.15) * scale / float(count);
    vec2 current = uv;
    float currentDepth = 0.0;
    float sampledDepth = 1.0 - textureGrad(baseColorTextures[nonuniformEXT(slot)], current, dx, dy).r;
    for (int step = 0; step < 16 && step < count && currentDepth < sampledDepth; ++step)
    {
        current -= stepUv;
        currentDepth += layer;
        sampledDepth = 1.0 - textureGrad(baseColorTextures[nonuniformEXT(slot)], current, dx, dy).r;
    }
    if (currentDepth <= 0.0) return uv;
    vec2 previous = current + stepUv;
    float before = 1.0 - textureGrad(baseColorTextures[nonuniformEXT(slot)], previous, dx, dy).r - (currentDepth - layer);
    float after = sampledDepth - currentDepth;
    return mix(current, previous, clamp(after / max(after - before, 1e-5), 0.0, 1.0));
}

bool visibilityMaterialUv(PackedVertex a, PackedVertex b, PackedVertex c, DrawEntry entry,
    VisibilityBarycentrics barycentrics, ivec2 pixel, vec2 ndc, vec2 ndcPerPixel,
    mat4 viewProjection, vec4 eyeAndSteep, uint materialCount, inout vec2 uv, inout vec2 dx, inout vec2 dy)
{
    if (entry.parallaxSlot >= materialCount) return true;
    mat3x2 texcoords = mat3x2(a.normalYZUv.zw, b.normalYZUv.zw, c.normalYZUv.zw);
    mat3 normals = mat3(finiteDirection(visibilityNormal(a), vec3(0,1,0)),
        finiteDirection(visibilityNormal(b), vec3(0,1,0)), finiteDirection(visibilityNormal(c), vec3(0,1,0)));
    mat3 tangents = mat3(unpackSnorm4x8(a.bakedBATangentBitangent.z).xyz,
        unpackSnorm4x8(b.bakedBATangentBitangent.z).xyz, unpackSnorm4x8(c.bakedBATangentBitangent.z).xyz);
    mat3 bitangents = mat3(unpackSnorm4x8(a.bakedBATangentBitangent.w).xyz,
        unpackSnorm4x8(b.bakedBATangentBitangent.w).xyz, unpackSnorm4x8(c.bakedBATangentBitangent.w).xyz);
    mat3 positions = mat3(a.positionNormalX.xyz, b.positionNormalX.xyz, c.positionNormalX.xyz);
    uv = parallaxUv(uv, dx, dy, entry.parallaxSlot, entry.bumpHeight, positions * barycentrics.weights,
        finiteDirection(normals * barycentrics.weights, vec3(0,1,0)), tangents * barycentrics.weights,
        bitangents * barycentrics.weights, eyeAndSteep);
    if (!finiteUv(uv)) return false;
    for (uint axis = 0u; axis < 2u; ++axis)
    {
        float direction = ((axis == 0u ? pixel.x : pixel.y) & 1) == 0 ? 1.0 : -1.0;
        vec2 neighborNdc = ndc + (axis == 0u ? vec2(direction * ndcPerPixel.x, 0.0) :
            vec2(0.0, direction * ndcPerPixel.y));
        VisibilityBarycentrics neighbor;
        if (!resolveVisibilityBarycentrics(visibilityClipPosition(a, viewProjection),
            visibilityClipPosition(b, viewProjection), visibilityClipPosition(c, viewProjection),
            neighborNdc, ndcPerPixel, neighbor)) continue;
        vec2 neighborUv = parallaxUv(visibilityBaseUv(entry, texcoords * neighbor.weights, 1.0),
            visibilityBaseUv(entry, texcoords * neighbor.dx, 0.0), visibilityBaseUv(entry, texcoords * neighbor.dy, 0.0),
            entry.parallaxSlot, entry.bumpHeight, positions * neighbor.weights,
            finiteDirection(normals * neighbor.weights, vec3(0,1,0)), tangents * neighbor.weights,
            bitangents * neighbor.weights, eyeAndSteep);
        if (finiteUv(neighborUv))
        {
            if (axis == 0u) dx = (neighborUv - uv) * direction;
            else dy = (neighborUv - uv) * direction;
        }
    }
    return true;
}
#endif
