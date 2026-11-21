#ifndef XRAY_VISIBILITY_GEOMETRY_GLSL
#define XRAY_VISIBILITY_GEOMETRY_GLSL

struct DrawEntry
{
    uint firstIndex;
    uint indexCount;
    int vertexOffset;
    uint materialSlot;
    uint geometryBatch;
    uint lightmapSlot;
};

// Matches the static LevelVertex stream, including attributes unused here.
struct PackedVertex
{
    vec4 positionNormalX;
    vec4 normalYZUv;
    vec4 lightmapUvBakedRG;
    uvec4 bakedBATangentBitangent;
};

layout(std430, binding = 1) readonly buffer DrawEntries { DrawEntry entries[]; };
layout(std430, binding = 2) readonly buffer Indices { uint indices[]; };
layout(std430, binding = 3) readonly buffer Vertices { PackedVertex vertices[]; };

vec4 visibilityClipPosition(PackedVertex vertex, mat4 viewProjection)
{
    vec4 clip = viewProjection * vec4(vertex.positionNormalX.xyz, 1.0);
    clip.y = -clip.y;
    return clip;
}

vec3 visibilityNormal(PackedVertex vertex)
{
    return vec3(vertex.positionNormalX.w, vertex.normalYZUv.xy);
}

bool visibilityVertices(DrawEntry entry, uint triangle, out uvec3 vertexIndex)
{
    uint index = triangle * 3u;
    uint capacity = uint(indices.length());
    if (index + 3u > entry.indexCount || entry.firstIndex > capacity || index + 3u > capacity - entry.firstIndex)
        return false;
    uint start = entry.firstIndex + index;
    vertexIndex = uvec3(indices[start], indices[start + 1u], indices[start + 2u]) + uint(entry.vertexOffset);
    return all(lessThan(vertexIndex, uvec3(vertices.length())));
}

vec2 visibilityScreenPosition(vec4 clip, vec2 origin, vec2 extent)
{
    return origin + (clip.xy / clip.w * 0.5 + 0.5) * extent;
}

#endif
