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
    float materialClass;
    uint pbrSlot;
    uint metallicSlot;
    uint roughnessSlot;
    uint aoSlot;
    uint bumpSlot;
    uint bumpAuxSlot;
    uint normalSlot;
    float bumpHeight;
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
    uint capacity = uint(indices.length());
    if (entry.firstIndex > capacity ||
        triangle >= min(entry.indexCount, capacity - entry.firstIndex) / 3u)
        return false;
    uint start = entry.firstIndex + triangle * 3u;
    uvec3 raw = uvec3(indices[start], indices[start + 1u], indices[start + 2u]);
    vertexIndex = raw + uint(entry.vertexOffset);
    // Signed base-vertex offsets are valid, but a wrapped index must never
    // alias a vertex in another mesh in the same packed geometry buffer.
    if (entry.vertexOffset < 0 ? any(greaterThan(vertexIndex, raw)) : any(lessThan(vertexIndex, raw)))
        return false;
    return all(lessThan(vertexIndex, uvec3(vertices.length())));
}

vec2 visibilityScreenPosition(vec4 clip, vec2 origin, vec2 extent)
{
    return origin + (clip.xy / clip.w * 0.5 + 0.5) * extent;
}

#endif
