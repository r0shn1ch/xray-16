#ifndef XRAY_VISIBILITY_COMMON_GLSL
#define XRAY_VISIBILITY_COMMON_GLSL

// Keep the R32_UINT attachment encoding consistent with xrRenderGraph/VisibilityId.h.
const uint visibilityTriangleBits = 7u;
const uint visibilityTriangleMask = (1u << visibilityTriangleBits) - 1u;

uint packVisibilityId(uint entry, uint triangle)
{
    if (entry >= (0xffffffffu >> visibilityTriangleBits) || triangle > visibilityTriangleMask)
        return 0u;
    return ((entry + 1u) << visibilityTriangleBits) | triangle;
}

bool visibilityIdValid(uint id) { return id > visibilityTriangleMask; }
uint visibilityEntry(uint id) { return (id >> visibilityTriangleBits) - 1u; }
uint visibilityTriangle(uint id) { return id & visibilityTriangleMask; }

float visibilityEdge(vec2 a, vec2 b, vec2 p)
{
    vec2 segment = b - a;
    vec2 offset = p - a;
    return segment.x * offset.y - segment.y * offset.x;
}

struct VisibilityBarycentrics
{
    vec3 weights;
    vec3 dx;
    vec3 dy;
};

// Solve for the original triangle in homogeneous coordinates. A clipped
// triangle still uses the original ID, including when an original w is zero.
bool resolveVisibilityBarycentrics(vec4 a, vec4 b, vec4 c, vec2 ndc, vec2 ndcPerPixel,
    out VisibilityBarycentrics barycentrics)
{
    vec3 x = vec3(a.x, b.x, c.x);
    vec3 y = vec3(a.y, b.y, c.y);
    vec3 w = vec3(a.w, b.w, c.w);
    // Translate to the sample before evaluating the determinants. This avoids
    // subtracting large screen-origin terms for small, off-center triangles.
    x -= ndc.x * w;
    y -= ndc.y * w;
    vec3 q = cross(x, y);
    vec3 qdx = cross(y, w);
    vec3 qdy = cross(w, x);
    float denominator = q.x + q.y + q.z;
    if (!(abs(denominator) > 1e-20))
        return false;
    barycentrics.weights = q / denominator;
    barycentrics.dx = (qdx - barycentrics.weights * (qdx.x + qdx.y + qdx.z)) * ndcPerPixel.x / denominator;
    barycentrics.dy = (qdy - barycentrics.weights * (qdy.x + qdy.y + qdy.z)) * ndcPerPixel.y / denominator;
    return true;
}

#endif
