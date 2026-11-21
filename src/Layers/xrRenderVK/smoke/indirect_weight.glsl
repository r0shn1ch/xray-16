// Compare distance to the receiver's tangent plane in world units. Unlike
// device-depth differences this is stable across near/far planes and preserves
// neighbors on an oblique surface.
float indirect_plane_weight(vec3 center, vec3 neighbor, vec3 center_normal)
{
    float separation = abs(dot(neighbor - center, center_normal));
    return exp(-32.0 * separation);
}
