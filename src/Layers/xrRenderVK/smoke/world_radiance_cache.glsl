const uint WORLD_CACHE_EMPTY = 0u;
const uint WORLD_CACHE_WRITING = 1u;
const uint WORLD_CACHE_VALID = 2u;
const uint WORLD_CACHE_UPDATING = 3u;
const uint WORLD_CACHE_PROBES = 8u;
const uint WORLD_CACHE_CELL_TTL = 18000u;
const uint WORLD_CACHE_FINE_LEVEL = 0u;
const uint WORLD_CACHE_COARSE_LEVEL = 1u;
const float WORLD_CACHE_FINE_CELL_SIZE = 4.0;
const float WORLD_CACHE_COARSE_CELL_SIZE = 16.0;

struct WorldRadianceCacheCell
{
    uint state;
    int cell_x;
    int cell_y;
    int cell_z;
    uint normal_class;
    uint sample_count;
    uint last_frame;
    uint cell_level;
    float radiance_r;
    float radiance_g;
    float radiance_b;
    float radiance_padding;
    float normal_x;
    float normal_y;
    float normal_z;
    float normal_padding;
    float direction_r_x;
    float direction_r_y;
    float direction_r_z;
    float direction_r_padding;
    float direction_g_x;
    float direction_g_y;
    float direction_g_z;
    float direction_g_padding;
    float direction_b_x;
    float direction_b_y;
    float direction_b_z;
    float direction_b_padding;
    float second_order_r[5];
    float second_order_g[5];
    float second_order_b[5];
};

#ifndef WORLD_CACHE_BINDING
#define WORLD_CACHE_BINDING 7
#endif
layout(set = 0, binding = WORLD_CACHE_BINDING, std430) readonly buffer WorldRadianceCache
{
    WorldRadianceCacheCell cells[];
} world_cache;

uint world_cache_normal_class(vec3 normal)
{
    vec3 a = abs(normal);
    if (a.x >= a.y && a.x >= a.z) return normal.x >= 0.0 ? 0u : 1u;
    if (a.y >= a.z) return normal.y >= 0.0 ? 2u : 3u;
    return normal.z >= 0.0 ? 4u : 5u;
}

uint world_cache_hash(ivec3 cell, uint normal_class, uint level)
{
    uint hash = uint(cell.x) * 0x8da6b343u ^ uint(cell.y) * 0xd8163841u ^
        uint(cell.z) * 0xcb1ab31fu ^ (normal_class + 1u) * 0x9e3779b9u ^
        (level + 1u) * 0x85ebca6bu;
    hash ^= hash >> 16;
    hash *= 0x7feb352du;
    hash ^= hash >> 15;
    hash *= 0x846ca68bu;
    return hash ^ (hash >> 16);
}

bool world_cache_lookup_cell(ivec3 cell, uint normal_class, uint level, vec3 receiver_position,
    vec3 receiver_normal,
    uint frame, out vec3 radiance, out float normal_weight)
{
    radiance = vec3(0.0);
    normal_weight = 0.0;
    uint capacity = uint(world_cache.cells.length());
    if (capacity == 0u) return false;
    uint start = world_cache_hash(cell, normal_class, level) % capacity;
    for (uint probe = 0u; probe < min(WORLD_CACHE_PROBES, capacity); ++probe)
    {
        WorldRadianceCacheCell entry = world_cache.cells[(start + probe) % capacity];
        if (entry.state == WORLD_CACHE_EMPTY) return false;
        if (entry.state != WORLD_CACHE_VALID || entry.cell_x != cell.x || entry.cell_y != cell.y ||
            entry.cell_z != cell.z || entry.normal_class != normal_class || entry.cell_level != level ||
            entry.sample_count == 0u ||
            frame - entry.last_frame > WORLD_CACHE_CELL_TTL)
            continue;
        vec3 cached_normal = vec3(entry.normal_x, entry.normal_y, entry.normal_z);
        float normal_alignment = dot(cached_normal, receiver_normal);
        if (normal_alignment < 0.35) continue;
        float cell_size = level == WORLD_CACHE_FINE_LEVEL ? WORLD_CACHE_FINE_CELL_SIZE :
            WORLD_CACHE_COARSE_CELL_SIZE;
        vec3 cell_center = (vec3(cell) + vec3(0.5)) * cell_size;
        vec3 to_receiver = receiver_position - cell_center;
        float direction_length2 = dot(to_receiver, to_receiver);
        vec3 direction = direction_length2 > 1e-8 ? to_receiver * inversesqrt(direction_length2) : receiver_normal;
        vec3 base = max(vec3(entry.radiance_r, entry.radiance_g, entry.radiance_b), vec3(0.0));
        vec3 directional = vec3(
            dot(vec3(entry.direction_r_x, entry.direction_r_y, entry.direction_r_z), direction),
            dot(vec3(entry.direction_g_x, entry.direction_g_y, entry.direction_g_z), direction),
            dot(vec3(entry.direction_b_x, entry.direction_b_y, entry.direction_b_z), direction));
        float second_order_basis[5] = float[5](
            2.0 * direction.x * direction.y,
            2.0 * direction.x * direction.z,
            2.0 * direction.y * direction.z,
            direction.x * direction.x - direction.z * direction.z,
            direction.y * direction.y - direction.z * direction.z);
        vec3 second_order = vec3(0.0);
        for (uint coefficient = 0u; coefficient < 5u; ++coefficient)
            second_order += vec3(entry.second_order_r[coefficient],
                entry.second_order_g[coefficient], entry.second_order_b[coefficient]) *
                second_order_basis[coefficient];
        radiance = max(base + directional + second_order * 2.0, vec3(0.0));
        normal_weight = smoothstep(0.35, 0.85, normal_alignment);
        return true;
    }
    return false;
}

vec3 world_cache_sample(vec3 position, vec3 normal, vec3 receiver_position, uint frame)
{
    float normal_length2 = dot(normal, normal);
    if (!(normal_length2 > 1e-10) || isnan(normal_length2) || isinf(normal_length2))
        return vec3(0.0);
    if (any(isnan(position)) || any(isinf(position)) ||
        any(isnan(receiver_position)) || any(isinf(receiver_position))) return vec3(0.0);
    normal *= inversesqrt(normal_length2);
    uint normal_class = world_cache_normal_class(normal);
    vec3 fine_sum = vec3(0.0);
    float fine_weight_sum = 0.0;
    vec3 coarse_sum = vec3(0.0);
    float coarse_weight_sum = 0.0;
    for (uint level = WORLD_CACHE_FINE_LEVEL; level <= WORLD_CACHE_COARSE_LEVEL; ++level)
    {
        float cell_size = level == WORLD_CACHE_FINE_LEVEL ? WORLD_CACHE_FINE_CELL_SIZE :
            WORLD_CACHE_COARSE_CELL_SIZE;
        // The update pass hashes the unshifted surface position. Interpolate around
        // cell centers so that the cell containing the surface always contributes.
        vec3 coordinates = position / cell_size - vec3(0.5);
        vec3 base = floor(coordinates);
        if (any(lessThan(base, vec3(-2147483648.0))) ||
            any(greaterThanEqual(base, vec3(2147483648.0)))) continue;
        vec3 fraction = fract(coordinates);
        for (int i = 0; i < 8; ++i)
        {
            ivec3 offset = ivec3(i & 1, (i >> 1) & 1, (i >> 2) & 1);
            vec3 axis_weights = mix(vec3(1.0) - fraction, fraction, vec3(offset));
            float spatial_weight = axis_weights.x * axis_weights.y * axis_weights.z;
            if (spatial_weight <= 0.0) continue;
            ivec3 cell = ivec3(base) + offset;
            vec3 sample_radiance;
            float normal_weight;
            if (world_cache_lookup_cell(cell, normal_class, level, receiver_position, normal, frame,
                    sample_radiance, normal_weight))
            {
                float weight = spatial_weight * normal_weight;
                if (level == WORLD_CACHE_FINE_LEVEL)
                {
                    fine_sum += min(sample_radiance, vec3(32.0)) * weight;
                    fine_weight_sum += weight;
                }
                else
                {
                    coarse_sum += min(sample_radiance, vec3(32.0)) * weight;
                    coarse_weight_sum += weight;
                }
            }
        }
    }
    // Prefer local detail; use broad cells only when the fine level has too
    // little coherent history to provide a stable estimate.
    if (fine_weight_sum >= 0.5 || coarse_weight_sum <= 0.0)
        return fine_weight_sum > 0.0 ? fine_sum / fine_weight_sum : vec3(0.0);
    if (fine_weight_sum <= 0.0)
        return coarse_sum / coarse_weight_sum;
    return mix(coarse_sum / coarse_weight_sum, fine_sum / fine_weight_sum,
        smoothstep(0.0, 0.5, fine_weight_sum));
}
