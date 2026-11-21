#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace xray::render::vulkan
{
struct GpuParticleCollisionNode
{
    float minimum[3]{}; uint32_t escape{};
    float maximum[3]{}; uint32_t first{};
    uint32_t count{}, padding[3]{};
};
struct GpuParticleCollisionTriangle { float a[4]{}, b[4]{}, c[4]{}; };
struct GpuParticleCollisionShape
{
    float row0[4]{}, row1[4]{}, row2[4]{}, data0[4]{}, data1[4]{};
    uint32_t type{}, triangle_first{}, triangle_count{}, padding{};
};
struct GpuParticleCollisionObject
{
    float spatial_sphere[4]{}, sphere[4]{}, row0[4]{}, row1[4]{}, row2[4]{};
    uint32_t first{}, count{}, mesh{}, padding{};
};
static_assert(sizeof(GpuParticleCollisionNode) == 48 && sizeof(GpuParticleCollisionTriangle) == 48);
static_assert(sizeof(GpuParticleCollisionShape) == 96 && sizeof(GpuParticleCollisionObject) == 96);

struct GpuParticleCollisionInput
{
    std::vector<GpuParticleCollisionNode> nodes, blocks;
    std::vector<GpuParticleCollisionTriangle> static_triangles, dynamic_triangles;
    std::vector<GpuParticleCollisionShape> shapes;
    std::vector<GpuParticleCollisionObject> objects;
};

// Layout matches gpu_particle_collision.h: four counts, six byte offsets,
// padding to 48 bytes, then six contiguous 16-byte-aligned sections.
inline bool pack_gpu_particle_collision(const GpuParticleCollisionInput& input, uint64_t max_bytes,
    std::vector<uint8_t>& output, std::string& error)
{
    const auto fits = [](uint32_t first, uint32_t count, size_t size) {
        return first <= size && count <= size - first;
    };
    const auto fail = [&](const char* message) { error = message; return false; };
    if (input.nodes.size() > UINT32_MAX || input.objects.size() > UINT32_MAX ||
        input.static_triangles.size() > UINT32_MAX || input.dynamic_triangles.size() > UINT32_MAX ||
        input.shapes.size() > UINT32_MAX || input.blocks.size() > UINT32_MAX)
        return fail("GPU particle collision count overflow");
    if (input.blocks.size() != (input.objects.size() + 15) / 16)
        return fail("GPU particle collision block count mismatch");
    for (size_t i = 0; i < input.nodes.size(); ++i)
    {
        const auto& node = input.nodes[i];
        if (node.escape <= i || node.escape > input.nodes.size() ||
            !fits(node.first, node.count, input.static_triangles.size()))
            return fail("invalid GPU particle static collision tree");
    }
    for (size_t i = 0; i < input.blocks.size(); ++i)
    {
        const auto& block = input.blocks[i];
        const size_t expected = i * 16;
        if (block.first != expected || !block.count || block.count > 16 ||
            block.count != std::min<size_t>(16, input.objects.size() - expected))
            return fail("invalid GPU particle collision block");
    }
    for (const auto& object : input.objects)
        if (!fits(object.first, object.count, input.shapes.size()))
            return fail("invalid GPU particle collision object shape range");
    for (const auto& shape : input.shapes)
        if (!fits(shape.triangle_first, shape.triangle_count, input.dynamic_triangles.size()))
            return fail("invalid GPU particle collision shape triangle range");
    const uint64_t lengths[]{uint64_t(input.nodes.size()) * sizeof(GpuParticleCollisionNode),
        uint64_t(input.static_triangles.size()) * sizeof(GpuParticleCollisionTriangle),
        uint64_t(input.dynamic_triangles.size()) * sizeof(GpuParticleCollisionTriangle),
        uint64_t(input.shapes.size()) * sizeof(GpuParticleCollisionShape),
        uint64_t(input.objects.size()) * sizeof(GpuParticleCollisionObject),
        uint64_t(input.blocks.size()) * sizeof(GpuParticleCollisionNode)};
    uint64_t bytes = 48;
    std::array<uint32_t, 6> offsets{};
    for (size_t i = 0; i < offsets.size(); ++i)
    {
        if (bytes > UINT32_MAX || lengths[i] > UINT32_MAX - bytes)
            return fail("GPU particle collision buffer exceeds 32-bit addressing");
        offsets[i] = static_cast<uint32_t>(bytes);
        bytes += lengths[i];
    }
    if (bytes > max_bytes || bytes > std::numeric_limits<size_t>::max())
        return fail("GPU particle collision buffer exceeds device limits");
    std::vector<uint8_t> packed(static_cast<size_t>(bytes), 0);
    const uint32_t counts[]{static_cast<uint32_t>(input.nodes.size()), static_cast<uint32_t>(input.objects.size()),
        static_cast<uint32_t>(input.static_triangles.size()), static_cast<uint32_t>(input.dynamic_triangles.size())};
    std::memcpy(packed.data(), counts, sizeof(counts));
    std::memcpy(packed.data() + 16, offsets.data(), sizeof(offsets));
    const auto copy = [&](size_t section, const auto& values) {
        if (!values.empty()) std::memcpy(packed.data() + offsets[section], values.data(), static_cast<size_t>(lengths[section]));
    };
    copy(0, input.nodes); copy(1, input.static_triangles); copy(2, input.dynamic_triangles);
    copy(3, input.shapes); copy(4, input.objects); copy(5, input.blocks);
    output = std::move(packed); error.clear(); return true;
}
}
