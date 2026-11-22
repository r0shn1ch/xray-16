#include "src/Layers/xrRenderVK/GpuParticleCollisionData.h"
#include "src/Layers/xrRenderVK/GpuParticleStaticCollision.h"
#include "src/Layers/xrRenderVK/GpuParticleDiagnostics.h"
#include <cassert>
#include <cstring>
#include <limits>

using namespace xray::render::vulkan;
int main()
{
    GpuParticleDiagnostics diagnostics;
    assert(diagnostics.should_report(1, 100));
    assert(!diagnostics.should_report(1, 101));
    assert(diagnostics.should_report(3, 102));
    assert(!diagnostics.should_report(0, 5101));
    assert(diagnostics.should_report(3, 5102));
    diagnostics.reset();
    assert(diagnostics.should_report(4, 0xfffffff0u));
    assert(!diagnostics.should_report(4, 0x20u));
    assert(diagnostics.should_report(4, 0x1380u));
    GpuParticleCollisionInput input;
    input.nodes.resize(1); input.nodes[0].escape = 1; input.nodes[0].count = 1;
    input.static_triangles.resize(1); input.static_triangles[0].a[0] = 3.f;
    input.dynamic_triangles.resize(2); input.dynamic_triangles[1].b[1] = 7.f;
    input.shapes.resize(1); input.shapes[0].triangle_first = 1; input.shapes[0].triangle_count = 1;
    input.objects.resize(1); input.objects[0].count = 1;
    input.blocks.resize(1); input.blocks[0].count = 1;
    std::vector<uint8_t> bytes;
    std::string error;
    assert(pack_gpu_particle_collision(input, 4096, bytes, error));
    uint32_t counts[4]{}, offsets[6]{};
    std::memcpy(counts, bytes.data(), sizeof(counts));
    std::memcpy(offsets, bytes.data() + 16, sizeof(offsets));
    assert(counts[0] == 1 && counts[1] == 1 && counts[2] == 1 && counts[3] == 2);
    assert(offsets[0] == 48 && offsets[1] == 96 && offsets[2] == 144 && offsets[3] == 240 &&
        offsets[4] == 336 && offsets[5] == 432 && bytes.size() == 480);
    GpuParticleCollisionTriangle triangle{};
    std::memcpy(&triangle, bytes.data() + offsets[2] + sizeof(triangle), sizeof(triangle));
    assert(triangle.b[1] == 7.f);
    const auto original = bytes;
    input.nodes[0].escape = 0;
    assert(!pack_gpu_particle_collision(input, 4096, bytes, error) && bytes == original);
    input.nodes[0].escape = 1;
    input.shapes[0].triangle_count = 2;
    assert(!pack_gpu_particle_collision(input, 4096, bytes, error) && bytes == original);
    input.shapes[0].triangle_count = 1;
    assert(!pack_gpu_particle_collision(input, 479, bytes, error) && bytes == original);
    input.blocks[0].count = 0;
    assert(!pack_gpu_particle_collision(input, 4096, bytes, error) && bytes == original);
    std::vector<GpuParticleCollisionTriangle> geometry(37);
    for (size_t i = 0; i < geometry.size(); ++i)
    {
        geometry[i].a[0] = geometry[i].b[0] = geometry[i].c[0] = static_cast<float>(i);
        geometry[i].b[1] = 1.f; geometry[i].c[2] = 1.f;
    }
    GpuParticleCollisionInput tree;
    tree.dynamic_triangles.resize(1);
    assert(build_gpu_particle_static_collision(geometry, tree, error));
    assert(tree.static_triangles.size() == 37 && tree.nodes.size() > 1 && tree.nodes[0].escape == tree.nodes.size() &&
        tree.dynamic_triangles.size() == 1);
    std::vector<unsigned> seen(37);
    for (size_t n = 0; n < tree.nodes.size(); ++n)
    {
        const auto& node = tree.nodes[n];
        assert(node.escape > n && node.escape <= tree.nodes.size());
        if (!node.count) { assert(n + 1 < node.escape); continue; }
        assert(node.count <= 8 && node.first + node.count <= tree.static_triangles.size());
        for (size_t i = node.first; i < node.first + node.count; ++i)
        {
            const auto& t = tree.static_triangles[i];
            ++seen[static_cast<size_t>(t.a[0])];
            for (const float* point : {t.a, t.b, t.c})
                for (unsigned axis = 0; axis < 3; ++axis)
                    assert(point[axis] >= node.minimum[axis] && point[axis] <= node.maximum[axis]);
        }
    }
    for (unsigned count : seen) assert(count == 1);
    assert(pack_gpu_particle_collision(tree, 100000, bytes, error));
    const auto previous = tree.static_triangles.size();
    geometry[0].a[1] = std::numeric_limits<float>::quiet_NaN();
    assert(!build_gpu_particle_static_collision(geometry, tree, error) && tree.static_triangles.size() == previous);
    assert(build_gpu_particle_static_collision({}, tree, error) && tree.nodes.empty() && tree.static_triangles.empty());
}
