#include "xrEngine/stdafx.h"
#include "GpuParticleLevelCollision.h"
#include "GpuParticleDynamicCollision.h"
#include "xrEngine/IGame_Level.h"

namespace xray::render::vulkan
{
bool GpuParticleLevelCollision::update(bool& changed, std::string& error)
{
    auto* level = g_pGameLevel;
    auto* model = level ? level->ObjectSpace.GetStaticModel() : nullptr;
    const CDB::TRI* triangles = model ? model->get_tris() : nullptr;
    const Fvector* vertices = model ? model->get_verts() : nullptr;
    const uint32_t triangle_count = model ? model->get_tris_count() : 0;
    const uint32_t vertex_count = model ? model->get_verts_count() : 0;
    if (level == level_ && model == model_ && triangles == triangles_ && vertices == vertices_ &&
        triangle_count == triangle_count_ && vertex_count == vertex_count_)
    { changed = false; error.clear(); return true; }
    if ((triangle_count && (!triangles || !vertices || !vertex_count)) || triangle_count > UINT32_MAX / 2)
    { error = "invalid GPU particle level collision geometry"; return false; }
    std::vector<GpuParticleCollisionTriangle> converted;
    converted.reserve(triangle_count);
    for (uint32_t i = 0; i < triangle_count; ++i)
    {
        GpuParticleCollisionTriangle out{};
        float* points[]{out.a, out.b, out.c};
        for (unsigned j = 0; j < 3; ++j)
        {
            const uint32_t index = triangles[i].verts[j];
            if (index >= vertex_count) { error = "GPU particle level collision index out of range"; return false; }
            points[j][0] = vertices[index].x; points[j][1] = vertices[index].y; points[j][2] = vertices[index].z;
        }
        converted.push_back(out);
    }
    if (!build_gpu_particle_static_collision(converted, data_, error)) return false;
    level_ = level; model_ = model; triangles_ = triangles; vertices_ = vertices;
    triangle_count_ = triangle_count; vertex_count_ = vertex_count;
    changed = true; error.clear(); return true;
}
void GpuParticleLevelCollision::clear()
{
    level_ = model_ = triangles_ = vertices_ = nullptr;
    triangle_count_ = vertex_count_ = 0; data_ = {};
}
bool GpuParticleLevelCollision::update_dynamic(bool enabled, bool& changed, std::string& error)
{
    return update_gpu_particle_dynamic_collision(data_, enabled, changed, error);
}
}
