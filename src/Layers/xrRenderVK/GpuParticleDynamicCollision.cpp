#include "xrEngine/stdafx.h"
#include "GpuParticleDynamicCollision.h"
#include "VulkanKinematics.h"
#include "xrEngine/IGame_Level.h"
#include "xrEngine/IGame_Persistent.h"
#include "xrEngine/xr_object.h"
#include "xrEngine/xr_collide_form.h"
#include "xrEngine/cf_dynamic_mesh.h"
#include "xrCore/Animation/Bone.hpp"
#include <cmath>

namespace xray::render::vulkan
{
namespace
{
void vector4(float* target, const Fvector& vector, float w = 0.f)
{ target[0] = vector.x; target[1] = vector.y; target[2] = vector.z; target[3] = w; }
void matrix_rows(const Fmatrix& matrix, float* row0, float* row1, float* row2)
{
    row0[0] = matrix._11; row0[1] = matrix._21; row0[2] = matrix._31; row0[3] = matrix._41;
    row1[0] = matrix._12; row1[1] = matrix._22; row1[2] = matrix._32; row1[3] = matrix._42;
    row2[0] = matrix._13; row2[1] = matrix._23; row2[2] = matrix._33; row2[3] = matrix._43;
}
template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b)
{ return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0); }
bool finite4(const float* values)
{ return std::isfinite(values[0]) && std::isfinite(values[1]) && std::isfinite(values[2]) && std::isfinite(values[3]); }
}
bool update_gpu_particle_dynamic_collision(GpuParticleCollisionInput& data, bool enabled,
    bool& changed, std::string& error)
{
    GpuParticleCollisionInput next;
    if (enabled && g_pGameLevel && g_pGameLevel->bReady && g_pGamePersistent)
    {
        auto& objects = g_pGameLevel->Objects;
        for (uint32_t index = 0; index < objects.o_count(); ++index)
        {
            IGameObject* object = objects.o_get_by_iterator(index);
            if (!object) continue;
            auto& spatial = object->GetSpatialData();
            if (!(spatial.type & STYPE_COLLIDEABLE) || !spatial.node_ptr || spatial.space != &g_pGamePersistent->SpatialSpace)
                continue;
            ICollisionForm* form = object->GetCForm();
            if (!form || form->Type() != cftObject) continue;
            auto* skeleton = dynamic_cast<CCF_Skeleton*>(form);
            if (!skeleton) { error = "unsupported GPU particle object collision form"; return false; }
            if (!object->Visual() || !object->Visual()->dcast_PKinematics())
            { error = "GPU particle collision object has no skeleton"; return false; }
            skeleton->RefreshCollisionGeometry();
            GpuParticleCollisionObject gpu{};
            if (next.shapes.size() > UINT32_MAX) { error = "GPU particle shape count overflow"; return false; }
            gpu.first = static_cast<uint32_t>(next.shapes.size());
            gpu.mesh = dynamic_cast<CCF_DynamicMesh*>(form) != nullptr;
            vector4(gpu.spatial_sphere, spatial.sphere.P, spatial.sphere.R);
            Fvector center; object->XFORM().transform_tiny(center, form->getSphere().P);
            vector4(gpu.sphere, center, form->getSphere().R);
            if (!finite4(gpu.spatial_sphere) || !finite4(gpu.sphere))
            { error = "nonfinite GPU particle object bounds"; return false; }
            std::vector<std::vector<GpuParticleCollisionTriangle>> triangles;
            if (gpu.mesh)
            {
                Fmatrix inverse; inverse.invert(object->XFORM());
                matrix_rows(inverse, gpu.row0, gpu.row1, gpu.row2);
                if (!finite4(gpu.row0) || !finite4(gpu.row1) || !finite4(gpu.row2))
                { error = "invalid GPU particle object transform"; return false; }
                auto* visual = object->Visual();
                auto* kinematics = visual ? dynamic_cast<VulkanKinematics*>(visual->dcast_PKinematics()) : nullptr;
                if (!kinematics || !kinematics->export_collision_triangles(triangles, error))
                { if (error.empty()) error = "unsupported GPU particle dynamic mesh"; return false; }
            }
            for (const auto& element : skeleton->_GetElements())
            {
                if (!element.valid()) continue;
                GpuParticleCollisionShape shape{};
                switch (element.type)
                {
                case SBoneShape::stBox:
                    shape.type = 1; matrix_rows(element.b_IM, shape.row0, shape.row1, shape.row2);
                    vector4(shape.data0, element.b_hsize); break;
                case SBoneShape::stSphere:
                    shape.type = 2; vector4(shape.data0, element.s_sphere.P, element.s_sphere.R); break;
                case SBoneShape::stCylinder:
                    shape.type = 3; vector4(shape.data0, element.c_cylinder.m_center, element.c_cylinder.m_radius);
                    vector4(shape.data1, element.c_cylinder.m_direction, element.c_cylinder.m_height); break;
                default: error = "unsupported GPU particle collision shape"; return false;
                }
                if (!finite4(shape.row0) || !finite4(shape.row1) || !finite4(shape.row2) ||
                    !finite4(shape.data0) || !finite4(shape.data1))
                { error = "nonfinite GPU particle collision shape"; return false; }
                if (next.dynamic_triangles.size() > UINT32_MAX) { error = "GPU particle triangle count overflow"; return false; }
                shape.triangle_first = static_cast<uint32_t>(next.dynamic_triangles.size());
                if (gpu.mesh)
                {
                    if (element.elem_id >= triangles.size() || triangles[element.elem_id].size() > UINT32_MAX - next.dynamic_triangles.size())
                    { error = "invalid GPU particle bone triangle range"; return false; }
                    const auto& bone = triangles[element.elem_id];
                    for (const auto& triangle : bone)
                        if (!finite4(triangle.a) || !finite4(triangle.b) || !finite4(triangle.c))
                        { error = "nonfinite GPU particle mesh triangle"; return false; }
                    next.dynamic_triangles.insert(next.dynamic_triangles.end(), bone.begin(), bone.end());
                    shape.triangle_count = static_cast<uint32_t>(bone.size());
                }
                next.shapes.push_back(shape);
                if (next.shapes.size() > UINT32_MAX) { error = "GPU particle shape count overflow"; return false; }
            }
            gpu.count = static_cast<uint32_t>(next.shapes.size()) - gpu.first;
            if (gpu.count) next.objects.push_back(gpu);
            if (next.objects.size() > UINT32_MAX) { error = "GPU particle object count overflow"; return false; }
        }
    }
    for (size_t first = 0; first < next.objects.size(); first += 16)
    {
        GpuParticleCollisionNode block{}; block.first = static_cast<uint32_t>(first);
        block.count = static_cast<uint32_t>(std::min<size_t>(16, next.objects.size() - first));
        for (unsigned axis = 0; axis < 3; ++axis)
        { block.minimum[axis] = std::numeric_limits<float>::max(); block.maximum[axis] = -block.minimum[axis]; }
        for (size_t i = first; i < first + block.count; ++i)
        {
            const auto& sphere = next.objects[i].spatial_sphere;
            const float margin = (std::max({std::abs(sphere[0]), std::abs(sphere[1]), std::abs(sphere[2]), std::abs(sphere[3])}) + 1.f) *
                (8.f * std::numeric_limits<float>::epsilon());
            const float radius = std::abs(sphere[3]) + margin;
            for (unsigned axis = 0; axis < 3; ++axis)
            { block.minimum[axis] = std::min(block.minimum[axis], sphere[axis] - radius);
              block.maximum[axis] = std::max(block.maximum[axis], sphere[axis] + radius);
              if (!std::isfinite(block.minimum[axis]) || !std::isfinite(block.maximum[axis]))
              { error = "GPU particle collision bounds overflow"; return false; } }
        }
        next.blocks.push_back(block);
    }
    changed = !same(data.dynamic_triangles, next.dynamic_triangles) || !same(data.shapes, next.shapes) ||
        !same(data.objects, next.objects) || !same(data.blocks, next.blocks);
    data.dynamic_triangles = std::move(next.dynamic_triangles); data.shapes = std::move(next.shapes);
    data.objects = std::move(next.objects); data.blocks = std::move(next.blocks);
    error.clear(); return true;
}
}
