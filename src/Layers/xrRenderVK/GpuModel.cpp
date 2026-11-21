#include "xrEngine/stdafx.h"
#include "GpuModel.h"
#include "ObjectVelocityPass.h"
#include "TangentBasis.h"
#include "ClusterLodSelection.h"
#include "ModelPoseConstants.h"
#include "ModelTessellation.h"
#include "SkeletonBones.h"
#include "VulkanSettings.h"
#include "Include/xrRender/Kinematics.h"
#include "xrCore/Animation/Bone.hpp"

#include <algorithm>
#include <cstring>

namespace xray::render::vulkan
{
bool make_model_ray_batch(VkDeviceAddress blas, const VkTransformMatrixKHR& transform,
    const std::vector<RayModelGeometry>& geometries, const std::vector<RayModelTextureSlots>& slots,
    RayGeometryBatch& result, std::string& error)
{
    if (!blas || geometries.empty() || geometries.size() != slots.size())
    { error = "model ray batch requires a BLAS and matching material slots"; return false; }
    RayGeometryBatch batch;
    batch.transform = transform;
    batch.external_blas = blas;
    for (size_t i = 0; i < geometries.size(); ++i)
    {
        const auto& geometry = geometries[i];
        const auto& slot = slots[i];
        if (!geometry.primitive_count || slot.diffuse == UINT32_MAX ||
            (geometry.alpha_cutoff > 0.f && slot.alpha == UINT32_MAX) ||
            (!geometry.pbr_texture.empty() && slot.pbr == UINT32_MAX) ||
            (!geometry.metallic_texture.empty() && slot.metallic == UINT32_MAX) ||
            (!geometry.roughness_texture.empty() && slot.roughness == UINT32_MAX) ||
            (!geometry.ao_texture.empty() && slot.ao == UINT32_MAX) ||
            (!geometry.normal_texture.empty() && slot.normal == UINT32_MAX) ||
            (!geometry.normal_aux_texture.empty() && slot.normal_aux == UINT32_MAX))
        { error = "model ray batch has incomplete material slots"; return false; }
        batch.geometries.push_back(geometry.geometry);
        batch.primitive_counts.push_back(geometry.primitive_count);
        batch.material_ids.push_back(slot.material_id);
        batch.alpha_cutoffs.push_back(geometry.alpha_cutoff);
        batch.texture_slots.push_back(slot.alpha);
        batch.diffuse_texture_slots.push_back(slot.diffuse);
        batch.pbr_texture_slots.push_back(slot.pbr);
        batch.metallic_texture_slots.push_back(slot.metallic);
        batch.roughness_texture_slots.push_back(slot.roughness);
        batch.ao_texture_slots.push_back(slot.ao);
        batch.normal_texture_slots.push_back(slot.normal);
        batch.normal_aux_texture_slots.push_back(slot.normal_aux);
        batch.material_flags.push_back(geometry.emissive ? RayMaterialEmissive : 0u);
    }
    result = std::move(batch);
    error.clear();
    return true;
}
bool GpuModel::add_meshes(ModelGeometry &&geometry, const std::string &inherited_texture, VkDevice device,
                          VkQueue queue, VkCommandPool pool, const VkPhysicalDeviceMemoryProperties &memory,
                          const BufferUploadDispatch &upload, GameTextureFactory &textures, DeferredPass &pass,
                          std::string &error)
{
    const std::string texture = geometry.texture.empty() ? inherited_texture : geometry.texture;
    auto children = std::move(geometry.children);
    geometry.children.clear();
    geometry.fast.reset();
    const auto upload_mesh = [&](ModelGeometry &&decoded) -> bool {
        if (texture.empty() || decoded.indices.empty() || decoded.indices.size() > UINT32_MAX)
        {
            error = "OGF model mesh is missing a texture or triangle indices";
            return false;
        }
        meshes_.emplace_back();
        auto &mesh = meshes_.back();
        mesh.geometry = std::move(decoded);
        mesh.wallmark = mesh.geometry.shader == "effects\\wallmark";
        mesh.ray_diffuse_texture = texture;
        MaterialDefinition definition;
        // An OGF can reference the same S_SET definition as level geometry.
        // Resolve its world flags before choosing a model shader; otherwise a
        // valid ADD/MUL token silently becomes ordinary alpha blending.
        if (!textures.surface_mode(mesh.geometry.shader, texture, mesh.geometry.mode, error, &mesh.alpha_ref,
                                   &mesh.screen_blend, false, false, true, false, nullptr, nullptr, &definition, true))
        {
            error = "OGF shader='" + mesh.geometry.shader + "' texture='" + texture + "': " + error;
            return false;
        }
        if (definition.class_name == "LmEbB   " || definition.class_name == "LaEmB   ")
        {
            error = "OGF shader='" + mesh.geometry.shader + "' class='" + definition.class_name +
                    "' requires level lightmap geometry";
            return false;
        }
        mesh.priority = definition.priority;
        mesh.emissive = definition.emissive;
        mesh.secondary_opaque = mesh.priority >= 2 && mesh.geometry.mode == SurfaceMode::Opaque;
        mesh.matrices = definition.matrices;
        if (definition.class_name == "S_SET   " && definition.role == MaterialDefinition::Role::Water)
        {
            error =
                "OGF shader='" + mesh.geometry.shader + "' class='S_SET   ' water pass requires level water geometry";
            return false;
        }
        mesh.screen_fog = definition.class_name == "S_SET   " && definition.pass.fog;
        if (definition.class_name != "S_SET   ")
            mesh.screen_blend = -1;
        GameTextureFactory::DetailMetadata metadata;
        if (!textures.texture_detail_metadata(texture, metadata, error))
        {
            error = "OGF texture='" + texture + "': " + error;
            return false;
        }
        if (tessellation_enabled && definition.tessellation)
        {
            const HeightField *height = nullptr;
            if (definition.tessellation & 2)
            {
                if (metadata.bump_mode < 2 || metadata.bump_texture.empty())
                {
                    error = "OGF HM tessellation requires a bump auxiliary texture: " + texture;
                    return false;
                }
                if (!textures.height_field(metadata.bump_texture + "#", height, error))
                    return false;
            }
            const size_t triangles = mesh.geometry.indices.size() / 3;
            if (!tessellate_model(mesh.geometry, definition.tessellation, height, metadata.bump_height, error))
            {
                error =
                    "OGF PN/HM tessellation shader='" + mesh.geometry.shader + "' texture='" + texture + "': " + error;
                return false;
            }
            Msg("[renderer-vulkan] OGF PN/HM tessellation shader='%s' mode=%u triangles=%zu/%zu",
                mesh.geometry.shader.c_str(), definition.tessellation, triangles, mesh.geometry.indices.size() / 3);
        }
        prepare_tangent_basis(mesh.geometry.vertices, mesh.geometry.indices);
        const bool animated = mesh.geometry.type == 4 || mesh.geometry.type == 5;
        // R2's opaque MODELEbB calls uber_deffer("model", "base") and never
        // samples oT2_Name. Only its blended forward element reads the cube.
        mesh.environment = definition.class_name == "MODELEbB" && mesh.geometry.mode == SurfaceMode::Transparent;
        // Tree surfaces are two-sided by blender contract. A user-controlled
        // shader name must never select a different raster/shader variant.
        mesh.tree_surface = definition.class_name == "D_TREE  ";
        mesh.ray_pbr_texture = metadata.pbr_texture;
        mesh.ray_metallic_texture = metadata.metallic_texture;
        mesh.ray_roughness_texture = metadata.roughness_texture;
        mesh.ray_ao_texture = metadata.ao_texture;
        mesh.ray_normal_texture = metadata.bump_mode >= 2 ? metadata.bump_texture : metadata.external_normal_texture;
        if (metadata.bump_mode >= 2 && !metadata.bump_texture.empty())
            mesh.ray_normal_aux_texture = metadata.bump_texture + "#";
        // CBlender_Tree's high quality element samples s_detail. MODEL does
        // not, even when the same base DDS carries detail metadata.
        mesh.detail = mesh.tree_surface && metadata.diffuse && !metadata.texture.empty() && metadata.texture != "$null";
        if (mesh.detail && mesh.environment)
        {
            error = "OGF shader='" + mesh.geometry.shader +
                    "': diffuse detail with environment cube has no Vulkan pipeline";
            return false;
        }
        const bool surface_roles = mesh.detail || metadata.normal || metadata.bump_mode >= 2 ||
                                   !metadata.external_normal_texture.empty() || metadata.material != 1 ||
                                   metadata.material_weight != 0.f ||
                                   (!metadata.pbr_texture.empty() || !metadata.metallic_texture.empty() ||
                                     !metadata.roughness_texture.empty() || !metadata.ao_texture.empty() ||
                                     !metadata.parallax_texture.empty()) ||
                                   definition.emissive;
        mesh.surface = surface_roles;
        const uint16_t roles = mesh.surface
                                   ? ((metadata.bump_mode >= 2 ? BumpRole | BumpAuxRole | GlossRole : 0u) |
                                      (metadata.normal || !metadata.external_normal_texture.empty() ? NormalRole : 0u))
                                   : 0u;
        if (!build_render_pass_graph({RenderConsumer::Model, mesh.geometry.mode, false, animated, mesh.environment,
                                      mesh.tree_surface, mesh.geometry.skin_weights, 1, roles, mesh.detail,
                                      mesh.screen_blend, false, definition.object_lod, mesh.screen_fog},
                                     mesh.pass_graph, error))
        {
            error = "OGF shader='" + mesh.geometry.shader + "' texture='" + texture + "': " + error;
            return false;
        }
        if (mesh.secondary_opaque)
        {
            // R2 submits priority 2/3 opaque objects after the lighting
            // combine. They replace forward color and write scene depth.
            auto &node = mesh.pass_graph.passes[0];
            node.target = LevelPassTarget::Forward;
            node.blend = LevelPassBlend::Replace;
            node.depth_test = node.depth_write = true;
            node.fragment = animated ? "vk\\skinned_priority_opaque.ps" : "vk\\object_priority_opaque.ps";
        }
        if (mesh.wallmark && mesh.geometry.mode == SurfaceMode::Transparent)
            mesh.pass_graph.passes[0].fragment = animated ? "vk\\skinned_object_wallmark_blended.ps" :
                                                            "vk\\object_wallmark_blended.ps";
        const bool cluster_requested = Core.Params && std::strstr(Core.Params, "-vk-cluster-dag");
        if (cluster_requested && !animated && !mesh.geometry.depth_only && !mesh.tree_surface &&
            !mesh.secondary_opaque && !mesh.wallmark &&
            mesh.geometry.mode != SurfaceMode::Transparent && mesh.pass_graph.count == 1 &&
            mesh.geometry.indices.size() >= 128u * 3u &&
            mesh.geometry.indices.size() <= std::numeric_limits<uint32_t>::max())
        {
            std::string cluster_error;
            std::vector<uint32_t> packed_cluster_indices;
            const size_t window_count = mesh.geometry.windows.empty() ? 1 : mesh.geometry.windows.size();
            mesh.cluster_lod_windows.reserve(window_count);
            bool cluster_windows_ready = true;
            size_t group_count = 0;
            size_t cluster_count = 0;
            for (size_t window_index = 0; window_index < window_count; ++window_index)
            {
                const SlideWindow window = mesh.geometry.windows.empty()
                    ? SlideWindow{0, static_cast<uint32_t>(mesh.geometry.indices.size()), 0}
                    : mesh.geometry.windows[window_index];
                ClusterLodData data;
                if (!build_cluster_lod(mesh.geometry, window, data, cluster_error) || data.indices.empty())
                {
                    if (cluster_error.empty()) cluster_error = "cluster DAG window produced no indices";
                    cluster_windows_ready = false;
                    break;
                }
                Mesh::ClusterWindow cluster_window;
                uint32_t cluster_index_count = 0;
                if (!append_packed_cluster_indices(packed_cluster_indices, data, cluster_window.first_index,
                        cluster_index_count, cluster_error) || !cluster_index_count)
                {
                    cluster_windows_ready = false;
                    break;
                }
                group_count += data.groups.size();
                cluster_count += data.clusters.size();
                cluster_window.data = std::move(data);
                mesh.cluster_lod_windows.push_back(std::move(cluster_window));
            }
            if (cluster_windows_ready && !packed_cluster_indices.empty())
            {
                if (upload_buffer(device, queue, pool, memory, upload, packed_cluster_indices.data(),
                        packed_cluster_indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                        mesh.cluster_indices, pending_, cluster_error))
                {
                    Msg("[renderer-vulkan] cluster DAG model baked shader='%s' windows=%zu groups=%zu clusters=%zu indices=%zu",
                        mesh.geometry.shader.c_str(), window_count, group_count, cluster_count,
                        packed_cluster_indices.size());
                }
            }
            else if (cluster_error.empty())
                cluster_error = "cluster DAG generated no usable window indices";
            if (!cluster_windows_ready || !mesh.cluster_indices.handle())
            {
                if (!cluster_error.empty())
                    Msg("! [renderer-vulkan] cluster DAG model bake skipped shader='%s': %s",
                        mesh.geometry.shader.c_str(), cluster_error.c_str());
                mesh.cluster_lod_windows.clear();
                mesh.cluster_indices.destroy();
            }
        }
        auto &node = mesh.pass_graph.passes[0];
        const std::string &vertex = node.vertex;
        const std::string &fragment = node.fragment;
        if (!pass.request_game_pipeline(vertex, fragment,
                                        mesh.secondary_opaque ? SurfaceMode::Transparent : mesh.geometry.mode, false,
                                        animated, error, node.shader_defines, &node.shader_variant_hash))
        {
            error = "OGF type=" + std::to_string(mesh.geometry.type) + " shader pair: " + error;
            return false;
        }
        if (mesh.screen_blend >= 0 || mesh.environment)
        {
            RenderPassGraph hud_graph;
            if (!build_render_pass_graph({RenderConsumer::Model, mesh.geometry.mode, true, animated,
                                          mesh.environment, false, mesh.geometry.skin_weights, 1, 0, false,
                                          mesh.screen_blend},
                                         hud_graph, error))
                return false;
            auto &hud_node = hud_graph.passes[0];
            if (!pass.request_game_pipeline(hud_node.vertex, hud_node.fragment, SurfaceMode::Opaque, true, animated,
                                            error, hud_node.shader_defines, &hud_node.shader_variant_hash))
            {
                error = "OGF HUD shader pair: " + error;
                return false;
            }
            mesh.hud_variant_hash = hud_node.shader_variant_hash;
        }
        Msg("[renderer-vulkan] model.material shader='%s' texture='%s' mode=%u priority=%u weights=%u bump=%u "
            "material=%u",
            mesh.geometry.shader.c_str(), texture.c_str(), unsigned(mesh.geometry.mode), mesh.priority,
            mesh.geometry.skin_weights, metadata.bump_mode, metadata.material);
        mesh.detail_scale = metadata.scale;
        bool surface_ready = false;
        if (mesh.surface)
        {
            std::array<std::string, SurfaceMaterialTextureCount> names;
            names.fill(texture);
            if (mesh.environment)
                names[2] = definition.environment_texture;
            if (mesh.detail)
                names[2] = metadata.texture;
            if (metadata.bump_mode >= 2)
            {
                names[7] = metadata.bump_texture;
                names[8] = metadata.bump_texture + "#";
            }
            if (!metadata.external_normal_texture.empty())
                names[15] = metadata.external_normal_texture;
            uint32_t surface_map_flags = metadata.external_normal_texture.empty() ? 0u : SurfaceMapExternalNormal;
            if (!metadata.parallax_texture.empty())
            {
                names[36] = metadata.parallax_texture;
                surface_map_flags |= SurfaceMapParallax;
            }
            {
                if (!metadata.pbr_texture.empty())
                {
                    names[19] = metadata.pbr_texture;
                    surface_map_flags |= SurfaceMapPackedPbr;
                }
                if (!metadata.metallic_texture.empty())
                {
                    names[16] = metadata.metallic_texture;
                    surface_map_flags |= SurfaceMapMetallic;
                }
                if (!metadata.roughness_texture.empty())
                {
                    names[17] = metadata.roughness_texture;
                    surface_map_flags |= SurfaceMapRoughness;
                }
                if (!metadata.ao_texture.empty())
                {
                    names[18] = metadata.ao_texture;
                    surface_map_flags |= SurfaceMapAo;
                }
            }
            SurfaceParameters surface;
            surface.bump_mode = metadata.bump_mode >= 2 ? float(metadata.bump_mode) : 0.f;
            surface.external_normal = float(surface_map_flags);
            surface.material = float(metadata.material);
            surface.material_weight = metadata.material_weight;
            surface.bump_height = metadata.bump_height;
            surface.emissive = definition.emissive ? 1.f : 0.f;
            if (mesh.environment)
                surface.terrain_bump = metadata.scale;
            if (metadata.normal)
            {
                GameTextureFactory::DetailMetadata detail_metadata;
                if (!textures.texture_detail_metadata(metadata.texture, detail_metadata, error))
                    return false;
                names[mesh.environment ? 3 : 2] = metadata.texture;
                names[9] = detail_metadata.bump_mode >= 2 ? detail_metadata.bump_texture : metadata.texture + "_bump";
                names[10] = names[9] + "#";
                surface.detail_normal = metadata.diffuse ? 2.f : 1.f;
            }
            surface_ready = textures.surface_material(names, surface, pass, mesh.material, error, mesh.environment,
                                                      definition.class_name == "S_SET   " && definition.pass.clamp);
        }
        const bool ray_eligible = ray_address_ && !mesh.tree_surface && !mesh.secondary_opaque &&
            mesh.geometry.mode != SurfaceMode::Transparent;
        const VkBufferUsageFlags ray_usage = ray_eligible ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR : 0;
        if (!(mesh.surface ? surface_ready
              : mesh.environment
                  ? textures.environment_material(texture, {}, definition.environment_texture, true, pass,
                                                  mesh.material, error)
                  : textures.material(texture, pass, mesh.material, error,
                                      (definition.class_name == "S_SET   " || definition.class_name == "PARTICLE") &&
                                          definition.pass.clamp)) ||
            !upload_buffer(device, queue, pool, memory, upload, mesh.geometry.indices.data(),
                           mesh.geometry.indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT | ray_usage,
                           mesh.indices, pending_, error))
            return false;
        if (animated && mesh.geometry.skin_weights >= 1 && mesh.geometry.skin_weights <= 4)
        {
            const std::string vertex = "vk\\skinned_" + std::to_string(mesh.geometry.skin_weights) + ".vs";
            mesh.gpu_skinning = pass.has_game_pipeline(vertex, fragment,
                mesh.pass_graph.passes[0].shader_variant_hash);
            for (const ModelVertex &vertex : mesh.geometry.vertices)
                for (unsigned i = 0; i < mesh.geometry.skin_weights; ++i)
                    if (vertex.weights[i] > 0)
                        mesh.max_bone_index = std::max(mesh.max_bone_index, vertex.bones[i]);
            if (!upload_buffer(device, queue, pool, memory, upload, mesh.geometry.vertices.data(),
                               mesh.geometry.vertices.size() * sizeof(ModelVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                               mesh.skinned_vertices, pending_, error))
                return false;
        }
        if (!animated)
        {
            std::vector<LevelVertex> vertices;
            vertices.reserve(mesh.geometry.vertices.size());
            for (const auto &source : mesh.geometry.vertices)
            {
                LevelVertex vertex;
                std::copy_n(source.position, 3, vertex.position);
                std::copy_n(source.normal, 3, vertex.normal);
                for (unsigned axis = 0; axis < 3; ++axis)
                {
                    tangent_store(vertex.tangent[axis], source.tangent[axis]);
                    tangent_store(vertex.bitangent[axis], source.bitangent[axis]);
                }
                std::copy_n(source.uv, 2, vertex.uv);
                vertices.push_back(vertex);
            }
            if (vertices.empty() ||
                !upload_buffer(device, queue, pool, memory, upload, vertices.data(),
                               vertices.size() * sizeof(LevelVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | ray_usage,
                               mesh.vertices[0], pending_, error))
            {
                if (error.empty())
                    error = "static OGF has no vertices";
                return false;
            }
            // Static OGF vertices never change. Keep one GPU buffer instead
            // of three host-visible copies plus the decoded CPU array.
            std::vector<ModelVertex>().swap(mesh.geometry.vertices);
        }
        else if (mesh.gpu_skinning && !ray_address_)
            std::vector<ModelVertex>().swap(mesh.geometry.vertices);
        std::vector<uint32_t>().swap(mesh.geometry.indices);
        return true;
    };
    if (!geometry.vertices.empty())
    {
        if (!upload_mesh(std::move(geometry)))
            return false;
        // OGF_FASTPATH/geomX is a depth-only proxy, never a shaded distance
        // LOD. Model shadows do not consume it yet; avoid uploading unused data.
    }
    for (auto &child : children)
        if (!add_meshes(std::move(child), texture, device, queue, pool, memory, upload, textures, pass, error))
            return false;
    return true;
}

bool GpuModel::load(const char *name, IReader *source, VkDevice device, VkQueue queue, VkCommandPool pool,
                    const VkPhysicalDeviceMemoryProperties &memory, const BufferUploadDispatch &upload,
                    GameTextureFactory &textures, DeferredPass &pass, std::string &error,
                    PFN_vkGetBufferDeviceAddress ray_address)
{
    ModelGeometry decoded;
    if (!load_engine_model_geometry(name, source, decoded, error))
        return false;
    return load_geometry(std::move(decoded), device, queue, pool, memory, upload, textures, pass, error, ray_address);
}

bool GpuModel::load_record(const VisualRecord &record, const std::string &inherited_texture, VkDevice device,
                           VkQueue queue, VkCommandPool pool, const VkPhysicalDeviceMemoryProperties &memory,
                           const BufferUploadDispatch &upload, GameTextureFactory &textures, DeferredPass &pass,
                           std::string &error, PFN_vkGetBufferDeviceAddress ray_address)
{
    ModelGeometry decoded;
    if (!decode_engine_model_geometry(record, decoded, error))
        return false;
    if (decoded.texture.empty())
        decoded.texture = inherited_texture;
    return load_geometry(std::move(decoded), device, queue, pool, memory, upload, textures, pass, error, ray_address);
}

bool GpuModel::load_geometry(ModelGeometry &&decoded, VkDevice device, VkQueue queue, VkCommandPool pool,
                             const VkPhysicalDeviceMemoryProperties &memory, const BufferUploadDispatch &upload,
                             GameTextureFactory &textures, DeferredPass &pass, std::string &error,
                             PFN_vkGetBufferDeviceAddress ray_address)
{
    destroy();
    if (!device || !queue || !pool)
    {
        error = "OGF model requires a Vulkan device, queue and command pool";
        return false;
    }
    device_ = device;
    queue_ = queue;
    pool_ = pool;
    memory_ = memory;
    upload_ = upload;
    textures_ = &textures;
    pass_ = &pass;
    ray_address_ = ray_address;
    if (!add_meshes(std::move(decoded), "", device, queue, pool, memory, upload, textures, pass, error) ||
        meshes_.empty())
    {
        if (error.empty())
            error = "OGF model has no renderable meshes";
        destroy();
        return false;
    }
    error.clear();
    return true;
}

bool GpuModel::retire_completed_uploads()
{
    if (device_)
        collect_completed_buffer_uploads(device_, pool_, upload_, pending_);
    return pending_.empty();
}

bool GpuModel::record(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16],
                      const float *pose, size_t bones, std::string &error, GeometryPhase phase, float lod,
                      const IKinematics *instance, const float *normal_rows, int priority_filter, float opacity,
                      const float *world_transform)
{
    if (!device_ || frame.frame_index >= FrameContext::FramesInFlight)
    {
        error = "Vulkan model has no acquired frame";
        return false;
    }
    const float camera[4]{Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z, 1.f};
    for (size_t index = 0; index < meshes_.size(); ++index)
    {
        auto &base = meshes_[index];
        // Distance LOD selects regular SWI; geomX is reserved for depth.
        auto &mesh = base;
        if (phase != GeometryPhase::Hud &&
            !(phase == GeometryPhase::SecondaryOpaque ? mesh.secondary_opaque
              : phase == GeometryPhase::Transparent
                  ? mesh.geometry.mode == SurfaceMode::Transparent &&
                        (priority_filter < 0 || mesh.priority == priority_filter)
                  : mesh.geometry.mode != SurfaceMode::Transparent && !mesh.secondary_opaque))
            continue;
        const bool skinned = mesh.geometry.type == 4 || mesh.geometry.type == 5;
        const bool gpu_skinning = skinned && mesh.gpu_skinning;
        std::array<BufferResource, FrameContext::FramesInFlight> *buffers = &mesh.vertices;
        if (skinned)
        {
            if (!instance)
            {
                error = "skinned Vulkan draw has no owning skeleton instance";
                return false;
            }
            if (!gpu_skinning)
            {
                buffers = &mesh.instance_vertices[instance];
                if (!(*buffers)[frame.frame_index].handle())
                    if (!(*buffers)[frame.frame_index].initialize(
                            device_, mesh.geometry.vertices.size() * sizeof(LevelVertex),
                            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory_,
                            upload_.buffer, error))
                        return false;
            }
        }
        const size_t window_index = select_slide_window_index(mesh.geometry.windows, lod);
        const SlideWindow window = select_slide_window(mesh.geometry.windows, lod, mesh.index_count());
        if (gpu_skinning)
        {
            const bool hud = phase == GeometryPhase::Hud;
            const SurfaceMode draw_mode = mesh.secondary_opaque ? SurfaceMode::Transparent : mesh.geometry.mode;
            if (!pose || !bones || bones > MaxVulkanSkeletonBones || mesh.max_bone_index >= bones)
            {
                error = "Vulkan GPU skinning references a bone outside its pose";
                return false;
            }
            auto &per_instance = mesh.poses[instance];
            auto &pose_buffer = per_instance.buffers[frame.frame_index];
            auto &descriptor = per_instance.descriptors[frame.frame_index];
            const size_t pose_bytes = (bones + 1) * 16 * sizeof(float);
            if (!pose_buffer.handle())
                if (!pose_buffer.initialize(device_, pose_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            memory_, upload_.buffer, error))
                    return false;
            if (pose_buffer.size() < pose_bytes)
            {
                error = "Vulkan skeletal pose grew after its descriptor was allocated";
                return false;
            }
            if (!descriptor && !pass_->pose_descriptor(pose_buffer.handle(), pose_buffer.size(), descriptor, error))
                return false;
            const size_t bone_bytes = bones * 16 * sizeof(float);
            const auto object_constants = model_pose_constants(normal_rows, pass.model_hemi_negative());
            if (!pose_buffer.write(0, pose, bone_bytes, error) ||
                !pose_buffer.write(bone_bytes, object_constants.data(), sizeof(object_constants), error))
                return false;
            RenderPassGraph hud_graph;
            if (hud && !build_render_pass_graph({RenderConsumer::Model, mesh.geometry.mode, true, true,
                                                 mesh.environment, mesh.tree_surface, mesh.geometry.skin_weights, 1, 0,
                                                 false, mesh.screen_blend},
                                                hud_graph, error))
            {
                error = "OGF HUD shader='" + mesh.geometry.shader + "': " + error;
                return false;
            }
            auto &node = hud ? hud_graph.passes[0] : mesh.pass_graph.passes[0];
            if (hud)
                node.shader_variant_hash = mesh.hud_variant_hash;
            const std::string &vertex_name = node.vertex;
            const char *fragment_name = node.fragment.c_str();
            if (!pass.record_skinned(frame, mesh.skinned_vertices.handle(), mesh.indices.handle(), window.index_count,
                                     mvp, mesh.material, descriptor, hud ? SurfaceMode::Opaque : draw_mode, hud,
                                     window.offset, vertex_name.c_str(), fragment_name,
                                     mesh.wallmark ? std::clamp(opacity, 0.f, 1.f) : float(mesh.alpha_ref) / 255.f,
                                     normal_rows, mesh.detail_scale, mesh.environment ? camera : nullptr,
                                     &mesh.matrices, node.shader_variant_hash))
            {
                error =
                    "Vulkan skeletal shader pair or render pass is unavailable: " + vertex_name + " / " + fragment_name;
                return false;
            }
            continue;
        }
        if (phase == GeometryPhase::Hud)
        {
            RenderPassGraph hud_graph;
            if (!build_render_pass_graph({RenderConsumer::Model, mesh.geometry.mode, true, false, mesh.environment,
                                          mesh.tree_surface, 0, 1, 0, false, mesh.screen_blend},
                                         hud_graph, error))
            {
                error = "OGF HUD shader='" + mesh.geometry.shader + "': " + error;
                return false;
            }
            BufferResource *vertex_buffer = &mesh.vertices[0];
            if (skinned)
            {
                std::vector<LevelVertex> vertices;
                if (!skin_model_mesh(mesh.geometry, pose, bones, vertices, error))
                    return false;
                vertex_buffer = &(*buffers)[frame.frame_index];
                if (!vertex_buffer->write(0, vertices.data(), vertices.size() * sizeof(LevelVertex), error))
                    return false;
            }
            auto &hud_node = hud_graph.passes[0];
            hud_node.shader_variant_hash = mesh.hud_variant_hash;
            if (!pass.record_hud(frame, vertex_buffer->handle(), mesh.indices.handle(), window.index_count, mvp,
                                 mesh.material, window.offset,
                                 hud_node.vertex.empty() ? nullptr : hud_node.vertex.c_str(),
                                 hud_node.fragment.empty() ? nullptr : hud_node.fragment.c_str(),
                                 float(mesh.alpha_ref) / 255.f, normal_rows, 0, &mesh.matrices,
                                 hud_node.shader_variant_hash))
            {
                if (error.empty())
                    error = "Vulkan HUD model geometry recording failed";
                return false;
            }
            continue;
        }
        const bool transparent = mesh.geometry.mode == SurfaceMode::Transparent || mesh.secondary_opaque;
        const char *vertex = mesh.pass_graph.passes[0].vertex.c_str();
        const char *fragment = mesh.pass_graph.passes[0].fragment.c_str();
        const uint64_t variant_hash = mesh.pass_graph.passes[0].shader_variant_hash;
        // Both the immutable static buffer and CPU-skinned frame buffers use
        // LevelVertex; pose-buffer variants use a separate GPU layout.
        const bool named = pass.require_game_pipeline(
            vertex, fragment, transparent ? SurfaceMode::Transparent : mesh.geometry.mode, false, false, error,
            variant_hash);
        if (!named)
            return false;
        BufferResource *vertex_buffer = &mesh.vertices[0];
        if (skinned)
        {
            std::vector<LevelVertex> vertices;
            if (!skin_model_mesh(mesh.geometry, pose, bones, vertices, error))
                return false;
            vertex_buffer = &(*buffers)[frame.frame_index];
            if (!vertex_buffer->write(0, vertices.data(), vertices.size() * sizeof(LevelVertex), error))
                return false;
        }
        const Mesh::ClusterWindow* cluster_window = window_index < mesh.cluster_lod_windows.size()
            ? &mesh.cluster_lod_windows[window_index] : nullptr;
        if (!transparent && phase == GeometryPhase::OpaqueAndAlphaTest && world_transform && cluster_window &&
            mesh.cluster_indices.handle() && !cluster_window->data.clusters.empty())
        {
            const std::array<float, 3> camera_position{
                Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z};
            const ClusterLodSelection selection = select_cluster_lod(cluster_window->data,
                frame.extent, mvp, world_transform, camera_position, Device.fFOV);
            if (selection.disposition == ClusterLodDisposition::FullyCulled)
                continue;
            const std::vector<uint32_t>& cluster_draws = selection.clusters;
            bool valid_cluster_draws = selection.disposition == ClusterLodDisposition::UseClusters;
            for (uint32_t cluster_index : cluster_draws)
            {
                if (cluster_index >= cluster_window->data.clusters.size())
                {
                    valid_cluster_draws = false;
                    break;
                }
                const ClusterLodCluster& cluster = cluster_window->data.clusters[cluster_index];
                const size_t packed_count = mesh.cluster_indices.size() / sizeof(uint32_t);
                if (cluster_window->first_index > packed_count ||
                    cluster.first_index > packed_count - cluster_window->first_index ||
                    cluster.index_count > packed_count - cluster_window->first_index - cluster.first_index)
                {
                    valid_cluster_draws = false;
                    break;
                }
            }
            if (valid_cluster_draws)
            {
                for (uint32_t cluster_index : cluster_draws)
                {
                    const ClusterLodCluster& cluster = cluster_window->data.clusters[cluster_index];
                    if (!pass.record_geometry(frame, vertex_buffer->handle(), mesh.cluster_indices.handle(),
                            cluster.index_count, mvp, mesh.material, mesh.geometry.mode,
                            cluster_window->first_index + cluster.first_index,
                            named ? vertex : nullptr, named ? fragment : nullptr,
                            float(mesh.alpha_ref) / 255.f, normal_rows, mesh.detail_scale,
                            mesh.environment ? camera : nullptr, nullptr, 0, &mesh.matrices, VK_NULL_HANDLE, 1,
                            VK_NULL_HANDLE, 0, 0, variant_hash))
                    {
                        error = "Vulkan clustered model geometry recording failed";
                        return false;
                    }
                }
                continue;
            }
        }
        if (!(transparent
                  ? pass.record_transparent(frame, vertex_buffer->handle(), mesh.indices.handle(), window.index_count,
                                            mvp, mesh.material, window.offset, named ? vertex : nullptr,
                                            named ? fragment : nullptr, nullptr,
                                            mesh.wallmark ? std::clamp(opacity, 0.f, 1.f) :
                                                            float(mesh.alpha_ref) / 255.f,
                                            normal_rows, mesh.detail_scale, 0, &mesh.matrices, variant_hash)
                  : pass.record_geometry(frame, vertex_buffer->handle(), mesh.indices.handle(), window.index_count, mvp,
                                         mesh.material, mesh.geometry.mode, window.offset, named ? vertex : nullptr,
                                         named ? fragment : nullptr, float(mesh.alpha_ref) / 255.f, normal_rows,
                                         mesh.detail_scale, mesh.environment ? camera : nullptr, nullptr, 0,
                                         &mesh.matrices, VK_NULL_HANDLE, 1, VK_NULL_HANDLE, 0, 0, variant_hash)))
        {
            error = "Vulkan model geometry recording failed";
            return false;
        }
    }
    error.clear();
    return true;
}

bool GpuModel::record_object_velocity(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
    const float* current_mvp, const float* previous_mvp, const IKinematics* instance,
    float lod, std::string& error, const float* world_transform) const
{
    static_assert(FrameContext::FramesInFlight == 2);
    if (!pass.ready() || !current_mvp || !previous_mvp || frame.frame_index >= FrameContext::FramesInFlight)
    { error = "object velocity needs a live raster pass and matrices"; return false; }
    for (const Mesh& mesh : meshes_)
    {
        // CPU-skinned meshes retain both vertex buffers; GPU-skinned meshes
        // retain per-instance pose descriptors in both frame slots.
        if ((mesh.geometry.mode != SurfaceMode::Opaque && mesh.geometry.mode != SurfaceMode::AlphaTest) ||
            mesh.secondary_opaque)
            continue;
        const size_t window_index = select_slide_window_index(mesh.geometry.windows, lod);
        const SlideWindow window = select_slide_window(mesh.geometry.windows, lod, mesh.index_count());
        const bool cutout = mesh.geometry.mode == SurfaceMode::AlphaTest;
        float base_rows[8]{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
        if (cutout && mesh.matrices.enabled[0])
        {
            const auto matrix = mesh.matrices.values[0].evaluate(Device.fTimeGlobal,
                reinterpret_cast<const float*>(&Device.mView));
            for (size_t row = 0; row < 2; ++row)
                std::copy(matrix[row].begin(), matrix[row].end(), base_rows + row * 4);
        }
        if (mesh.gpu_skinning)
        {
            const auto found = mesh.poses.find(instance);
            if (!instance || found == mesh.poses.end() || !mesh.skinned_vertices.handle() ||
                !found->second.descriptors[frame.frame_index] ||
                !found->second.descriptors[frame.frame_index ^ 1])
                continue;
            pass.draw_skinned(frame, mesh.skinned_vertices.handle(), mesh.indices.handle(),
                window.index_count, window.offset, current_mvp, previous_mvp,
                found->second.descriptors[frame.frame_index],
                found->second.descriptors[frame.frame_index ^ 1],
                cutout ? mesh.material : VK_NULL_HANDLE, float(mesh.alpha_ref) / 255.f, base_rows);
            continue;
        }
        VkBuffer current_vertices = mesh.vertices[0].handle();
        VkBuffer previous_vertices = current_vertices;
        if (mesh.geometry.type == 4 || mesh.geometry.type == 5)
        {
            const auto found = mesh.instance_vertices.find(instance);
            if (!instance || frame.frame_index >= FrameContext::FramesInFlight ||
                found == mesh.instance_vertices.end())
                continue;
            current_vertices = found->second[frame.frame_index].handle();
            previous_vertices = found->second[frame.frame_index ^ 1].handle();
            if (!previous_vertices) continue;
        }
        if (!current_vertices || !mesh.indices.handle())
        { error = "object velocity model buffer is unavailable"; return false; }
        const Mesh::ClusterWindow* cluster_window = window_index < mesh.cluster_lod_windows.size()
            ? &mesh.cluster_lod_windows[window_index] : nullptr;
        if (world_transform && cluster_window && mesh.cluster_indices.handle() &&
            !cluster_window->data.clusters.empty())
        {
            const std::array<float, 3> camera_position{
                Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z};
            const ClusterLodSelection selection = select_cluster_lod(cluster_window->data,
                frame.extent, current_mvp, world_transform, camera_position, Device.fFOV);
            if (selection.disposition == ClusterLodDisposition::FullyCulled)
                continue;
            const std::vector<uint32_t>& cluster_draws = selection.clusters;
            bool valid_cluster_draws = selection.disposition == ClusterLodDisposition::UseClusters;
            for (uint32_t cluster_index : cluster_draws)
            {
                if (cluster_index >= cluster_window->data.clusters.size())
                {
                    valid_cluster_draws = false;
                    break;
                }
                const ClusterLodCluster& cluster = cluster_window->data.clusters[cluster_index];
                const size_t packed_count = mesh.cluster_indices.size() / sizeof(uint32_t);
                if (cluster_window->first_index > packed_count ||
                    cluster.first_index > packed_count - cluster_window->first_index ||
                    cluster.index_count > packed_count - cluster_window->first_index - cluster.first_index)
                {
                    valid_cluster_draws = false;
                    break;
                }
            }
            if (valid_cluster_draws)
            {
                for (uint32_t cluster_index : cluster_draws)
                {
                    const ClusterLodCluster& cluster = cluster_window->data.clusters[cluster_index];
                    pass.draw(frame, current_vertices, previous_vertices, mesh.cluster_indices.handle(),
                        cluster.index_count, cluster_window->first_index + cluster.first_index, 0,
                        current_mvp, previous_mvp,
                        cutout ? mesh.material : VK_NULL_HANDLE, float(mesh.alpha_ref) / 255.f, base_rows);
                }
                continue;
            }
        }
        pass.draw(frame, current_vertices, previous_vertices, mesh.indices.handle(), window.index_count,
            window.offset, 0, current_mvp, previous_mvp, cutout ? mesh.material : VK_NULL_HANDLE,
            float(mesh.alpha_ref) / 255.f, base_rows);
    }
    error.clear();
    return true;
}

bool GpuModel::record_shadow(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16],
                             IKinematics *skeleton, std::string &error, float lod,
                             const float *world_transform)
{
    if (frame.frame_index >= FrameContext::FramesInFlight)
        return false;
    std::vector<float> pose;
    for (auto &mesh : meshes_)
    {
        if (mesh.geometry.mode == SurfaceMode::Transparent)
            continue;
        const bool skinned = mesh.geometry.type == 4 || mesh.geometry.type == 5;
        VkDescriptorSet descriptor = VK_NULL_HANDLE;
        VkBuffer vertices = mesh.vertices[0].handle();
        if (skinned)
        {
            if (!skeleton || !mesh.skinned_vertices.handle())
            {
                error = "shadow requires GPU skinning vertices and a skeleton";
                return false;
            }
            auto &state = mesh.poses[skeleton];
            auto &buffer = state.buffers[frame.frame_index];
            descriptor = state.descriptors[frame.frame_index];
            if (state.shadow_frames[frame.frame_index] != Device.dwFrame)
            {
                if (pose.empty())
                {
                    skeleton->CalculateBones();
                    const size_t bones = skeleton->LL_BoneCount();
                    if (!bones || bones > MaxVulkanSkeletonBones || mesh.max_bone_index >= bones)
                    {
                        error = "invalid shadow skeleton";
                        return false;
                    }
                    pose.resize((bones + 1) * 16, 0);
                    for (size_t i = 0; i < bones; ++i)
                        std::memcpy(pose.data() + i * 16, &skeleton->LL_GetBoneInstance(u16(i)).mRenderTransform, 64);
                }
                if (mesh.max_bone_index >= pose.size() / 16 - 1)
                {
                    error = "shadow mesh bone is outside its pose";
                    return false;
                }
                const size_t bytes = pose.size() * sizeof(float);
                if (!buffer.handle() &&
                    !buffer.initialize(device_, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       memory_, upload_.buffer, error))
                    return false;
                if (!state.descriptors[frame.frame_index] &&
                    !pass_->pose_descriptor(buffer.handle(), bytes, state.descriptors[frame.frame_index], error))
                    return false;
                if (buffer.size() < bytes)
                {
                    error = "shadow pose grew after allocation";
                    return false;
                }
                // Shadow vertices read bones only. Do not overwrite the final object/ROS
                // matrix used by G-buffer and forward commands sharing this frame slot.
                if (!buffer.write(0, pose.data(), bytes - 16 * sizeof(float), error))
                    return false;
                state.shadow_frames[frame.frame_index] = Device.dwFrame;
                descriptor = state.descriptors[frame.frame_index];
            }
            vertices = mesh.skinned_vertices.handle();
        }
        const size_t window_index = select_slide_window_index(mesh.geometry.windows, lod);
        const auto window = select_slide_window(mesh.geometry.windows, lod, mesh.index_count());
        const Mesh::ClusterWindow* cluster_window = window_index < mesh.cluster_lod_windows.size()
            ? &mesh.cluster_lod_windows[window_index] : nullptr;
        if (!skinned && world_transform && cluster_window && mesh.cluster_indices.handle() &&
            !cluster_window->data.clusters.empty())
        {
            const std::array<float, 3> camera_position{
                Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z};
            const ClusterLodSelection selection = select_cluster_lod(cluster_window->data,
                frame.extent, mvp, world_transform, camera_position, Device.fFOV);
            if (selection.disposition == ClusterLodDisposition::FullyCulled)
                continue;
            bool valid_cluster_draws = selection.disposition == ClusterLodDisposition::UseClusters;
            const size_t packed_count = mesh.cluster_indices.size() / sizeof(uint32_t);
            for (uint32_t cluster_index : selection.clusters)
            {
                if (cluster_index >= cluster_window->data.clusters.size())
                {
                    valid_cluster_draws = false;
                    break;
                }
                const ClusterLodCluster& cluster = cluster_window->data.clusters[cluster_index];
                if (cluster_window->first_index > packed_count ||
                    cluster.first_index > packed_count - cluster_window->first_index ||
                    cluster.index_count > packed_count - cluster_window->first_index - cluster.first_index)
                {
                    valid_cluster_draws = false;
                    break;
                }
            }
            if (valid_cluster_draws)
            {
                for (uint32_t cluster_index : selection.clusters)
                {
                    const ClusterLodCluster& cluster = cluster_window->data.clusters[cluster_index];
                    if (!pass.record_sun_shadow(frame, vertices, mesh.cluster_indices.handle(), cluster.index_count,
                            mvp, mesh.material, mesh.geometry.mode == SurfaceMode::AlphaTest,
                            cluster_window->first_index + cluster.first_index, float(mesh.alpha_ref) / 255.f, 0,
                            descriptor, false, &mesh.matrices))
                    {
                        if (error.empty())
                            error = "shadow clustered model geometry recording failed";
                        return false;
                    }
                }
                continue;
            }
        }
        if (!pass.record_sun_shadow(frame, vertices, mesh.indices.handle(), window.index_count, mvp, mesh.material,
                                    mesh.geometry.mode == SurfaceMode::AlphaTest, window.offset,
                                    float(mesh.alpha_ref) / 255.f, 0, descriptor, false, &mesh.matrices))
        {
            if (error.empty())
                error = "shadow pipeline/buffer mismatch type=" + std::to_string(mesh.geometry.type) +
                        " skinned=" + std::to_string(skinned);
            return false;
        }
    }
    return true;
}

bool GpuModel::record_animated(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16],
                               IKinematics &skeleton, std::string &error, GeometryPhase phase, float lod,
                               const float *normal_rows, int priority_filter, float opacity,
                               const float *world_transform)
{
    static_assert(sizeof(Fmatrix) == 16 * sizeof(float));
    skeleton.CalculateBones();
    const size_t count = skeleton.LL_BoneCount();
    if (!count || count > MaxVulkanSkeletonBones)
    {
        error = "animated OGF model has an invalid bone count";
        return false;
    }
    std::vector<float> pose(count * 16);
    for (size_t bone = 0; bone < count; ++bone)
        std::memcpy(pose.data() + bone * 16, &skeleton.LL_GetBoneInstance(static_cast<u16>(bone)).mRenderTransform,
                    16 * sizeof(float));
    return record(frame, pass, mvp, pose.data(), count, error, phase, lod, &skeleton, normal_rows, priority_filter,
                  opacity, world_transform);
}

bool GpuModel::has_secondary_priority() const
{
    return std::any_of(meshes_.begin(), meshes_.end(), [](const Mesh &mesh) { return mesh.secondary_opaque; });
}

bool GpuModel::has_transparent_priority(uint8_t priority) const
{
    return std::any_of(meshes_.begin(), meshes_.end(), [priority](const Mesh &mesh) {
        return mesh.geometry.mode == SurfaceMode::Transparent && mesh.priority == priority;
    });
}

bool GpuModel::collect_static_ray_geometry(std::vector<RayModelGeometry>& output, std::string& error)
{
    if (!device_ || !ray_address_)
    { error = "model was not loaded with ray geometry support"; return false; }
    if (!wait_for_buffer_uploads(device_, pool_, upload_, pending_))
    { error = "model ray geometry uploads did not complete"; return false; }
    std::vector<RayModelGeometry> result;
    for (const auto& mesh : meshes_)
    {
        if (mesh.geometry.type == 4 || mesh.geometry.type == 5 || mesh.tree_surface || mesh.secondary_opaque ||
            mesh.geometry.mode == SurfaceMode::Transparent)
            continue;
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = mesh.vertices[0].handle();
        const VkDeviceAddress vertices = ray_address_(device_, &info);
        info.buffer = mesh.indices.handle();
        const VkDeviceAddress indices = ray_address_(device_, &info);
        RayIndexedMesh input{};
        input.vertices = vertices;
        input.indices = indices;
        input.vertex_count = static_cast<uint32_t>(mesh.vertices[0].size() / sizeof(LevelVertex));
        input.index_count = static_cast<uint32_t>(mesh.index_count());
        input.vertex_stride = sizeof(LevelVertex);
        input.opaque = mesh.geometry.mode == SurfaceMode::Opaque;
        RayModelGeometry geometry{};
        if (!make_ray_triangle_geometry(input, geometry.geometry, geometry.primitive_count, error)) return false;
        geometry.diffuse_texture = mesh.ray_diffuse_texture;
        geometry.pbr_texture = mesh.ray_pbr_texture;
        geometry.metallic_texture = mesh.ray_metallic_texture;
        geometry.roughness_texture = mesh.ray_roughness_texture;
        geometry.ao_texture = mesh.ray_ao_texture;
        geometry.normal_texture = mesh.ray_normal_texture;
        geometry.normal_aux_texture = mesh.ray_normal_aux_texture;
        geometry.texture_matrices = mesh.matrices;
        geometry.alpha_cutoff = input.opaque ? 0.f : float(mesh.alpha_ref) / 255.f;
        geometry.emissive = mesh.emissive;
        result.push_back(std::move(geometry));
    }
    output = std::move(result);
    error.clear();
    return true;
}

bool GpuModel::prepare_static_ray_blas(const AccelerationStructureDispatch& dispatch,
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error)
{
    if (ray_blas_) { error.clear(); return true; }
    std::vector<RayModelGeometry> metadata;
    if (!collect_static_ray_geometry(metadata, error)) return false;
    if (metadata.empty()) { error.clear(); return true; }
    if (metadata.size() > properties.maxGeometryCount)
    { error = "model ray geometry exceeds device geometry limit"; return false; }
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<uint32_t> counts;
    uint64_t primitive_total = 0;
    for (const auto& mesh : metadata)
    {
        geometries.push_back(mesh.geometry);
        counts.push_back(mesh.primitive_count);
        primitive_total += mesh.primitive_count;
    }
    if (primitive_total > properties.maxPrimitiveCount)
    { error = "model ray geometry exceeds device primitive limit"; return false; }
    auto blas = std::make_unique<AccelerationStructure>();
    if (!blas->initialize(device_, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometries, counts,
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            properties.minAccelerationStructureScratchOffsetAlignment, memory_, dispatch, error) ||
        !build_ray_blas_sync(device_, queue_, pool_, upload_, *blas, geometries, counts, error))
        return false;
    blas->release_scratch();
    ray_blas_ = std::move(blas);
    ray_geometry_ = std::move(metadata);
    error.clear();
    return true;
}

bool GpuModel::prepare_skinned_ray_blas(uint32_t slot, VkCommandBuffer command, IKinematics& skeleton,
    const AccelerationStructureDispatch& dispatch,
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties,
    VkDeviceAddress& address, const std::vector<RayModelGeometry>*& geometry,
    bool& recorded_update, std::string& error)
{
    address = 0; geometry = nullptr; recorded_update = false;
    if (!device_ || !ray_address_ || !command || slot >= FrameContext::FramesInFlight || !upload_.cmd_pipeline_barrier)
    { error = "skinned ray geometry requires an acquired frame and device-address buffers"; return false; }
    const size_t bones = skeleton.LL_BoneCount();
    if (!bones || bones > MaxVulkanSkeletonBones)
    { error = "skinned ray geometry has an invalid bone count"; return false; }
    std::vector<float> pose(bones * 16);
    for (size_t bone = 0; bone < bones; ++bone)
        std::memcpy(pose.data() + bone * 16, &skeleton.LL_GetBoneInstance(static_cast<u16>(bone)).mRenderTransform, 64);
    if (!wait_for_buffer_uploads(device_, pool_, upload_, pending_))
    { error = "skinned ray geometry uploads did not complete"; return false; }
    std::vector<const Mesh*> eligible;
    uint64_t primitive_total = 0;
    for (const auto& mesh : meshes_)
        if ((mesh.geometry.type == 4 || mesh.geometry.type == 5) && !mesh.tree_surface && !mesh.secondary_opaque &&
            mesh.geometry.mode != SurfaceMode::Transparent)
        {
            eligible.push_back(&mesh);
            primitive_total += mesh.index_count() / 3;
        }
    if (eligible.empty()) { error.clear(); return true; }
    if (eligible.size() > properties.maxGeometryCount || primitive_total > properties.maxPrimitiveCount)
    { error = "skinned ray geometry exceeds acceleration structure limits"; return false; }
    auto& state = ray_skin_[&skeleton][slot];
    if (!state.blas)
    {
        state.vertices.clear(); state.geometry.clear();
        state.vertices.resize(eligible.size());
        state.geometry.resize(eligible.size());
    }
    const auto usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<uint32_t> counts;
    for (size_t i = 0; i < eligible.size(); ++i)
    {
        const Mesh& mesh = *eligible[i];
        std::vector<LevelVertex> vertices;
        if (!skin_model_mesh(mesh.geometry, pose.data(), bones, vertices, error)) return false;
        auto& buffer = state.vertices[i];
        if (!buffer.handle() && !buffer.initialize(device_, vertices.size() * sizeof(LevelVertex), usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory_, upload_.buffer, error)) return false;
        if (!buffer.write(0, vertices.data(), vertices.size() * sizeof(LevelVertex), error)) return false;
        VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        info.buffer = buffer.handle();
        const VkDeviceAddress vertex_address = ray_address_(device_, &info);
        info.buffer = mesh.indices.handle();
        RayIndexedMesh input{};
        input.vertices = vertex_address;
        input.indices = ray_address_(device_, &info);
        input.vertex_count = static_cast<uint32_t>(vertices.size());
        input.index_count = static_cast<uint32_t>(mesh.index_count());
        input.vertex_stride = sizeof(LevelVertex);
        input.opaque = mesh.geometry.mode == SurfaceMode::Opaque;
        RayModelGeometry item{};
        if (!make_ray_triangle_geometry(input, item.geometry, item.primitive_count, error)) return false;
        item.diffuse_texture = mesh.ray_diffuse_texture;
        item.pbr_texture = mesh.ray_pbr_texture;
        item.metallic_texture = mesh.ray_metallic_texture;
        item.roughness_texture = mesh.ray_roughness_texture;
        item.ao_texture = mesh.ray_ao_texture;
        item.normal_texture = mesh.ray_normal_texture;
        item.normal_aux_texture = mesh.ray_normal_aux_texture;
        item.texture_matrices = mesh.matrices;
        item.alpha_cutoff = input.opaque ? 0.f : float(mesh.alpha_ref) / 255.f;
        item.emissive = mesh.emissive;
        geometries.push_back(item.geometry);
        counts.push_back(item.primitive_count);
        state.geometry[i] = std::move(item);
    }
    if (!state.blas)
    {
        auto blas = std::make_unique<AccelerationStructure>();
        if (!blas->initialize(device_, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometries, counts,
                VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                properties.minAccelerationStructureScratchOffsetAlignment, memory_, dispatch, error) ||
            !build_ray_blas_sync(device_, queue_, pool_, upload_, *blas, geometries, counts, error)) return false;
        state.blas = std::move(blas);
    }
    else
    {
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        before.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        upload_.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &before, 0, nullptr, 0, nullptr);
        if (!state.blas->record_build(command, geometries, counts, true, error)) return false;
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        after.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        upload_.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &after, 0, nullptr, 0, nullptr);
        recorded_update = true;
    }
    address = state.blas->address();
    geometry = &state.geometry;
    error.clear();
    return true;
}

void GpuModel::release_instance(const IKinematics *skeleton)
{
    ray_skin_.erase(skeleton);
    for (auto &mesh : meshes_)
    {
        auto it = mesh.poses.find(skeleton);
        if (it != mesh.poses.end())
        {
            for (auto &descriptor : it->second.descriptors)
                if (pass_)
                    pass_->release_pose_descriptor(descriptor);
            mesh.poses.erase(it);
        }
        mesh.instance_vertices.erase(skeleton);
    }
}

void GpuModel::destroy()
{
    if (device_)
        wait_for_buffer_uploads(device_, pool_, upload_, pending_);
    if (textures_ && pass_)
        for (auto &mesh : meshes_)
        {
            for (auto &[instance, pose] : mesh.poses)
                for (auto &descriptor : pose.descriptors)
                    pass_->release_pose_descriptor(descriptor);
            if (mesh.environment || mesh.surface)
                textures_->release_terrain_material(mesh.material, *pass_);
            else
                textures_->release_material(mesh.material, *pass_);
        }
    ray_blas_.reset();
    ray_geometry_.clear();
    ray_skin_.clear();
    meshes_.clear();
    pending_.clear();
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    memory_ = {};
    upload_ = {};
    textures_ = nullptr;
    pass_ = nullptr;
    ray_address_ = nullptr;
}
} // namespace xray::render::vulkan
