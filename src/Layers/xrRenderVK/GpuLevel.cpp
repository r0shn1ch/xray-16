#include "xrEngine/stdafx.h"
#include "GpuLevel.h"
#include "ObjectVelocityPass.h"
#include "TangentBasis.h"
#include "DetailInstance.h"
#include "LevelLightmap.h"
#include "VulkanSettings.h"
#include "PackedLevelGeometry.h"
#include "ClusterLodSelection.h"
#include "LevelVisibility.h"
#include "VulkanVisual.h"
#include "SpecialVisuals.h"
#include "SlidingWindows.h"
#include "ModelTessellation.h"
#include "xrEngine/device.h"
#if defined(XR_PLATFORM_ANDROID)
#include "xrEngine/x_ray.h"
#endif
#include "xrEngine/IGame_Level.h"
#include "xrCDB/xr_area.h"
#include "xrCDB/Intersect.hpp"
#include "xrMaterialSystem/GameMtlLib.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_set>
#if defined(XR_PLATFORM_ANDROID)
#include <unistd.h>
#endif

namespace xray::render::vulkan
{
namespace
{
#if defined(XR_PLATFORM_ANDROID)
unsigned long long descriptor_number(VkDescriptorSet set)
{
    uint64_t bits = 0;
    static_assert(sizeof(set) <= sizeof(bits));
    std::memcpy(&bits, &set, sizeof(set));
    return static_cast<unsigned long long>(bits);
}
size_t current_rss_kib()
{
    FILE *statm = std::fopen("/proc/self/statm", "r");
    if (!statm)
        return 0;
    unsigned long total = 0, resident = 0;
    const bool parsed = std::fscanf(statm, "%lu %lu", &total, &resident) == 2;
    std::fclose(statm);
    const long page_size = sysconf(_SC_PAGESIZE);
    return parsed && page_size > 0 ? size_t(resident) * size_t(page_size) / 1024 : 0;
}
#endif
bool visible_sphere(const float (&mvp)[16], float x, float y, float z, float radius)
{
    if (!std::isfinite(radius) || radius <= 0.f)
        return true;
    if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        return true;
    const auto component = [&](int column) {
        return mvp[column] * x + mvp[column + 4] * y + mvp[column + 8] * z + mvp[column + 12];
    };
    const float cx = component(0), cy = component(1), cz = component(2), cw = component(3);
    const auto outside = [&](float distance, float a, float b, float c) {
        return distance < -radius * std::sqrt(a * a + b * b + c * c);
    };
    return !outside(cx + cw, mvp[0] + mvp[3], mvp[4] + mvp[7], mvp[8] + mvp[11]) &&
           !outside(cw - cx, mvp[3] - mvp[0], mvp[7] - mvp[4], mvp[11] - mvp[8]) &&
           !outside(cy + cw, mvp[1] + mvp[3], mvp[5] + mvp[7], mvp[9] + mvp[11]) &&
           !outside(cw - cy, mvp[3] - mvp[1], mvp[7] - mvp[5], mvp[11] - mvp[9]) &&
           !outside(cz, mvp[2], mvp[6], mvp[10]) &&
           !outside(cw - cz, mvp[3] - mvp[2], mvp[7] - mvp[6], mvp[11] - mvp[10]);
}

float lod_billboard_alpha(const LevelVisual& visual, VkExtent2D extent, float fov,
    const Fvector& camera_position)
{
    if (visual.type != 6 || visual.bounds[9] <= 0.f || !extent.width || !extent.height)
        return 0.f;
    const auto& bounds = visual.bounds;
    const float radius = bounds[9];
    std::array<float, 3> half_extents{(bounds[3] - bounds[0]) * .5f,
        (bounds[4] - bounds[1]) * .5f, (bounds[5] - bounds[2]) * .5f};
    std::sort(half_extents.begin(), half_extents.end());
    const float ratio = std::clamp(half_extents[1] / radius, 0.f, 1.f);
    const float factor = (2.f / 3.14159265f) *
        (std::asin(ratio) + ratio * std::sqrt(1.f - ratio * ratio));
    const float dx = bounds[6] - camera_position.x;
    const float dy = bounds[7] - camera_position.y;
    const float dz = bounds[8] - camera_position.z;
    const float fov_scale = 90.f / std::max(fov, 1.f);
    const float screen_area = radius * factor / std::max(dx * dx + dy * dy + dz * dz, 1e-6f) *
        float(extent.width) * float(extent.height) * fov_scale * fov_scale * .75f;
    if (screen_area < 3.5f * 3.5f)
        return -1.f;
    constexpr float near_area = (64.f / 3.f) * (64.f / 3.f);
    constexpr float far_area = (48.f / 3.f) * (48.f / 3.f);
    return std::clamp((near_area - screen_area) / (near_area - far_area), 0.f, 1.f);
}

ClusterLodSelection select_cluster_lod(const ClusterLodData& lod_data,
    const FrameRecordingContext& frame, const float (&mvp)[16], const Fvector& camera, float fov)
{
    const std::array<float, 3> camera_position{camera.x, camera.y, camera.z};
    return select_cluster_lod(lod_data, frame.extent, mvp, nullptr, camera_position, fov);
}

std::string lightmap_texture(const std::string &textures, const MaterialDefinition &definition)
{
    const std::string &cls = definition.class_name;
    if (cls != "LM      " && cls != "LM_AREF " && cls != "BmmD    " && cls != "BmmDold " && cls != "LmBmmD  " &&
        cls != "LmEbB   " && cls != "LaEmB   ")
        return {};
    return select_level_lightmap(textures, cls == "LaEmB   ");
}
} // namespace
GpuLevel::~GpuLevel()
{
    destroy();
}

bool GpuLevel::build_ray_scene(VkQueue queue, const AccelerationStructureDispatch& dispatch,
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error)
{
    return ray_scene_.build(device_, queue, pool_, memory_, upload_, pending_, ray_batches_,
        dispatch, properties, error);
}

void GpuLevel::report_visual_record_failure(size_t visual, int32_t mesh, GeometryPhase phase, bool water,
                                            uint32_t indices, float lod, const char *reason) const
{
    const std::string cause = reason ? reason : "unknown";
    const Mesh *resource = mesh >= 0 && size_t(mesh) < meshes_.size() ? &meshes_[mesh] : nullptr;
    const std::string material_id = resource ? std::to_string(resource->material_id) : "unknown";
    const std::string key = "profile=" + std::string(level_profile_name(profile_)) +
                            " visual=" + std::to_string(visual) + " material=" + material_id + " shader='" +
                            (resource ? resource->shader : "unknown") + "' mesh=" + std::to_string(mesh) +
                            " phase=" + std::to_string(static_cast<int>(phase)) + " reason=" + cause;
    if (!draw_failures_.record(key))
        return;
    Msg("! [renderer-vulkan] profile=%s file=$level$/level.geom visual=%zu material=%s shader='%s' mesh=%d phase=%d "
        "water=%d indices=%u lod=%.3f reason=%s (repeats summarized on unload)",
        level_profile_name(profile_), visual, material_id.c_str(), resource ? resource->shader.c_str() : "unknown",
        mesh, static_cast<int>(phase), water ? 1 : 0, indices, lod, cause.c_str());
}

bool GpuLevel::load(IReader &level, VkDevice device, VkQueue queue, VkCommandPool pool,
                    const VkPhysicalDeviceMemoryProperties &memory, const BufferUploadDispatch &upload,
                    GameTextureFactory &textures, DeferredPass &pass, std::string &error,
                    PFN_vkGetBufferDeviceAddress get_buffer_address,
                    PFN_vkGetDeviceProcAddr get_device_proc, VkDeviceSize max_storage_buffer_range)
{
    if (!device || !queue || !pool)
    {
        error = "Vulkan level requires a device and command pool";
        return false;
    }
    textures.invalidate_metadata();
    LevelModelData models;
    if (!load_engine_level_models(level, models, error))
        return false;
    Msg("[renderer-vulkan] level.geometry decoded models=%zu materials=%zu visuals=%zu", models.models.size(),
        models.materials.size(), models.visuals.size());
    // The export build audits every material before the first GPU upload.
    // One unsupported blender must not hide the rest of the level's material
    // requirements in the next device log.
    if (std::strstr(Core.Params, "-vk_export_shaders_xr"))
    {
        size_t unsupported = 0;
        for (size_t id = 0; id < models.materials.size(); ++id)
        {
            const auto &material = models.materials[id];
            if (material.shader.empty() && material.textures.empty())
                continue; // Reserved table slot.
            SurfaceMode mode{};
            std::string issue;
            if (!textures.surface_mode(material.shader, material.textures, mode, issue, nullptr, nullptr, false, false,
                                       true))
            {
                ++unsupported;
                Msg("! [renderer-vulkan] level.material.audit id=%zu shader='%s' textures='%s': %s", id,
                    material.shader.c_str(), material.textures.c_str(), issue.c_str());
            }
        }
        Msg("[renderer-vulkan] level.material.audit total=%zu unsupported=%zu", models.materials.size(), unsupported);
    }
    DetailAssets details;
    Msg("[renderer-vulkan] level.details decoding");
    if (!load_engine_detail_assets(details, error))
        return false;
    Msg("[renderer-vulkan] level.details decoded prototypes=%zu", details.prototypes.size());
    // Portal data is an optimization input. If an old or incomplete level
    // omits it, the renderer uses the complete root list rather than hiding
    // geometry based on partial visibility information.
    std::string visibility_error;
    if (!load_engine_level_visibility(level, models, visibility_error))
        Msg("! Vulkan: level visibility data unavailable (%s); rendering all level roots", visibility_error.c_str());
    GpuLevel prepared;
    prepared.device_ = device;
    prepared.profile_ = models.profile;
    prepared.memory_ = memory;
    prepared.pool_ = pool;
    prepared.upload_ = upload;
    prepared.textures_ = &textures;
    prepared.pass_ = &pass;
    prepared.visuals_ = std::move(models.visuals);
    prepared.roots_ = std::move(models.roots);
    prepared.sectors_ = std::move(models.sectors);
    prepared.portals_ = std::move(models.portals);
    prepared.meshes_.reserve(models.models.size());
    const auto upload_start = std::chrono::steady_clock::now();
    size_t uploaded_models = 0;
    [[maybe_unused]] bool direct_upload = true;
    const auto upload_geometry = [&](const void *data, size_t bytes, VkBufferUsageFlags usage,
                                     BufferResource &destination) -> bool {
#if defined(XR_PLATFORM_ANDROID)
        if (direct_upload)
        {
            if (destination.initialize(device, bytes, usage,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                       memory, upload.buffer, error, true))
                return destination.write(0, data, bytes, error);
            if (error != "no Vulkan memory type satisfies the buffer requirements")
                return false;
            direct_upload = false;
            Msg("[renderer-vulkan] level.geometry host-visible buffers unavailable; using staging uploads");
        }
#endif
        return upload_buffer(device, queue, pool, memory, upload, data, bytes, usage, destination, prepared.pending_,
                             error, true);
    };
    PackedLevelGeometry packing;
    std::vector<Mesh *> awaiting_batch;
    std::unordered_map<std::string, uint32_t> ray_texture_slots;
    std::unordered_map<std::string, uint32_t> ray_diffuse_texture_slots;
    std::unordered_map<uint32_t, const Mesh *> material_cache;
    std::unordered_set<uint16_t> reported_multiply;
    const auto flush_geometry = [&]() -> bool {
        if (awaiting_batch.empty())
            return true;
        GeometryBatch batch;
        const VkBufferUsageFlags ray_input = get_buffer_address ?
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR : 0;
        if (!upload_geometry(packing.vertices.data(), packing.vertices.size() * sizeof(LevelVertex),
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | ray_input, batch.vertices) ||
            !upload_geometry(packing.indices.data(), packing.indices.size() * sizeof(uint32_t),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT | ray_input, batch.indices))
            return false;
        if (!packing.cluster_indices.empty() &&
            !upload_geometry(packing.cluster_indices.data(), packing.cluster_indices.size() * sizeof(uint32_t),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT, batch.cluster_indices))
            return false;
        RayGeometryBatch ray_batch;
        std::vector<TextureMatrixSet> ray_matrices;
        if (get_buffer_address)
        {
            VkBufferDeviceAddressInfo info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
            info.buffer = batch.vertices.handle();
            const VkDeviceAddress vertices = get_buffer_address(device, &info);
            info.buffer = batch.indices.handle();
            const VkDeviceAddress indices = get_buffer_address(device, &info);
            if (!vertices || !indices)
            {
                error = "level ray geometry device address unavailable";
                wait_for_buffer_uploads(device, pool, upload, prepared.pending_);
                return false;
            }
            for (const Mesh* mesh : awaiting_batch)
            {
                if (mesh->depth_proxy || mesh->tree || mesh->water || mesh->lod_atlas ||
                    mesh->wallmark_multiply || mesh->wallmark_blend ||
                    mesh->mode == SurfaceMode::Transparent)
                    continue;
                RayIndexedMesh input{};
                input.vertices = vertices;
                input.indices = indices;
                input.first_vertex = mesh->first_vertex;
                input.vertex_count = static_cast<uint32_t>(mesh->vertex_bytes / sizeof(LevelVertex));
                input.first_index = mesh->first_index;
                input.index_count = mesh->index_count;
                input.vertex_stride = sizeof(LevelVertex);
                input.opaque = mesh->mode == SurfaceMode::Opaque;
                VkAccelerationStructureGeometryKHR geometry{};
                uint32_t count = 0;
                std::string issue;
                if (!make_ray_triangle_geometry(input, geometry, count, issue))
                {
                    Msg("! [renderer-vulkan] level ray geometry material=%u skipped: %s",
                        mesh->material_id, issue.c_str());
                    continue;
                }
                ray_batch.geometries.push_back(geometry);
                ray_batch.primitive_counts.push_back(count);
                ray_batch.material_ids.push_back(mesh->material_id);
                ray_batch.alpha_cutoffs.push_back(input.opaque ? 0.f : float(mesh->alpha_ref) / 255.f);
                uint32_t texture_slot = UINT32_MAX;
                uint32_t diffuse_texture_slot = UINT32_MAX;
                std::string diffuse_name;
                if (mesh->material_id < models.materials.size())
                {
                    const std::string& list = models.materials[mesh->material_id].textures;
                    const size_t comma = list.find(',');
                    std::string diffuse = list.substr(0, comma);
                    while (!diffuse.empty() && std::isspace(static_cast<unsigned char>(diffuse.back())))
                        diffuse.pop_back();
                    const size_t first = diffuse.find_first_not_of(" \t");
                    if (first != std::string::npos) diffuse.erase(0, first);
                    if (!diffuse.empty())
                    {
                        diffuse_name = diffuse;
                        const auto [diffuse_slot, diffuse_inserted] = ray_diffuse_texture_slots.emplace("C:" + diffuse,
                            static_cast<uint32_t>(prepared.ray_diffuse_texture_names_.size()));
                        if (diffuse_inserted)
                        {
                            prepared.ray_diffuse_texture_names_.push_back(diffuse);
                            prepared.ray_diffuse_texture_linear_.push_back(0);
                        }
                        diffuse_texture_slot = diffuse_slot->second;
                        if (!input.opaque)
                        {
                            const auto [slot, inserted] = ray_texture_slots.emplace(diffuse,
                                static_cast<uint32_t>(prepared.ray_texture_names_.size()));
                            if (inserted) prepared.ray_texture_names_.push_back(diffuse);
                            texture_slot = slot->second;
                        }
                    }
                }
                ray_batch.texture_slots.push_back(texture_slot);
                ray_batch.diffuse_texture_slots.push_back(diffuse_texture_slot);
                uint32_t pbr_slot = UINT32_MAX;
                if (!mesh->pbr_texture.empty())
                {
                    const auto [slot, inserted] = ray_diffuse_texture_slots.emplace("L:" + mesh->pbr_texture,
                        static_cast<uint32_t>(prepared.ray_diffuse_texture_names_.size()));
                    if (inserted)
                    {
                        prepared.ray_diffuse_texture_names_.push_back(mesh->pbr_texture);
                        prepared.ray_diffuse_texture_linear_.push_back(1);
                    }
                    pbr_slot = slot->second;
                }
                ray_batch.pbr_texture_slots.push_back(pbr_slot);
                auto acquire_linear_slot = [&](const std::string& name) -> uint32_t {
                    if (name.empty()) return UINT32_MAX;
                    const auto [slot, inserted] = ray_diffuse_texture_slots.emplace("L:" + name,
                        static_cast<uint32_t>(prepared.ray_diffuse_texture_names_.size()));
                    if (inserted)
                    {
                        prepared.ray_diffuse_texture_names_.push_back(name);
                        prepared.ray_diffuse_texture_linear_.push_back(1);
                    }
                    return slot->second;
                };
                ray_batch.metallic_texture_slots.push_back(acquire_linear_slot(mesh->metallic_texture));
                ray_batch.roughness_texture_slots.push_back(acquire_linear_slot(mesh->roughness_texture));
                ray_batch.ao_texture_slots.push_back(acquire_linear_slot(mesh->ao_texture));
                uint32_t normal_slot = UINT32_MAX;
                if (!mesh->ray_normal_texture.empty())
                {
                    const auto [slot, inserted] = ray_diffuse_texture_slots.emplace("L:" + mesh->ray_normal_texture,
                        static_cast<uint32_t>(prepared.ray_diffuse_texture_names_.size()));
                    if (inserted)
                    {
                        prepared.ray_diffuse_texture_names_.push_back(mesh->ray_normal_texture);
                        prepared.ray_diffuse_texture_linear_.push_back(1);
                    }
                    normal_slot = slot->second;
                }
                ray_batch.normal_texture_slots.push_back(normal_slot);
                uint32_t normal_aux_slot = UINT32_MAX;
                if (!mesh->ray_normal_texture.empty() && mesh->bump_mode >= 2)
                {
                    const std::string aux_name = mesh->ray_normal_texture + "#";
                    const auto [slot, inserted] = ray_diffuse_texture_slots.emplace("L:" + aux_name,
                        static_cast<uint32_t>(prepared.ray_diffuse_texture_names_.size()));
                    if (inserted)
                    {
                        prepared.ray_diffuse_texture_names_.push_back(aux_name);
                        prepared.ray_diffuse_texture_linear_.push_back(1);
                    }
                    normal_aux_slot = slot->second;
                }
                ray_batch.normal_aux_texture_slots.push_back(normal_aux_slot);
                std::array<uint32_t, 33> terrain_slots;
                terrain_slots.fill(UINT32_MAX);
                if (mesh->terrain && !diffuse_name.empty())
                {
                    auto acquire_color_slot = [&](const std::string& name) -> uint32_t {
                        if (name.empty()) return UINT32_MAX;
                        const auto [slot, inserted] = ray_diffuse_texture_slots.emplace("C:" + name,
                            static_cast<uint32_t>(prepared.ray_diffuse_texture_names_.size()));
                        if (inserted)
                        {
                            prepared.ray_diffuse_texture_names_.push_back(name);
                            prepared.ray_diffuse_texture_linear_.push_back(0);
                        }
                        return slot->second;
                    };
                    terrain_slots[0] = acquire_linear_slot(diffuse_name + "_mask");
                    for (size_t layer = 0; layer < mesh->terrain_ray_layers.size(); ++layer)
                    {
                        const auto& source = mesh->terrain_ray_layers[layer];
                        terrain_slots[1 + layer] = acquire_color_slot(source.diffuse);
                        terrain_slots[5 + layer] = acquire_linear_slot(source.pbr);
                        terrain_slots[9 + layer] = acquire_linear_slot(source.metallic);
                        terrain_slots[13 + layer] = acquire_linear_slot(source.roughness);
                        terrain_slots[17 + layer] = acquire_linear_slot(source.ao);
                        terrain_slots[21 + layer] = acquire_linear_slot(source.bump);
                        terrain_slots[25 + layer] = acquire_linear_slot(source.bump_aux);
                        terrain_slots[29 + layer] = acquire_linear_slot(source.parallax);
                    }
                }
                else if (!mesh->terrain)
                    terrain_slots[28] = acquire_linear_slot(mesh->parallax_texture);
                ray_batch.terrain_texture_slots.push_back(terrain_slots);
                ray_batch.terrain_scales.push_back(mesh->terrain ? mesh->terrain_scale : 1.f);
                ray_batch.parallax_heights.push_back(mesh->bump_height);
                ray_batch.material_flags.push_back(mesh->emissive ? RayMaterialEmissive : 0u);
                ray_matrices.push_back(mesh->matrices);
            }
        }
        const VkBuffer vertices = batch.vertices.handle();
        const VkBuffer indices = batch.indices.handle();
        const VkBuffer cluster_indices = batch.cluster_indices.handle();
        prepared.geometry_batches_.push_back(std::move(batch));
        prepared.ray_batches_.push_back(std::move(ray_batch));
        prepared.ray_texture_matrices_.push_back(std::move(ray_matrices));
        for (Mesh *mesh : awaiting_batch)
        {
            mesh->packed_vertices = vertices;
            mesh->packed_indices = indices;
            if (mesh->has_cluster_lod())
                mesh->packed_cluster_indices = cluster_indices;
        }
        awaiting_batch.clear();
        packing.reset();
        return true;
    };
    const auto upload_mesh = [&](LevelModel &model, Mesh &mesh) -> bool {
        if (prepared.pending_.size() >= 128 && !wait_for_buffer_uploads(device, pool, upload, prepared.pending_))
        {
            error = "Vulkan level geometry upload batch did not complete";
            return false;
        }
        if (model.material >= models.materials.size() || model.vertices.empty() || model.indices.empty() ||
            model.indices.size() > std::numeric_limits<uint32_t>::max())
        {
            error = "invalid or empty Vulkan level model";
            return false;
        }
        const auto &material = models.materials[model.material];
        mesh.material_id = model.material;
        mesh.shader = material.shader;
        const auto tessellate = [&]() -> bool
        {
            if (mesh.tree_wave && (mesh.lightmapped || !pack_tree_wind_attributes(model)))
            {
                error = "tree wind requires unused lightmap coordinates and matching rigidity values";
                return false;
            }
            if (mesh.wallmark_multiply || mesh.wallmark_blend)
                offset_level_wallmark(model);
            if (model.lod_pairs) return true;
            if (!mesh.tessellation)
            {
                prepare_tangent_basis(model.vertices, model.indices);
                return true;
            }
            const HeightField *height = nullptr;
            if (mesh.tessellation & 2)
            {
                if (mesh.bump_mode < 2 || mesh.bump_texture.empty())
                {
                    error = "level HM tessellation requires a bump auxiliary texture";
                    return false;
                }
                if (!textures.height_field(mesh.bump_texture + "#", height, error))
                    return false;
            }
            if (!tessellate_level(model, mesh.tessellation, height, mesh.bump_height, error))
                return false;
            prepare_tangent_basis(model.vertices, model.indices);
            return true;
        };
        const auto prepare_cluster_lod = [&]() {
            if (!Core.Params || !std::strstr(Core.Params, "-vk-cluster-dag"))
                return;
            const LevelClusterLodEligibility eligibility{model.lod_pairs, mesh.tree,
                mesh.tree && !mesh.tree_wave, mesh.water, mesh.glass,
                mesh.lod_atlas, mesh.depth_proxy,
                mesh.wallmark_multiply, mesh.wallmark_blend, mesh.mode, &mesh.pass_graph};
            if (!supports_level_cluster_lod(eligibility))
                return;
            std::string cluster_error;
            if (model.windows.empty())
            {
                if (!build_cluster_lod(model, mesh.cluster_lod, cluster_error))
                {
                    Msg("! [renderer-vulkan] cluster DAG bake skipped material=%u vertices=%zu indices=%zu: %s",
                        mesh.material_id, model.vertices.size(), model.indices.size(), cluster_error.c_str());
                    return;
                }
            }
            else
            {
                mesh.cluster_lod_windows.reserve(model.windows.size());
                for (const SlideWindow& source_window : model.windows)
                {
                    SlideWindow cluster_window = source_window;
                    // Tessellation replaces the progressive vertex stream with
                    // per-triangle vertices; the authored active-vertex limit
                    // no longer describes the tessellated indices.
                    if (mesh.tessellation)
                        cluster_window.vertex_count = 0;
                    Mesh::ClusterWindow window;
                    if (!build_cluster_lod(model, cluster_window, window.data, cluster_error) ||
                        window.data.indices.empty())
                    {
                        if (cluster_error.empty())
                            cluster_error = "progressive level window produced no cluster indices";
                        mesh.cluster_lod_windows.clear();
                        Msg("! [renderer-vulkan] cluster DAG window bake skipped material=%u: %s",
                            mesh.material_id, cluster_error.c_str());
                        return;
                    }
                    mesh.cluster_lod_windows.push_back(std::move(window));
                }
            }
            mesh.cluster_index_bytes = mesh.cluster_index_count() * sizeof(uint32_t);
        };
        const auto append_cluster_indices = [&]() -> bool {
            if (!mesh.cluster_lod_windows.empty())
            {
                for (Mesh::ClusterWindow& window : mesh.cluster_lod_windows)
                {
                    if (!append_packed_cluster_indices(packing.cluster_indices, window.data,
                            window.first_index, window.index_count, error))
                        return false;
                }
                return true;
            }
            uint32_t index_count = 0;
            if (!append_packed_cluster_indices(packing.cluster_indices, mesh.cluster_lod,
                    mesh.cluster_first_index, index_count, error))
                return false;
            return true;
        };
        const uint32_t material_key = uint32_t(model.material) |
            (uint32_t(model.lod_pairs) << 16) | (uint32_t(model.tree) << 17);
        if (const auto cached = material_cache.find(material_key); cached != material_cache.end())
        {
            const Mesh &source = *cached->second;
            if (source.lightmapped && !model.lightmap_uv)
            {
                error = "level lightmap material has no TEXCOORD1: " + material.textures;
                return false;
            }
            mesh.material = source.material;
            mesh.mode = source.mode;
            mesh.secondary_opaque = source.secondary_opaque;
            mesh.priority = source.priority;
            mesh.alpha_ref = source.alpha_ref;
            mesh.wallmark_multiply = source.wallmark_multiply;
            mesh.wallmark_blend = source.wallmark_blend;
            mesh.glow_alpha_add = source.glow_alpha_add;
            mesh.fog = source.fog;
            mesh.lightmapped = source.lightmapped;
            mesh.lod_atlas = source.lod_atlas;
            mesh.terrain = source.terrain;
            mesh.surface = source.surface;
            mesh.single_detail = source.single_detail;
            mesh.environment = source.environment;
            mesh.laemb = source.laemb;
            mesh.environment_constant = source.environment_constant;
            mesh.pass_graph = source.pass_graph;
            mesh.constant_waveforms = source.constant_waveforms;
            mesh.matrices = source.matrices;
            mesh.tessellation = source.tessellation;
            mesh.bump_texture = source.bump_texture;
            mesh.pbr_texture = source.pbr_texture;
            mesh.ray_normal_texture = source.ray_normal_texture;
            mesh.terrain_ray_layers = source.terrain_ray_layers;
            mesh.bump_mode = source.bump_mode;
            mesh.bump_height = source.bump_height;
            mesh.terrain_scale = source.terrain_scale;
            mesh.water = source.water;
            mesh.glass = source.glass;
            mesh.tree = model.tree;
            mesh.tree_wave = source.tree_wave;
            prepared.has_water_ |= mesh.water;
            if (!tessellate())
                return false;
            prepare_cluster_lod();
            if (packing.would_flush(model, mesh.cluster_index_count()) && !flush_geometry())
                return false;
            if (!packing.append(model, mesh.first_vertex, mesh.first_index, error))
                return false;
            if (!append_cluster_indices())
                return false;
            awaiting_batch.push_back(&mesh);
            mesh.vertex_bytes = model.vertices.size() * sizeof(LevelVertex);
            mesh.index_bytes = model.indices.size() * sizeof(uint32_t);
            mesh.index_count = static_cast<uint32_t>(model.indices.size());
            mesh.windows = model.windows;
            return true;
        }
        SurfaceMode mode;
        int blend_mode = -1;
        uint32_t material_flags = 0;
        TerrainMaterialTextures terrain;
        MaterialDefinition definition;
        if (!textures.surface_mode(material.shader, material.textures, mode, error, &mesh.alpha_ref, &blend_mode, false,
                                   false, true, false, &material_flags, &terrain, &definition))
        {
            error = "level material id=" + std::to_string(model.material) + " (" + material.shader + "/" +
                    material.textures + "): " + error;
            return false;
        }
        Msg("[renderer-vulkan] level.material id=%u shader='%s' textures='%s' mode=%u priority=%u blend=%d "
            "flags=0x%x lightmap-uv=%d",
            model.material, material.shader.c_str(), material.textures.c_str(), unsigned(mode),
            definition.priority, blend_mode, material_flags, model.lightmap_uv);
        if ((blend_mode == 3 || blend_mode == 4 || blend_mode == 6) && reported_multiply.insert(model.material).second)
            Msg("[renderer-vulkan] level.multiply material=%u shader='%s' textures='%s' blend=%d flags=0x%x",
                model.material, material.shader.c_str(), material.textures.c_str(), blend_mode, material_flags);
        if (definition.class_name == "MODELEbB")
        {
            error = "level material id=" + std::to_string(model.material) + " shader='" + material.shader +
                    "' class='MODELEbB' requires OGF model geometry";
            return false;
        }
        if (definition.class_name == "LaEmB   " && mode != SurfaceMode::Opaque)
        {
            error = "level material id=" + std::to_string(model.material) + " shader='" + material.shader +
                    "' class='LaEmB' requires opaque two-pass rendering";
            return false;
        }
        std::string lightmap = lightmap_texture(material.textures, definition);
        const size_t comma = material.textures.find(',');
        const std::string diffuse = material.textures.substr(0, comma);
        std::string detail_texture = definition.detail_texture;
        GameTextureFactory::DetailMetadata metadata;
        const bool implicit_detail = definition.class_name == "LM      " || definition.class_name == "LM_AREF " ||
                                     definition.class_name == "V       " || definition.class_name == "V_AREF  ";
        if (!textures.texture_detail_metadata(diffuse, metadata, error))
        {
            error =
                "level material id=" + std::to_string(model.material) + " shader='" + material.shader + "': " + error;
            return false;
        }
        if (implicit_detail && (metadata.diffuse || metadata.normal))
            detail_texture = metadata.texture;
        mesh.secondary_opaque = definition.priority >= 2 && definition.mode == SurfaceMode::Opaque;
        mesh.priority = definition.priority;
        if (mesh.secondary_opaque)
            mode = SurfaceMode::Transparent;
        if (mode == SurfaceMode::Transparent && !mesh.secondary_opaque && detail_texture.empty() &&
            definition.class_name != "LmEbB   ")
            lightmap.clear();
        if ((terrain.enabled || !definition.detail_texture.empty() || definition.class_name == "LmEbB   " ||
             definition.class_name == "LaEmB   ") &&
            lightmap.empty())
        {
            error = "level material id=" + std::to_string(model.material) + " shader='" + material.shader +
                    "' requires lightmap texture slot 1: " + material.textures;
            return false;
        }
        if (!detail_texture.empty() && mode == SurfaceMode::AlphaTest && lightmap.empty() &&
            definition.class_name != "V_AREF  ")
        {
            error = "level material id=" + std::to_string(model.material) + " shader='" + material.shader +
                    "': cutout detail requires lightmap texture slot 1";
            return false;
        }
        if (!lightmap.empty() && !model.lightmap_uv && !terrain.enabled)
        {
            error = "level lightmap material has no TEXCOORD1: " + material.textures;
            return false;
        }
        mesh.wallmark_multiply = blend_mode == 6;
        mesh.wallmark_blend = definition.role == MaterialDefinition::Role::WallmarkBlend;
        mesh.glow_alpha_add = blend_mode == 5;
        mesh.emissive = definition.emissive;
        mesh.fog = definition.class_name == "S_SET   " && definition.pass.fog;
        mesh.laemb = definition.class_name == "LaEmB   ";
        mesh.environment = definition.class_name == "LmEbB   " || mesh.laemb;
        mesh.environment_constant = definition.has_environment_constant;
        mesh.constant_waveforms = definition.constant_waveforms;
        mesh.matrices = definition.matrices;
        mesh.tessellation = tessellation_enabled ? definition.tessellation : 0;
        mesh.bump_texture = metadata.bump_texture;
        mesh.bump_mode = metadata.bump_mode;
        mesh.bump_height = metadata.bump_height;
        mesh.lightmapped = !lightmap.empty();
        mesh.terrain = terrain.enabled;
        if (!mesh.terrain)
        {
            mesh.parallax_texture = metadata.parallax_texture;
            mesh.pbr_texture = metadata.pbr_texture;
            mesh.metallic_texture = metadata.metallic_texture;
            mesh.roughness_texture = metadata.roughness_texture;
            mesh.ao_texture = metadata.ao_texture;
        }
        if (mode != SurfaceMode::Transparent)
            mesh.ray_normal_texture = metadata.bump_mode >= 2 ? metadata.bump_texture : metadata.external_normal_texture;
        // The four-channel impl pass ignores the serialized Detail map/Name;
        // slot 2 belongs to its mask, slots 3–6 to R2-R/G/B/A.
        mesh.single_detail = !detail_texture.empty() && !mesh.terrain;
        mesh.water = mode == SurfaceMode::Transparent && definition.role == MaterialDefinition::Role::Water;
        mesh.glass = false;
        prepared.has_water_ |= mesh.water;
        const uint16_t roles = (metadata.bump_mode >= 2 ? BumpRole | BumpAuxRole | GlossRole : 0u) |
                               (!metadata.external_normal_texture.empty() || metadata.normal ? NormalRole : 0u);
        if (mesh.secondary_opaque)
            mesh.pass_graph = secondary_opaque_level_graph(roles, mesh.lightmapped, mesh.single_detail);
        else if (!build_render_pass_graph(LevelPassSelection{mode, mesh.lightmapped, mesh.terrain, mesh.single_detail,
                                                             mesh.environment, mesh.laemb, mesh.wallmark_multiply,
                                                             mesh.glow_alpha_add, mesh.fog, roles,
                                                             definition.class_name == "S_SET   " &&
                                                                     definition.role != MaterialDefinition::Role::Lod
                                                                 ? blend_mode
                                                                 : -1,
                                                             definition.object_lod || model.tree},
                                          mesh.pass_graph, error))
        {
            error =
                "level / material id=" + std::to_string(model.material) + " shader='" + material.shader + "': " + error;
            return false;
        }
        if (mesh.wallmark_blend)
            mesh.pass_graph.passes[0].fragment = "vk\\wallmark_blended.ps";
        mesh.tree_wave = model.tree && definition.class_name == "D_TREE  " && !definition.object_lod;
        if (mesh.tree_wave) mesh.pass_graph.passes[0].vertex = "vk\\level_tree_w.vs";
        for (uint8_t index = 0; index < mesh.pass_graph.count; ++index)
        {
            auto &node = mesh.pass_graph.passes[index];
            if (!pass.request_game_pipeline(node.vertex, node.fragment, mode, false, false, error,
                    node.shader_defines, &node.shader_variant_hash))
            {
                error = "level / material id=" + std::to_string(model.material) + " shader='" + material.shader +
                        "' pass=" + std::to_string(index) + " shader pair: " + error;
                return false;
            }
        }
        if (mesh.terrain)
            mesh.terrain_scale = metadata.scale;
        if (mesh.single_detail)
            mesh.terrain_scale = metadata.scale;
        const bool surface_roles = !model.lod_pairs && definition.role != MaterialDefinition::Role::Lod &&
            (metadata.bump_mode >= 2 || metadata.normal || !metadata.external_normal_texture.empty() ||
             metadata.material != 1 || metadata.material_weight != 0.f || mesh.terrain ||
             (!mesh.pbr_texture.empty() || !mesh.metallic_texture.empty() ||
              !mesh.roughness_texture.empty() || !mesh.ao_texture.empty() ||
              !metadata.parallax_texture.empty()) ||
             definition.emissive);
        mesh.surface = surface_roles;
        bool surface_ready = false;
        if (surface_roles)
        {
            std::array<std::string, SurfaceMaterialTextureCount> names;
            names.fill(diffuse);
            names[1] = lightmap.empty() ? diffuse : lightmap;
            if (mesh.environment)
                names[2] = definition.environment_texture;
            uint32_t terrain_map_flags = 0u;
            if (mesh.terrain)
            {
                names[2] = diffuse + "_mask";
                for (size_t channel = 0; channel < 4; ++channel)
                {
                    names[3 + channel] = terrain.detail[channel];
                    GameTextureFactory::DetailMetadata channel_metadata;
                    if (!textures.texture_detail_metadata(terrain.detail[channel], channel_metadata, error))
                        return false;
                    auto& ray_layer = mesh.terrain_ray_layers[channel];
                    ray_layer.diffuse = terrain.detail[channel];
                    ray_layer.pbr = channel_metadata.pbr_texture;
                    ray_layer.metallic = channel_metadata.metallic_texture;
                    ray_layer.roughness = channel_metadata.roughness_texture;
                    ray_layer.ao = channel_metadata.ao_texture;
                    ray_layer.parallax = channel_metadata.parallax_texture;
                    if (channel_metadata.bump_mode >= 2)
                    {
                        ray_layer.bump = channel_metadata.bump_texture;
                        ray_layer.bump_aux = channel_metadata.bump_texture + "#";
                        names[11 + channel] = channel_metadata.bump_texture;
                        names[16 + channel] = channel_metadata.bump_texture + "#";
                        terrain_map_flags |= 1u << channel;
                    }
                    if (!channel_metadata.pbr_texture.empty())
                    {
                        names[20 + channel] = channel_metadata.pbr_texture;
                        terrain_map_flags |= 1u << (4 + channel);
                    }
                    if (!channel_metadata.metallic_texture.empty())
                    {
                        names[24 + channel] = channel_metadata.metallic_texture;
                        terrain_map_flags |= 1u << (8 + channel);
                    }
                    if (!channel_metadata.roughness_texture.empty())
                    {
                        names[28 + channel] = channel_metadata.roughness_texture;
                        terrain_map_flags |= 1u << (12 + channel);
                    }
                    if (!channel_metadata.ao_texture.empty())
                    {
                        names[32 + channel] = channel_metadata.ao_texture;
                        terrain_map_flags |= 1u << (16 + channel);
                    }
                    if (!channel_metadata.parallax_texture.empty())
                    {
                        names[37 + channel] = channel_metadata.parallax_texture;
                        terrain_map_flags |= 1u << (20 + channel);
                    }
                }
            }
            if (mesh.single_detail)
            {
                for (size_t slot = 2; slot <= 6; ++slot)
                    names[slot] = detail_texture;
            }
            if (metadata.bump_mode >= 2)
            {
                names[7] = metadata.bump_texture;
                names[8] = metadata.bump_texture + "#";
            }
            if (!metadata.external_normal_texture.empty())
                names[15] = metadata.external_normal_texture;
            uint32_t surface_map_flags = metadata.external_normal_texture.empty() ? 0u : SurfaceMapExternalNormal;
            if (!mesh.terrain)
            {
                if (!metadata.parallax_texture.empty())
                {
                    names[36] = metadata.parallax_texture;
                    surface_map_flags |= SurfaceMapParallax;
                }
                if (!mesh.pbr_texture.empty())
                {
                    names[19] = mesh.pbr_texture;
                    surface_map_flags |= SurfaceMapPackedPbr;
                }
                if (!mesh.metallic_texture.empty())
                {
                    names[16] = mesh.metallic_texture;
                    surface_map_flags |= SurfaceMapMetallic;
                }
                if (!mesh.roughness_texture.empty())
                {
                    names[17] = mesh.roughness_texture;
                    surface_map_flags |= SurfaceMapRoughness;
                }
                if (!mesh.ao_texture.empty())
                {
                    names[18] = mesh.ao_texture;
                    surface_map_flags |= SurfaceMapAo;
                }
            }
            SurfaceParameters surface;
            surface.bump_mode = float(metadata.bump_mode >= 2 ? metadata.bump_mode : 0);
            surface.external_normal = float(surface_map_flags);
            surface.material = float(metadata.material);
            surface.material_weight = metadata.material_weight;
            surface.bump_height = metadata.bump_height;
            surface.emissive = definition.emissive ? 1.f : 0.f;
            if (mesh.environment)
                surface.terrain_bump = metadata.scale;
            if (mesh.terrain)
                surface.terrain_bump = float(terrain_map_flags);
            if (metadata.normal && (mesh.single_detail || mesh.environment))
            {
                GameTextureFactory::DetailMetadata detail_metadata;
                const std::string normal_detail = mesh.environment ? metadata.texture : detail_texture;
                if (!textures.texture_detail_metadata(normal_detail, detail_metadata, error))
                    return false;
                if (mesh.environment)
                    names[3] = normal_detail;
                names[9] = detail_metadata.bump_mode >= 2 ? detail_metadata.bump_texture : normal_detail + "_bump";
                names[10] = names[9] + "#";
                surface.detail_normal = metadata.diffuse ? 2.f : 1.f;
            }
            surface_ready =
                textures.surface_material(names, surface, pass, mesh.material, error, mesh.environment && !mesh.laemb,
                                          definition.class_name == "S_SET   " && definition.pass.clamp);
        }
        if (model.lod_pairs && textures.lightmapped_material(diffuse, diffuse + "_nm", pass, mesh.material, error, true))
            mesh.lod_atlas = true;
        const bool material_ready = mesh.lod_atlas ||
            (surface_roles        ? surface_ready
            : mesh.terrain       ? textures.terrain_material(diffuse, lightmap, terrain, pass, mesh.material, error)
            : mesh.single_detail ? textures.detail_material(diffuse, lightmap.empty() ? diffuse : lightmap,
                                                            detail_texture, pass, mesh.material, error)
            : mesh.environment   ? textures.environment_material(diffuse, lightmap, definition.environment_texture,
                                                                 !mesh.laemb, pass, mesh.material, error)
            : lightmap.empty()   ? textures.material(material.textures, pass, mesh.material, error,
                                                     definition.class_name == "S_SET   " && definition.pass.clamp,
                                                     definition.role == MaterialDefinition::Role::Lod)
                                 : textures.lightmapped_material(diffuse, lightmap, pass, mesh.material, error));
        if (!material_ready)
        {
            error = "level material id=" + std::to_string(model.material) + " shader='" + material.shader +
                    "' textures='" + material.textures + "': " + error;
            return false;
        }
        mesh.owns_material = true;
        mesh.mode = mode;
        mesh.tree = model.tree;
        if (!tessellate())
            return false;
        prepare_cluster_lod();
        if (packing.would_flush(model, mesh.cluster_index_count()) && !flush_geometry())
        {
            error = "level material id=" + std::to_string(model.material) + " geometry batch upload: " + error;
            return false;
        }
        if (!packing.append(model, mesh.first_vertex, mesh.first_index, error))
            return false;
        if (!append_cluster_indices())
            return false;
        awaiting_batch.push_back(&mesh);
        mesh.vertex_bytes = model.vertices.size() * sizeof(LevelVertex);
        mesh.index_bytes = model.indices.size() * sizeof(uint32_t);
        mesh.index_count = static_cast<uint32_t>(model.indices.size());
        mesh.windows = model.windows;
        material_cache.emplace(material_key, &mesh);
        return true;
    };
    size_t released_cpu_bytes = 0;
    const auto describe_upload_failure = [&](const LevelModel &model, size_t mesh_id, const char *geometry) {
        std::string context = "profile=" + std::string(level_profile_name(models.profile)) +
                              " file=$level$/level.geom mesh=" + std::to_string(mesh_id) + " geometry=" + geometry +
                              " material_id=" + std::to_string(model.material);
        if (model.material < models.materials.size())
            context += " shader='" + models.materials[model.material].shader + "' textures='" +
                       models.materials[model.material].textures + "'";
        error = context + " / GPU preparation: " + error;
    };
    if (!pass.request_game_pipeline("vk\\level_lod.vs", "vk\\level_lod.ps", SurfaceMode::AlphaTest, false, false, error))
        return false;
    for (LevelModel &model : models.models)
    {
        if (uploaded_models % 64 == 0)
            Msg("[renderer-vulkan] level.geometry progress=%zu/%zu pending=%zu elapsed-ms=%lld", uploaded_models,
                models.models.size(), prepared.pending_.size(),
                static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                           std::chrono::steady_clock::now() - upload_start)
                                           .count()));
        prepared.meshes_.emplace_back();
        Mesh &mesh = prepared.meshes_.back();
        if (!upload_mesh(model, mesh))
        {
            describe_upload_failure(model, uploaded_models, "regular");
            return false;
        }
        if (mesh.tree_wave)
            for (const auto& vertex : model.vertices)
            {
                const float extent = std::abs((vertex.position[1] - vertex.lightmap_uv[1]) * vertex.lightmap_uv[0]);
                if (!std::isfinite(extent)) { mesh.tree_wind_extent = std::numeric_limits<float>::infinity(); break; }
                mesh.tree_wind_extent = std::max(mesh.tree_wind_extent, extent);
            }
        if (model.fast)
        {
            mesh.fast = std::make_unique<Mesh>();
            Mesh &fast = *mesh.fast;
            // geomX is a depth proxy. It has no lightmap/normal/UV contract;
            // share the regular material and bypass all shading/tessellation.
            fast.material = mesh.material;
            fast.material_id = mesh.material_id;
            fast.mode = mesh.mode;
            fast.alpha_ref = mesh.alpha_ref;
            fast.windows = model.fast->windows;
            fast.depth_proxy = true;
            if (Core.Params && std::strstr(Core.Params, "-vk-cluster-dag"))
            {
                std::string cluster_error;
                bool cluster_valid = true;
                if (model.fast->windows.empty())
                    cluster_valid = build_cluster_lod(*model.fast, fast.cluster_lod, cluster_error);
                else
                {
                    fast.cluster_lod_windows.reserve(model.fast->windows.size());
                    for (const SlideWindow& source_window : model.fast->windows)
                    {
                        Mesh::ClusterWindow window;
                        if (!build_cluster_lod(*model.fast, source_window, window.data, cluster_error) ||
                            window.data.indices.empty())
                        {
                            if (cluster_error.empty())
                                cluster_error = "progressive depth proxy produced no cluster indices";
                            cluster_valid = false;
                            break;
                        }
                        fast.cluster_lod_windows.push_back(std::move(window));
                    }
                }
                if (!cluster_valid)
                {
                    fast.cluster_lod = {};
                    fast.cluster_lod_windows.clear();
                    Msg("! [renderer-vulkan] depth-proxy cluster bake skipped material=%u: %s",
                        mesh.material_id, cluster_error.c_str());
                }
                else
                {
                    const size_t cluster_index_count = fast.cluster_index_count();
                    if (cluster_index_count > std::numeric_limits<size_t>::max() / sizeof(uint32_t))
                    {
                        fast.cluster_lod = {};
                        fast.cluster_lod_windows.clear();
                        Msg("! [renderer-vulkan] depth-proxy cluster index size overflow material=%u",
                            mesh.material_id);
                    }
                    else
                        fast.cluster_index_bytes = cluster_index_count * sizeof(uint32_t);
                }
            }
            if (packing.would_flush(*model.fast, fast.cluster_index_count()) && !flush_geometry())
                return false;
            if (!packing.append(*model.fast, fast.first_vertex, fast.first_index, error))
            {
                describe_upload_failure(*model.fast, uploaded_models, "fast level.geomX");
                return false;
            }
            if (!fast.cluster_lod_windows.empty())
            {
                for (Mesh::ClusterWindow& window : fast.cluster_lod_windows)
                    if (!append_packed_cluster_indices(packing.cluster_indices, window.data,
                            window.first_index, window.index_count, error))
                        return false;
            }
            else if (!fast.cluster_lod.indices.empty())
            {
                uint32_t cluster_index_count = 0;
                if (!append_packed_cluster_indices(packing.cluster_indices, fast.cluster_lod,
                        fast.cluster_first_index, cluster_index_count, error))
                    return false;
            }
            fast.index_count = static_cast<uint32_t>(model.fast->indices.size());
            fast.vertex_bytes = model.fast->vertices.size() * sizeof(LevelVertex);
            fast.index_bytes = model.fast->indices.size() * sizeof(uint32_t);
            awaiting_batch.push_back(&fast);
            released_cpu_bytes += model.fast->vertices.capacity() * sizeof(LevelVertex) +
                                  model.fast->indices.capacity() * sizeof(uint32_t) +
                                  fast.cluster_index_bytes;
            model.fast.reset();
        }
        // The CPU batch now owns a copy. Release each decoded mesh while the
        // batch is assembled, before allocating its Vulkan buffers.
        released_cpu_bytes +=
            model.vertices.capacity() * sizeof(LevelVertex) + model.indices.capacity() * sizeof(uint32_t) +
            mesh.cluster_index_bytes;
        std::vector<LevelVertex>().swap(model.vertices);
        std::vector<uint32_t>().swap(model.indices);
        std::vector<SlideWindow>().swap(model.windows);
        std::vector<float>().swap(model.tree_wind_weights);
        ++uploaded_models;
    }
    if (!flush_geometry())
    {
        error = "level geometry final batch upload: " + error;
        return false;
    }
    uint64_t cluster_draw_count = 0;
    for (Mesh& mesh : prepared.meshes_)
    {
        const auto reserve_commands = [&](size_t cluster_count, uint32_t& first_command) {
            if (!cluster_count)
                return true;
            if (cluster_draw_count > std::numeric_limits<uint32_t>::max() ||
                cluster_count > std::numeric_limits<uint32_t>::max() - cluster_draw_count)
                return false;
            first_command = static_cast<uint32_t>(cluster_draw_count);
            cluster_draw_count += cluster_count;
            return true;
        };
        if (!mesh.cluster_lod.groups.empty() && !mesh.cluster_lod.clusters.empty() &&
            !reserve_commands(mesh.cluster_lod.clusters.size(), mesh.cluster_first_draw_command))
        {
            cluster_draw_count = 0;
            break;
        }
        bool valid = true;
        for (Mesh::ClusterWindow& window : mesh.cluster_lod_windows)
        {
            if (!window.data.groups.empty() && !window.data.clusters.empty() &&
                !reserve_commands(window.data.clusters.size(), window.first_draw_command))
            {
                valid = false;
                break;
            }
        }
        if (valid && mesh.fast)
        {
            Mesh& fast = *mesh.fast;
            if (!fast.cluster_lod.groups.empty() && !fast.cluster_lod.clusters.empty() &&
                !reserve_commands(fast.cluster_lod.clusters.size(), fast.cluster_first_draw_command))
                valid = false;
            for (Mesh::ClusterWindow& window : fast.cluster_lod_windows)
            {
                if (!window.data.groups.empty() && !window.data.clusters.empty() &&
                    !reserve_commands(window.data.clusters.size(), window.first_draw_command))
                {
                    valid = false;
                    break;
                }
            }
        }
        if (!valid)
        {
            cluster_draw_count = 0;
            break;
        }
    }
    prepared.cluster_draw_command_count_ = static_cast<uint32_t>(cluster_draw_count);
    if (cluster_draw_count && pass.supports_multi_draw_indirect() && get_device_proc && max_storage_buffer_range)
    {
        std::vector<ClusterCullGroup> cull_groups;
        std::vector<ClusterCullRecord> cull_clusters;
        std::vector<ClusterCullSpan> cull_spans;
        bool metadata_valid = true;
        const auto append_hierarchy = [&](const ClusterLodData& data, uint32_t command_first,
            uint32_t first_index, int32_t vertex_offset, uint32_t& group_first, uint32_t& group_count,
            uint32_t& cluster_first, uint32_t& cluster_count) -> bool
        {
            if (data.groups.empty() || data.clusters.empty())
                return true;
            if (data.groups.size() > std::numeric_limits<uint32_t>::max() ||
                data.clusters.size() > std::numeric_limits<uint32_t>::max() ||
                cull_groups.size() > std::numeric_limits<uint32_t>::max() - data.groups.size() ||
                cull_clusters.size() > std::numeric_limits<uint32_t>::max() - data.clusters.size())
                return false;
            ClusterCullSpan span{};
            span.group_first = static_cast<uint32_t>(cull_groups.size());
            span.group_count = static_cast<uint32_t>(data.groups.size());
            span.cluster_first = static_cast<uint32_t>(cull_clusters.size());
            span.cluster_count = static_cast<uint32_t>(data.clusters.size());
            span.command_first = command_first;
            span.first_index = first_index;
            span.vertex_offset = vertex_offset;
            group_first = span.group_first;
            group_count = span.group_count;
            cluster_first = span.cluster_first;
            cluster_count = span.cluster_count;

            std::vector<int32_t> cluster_owners(data.clusters.size(), -1);
            for (uint32_t group_index = 0; group_index < span.group_count; ++group_index)
            {
                const ClusterLodGroup& source = data.groups[group_index];
                if (source.first_cluster > cluster_owners.size() ||
                    source.cluster_count > cluster_owners.size() - source.first_cluster)
                    return false;
                ClusterCullGroup target{};
                std::copy_n(source.center, 3, target.bounds);
                target.bounds[3] = source.radius;
                target.error = source.error;
                cull_groups.push_back(target);
                for (uint32_t i = 0; i < source.cluster_count; ++i)
                    cluster_owners[source.first_cluster + i] = static_cast<int32_t>(group_index);
            }
            for (size_t i = 0; i < data.clusters.size(); ++i)
            {
                const ClusterLodCluster& source = data.clusters[i];
                ClusterCullRecord target{};
                std::copy_n(source.center, 3, target.bounds);
                target.bounds[3] = source.radius;
                target.refined_group = source.refined_group >= 0 &&
                    size_t(source.refined_group) < span.group_count ? source.refined_group : -1;
                target.owner_group = cluster_owners[i];
                target.first_index = source.first_index;
                target.index_count = source.index_count;
                cull_clusters.push_back(target);
            }
            cull_spans.push_back(span);
            return true;
        };
        for (Mesh& mesh : prepared.meshes_)
        {
            if (!append_hierarchy(mesh.cluster_lod, mesh.cluster_first_draw_command,
                    mesh.cluster_first_index, static_cast<int32_t>(mesh.first_vertex),
                    mesh.cluster_cull_group_first, mesh.cluster_cull_group_count,
                    mesh.cluster_cull_cluster_first, mesh.cluster_cull_cluster_count))
            {
                metadata_valid = false;
                break;
            }
            for (Mesh::ClusterWindow& window : mesh.cluster_lod_windows)
            {
                if (!append_hierarchy(window.data, window.first_draw_command, window.first_index,
                        static_cast<int32_t>(mesh.first_vertex), window.cull_group_first,
                        window.cull_group_count, window.cull_cluster_first, window.cull_cluster_count))
                {
                    metadata_valid = false;
                    break;
                }
            }
            if (metadata_valid && mesh.fast)
            {
                Mesh& fast = *mesh.fast;
                if (!append_hierarchy(fast.cluster_lod, fast.cluster_first_draw_command,
                        fast.cluster_first_index, static_cast<int32_t>(fast.first_vertex),
                        fast.cluster_cull_group_first, fast.cluster_cull_group_count,
                        fast.cluster_cull_cluster_first, fast.cluster_cull_cluster_count))
                    metadata_valid = false;
                for (Mesh::ClusterWindow& window : fast.cluster_lod_windows)
                {
                    if (!metadata_valid || !append_hierarchy(window.data, window.first_draw_command,
                            window.first_index, static_cast<int32_t>(fast.first_vertex),
                            window.cull_group_first, window.cull_group_count,
                            window.cull_cluster_first, window.cull_cluster_count))
                    {
                        metadata_valid = false;
                        break;
                    }
                }
            }
            if (!metadata_valid)
                break;
        }
        if (metadata_valid && !cull_spans.empty() &&
            !prepared.cluster_visibility_.initialize(device, get_device_proc, pass.pipeline_cache(), memory,
                upload.buffer, cull_groups, cull_clusters, cull_spans,
                static_cast<uint32_t>(cluster_draw_count), max_storage_buffer_range, error))
        {
            Msg("! [renderer-vulkan] GPU cluster visibility unavailable; retaining CPU selection: %s", error.c_str());
            error.clear();
        }
        else if (!metadata_valid)
            Msg("! [renderer-vulkan] GPU cluster visibility metadata is invalid; retaining CPU selection");
    }
    if (cluster_draw_count && pass.supports_multi_draw_indirect() && !prepared.cluster_visibility_.ready())
    {
        if (cluster_draw_count > std::numeric_limits<VkDeviceSize>::max() /
                sizeof(VkDrawIndexedIndirectCommand))
        {
            Msg("! [renderer-vulkan] cluster DAG indirect argument size overflow; retaining CPU draws");
        }
        else
        {
            const VkDeviceSize bytes = cluster_draw_count * sizeof(VkDrawIndexedIndirectCommand);
            for (BufferResource& commands : prepared.cluster_draw_commands_)
            {
                if (!commands.initialize(device, bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        memory, upload.buffer, error, true))
                {
                    for (BufferResource& allocated : prepared.cluster_draw_commands_)
                        allocated.destroy();
                    error.clear();
                    Msg("! [renderer-vulkan] cluster DAG indirect buffers unavailable; retaining CPU draws");
                    cluster_draw_count = 0;
                    break;
                }
            }
        }
    }
    const size_t geometry_buffers = prepared.geometry_batches_.size() * 2 +
        std::count_if(prepared.geometry_batches_.begin(), prepared.geometry_batches_.end(),
            [](const GeometryBatch& batch) { return batch.cluster_indices.handle() != VK_NULL_HANDLE; });
    Msg("[renderer-vulkan] level.geometry batches=%zu vk-buffers=%zu materials=%zu (regular/fast meshes=%zu)",
        prepared.geometry_batches_.size(), geometry_buffers, material_cache.size(), uploaded_models);
    Msg("[renderer-vulkan] level.geometry uploaded=%zu elapsed-ms=%lld pending=%zu cpu-released=%zu", uploaded_models,
        static_cast<long long>(
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - upload_start)
                .count()),
        prepared.pending_.size(), released_cpu_bytes);
#if defined(XR_PLATFORM_ANDROID)
    Msg("[renderer-vulkan] level.geometry resident-current=%zuK resident-peak=%zuK", current_rss_kib(),
        Memory.mem_usage() / 1024);
#endif
    prepared.details_ = std::move(details);
    prepared.detail_meshes_.reserve(prepared.details_.prototypes.size());
    for (const DetailPrototype &prototype : prepared.details_.prototypes)
    {
        if (prepared.pending_.size() >= 128 && !wait_for_buffer_uploads(device, pool, upload, prepared.pending_))
        {
            error = "Vulkan level detail upload batch did not complete";
            return false;
        }
        prepared.detail_meshes_.emplace_back();
        Mesh &mesh = prepared.detail_meshes_.back();
        MaterialDefinition detail_definition;
        if (!textures.surface_mode(prototype.shader, prototype.texture, mesh.mode, error, &mesh.alpha_ref, nullptr,
                                   false, false, true, false, nullptr, nullptr, &detail_definition))
        {
            error = "level.details / " + prototype.shader + ": " + error;
            return false;
        }
        mesh.emissive = detail_definition.emissive;
        mesh.matrices = detail_definition.matrices;
        GameTextureFactory::DetailMetadata metadata;
        if (!textures.texture_detail_metadata(prototype.texture, metadata, error))
        {
            error = "level.details / " + prototype.texture + ": " + error;
            return false;
        }
        mesh.single_detail = metadata.diffuse;
        mesh.terrain_scale = metadata.scale;
        mesh.bump_height = metadata.bump_height;
        mesh.parallax_texture = metadata.parallax_texture;
        const bool surface_roles = metadata.bump_mode >= 2 || metadata.normal ||
                                   !metadata.external_normal_texture.empty() || metadata.material != 1 ||
                                   metadata.material_weight != 0.f ||
                                   (!metadata.pbr_texture.empty() || !metadata.metallic_texture.empty() ||
                                    !metadata.roughness_texture.empty() || !metadata.ao_texture.empty() ||
                                    !metadata.parallax_texture.empty()) ||
                                   detail_definition.emissive;
        mesh.surface = surface_roles;
        const uint16_t roles = (metadata.bump_mode >= 2 ? BumpRole | BumpAuxRole | GlossRole : 0u) |
                               (metadata.normal || !metadata.external_normal_texture.empty() ? NormalRole : 0u);
        if (!build_vegetation_pass_graph(mesh.mode, mesh.single_detail, roles, mesh.pass_graph, error))
        {
            error = "level.details / " + prototype.shader + ": " + error;
            return false;
        }
        if (mesh.mode != SurfaceMode::Transparent)
            mesh.pass_graph.passes[0].vertex = "vk\\level_detail_instanced.vs";
        auto &node = mesh.pass_graph.passes[0];
        bool material_ready = false;
        if (surface_roles)
        {
            std::array<std::string, SurfaceMaterialTextureCount> names;
            names.fill(prototype.texture);
            if (mesh.single_detail)
                for (size_t slot = 2; slot <= 6; ++slot)
                    names[slot] = metadata.texture;
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
            surface.emissive = detail_definition.emissive ? 1.f : 0.f;
            if (metadata.normal)
            {
                GameTextureFactory::DetailMetadata detail;
                if (!textures.texture_detail_metadata(metadata.texture, detail, error))
                {
                    error = "level.details / " + metadata.texture + ": " + error;
                    return false;
                }
                names[9] = detail.bump_mode >= 2 ? detail.bump_texture : metadata.texture + "_bump";
                names[10] = names[9] + "#";
                surface.detail_normal = mesh.single_detail ? 2.f : 1.f;
            }
            material_ready = textures.surface_material(names, surface, pass, mesh.material, error);
        }
        else if (mesh.single_detail)
            material_ready = textures.detail_material(prototype.texture, prototype.texture, metadata.texture, pass,
                                                      mesh.material, error);
        else
            material_ready = textures.material(prototype.texture, pass, mesh.material, error);
        if (!pass.request_game_pipeline(node.vertex, node.fragment, mesh.mode, false, false, error,
                node.shader_defines, &node.shader_variant_hash) ||
            !material_ready ||
            !upload_geometry(prototype.vertices.data(), prototype.vertices.size() * sizeof(LevelVertex),
                             VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, mesh.vertices) ||
            !upload_geometry(prototype.indices.data(), prototype.indices.size() * sizeof(uint32_t),
                             VK_BUFFER_USAGE_INDEX_BUFFER_BIT, mesh.indices))
        {
            error = "level.details / " + prototype.texture + ": " + error;
            return false;
        }
        mesh.index_count = static_cast<uint32_t>(prototype.indices.size());
        mesh.vertex_bytes = prototype.vertices.size() * sizeof(LevelVertex);
        mesh.index_bytes = prototype.indices.size() * sizeof(uint32_t);
    }
    // The old level may still have been submitted. Its owner must wait for
    // those frames before replacing the level. Uploads within prepared use
    // the same queue as the first draw and preserve staging until retired.
    destroy();
    device_ = prepared.device_;
    profile_ = prepared.profile_;
    memory_ = prepared.memory_;
    pool_ = prepared.pool_;
    upload_ = prepared.upload_;
    textures_ = prepared.textures_;
    pass_ = prepared.pass_;
    pending_ = std::move(prepared.pending_);
    geometry_batches_ = std::move(prepared.geometry_batches_);
    for (size_t frame = 0; frame < cluster_draw_commands_.size(); ++frame)
        cluster_draw_commands_[frame] = std::move(prepared.cluster_draw_commands_[frame]);
    cluster_draw_command_count_ = prepared.cluster_draw_command_count_;
    cluster_visibility_ = std::move(prepared.cluster_visibility_);
    ray_batches_ = std::move(prepared.ray_batches_);
    ray_texture_matrices_ = std::move(prepared.ray_texture_matrices_);
    ray_texture_names_ = std::move(prepared.ray_texture_names_);
    ray_diffuse_texture_names_ = std::move(prepared.ray_diffuse_texture_names_);
    ray_diffuse_texture_linear_ = std::move(prepared.ray_diffuse_texture_linear_);
    meshes_ = std::move(prepared.meshes_);
    has_water_ = prepared.has_water_;
    details_ = std::move(prepared.details_);
    detail_meshes_ = std::move(prepared.detail_meshes_);
    visuals_ = std::move(prepared.visuals_);
    cull_bounds_ = level_hierarchy_bounds(visuals_);
    std::vector<float> wind_extents;
    wind_extents.reserve(meshes_.size());
    for (const auto& mesh : meshes_) wind_extents.push_back(mesh.tree_wind_extent);
    cull_wind_extents_ = level_hierarchy_wind_extents(visuals_, wind_extents);
    visual_phases_.assign(visuals_.size(), 0);
    visual_priorities_.assign(visuals_.size(), 0);
    std::vector<uint8_t> phase_state(visuals_.size(), 0);
    const auto phases = [&](const auto &self, size_t index) -> uint8_t {
        if (index >= visuals_.size())
            return 3;
        if (phase_state[index] == 2)
            return visual_phases_[index];
        if (phase_state[index] == 1)
            return 3; // malformed cycle: never omit a pass
        phase_state[index] = 1;
        const LevelVisual &visual = visuals_[index];
        uint8_t mask = 0;
        if (visual.mesh >= 0 && size_t(visual.mesh) < meshes_.size())
        {
            const Mesh &mesh = meshes_[visual.mesh];
            mask |= mesh.mode == SurfaceMode::Transparent ? 2 : 1;
            if (mesh.secondary_opaque)
                mask |= 4;
            else if (mesh.mode == SurfaceMode::Transparent)
                visual_priorities_[index] |= uint8_t(1u << mesh.priority);
        }
        for (int32_t facet : visual.lod_facets)
            if (facet >= 0 && size_t(facet) < meshes_.size())
            {
                const Mesh &mesh = meshes_[facet];
                mask |= mesh.mode == SurfaceMode::Transparent ? 2 : 1;
                if (mesh.secondary_opaque)
                    mask |= 4;
                else if (mesh.mode == SurfaceMode::Transparent)
                    visual_priorities_[index] |= uint8_t(1u << mesh.priority);
            }
        for (uint32_t child : visual.children)
        {
            mask |= self(self, child);
            if (child < visual_priorities_.size())
                visual_priorities_[index] |= visual_priorities_[child];
        }
        phase_state[index] = 2;
        return visual_phases_[index] = mask;
    };
    for (size_t index = 0; index < visuals_.size(); ++index)
        phases(phases, index);
    roots_ = std::move(prepared.roots_);
    sectors_ = std::move(prepared.sectors_);
    portals_ = std::move(prepared.portals_);
    visual_objects_.reserve(visuals_.size());
    for (size_t index = 0; index < visuals_.size(); ++index)
    {
        visual_objects_.emplace_back(
            std::make_unique<VulkanVisual>(*this, static_cast<uint32_t>(index), visuals_[index]));
        visual_indices_.emplace(visual_objects_.back().get(), static_cast<uint32_t>(index));
    }
    prepared.device_ = VK_NULL_HANDLE;
    prepared.textures_ = nullptr;
    prepared.pass_ = nullptr;
    prepared.pool_ = VK_NULL_HANDLE;
    Msg("[renderer-vulkan] level geometry buffers ready models=%zu details=%zu pending-uploads=%zu bytes=%llu",
        meshes_.size(), detail_meshes_.size(), pending_.size(), static_cast<unsigned long long>(geometry_bytes()));
    error.clear();
    return true;
}

bool GpuLevel::record(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16]) const
{
    for (GeometryPhase phase : {GeometryPhase::OpaqueAndAlphaTest, GeometryPhase::Transparent})
        if (!record(frame, pass, mvp, phase))
            return false;
    return true;
}

bool GpuLevel::record_cluster_visibility(const FrameRecordingContext& frame,
    const float (&camera_mvp)[16], const float (&sun_mvp)[16], uint32_t shadow_height,
    std::string& error, const float* previous_mvp, VkImage previous_hiz_image,
    VkImageView previous_hiz, VkSampler hiz_sampler,
    const FrameImageSharing* previous_hiz_sharing, uint32_t previous_hiz_levels, bool hiz_valid) const
{
    if (!cluster_visibility_.ready())
    {
        error.clear();
        return true;
    }
    const float camera[3]{Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z};
    return cluster_visibility_.record(frame, camera_mvp, sun_mvp, camera, Device.fFOV, shadow_height,
        error, previous_mvp, previous_hiz_image, previous_hiz, hiz_sampler,
        previous_hiz_sharing, previous_hiz_levels, hiz_valid);
}

bool GpuLevel::record_cluster_shadow_visibility(const FrameRecordingContext& frame,
    const float (&shadow_mvp)[16], std::string& error) const
{
    if (!cluster_visibility_.ready())
    {
        error.clear();
        return true;
    }
    const float camera[3]{Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z};
    return cluster_visibility_.record_shadow_view(frame, shadow_mvp, camera, Device.fFOV, error);
}

bool GpuLevel::record(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16],
                      GeometryPhase phase) const
{
    for (uint32_t root : roots_)
        if (!record_visual(root, frame, pass, mvp, phase))
            return false;
    if (phase == GeometryPhase::OpaqueAndAlphaTest && !record_details(frame, pass, mvp))
        return false;
    return true;
}

bool GpuLevel::record_sun_shadow(const FrameRecordingContext &frame, const DeferredPass &pass,
                                 const float (&sun_mvp)[16], bool include_static, bool include_trees,
                                 bool use_cluster_visibility) const
{
    auto command_batch = pass.geometry_batch();
    std::vector<uint8_t> visited(visuals_.size(), 0);
    const auto draw = [&](const auto &self, uint32_t index) -> bool {
        if (index >= visuals_.size())
            return false;
        if (visited[index])
            return true;
        visited[index] = 1;
        const LevelVisual &visual = visuals_[index];
        if (index < cull_bounds_.size())
        {
            const auto &sphere = cull_bounds_[index];
            const float radius = wind_expanded_radius(sphere[3], cull_wind_extents_[index], pass.tree_wind_amplitude());
            if (!visible_sphere(sun_mvp, sphere[0], sphere[1], sphere[2], radius))
                return true;
        }
        if (visual.mesh >= 0)
        {
            if (size_t(visual.mesh) >= meshes_.size())
                return false;
            const Mesh &base = meshes_[visual.mesh];
            if ((base.tree && include_trees) || (!base.tree && include_static))
            {
                // Alpha-tested casters need original UVs; displaced meshes need
                // their tessellated silhouette. Only opaque flat proxies qualify.
                const Mesh &mesh = base.fast && use_fast_depth_proxy(base.mode, base.tessellation) ? *base.fast : base;
                const float dx = visual.bounds[6] - Device.vCameraPosition.x;
                const float dy = visual.bounds[7] - Device.vCameraPosition.y;
                const float dz = visual.bounds[8] - Device.vCameraPosition.z;
                const float lod = lod_for_distance(visual.bounds[9], dx * dx + dy * dy + dz * dz);
                const SlideWindow window = select_slide_window(mesh.windows, lod, mesh.index_count);
                const size_t window_index = select_slide_window_index(mesh.windows, lod);
                const Mesh::ClusterWindow* window_cluster = window_index < mesh.cluster_lod_windows.size()
                    ? &mesh.cluster_lod_windows[window_index] : nullptr;
                bool cluster_rendered = false;
                if (window_cluster && mesh.packed_cluster_indices &&
                    !window_cluster->data.groups.empty() && !window_cluster->data.clusters.empty())
                {
                    if (cluster_visibility_.ready() &&
                        (pass.is_sun_shadow_pass(frame.render_pass) || use_cluster_visibility) &&
                        window_cluster->cull_cluster_count == window_cluster->data.clusters.size())
                    {
                        const VkBuffer commands = cluster_visibility_.command_buffer(frame.frame_index);
                        if (!commands || !pass.record_sun_shadow(frame, mesh.vertex_buffer(),
                                mesh.packed_cluster_indices, 0, sun_mvp, mesh.material,
                                mesh.mode == SurfaceMode::AlphaTest, 0, float(mesh.alpha_ref) / 255.f,
                                mesh.first_vertex, VK_NULL_HANDLE, false, &mesh.matrices, commands,
                                cluster_visibility_.command_offset(1, window_cluster->first_draw_command),
                                window_cluster->cull_cluster_count))
                            return false;
                        cluster_rendered = true;
                    }
                    if (!cluster_rendered)
                    {
                    const ClusterLodSelection selection = select_cluster_lod(
                        window_cluster->data, frame, sun_mvp, Device.vCameraPosition, Device.fFOV);
                    if (selection.disposition == ClusterLodDisposition::FullyCulled)
                        cluster_rendered = true;
                    else if (selection.disposition == ClusterLodDisposition::UseClusters)
                    {
                        bool valid = true;
                        for (uint32_t cluster_index : selection.clusters)
                        {
                            if (cluster_index >= window_cluster->data.clusters.size())
                            {
                                valid = false;
                                break;
                            }
                            const ClusterLodCluster& cluster = window_cluster->data.clusters[cluster_index];
                            if (!cluster.index_count || cluster.index_count % 3 ||
                                cluster.first_index > window_cluster->index_count ||
                                cluster.index_count > window_cluster->index_count - cluster.first_index ||
                                window_cluster->first_index > UINT32_MAX - cluster.first_index ||
                                cluster.index_count > UINT32_MAX - window_cluster->first_index - cluster.first_index)
                            {
                                valid = false;
                                break;
                            }
                        }
                        if (valid)
                        {
                            for (uint32_t cluster_index : selection.clusters)
                            {
                                const ClusterLodCluster& cluster = window_cluster->data.clusters[cluster_index];
                                if (!pass.record_sun_shadow(frame, mesh.vertex_buffer(),
                                        mesh.packed_cluster_indices, cluster.index_count, sun_mvp, mesh.material,
                                        mesh.mode == SurfaceMode::AlphaTest,
                                        window_cluster->first_index + cluster.first_index,
                                        float(mesh.alpha_ref) / 255.f, mesh.first_vertex, VK_NULL_HANDLE, false,
                                        &mesh.matrices))
                                    return false;
                            }
                            cluster_rendered = true;
                        }
                    }
                    }
                }
                else if (!window_cluster && mesh.packed_cluster_indices && !mesh.cluster_lod.groups.empty())
                {
                    if (cluster_visibility_.ready() &&
                        (pass.is_sun_shadow_pass(frame.render_pass) || use_cluster_visibility))
                    {
                        const VkBuffer commands = cluster_visibility_.command_buffer(frame.frame_index);
                        if (!commands || !pass.record_sun_shadow(frame, mesh.vertex_buffer(),
                                mesh.packed_cluster_indices, 0, sun_mvp, mesh.material,
                                mesh.mode == SurfaceMode::AlphaTest, 0, float(mesh.alpha_ref) / 255.f,
                                mesh.first_vertex, VK_NULL_HANDLE, false, &mesh.matrices, commands,
                                cluster_visibility_.command_offset(1, mesh.cluster_first_draw_command),
                                mesh.cluster_cull_cluster_count))
                            return false;
                        cluster_rendered = true;
                    }
                    else
                    {
                        const ClusterLodSelection selection = select_cluster_lod(
                            mesh.cluster_lod, frame, sun_mvp, Device.vCameraPosition, Device.fFOV);
                        if (selection.disposition == ClusterLodDisposition::FullyCulled)
                        {
                            cluster_rendered = true;
                        }
                        else if (selection.disposition == ClusterLodDisposition::UseClusters)
                        {
                            const std::vector<uint32_t>& cluster_draws = selection.clusters;
                            for (uint32_t cluster_index : cluster_draws)
                            {
                                if (cluster_index >= mesh.cluster_lod.clusters.size())
                                    return false;
                                const ClusterLodCluster& cluster = mesh.cluster_lod.clusters[cluster_index];
                                if (!pass.record_sun_shadow(frame, mesh.vertex_buffer(), mesh.packed_cluster_indices,
                                        cluster.index_count, sun_mvp, mesh.material,
                                        mesh.mode == SurfaceMode::AlphaTest,
                                        mesh.cluster_first_index + cluster.first_index,
                                        float(mesh.alpha_ref) / 255.f, mesh.first_vertex, VK_NULL_HANDLE, false,
                                        &mesh.matrices))
                                    return false;
                            }
                            cluster_rendered = true;
                        }
                    }
                }
                if (mesh.mode != SurfaceMode::Transparent && mesh.index_count &&
                    !cluster_rendered && !pass.record_sun_shadow(frame, mesh.vertex_buffer(), mesh.index_buffer(), window.index_count, sun_mvp,
                                            mesh.material, mesh.mode == SurfaceMode::AlphaTest,
                                            mesh.first_index + window.offset, float(mesh.alpha_ref) / 255.f,
                                            mesh.first_vertex, VK_NULL_HANDLE, base.tree_wave, &mesh.matrices))
                    return false;
            }
        }
        for (uint32_t child : visual.children)
            if (!self(self, child))
                return false;
        return true;
    };
    for (uint32_t root : roots_)
        if (!draw(draw, root))
            return false;
    return true;
}

namespace
{
uint32_t detail_random(uint32_t value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
} // namespace

void GpuLevel::prepare_details(const Fvector &camera, float max_distance)
{
    visible_details_.clear();
    if (details_.cells.empty() || !g_pGameLevel || !std::isfinite(camera.x) || !std::isfinite(camera.y) ||
        !std::isfinite(camera.z))
        return;
    max_distance = std::clamp(max_distance, 0.f, 80.f);
    if (max_distance < 1.f)
        return;
    const float spacing = std::isfinite(detail_spacing) ? std::clamp(detail_spacing, .25f, 1.f) : .5f;
    const unsigned grid_size = detail_grid_size(spacing);
    const float sample_spacing = 2.f / grid_size;
    const float density = sample_spacing * sample_spacing / (spacing * spacing);
    if (detail_spacing_ != spacing)
    {
        detail_cache_.clear();
        detail_spacing_ = spacing;
    }
    const int cx = static_cast<int>(std::floor(camera.x / 2.f)) + details_.offset_x;
    const int cz = static_cast<int>(std::floor(camera.z / 2.f)) + details_.offset_z;
    const int radius = static_cast<int>(max_distance / 2.f) + 1;
    CDB::COLLIDER collider;
    std::vector<size_t> cells;
    for (int z = std::max(0, cz - radius); z <= std::min(int(details_.height) - 1, cz + radius); ++z)
        for (int x = std::max(0, cx - radius); x <= std::min(int(details_.width) - 1, cx + radius); ++x)
            cells.push_back(size_t(z) * details_.width + x);
    const auto distance = [&](size_t index) {
        const float x = float(int(index % details_.width) - details_.offset_x) * 2.f + 1.f - camera.x;
        const float z = float(int(index / details_.width) - details_.offset_z) * 2.f + 1.f - camera.z;
        return x * x + z * z;
    };
    std::stable_sort(cells.begin(), cells.end(), [&](size_t a, size_t b) { return distance(a) < distance(b); });
#if defined(XR_PLATFORM_ANDROID)
    const auto prepare_begin = std::chrono::steady_clock::now();
#endif
    int updated = 0;
    for (size_t index : cells)
    {
        const int x = int(index % details_.width), z = int(index / details_.width);
        const DetailCell &cell = details_.cells[index];
        const float slot_x = float(x - details_.offset_x) * 2.f;
        const float slot_z = float(z - details_.offset_z) * 2.f;
        const float dx = slot_x + 1.f - camera.x, dz = slot_z + 1.f - camera.z;
        if (dx * dx + dz * dz > max_distance * max_distance ||
            std::abs(cell.ground - camera.y) > max_distance + cell.height)
            continue;
        if (std::all_of(cell.ids.begin(), cell.ids.end(), [](uint8_t id) { return id == 63; }))
            continue;
        auto found = detail_cache_.find(index);
        if (found == detail_cache_.end())
        {
            if (updated >= detail_update_cells)
                continue;
#if defined(XR_PLATFORM_ANDROID)
            if (updated &&
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - prepare_begin).count() >=
                    detail_prepare_ms)
                break;
#endif
            ++updated;
            std::vector<DetailPlacement> placements;
            // Query the slot once, then test its triangles for every
            // placement. A ray query per blade causes long frame stalls
            // whenever a new group of cells enters the camera radius.
            Fvector box_center, box_half;
            box_center.set(slot_x + 1.f, cell.ground + cell.height * .5f, slot_z + 1.f);
            box_half.set(1.f, cell.height * .5f + 5.f, 1.f);
            collider.box_query(CDB::OPT_FULL_TEST, g_pGameLevel->ObjectSpace.GetStaticModel(), box_center, box_half);
            const auto *tris = g_pGameLevel->ObjectSpace.GetStaticTris();
            const auto *verts = g_pGameLevel->ObjectSpace.GetStaticVerts();
            for (unsigned rz = 0; rz < grid_size; ++rz)
                for (unsigned rx = 0; rx < grid_size; ++rx)
                {
                    const uint32_t seed = detail_random(uint32_t(index) ^ (uint32_t(rx + rz * grid_size) * 0x9e3779b9u));
                    const float u = (float(rx) + .5f) / grid_size;
                    const float v = (float(rz) + .5f) / grid_size;
                    float weights[4]{};
                    float sum = 0.f;
                    for (unsigned n = 0; n < 4; ++n)
                    {
                        if (cell.ids[n] == 63)
                            continue;
                        const auto &p = cell.palette[n];
                        weights[n] = (p[0] * (1 - u) + p[1] * u) * (1 - v) + (p[2] * (1 - u) + p[3] * u) * v;
                        sum += weights[n];
                    }
                    if (sum <= 0.f || float(seed & 255u) / 255.f > density * std::min(1.f, sum / 15.f))
                        continue;
                    float choice = float((seed >> 8) & 65535u) / 65536.f * sum;
                    unsigned model = 0;
                    for (; model < 3 && choice >= weights[model]; ++model)
                        choice -= weights[model];
                    if (cell.ids[model] >= detail_meshes_.size())
                        continue;
                    const float jx = float((seed >> 16) & 15u) / 16.f - .5f;
                    const float jz = float((seed >> 20) & 15u) / 16.f - .5f;
                    Fvector origin, down;
                    origin.set(slot_x + (float(rx) + .5f + jx * .5f) * sample_spacing, cell.ground + cell.height + 5.f,
                               slot_z + (float(rz) + .5f + jz * .5f) * sample_spacing);
                    down.set(0.f, -1.f, 0.f);
                    // Match the GLES decompressor: ignore passable faces
                    // and intersect only triangles belonging to this slot.
                    float ground = cell.ground - 5.f;
                    for (const auto &hit : *collider.r_get())
                    {
                        const auto &triangle = tris[hit.id];
                        const auto *material = GMLib.GetMaterialByIdx(triangle.material);
                        if (!material || material->Flags.test(SGameMtl::flPassable))
                            continue;
                        Fvector points[3]{verts[triangle.verts[0]], verts[triangle.verts[1]], verts[triangle.verts[2]]};
                        float u, v, range;
                        if (CDB::TestRayTri(origin, down, points, u, v, range, TRUE) && range >= 0.f &&
                            range <= cell.height + 10.f)
                        {
                            const float height = origin.y - range;
                            if (height >= cell.ground && height <= cell.ground + cell.height + .1f)
                                ground = std::max(ground, height);
                        }
                    }
                    if (ground < cell.ground)
                        continue;
                    const DetailPrototype &prototype = details_.prototypes[cell.ids[model]];
                    const float t = float((seed >> 24) & 255u) / 255.f;
                    placements.push_back({cell.ids[model], origin.x, ground, origin.z,
                                          float(seed & 65535u) * (6.2831853f / 65536.f),
                                          prototype.min_scale + t * (prototype.max_scale - prototype.min_scale)});
                }
            found = detail_cache_.emplace(index, std::move(placements)).first;
        }
    }
    // Keep draw order independent of generation priority, so camera motion
    // does not invalidate the cached detail vertex buffers by reordering cells.
    std::sort(cells.begin(), cells.end());
    for (size_t index : cells)
    {
        const auto &cell = details_.cells[index];
        if (distance(index) > max_distance * max_distance ||
            std::abs(cell.ground - camera.y) > max_distance + cell.height)
            continue;
        const auto found = detail_cache_.find(index);
        if (found != detail_cache_.end())
            visible_details_.insert(visible_details_.end(), found->second.begin(), found->second.end());
    }
    if (detail_cache_.size() > 4096)
    {
        for (auto it = detail_cache_.begin(); it != detail_cache_.end();)
        {
            const int x = static_cast<int>(it->first % details_.width);
            const int z = static_cast<int>(it->first / details_.width);
            if (std::abs(x - cx) > radius * 2 || std::abs(z - cz) > radius * 2)
                it = detail_cache_.erase(it);
            else
                ++it;
        }
    }
}

bool GpuLevel::record_details(const FrameRecordingContext &frame, const DeferredPass &pass,
                              const float (&mvp)[16]) const
{
    if (frame.frame_index >= detail_batches_.size())
        return false;
    auto &buffers = detail_batches_[frame.frame_index];
    if (buffers.size() != detail_meshes_.size())
        buffers.resize(detail_meshes_.size());
    uint64_t signature = 14695981039346656037ull;
    std::vector<std::vector<DetailInstance>> grouped(detail_meshes_.size());
    for (const auto &p : visible_details_)
    {
        if (p.model >= detail_meshes_.size())
            return false;
        if (detail_meshes_[p.model].mode == SurfaceMode::Transparent || !visible_sphere(mvp, p.x, p.y, p.z, 6.f))
            continue;
        auto instance = detail_instance(p.x, p.y, p.z, p.yaw, p.scale);
        const auto *bytes = reinterpret_cast<const uint8_t *>(&instance);
        for (size_t i = 0; i < sizeof(instance); ++i)
            signature = (signature ^ bytes[i]) * 1099511628211ull;
        signature = (signature ^ p.model) * 1099511628211ull;
        grouped[p.model].push_back(instance);
    }
    const bool changed = detail_signatures_[frame.frame_index] != signature;
    for (size_t i = 0; i < grouped.size(); ++i)
    {
        auto &buffer = buffers[i];
        const auto &instances = grouped[i];
        const auto &mesh = detail_meshes_[i];
        if (changed)
        {
            buffer.instance_count = uint32_t(instances.size());
            if (!instances.empty())
            {
                std::string error;
                const size_t bytes = instances.size() * sizeof(DetailInstance);
                if ((buffer.vertices.size() < bytes &&
                     !buffer.vertices.initialize(
                         device_, std::max<VkDeviceSize>(bytes, buffer.vertices.size() * 2 + 4096),
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory_,
                         upload_.buffer, error)) ||
                    !buffer.vertices.write(0, instances.data(), bytes, error))
                {
                    Msg("! [renderer-vulkan] detail instances: %s", error.c_str());
                    return false;
                }
            }
        }
        if (!buffer.instance_count)
            continue;
        const auto &node = mesh.pass_graph.passes[0];
        if (!pass.record_geometry(frame, mesh.vertex_buffer(), mesh.index_buffer(), mesh.index_count, mvp,
                                  mesh.material, mesh.mode, 0, node.vertex.c_str(), node.fragment.c_str(),
                                  float(mesh.alpha_ref) / 255.f, nullptr, mesh.terrain_scale, nullptr, nullptr, 0,
                                  &mesh.matrices, buffer.vertices.handle(), buffer.instance_count,
                                  VK_NULL_HANDLE, 0, 0, node.shader_variant_hash))
            return false;
    }
    detail_signatures_[frame.frame_index] = signature;
    return true;
}

bool GpuLevel::prepare_transparent_details(const FrameRecordingContext &frame, const float (&mvp)[16],
                                           std::vector<float> &sort_distances) const
{
    if (frame.frame_index >= transparent_detail_batches_.size())
        return false;
    sort_distances.clear();
    auto &draws = transparent_detail_draws_[frame.frame_index];
    draws.clear();
    std::vector<const DetailPlacement *> visible;
    visible.reserve(visible_details_.size());
    for (const DetailPlacement &placement : visible_details_)
    {
        if (placement.model >= detail_meshes_.size())
            return false;
        if (detail_meshes_[placement.model].mode == SurfaceMode::Transparent &&
            visible_sphere(mvp, placement.x, placement.y, placement.z, 6.f))
            visible.push_back(&placement);
    }
    const auto distance = [](const DetailPlacement *placement) {
        const float dx = placement->x - Device.vCameraPosition.x;
        const float dy = placement->y - Device.vCameraPosition.y;
        const float dz = placement->z - Device.vCameraPosition.z;
        return dx * dx + dy * dy + dz * dz;
    };
    if (visible.empty())
        return true;

    std::vector<LevelVertex> vertices;
    std::vector<uint32_t> indices;
    for (const DetailPlacement *placement : visible)
    {
        const DetailPrototype &prototype = details_.prototypes[placement->model];
        if (vertices.size() > UINT32_MAX - prototype.vertices.size() ||
            indices.size() > UINT32_MAX - prototype.indices.size())
            return false;
        draws.push_back(
            {placement->model, static_cast<uint32_t>(indices.size()), static_cast<uint32_t>(prototype.indices.size())});
        sort_distances.push_back(distance(placement));
        const uint32_t first = static_cast<uint32_t>(vertices.size());
        Fmatrix rotation, scale, world;
        Fvector position;
        position.set(placement->x, placement->y, placement->z);
        rotation.rotateY(placement->yaw);
        rotation.translate_over(position);
        scale.scale(placement->scale, placement->scale, placement->scale);
        world.mul_43(rotation, scale);
        for (const LevelVertex &original : prototype.vertices)
        {
            LevelVertex vertex = original;
            Fvector local, transformed, normal;
            local.set(original.position[0], original.position[1], original.position[2]);
            world.transform_tiny(transformed, local);
            local.set(original.normal[0], original.normal[1], original.normal[2]);
            rotation.transform_dir(normal, local);
            normal.normalize_safe();
            std::copy_n(&transformed.x, 3, vertex.position);
            std::copy_n(&normal.x, 3, vertex.normal);
            vertices.push_back(vertex);
        }
        for (uint32_t index : prototype.indices)
            indices.push_back(first + index);
    }
    auto &batches = transparent_detail_batches_[frame.frame_index];
    if (batches.empty())
        batches.resize(1);
    auto &batch = batches[0];
    const auto write = [this](BufferResource &buffer, const void *data, size_t bytes, VkBufferUsageFlags usage,
                              std::string &error) {
        if (buffer.size() < bytes &&
            !buffer.initialize(device_, std::max<VkDeviceSize>(bytes, buffer.size() ? buffer.size() * 2 : 4096), usage,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory_,
                               upload_.buffer, error))
            return false;
        return buffer.write(0, data, bytes, error);
    };
    std::string error;
    if (!write(batch.vertices, vertices.data(), vertices.size() * sizeof(LevelVertex),
               VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, error) ||
        !write(batch.indices, indices.data(), indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
               error))
    {
        Msg("! [renderer-vulkan] transparent detail upload: %s", error.c_str());
        return false;
    }
    return true;
}

bool GpuLevel::record_transparent_detail(size_t draw_index, const FrameRecordingContext &frame,
                                         const DeferredPass &pass, const float (&mvp)[16]) const
{
    if (frame.frame_index >= transparent_detail_batches_.size() ||
        draw_index >= transparent_detail_draws_[frame.frame_index].size() ||
        transparent_detail_batches_[frame.frame_index].empty())
        return false;
    const auto &draw = transparent_detail_draws_[frame.frame_index][draw_index];
    const auto &batch = transparent_detail_batches_[frame.frame_index][0];
    const Mesh &mesh = detail_meshes_[draw.model];
    const auto &node = mesh.pass_graph.passes[0];
    {
        const char *failure = nullptr;
        if (!pass.record_transparent(frame, batch.vertices.handle(), batch.indices.handle(), draw.count, mvp,
                                     mesh.material, draw.first, node.vertex.c_str(), node.fragment.c_str(), &failure,
                                     float(mesh.alpha_ref) / 255.f, nullptr, mesh.terrain_scale, 0, &mesh.matrices,
                                     node.shader_variant_hash))
        {
            Msg("! [renderer-vulkan] transparent detail shader='%s' texture='%s': %s",
                details_.prototypes[draw.model].shader.c_str(), details_.prototypes[draw.model].texture.c_str(),
                failure ? failure : "unknown");
            return false;
        }
    }
    return true;
}

bool GpuLevel::record_visual(size_t index, const FrameRecordingContext &frame, const DeferredPass &pass,
                             const float (&mvp)[16], GeometryPhase phase, float lod, bool cull_static,
                             bool secondary_only, int priority_filter) const
{
    if (index >= visuals_.size())
        return false;
    if (index < visual_phases_.size() && !(visual_phases_[index] & (secondary_only                        ? 4
                                                                    : phase == GeometryPhase::Transparent ? 2
                                                                                                          : 1)))
        return true;
    if (phase == GeometryPhase::Transparent && !secondary_only && priority_filter >= 0 &&
        !has_transparent_priority(index, static_cast<uint8_t>(priority_filter)))
        return true;
    const LevelVisual &visual = visuals_[index];
    // Rebuilt bounds enclose every descendant. Reject an offscreen branch
    // once instead of walking its leaves again for each material priority.
    const auto &sphere = cull_bounds_[index];
    const float radius = wind_expanded_radius(sphere[3], cull_wind_extents_[index], pass.tree_wind_amplitude());
    if (cull_static && !visible_sphere(mvp, sphere[0], sphere[1], sphere[2], radius))
    {
        static std::atomic<uint64_t> culled{0};
        const uint64_t count = ++culled;
        if (count <= 4 || count % 1048576 == 0)
            Msg("[renderer-vulkan] level.frustum culled=%llu last-visual=%zu phase=%d",
                static_cast<unsigned long long>(count), index, static_cast<int>(phase));
        return true;
    }
#if defined(XR_PLATFORM_ANDROID)
    if (Device.dwPrecacheFrame)
    {
        char context[96];
        std::snprintf(context, sizeof(context), "vulkan level visual=%zu mesh=%d phase=%d", index, visual.mesh,
                      static_cast<int>(phase));
        android_set_load_context(context);
    }
#endif
    const float billboard_alpha = lod_billboard_alpha(
        visual, frame.extent, Device.fFOV, Device.vCameraPosition);
    if (visual.type == 6 && visual.bounds[9] > 0.f && billboard_alpha < 0.f)
        return true;
    if (billboard_alpha >= 96.f/255.f && phase == GeometryPhase::OpaqueAndAlphaTest)
    {
        Fvector direction;
        direction.set(visual.bounds[6] - Device.vCameraPosition.x, visual.bounds[7] - Device.vCameraPosition.y,
                      visual.bounds[8] - Device.vCameraPosition.z);
        auto views = select_lod_views(visual.lod_normals, {direction.x, direction.y, direction.z});
        if (views.first > views.second)
        {
            std::swap(views.first, views.second);
            views.weight = 1.f - views.weight;
        }
        const auto facet = views.first;
        const int32_t mesh_index = visual.lod_facets[facet];
        if (mesh_index < 0 || size_t(mesh_index) >= meshes_.size())
            return false;
        const Mesh &mesh = meshes_[mesh_index];
        const bool transparent = mesh.mode == SurfaceMode::Transparent;
        if ((phase == GeometryPhase::Transparent) != transparent ||
            (transparent && (mesh.secondary_opaque != secondary_only ||
                             (!secondary_only && priority_filter >= 0 && mesh.priority != priority_filter))))
            return true;
        const char *vertex = "vk\\level_lod.vs";
        const char *fragment = "vk\\level_lod.ps";
        std::string shader_error;
        const bool named = pass.require_game_pipeline(vertex, fragment, mesh.mode, false, false, shader_error);
        if (!named)
        {
            Msg("! Vulkan level: %s", shader_error.c_str());
            return false;
        }
#if defined(XR_PLATFORM_ANDROID)
        if (Device.dwPrecacheFrame)
        {
            char context[192];
            std::snprintf(context, sizeof(context),
                          "vulkan level visual=%zu mesh=%d phase=%d water=%d indices=%u set=%llu", index, mesh_index,
                          static_cast<int>(phase), mesh.water ? 1 : 0, mesh.index_count,
                          descriptor_number(mesh.material));
            android_set_load_context(context);
        }
#endif
        const char *failure = nullptr;
        Fmatrix view_projection, offset, shifted;
        std::memcpy(&view_projection, mvp, sizeof(mvp));
        direction.normalize_safe().mul(-.5f * visual.bounds[9]);
        offset.translate(direction);
        shifted.mul(view_projection, offset);
        float billboard_mvp[16];
        std::memcpy(billboard_mvp, &shifted, sizeof(billboard_mvp));
        const bool completed = pass.record_geometry(frame, mesh.vertex_buffer(), mesh.index_buffer(), 6, billboard_mvp,
            mesh.material, SurfaceMode::AlphaTest, mesh.first_index + (views.second - views.first) * 6,
            vertex, fragment, (mesh.lod_atlas ? 1.f : -1.f) * (96.f/255.f) / billboard_alpha,
            nullptr, views.weight, nullptr, nullptr, mesh.first_vertex);
        if (!completed)
            report_visual_record_failure(index, mesh_index, phase, mesh.water, mesh.index_count, lod, failure);
        if (!completed || billboard_alpha >= 1.f)
            return completed;
    }
    if (billboard_alpha >= 1.f)
        return true;
    if (visual.mesh >= 0)
    {
        if (static_cast<size_t>(visual.mesh) >= meshes_.size())
            return false;
        const Mesh &base = meshes_[visual.mesh];
        const Mesh &mesh = base;
        const bool transparent = mesh.mode == SurfaceMode::Transparent;
        const bool selected = phase == GeometryPhase::Transparent
                                  ? transparent && mesh.secondary_opaque == secondary_only &&
                                        (secondary_only || priority_filter < 0 || mesh.priority == priority_filter)
                                  : phase == GeometryPhase::OpaqueAndAlphaTest && !transparent;
        const char *failure = nullptr;
        const SlideWindow window = select_slide_window(mesh.windows, lod, mesh.index_count);
        const size_t window_index = select_slide_window_index(mesh.windows, lod);
        const Mesh::ClusterWindow* window_cluster = window_index < mesh.cluster_lod_windows.size()
            ? &mesh.cluster_lod_windows[window_index] : nullptr;
        const auto &first = mesh.pass_graph.passes[0];
        const char *vertex = first.vertex_for_visual(&mesh != &base, visual.type);
        const char *fragment = first.fragment.c_str();
        if (selected)
        {
            const float camera[4]{Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z, 0.f};
            float factor[4]{1.f, 1.f, 1.f, 1.f};
            if (mesh.environment_constant)
                for (unsigned channel = 0; channel < 4; ++channel)
                {
                    WaveForm waveform = mesh.constant_waveforms[channel];
                    factor[channel] = waveform.Calculate(Device.fTimeGlobal);
                }
            std::string shader_error;
            const bool named = pass.require_game_pipeline(vertex, fragment, mesh.mode, false, false, shader_error,
                first.shader_variant_hash);
            if (!named)
            {
                Msg("! Vulkan level: %s", shader_error.c_str());
                return false;
            }
#if defined(XR_PLATFORM_ANDROID)
            if (Device.dwPrecacheFrame)
            {
                char context[192];
                std::snprintf(context, sizeof(context),
                              "vulkan level visual=%zu mesh=%d phase=%d water=%d first=%u count=%u set=%llu", index,
                              visual.mesh, static_cast<int>(phase), mesh.water ? 1 : 0, window.offset,
                              window.index_count, descriptor_number(mesh.material));
                android_set_load_context(context);
            }
#endif
            const auto record_followup_material_pass = [&](VkBuffer indices, uint32_t index_count,
                uint32_t first_index, VkBuffer indirect_commands, VkDeviceSize indirect_offset,
                uint32_t indirect_draw_count) {
                if (mesh.pass_graph.count == 1)
                    return true;
                if (!supports_cluster_lod_pass_graph(mesh.pass_graph))
                    return false;
                const LevelPassNode& node = mesh.pass_graph.passes[1];
                return pass.record_geometry(frame, mesh.vertex_buffer(), indices, index_count, mvp,
                    mesh.material, mesh.mode, first_index, node.vertex.c_str(), node.fragment.c_str(),
                    128.f / 255.f, nullptr, 1.f, nullptr, nullptr, mesh.first_vertex, &mesh.matrices,
                    VK_NULL_HANDLE, 1, indirect_commands, indirect_offset, indirect_draw_count,
                    node.shader_variant_hash);
            };
            bool cluster_rendered = false;
            if (!transparent && window_cluster && mesh.packed_cluster_indices &&
                !window_cluster->data.groups.empty() && !window_cluster->data.clusters.empty())
            {
                const bool gpu_window_ready = cluster_visibility_.ready() &&
                    supports_cluster_lod_pass_graph(mesh.pass_graph) &&
                    window_cluster->cull_cluster_count == window_cluster->data.clusters.size();
                if (gpu_window_ready)
                {
                    const VkBuffer commands = cluster_visibility_.command_buffer(frame.frame_index);
                    const VkDeviceSize command_offset = cluster_visibility_.command_offset(
                        0, window_cluster->first_draw_command);
                    if (commands && pass.record_geometry(frame, mesh.vertex_buffer(), mesh.packed_cluster_indices,
                            0, mvp, mesh.material, mesh.mode, 0, named ? vertex : nullptr,
                            named ? fragment : nullptr, float(mesh.alpha_ref) / 255.f, nullptr,
                            mesh.terrain_scale, mesh.environment ? camera : nullptr,
                            mesh.laemb ? factor : nullptr, mesh.first_vertex, &mesh.matrices,
                            VK_NULL_HANDLE, 1, commands, command_offset,
                            window_cluster->cull_cluster_count, first.shader_variant_hash))
                    {
                        if (!record_followup_material_pass(mesh.packed_cluster_indices, 0, 0, commands,
                                command_offset, window_cluster->cull_cluster_count))
                        {
                            report_visual_record_failure(index, visual.mesh, phase, false, window.index_count, lod,
                                "windowed cluster DAG material pass recording failed");
                            return false;
                        }
                        cluster_rendered = true;
                    }
                }
                if (!cluster_rendered)
                {
                    const ClusterLodSelection selection = select_cluster_lod(
                        window_cluster->data, frame, mvp, Device.vCameraPosition, Device.fFOV);
                    if (selection.disposition == ClusterLodDisposition::FullyCulled)
                        cluster_rendered = true;
                    else if (selection.disposition == ClusterLodDisposition::UseClusters)
                    {
                        bool valid = true;
                        for (uint32_t cluster_index : selection.clusters)
                        {
                            if (cluster_index >= window_cluster->data.clusters.size())
                            {
                                valid = false;
                                break;
                            }
                            const ClusterLodCluster& cluster = window_cluster->data.clusters[cluster_index];
                            if (!cluster.index_count || cluster.index_count % 3 ||
                                cluster.first_index > window_cluster->index_count ||
                                cluster.index_count > window_cluster->index_count - cluster.first_index ||
                                window_cluster->first_index > UINT32_MAX - cluster.first_index ||
                                cluster.index_count > UINT32_MAX - window_cluster->first_index - cluster.first_index)
                            {
                                valid = false;
                                break;
                            }
                        }
                        if (valid)
                        {
                            for (uint32_t cluster_index : selection.clusters)
                            {
                                const ClusterLodCluster& cluster = window_cluster->data.clusters[cluster_index];
                                const uint32_t first_index = window_cluster->first_index + cluster.first_index;
                                if (!pass.record_geometry(frame, mesh.vertex_buffer(), mesh.packed_cluster_indices,
                                        cluster.index_count, mvp, mesh.material, mesh.mode, first_index,
                                        named ? vertex : nullptr, named ? fragment : nullptr,
                                        float(mesh.alpha_ref) / 255.f, nullptr, mesh.terrain_scale,
                                        mesh.environment ? camera : nullptr, mesh.laemb ? factor : nullptr,
                                        mesh.first_vertex, &mesh.matrices, VK_NULL_HANDLE, 1, VK_NULL_HANDLE, 0, 0,
                                        first.shader_variant_hash) ||
                                    !record_followup_material_pass(mesh.packed_cluster_indices, cluster.index_count,
                                        first_index, VK_NULL_HANDLE, 0, 0))
                                {
                                    report_visual_record_failure(index, visual.mesh, phase, false,
                                        cluster.index_count, lod, "windowed cluster DAG draw failed");
                                    return false;
                                }
                            }
                            cluster_rendered = true;
                        }
                    }
                }
            }
            if (!cluster_rendered && !window_cluster && !transparent &&
                supports_cluster_lod_pass_graph(mesh.pass_graph) &&
                mesh.packed_cluster_indices && !mesh.cluster_lod.groups.empty())
            {
                if (cluster_visibility_.ready())
                {
                    const VkBuffer commands = cluster_visibility_.command_buffer(frame.frame_index);
                    const VkDeviceSize command_offset = cluster_visibility_.command_offset(
                        0, mesh.cluster_first_draw_command);
                    if (commands && mesh.cluster_cull_cluster_count == mesh.cluster_lod.clusters.size() &&
                        pass.record_geometry(frame, mesh.vertex_buffer(), mesh.packed_cluster_indices,
                            0, mvp, mesh.material, mesh.mode, 0, named ? vertex : nullptr,
                            named ? fragment : nullptr, float(mesh.alpha_ref) / 255.f, nullptr,
                            mesh.terrain_scale, mesh.environment ? camera : nullptr,
                            mesh.laemb ? factor : nullptr, mesh.first_vertex, &mesh.matrices,
                            VK_NULL_HANDLE, 1, commands, command_offset,
                            mesh.cluster_cull_cluster_count, first.shader_variant_hash))
                    {
                        if (!record_followup_material_pass(mesh.packed_cluster_indices, 0, 0, commands,
                                command_offset, mesh.cluster_cull_cluster_count))
                        {
                            report_visual_record_failure(index, visual.mesh, phase, false, window.index_count, lod,
                                "cluster DAG material pass 1 recording failed");
                            return false;
                        }
                        cluster_rendered = true;
                    }
                    // GPU visibility is an optional draw optimization. Keep the
                    // authored mesh visible if per-frame indirect data or its
                    // metadata is unavailable on a device or frame.
                }
                else
                {
                    const ClusterLodSelection selection = select_cluster_lod(
                        mesh.cluster_lod, frame, mvp, Device.vCameraPosition, Device.fFOV);
                    if (selection.disposition == ClusterLodDisposition::FullyCulled)
                    {
                        cluster_rendered = true;
                    }
                    else if (selection.disposition == ClusterLodDisposition::UseClusters)
                    {
                        const std::vector<uint32_t>& cluster_draws = selection.clusters;
                        bool recorded_indirect = false;
                        if (pass.supports_multi_draw_indirect() && frame.frame_index < cluster_draw_commands_.size() &&
                            cluster_draw_commands_[frame.frame_index].handle() &&
                            mesh.cluster_first_draw_command <= cluster_draw_command_count_ &&
                            mesh.cluster_lod.clusters.size() <= cluster_draw_command_count_ -
                                mesh.cluster_first_draw_command)
                        {
                        std::vector<VkDrawIndexedIndirectCommand> commands(mesh.cluster_lod.clusters.size());
                        for (uint32_t cluster_index : cluster_draws)
                        {
                            if (cluster_index >= mesh.cluster_lod.clusters.size())
                                return false;
                            const ClusterLodCluster& cluster = mesh.cluster_lod.clusters[cluster_index];
                            auto& command = commands[cluster_index];
                            command.indexCount = cluster.index_count;
                            command.instanceCount = 1;
                            command.firstIndex = mesh.cluster_first_index + cluster.first_index;
                            command.vertexOffset = static_cast<int32_t>(mesh.first_vertex);
                        }
                        const VkDeviceSize command_offset = VkDeviceSize(mesh.cluster_first_draw_command) *
                            sizeof(VkDrawIndexedIndirectCommand);
                        std::string write_error;
                        if (cluster_draw_commands_[frame.frame_index].write(command_offset, commands.data(),
                                commands.size() * sizeof(VkDrawIndexedIndirectCommand), write_error) &&
                            pass.record_geometry(frame, mesh.vertex_buffer(), mesh.packed_cluster_indices,
                                0, mvp, mesh.material, mesh.mode, 0, named ? vertex : nullptr,
                                named ? fragment : nullptr, float(mesh.alpha_ref) / 255.f, nullptr,
                                mesh.terrain_scale, mesh.environment ? camera : nullptr,
                                mesh.laemb ? factor : nullptr, mesh.first_vertex, &mesh.matrices,
                                VK_NULL_HANDLE, 1, cluster_draw_commands_[frame.frame_index].handle(),
                                command_offset, static_cast<uint32_t>(commands.size()), first.shader_variant_hash))
                        {
                            if (!record_followup_material_pass(mesh.packed_cluster_indices, 0, 0,
                                    cluster_draw_commands_[frame.frame_index].handle(), command_offset,
                                    static_cast<uint32_t>(commands.size())))
                            {
                                report_visual_record_failure(index, visual.mesh, phase, false, window.index_count,
                                    lod, "cluster DAG material pass 1 recording failed");
                                return false;
                            }
                            recorded_indirect = true;
                        }
                        }
                        if (!recorded_indirect)
                        {
                            for (uint32_t cluster_index : cluster_draws)
                            {
                                if (cluster_index >= mesh.cluster_lod.clusters.size())
                                    return false;
                                const ClusterLodCluster& cluster = mesh.cluster_lod.clusters[cluster_index];
                                if (!pass.record_geometry(frame, mesh.vertex_buffer(), mesh.packed_cluster_indices,
                                        cluster.index_count, mvp, mesh.material, mesh.mode,
                                        mesh.cluster_first_index + cluster.first_index,
                                        named ? vertex : nullptr, named ? fragment : nullptr,
                                        float(mesh.alpha_ref) / 255.f, nullptr, mesh.terrain_scale,
                                        mesh.environment ? camera : nullptr, mesh.laemb ? factor : nullptr,
                                        mesh.first_vertex, &mesh.matrices, VK_NULL_HANDLE, 1, VK_NULL_HANDLE, 0, 0,
                                        first.shader_variant_hash))
                                {
                                    report_visual_record_failure(index, visual.mesh, phase, mesh.water,
                                        cluster.index_count, lod, "cluster DAG geometry draw failed");
                                    return false;
                                }
                                if (!record_followup_material_pass(mesh.packed_cluster_indices, cluster.index_count,
                                        mesh.cluster_first_index + cluster.first_index, VK_NULL_HANDLE, 0, 0))
                                {
                                    report_visual_record_failure(index, visual.mesh, phase, false,
                                        cluster.index_count, lod, "cluster DAG material pass 1 recording failed");
                                    return false;
                                }
                            }
                        }
                        cluster_rendered = true;
                    }
                }
            }
            if (!cluster_rendered && !(transparent
                      ? mesh.water
                            ? pass.record_water(frame, mesh.vertex_buffer(), mesh.index_buffer(), window.index_count,
                                                mvp, mesh.material, mesh.first_index + window.offset,
                                                Device.fTimeGlobal, mesh.glass ? -.35f : .72f, mesh.first_vertex)
                            : pass.record_transparent(
                                  frame, mesh.vertex_buffer(), mesh.index_buffer(), window.index_count, mvp,
                                  mesh.material, mesh.first_index + window.offset, named ? vertex : nullptr,
                                  named ? fragment : nullptr, &failure, float(mesh.alpha_ref) / 255.f, nullptr,
                                  mesh.terrain_scale, mesh.first_vertex, &mesh.matrices, first.shader_variant_hash)
                      : pass.record_geometry(frame, mesh.vertex_buffer(), mesh.index_buffer(), window.index_count, mvp,
                                             mesh.material, mesh.mode, mesh.first_index + window.offset,
                                             named ? vertex : nullptr, named ? fragment : nullptr,
                                             float(mesh.alpha_ref) / 255.f, nullptr, mesh.terrain_scale,
                                             mesh.environment ? camera : nullptr, mesh.laemb ? factor : nullptr,
                                             mesh.first_vertex, &mesh.matrices, VK_NULL_HANDLE, 1, VK_NULL_HANDLE, 0,
                                             0, first.shader_variant_hash)))
            {
                report_visual_record_failure(index, visual.mesh, phase, mesh.water, window.index_count, lod, failure);
                return false;
            }
            if (!cluster_rendered && mesh.pass_graph.count > 1 &&
                !record_followup_material_pass(mesh.index_buffer(), window.index_count,
                    mesh.first_index + window.offset, VK_NULL_HANDLE, 0, 0))
            {
                report_visual_record_failure(index, visual.mesh, phase, false, window.index_count, lod,
                                             "material pass 1 multiply failed");
                return false;
            }
        }
    }
    for (uint32_t child : visual.children)
    {
        if (child >= visuals_.size())
            return false;
        const auto &bounds = visuals_[child].bounds;
        const float dx = bounds[6] - Device.vCameraPosition.x;
        const float dy = bounds[7] - Device.vCameraPosition.y;
        const float dz = bounds[8] - Device.vCameraPosition.z;
        const float child_lod = lod_for_distance(bounds[9], dx * dx + dy * dy + dz * dz);
        if (!record_visual(child, frame, pass, mvp, phase, child_lod, cull_static, secondary_only, priority_filter))
            return false;
    }
    return true;
}

bool GpuLevel::record_tree_velocity(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
    const float (&current_mvp)[16], const float (&previous_mvp)[16],
    const TreeWindParameters& current_wind, const TreeWindParameters& previous_wind) const
{
    if (!pass.ready()) return false;
    std::vector<uint8_t> visited(visuals_.size(), 0);
    const auto visit = [&](const auto& self, uint32_t index) -> bool {
        if (index >= visuals_.size()) return false;
        if (visited[index]) return true;
        visited[index] = 1;
        const auto& visual = visuals_[index];
        if (index < cull_bounds_.size())
        {
            const auto& sphere = cull_bounds_[index];
            const float amplitude = std::max(std::hypot(current_wind.direction[0], current_wind.direction[1]),
                std::hypot(previous_wind.direction[0], previous_wind.direction[1]));
            if (!visible_sphere(current_mvp, sphere[0], sphere[1], sphere[2],
                    wind_expanded_radius(sphere[3], cull_wind_extents_[index], amplitude))) return true;
        }
        if (visual.mesh >= 0)
        {
            if (size_t(visual.mesh) >= meshes_.size()) return false;
            const auto& mesh = meshes_[visual.mesh];
            if (mesh.tree_wave && !mesh.secondary_opaque &&
                (mesh.mode == SurfaceMode::Opaque || mesh.mode == SurfaceMode::AlphaTest))
            {
                const float dx = visual.bounds[6] - Device.vCameraPosition.x;
                const float dy = visual.bounds[7] - Device.vCameraPosition.y;
                const float dz = visual.bounds[8] - Device.vCameraPosition.z;
                const auto window = select_slide_window(mesh.windows,
                    lod_for_distance(visual.bounds[9], dx * dx + dy * dy + dz * dz), mesh.index_count);
                float rows[8]{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
                if (mesh.matrices.enabled[0])
                {
                    const auto matrix = mesh.matrices.values[0].evaluate(Device.fTimeGlobal,
                        reinterpret_cast<const float*>(&Device.mView));
                    for (size_t row = 0; row < 2; ++row)
                        std::copy(matrix[row].begin(), matrix[row].end(), rows + row * 4);
                }
                pass.draw(frame, mesh.vertex_buffer(), mesh.vertex_buffer(), mesh.index_buffer(),
                    window.index_count, mesh.first_index + window.offset, static_cast<int32_t>(mesh.first_vertex),
                    current_mvp, previous_mvp, mesh.mode == SurfaceMode::AlphaTest ? mesh.material : VK_NULL_HANDLE,
                    float(mesh.alpha_ref) / 255.f, rows, &current_wind, &previous_wind);
            }
        }
        for (uint32_t child : visual.children) if (!self(self, child)) return false;
        return true;
    };
    for (uint32_t root : roots_) if (!visit(visit, root)) return false;
    return true;
}

bool GpuLevel::supports_dynamic_occluder_mask(size_t index) const
{
    const auto visit = [&](const auto& self, uint32_t visual_index) -> bool {
        if (visual_index >= visuals_.size()) return false;
        const LevelVisual& visual = visuals_[visual_index];
        if (visual.mesh >= 0)
        {
            if (size_t(visual.mesh) >= meshes_.size()) return false;
            const Mesh& mesh = meshes_[visual.mesh];
            if ((mesh.mode == SurfaceMode::Opaque || mesh.mode == SurfaceMode::AlphaTest) &&
                mesh.secondary_opaque)
                return false;
        }
        for (uint32_t child : visual.children)
            if (!self(self, child)) return false;
        return true;
    };
    return visit(visit, static_cast<uint32_t>(index));
}

bool GpuLevel::record_object_velocity_visual(size_t index, const FrameRecordingContext& frame,
    const ObjectVelocityPass& pass, const float (&current_mvp)[16],
    const float (&previous_mvp)[16], float lod, bool mark_base_geometry) const
{
    if (index >= visuals_.size() || !pass.ready()) return false;
    if (index < visual_phases_.size() && !(visual_phases_[index] & 1u)) return true;
    const LevelVisual& visual = visuals_[index];
    const auto& sphere = cull_bounds_[index];
    if (!visible_sphere(current_mvp, sphere[0], sphere[1], sphere[2], sphere[3]))
        return true;
    const float billboard_alpha = lod_billboard_alpha(
        visual, frame.extent, Device.fFOV, Device.vCameraPosition);
    if (visual.type == 6 && visual.bounds[9] > 0.f && billboard_alpha < 0.f)
        return true;
    if (billboard_alpha >= 96.f / 255.f)
    {
        Fvector direction;
        direction.set(visual.bounds[6] - Device.vCameraPosition.x,
            visual.bounds[7] - Device.vCameraPosition.y,
            visual.bounds[8] - Device.vCameraPosition.z);
        auto views = select_lod_views(visual.lod_normals, {direction.x, direction.y, direction.z});
        if (views.first > views.second)
        {
            std::swap(views.first, views.second);
            views.weight = 1.f - views.weight;
        }
        const int32_t mesh_index = visual.lod_facets[views.first];
        if (mesh_index < 0 || size_t(mesh_index) >= meshes_.size()) return false;
        const Mesh& billboard = meshes_[mesh_index];
        if (billboard.mode == SurfaceMode::Transparent)
            return true;
        if (!billboard.material || !billboard.vertex_buffer() || !billboard.index_buffer())
            return false;
        direction.normalize_safe().mul(-.5f * visual.bounds[9]);
        Fmatrix view_projection, offset, shifted;
        std::memcpy(&view_projection, current_mvp, sizeof(view_projection));
        offset.translate(direction);
        shifted.mul(view_projection, offset);
        float billboard_mvp[16];
        std::memcpy(billboard_mvp, &shifted, sizeof(billboard_mvp));
        const float alpha_ref = (billboard.lod_atlas ? 1.f : -1.f) *
            (96.f / 255.f) / billboard_alpha;
        pass.draw_lod(frame, billboard.vertex_buffer(), billboard.index_buffer(), 6,
            billboard.first_index + (views.second - views.first) * 6,
            billboard_mvp, billboard.material, alpha_ref, views.weight);
        if (billboard_alpha >= 1.f)
            return true;
    }
    if (mark_base_geometry && visual.mesh >= 0)
    {
        if (size_t(visual.mesh) >= meshes_.size()) return false;
        const Mesh& mesh = meshes_[visual.mesh];
        const bool cutout = mesh.mode == SurfaceMode::AlphaTest && !mesh.lod_atlas;
        if ((mesh.mode == SurfaceMode::Opaque || cutout) && !mesh.secondary_opaque)
        {
            const SlideWindow window = select_slide_window(mesh.windows, lod, mesh.index_count);
            const size_t window_index = select_slide_window_index(mesh.windows, lod);
            const Mesh::ClusterWindow* window_cluster = window_index < mesh.cluster_lod_windows.size()
                ? &mesh.cluster_lod_windows[window_index] : nullptr;
            if (!mesh.vertex_buffer() || !mesh.index_buffer()) return false;
            float base_rows[8]{1.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f};
            if (cutout && mesh.matrices.enabled[0])
            {
                const auto matrix = mesh.matrices.values[0].evaluate(Device.fTimeGlobal,
                    reinterpret_cast<const float*>(&Device.mView));
                for (size_t row = 0; row < 2; ++row)
                    std::copy(matrix[row].begin(), matrix[row].end(), base_rows + row * 4);
            }
            const std::array<float, 3> camera_position{
                Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z};
            const ClusterLodData* cluster_data = nullptr;
            uint32_t cluster_first_index = 0;
            uint32_t cluster_first_draw_command = 0;
            uint32_t cluster_cull_count = 0;
            if (window_cluster)
            {
                cluster_data = &window_cluster->data;
                cluster_first_index = window_cluster->first_index;
                cluster_first_draw_command = window_cluster->first_draw_command;
                cluster_cull_count = window_cluster->cull_cluster_count;
            }
            else if (!mesh.cluster_lod.groups.empty() && !mesh.cluster_lod.clusters.empty())
            {
                cluster_data = &mesh.cluster_lod;
                cluster_first_index = mesh.cluster_first_index;
                cluster_first_draw_command = mesh.cluster_first_draw_command;
                cluster_cull_count = mesh.cluster_cull_cluster_count;
            }
            bool clustered = false;
            if (cluster_data && mesh.packed_cluster_indices &&
                !cluster_data->groups.empty() && !cluster_data->clusters.empty())
            {
                if (cluster_visibility_.ready() && cluster_cull_count == cluster_data->clusters.size())
                {
                    const VkBuffer commands = cluster_visibility_.command_buffer(frame.frame_index);
                    if (commands)
                    {
                        pass.draw(frame, mesh.vertex_buffer(), mesh.vertex_buffer(),
                            mesh.packed_cluster_indices, 0, 0, static_cast<int32_t>(mesh.first_vertex),
                            current_mvp, previous_mvp, cutout ? mesh.material : VK_NULL_HANDLE,
                            float(mesh.alpha_ref) / 255.f, base_rows, nullptr, nullptr, commands,
                            cluster_visibility_.command_offset(0, cluster_first_draw_command),
                            cluster_cull_count);
                        clustered = true;
                    }
                }
                if (!clustered)
                {
                    const ClusterLodSelection selection = select_cluster_lod(*cluster_data,
                        frame.extent, current_mvp, nullptr, camera_position, Device.fFOV);
                    if (selection.disposition == ClusterLodDisposition::FullyCulled)
                        clustered = true;
                    else if (selection.disposition == ClusterLodDisposition::UseClusters)
                    {
                        bool valid = true;
                        for (uint32_t cluster_index : selection.clusters)
                        {
                            if (cluster_index >= cluster_data->clusters.size())
                            {
                                valid = false;
                                break;
                            }
                            const ClusterLodCluster& cluster = cluster_data->clusters[cluster_index];
                            if (!cluster.index_count || cluster.index_count % 3 ||
                                cluster.first_index > cluster_data->indices.size() ||
                                cluster.index_count > cluster_data->indices.size() - cluster.first_index ||
                                cluster_first_index > UINT32_MAX - cluster.first_index ||
                                cluster.index_count > UINT32_MAX - cluster_first_index - cluster.first_index)
                            {
                                valid = false;
                                break;
                            }
                        }
                        if (valid)
                        {
                            for (uint32_t cluster_index : selection.clusters)
                            {
                                const ClusterLodCluster& cluster = cluster_data->clusters[cluster_index];
                                pass.draw(frame, mesh.vertex_buffer(), mesh.vertex_buffer(),
                                    mesh.packed_cluster_indices, cluster.index_count,
                                    cluster_first_index + cluster.first_index,
                                    static_cast<int32_t>(mesh.first_vertex), current_mvp, previous_mvp,
                                    cutout ? mesh.material : VK_NULL_HANDLE,
                                    float(mesh.alpha_ref) / 255.f, base_rows);
                            }
                            clustered = true;
                        }
                    }
                }
            }
            if (!clustered)
                pass.draw(frame, mesh.vertex_buffer(), mesh.vertex_buffer(), mesh.index_buffer(),
                    window.index_count, mesh.first_index + window.offset,
                    static_cast<int32_t>(mesh.first_vertex), current_mvp, previous_mvp,
                    cutout ? mesh.material : VK_NULL_HANDLE, float(mesh.alpha_ref) / 255.f, base_rows);
        }
    }
    for (uint32_t child : visual.children)
    {
        if (child >= visuals_.size()) return false;
        const auto& bounds = visuals_[child].bounds;
        const float dx = bounds[6] - Device.vCameraPosition.x;
        const float dy = bounds[7] - Device.vCameraPosition.y;
        const float dz = bounds[8] - Device.vCameraPosition.z;
        const float child_lod = lod_for_distance(bounds[9], dx * dx + dy * dy + dz * dz);
        if (!record_object_velocity_visual(child, frame, pass, current_mvp, previous_mvp,
                child_lod, mark_base_geometry))
            return false;
    }
    return true;
}

bool GpuLevel::record_hud_visual(size_t index, const FrameRecordingContext &frame, const DeferredPass &pass,
                                 const float (&mvp)[16], float lod) const
{
    if (index >= visuals_.size())
        return false;
    const LevelVisual &visual = visuals_[index];
    if (visual.mesh >= 0)
    {
        if (static_cast<size_t>(visual.mesh) >= meshes_.size())
            return false;
        const Mesh &base = meshes_[visual.mesh];
        const Mesh &mesh = base;
        const SlideWindow window = select_slide_window(mesh.windows, lod, mesh.index_count);
        if (!pass.record_hud(frame, mesh.vertex_buffer(), mesh.index_buffer(), window.index_count, mvp, mesh.material,
                             mesh.first_index + window.offset, nullptr, nullptr, 0.f, nullptr, mesh.first_vertex,
                             &mesh.matrices))
            return false;
    }
    for (uint32_t child : visual.children)
        if (!record_hud_visual(child, frame, pass, mvp, lod))
            return false;
    return true;
}

IRenderVisual *GpuLevel::get_visual(size_t index) const
{
    return index < visual_objects_.size() ? visual_objects_[index].get() : nullptr;
}

int GpuLevel::find_visual_index(const IRenderVisual *visual) const
{
    const auto it = visual_indices_.find(visual);
    return it == visual_indices_.end() ? -1 : static_cast<int>(it->second);
}

const LevelVisual *GpuLevel::visual_node(size_t index) const
{
    return index < visuals_.size() ? &visuals_[index] : nullptr;
}

size_t GpuLevel::camera_sector(const Fvector &position, const Fvector &direction, float geometry_distance,
                               size_t geometry_sector) const
{
    return camera_sector_from_portals(portals_, position, direction, geometry_distance, geometry_sector);
}

bool GpuLevel::visible_sector_roots(size_t camera_sector, const Fmatrix &view_projection,
                                    const Fvector &camera_position, std::vector<uint32_t> &roots) const
{
    if (!select_visible_sector_roots(sectors_, portals_, visuals_.size(), camera_sector, view_projection,
                                     camera_position, roots))
        return false;
    append_unsectored_level_roots(roots_, sectors_, visuals_.size(), roots);
    return true;
}

void GpuLevel::all_level_roots(std::vector<uint32_t> &roots) const
{
    roots.clear();
    std::vector<uint8_t> added(visuals_.size(), 0);
    const auto add = [&](uint32_t root) {
        if (root < visuals_.size() && !added[root])
        {
            roots.push_back(root);
            added[root] = 1;
        }
    };
    for (uint32_t root : roots_)
        add(root);
    for (const LevelSector &sector : sectors_)
        add(sector.root);
}

bool GpuLevel::retire_completed_uploads()
{
    if (device_)
        collect_completed_buffer_uploads(device_, pool_, upload_, pending_);
    return pending_.empty();
}

void GpuLevel::destroy()
{
    draw_failures_.report_repeats([](const std::string &key, uint64_t count) {
        Msg("! [renderer-vulkan] file=$level$/level.geom repeated draw failure %s occurrences=%llu", key.c_str(),
            static_cast<unsigned long long>(count));
    });
    draw_failures_.clear();
    ++revision_;
    if (device_)
        wait_for_buffer_uploads(device_, pool_, upload_, pending_);
    ray_scene_.destroy();
    visual_indices_.clear();
    visual_objects_.clear();
    if (textures_ && pass_)
        for (const Mesh &mesh : meshes_)
        {
            if (!mesh.owns_material || !mesh.material)
                continue;
            if (mesh.lod_atlas)
                textures_->release_lightmapped_material(mesh.material, *pass_);
            else if (mesh.surface || mesh.terrain || mesh.single_detail || mesh.environment)
                textures_->release_terrain_material(mesh.material, *pass_);
            else if (mesh.lightmapped)
                textures_->release_lightmapped_material(mesh.material, *pass_);
            else
                textures_->release_material(mesh.material, *pass_);
        }
    if (textures_ && pass_)
        for (const Mesh &mesh : detail_meshes_)
            if (mesh.surface || mesh.single_detail)
                textures_->release_terrain_material(mesh.material, *pass_);
            else
                textures_->release_material(mesh.material, *pass_);
    cluster_visibility_.destroy();
    meshes_.clear();
    geometry_batches_.clear();
    for (BufferResource& commands : cluster_draw_commands_)
        commands.destroy();
    cluster_draw_command_count_ = 0;
    ray_batches_.clear();
    ray_texture_matrices_.clear();
    ray_texture_names_.clear();
    ray_diffuse_texture_names_.clear();
    ray_diffuse_texture_linear_.clear();
    has_water_ = false;
    detail_meshes_.clear();
    details_ = {};
    detail_cache_.clear();
    visible_details_.clear();
    for (auto &buffers : detail_batches_)
        buffers.clear();
    for (auto &buffers : transparent_detail_batches_)
        buffers.clear();
    for (auto &draws : transparent_detail_draws_)
        draws.clear();
    detail_signatures_.fill(0);
    visuals_.clear();
    cull_bounds_.clear();
    cull_wind_extents_.clear();
    visual_phases_.clear();
    visual_priorities_.clear();
    roots_.clear();
    sectors_.clear();
    portals_.clear();
    pending_.clear();
    device_ = VK_NULL_HANDLE;
    profile_ = LevelGameProfile::Unknown;
    pool_ = VK_NULL_HANDLE;
    upload_ = {};
    textures_ = nullptr;
    pass_ = nullptr;
}
} // namespace xray::render::vulkan
