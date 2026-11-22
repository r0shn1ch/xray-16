#pragma once

#include "BufferUpload.h"
#include "DeferredPass.h"
#include "EngineLevelModels.h"
#include "GameTextureFactory.h"
#include "LevelPassGraph.h"
#include "ClusterLodBuilder.h"
#include "AccelerationStructure.h"
#include "LevelRayScene.h"

#include <array>
#include <unordered_map>

class IKinematics;
namespace xray::render::vulkan
{
class ObjectVelocityPass;
struct RayModelGeometry
{
    VkAccelerationStructureGeometryKHR geometry{};
    uint32_t primitive_count{};
    std::string diffuse_texture;
    std::string pbr_texture, metallic_texture, roughness_texture, ao_texture;
    std::string normal_texture;
    std::string normal_aux_texture;
    TextureMatrixSet texture_matrices;
    float alpha_cutoff{};
    bool emissive{};
};
struct RayModelTextureSlots
{
    uint32_t material_id{UINT32_MAX};
    uint32_t alpha{UINT32_MAX}, diffuse{UINT32_MAX}, pbr{UINT32_MAX};
    uint32_t metallic{UINT32_MAX}, roughness{UINT32_MAX}, ao{UINT32_MAX};
    uint32_t normal{UINT32_MAX}, normal_aux{UINT32_MAX};
};
// The caller supplies leased slots from the same tables used by ray queries.
bool make_model_ray_batch(VkDeviceAddress blas, const VkTransformMatrixKHR& transform,
    const std::vector<RayModelGeometry>& geometries, const std::vector<RayModelTextureSlots>& slots,
    RayGeometryBatch& result, std::string& error);
// A standalone OGF model. Vertex buffers are per-flight-slot because an
// animated pose may change while an earlier frame still reads its vertices.
class GpuModel
{
  public:
    ~GpuModel()
    {
        destroy();
    }
    GpuModel(const GpuModel &) = delete;
    GpuModel &operator=(const GpuModel &) = delete;
    GpuModel() = default;

    bool load(const char *name, IReader *source, VkDevice device, VkQueue queue, VkCommandPool pool,
              const VkPhysicalDeviceMemoryProperties &memory, const BufferUploadDispatch &upload,
              GameTextureFactory &textures, DeferredPass &pass, std::string &error,
              PFN_vkGetBufferDeviceAddress ray_address = nullptr);
    bool load_record(const VisualRecord &record, const std::string &inherited_texture, VkDevice device, VkQueue queue,
                     VkCommandPool pool, const VkPhysicalDeviceMemoryProperties &memory,
                     const BufferUploadDispatch &upload, GameTextureFactory &textures, DeferredPass &pass,
                     std::string &error, PFN_vkGetBufferDeviceAddress ray_address = nullptr);
    bool load_decoded(ModelGeometry &&geometry, VkDevice device, VkQueue queue, VkCommandPool pool,
                      const VkPhysicalDeviceMemoryProperties &memory, const BufferUploadDispatch &upload,
                      GameTextureFactory &textures, DeferredPass &pass, std::string &error,
                      PFN_vkGetBufferDeviceAddress ray_address = nullptr)
    {
        return load_geometry(std::move(geometry), device, queue, pool, memory, upload, textures, pass, error, ray_address);
    }
    // Call from the acquired frame's geometry recorder. Pose matrices have
    // the same layout as Fmatrix and include inverse bind transforms.
    bool record(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16], const float *pose,
                size_t bones, std::string &error, GeometryPhase phase = GeometryPhase::OpaqueAndAlphaTest,
                float lod = 1.f, const IKinematics *instance = nullptr, const float *normal_rows = nullptr,
                int priority_filter = -1, float opacity = 1.f, const float *world_transform = nullptr);
    bool record_animated(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16],
                         IKinematics &skeleton, std::string &error,
                         GeometryPhase phase = GeometryPhase::OpaqueAndAlphaTest, float lod = 1.f,
                         const float *normal_rows = nullptr, int priority_filter = -1, float opacity = 1.f,
                         const float *world_transform = nullptr);
    bool record_shadow(const FrameRecordingContext &frame, const DeferredPass &pass, const float (&mvp)[16],
                       IKinematics *skeleton, std::string &error, float lod = 1.f,
                       const float *world_transform = nullptr);
    bool record_object_velocity(const FrameRecordingContext& frame, const ObjectVelocityPass& pass,
        const float* current_mvp, const float* previous_mvp, const IKinematics* instance,
        float lod, std::string& error, const float* world_transform = nullptr,
        bool previous_pose_valid = true) const;
    bool has_secondary_priority() const;
    // The caller retains this model until builds and queries using these addresses retire.
    bool collect_static_ray_geometry(std::vector<RayModelGeometry>& output, std::string& error);
    bool ray_geometry_enabled() const { return ray_address_ != nullptr; }
    bool prepare_static_ray_blas(const AccelerationStructureDispatch& dispatch,
        const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error);
    bool prepare_skinned_ray_blas(uint32_t slot, VkCommandBuffer command, IKinematics& skeleton,
        const AccelerationStructureDispatch& dispatch,
        const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties,
        VkDeviceAddress& address, const std::vector<RayModelGeometry>*& geometry,
        bool& recorded_update, std::string& error);
    VkDeviceAddress static_ray_blas_address() const { return ray_blas_ ? ray_blas_->address() : 0; }
    const std::vector<RayModelGeometry>& static_ray_geometry() const { return ray_geometry_; }
    bool has_transparent_priority(uint8_t priority) const;
    void destroy(); // Caller waits for all submitted frames first.
    bool retire_completed_uploads();
    uint64_t geometry_bytes() const
    {
        uint64_t bytes = 0;
        for (const auto &mesh : meshes_)
        {
            bytes += mesh.skinned_vertices.size() + mesh.indices.size();
            bytes += mesh.cluster_indices.size();
            for (const auto &buffer : mesh.vertices)
                bytes += buffer.size();
            for (const auto &[instance, buffers] : mesh.instance_vertices)
                for (const auto &buffer : buffers)
                    bytes += buffer.size();
            for (const auto &[instance, pose] : mesh.poses)
                for (const auto &buffer : pose.buffers)
                    bytes += buffer.size();
        }
        for (const auto& [instance, frames] : ray_skin_)
            for (const auto& frame : frames)
                for (const auto& buffer : frame.vertices)
                    bytes += buffer.size();
        return bytes;
    }
    void release_instance(const IKinematics *skeleton); // Caller waits for submitted frames first.

  private:
    bool load_geometry(ModelGeometry &&decoded, VkDevice device, VkQueue queue, VkCommandPool pool,
                       const VkPhysicalDeviceMemoryProperties &memory, const BufferUploadDispatch &upload,
                       GameTextureFactory &textures, DeferredPass &pass, std::string &error,
                       PFN_vkGetBufferDeviceAddress ray_address);
    struct Mesh
    {
        struct PoseInstance
        {
            std::array<BufferResource, FrameContext::FramesInFlight> buffers;
            std::array<VkDescriptorSet, FrameContext::FramesInFlight> descriptors{};
            std::array<uint32_t, FrameContext::FramesInFlight> shadow_frames{UINT32_MAX, UINT32_MAX};
        };
        ModelGeometry geometry;
        BufferResource skinned_vertices;
        std::array<BufferResource, FrameContext::FramesInFlight> vertices;
        std::unordered_map<const IKinematics *, std::array<BufferResource, FrameContext::FramesInFlight>>
            instance_vertices;
        std::unordered_map<const IKinematics *, PoseInstance> poses;
        BufferResource indices;
        size_t index_count() const { return static_cast<size_t>(indices.size() / sizeof(uint32_t)); }
        BufferResource cluster_indices;
        struct ClusterWindow
        {
            ClusterLodData data;
            uint32_t first_index{};
        };
        std::vector<ClusterWindow> cluster_lod_windows;
        VkDescriptorSet material{};
        int alpha_ref{128};
        int screen_blend{-1};
        bool gpu_skinning{};
        bool wallmark{};
        uint16_t max_bone_index{};
        bool environment{};
        bool screen_fog{};
        bool surface{};
        bool emissive{};
        bool detail{};
        float detail_scale{1.f};
        TextureMatrixSet matrices;
        uint64_t hud_variant_hash{};
        uint8_t priority{1};
        bool secondary_opaque{};
        bool tree_surface{};
        std::string ray_diffuse_texture;
        std::string ray_pbr_texture, ray_metallic_texture, ray_roughness_texture, ray_ao_texture;
        std::string ray_normal_texture, ray_normal_aux_texture;
        RenderPassGraph pass_graph;
    };
    bool add_meshes(ModelGeometry &&geometry, const std::string &inherited_texture, VkDevice device, VkQueue queue,
                    VkCommandPool pool, const VkPhysicalDeviceMemoryProperties &memory,
                    const BufferUploadDispatch &upload, GameTextureFactory &textures, DeferredPass &pass,
                    std::string &error);

    VkDevice device_{};
    VkQueue queue_{};
    VkCommandPool pool_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    BufferUploadDispatch upload_{};
    GameTextureFactory *textures_{};
    DeferredPass *pass_{};
    std::vector<PendingBufferUpload> pending_;
    std::vector<Mesh> meshes_;
    PFN_vkGetBufferDeviceAddress ray_address_{};
    std::unique_ptr<AccelerationStructure> ray_blas_;
    std::vector<RayModelGeometry> ray_geometry_;
    struct RaySkinSlot
    {
        std::vector<BufferResource> vertices;
        std::vector<RayModelGeometry> geometry;
        std::unique_ptr<AccelerationStructure> blas;
    };
    std::unordered_map<const IKinematics*, std::array<RaySkinSlot, FrameContext::FramesInFlight>> ray_skin_;
};
} // namespace xray::render::vulkan
