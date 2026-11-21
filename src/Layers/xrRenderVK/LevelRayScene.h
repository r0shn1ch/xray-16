#pragma once

#include "AccelerationStructure.h"
#include "BufferUpload.h"

#include <memory>
#include <vector>
#include <array>

namespace xray::render::vulkan
{
inline constexpr uint32_t RayMaterialEmissive = 1u << 0;

struct RayGeometryBatch
{
    VkTransformMatrixKHR transform{{{1.f, 0.f, 0.f, 0.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}}};
    // Optional prebuilt BLAS. Its owner and the exact ordered geometry buffers
    // must outlive this scene and every submitted query that references it.
    VkDeviceAddress external_blas{};
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<uint32_t> primitive_counts;
    std::vector<uint32_t> material_ids;
    std::vector<float> alpha_cutoffs;
    std::vector<uint32_t> texture_slots;
    std::vector<uint32_t> diffuse_texture_slots;
    std::vector<uint32_t> pbr_texture_slots;
    std::vector<uint32_t> metallic_texture_slots;
    std::vector<uint32_t> roughness_texture_slots;
    std::vector<uint32_t> ao_texture_slots;
    std::vector<uint32_t> normal_texture_slots;
    std::vector<uint32_t> normal_aux_texture_slots;
    // Terrain: mask, four detail albedos, packed/metallic/roughness/AO/bump/
    // bump auxiliary maps, then four parallax maps. For non-terrain geometry,
    // slot 28 holds the base parallax map. UINT32_MAX means absent.
    std::vector<std::array<uint32_t, 33>> terrain_texture_slots;
    std::vector<float> terrain_scales;
    std::vector<float> parallax_heights;
    // Material flags are independent of VkAccelerationStructureGeometry flags.
    // Empty means all zero for callers that do not provide extra material data.
    std::vector<uint32_t> material_flags;
};

struct RayHitBatchInfo
{
    uint32_t first_geometry{};
    uint32_t geometry_count{};
    uint32_t padding[2]{};
    float row0[4]{1.f, 0.f, 0.f, 0.f};
    float row1[4]{0.f, 1.f, 0.f, 0.f};
    float row2[4]{0.f, 0.f, 1.f, 0.f};
};
static_assert(sizeof(RayHitBatchInfo) == 64);

// std430-compatible information for indexed triangle fetch after a ray hit.
struct RayHitGeometryInfo
{
    VkDeviceAddress vertices{};
    VkDeviceAddress indices{};
    uint32_t vertex_stride{};
    uint32_t max_vertex{};
    uint32_t material_id{};
    uint32_t material_flags{};
    float alpha_cutoff{};
    uint32_t texture_slot{};
    uint32_t diffuse_texture_slot{};
    uint32_t pbr_texture_slot{};
    uint32_t metallic_texture_slot{};
    uint32_t roughness_texture_slot{};
    uint32_t ao_texture_slot{};
    uint32_t normal_texture_slot{};
    uint32_t normal_aux_texture_slot{};
    uint32_t terrain_texture_slots[33];
    float terrain_scale{1.f};
    float parallax_height{};
};
static_assert(sizeof(RayHitGeometryInfo) == 208);
struct RayTextureTransform
{
    float row0[4]{1.f, 0.f, 0.f, 0.f};
    float row1[4]{0.f, 1.f, 0.f, 0.f};
    float detail_row0[4]{1.f, 0.f, 0.f, 0.f};
    float detail_row1[4]{0.f, 1.f, 0.f, 0.f};
};
static_assert(sizeof(RayTextureTransform) == 64);

// Static level AS owner. Geometry input buffers and the graphics queue belong
// to the caller and must stay alive until the build completes.
class LevelRayScene
{
public:
    ~LevelRayScene() { destroy(); }
    LevelRayScene() = default;
    LevelRayScene(const LevelRayScene&) = delete;
    LevelRayScene& operator=(const LevelRayScene&) = delete;

    bool build(VkDevice device, VkQueue queue, VkCommandPool pool,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferUploadDispatch& upload,
        std::vector<PendingBufferUpload>& pending, const std::vector<RayGeometryBatch>& batches,
        const AccelerationStructureDispatch& dispatch,
        const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error,
        bool allow_transform_updates = false);
    // Caller must wait for all previous users of this scene before this call,
    // then submit the recorded command before any subsequent scene use.
    bool record_transform_update(VkCommandBuffer command,
        const std::vector<VkTransformMatrixKHR>& transforms, std::string& error);
    void destroy();
    VkAccelerationStructureKHR handle() const { return ray_tlas_ ? ray_tlas_->handle() : VK_NULL_HANDLE; }
    VkDeviceAddress address() const { return ray_tlas_ ? ray_tlas_->address() : 0; }
    size_t blas_count() const { return ray_blas_.size(); }
    VkDeviceAddress batch_blas_address(size_t batch) const
    { return batch < batch_blas_addresses_.size() ? batch_blas_addresses_[batch] : 0; }
    const std::vector<uint32_t>& instance_batch_indices() const { return instance_batch_indices_; }
    const BufferResource& hit_batch_buffer() const { return hit_batches_; }
    const BufferResource& hit_geometry_buffer() const { return hit_geometries_; }
    size_t hit_batch_count() const { return hit_batch_count_; }
    size_t hit_geometry_count() const { return hit_geometry_count_; }
    static constexpr uint32_t FrameSlots = 2;
    const BufferResource& texture_transform_buffer(uint32_t slot) const { return texture_transforms_.at(slot); }
    bool write_texture_transforms(uint32_t slot, const std::vector<RayTextureTransform>& transforms,
        std::string& error) const;

private:
    std::vector<std::unique_ptr<AccelerationStructure>> ray_blas_;
    std::unique_ptr<AccelerationStructure> ray_tlas_;
    BufferResource ray_instances_;
    BufferResource hit_batches_;
    BufferResource hit_geometries_;
    mutable std::array<BufferResource, FrameSlots> texture_transforms_;
    size_t hit_batch_count_{};
    size_t hit_geometry_count_{};
    std::vector<uint32_t> instance_batch_indices_;
    std::vector<VkDeviceAddress> batch_blas_addresses_;
    std::vector<VkAccelerationStructureInstanceKHR> instance_data_;
    std::vector<RayHitBatchInfo> batch_data_;
    VkDeviceAddress instance_address_{};
    PFN_vkCmdPipelineBarrier update_barrier_{};
    bool transform_updates_{};
};

// One independently writable TLAS and hit table per in-flight frame. The
// caller must wait for the matching frame fence before preparing that slot,
// keep every referenced external BLAS alive, and change topology_revision
// whenever geometry, material slots, or BLAS ownership changes.
class FrameRayScenes
{
public:
    bool prepare(uint32_t slot, uint64_t topology_revision, VkCommandBuffer command,
        VkDevice device, VkQueue queue, VkCommandPool pool, const VkPhysicalDeviceMemoryProperties& memory,
        const BufferUploadDispatch& upload, std::vector<PendingBufferUpload>& pending,
        const std::vector<RayGeometryBatch>& batches, const AccelerationStructureDispatch& dispatch,
        const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error,
        bool force_update = false);
    const LevelRayScene* scene(uint32_t slot) const;
    // On aborted recording or failed submit, discard the slot after it is idle.
    void discard(uint32_t slot);
    void destroy(); // All slots must be idle.
private:
    struct Slot
    {
        LevelRayScene scene;
        std::vector<VkTransformMatrixKHR> transforms;
        uint64_t revision{};
        bool valid{};
    };
    std::array<Slot, LevelRayScene::FrameSlots> slots_;
};
}
