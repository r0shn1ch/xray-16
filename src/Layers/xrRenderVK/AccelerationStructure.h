#pragma once

#include "BufferResource.h"
#include "BufferUpload.h"
#include <vector>

namespace xray::render::vulkan
{
struct AccelerationStructureDispatch
{
    BufferResourceDispatch buffers{};
    PFN_vkGetBufferDeviceAddress get_buffer_address{};
    PFN_vkCreateAccelerationStructureKHR create{};
    PFN_vkDestroyAccelerationStructureKHR destroy{};
    PFN_vkGetAccelerationStructureBuildSizesKHR get_build_sizes{};
    PFN_vkCmdBuildAccelerationStructuresKHR cmd_build{};
    PFN_vkGetAccelerationStructureDeviceAddressKHR get_address{};
};

AccelerationStructureDispatch load_acceleration_structure_dispatch(VkDevice device,
    PFN_vkGetDeviceProcAddr get_proc, bool core_buffer_address);

// Addresses refer to the start of packed vertex/index buffers. Index values
// remain local to one mesh, as with vkCmdDrawIndexed's vertexOffset.
struct RayIndexedMesh
{
    VkDeviceAddress vertices{};
    VkDeviceAddress indices{};
    uint32_t first_vertex{};
    uint32_t vertex_count{};
    uint32_t first_index{};
    uint32_t index_count{};
    uint32_t vertex_stride{};
    uint32_t position_offset{};
    bool opaque{};
};

bool make_ray_triangle_geometry(const RayIndexedMesh& mesh,
    VkAccelerationStructureGeometryKHR& geometry, uint32_t& primitive_count,
    std::string& error);

// Owns the AS and its build scratch. Callers keep geometry inputs alive until
// the submitted build completes and synchronize subsequent reads or updates.
class AccelerationStructure
{
public:
    ~AccelerationStructure() { destroy(); }
    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;
    AccelerationStructure() = default;

    bool initialize(VkDevice device, VkAccelerationStructureTypeKHR type,
        const VkAccelerationStructureGeometryKHR& geometry, uint32_t primitive_count,
        VkBuildAccelerationStructureFlagsKHR flags, VkDeviceSize scratch_alignment,
        const VkPhysicalDeviceMemoryProperties& memory,
        const AccelerationStructureDispatch& dispatch, std::string& error);
    bool initialize(VkDevice device, VkAccelerationStructureTypeKHR type,
        const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
        const std::vector<uint32_t>& primitive_counts,
        VkBuildAccelerationStructureFlagsKHR flags, VkDeviceSize scratch_alignment,
        const VkPhysicalDeviceMemoryProperties& memory,
        const AccelerationStructureDispatch& dispatch, std::string& error);
    bool record_build(VkCommandBuffer command_buffer,
        const VkAccelerationStructureGeometryKHR& geometry, uint32_t primitive_count,
        bool update, std::string& error) const;
    bool record_build(VkCommandBuffer command_buffer,
        const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
        const std::vector<uint32_t>& primitive_counts,
        bool update, std::string& error) const;
    void destroy();
    // Static structures can discard build scratch after their build fence completes.
    void release_scratch();

    VkAccelerationStructureKHR handle() const { return as_; }
    VkDeviceAddress address() const { return address_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkAccelerationStructureKHR as_ = VK_NULL_HANDLE;
    VkDeviceAddress address_ = 0;
    VkDeviceAddress scratch_address_ = 0;
    VkAccelerationStructureTypeKHR type_ = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR;
    VkBuildAccelerationStructureFlagsKHR flags_ = 0;
    std::vector<uint32_t> max_primitives_;
    BufferResource storage_;
    BufferResource scratch_;
    AccelerationStructureDispatch vk_{};
};

// Submit one BLAS build after its uploads and wait before returning. Intended
// for asset preparation, not frame recording; the caller owns geometry buffers.
bool build_ray_blas_sync(VkDevice device, VkQueue queue, VkCommandPool pool,
    const BufferUploadDispatch& upload, AccelerationStructure& structure,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<uint32_t>& primitive_counts, std::string& error);
}
