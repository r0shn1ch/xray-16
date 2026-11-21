#include "AccelerationStructure.h"

#include <algorithm>
#include <limits>

namespace xray::render::vulkan
{
AccelerationStructureDispatch load_acceleration_structure_dispatch(VkDevice device,
    PFN_vkGetDeviceProcAddr get_proc, bool core_buffer_address)
{
    AccelerationStructureDispatch vk{};
    if (!device || !get_proc) return vk;
#define XR_RT_PROC(field, type, name) vk.field = reinterpret_cast<type>(get_proc(device, name))
    XR_RT_PROC(buffers.create_buffer, PFN_vkCreateBuffer, "vkCreateBuffer");
    XR_RT_PROC(buffers.destroy_buffer, PFN_vkDestroyBuffer, "vkDestroyBuffer");
    XR_RT_PROC(buffers.get_buffer_memory_requirements, PFN_vkGetBufferMemoryRequirements,
        "vkGetBufferMemoryRequirements");
    XR_RT_PROC(buffers.allocate_memory, PFN_vkAllocateMemory, "vkAllocateMemory");
    XR_RT_PROC(buffers.free_memory, PFN_vkFreeMemory, "vkFreeMemory");
    XR_RT_PROC(buffers.bind_buffer_memory, PFN_vkBindBufferMemory, "vkBindBufferMemory");
    XR_RT_PROC(buffers.map_memory, PFN_vkMapMemory, "vkMapMemory");
    XR_RT_PROC(buffers.unmap_memory, PFN_vkUnmapMemory, "vkUnmapMemory");
    XR_RT_PROC(get_buffer_address, PFN_vkGetBufferDeviceAddress,
        core_buffer_address ? "vkGetBufferDeviceAddress" : "vkGetBufferDeviceAddressKHR");
    XR_RT_PROC(create, PFN_vkCreateAccelerationStructureKHR, "vkCreateAccelerationStructureKHR");
    XR_RT_PROC(destroy, PFN_vkDestroyAccelerationStructureKHR, "vkDestroyAccelerationStructureKHR");
    XR_RT_PROC(get_build_sizes, PFN_vkGetAccelerationStructureBuildSizesKHR,
        "vkGetAccelerationStructureBuildSizesKHR");
    XR_RT_PROC(cmd_build, PFN_vkCmdBuildAccelerationStructuresKHR,
        "vkCmdBuildAccelerationStructuresKHR");
    XR_RT_PROC(get_address, PFN_vkGetAccelerationStructureDeviceAddressKHR,
        "vkGetAccelerationStructureDeviceAddressKHR");
#undef XR_RT_PROC
    return vk;
}

bool make_ray_triangle_geometry(const RayIndexedMesh& mesh,
    VkAccelerationStructureGeometryKHR& geometry, uint32_t& primitive_count,
    std::string& error)
{
    geometry = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    primitive_count = 0;
    if (!mesh.vertices || !mesh.indices || mesh.vertex_count < 3 ||
        mesh.vertices % alignof(float) || mesh.indices % alignof(uint32_t) ||
        !mesh.index_count || mesh.index_count % 3 ||
        mesh.vertex_stride < 3 * sizeof(float) ||
        mesh.position_offset > mesh.vertex_stride - 3 * sizeof(float) ||
        mesh.position_offset % alignof(float) ||
        mesh.vertex_stride % alignof(float) ||
        mesh.vertices > UINT64_MAX - mesh.position_offset ||
        mesh.first_vertex > (UINT64_MAX - mesh.vertices - mesh.position_offset) / mesh.vertex_stride ||
        mesh.first_index > (UINT64_MAX - mesh.indices) / sizeof(uint32_t))
    {
        error = "invalid indexed ray triangle mesh or device address overflow";
        return false;
    }
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.flags = mesh.opaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
    geometry.geometry.triangles.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    geometry.geometry.triangles.vertexData.deviceAddress = mesh.vertices +
        VkDeviceAddress(mesh.first_vertex) * mesh.vertex_stride + mesh.position_offset;
    geometry.geometry.triangles.vertexStride = mesh.vertex_stride;
    geometry.geometry.triangles.maxVertex = mesh.vertex_count - 1;
    geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    geometry.geometry.triangles.indexData.deviceAddress = mesh.indices +
        VkDeviceAddress(mesh.first_index) * sizeof(uint32_t);
    primitive_count = mesh.index_count / 3;
    error.clear();
    return true;
}

bool AccelerationStructure::initialize(VkDevice device, VkAccelerationStructureTypeKHR type,
    const VkAccelerationStructureGeometryKHR& geometry, uint32_t primitive_count,
    VkBuildAccelerationStructureFlagsKHR flags, VkDeviceSize scratch_alignment,
    const VkPhysicalDeviceMemoryProperties& memory,
    const AccelerationStructureDispatch& dispatch, std::string& error)
{
    return initialize(device, type, std::vector<VkAccelerationStructureGeometryKHR>{geometry},
        std::vector<uint32_t>{primitive_count}, flags,
        scratch_alignment, memory, dispatch, error);
}

bool AccelerationStructure::initialize(VkDevice device, VkAccelerationStructureTypeKHR type,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<uint32_t>& primitive_counts,
    VkBuildAccelerationStructureFlagsKHR flags, VkDeviceSize scratch_alignment,
    const VkPhysicalDeviceMemoryProperties& memory,
    const AccelerationStructureDispatch& dispatch, std::string& error)
{
    destroy();
    if (!device || (type != VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR &&
        type != VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) || geometries.empty() ||
        geometries.size() != primitive_counts.size() || geometries.size() > UINT32_MAX ||
        !scratch_alignment || (scratch_alignment & (scratch_alignment - 1)) ||
        !dispatch.create || !dispatch.destroy || !dispatch.get_build_sizes ||
        !dispatch.cmd_build || !dispatch.get_address || !dispatch.get_buffer_address)
    {
        error = "invalid acceleration structure type, geometry, alignment or Vulkan procedures";
        return false;
    }
    if ((type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR && geometries.size() != 1) ||
        std::any_of(geometries.begin(), geometries.end(), [&](const auto& geometry)
        { return (type == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) !=
            (geometry.geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR); }) ||
        std::any_of(primitive_counts.begin(), primitive_counts.end(),
            [](uint32_t count) { return count == 0; }))
    {
        error = "acceleration structure type, geometry or primitive counts are invalid";
        return false;
    }
    VkAccelerationStructureBuildGeometryInfoKHR build{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = type;
    build.flags = flags;
    build.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.geometryCount = static_cast<uint32_t>(geometries.size());
    build.pGeometries = geometries.data();
    VkAccelerationStructureBuildSizesInfoKHR sizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    dispatch.get_build_sizes(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &build, primitive_counts.data(), &sizes);
    const VkDeviceSize scratch_size = std::max(sizes.buildScratchSize,
        (flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR) ? sizes.updateScratchSize : 0);
    if (!sizes.accelerationStructureSize || !scratch_size ||
        scratch_size > std::numeric_limits<VkDeviceSize>::max() - (scratch_alignment - 1))
    {
        error = "invalid acceleration structure build size";
        return false;
    }
    vk_ = dispatch;
    device_ = device;
    type_ = type;
    flags_ = flags;
    max_primitives_ = primitive_counts;
    if (!storage_.initialize(device, sizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, dispatch.buffers, error) ||
        !scratch_.initialize(device, scratch_size + scratch_alignment - 1,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, dispatch.buffers, error))
    {
        destroy();
        return false;
    }
    VkAccelerationStructureCreateInfoKHR create{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    create.buffer = storage_.handle();
    create.size = sizes.accelerationStructureSize;
    create.type = type;
    if (dispatch.create(device, &create, nullptr, &as_) != VK_SUCCESS)
    {
        error = "vkCreateAccelerationStructureKHR failed";
        destroy();
        return false;
    }
    VkBufferDeviceAddressInfo scratch_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    scratch_info.buffer = scratch_.handle();
    const VkDeviceAddress raw_scratch = dispatch.get_buffer_address(device, &scratch_info);
    scratch_address_ = (raw_scratch + scratch_alignment - 1) & ~(scratch_alignment - 1);
    VkAccelerationStructureDeviceAddressInfoKHR address_info{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    address_info.accelerationStructure = as_;
    address_ = dispatch.get_address(device, &address_info);
    if (!raw_scratch || !address_ || scratch_address_ < raw_scratch)
    {
        error = "acceleration structure or scratch device address unavailable";
        destroy();
        return false;
    }
    error.clear();
    return true;
}

bool AccelerationStructure::record_build(VkCommandBuffer command_buffer,
    const VkAccelerationStructureGeometryKHR& geometry, uint32_t primitive_count,
    bool update, std::string& error) const
{
    return record_build(command_buffer, std::vector<VkAccelerationStructureGeometryKHR>{geometry},
        std::vector<uint32_t>{primitive_count}, update, error);
}

bool AccelerationStructure::record_build(VkCommandBuffer command_buffer,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<uint32_t>& primitive_counts,
    bool update, std::string& error) const
{
    if (!command_buffer || !as_ || !scratch_address_ || geometries.size() != max_primitives_.size() ||
        primitive_counts.size() != max_primitives_.size() ||
        (update && !(flags_ & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR)) ||
        (type_ == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR && geometries.size() != 1))
    {
        error = "invalid acceleration structure build request";
        return false;
    }
    for (size_t i = 0; i < geometries.size(); ++i)
        if (!primitive_counts[i] || primitive_counts[i] > max_primitives_[i] ||
            ((type_ == VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR) !=
                (geometries[i].geometryType == VK_GEOMETRY_TYPE_INSTANCES_KHR)))
        {
            error = "invalid acceleration structure geometry or primitive count";
            return false;
        }
    VkAccelerationStructureBuildGeometryInfoKHR build{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    build.type = type_;
    build.flags = flags_;
    build.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
        VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build.srcAccelerationStructure = update ? as_ : VK_NULL_HANDLE;
    build.dstAccelerationStructure = as_;
    build.geometryCount = static_cast<uint32_t>(geometries.size());
    build.pGeometries = geometries.data();
    build.scratchData.deviceAddress = scratch_address_;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(geometries.size());
    for (size_t i = 0; i < ranges.size(); ++i)
        ranges[i].primitiveCount = primitive_counts[i];
    const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = ranges.data();
    vk_.cmd_build(command_buffer, 1, &build, &range_ptr);
    error.clear();
    return true;
}

void AccelerationStructure::destroy()
{
    if (device_ && as_ && vk_.destroy) vk_.destroy(device_, as_, nullptr);
    as_ = VK_NULL_HANDLE;
    address_ = 0;
    scratch_address_ = 0;
    scratch_.destroy();
    storage_.destroy();
    device_ = VK_NULL_HANDLE;
    type_ = VK_ACCELERATION_STRUCTURE_TYPE_GENERIC_KHR;
    flags_ = 0;
    max_primitives_.clear();
    vk_ = {};
}

void AccelerationStructure::release_scratch()
{
    scratch_.destroy();
    scratch_address_ = 0;
}

bool build_ray_blas_sync(VkDevice device, VkQueue queue, VkCommandPool pool,
    const BufferUploadDispatch& upload, AccelerationStructure& structure,
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<uint32_t>& primitive_counts, std::string& error)
{
    if (!device || !queue || !pool || !upload.allocate_command_buffers || !upload.free_command_buffers ||
        !upload.begin_command_buffer || !upload.end_command_buffer || !upload.cmd_pipeline_barrier ||
        !upload.create_fence || !upload.destroy_fence || !upload.queue_submit || !upload.wait_for_fences)
    { error = "ray BLAS build needs a queue and complete submission procedures"; return false; }
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false, completed = false;
    do
    {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool;
        allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocate.commandBufferCount = 1;
        if (upload.allocate_command_buffers(device, &allocate, &command) != VK_SUCCESS)
        { error = "ray BLAS command allocation failed"; break; }
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        if (upload.begin_command_buffer(command, &begin) != VK_SUCCESS)
        { error = "ray BLAS command begin failed"; break; }
        VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
        before.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        upload.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &before, 0, nullptr, 0, nullptr);
        if (!structure.record_build(command, geometries, primitive_counts, false, error)) break;
        VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        after.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        after.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        upload.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &after, 0, nullptr, 0, nullptr);
        if (upload.end_command_buffer(command) != VK_SUCCESS)
        { error = "ray BLAS command end failed"; break; }
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (upload.create_fence(device, &fence_info, nullptr, &fence) != VK_SUCCESS)
        { error = "ray BLAS fence creation failed"; break; }
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        if (upload.queue_submit(queue, 1, &submit, fence) != VK_SUCCESS)
        { error = "ray BLAS queue submission failed"; break; }
        submitted = true;
        if (upload.wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        { error = "ray BLAS build did not complete"; break; }
        completed = true;
    } while (false);
    if (submitted && !completed) upload.wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX);
    if (fence) upload.destroy_fence(device, fence, nullptr);
    if (command) upload.free_command_buffers(device, pool, 1, &command);
    if (completed) error.clear();
    return completed;
}
}
