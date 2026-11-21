#include "LevelRayScene.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace xray::render::vulkan
{
namespace
{
bool valid_transform(const VkTransformMatrixKHR& transform)
{
    const auto& m = transform.matrix;
    for (const auto& row : m) for (float value : row) if (!std::isfinite(value)) return false;
    const float determinant = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
        m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
        m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    return std::isfinite(determinant) && std::abs(determinant) >= 1e-12f;
}
void copy_transform(RayHitBatchInfo& batch, const VkTransformMatrixKHR& transform)
{
    std::memcpy(batch.row0, transform.matrix[0], sizeof(batch.row0));
    std::memcpy(batch.row1, transform.matrix[1], sizeof(batch.row1));
    std::memcpy(batch.row2, transform.matrix[2], sizeof(batch.row2));
}
}
bool LevelRayScene::build(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferUploadDispatch& upload,
    std::vector<PendingBufferUpload>& pending, const std::vector<RayGeometryBatch>& batches,
    const AccelerationStructureDispatch& vk,
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error,
    bool allow_transform_updates)
{
    destroy();
    size_t batch_count = 0;
    for (const auto& batch : batches)
        batch_count += !batch.geometries.empty();
    if (!batch_count)
    {
        error.clear();
        return true;
    }
    if (!device || !queue || !pool || !vk.create || !vk.destroy || !vk.get_build_sizes ||
        !vk.cmd_build || !vk.get_address || !vk.get_buffer_address ||
        !upload.allocate_command_buffers || !upload.free_command_buffers ||
        !upload.begin_command_buffer || !upload.end_command_buffer ||
        !upload.cmd_pipeline_barrier || !upload.queue_submit || !upload.create_fence ||
        !upload.destroy_fence || !upload.wait_for_fences ||
        !properties.minAccelerationStructureScratchOffsetAlignment ||
        batch_count > properties.maxInstanceCount || batches.size() > 0x1000000u)
    {
        error = "level ray scene device capabilities or dispatch are incomplete";
        return false;
    }
    if (!wait_for_buffer_uploads(device, pool, upload, pending))
    {
        error = "level ray geometry uploads did not complete";
        return false;
    }

    // Submit and retire each build before the next one allocates scratch.
    // Packed levels can contain many batches; keeping every scratch buffer
    // alive until the TLAS completes can exhaust memory during level loading.
    const auto submit_build = [&](auto&& record) -> bool
    {
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        bool submitted = false;
        bool completed = false;
        do
        {
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            if (upload.allocate_command_buffers(device, &allocate, &command) != VK_SUCCESS)
            {
                error = "level ray build command allocation failed";
                break;
            }
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (upload.begin_command_buffer(command, &begin) != VK_SUCCESS)
            {
                error = "level ray build command begin failed";
                break;
            }
            if (!record(command))
            {
                if (error.empty()) error = "level ray build recording failed";
                break;
            }
            if (upload.end_command_buffer(command) != VK_SUCCESS)
            {
                error = "level ray build command end failed";
                break;
            }
            VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (upload.create_fence(device, &fence_info, nullptr, &fence) != VK_SUCCESS)
            {
                error = "level ray build fence creation failed";
                break;
            }
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            if (upload.queue_submit(queue, 1, &submit, fence) != VK_SUCCESS)
            {
                error = "level ray build submission failed";
                break;
            }
            submitted = true;
            if (upload.wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            {
                error = "level ray build did not complete";
                break;
            }
            completed = true;
        } while (false);
        if (submitted && !completed)
            upload.wait_for_fences(device, 1, &fence, VK_TRUE, UINT64_MAX);
        if (fence) upload.destroy_fence(device, fence, nullptr);
        if (command) upload.free_command_buffers(device, pool, 1, &command);
        return completed;
    };

    std::vector<std::unique_ptr<AccelerationStructure>> blas;
    blas.reserve(batch_count);
    std::vector<uint32_t> batch_indices;
    batch_indices.reserve(batch_count);
    std::vector<VkDeviceAddress> batch_addresses(batches.size());
    std::vector<RayHitBatchInfo> hit_batches(batches.size());
    std::vector<RayHitGeometryInfo> hit_geometries;
    for (size_t batch_index = 0; batch_index < batches.size(); ++batch_index)
    {
        const auto& batch = batches[batch_index];
        if (!valid_transform(batch.transform))
        { error = "level ray instance transform is singular or non-finite"; return false; }
        if (batch.geometries.size() != batch.primitive_counts.size() ||
            batch.geometries.size() != batch.material_ids.size() ||
            batch.geometries.size() != batch.alpha_cutoffs.size() ||
            batch.geometries.size() != batch.texture_slots.size() ||
            batch.geometries.size() != batch.diffuse_texture_slots.size() ||
            batch.geometries.size() != batch.pbr_texture_slots.size() ||
            batch.geometries.size() != batch.metallic_texture_slots.size() ||
            batch.geometries.size() != batch.roughness_texture_slots.size() ||
            batch.geometries.size() != batch.ao_texture_slots.size() ||
            batch.geometries.size() != batch.normal_texture_slots.size() ||
            batch.geometries.size() != batch.normal_aux_texture_slots.size() ||
            (!batch.terrain_texture_slots.empty() && batch.geometries.size() != batch.terrain_texture_slots.size()) ||
            (!batch.terrain_scales.empty() && batch.geometries.size() != batch.terrain_scales.size()) ||
            (!batch.parallax_heights.empty() && batch.geometries.size() != batch.parallax_heights.size()) ||
            (!batch.material_flags.empty() && batch.geometries.size() != batch.material_flags.size()) ||
            batch.geometries.size() > UINT32_MAX ||
            hit_geometries.size() > UINT32_MAX - batch.geometries.size())
        {
            error = "level ray hit geometry metadata is inconsistent";
            return false;
        }
        auto& hit_batch = hit_batches[batch_index];
        hit_batch.first_geometry = static_cast<uint32_t>(hit_geometries.size());
        hit_batch.geometry_count = static_cast<uint32_t>(batch.geometries.size());
        copy_transform(hit_batch, batch.transform);
        for (size_t geometry_index = 0; geometry_index < batch.geometries.size(); ++geometry_index)
        {
            const auto& geometry = batch.geometries[geometry_index];
            const auto& triangles = geometry.geometry.triangles;
            if (geometry.geometryType != VK_GEOMETRY_TYPE_TRIANGLES_KHR ||
                triangles.vertexFormat != VK_FORMAT_R32G32B32_SFLOAT ||
                triangles.indexType != VK_INDEX_TYPE_UINT32 ||
                triangles.vertexStride > UINT32_MAX ||
                !triangles.vertexData.deviceAddress || !triangles.indexData.deviceAddress)
            {
                error = "level ray hit geometry format is unsupported";
                return false;
            }
            RayHitGeometryInfo hit{};
            hit.vertices = triangles.vertexData.deviceAddress;
            hit.indices = triangles.indexData.deviceAddress;
            hit.vertex_stride = static_cast<uint32_t>(triangles.vertexStride);
            hit.max_vertex = triangles.maxVertex;
            hit.material_id = batch.material_ids[geometry_index];
            hit.material_flags = batch.material_flags.empty() ? 0u : batch.material_flags[geometry_index];
            hit.alpha_cutoff = batch.alpha_cutoffs[geometry_index];
            hit.texture_slot = batch.texture_slots[geometry_index];
            hit.diffuse_texture_slot = batch.diffuse_texture_slots[geometry_index];
            hit.pbr_texture_slot = batch.pbr_texture_slots[geometry_index];
            hit.metallic_texture_slot = batch.metallic_texture_slots[geometry_index];
            hit.roughness_texture_slot = batch.roughness_texture_slots[geometry_index];
            hit.ao_texture_slot = batch.ao_texture_slots[geometry_index];
            hit.normal_texture_slot = batch.normal_texture_slots[geometry_index];
            hit.normal_aux_texture_slot = batch.normal_aux_texture_slots[geometry_index];
            std::fill(std::begin(hit.terrain_texture_slots), std::end(hit.terrain_texture_slots), UINT32_MAX);
            if (!batch.terrain_texture_slots.empty())
                std::copy(batch.terrain_texture_slots[geometry_index].begin(),
                    batch.terrain_texture_slots[geometry_index].end(), std::begin(hit.terrain_texture_slots));
            hit.terrain_scale = batch.terrain_scales.empty() ? 1.f : batch.terrain_scales[geometry_index];
            hit.parallax_height = batch.parallax_heights.empty() ? 0.f : batch.parallax_heights[geometry_index];
            hit_geometries.push_back(hit);
        }
        if (batch.geometries.empty()) continue;
        uint64_t primitives = 0;
        for (uint32_t count : batch.primitive_counts) primitives += count;
        if (batch.geometries.size() > properties.maxGeometryCount ||
            primitives > properties.maxPrimitiveCount)
        {
            error = "level ray geometry exceeds acceleration structure limits";
            return false;
        }
        if (batch.external_blas)
        {
            batch_addresses[batch_index] = batch.external_blas;
            batch_indices.push_back(static_cast<uint32_t>(batch_index));
            continue;
        }
        auto structure = std::make_unique<AccelerationStructure>();
        if (!structure->initialize(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                batch.geometries, batch.primitive_counts,
                VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                properties.minAccelerationStructureScratchOffsetAlignment, memory, vk, error))
            return false;
        if (!submit_build([&](VkCommandBuffer command) {
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
                upload.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
                    &barrier, 0, nullptr, 0, nullptr);
                return structure->record_build(command, batch.geometries,
                    batch.primitive_counts, false, error);
            }))
            return false;
        structure->release_scratch();
        batch_addresses[batch_index] = structure->address();
        blas.push_back(std::move(structure));
        batch_indices.push_back(static_cast<uint32_t>(batch_index));
    }

    BufferResource hit_batch_buffer;
    BufferResource hit_geometry_buffer;
    const auto storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const auto host_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!hit_batch_buffer.initialize(device, hit_batches.size() * sizeof(hit_batches[0]),
            storage_usage, host_memory, memory, vk.buffers, error) ||
        !hit_batch_buffer.write(0, hit_batches.data(), hit_batches.size() * sizeof(hit_batches[0]), error) ||
        !hit_geometry_buffer.initialize(device, hit_geometries.size() * sizeof(hit_geometries[0]),
            storage_usage, host_memory, memory, vk.buffers, error) ||
        !hit_geometry_buffer.write(0, hit_geometries.data(),
            hit_geometries.size() * sizeof(hit_geometries[0]), error))
        return false;
    std::array<BufferResource, FrameSlots> transform_buffers;
    const std::vector<RayTextureTransform> identity(hit_geometries.size());
    for (auto& buffer : transform_buffers)
        if (!buffer.initialize(device, identity.size() * sizeof(identity[0]), storage_usage,
                host_memory, memory, vk.buffers, error) ||
            !buffer.write(0, identity.data(), identity.size() * sizeof(identity[0]), error))
            return false;

    std::vector<VkAccelerationStructureInstanceKHR> instances(batch_count);
    for (size_t i = 0; i < batch_count; ++i)
    {
        auto& instance = instances[i];
        instance.transform = batches[batch_indices[i]].transform;
        instance.instanceCustomIndex = batch_indices[i];
        instance.mask = 0xff;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = batch_addresses[batch_indices[i]];
    }
    BufferResource instance_buffer;
    const VkBufferUsageFlags input_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    if (!instance_buffer.initialize(device, instances.size() * sizeof(instances[0]), input_usage,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            memory, vk.buffers, error) ||
        !instance_buffer.write(0, instances.data(), instances.size() * sizeof(instances[0]), error))
        return false;
    VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    address_info.buffer = instance_buffer.handle();
    const VkDeviceAddress instance_address = vk.get_buffer_address(device, &address_info);
    if (!instance_address)
    {
        error = "level ray instance buffer device address unavailable";
        return false;
    }
    VkAccelerationStructureGeometryKHR top_geometry{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    top_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    top_geometry.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    top_geometry.geometry.instances.data.deviceAddress = instance_address;
    auto tlas = std::make_unique<AccelerationStructure>();
    if (!tlas->initialize(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            top_geometry, static_cast<uint32_t>(batch_count),
            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                (allow_transform_updates ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0),
            properties.minAccelerationStructureScratchOffsetAlignment, memory, vk, error))
        return false;

    if (!submit_build([&](VkCommandBuffer command) {
            VkMemoryBarrier build_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            build_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
                VK_ACCESS_HOST_WRITE_BIT;
            build_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            upload.cmd_pipeline_barrier(command,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
                &build_barrier, 0, nullptr, 0, nullptr);
            if (!tlas->record_build(command, top_geometry,
                    static_cast<uint32_t>(batch_count), false, error))
                return false;
            VkMemoryBarrier ray_read_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            ray_read_barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
            ray_read_barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
            upload.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &ray_read_barrier, 0, nullptr, 0, nullptr);
            return true;
        }))
        return false;
    if (!allow_transform_updates) tlas->release_scratch();
    ray_blas_ = std::move(blas);
    ray_tlas_ = std::move(tlas);
    ray_instances_ = std::move(instance_buffer);
    hit_batches_ = std::move(hit_batch_buffer);
    hit_geometries_ = std::move(hit_geometry_buffer);
    texture_transforms_ = std::move(transform_buffers);
    hit_batch_count_ = hit_batches.size();
    hit_geometry_count_ = hit_geometries.size();
    instance_batch_indices_ = std::move(batch_indices);
    batch_blas_addresses_ = std::move(batch_addresses);
    if (allow_transform_updates)
    {
        instance_data_ = std::move(instances);
        batch_data_ = std::move(hit_batches);
        instance_address_ = instance_address;
        update_barrier_ = upload.cmd_pipeline_barrier;
        transform_updates_ = true;
    }
    error.clear();
    return true;
}

void LevelRayScene::destroy()
{
    ray_tlas_.reset();
    ray_blas_.clear();
    ray_instances_.destroy();
    hit_batches_.destroy();
    hit_geometries_.destroy();
    for (auto& buffer : texture_transforms_) buffer.destroy();
    hit_batch_count_ = 0;
    hit_geometry_count_ = 0;
    instance_batch_indices_.clear();
    batch_blas_addresses_.clear();
    instance_data_.clear();
    batch_data_.clear();
    instance_address_ = 0;
    update_barrier_ = nullptr;
    transform_updates_ = false;
}

bool LevelRayScene::record_transform_update(VkCommandBuffer command,
    const std::vector<VkTransformMatrixKHR>& transforms, std::string& error)
{
    if (!transform_updates_ || !command || transforms.size() != batch_data_.size() || !ray_tlas_ ||
        !instance_address_ || !update_barrier_)
    { error = "ray scene does not support this transform update"; return false; }
    for (const auto& transform : transforms)
        if (!valid_transform(transform))
        { error = "ray scene transform update is singular or non-finite"; return false; }
    auto updated_batches = batch_data_;
    auto updated_instances = instance_data_;
    for (size_t i = 0; i < transforms.size(); ++i) copy_transform(updated_batches[i], transforms[i]);
    for (size_t i = 0; i < updated_instances.size(); ++i)
        updated_instances[i].transform = transforms[instance_batch_indices_[i]];
    if (!ray_instances_.write(0, updated_instances.data(), updated_instances.size() * sizeof(updated_instances[0]), error) ||
        !hit_batches_.write(0, updated_batches.data(), updated_batches.size() * sizeof(updated_batches[0]), error))
        return false;
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    before.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
        VK_ACCESS_SHADER_READ_BIT;
    update_barrier_(command, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &before, 0, nullptr, 0, nullptr);
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.data.deviceAddress = instance_address_;
    if (!ray_tlas_->record_build(command, geometry, static_cast<uint32_t>(updated_instances.size()), true, error)) return false;
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    after.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    update_barrier_(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    batch_data_ = std::move(updated_batches);
    instance_data_ = std::move(updated_instances);
    error.clear();
    return true;
}

bool LevelRayScene::write_texture_transforms(uint32_t slot,
    const std::vector<RayTextureTransform>& transforms, std::string& error) const
{
    if (slot >= FrameSlots || transforms.size() != hit_geometry_count_ || transforms.empty())
    { error = "ray texture transform count or frame slot is invalid"; return false; }
    return texture_transforms_[slot].write(0, transforms.data(),
        transforms.size() * sizeof(transforms[0]), error);
}

bool FrameRayScenes::prepare(uint32_t slot, uint64_t topology_revision, VkCommandBuffer command,
    VkDevice device, VkQueue queue, VkCommandPool pool, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferUploadDispatch& upload, std::vector<PendingBufferUpload>& pending,
    const std::vector<RayGeometryBatch>& batches, const AccelerationStructureDispatch& dispatch,
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& properties, std::string& error, bool force_update)
{
    if (slot >= slots_.size() || !command || batches.empty())
    { error = "frame ray scene needs a valid slot, command and geometry"; return false; }
    for (const auto& batch : batches)
        if (!batch.geometries.empty() && !batch.external_blas)
        { error = "frame ray scene requires retained external BLAS inputs"; return false; }
    auto& frame = slots_[slot];
    std::vector<VkTransformMatrixKHR> transforms;
    transforms.reserve(batches.size());
    for (const auto& batch : batches) transforms.push_back(batch.transform);
    if (!frame.valid || frame.revision != topology_revision)
    {
        frame.valid = false;
        if (!frame.scene.build(device, queue, pool, memory, upload, pending, batches, dispatch, properties, error, true))
            return false;
        if (!frame.scene.handle())
        { error = "frame ray scene has no nonempty geometry"; return false; }
        frame.revision = topology_revision;
        frame.transforms = std::move(transforms);
        frame.valid = true;
        if (force_update && !frame.scene.record_transform_update(command, frame.transforms, error))
        { frame.valid = false; return false; }
        error.clear();
        return true;
    }
    if (transforms.size() != frame.transforms.size())
    { error = "frame ray scene topology changed without a revision"; return false; }
    bool changed = false;
    for (size_t i = 0; i < transforms.size(); ++i)
        changed |= std::memcmp(&transforms[i], &frame.transforms[i], sizeof(transforms[i])) != 0;
    if (changed || force_update)
    {
        if (!frame.scene.record_transform_update(command, transforms, error)) return false;
        frame.transforms = std::move(transforms);
    }
    error.clear();
    return true;
}

const LevelRayScene* FrameRayScenes::scene(uint32_t slot) const
{
    return slot < slots_.size() && slots_[slot].valid ? &slots_[slot].scene : nullptr;
}

void FrameRayScenes::discard(uint32_t index)
{
    if (index >= slots_.size()) return;
    auto& slot = slots_[index];
    slot.scene.destroy();
    slot.transforms.clear();
    slot.valid = false;
    slot.revision = 0;
}

void FrameRayScenes::destroy()
{
    for (uint32_t i = 0; i < slots_.size(); ++i) discard(i);
}
}
