#include "src/Layers/xrRenderVK/VulkanHardware.h"
#include "src/Layers/xrRenderVK/AccelerationStructure.h"
#include "src/Layers/xrRenderVK/LevelRayScene.h"
#include "src/Layers/xrRenderVK/RayQueryBatch.h"
#include "src/Layers/xrRenderVK/RaySurface.h"
#include "src/Layers/xrRenderVK/IndirectRayFrames.h"
#include "src/Layers/xrRenderVK/IndirectLocalShaders.h"
#include "src/Layers/xrRenderVK/RayGeneratePass.h"
#include "src/Layers/xrRenderVK/RayVisibilityPass.h"
#include "src/Layers/xrRenderVK/VsmPageAllocator.h"
#include "src/Layers/xrRenderVK/VsmDemandPass.h"
#include "src/Layers/xrRenderVK/VsmStaticCache.h"
#include "src/Layers/xrRenderVK/VsmFrameResources.h"
#include "src/Layers/xrRenderVK/VsmClipmap.h"
#include "src/Layers/xrRenderVK/VsmAtlas.h"
#include "src/Layers/xrRenderVK/VsmResolvePass.h"
#include "src/Layers/xrRenderVK/VsmCasterBinning.h"
#include "src/Layers/xrRenderVK/VsmInvalidation.h"
#include "src/Layers/xrRenderVK/VsmPageRender.h"
#include "src/Layers/xrRenderVK/BindlessTextureTable.h"
#include "src/Layers/xrRenderVK/GpuParticleCompute.h"
#include "src/Layers/xrRenderVK/GpuParticleUpload.h"
#include "src/Layers/xrRenderVK/GpuParticleUniforms.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <vector>

using namespace xray::render::vulkan;

static bool exercise_particle_pipelines(VkPhysicalDevice physical, VkDevice device, VkQueue queue,
    uint32_t family, const VkPhysicalDeviceLimits& limits, const char* directory)
{
    const char* names[]{"vk/gpu_particle_init.cs.spv", "vk/gpu_particle_simulate.cs.spv",
        "vk/gpu_particle_compact.cs.spv", "vk/gpu_particle_motion_capture.cs.spv"};
    std::array<std::vector<uint32_t>, 4> code;
    std::array<GpuParticleCompute::Shader, 4> shaders{};
    for (size_t i = 0; i < code.size(); ++i)
    {
        std::ifstream input(std::string(directory) + "/" + names[i], std::ios::binary | std::ios::ate);
        const auto size = input.tellg();
        if (!input || size <= 0 || size % 4 != 0) return false;
        code[i].resize(static_cast<size_t>(size) / 4);
        input.seekg(0); input.read(reinterpret_cast<char*>(code[i].data()), size);
        if (!input) return false;
        shaders[i] = {code[i].data(), static_cast<size_t>(size)};
    }
    GpuParticleCompute pipelines;
    std::string error;
    if (!pipelines.initialize(device, vkGetDeviceProcAddr, limits, 2, VK_NULL_HANDLE, shaders, error))
    { std::cerr << "GPU particle pipelines: " << error << '\n'; return false; }
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    BufferResourceDispatch vk{vkCreateBuffer, vkDestroyBuffer, vkGetBufferMemoryRequirements,
        vkAllocateMemory, vkFreeMemory, vkBindBufferMemory, vkMapMemory, vkUnmapMemory};
    BufferResource buffers[5];
    const VkDeviceSize sizes[]{32, 272 * 2, 4 * 8, 4 * 8, 4};
    const uint32_t constants[]{8, 2, 8, 2, 0, 0, 0, 0};
    std::vector<VkDescriptorBufferInfo> descriptors;
    for (size_t i = 0; i < 5; ++i)
    {
        if (!buffers[i].initialize(device, sizes[i], i == 0 ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error))
        { std::cerr << error << '\n'; return false; }
        std::vector<uint8_t> initial(static_cast<size_t>(sizes[i]), 0xff);
        if (i == 0) std::memcpy(initial.data(), constants, sizeof(constants));
        if (!buffers[i].write(0, initial.data(), initial.size(), error)) return false;
        descriptors.push_back({buffers[i].handle(), 0, sizes[i]});
    }
    VkCommandPool pool{};
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    if (vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS) return false;
    bool ok = false;
    do
    {
        VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool = pool; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount = 1;
        VkCommandBuffer command{};
        if (vkAllocateCommandBuffers(device, &allocate, &command) != VK_SUCCESS) break;
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS ||
            !pipelines.record(GpuParticleCompute::Pass::Initialize, command, 0, descriptors, 1, error)) break;
        VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &host_barrier, 0, nullptr, 0, nullptr);
        if (vkEndCommandBuffer(command) != VK_SUCCESS) break;
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
        if (vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) != VK_SUCCESS || vkQueueWaitIdle(queue) != VK_SUCCESS) break;
        ok = true;
        for (size_t i = 1; i < 5; ++i)
        {
            std::vector<uint8_t> result(static_cast<size_t>(sizes[i]));
            if (!buffers[i].read(0, result.data(), result.size(), error) ||
                std::any_of(result.begin(), result.end(), [](uint8_t value) { return value != 0; }))
            { ok = false; error = "GPU particle initialization produced unexpected data"; break; }
        }
    } while (false);
    vkDestroyCommandPool(device, pool, nullptr);
    if (ok)
    {
        // Submit one real Create command through the simulation kernel after
        // initialization, then inspect the status and allocation buffers.
        BufferResource extra[9];
        const VkDeviceSize extra_sizes[]{96 * 8, 16, 96, 560, 112, 8, 514 * 16, 48, 48};
        std::vector<VkDescriptorBufferInfo> sim(14);
        sim[0] = descriptors[0]; sim[2] = descriptors[1]; sim[9] = descriptors[2];
        sim[10] = descriptors[3]; sim[11] = descriptors[4];
        const uint32_t bindings[]{1, 3, 4, 5, 6, 7, 8, 12, 13};
        for (size_t i = 0; i < 9 && ok; ++i)
        {
            ok = extra[i].initialize(device, extra_sizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error);
            if (!ok) break;
            std::vector<uint8_t> initial(static_cast<size_t>(extra_sizes[i]), 0);
            ok = extra[i].write(0, initial.data(), initial.size(), error);
            sim[bindings[i]] = {extra[i].handle(), 0, extra_sizes[i]};
        }
        uint32_t sim_constants[]{8, 2, 8, 1, 1, 1, 0, 0};
        GpuParticleUniforms uniforms;
        if (ok) ok = uniforms.initialize(device, limits, memory, vk, 2, error);
        VkDescriptorBufferInfo ignored{};
        if (ok) ok = uniforms.append(sim_constants, sizeof(sim_constants), ignored, error) &&
            uniforms.append(sim_constants, sizeof(sim_constants), sim[0], error) &&
            sim[0].offset % std::max<VkDeviceSize>(limits.minUniformBufferOffsetAlignment, 1) == 0 &&
            !uniforms.append(sim_constants, sizeof(sim_constants), ignored, error);
        VkCommandPool sim_pool{};
        GpuParticleUpload uploader;
        if (ok) ok = uploader.initialize(device, vkGetDeviceProcAddr, 64, memory, vk, error);
        if (ok) ok = vkCreateCommandPool(device, &pool_info, nullptr, &sim_pool) == VK_SUCCESS;
        if (ok)
        {
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = sim_pool; allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; allocate.commandBufferCount = 1;
            VkCommandBuffer command{};
            ok = vkAllocateCommandBuffers(device, &allocate, &command) == VK_SUCCESS;
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            const uint32_t capacity = 2, header[]{0, 0, 1, 1}, range[]{0, 1};
            const std::vector<GpuParticleUpload::Update> updates{
                {&extra[2], 60, &capacity, sizeof(capacity)}, {&extra[4], 0, header, sizeof(header)},
                {&extra[5], 0, range, sizeof(range)}};
            if (ok) ok = vkBeginCommandBuffer(command, &begin) == VK_SUCCESS;
            if (ok) ok = !uploader.record(command, {{&extra[4], 0, header, sizeof(header)},
                {&extra[4], 4, &capacity, sizeof(capacity)}}, VK_QUEUE_TRANSFER_BIT, error);
            if (ok) ok = !uploader.record(command, {{&extra[4], 110, &capacity, sizeof(capacity)}},
                VK_QUEUE_TRANSFER_BIT, error);
            if (ok) ok = uploader.record(command, updates, VK_QUEUE_TRANSFER_BIT, error) &&
                pipelines.record(GpuParticleCompute::Pass::Simulate, command, 0, sim, 1, error);
            if (ok)
            {
                VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                    0, 1, &barrier, 0, nullptr, 0, nullptr);
                ok = vkEndCommandBuffer(command) == VK_SUCCESS;
                VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &command;
                if (ok) ok = vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS;
            }
            uint32_t status[12]{}, allocations[8]{}, emitter[68]{};
            if (ok) ok = extra[7].read(0, status, sizeof(status), error) &&
                buffers[2].read(0, allocations, sizeof(allocations), error) &&
                buffers[1].read(0, emitter, sizeof(emitter), error);
            if (ok && (status[0] != 1 || status[1] != 1 || emitter[31] != 1 ||
                allocations[0] != 2 || allocations[1] != 2))
            { ok = false; error = "GPU particle Create command produced unexpected state"; }
        }
        if (sim_pool) vkDestroyCommandPool(device, sim_pool, nullptr);
    }
    if (!ok) std::cerr << "GPU particle initialization: " << error << '\n';
    return ok;
}

static bool record_test_vsm_page(const FrameRecordingContext& frame, const float* matrix,
    const VsmPageJob& job, void*, std::string& error)
{
    if (!matrix || std::abs(matrix[0] - 2.f) > .001f ||
        std::abs(matrix[12] - (job.x == 0 ? 1.f : -1.f)) > .001f ||
        frame.viewport_offset.x != int32_t(job.slot * 8) || frame.extent.width != 8)
    { error = "VSM page projection or viewport mismatch"; return false; }
    if (job.slot == 0)
    {
        VkClearAttachment clear{}; clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        clear.clearValue.depthStencil = {.25f, 0};
        const VkClearRect rect{{frame.viewport_offset, frame.extent}, 0, 1};
        vkCmdClearAttachments(frame.command_buffer, 1, &clear, 1, &rect);
    }
    return true;
}

static bool exercise_bindless_table(VkPhysicalDevice physical, VkDevice device)
{
    BindlessTextureTable table;
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;
    std::string error;
    bool ok = false;
    do
    {
        if (!table.initialize(device, vkGetDeviceProcAddr, 1, error)) break;
        VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_info.extent = {1, 1, 1};
        image_info.mipLevels = image_info.arrayLayers = 1;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(device, &image_info, nullptr, &image) != VK_SUCCESS) break;
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, image, &requirements);
        VkPhysicalDeviceMemoryProperties memory{};
        vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        uint32_t type = 0;
        while (type < memory.memoryTypeCount && !(requirements.memoryTypeBits & (1u << type))) ++type;
        if (type == memory.memoryTypeCount) break;
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = type;
        if (vkAllocateMemory(device, &allocate, nullptr, &allocation) != VK_SUCCESS ||
            vkBindImageMemory(device, image, allocation, 0) != VK_SUCCESS) break;
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = image_info.format;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &view_info, nullptr, &view) != VK_SUCCESS) break;
        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = sampler_info.minFilter = VK_FILTER_NEAREST;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        if (vkCreateSampler(device, &sampler_info, nullptr, &sampler) != VK_SUCCESS) break;
        ok = table.write(0, view, sampler, error) && table.set() && table.layout() &&
            table.capacity() == 1 && !table.write(1, view, sampler, error);
    } while (false);
    if (!ok) std::cerr << "bindless table: " << error << '\n';
    table.destroy();
    if (sampler) vkDestroySampler(device, sampler, nullptr);
    if (view) vkDestroyImageView(device, view, nullptr);
    if (image) vkDestroyImage(device, image, nullptr);
    if (allocation) vkFreeMemory(device, allocation, nullptr);
    return ok;
}

static bool build_triangle_scene(VkPhysicalDevice physical, VkDevice device,
    VkQueue queue, uint32_t family, uint32_t api_version)
{
    auto vk = load_acceleration_structure_dispatch(device, vkGetDeviceProcAddr,
        api_version >= VK_API_VERSION_1_2);
    if (!vk.create || !vk.destroy || !vk.get_build_sizes || !vk.cmd_build ||
        !vk.get_address || !vk.get_buffer_address) return false;
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(physical, &memory);
    VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceDescriptorIndexingProperties indexing_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &as_properties;
    as_properties.pNext = &indexing_properties;
    vkGetPhysicalDeviceProperties2(physical, &properties);
    std::string error;
    BufferResource vertices;
    BufferResource indices;
    BufferResource instances;
    BufferResource query_rays;
    BufferResource query_hits;
    BufferResource bump_hits;
    BufferResource query_surfaces;
    BufferResource vsm_needed, vsm_table, vsm_pages, vsm_counters, vsm_parameters, vsm_cache_slots, vsm_cache_work;
    BufferResource vsm_atlas_readback;
    BufferResource vsm_resolve_table, vsm_resolve_parameters, vsm_resolve_readback;
    BufferResource vsm_casters, vsm_bin_parameters, vsm_pairs;
    BufferResource vsm_invalidation_parameters;
    BufferResource bump_surfaces;
    BufferResource indirect_surface_readback;
    BufferResource indirect_sun_readback;
    BufferResource indirect_radiance_readback;
    BufferResource indirect_image_readback;
    BufferResource local_test_surfaces, local_test_lights, local_test_rays, local_test_hits;
    BufferResource generated_hit;
    BufferResource large_rays;
    BufferResource large_hits;
    BufferResource generated_ray;
    BufferResource indirect_ray;
    BufferResource indirect_primary_surfaces;
    BufferResource background_ray;
    BufferResource visibility_readback;
    RayGeneratePass generate_pass;
    RayGeneratePass indirect_pass;
    RayGeneratePass background_pass;
    RayVisibilityPass visibility_pass;
    VkImage visibility_image{};
    VkDeviceMemory visibility_memory{};
    VkImageView visibility_view{};
    VkImage indirect_image{};
    VkDeviceMemory indirect_image_memory{};
    VkImageView indirect_image_view{};
    BufferResource alpha_upload;
    BindlessTextureTable alpha_textures;
    VkImage alpha_images[3]{};
    VkDeviceMemory alpha_memory[3]{};
    VkImageView alpha_views[3]{};
    VkSampler alpha_sampler = VK_NULL_HANDLE;
    AccelerationStructure blas;
    AccelerationStructure tlas;
    RayQueryBatch query_pass;
    RayQueryBatch bump_query_pass;
    RaySurfacePass surface_pass;
    VsmPageAllocator vsm_allocator;
    VsmDemandPass vsm_demand;
    VsmStaticCache vsm_static_cache;
    VsmAtlas vsm_atlas;
    VsmResolvePass vsm_resolve;
    VsmCasterBinning vsm_binning;
    VsmInvalidation vsm_invalidation;
    VkImage vsm_resolve_image{};
    VkDeviceMemory vsm_resolve_memory{};
    VkImageView vsm_resolve_view{};
    RaySurfacePass bump_surface_pass;
    RaySurfacePass indirect_surface_pass;
    IndirectRayFrames indirect_frames;
    StorageComputePass local_test_generator;
    RayQueryBatch local_test_query;
    FrameImagePool indirect_pool;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool ok = false;
    do
    {
        constexpr float triangle_data[] = {
            -1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
             1.f, -1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
             0.f,  1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f,
            -1.f, -1.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
             1.f, -1.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f,
             0.f,  1.f, 1.f, 0.f, 0.f, 1.f, 0.f, 0.f};
        struct TestVertex { float attributes[14]{}; uint32_t tangent{}, bitangent{}; };
        TestVertex triangle[6]{};
        for (size_t i = 0; i < 6; ++i)
        {
            std::copy_n(triangle_data + i * 8, 8, triangle[i].attributes);
            triangle[i].tangent = 0x0000007f; // +X
            triangle[i].bitangent = 0x00007f00; // +Y
        }
        constexpr uint32_t local_indices[] = {0, 1, 2, 0, 1, 2};
        const VkBufferUsageFlags input_usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
        if (!vertices.initialize(device, sizeof(triangle), input_usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, memory, vk.buffers, error) ||
            !vertices.write(0, triangle, sizeof(triangle), error) ||
            !indices.initialize(device, sizeof(local_indices), input_usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, memory, vk.buffers, error) ||
            !indices.write(0, local_indices, sizeof(local_indices), error)) break;
        VkBufferDeviceAddressInfo address_info{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        address_info.buffer = vertices.handle();
        const auto vertex_address = vk.get_buffer_address(device, &address_info);
        address_info.buffer = indices.handle();
        const auto index_address = vk.get_buffer_address(device, &address_info);
        if (!vertex_address || !index_address)
        { error = "triangle buffer address is zero"; break; }
        RayIndexedMesh unaligned{};
        unaligned.vertices = vertex_address + 1;
        unaligned.indices = index_address;
        unaligned.vertex_count = unaligned.index_count = 3;
        unaligned.vertex_stride = 8 * sizeof(float);
        VkAccelerationStructureGeometryKHR rejected{};
        uint32_t rejected_count = 0;
        if (make_ray_triangle_geometry(unaligned, rejected, rejected_count, error))
        { error = "unaligned vertex address was accepted"; break; }
        unaligned.vertices = vertex_address;
        unaligned.indices = index_address + 1;
        if (make_ray_triangle_geometry(unaligned, rejected, rejected_count, error))
        { error = "unaligned index address was accepted"; break; }
        error.clear();
        std::vector<VkAccelerationStructureGeometryKHR> blas_geometry(2);
        std::vector<uint32_t> blas_counts(2);
        bool geometry_ready = true;
        for (uint32_t mesh = 0; mesh < 2; ++mesh)
        {
            RayIndexedMesh input{};
            input.vertices = vertex_address;
            input.indices = index_address;
            input.first_vertex = mesh * 3;
            input.vertex_count = 3;
            input.first_index = mesh * 3;
            input.index_count = 3;
            input.vertex_stride = sizeof(TestVertex);
            input.opaque = false;
            if (!make_ray_triangle_geometry(input, blas_geometry[mesh], blas_counts[mesh], error))
            { geometry_ready = false; break; }
        }
        if (!geometry_ready) break;
        if (blas_geometry[1].geometry.triangles.vertexData.deviceAddress !=
                vertex_address + 3 * sizeof(TestVertex) ||
            blas_geometry[1].geometry.triangles.indexData.deviceAddress !=
                index_address + 3 * sizeof(uint32_t) ||
            blas_geometry[1].geometry.triangles.maxVertex != 2 ||
            blas_counts[1] != 1)
        { error = "packed mesh RT offsets are incorrect"; break; }
        if (!blas.initialize(device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                blas_geometry, blas_counts, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                as_properties.minAccelerationStructureScratchOffsetAlignment,
                memory, vk, error)) break;
        VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = family;
        if (vkCreateCommandPool(device, &pool_info, nullptr, &pool) != VK_SUCCESS)
        { error = "vkCreateCommandPool failed"; break; }
        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        VkCommandBuffer command = VK_NULL_HANDLE;
        if (vkAllocateCommandBuffers(device, &command_info, &command) != VK_SUCCESS)
        { error = "vkAllocateCommandBuffers failed"; break; }
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(device, &fence_info, nullptr, &fence) != VK_SUCCESS)
        { error = "vkCreateFence failed"; break; }
        auto submit = [&](const AccelerationStructure& as,
            const VkAccelerationStructureGeometryKHR& geometry, bool update)
        {
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            if (vkBeginCommandBuffer(command, &begin) != VK_SUCCESS ||
                !(as.handle() == blas.handle() ?
                    as.record_build(command, blas_geometry, blas_counts, update, error) :
                    as.record_build(command, geometry, 1, update, error)) ||
                vkEndCommandBuffer(command) != VK_SUCCESS)
                return false;
            VkSubmitInfo submission{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submission.commandBufferCount = 1;
            submission.pCommandBuffers = &command;
            if (vkQueueSubmit(queue, 1, &submission, fence) != VK_SUCCESS ||
                vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            { error = "acceleration structure GPU build failed"; return false; }
            return vkResetFences(device, 1, &fence) == VK_SUCCESS &&
                vkResetCommandBuffer(command, 0) == VK_SUCCESS;
        };
        BufferUploadDispatch blas_upload{};
        blas_upload.allocate_command_buffers = vkAllocateCommandBuffers;
        blas_upload.free_command_buffers = vkFreeCommandBuffers;
        blas_upload.begin_command_buffer = vkBeginCommandBuffer;
        blas_upload.end_command_buffer = vkEndCommandBuffer;
        blas_upload.cmd_pipeline_barrier = vkCmdPipelineBarrier;
        blas_upload.queue_submit = vkQueueSubmit;
        blas_upload.create_fence = vkCreateFence;
        blas_upload.destroy_fence = vkDestroyFence;
        blas_upload.wait_for_fences = vkWaitForFences;
        if (!build_ray_blas_sync(device, queue, pool, blas_upload, blas, blas_geometry, blas_counts, error)) break;
        // Exercise the update path on the same geometry before TLAS references it.
        if (!submit(blas, blas_geometry[0], true)) break;
        VkAccelerationStructureInstanceKHR instance_data{};
        instance_data.transform.matrix[0][0] = 1.f;
        instance_data.transform.matrix[1][1] = 1.f;
        instance_data.transform.matrix[2][2] = 1.f;
        instance_data.mask = 0xff;
        instance_data.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance_data.accelerationStructureReference = blas.address();
        if (!instances.initialize(device, sizeof(instance_data), input_usage,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, memory, vk.buffers, error) ||
            !instances.write(0, &instance_data, sizeof(instance_data), error)) break;
        address_info.buffer = instances.handle();
        const auto instance_address = vk.get_buffer_address(device, &address_info);
        if (!instance_address) { error = "instance buffer address is zero"; break; }
        VkAccelerationStructureGeometryKHR instance_geometry{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        instance_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
        instance_geometry.geometry.instances.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
        instance_geometry.geometry.instances.data.deviceAddress = instance_address;
        if (!tlas.initialize(device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                instance_geometry, 1, VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR,
                as_properties.minAccelerationStructureScratchOffsetAlignment,
                memory, vk, error) || !submit(tlas, instance_geometry, false) ||
            !submit(tlas, instance_geometry, true)) break;
        BufferUploadDispatch upload{};
        upload.buffer = vk.buffers;
        upload.allocate_command_buffers = vkAllocateCommandBuffers;
        upload.free_command_buffers = vkFreeCommandBuffers;
        upload.begin_command_buffer = vkBeginCommandBuffer;
        upload.end_command_buffer = vkEndCommandBuffer;
        upload.cmd_pipeline_barrier = vkCmdPipelineBarrier;
        upload.queue_submit = vkQueueSubmit;
        upload.create_fence = vkCreateFence;
        upload.destroy_fence = vkDestroyFence;
        upload.wait_for_fences = vkWaitForFences;
        std::vector<RayGeometryBatch> level_batches(3);
        level_batches[0].geometries = {blas_geometry[0]};
        level_batches[0].primitive_counts = {1};
        level_batches[0].material_ids = {11};
        level_batches[0].alpha_cutoffs = {0.25f};
        level_batches[0].texture_slots = {0};
        level_batches[0].diffuse_texture_slots = {0};
        level_batches[0].pbr_texture_slots = {UINT32_MAX};
        level_batches[0].metallic_texture_slots = {UINT32_MAX};
        level_batches[0].roughness_texture_slots = {UINT32_MAX};
        level_batches[0].ao_texture_slots = {UINT32_MAX};
        level_batches[0].normal_texture_slots = {UINT32_MAX};
        level_batches[0].normal_aux_texture_slots = {UINT32_MAX};
        level_batches[2].geometries = {blas_geometry[1]};
        level_batches[2].primitive_counts = {1};
        level_batches[2].material_ids = {22};
        level_batches[2].alpha_cutoffs = {0.5f};
        level_batches[2].texture_slots = {1};
        level_batches[2].diffuse_texture_slots = {1};
        level_batches[2].pbr_texture_slots = {0};
        level_batches[2].metallic_texture_slots = {UINT32_MAX};
        level_batches[2].roughness_texture_slots = {UINT32_MAX};
        level_batches[2].ao_texture_slots = {UINT32_MAX};
        level_batches[2].normal_texture_slots = {0};
        level_batches[2].normal_aux_texture_slots = {UINT32_MAX};
        std::vector<PendingBufferUpload> pending;
        LevelRayScene level_scene;
        if (!level_scene.build(device, queue, pool, memory, upload, pending, level_batches,
                vk, as_properties, error)) break;
        auto bump_batches = level_batches;
        bump_batches[2].external_blas = level_scene.batch_blas_address(2);
        if (!bump_batches[2].external_blas) { error = "source model BLAS address missing"; break; }
        bump_batches[2].transform.matrix[0][0] = 0.f;
        bump_batches[2].transform.matrix[0][1] = -1.f;
        bump_batches[2].transform.matrix[1][0] = 1.f;
        bump_batches[2].transform.matrix[1][1] = 0.f;
        bump_batches[2].transform.matrix[2][3] = .25f;
        bump_batches[2].normal_texture_slots = {1};
        bump_batches[2].normal_aux_texture_slots = {0};
        LevelRayScene bump_scene;
        if (!bump_scene.build(device, queue, pool, memory, upload, pending, bump_batches,
                vk, as_properties, error, true)) break;
        if (bump_scene.blas_count() != 1 ||
            bump_scene.batch_blas_address(2) != level_scene.batch_blas_address(2))
        { error = "external BLAS was not reused"; break; }
        RayHitBatchInfo hit_batches[3]{};
        RayHitGeometryInfo hit_geometries[2]{};
        const bool metadata_ok = tlas.address() != 0 && level_scene.address() != 0 &&
            level_scene.blas_count() == 2 && pending.empty() &&
            level_scene.instance_batch_indices() == std::vector<uint32_t>({0, 2}) &&
            level_scene.hit_batch_count() == 3 && level_scene.hit_geometry_count() == 2 &&
            level_scene.hit_batch_buffer().read(0, hit_batches, sizeof(hit_batches), error) &&
            level_scene.hit_geometry_buffer().read(0, hit_geometries, sizeof(hit_geometries), error) &&
            hit_batches[0].first_geometry == 0 && hit_batches[0].geometry_count == 1 &&
            hit_batches[1].first_geometry == 1 && hit_batches[1].geometry_count == 0 &&
            hit_batches[2].first_geometry == 1 && hit_batches[2].geometry_count == 1 &&
            hit_geometries[0].material_id == 11 && hit_geometries[1].material_id == 22 &&
            hit_geometries[0].alpha_cutoff == 0.25f &&
            hit_geometries[1].alpha_cutoff == 0.5f &&
            hit_geometries[0].texture_slot == 0 && hit_geometries[1].texture_slot == 1 &&
            hit_geometries[0].vertices == blas_geometry[0].geometry.triangles.vertexData.deviceAddress &&
            hit_geometries[1].indices == blas_geometry[1].geometry.triangles.indexData.deviceAddress;
        if (!metadata_ok) { error = "level ray hit metadata mismatch"; break; }
        const RayQueryInput ray_data[] = {
            {{0.f, 0.f, -1.f, .001f}, {0.f, 0.f, 1.f, 10.f}},
            {{0.f, 0.f, .25f, .001f}, {0.f, 0.f, 1.f, 10.f}},
            {{4.f, 4.f, -1.f, .001f}, {0.f, 0.f, 1.f, 10.f}},
            {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}}};
        const auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        const auto coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        FrameDispatch atlas_dispatch{};
        atlas_dispatch.get_format_properties = vkGetPhysicalDeviceFormatProperties;
        atlas_dispatch.create_image = vkCreateImage; atlas_dispatch.destroy_image = vkDestroyImage;
        atlas_dispatch.get_image_memory_requirements = vkGetImageMemoryRequirements;
        atlas_dispatch.allocate_memory = vkAllocateMemory; atlas_dispatch.free_memory = vkFreeMemory;
        atlas_dispatch.bind_image_memory = vkBindImageMemory; atlas_dispatch.create_image_view = vkCreateImageView;
        atlas_dispatch.destroy_image_view = vkDestroyImageView; atlas_dispatch.create_framebuffer = vkCreateFramebuffer;
        atlas_dispatch.destroy_framebuffer = vkDestroyFramebuffer; atlas_dispatch.create_render_pass = vkCreateRenderPass;
        atlas_dispatch.destroy_render_pass = vkDestroyRenderPass; atlas_dispatch.cmd_begin_render_pass = vkCmdBeginRenderPass;
        atlas_dispatch.cmd_end_render_pass = vkCmdEndRenderPass; atlas_dispatch.cmd_pipeline_barrier = vkCmdPipelineBarrier;
        atlas_dispatch.cmd_clear_attachments = vkCmdClearAttachments;
        if (!vsm_atlas.initialize(physical, device, vkGetDeviceProcAddr, VK_FORMAT_D32_SFLOAT, 8, 2, 1,
                properties.properties.limits, memory, atlas_dispatch, error, true) ||
            !vsm_atlas_readback.initialize(device, 16 * 8 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !vsm_resolve_table.initialize(device, 8 * sizeof(uint32_t), storage, coherent, memory, vk.buffers, error) ||
            !vsm_resolve_parameters.initialize(device, sizeof(VsmResolveParameters), storage, coherent, memory, vk.buffers, error) ||
            !vsm_resolve_readback.initialize(device, sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error)) break;
        const uint32_t requested_pages[]{1, 1, 0, 1, 0, 1, 1, 1};
        if (!vsm_needed.initialize(device, sizeof(requested_pages), storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !vsm_needed.write(0, requested_pages, sizeof(requested_pages), error) ||
            !vsm_table.initialize(device, sizeof(requested_pages), storage, coherent, memory, vk.buffers, error) ||
            !vsm_pages.initialize(device, 3 * 16, storage, coherent, memory, vk.buffers, error) ||
            !vsm_counters.initialize(device, 3 * 4, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !vsm_allocator.initialize(device, vkGetDeviceProcAddr, 1, properties.properties.limits.maxStorageBufferRange,
                VK_NULL_HANDLE, error) ||
            !vsm_parameters.initialize(device, sizeof(VsmDemandParameters), storage, coherent, memory, vk.buffers, error) ||
            !vsm_cache_slots.initialize(device, 8 * 16, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !vsm_cache_work.initialize(device, 16 + 8 * 16, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !vsm_casters.initialize(device, 3 * sizeof(VsmCasterBounds), storage, coherent, memory, vk.buffers, error) ||
            !vsm_bin_parameters.initialize(device, sizeof(VsmCasterBinningParameters), storage,
                coherent, memory, vk.buffers, error) ||
            !vsm_invalidation_parameters.initialize(device, sizeof(VsmInvalidationParameters), storage,
                coherent, memory, vk.buffers, error) ||
            !vsm_pairs.initialize(device, 16 + 3 * 8, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !vsm_static_cache.initialize(device, vkGetDeviceProcAddr, 1, properties.properties.limits.maxStorageBufferRange,
                VK_NULL_HANDLE, error) ||
            !vsm_binning.initialize(device, vkGetDeviceProcAddr, 1, properties.properties.limits.maxStorageBufferRange,
                VK_NULL_HANDLE, error) ||
            !vsm_invalidation.initialize(device, vkGetDeviceProcAddr, 1, properties.properties.limits.maxStorageBufferRange,
                VK_NULL_HANDLE, error)) break;
        const uint8_t alpha_pixels[16]{128, 128, 255, 0, 128, 128, 255, 255,
            128, 128, 255, 255, 255, 255, 255, 255};
        if (!alpha_upload.initialize(device, sizeof(alpha_pixels), VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                coherent, memory, vk.buffers, error) ||
            !alpha_upload.write(0, alpha_pixels, sizeof(alpha_pixels), error) ||
            !alpha_textures.initialize(device, vkGetDeviceProcAddr, 2, error)) break;
        VkImageCreateInfo alpha_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        alpha_info.imageType = VK_IMAGE_TYPE_2D;
        alpha_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        alpha_info.extent = {1, 1, 1};
        alpha_info.mipLevels = alpha_info.arrayLayers = 1;
        alpha_info.samples = VK_SAMPLE_COUNT_1_BIT;
        alpha_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        alpha_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        alpha_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        bool alpha_ready = true;
        for (uint32_t slot = 0; slot < 3; ++slot)
        {
            alpha_info.extent.width = slot == 0 ? 2 : 1;
            if (vkCreateImage(device, &alpha_info, nullptr, &alpha_images[slot]) != VK_SUCCESS)
            { alpha_ready = false; break; }
            VkMemoryRequirements requirements{};
            vkGetImageMemoryRequirements(device, alpha_images[slot], &requirements);
            uint32_t type = 0;
            while (type < memory.memoryTypeCount && !(requirements.memoryTypeBits & (1u << type))) ++type;
            if (type == memory.memoryTypeCount) { alpha_ready = false; break; }
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = type;
            if (vkAllocateMemory(device, &allocation, nullptr, &alpha_memory[slot]) != VK_SUCCESS ||
                vkBindImageMemory(device, alpha_images[slot], alpha_memory[slot], 0) != VK_SUCCESS)
            { alpha_ready = false; break; }
            VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view_info.image = alpha_images[slot];
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = alpha_info.format;
            view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            if (vkCreateImageView(device, &view_info, nullptr, &alpha_views[slot]) != VK_SUCCESS)
            { alpha_ready = false; break; }
        }
        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = sampler_info.minFilter = VK_FILTER_NEAREST;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        if (!alpha_ready || vkCreateSampler(device, &sampler_info, nullptr, &alpha_sampler) != VK_SUCCESS ||
            !alpha_textures.write(0, alpha_views[0], alpha_sampler, error) ||
            !alpha_textures.write(1, alpha_views[1], alpha_sampler, error))
        { if (error.empty()) error = "alpha texture setup failed"; break; }
        if (!vsm_demand.initialize(device, vkGetDeviceProcAddr, {1, 1}, {alpha_views[1]}, alpha_sampler,
                {&vsm_parameters}, {&vsm_needed}, VK_NULL_HANDLE, error)) break;
        VkImageCreateInfo visibility_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        visibility_info.imageType = VK_IMAGE_TYPE_2D;
        visibility_info.format = VK_FORMAT_R32_SFLOAT;
        visibility_info.extent = {4, 1, 1};
        visibility_info.mipLevels = visibility_info.arrayLayers = 1;
        visibility_info.samples = VK_SAMPLE_COUNT_1_BIT;
        visibility_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        visibility_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
        if (vkCreateImage(device, &visibility_info, nullptr, &visibility_image) != VK_SUCCESS)
        { error = "visibility image creation failed"; break; }
        VkMemoryRequirements visibility_requirements{};
        vkGetImageMemoryRequirements(device, visibility_image, &visibility_requirements);
        uint32_t visibility_type = 0;
        while (visibility_type < memory.memoryTypeCount &&
            !(visibility_requirements.memoryTypeBits & (1u << visibility_type))) ++visibility_type;
        if (visibility_type == memory.memoryTypeCount) { error = "visibility memory unavailable"; break; }
        VkMemoryAllocateInfo visibility_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        visibility_allocation.allocationSize = visibility_requirements.size;
        visibility_allocation.memoryTypeIndex = visibility_type;
        if (vkAllocateMemory(device, &visibility_allocation, nullptr, &visibility_memory) != VK_SUCCESS ||
            vkBindImageMemory(device, visibility_image, visibility_memory, 0) != VK_SUCCESS)
        { error = "visibility memory allocation failed"; break; }
        VkImageViewCreateInfo visibility_view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        visibility_view_info.image = visibility_image;
        visibility_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        visibility_view_info.format = visibility_info.format;
        visibility_view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (vkCreateImageView(device, &visibility_view_info, nullptr, &visibility_view) != VK_SUCCESS)
        { error = "visibility view creation failed"; break; }
        VkImageCreateInfo vsm_resolve_info = visibility_info;
        vsm_resolve_info.extent = {1, 1, 1};
        if (vkCreateImage(device, &vsm_resolve_info, nullptr, &vsm_resolve_image) != VK_SUCCESS)
        { error = "VSM resolve image creation failed"; break; }
        VkMemoryRequirements vsm_resolve_requirements{};
        vkGetImageMemoryRequirements(device, vsm_resolve_image, &vsm_resolve_requirements);
        uint32_t vsm_resolve_type = 0;
        while (vsm_resolve_type < memory.memoryTypeCount &&
            !(vsm_resolve_requirements.memoryTypeBits & (1u << vsm_resolve_type))) ++vsm_resolve_type;
        if (vsm_resolve_type == memory.memoryTypeCount) { error = "VSM resolve memory unavailable"; break; }
        VkMemoryAllocateInfo vsm_resolve_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        vsm_resolve_allocation.allocationSize = vsm_resolve_requirements.size;
        vsm_resolve_allocation.memoryTypeIndex = vsm_resolve_type;
        if (vkAllocateMemory(device, &vsm_resolve_allocation, nullptr, &vsm_resolve_memory) != VK_SUCCESS ||
            vkBindImageMemory(device, vsm_resolve_image, vsm_resolve_memory, 0) != VK_SUCCESS)
        { error = "VSM resolve memory allocation failed"; break; }
        VkImageViewCreateInfo vsm_resolve_view_info = visibility_view_info;
        vsm_resolve_view_info.image = vsm_resolve_image;
        if (vkCreateImageView(device, &vsm_resolve_view_info, nullptr, &vsm_resolve_view) != VK_SUCCESS ||
            !vsm_resolve.initialize(device, vkGetDeviceProcAddr, {1, 1}, {alpha_views[1]}, vsm_atlas.view(),
                vsm_atlas.view(), vsm_atlas.sampler(), {&vsm_resolve_table}, {&vsm_resolve_table},
                {&vsm_resolve_parameters}, {vsm_resolve_image}, {vsm_resolve_view}, VK_NULL_HANDLE, error))
        { if (error.empty()) error = "VSM resolve setup failed"; break; }
        VkImageCreateInfo indirect_info = visibility_info;
        indirect_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        indirect_info.extent = {1, 1, 1};
        if (vkCreateImage(device, &indirect_info, nullptr, &indirect_image) != VK_SUCCESS)
        { error = "indirect image creation failed"; break; }
        VkMemoryRequirements indirect_requirements{};
        vkGetImageMemoryRequirements(device, indirect_image, &indirect_requirements);
        uint32_t indirect_type = 0;
        while (indirect_type < memory.memoryTypeCount &&
            !(indirect_requirements.memoryTypeBits & (1u << indirect_type))) ++indirect_type;
        if (indirect_type == memory.memoryTypeCount) { error = "indirect image memory unavailable"; break; }
        VkMemoryAllocateInfo indirect_alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        indirect_alloc.allocationSize = indirect_requirements.size;
        indirect_alloc.memoryTypeIndex = indirect_type;
        if (vkAllocateMemory(device, &indirect_alloc, nullptr, &indirect_image_memory) != VK_SUCCESS ||
            vkBindImageMemory(device, indirect_image, indirect_image_memory, 0) != VK_SUCCESS)
        { error = "indirect image memory allocation failed"; break; }
        VkImageViewCreateInfo indirect_view_info = visibility_view_info;
        indirect_view_info.image = indirect_image;
        indirect_view_info.format = indirect_info.format;
        if (vkCreateImageView(device, &indirect_view_info, nullptr, &indirect_image_view) != VK_SUCCESS)
        { error = "indirect image view creation failed"; break; }
        if (!query_rays.initialize(device, sizeof(ray_data), storage, coherent, memory, vk.buffers, error) ||
            !query_rays.write(0, ray_data, sizeof(ray_data), error) ||
            !generated_ray.initialize(device, sizeof(RayQueryInput), storage, coherent,
                memory, vk.buffers, error) ||
            !indirect_ray.initialize(device, sizeof(RayQueryInput), storage, coherent,
                memory, vk.buffers, error) ||
            !background_ray.initialize(device, sizeof(RayQueryInput), storage, coherent,
                memory, vk.buffers, error) ||
            !generate_pass.initialize(device, vkGetDeviceProcAddr, {1, 1}, {alpha_views[1]},
                {alpha_views[1]}, alpha_sampler, {&generated_ray}, VK_NULL_HANDLE, error) ||
            !indirect_primary_surfaces.initialize(device, sizeof(uint32_t) * 2, storage,
                coherent, memory, vk.buffers, error) ||
            !indirect_pass.initialize(device, vkGetDeviceProcAddr, {1, 1}, {alpha_views[1]},
                {alpha_views[1]}, alpha_sampler, {&indirect_ray}, VK_NULL_HANDLE, error,
                RayGeneratePass::Mode::Indirect, {&indirect_primary_surfaces}) ||
            !background_pass.initialize(device, vkGetDeviceProcAddr, {1, 1}, {alpha_views[2]},
                {alpha_views[1]}, alpha_sampler, {&background_ray}, VK_NULL_HANDLE, error) ||
            !query_hits.initialize(device, sizeof(RayQueryHit) * 4, storage, coherent,
                memory, vk.buffers, error) ||
            !bump_hits.initialize(device, sizeof(RayQueryHit) * 4, storage, coherent,
                memory, vk.buffers, error) ||
            !query_surfaces.initialize(device, sizeof(RaySurface) * 4, storage, coherent,
                memory, vk.buffers, error) ||
            !bump_surfaces.initialize(device, sizeof(RaySurface) * 4, storage, coherent,
                memory, vk.buffers, error) ||
            !surface_pass.initialize(device, vkGetDeviceProcAddr, 2, properties.properties.limits,
                indexing_properties, alpha_textures, VK_NULL_HANDLE, error) ||
            !bump_surface_pass.initialize(device, vkGetDeviceProcAddr, 1, properties.properties.limits,
                indexing_properties, alpha_textures, VK_NULL_HANDLE, error) ||
            !indirect_surface_pass.initialize(device, vkGetDeviceProcAddr, 2, properties.properties.limits,
                indexing_properties, alpha_textures, VK_NULL_HANDLE, error) ||
            !indirect_frames.initialize(device, vkGetDeviceProcAddr, {1, 1}, {alpha_views[1]},
                {alpha_views[1]}, alpha_sampler, properties.properties, memory, vk.buffers,
                alpha_textures, VK_NULL_HANDLE, error) ||
            !indirect_frames.attach_output(device, vkGetDeviceProcAddr, {indirect_image},
                {indirect_image_view}, VK_NULL_HANDLE, error) ||
            !indirect_surface_readback.initialize(device, sizeof(RaySurface), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !indirect_sun_readback.initialize(device, sizeof(RayQueryInput), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !indirect_radiance_readback.initialize(device, sizeof(float) * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !indirect_image_readback.initialize(device, sizeof(uint16_t) * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !local_test_surfaces.initialize(device, sizeof(RaySurface) * 2, storage, coherent, memory, vk.buffers, error) ||
            !local_test_lights.initialize(device, sizeof(IndirectLocalLight), storage, coherent, memory, vk.buffers, error) ||
            !local_test_rays.initialize(device, sizeof(RayQueryInput) * 2, storage, coherent, memory, vk.buffers, error) ||
            !local_test_hits.initialize(device, sizeof(RayQueryHit) * 2, storage, coherent, memory, vk.buffers, error) ||
            !local_test_generator.initialize(device, vkGetDeviceProcAddr, 1, 3, 16,
                properties.properties.limits.maxStorageBufferRange, VK_NULL_HANDLE, indirect_local_shaders::Generate,
                sizeof(indirect_local_shaders::Generate), error) ||
            !local_test_query.initialize(device, vkGetDeviceProcAddr, 1, VK_NULL_HANDLE, alpha_textures, error) ||
            !visibility_readback.initialize(device, 4 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                coherent, memory, vk.buffers, error) ||
            !visibility_pass.initialize(device, vkGetDeviceProcAddr, {4, 1}, {visibility_image},
                {visibility_view}, {&query_hits}, VK_NULL_HANDLE, error) ||
            !generated_hit.initialize(device, sizeof(RayQueryHit), storage, coherent,
                memory, vk.buffers, error) ||
            !query_pass.initialize(device, vkGetDeviceProcAddr, 2, VK_NULL_HANDLE,
                alpha_textures, error) ||
            !bump_query_pass.initialize(device, vkGetDeviceProcAddr, 1, VK_NULL_HANDLE,
                alpha_textures, error)) break;
        const auto& ray_limits = properties.properties.limits;
        const uint64_t path_trace_sampled_descriptors = uint64_t(alpha_textures.capacity()) +
            std::min(8u, alpha_textures.capacity()) + 1u;
        const bool path_trace_limits_supported = ray_limits.maxPerStageDescriptorStorageBuffers >= 10 &&
            ray_limits.maxDescriptorSetStorageBuffers >= 10 && ray_limits.maxBoundDescriptorSets >= 3 &&
            path_trace_sampled_descriptors <= ray_limits.maxPerStageDescriptorSampledImages &&
            path_trace_sampled_descriptors <= ray_limits.maxDescriptorSetSampledImages &&
            path_trace_sampled_descriptors <= ray_limits.maxPerStageDescriptorSamplers &&
            path_trace_sampled_descriptors <= ray_limits.maxDescriptorSetSamplers &&
            sizeof(ReferencePathTraceHistoryCell) <= ray_limits.maxStorageBufferRange;
        if (path_trace_limits_supported && !indirect_frames.reference_path_trace_enabled())
        {
            error = "reference path trace pipeline was disabled despite sufficient descriptor limits";
            break;
        }
        std::cout << properties.properties.deviceName << ": reference path trace "
            << (indirect_frames.reference_path_trace_enabled() ? "enabled" : "unsupported by device limits")
            << '\n';
        VkCommandBufferBeginInfo query_begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if (vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS)
        { error = "ray query command begin failed"; break; }
        if (!vsm_allocator.record(command, 0, 2, 2, 3, vsm_needed, vsm_table, vsm_pages, vsm_counters, error)) break;
        VkImageMemoryBarrier alpha_barriers[3]{};
        for (uint32_t slot = 0; slot < 3; ++slot)
        {
            auto& barrier = alpha_barriers[slot];
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = alpha_images[slot];
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 3, alpha_barriers);
        for (uint32_t slot = 0; slot < 3; ++slot)
        {
            VkBufferImageCopy copy{};
            copy.bufferOffset = slot == 0 ? 0 : slot == 1 ? 8 : 12;
            copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            copy.imageExtent = {slot == 0 ? 2u : 1u, 1, 1};
            vkCmdCopyBufferToImage(command, alpha_upload.handle(), alpha_images[slot],
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            alpha_barriers[slot].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            alpha_barriers[slot].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            alpha_barriers[slot].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            alpha_barriers[slot].newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 3, alpha_barriers);
        const float inverse[] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        const float direction[] = {0, 0, 1, 10};
        const float indirect_direction[] = {0, 1, 1, 10};
        FrameRecordingContext generate_frame{command, VK_NULL_HANDLE, VK_NULL_HANDLE, {1, 1}, 0, 0};
        generate_frame.command_queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        generate_frame.graphics_queue_flags = generate_frame.command_queue_flags;
        generate_frame.compute_queue_flags = generate_frame.command_queue_flags;
        if (!generate_pass.record(generate_frame, inverse, direction) ||
            !indirect_pass.record(generate_frame, inverse, indirect_direction) ||
            !background_pass.record(generate_frame, inverse, direction))
        { error = "ray generation recording failed"; break; }
        if (!query_pass.record(command, 1, level_scene, generated_ray, generated_hit, 1, error))
        { if (error.empty()) error = "generated ray query recording failed"; break; }
        VkMemoryBarrier host_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_barrier, 0, nullptr, 0, nullptr);
        if (!query_pass.record(command, 0, level_scene, query_rays, query_hits, 4, error))
        { if (error.empty()) error = "ray query recording failed"; break; }
        if (!surface_pass.record(command, 0, level_scene, query_rays, query_hits, query_surfaces, 4, error))
        { if (error.empty()) error = "ray surface recording failed"; break; }
        if (!bump_query_pass.record(command, 0, bump_scene, query_rays, bump_hits, 4, error) ||
            !bump_surface_pass.record(command, 0, bump_scene, query_rays, bump_hits, bump_surfaces, 4, error))
        { if (error.empty()) error = "bump ray surface recording failed"; break; }
        const float secondary_sun[]{0.f, 0.f, -1.f, 10.f};
        const float sky_color[]{.2f, .2f, .2f};
        const float sun_color[]{1.f, 1.f, 1.f};
        RaySurface local_test_surface_data[2]{};
        local_test_surface_data[0].position_valid[2] = 2.f;
        local_test_surface_data[0].position_valid[3] = 1.f;
        local_test_surface_data[0].normal[2] = -1.f;
        local_test_surface_data[0].geometric_normal[2] = -1.f;
        local_test_surface_data[1].position_valid[0] = 4.f;
        local_test_surface_data[1].position_valid[2] = 2.f;
        local_test_surface_data[1].position_valid[3] = 1.f;
        local_test_surface_data[1].normal[0] = -1.f;
        local_test_surface_data[1].geometric_normal[0] = -1.f;
        IndirectLocalLight local_test_light{};
        local_test_light.position_range[2] = .5f;
        local_test_light.position_range[3] = 10.f;
        local_test_light.color_type[0] = local_test_light.color_type[1] = local_test_light.color_type[2] = 1.f;
        local_test_light.color_type[3] = 2.f;
        if (!local_test_surfaces.write(0, local_test_surface_data, sizeof(local_test_surface_data), error) ||
            !local_test_lights.write(0, &local_test_light, sizeof(local_test_light), error)) break;
        const uint32_t local_test_constants[]{2, 1, 1, 0};
        const std::vector<VkDescriptorBufferInfo> local_test_buffers{
            {local_test_surfaces.handle(), 0, local_test_surfaces.size()},
            {local_test_lights.handle(), 0, local_test_lights.size()},
            {local_test_rays.handle(), 0, local_test_rays.size()}};
        if (!local_test_generator.record(command, 0, local_test_buffers, local_test_constants, 1, 1, error) ||
            !local_test_query.record(command, 0, level_scene, local_test_rays, local_test_hits, 2, error)) break;
        IndirectLocalLight local_light{};
        local_light.position_range[2] = .5f;
        local_light.position_range[3] = 1.f;
        local_light.color_type[0] = local_light.color_type[1] = local_light.color_type[2] = 1.f;
        IndirectLocalLight distant_light = local_light;
        distant_light.position_range[0] = 100.f;
        if (!indirect_frames.record(generate_frame, level_scene, indirect_surface_pass, inverse, 10.f, 0,
                secondary_sun, sky_color, sun_color, {local_light, distant_light, distant_light}, error,
                {}, alpha_sampler, alpha_views[1]))
        { if (error.empty()) error = "indirect ray sequence recording failed"; break; }
        VkBufferMemoryBarrier indirect_readback_barriers[3]{};
        const VkBuffer indirect_readback_buffers[]{indirect_frames.surfaces(0).handle(),
            indirect_frames.sun_rays(0).handle(), indirect_frames.radiance(0).handle()};
        const VkDeviceSize indirect_readback_sizes[]{sizeof(RaySurface), sizeof(RayQueryInput), sizeof(float) * 4};
        for (uint32_t index = 0; index < 3; ++index)
        {
            auto& barrier = indirect_readback_barriers[index];
            barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.buffer = indirect_readback_buffers[index];
            barrier.offset = 0;
            barrier.size = indirect_readback_sizes[index];
        }
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 3, indirect_readback_barriers, 0, nullptr);
        VkBufferCopy surface_copy{0, 0, sizeof(RaySurface)};
        vkCmdCopyBuffer(command, indirect_frames.surfaces(0).handle(), indirect_surface_readback.handle(), 1, &surface_copy);
        VkBufferCopy sun_copy{0, 0, sizeof(RayQueryInput)};
        vkCmdCopyBuffer(command, indirect_frames.sun_rays(0).handle(), indirect_sun_readback.handle(), 1, &sun_copy);
        VkBufferCopy radiance_copy{0, 0, sizeof(float) * 4};
        vkCmdCopyBuffer(command, indirect_frames.radiance(0).handle(), indirect_radiance_readback.handle(), 1, &radiance_copy);
        VkImageMemoryBarrier indirect_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        indirect_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        indirect_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        indirect_copy.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        indirect_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        indirect_copy.srcQueueFamilyIndex = indirect_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        indirect_copy.image = indirect_image;
        indirect_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &indirect_copy);
        VkBufferImageCopy indirect_region{};
        indirect_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        indirect_region.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(command, indirect_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            indirect_image_readback.handle(), 1, &indirect_region);
        VkMemoryBarrier indirect_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        indirect_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        indirect_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &indirect_host, 0, nullptr, 0, nullptr);
        FrameRecordingContext visibility_frame{command, VK_NULL_HANDLE, VK_NULL_HANDLE, {4, 1}, 0, 0};
        if (!visibility_pass.record(visibility_frame, VK_IMAGE_LAYOUT_UNDEFINED))
        { error = "visibility recording failed"; break; }
        VkImageMemoryBarrier visibility_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        visibility_copy.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        visibility_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        visibility_copy.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        visibility_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        visibility_copy.srcQueueFamilyIndex = visibility_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        visibility_copy.image = visibility_image;
        visibility_copy.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &visibility_copy);
        VkBufferImageCopy visibility_region{};
        visibility_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        visibility_region.imageExtent = {4, 1, 1};
        vkCmdCopyImageToBuffer(command, visibility_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            visibility_readback.handle(), 1, &visibility_region);
        VkMemoryBarrier visibility_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        visibility_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        visibility_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &visibility_host, 0, nullptr, 0, nullptr);
        if (vkEndCommandBuffer(command) != VK_SUCCESS) { error = "command end failed"; break; }
        VkSubmitInfo query_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        query_submit.commandBufferCount = 1;
        query_submit.pCommandBuffers = &command;
        if (vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        { error = "ray query dispatch failed"; break; }
        RayQueryHit query_result[4]{};
        RaySurface surface_result[4]{};
        RaySurface indirect_surface{};
        RayQueryInput indirect_sun{};
        RayQueryHit local_test_result[2]{};
        if (!local_test_hits.read(0, local_test_result, sizeof(local_test_result), error)) break;
        if (local_test_result[0].found != 1 || local_test_result[1].found != 0)
        { error = "local indirect occlusion mismatch"; break; }
        float indirect_radiance[4]{};
        uint16_t indirect_pixel[4]{};
        if (!indirect_surface_readback.read(0, &indirect_surface, sizeof(indirect_surface), error)) break;
        if (!indirect_sun_readback.read(0, &indirect_sun, sizeof(indirect_sun), error)) break;
        if (!indirect_radiance_readback.read(0, indirect_radiance, sizeof(indirect_radiance), error)) break;
        if (!indirect_image_readback.read(0, indirect_pixel, sizeof(indirect_pixel), error)) break;
        if (indirect_pixel[0] < 0x3aa0 || indirect_pixel[0] > 0x3ae0 || indirect_pixel[3] != 0x3c00)
        {
            error = "indirect radiance image mismatch: red=" + std::to_string(indirect_pixel[0]) +
                " alpha=" + std::to_string(indirect_pixel[3]);
            break;
        }
        indirect_frames.commit_reservoir_history();
        bool indirect_output_in_transfer_layout = true;
        if (indirect_frames.reference_path_trace_enabled())
        {
            if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
                vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS)
            { error = "path-trace history command begin failed"; break; }
            indirect_copy.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            indirect_copy.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            indirect_copy.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            indirect_copy.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &indirect_copy);
            indirect_output_in_transfer_layout = false;
            indirect_frames.mark_output_recorded(0);
            if (!indirect_frames.record(generate_frame, level_scene, indirect_surface_pass, inverse, 10.f, 1,
                    secondary_sun, sky_color, sun_color, {local_light, distant_light, distant_light}, error,
                    {}, alpha_sampler, alpha_views[1], true, true, alpha_images[1]) ||
                !indirect_frames.reference_path_trace_history_reused())
            {
                if (error.empty()) error = "reference path trace did not reuse valid motion-reprojected history";
                break;
            }
            VkBufferMemoryBarrier radiance_readback_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            radiance_readback_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            radiance_readback_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            radiance_readback_barrier.srcQueueFamilyIndex = radiance_readback_barrier.dstQueueFamilyIndex =
                VK_QUEUE_FAMILY_IGNORED;
            radiance_readback_barrier.buffer = indirect_frames.radiance(0).handle();
            radiance_readback_barrier.offset = 0;
            radiance_readback_barrier.size = sizeof(float) * 4;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 1, &radiance_readback_barrier, 0, nullptr);
            vkCmdCopyBuffer(command, indirect_frames.radiance(0).handle(), indirect_radiance_readback.handle(),
                1, &radiance_copy);
            VkMemoryBarrier history_host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            history_host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            history_host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                0, 1, &history_host, 0, nullptr, 0, nullptr);
            if (vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
                vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            { error = "reference path trace history dispatch failed"; break; }
            float history_radiance[4]{};
            if (!indirect_radiance_readback.read(0, history_radiance, sizeof(history_radiance), error) ||
                !std::all_of(std::begin(history_radiance), std::end(history_radiance),
                    [](float value) { return std::isfinite(value) && value >= 0.f; }))
            { if (error.empty()) error = "reference path trace history produced invalid radiance"; break; }
            indirect_frames.commit_reservoir_history();
            std::cout << properties.properties.deviceName << ": reference path trace temporal history passed\n";
        }
        if (indirect_frames.projection_capacity())
        {
            indirect_frames.mark_output_recorded(0);
            IndirectLocalLight spot = local_light;
            spot.color_type[3] = 1.f; spot.direction_cone[2] = 1.f; spot.direction_cone[3] = .5f;
            spot.spot_right[0] = 1.f; spot.spot_right[3] = 1.f; spot.projection_slot = 0;
            float results[2]{};
            bool projection_ok = true;
            for (unsigned sample = 0; sample < 2 && projection_ok; ++sample)
            {
                projection_ok = vkResetFences(device, 1, &fence) == VK_SUCCESS &&
                    vkResetCommandBuffer(command, 0) == VK_SUCCESS && vkBeginCommandBuffer(command, &query_begin) == VK_SUCCESS;
                if (!projection_ok) break;
                if (sample == 0 && indirect_output_in_transfer_layout)
                {
                    indirect_copy.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; indirect_copy.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                    indirect_copy.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                    indirect_copy.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                    vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &indirect_copy);
                    indirect_output_in_transfer_layout = false;
                }
                projection_ok = indirect_frames.record(generate_frame, level_scene, indirect_surface_pass, inverse, 10.f, 0,
                    secondary_sun, sky_color, sun_color, {spot}, error, {alpha_views[sample + 1]},
                    alpha_sampler, alpha_views[1]);
                if (!projection_ok) break;
                vkCmdCopyBuffer(command, indirect_frames.radiance(0).handle(), indirect_radiance_readback.handle(), 1, &radiance_copy);
                VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
                host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
                vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                    0, 1, &host, 0, nullptr, 0, nullptr);
                projection_ok = vkEndCommandBuffer(command) == VK_SUCCESS &&
                    vkQueueSubmit(queue, 1, &query_submit, fence) == VK_SUCCESS &&
                    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
                float values[4]{};
                if (projection_ok) projection_ok = indirect_radiance_readback.read(0, values, sizeof(values), error);
                results[sample] = values[0];
                indirect_frames.mark_output_recorded(0);
            }
            if (!projection_ok || results[1] <= results[0] + .02f)
            { if (error.empty()) error = "indirect spotlight projection texture mismatch"; break; }
            std::cout << properties.properties.deviceName << ": indirect spotlight projection passed\n";
        }
        const FrameImagePoolDispatch indirect_pool_dispatch{vkCreateImage, vkDestroyImage,
            vkGetImageMemoryRequirements, vkAllocateMemory, vkFreeMemory, vkBindImageMemory};
        FrameDispatch indirect_frame_dispatch{};
        indirect_frame_dispatch.get_format_properties = vkGetPhysicalDeviceFormatProperties;
        indirect_frame_dispatch.create_image_view = vkCreateImageView;
        indirect_frame_dispatch.destroy_image_view = vkDestroyImageView;
        if (!indirect_pool.initialize(device, memory, indirect_pool_dispatch) ||
            !indirect_frames.prepare_output(physical, device, vkGetDeviceProcAddr, indirect_frame_dispatch,
                indirect_pool, VK_NULL_HANDLE, error) || !indirect_frames.output_view(0))
        { if (error.empty()) error = "indirect pooled output creation failed"; break; }
        {
            VsmFrameResources resources;
            if (!resources.initialize(physical, device, vkGetDeviceProcAddr, {1, 1},
                    {alpha_views[1], alpha_views[1]}, alpha_sampler, VK_FORMAT_D32_SFLOAT, 2, 2, 8, 2, 1,
                    properties.properties, memory, vk.buffers, atlas_dispatch, indirect_pool, VK_NULL_HANDLE, error) ||
                resources.count() != 2 || !resources.image(0).output_view || !resources.image(1).output_view ||
                resources.static_atlas.page_count() != 8 || resources.dynamic_atlas.page_count() != 2 ||
                !resources.image(0).static_work.host_visible() || !resources.image(1).dynamic_pages.host_visible())
            { if (error.empty()) error = "VSM frame resource creation failed"; break; }
        }
        const float material = 128.f / 255.f;
        const float f0 = .04f * (1.f - material) + material * material;
        const float env_r0 = 1.f - material;
        const float env_r1 = .0425f - .0275f * material;
        const float env_r2 = 1.04f - .572f * material;
        const float env_r3 = -.04f + .022f * material;
        const float env_a004 = std::min(env_r0 * env_r0, std::exp2(-9.28f)) * env_r0 + env_r1;
        const float environment_brdf = f0 * (-1.04f * env_a004 + env_r2) + 1.04f * env_a004 + env_r3;
        const float expected_pbr = .2f * environment_brdf + (material * (1.f - material) * .2f +
            1.75f * ((1.f - f0) * (1.f - material) * material + f0 / (4.f * std::pow(material, 4.f)))) /
            3.14159265f;
        // The seeded RIS sample includes the single in-range local light; the
        // other two test lights are outside its range.
        const float expected_local_pbr = .094f;
        if (std::abs(indirect_radiance[0] - (expected_pbr + expected_local_pbr)) > .015f)
        {
            error = "indirect radiance mismatch: actual=" + std::to_string(indirect_radiance[0]) +
                " expected=" + std::to_string(expected_pbr + expected_local_pbr);
            break;
        }
        if (std::abs(indirect_sun.direction_max[2] + 1.f) > .001f ||
            std::abs(indirect_sun.origin_min[2] - .99f) > .001f ||
            std::abs(indirect_sun.origin_min[0] - indirect_surface.position_valid[0]) > 1e-6f ||
            std::abs(indirect_sun.origin_min[1] - indirect_surface.position_valid[1]) > 1e-6f ||
            indirect_sun.direction_max[3] != 10.f)
        { error = "secondary sun ray mismatch"; break; }
        if (indirect_surface.position_valid[3] != 1.f || indirect_surface.material_id != 22 ||
            std::abs(indirect_surface.position_valid[2] - 1.f) > .001f ||
            std::abs(indirect_surface.albedo[0] - 128.f / 255.f) > .001f)
        { error = "indirect ray sequence surface mismatch"; break; }
        if (!query_surfaces.read(0, surface_result, sizeof(surface_result), error)) break;
        uint32_t page_table[8]{}, page_list[3][4]{}, page_counts[3]{};
        if (!vsm_table.read(0, page_table, sizeof(page_table), error) ||
            !vsm_pages.read(0, page_list, sizeof(page_list), error) ||
            !vsm_counters.read(0, page_counts, sizeof(page_counts), error)) break;
        if (page_counts[0] != 6 || page_counts[1] + page_counts[2] != 3)
        { error = "VSM allocation counters mismatch"; break; }
        bool seen[3]{};
        for (uint32_t i = 0; i < 8; ++i)
        {
            const auto slot = page_table[i];
            if (!requested_pages[i] && slot != UINT32_MAX)
            { error = "VSM allocated an unrequested page"; break; }
            if (slot == UINT32_MAX) continue;
            if (slot >= 3 || seen[slot] || page_list[slot][0] != i / 4 ||
                page_list[slot][1] != i % 2 || page_list[slot][2] != (i % 4) / 2)
            { error = "VSM page mapping mismatch"; break; }
            seen[slot] = true;
        }
        if (!error.empty() || !seen[0] || !seen[1] || !seen[2])
        { if (error.empty()) error = "VSM physical allocation incomplete"; break; }
        std::vector<VsmPageJob> overflow_jobs;
        if (!read_vsm_dynamic_jobs(vsm_counters, vsm_pages, 2, 2, 3, overflow_jobs, error) ||
            overflow_jobs.size() != 3)
        { if (error.empty()) error = "VSM overflow did not produce three renderable jobs"; break; }
        const uint32_t inconsistent_counts[]{6, 0, 0};
        std::string readback_error;
        if (!vsm_counters.write(0, inconsistent_counts, sizeof(inconsistent_counts), error) ||
            read_vsm_dynamic_jobs(vsm_counters, vsm_pages, 2, 2, 3, overflow_jobs, readback_error) ||
            !overflow_jobs.empty() || readback_error != "inconsistent VSM dynamic page counters" ||
            !vsm_counters.write(0, page_counts, sizeof(page_counts), error))
        { if (error.empty()) error = "VSM readback accepted inconsistent overflow counters"; break; }
        RaySurface bump_result[4]{};
        if (!bump_surfaces.read(0, bump_result, sizeof(bump_result), error)) break;
        const float bump_component = 1.f / std::sqrt(3.f);
        if (std::abs(bump_result[0].normal[0] + bump_component) > .001f ||
            std::abs(bump_result[0].normal[1] - bump_component) > .001f ||
            std::abs(bump_result[0].normal[2] + bump_component) > .001f ||
            std::abs(bump_result[0].position_valid[2] - 1.25f) > .001f)
        { error = "bump ray surface normal mismatch"; break; }
        if (surface_result[0].position_valid[3] != 1.f ||
            std::abs(surface_result[0].position_valid[2] - 1.f) > .001f ||
            std::abs(surface_result[0].normal[2] + 1.f) > .001f ||
            std::abs(surface_result[0].normal[0] - 1.f / 255.f) > .001f ||
            std::abs(surface_result[0].normal[1] - 1.f / 255.f) > .001f ||
            std::abs(surface_result[0].geometric_normal[2] + 1.f) > .001f ||
            surface_result[0].material_id != 22 || surface_result[0].texture_slot != 1 ||
            std::abs(surface_result[0].albedo[0] - 128.f / 255.f) > .001f ||
            std::abs(surface_result[0].pbr[0] - 128.f / 255.f) > .001f ||
            std::abs(surface_result[0].pbr[1] - 128.f / 255.f) > .001f ||
            surface_result[0].pbr[2] != 1.f || surface_result[0].pbr[3] != 1.f ||
            surface_result[2].position_valid[3] != 0.f ||
            surface_result[3].position_valid[3] != 0.f)
        { error = "ray surface reconstruction mismatch"; break; }
        float visibility_values[4]{};
        if (!visibility_readback.read(0, visibility_values, sizeof(visibility_values), error)) break;
        if (visibility_values[0] != 0.f || visibility_values[1] != 0.f ||
            visibility_values[2] != 1.f || visibility_values[3] != 1.f)
        { error = "ray visibility hit/miss result mismatch"; break; }
        RayQueryHit generated_result{};
        RayQueryInput generated{};
        RayQueryInput indirect{};
        RayQueryInput background{};
        if (!generated_ray.read(0, &generated, sizeof(generated), error)) break;
        if (!indirect_ray.read(0, &indirect, sizeof(indirect), error)) break;
        const float indirect_length = indirect.direction_max[0] * indirect.direction_max[0] +
            indirect.direction_max[1] * indirect.direction_max[1] +
            indirect.direction_max[2] * indirect.direction_max[2];
        if (std::abs(indirect_length - 1.f) > .001f || indirect.direction_max[2] <= 0.f ||
            indirect.direction_max[3] != 10.f)
        { error = "indirect ray generation mismatch"; break; }
        if (!background_ray.read(0, &background, sizeof(background), error)) break;
        if (!generated_hit.read(0, &generated_result, sizeof(generated_result), error)) break;
        if (!generated_result.found || generated_result.material_id != 22 ||
            std::abs(generated_result.distance - (1.f - 128.f / 255.f - .01f)) > .001f)
        { error = "generated ray did not hit the expected triangle"; break; }
        if (background.direction_max[3] != 0.f || background.origin_min[3] != 0.f)
        { error = "background pixel emitted an active ray"; break; }
        if (std::abs(generated.origin_min[0]) > .001f ||
            std::abs(generated.origin_min[1]) > .001f ||
            std::abs(generated.origin_min[2] - (128.f / 255.f + .01f)) > .001f ||
            std::abs(generated.direction_max[2] - 1.f) > .001f ||
            generated.direction_max[3] != 10.f)
        { error = "G-buffer ray generation mismatch"; break; }
        if (!query_hits.read(0, query_result, sizeof(query_result), error)) break;
        const bool query_ok = query_result[0].found == 1 && query_result[0].instance == 2 &&
            query_result[0].geometry == 0 && query_result[0].primitive == 0 &&
            query_result[0].material_id == 22 &&
            query_result[0].barycentric[0] >= 0.f && query_result[0].barycentric[1] >= 0.f &&
            query_result[0].barycentric[0] + query_result[0].barycentric[1] <= 1.001f &&
            std::abs(query_result[0].distance - 2.f) < .001f &&
            query_result[1].found == 1 && query_result[1].instance == 2 &&
            query_result[1].geometry == 0 && query_result[1].primitive == 0 &&
            query_result[1].material_id == 22 &&
            query_result[1].barycentric[0] >= 0.f && query_result[1].barycentric[1] >= 0.f &&
            query_result[1].barycentric[0] + query_result[1].barycentric[1] <= 1.001f &&
            std::abs(query_result[1].distance - .75f) < .001f &&
            query_result[2].found == 0 && query_result[2].instance == UINT32_MAX &&
            query_result[2].geometry == UINT32_MAX && query_result[2].primitive == UINT32_MAX &&
            query_result[2].material_id == UINT32_MAX &&
            query_result[3].found == 0 && query_result[3].distance == -1.f &&
            query_result[3].instance == UINT32_MAX && query_result[3].material_id == UINT32_MAX;
        if (!query_ok) { error = "ray query hit or miss result mismatch"; break; }
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_atlas.initialize_layout(command, error) || vkEndCommandBuffer(command) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        { if (error.empty()) error = "VSM atlas initialization failed"; break; }
        vsm_atlas.mark_initialized();
        VsmPageProjection page_projection{};
        std::copy_n(inverse, 16, page_projection.light_view);
        page_projection.level[0][0] = page_projection.level[0][1] = -1.f;
        page_projection.level[0][2] = 2.f;
        page_projection.axis = 2; page_projection.levels = 1;
        page_projection.near_depth = 0.f; page_projection.far_depth = 1.f;
        VsmClipmap clipmap{}, moved_clipmap{};
        const float direction_a[3]{.25f, -.95f, .15f};
        const float direction_b[3]{.25000001f, -.95f, .15f};
        const float zero_direction[3]{};
        float stable_a[3]{}, stable_b[3]{};
        if (!stabilize_vsm_direction(direction_a, stable_a) || !stabilize_vsm_direction(direction_b, stable_b) ||
            stabilize_vsm_direction(zero_direction, stable_b) || std::memcmp(stable_a, stable_b, sizeof(stable_a)) != 0)
        { error = "VSM light direction stabilization mismatch"; break; }
        const float camera[3]{.1f, -.1f, 0.f}, moved_camera[3]{.9f, -.1f, 0.f};
        if (!build_vsm_clipmap(inverse, inverse, camera, 2, 2, 2.f, 0.f, 1.f, clipmap, error) ||
            !build_vsm_clipmap(inverse, inverse, moved_camera, 2, 2, 2.f, 0.f, 1.f, moved_clipmap, error) ||
            clipmap.cache.base[0][0] != -1 || clipmap.cache.base[0][1] != -2 ||
            moved_clipmap.cache.base[0][0] != -1 || clipmap.demand.level[0][0] != -1.f ||
            clipmap.resolve.level[0][1] != -2.f || clipmap.projection.level[1][2] != 4.f)
        { if (error.empty()) error = "VSM clipmap stabilization mismatch"; break; }
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !record_vsm_pages(vsm_atlas, generate_frame, page_projection,
                {{0, 0, 0, 0}, {1, 0, 1, 0}}, record_test_vsm_page, nullptr, error))
        { if (error.empty()) error = "VSM page rendering failed"; break; }
        VkImageMemoryBarrier atlas_copy{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        atlas_copy.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        atlas_copy.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        atlas_copy.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        atlas_copy.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        atlas_copy.srcQueueFamilyIndex = atlas_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        atlas_copy.image = vsm_atlas.image(); atlas_copy.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &atlas_copy);
        VkBufferImageCopy atlas_region{};
        atlas_region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        atlas_region.imageExtent = {16, 8, 1};
        vkCmdCopyImageToBuffer(command, vsm_atlas.image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vsm_atlas_readback.handle(), 1, &atlas_region);
        atlas_copy.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; atlas_copy.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        atlas_copy.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        atlas_copy.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, nullptr, 0, nullptr, 1, &atlas_copy);
        if (vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        { error = "VSM atlas page submission failed"; break; }
        float atlas_pixels[16 * 8]{};
        if (!vsm_atlas_readback.read(0, atlas_pixels, sizeof(atlas_pixels), error)) break;
        if (std::abs(atlas_pixels[0] - .25f) > .001f || std::abs(atlas_pixels[8] - 1.f) > .001f)
        { error = "VSM atlas page clear damaged a neighboring page"; break; }
        VsmResolveParameters resolve_params{};
        std::copy_n(inverse, 16, resolve_params.inverse_view_projection);
        std::copy_n(inverse, 16, resolve_params.light_view);
        resolve_params.level[0][0] = resolve_params.level[0][1] = -1.f; resolve_params.level[0][2] = 2.f;
        resolve_params.level[1][0] = resolve_params.level[1][1] = -2.f; resolve_params.level[1][2] = 4.f;
        resolve_params.axis = 2; resolve_params.levels = 2; resolve_params.page_size = 8;
        resolve_params.static_columns = 2; resolve_params.static_slots = 2; resolve_params.inverse_depth_range = 1.f;
        bool resolve_ok = true;
        for (uint32_t scenario = 0; scenario < 9 && resolve_ok; ++scenario)
        {
            uint32_t mapping[8]; std::fill_n(mapping, 8, UINT32_MAX);
            if (scenario < 2) mapping[scenario == 0 ? 3 : 7] = 0;
            if (scenario >= 3 && scenario <= 5) mapping[3] = scenario == 3 ? 0 : 1;
            if (scenario == 6 || scenario == 7) { mapping[3] = 2; mapping[7] = 0; }
            if (scenario == 8) { mapping[0] = mapping[1] = mapping[2] = 1; mapping[3] = 0; }
            const bool dynamic = (scenario >= 3 && scenario <= 5) || scenario == 7;
            const uint32_t declared_slots = scenario >= 6 ? 3 : 2;
            resolve_params.static_slots = dynamic ? 0 : declared_slots;
            resolve_params.dynamic_enabled = dynamic ? 1 : 0;
            resolve_params.dynamic_columns = 2; resolve_params.dynamic_slots = declared_slots;
            BufferResource& resolve_mapping = dynamic ? vsm_table : vsm_resolve_table;
            if (!resolve_mapping.write(0, mapping, sizeof(mapping), error) ||
                vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
                vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
                !vsm_resolve.record(generate_frame, resolve_params, scenario == 0 ? VK_IMAGE_LAYOUT_UNDEFINED :
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, error, UINT32_MAX, &resolve_mapping, &resolve_mapping,
                    scenario == 5 ? alpha_views[1] : VK_NULL_HANDLE, scenario == 5 ? alpha_sampler : VK_NULL_HANDLE))
            { resolve_ok = false; break; }
            VkImageMemoryBarrier read{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            read.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; read.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            read.srcQueueFamilyIndex = read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            read.image = vsm_resolve_image; read.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &read);
            VkBufferImageCopy result_copy{};
            result_copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; result_copy.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(command, vsm_resolve_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vsm_resolve_readback.handle(), 1, &result_copy);
            read.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT; read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL; read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, nullptr, 0, nullptr, 1, &read);
            float visible{};
            if (vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
                vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
                !vsm_resolve_readback.read(0, &visible, sizeof(visible), error) ||
                std::abs(visible - (scenario == 8 ? 5.f / 9.f : scenario == 5 ? 128.f / 255.f :
                    scenario == 2 || scenario == 4 ? 1.f : 0.f)) > .001f)
            { resolve_ok = false; break; }
        }
        if (!resolve_ok) { if (error.empty()) error = "VSM visibility or parent-page fallback mismatch"; break; }
        VsmDemandParameters demand{};
        std::copy_n(inverse, 16, demand.inverse_view_projection);
        std::copy_n(inverse, 16, demand.light_view);
        demand.level[0][0] = demand.level[0][1] = -1.f; demand.level[0][2] = 2.f;
        demand.level[1][0] = demand.level[1][1] = -2.f; demand.level[1][2] = 4.f;
        demand.axis = 2; demand.levels = 2; demand.step = 1;
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_demand.record(generate_frame, demand, error) ||
            !vsm_allocator.record(command, 0, 2, 2, 3, vsm_needed, vsm_table, vsm_pages, vsm_counters, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_needed.read(0, page_table, sizeof(page_table), error) ||
            !vsm_counters.read(0, page_counts, sizeof(page_counts), error))
        { if (error.empty()) error = "VSM demand and allocation failed"; break; }
        for (uint32_t i = 0; i < 8; ++i)
            if (page_table[i] != ((i == 3 || i == 7) ? 1u : 0u))
            { error = "VSM demand marking mismatch"; break; }
        if (!error.empty() || page_counts[0] != 2 || page_counts[1] != 1 || page_counts[2] != 1)
        { if (error.empty()) error = "VSM demand allocation counters mismatch"; break; }
        std::vector<VsmPageJob> decoded_jobs;
        if (!read_vsm_dynamic_jobs(vsm_counters, vsm_pages, 2, 2, 3, decoded_jobs, error) ||
            decoded_jobs.size() != 2 || decoded_jobs[0].slot != 0 || decoded_jobs[1].slot != 1 ||
            decoded_jobs[0].x != 1 || decoded_jobs[0].y != 1 || decoded_jobs[1].x != 1 ||
            decoded_jobs[1].y != 1 || decoded_jobs[0].level == decoded_jobs[1].level)
        { if (error.empty()) error = "VSM dynamic job readback mismatch"; break; }
        demand.lod_bias = 1;
        demand.page_size = 8;
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_demand.record(generate_frame, demand, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_needed.read(0, page_table, sizeof(page_table), error))
        { if (error.empty()) error = "VSM filter demand failed"; break; }
        for (uint32_t i = 0; i < 8; ++i)
            if (page_table[i] != (i >= 4 ? 1u : 0u))
            { error = "VSM filter demand omitted a neighbor or escaped its clipmap level"; break; }
        if (!error.empty()) break;
        demand.page_size = 0;
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_demand.record(generate_frame, demand, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_needed.read(0, page_table, sizeof(page_table), error))
        { if (error.empty()) error = "VSM biased demand failed"; break; }
        for (uint32_t i = 0; i < 8; ++i)
            if (page_table[i] != (i == 7 ? 1u : 0u))
            { error = "VSM biased demand or per-frame clear mismatch"; break; }
        if (!error.empty()) break;
        VsmStaticCacheParameters cache_params{};
        cache_params.axis = 2; cache_params.levels = 2; cache_params.refresh_interval = 20; cache_params.budget = 8;
        uint32_t cache_work[36]{};
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_static_cache.reset(command, vsm_cache_slots, error) ||
            !vsm_static_cache.prepare(command, 0, cache_params, vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_table.read(0, page_table, sizeof(page_table), error) ||
            !vsm_cache_work.read(0, cache_work, sizeof(cache_work), error))
        { if (error.empty()) error = "VSM static cache prepare failed"; break; }
        if (page_table[7] != UINT32_MAX || cache_work[0] != 1 || cache_work[4] != 7 || cache_work[5] != 7)
        { error = "VSM static page was visible before publication"; break; }
        if (!read_vsm_static_jobs(vsm_cache_work, 2, 2, 8, 8, decoded_jobs, error) ||
            decoded_jobs.size() != 1 || decoded_jobs[0].slot != 7 || decoded_jobs[0].level != 1 ||
            decoded_jobs[0].x != 1 || decoded_jobs[0].y != 1)
        { if (error.empty()) error = "VSM static job readback mismatch"; break; }
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_static_cache.publish(command, 0, cache_params, vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_table.read(0, page_table, sizeof(page_table), error))
        { if (error.empty()) error = "VSM static page publication failed"; break; }
        if (page_table[7] != 7) { error = "VSM static page publication mismatch"; break; }
        cache_params.frame = 1; cache_params.force_dirty = 1; cache_params.budget = 0;
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_static_cache.prepare(command, 0, cache_params, vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_table.read(0, page_table, sizeof(page_table), error) ||
            !vsm_cache_work.read(0, cache_work, sizeof(cache_work), error))
        { if (error.empty()) error = "VSM forced invalidation failed"; break; }
        if (page_table[7] != UINT32_MAX || cache_work[0] != 1)
        { error = "VSM forced invalidation exposed stale page"; break; }
        cache_params.frame = 1;
        cache_params.force_dirty = 0; cache_params.budget = 8;
        cache_params.base[1][0] = 1;
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_static_cache.prepare(command, 0, cache_params, vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_table.read(0, page_table, sizeof(page_table), error) ||
            !vsm_cache_work.read(0, cache_work, sizeof(cache_work), error))
        { if (error.empty()) error = "VSM static cache movement failed"; break; }
        if (page_table[7] != UINT32_MAX || cache_work[0] != 1 || cache_work[4] != 6 || cache_work[5] != 7)
        { error = "VSM toroidal movement failed to invalidate stale page"; break; }
        VsmCasterBounds casters[3]{};
        casters[0].minimum[0] = casters[0].minimum[1] = .5f; casters[0].minimum[2] = -.5f;
        casters[0].maximum[0] = casters[0].maximum[1] = 1.f; casters[0].maximum[2] = .5f;
        casters[1] = casters[0]; casters[1].minimum[0] = 3.f; casters[1].maximum[0] = 4.f;
        casters[2] = casters[0]; casters[2].minimum[0] = 1.5f; casters[2].maximum[0] = 2.5f;
        VsmCasterBinningParameters bin_params{};
        bin_params.level[0][0] = bin_params.level[0][1] = -1.f; bin_params.level[0][2] = 2.f;
        bin_params.level[1][0] = bin_params.level[1][1] = -2.f; bin_params.level[1][2] = 4.f;
        bin_params.axis = 2; bin_params.levels = 2; bin_params.zrange[0] = -1.f; bin_params.zrange[1] = 1.f;
        if (!vsm_casters.write(0, casters, sizeof(casters), error)) break;
        bool bin_ok = true;
        for (uint32_t capacity : {1u, 3u})
        {
            if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
                vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
                !vsm_binning.record(command, 0, 3, 8, capacity, bin_params, vsm_casters, vsm_cache_work,
                    vsm_bin_parameters, vsm_pairs, error) ||
                !vsm_static_cache.publish(command, 0, cache_params, vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, error) ||
                vkEndCommandBuffer(command) != VK_SUCCESS ||
                vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
                vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
            { bin_ok = false; break; }
            uint32_t pairs[10]{};
            if (!vsm_pairs.read(0, pairs, sizeof(pairs), error) ||
                !vsm_table.read(0, page_table, sizeof(page_table), error) ||
                !vsm_cache_work.read(0, cache_work, sizeof(cache_work), error) || pairs[0] != 2 ||
                pairs[1] != (capacity == 1 ? 1u : 0u) || pairs[5] != 6 ||
                cache_work[1] != (capacity == 1 ? 1u : 0u) || page_table[7] != (capacity == 1 ? UINT32_MAX : 6u) ||
                (pairs[4] != 0 && pairs[4] != 2) ||
                (capacity == 3 && (pairs[7] != 6 || pairs[6] == pairs[4] ||
                    (pairs[6] != 0 && pairs[6] != 2))))
            { bin_ok = false; break; }
            if (capacity == 1)
            {
                if (read_vsm_static_jobs(vsm_cache_work, 2, 2, 8, 8, decoded_jobs, error) ||
                    !decoded_jobs.empty()) { bin_ok = false; break; }
                error.clear();
            }
            else if (!read_vsm_static_jobs(vsm_cache_work, 2, 2, 8, 8, decoded_jobs, error) ||
                decoded_jobs.size() != 1 || decoded_jobs[0].slot != 6)
            { bin_ok = false; break; }
        }
        if (!bin_ok) { if (error.empty()) error = "VSM caster binning or overflow mismatch"; break; }
        VsmInvalidationParameters invalidate_params{};
        invalidate_params.axis = 2; invalidate_params.levels = 2;
        invalidate_params.level[0][0] = invalidate_params.level[0][1] = -1.f; invalidate_params.level[0][2] = 2.f;
        invalidate_params.level[1][0] = invalidate_params.level[1][1] = -2.f; invalidate_params.level[1][2] = 4.f;
        invalidate_params.base[1][0] = 1;
        invalidate_params.zrange[0] = -1.f; invalidate_params.zrange[1] = 1.f;
        cache_params.frame = 2;
        for (uint32_t scenario = 0; scenario < 2; ++scenario)
        {
            const auto& change = casters[scenario == 0 ? 1 : 0];
            if (!vsm_casters.write(0, &change, sizeof(change), error) ||
                vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
                vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
                !vsm_invalidation.record(command, 0, 1, invalidate_params, vsm_casters,
                    vsm_invalidation_parameters, vsm_cache_slots, error) ||
                !vsm_static_cache.prepare(command, 0, cache_params, vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, error) ||
                vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
                vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
                !vsm_table.read(0, page_table, sizeof(page_table), error) ||
                !vsm_cache_work.read(0, cache_work, sizeof(cache_work), error))
            { bin_ok = false; break; }
            if (page_table[7] != (scenario == 0 ? 6u : UINT32_MAX) || cache_work[0] != (scenario == 0 ? 0u : 1u))
            { bin_ok = false; break; }
        }
        if (!bin_ok) { if (error.empty()) error = "VSM spatial invalidation mismatch"; break; }
        // Exercise the complete static update across the GPU/CPU boundary:
        // prepare, wait/read jobs, draw atlas pages, then publish the table.
        VsmStaticCacheParameters single_cache{};
        single_cache.axis = single_cache.levels = single_cache.budget = 1;
        single_cache.refresh_interval = 20;
        VsmPageProjection single_projection = page_projection;
        single_projection.axis = 1; single_projection.level[0][2] = 1.f;
        const uint32_t required_page = 1;
        if (!vsm_needed.write(0, &required_page, sizeof(required_page), error) ||
            vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_static_cache.reset(command, vsm_cache_slots, error) ||
            !vsm_static_cache.prepare(command, 0, single_cache, vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_table.read(0, page_table, sizeof(uint32_t), error) || page_table[0] != UINT32_MAX)
        { if (error.empty()) error = "VSM static update preparation failed"; break; }
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS || vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !vsm_static_cache.render_and_publish(vsm_atlas, generate_frame, single_projection, single_cache,
                vsm_needed, vsm_table, vsm_cache_slots, vsm_cache_work, record_test_vsm_page, nullptr, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !vsm_table.read(0, page_table, sizeof(uint32_t), error) || page_table[0] != 0)
        { if (error.empty()) error = "VSM static update rendering or publication failed"; break; }
        std::vector<RayTextureTransform> shifted(2);
        shifted[0].row0[3] = .5f;
        if (!level_scene.write_texture_transforms(0, shifted, error) ||
            vkResetFences(device, 1, &fence) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !query_pass.record(command, 0, level_scene, query_rays, query_hits, 4, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !query_hits.read(0, query_result, sizeof(query_result), error))
        { if (error.empty()) error = "transformed ray query failed"; break; }
        if (!query_result[0].found || query_result[0].material_id != 11 ||
            std::abs(query_result[0].distance - 1.f) > .001f)
        { error = "ray alpha test ignored the texture transform"; break; }
        // One ray beyond the row width requires a second dispatch row. Fill
        // inactive rays on the GPU and inspect both sides of the row boundary.
        constexpr uint32_t boundary = RayQueryBatch::MaxGroupsX * 64u;
        constexpr uint32_t large_count = boundary + 1;
        const VkDeviceSize large_bytes = VkDeviceSize(large_count) * sizeof(RayQueryInput);
        if (properties.properties.limits.maxStorageBufferRange >= large_bytes)
        {
            if (!large_rays.initialize(device, large_bytes, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    coherent, memory, vk.buffers, error) ||
                !large_hits.initialize(device, large_bytes, storage, coherent, memory, vk.buffers, error) ||
                vkResetFences(device, 1, &fence) != VK_SUCCESS ||
                vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
                vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS) break;
            vkCmdFillBuffer(command, large_rays.handle(), 0, large_bytes, 0);
            VkMemoryBarrier fill_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            fill_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            fill_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &fill_barrier, 0, nullptr, 0, nullptr);
            if (!query_pass.record(command, 0, level_scene, large_rays, large_hits, large_count, error)) break;
            VkResult large_status = vkEndCommandBuffer(command);
            if (large_status == VK_SUCCESS) large_status = vkQueueSubmit(queue, 1, &query_submit, fence);
            if (large_status == VK_SUCCESS) large_status = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
            if (large_status != VK_SUCCESS)
            { error = "large ray batch Vulkan result " + std::to_string(large_status); break; }
            RayQueryHit boundary_hits[2]{};
            if (!large_hits.read(VkDeviceSize(boundary - 1) * sizeof(RayQueryHit),
                    boundary_hits, sizeof(boundary_hits), error)) break;
            if (boundary_hits[0].distance != -1.f || boundary_hits[1].distance != -1.f ||
                boundary_hits[0].instance != UINT32_MAX || boundary_hits[1].instance != UINT32_MAX)
            { error = "ray dispatch did not cover both sides of the row boundary"; break; }
            std::cout << properties.properties.deviceName << ": two-row ray dispatch passed\n";
        }
        std::vector<VkTransformMatrixKHR> moved;
        for (const auto& batch : bump_batches) moved.push_back(batch.transform);
        moved[2].matrix[2][3] = .5f;
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !bump_scene.record_transform_update(command, moved, error) ||
            !bump_query_pass.record(command, 0, bump_scene, query_rays, bump_hits, 4, error) ||
            !bump_surface_pass.record(command, 0, bump_scene, query_rays, bump_hits, bump_surfaces, 4, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS ||
            vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !bump_surfaces.read(0, bump_result, sizeof(bump_result), error))
        { if (error.empty()) error = "ray scene transform update failed"; break; }
        if (std::abs(bump_result[0].position_valid[2] - 1.5f) > .001f ||
            std::abs(bump_result[0].normal[0] + bump_component) > .001f)
        { error = "updated ray scene and hit attributes disagree"; break; }
        moved[2].matrix[2][2] = 0.f;
        if (bump_scene.record_transform_update(command, moved, error))
        { error = "ray scene accepted singular transform"; break; }
        error.clear();
        FrameRayScenes frame_scenes;
        auto frame_batches = bump_batches;
        frame_batches[0].external_blas = level_scene.batch_blas_address(0);
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !frame_scenes.prepare(0, 1, command, device, queue, pool, memory, upload, pending,
                frame_batches, vk, as_properties, error))
        { if (error.empty()) error = "first frame ray scene failed"; break; }
        frame_batches[2].transform.matrix[2][3] = .5f;
        if (!frame_scenes.prepare(1, 1, command, device, queue, pool, memory, upload, pending,
                frame_batches, vk, as_properties, error) ||
            !bump_query_pass.record(command, 0, *frame_scenes.scene(1), query_rays, bump_hits, 4, error) ||
            !bump_surface_pass.record(command, 0, *frame_scenes.scene(1), query_rays, bump_hits, bump_surfaces, 4, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !bump_surfaces.read(0, bump_result, sizeof(bump_result), error))
        { if (error.empty()) error = "second frame ray scene failed"; break; }
        if (std::abs(bump_result[0].position_valid[2] - 1.5f) > .001f)
        { error = "second frame ray scene transform mismatch"; break; }
        frame_batches[2].transform.matrix[2][3] = .75f;
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !frame_scenes.prepare(0, 1, command, device, queue, pool, memory, upload, pending,
                frame_batches, vk, as_properties, error) ||
            !bump_query_pass.record(command, 0, *frame_scenes.scene(0), query_rays, bump_hits, 4, error) ||
            !bump_surface_pass.record(command, 0, *frame_scenes.scene(0), query_rays, bump_hits, bump_surfaces, 4, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !bump_surfaces.read(0, bump_result, sizeof(bump_result), error))
        { if (error.empty()) error = "frame ray scene update failed"; break; }
        RayHitBatchInfo second_slot[3]{};
        if (std::abs(bump_result[0].position_valid[2] - 1.75f) > .001f ||
            !frame_scenes.scene(1)->hit_batch_buffer().read(0, second_slot, sizeof(second_slot), error) ||
            std::abs(second_slot[2].row2[3] - .5f) > .001f)
        { error = "frame ray scene slots were not independent"; break; }
        frame_scenes.destroy();
        RayGeometryBatch deform_batch;
        deform_batch.external_blas = blas.address();
        deform_batch.geometries = blas_geometry;
        deform_batch.primitive_counts = blas_counts;
        deform_batch.material_ids = {11, 22};
        deform_batch.alpha_cutoffs = {0.f, 0.f};
        deform_batch.texture_slots = {UINT32_MAX, UINT32_MAX};
        deform_batch.diffuse_texture_slots = {0, 1};
        deform_batch.pbr_texture_slots = {UINT32_MAX, UINT32_MAX};
        deform_batch.metallic_texture_slots = {UINT32_MAX, UINT32_MAX};
        deform_batch.roughness_texture_slots = {UINT32_MAX, UINT32_MAX};
        deform_batch.ao_texture_slots = {UINT32_MAX, UINT32_MAX};
        deform_batch.normal_texture_slots = {UINT32_MAX, UINT32_MAX};
        deform_batch.normal_aux_texture_slots = {UINT32_MAX, UINT32_MAX};
        std::vector<RayGeometryBatch> deform_batches{deform_batch};
        if (vkResetFences(device, 1, &fence) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS ||
            !frame_scenes.prepare(0, 2, command, device, queue, pool, memory, upload, pending,
                deform_batches, vk, as_properties, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        { if (error.empty()) error = "deforming ray scene setup failed"; break; }
        for (size_t i = 0; i < 3; ++i) triangle[i].attributes[2] = .4f;
        if (!vertices.write(0, triangle, sizeof(triangle), error) ||
            vkResetFences(device, 1, &fence) != VK_SUCCESS ||
            vkResetCommandBuffer(command, 0) != VK_SUCCESS ||
            vkBeginCommandBuffer(command, &query_begin) != VK_SUCCESS) break;
        VkMemoryBarrier deform_before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        deform_before.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        deform_before.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &deform_before, 0, nullptr, 0, nullptr);
        if (!blas.record_build(command, blas_geometry, blas_counts, true, error)) break;
        VkMemoryBarrier deform_after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        deform_after.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        deform_after.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &deform_after, 0, nullptr, 0, nullptr);
        if (!frame_scenes.prepare(0, 2, command, device, queue, pool, memory, upload, pending,
                deform_batches, vk, as_properties, error, true) ||
            !bump_query_pass.record(command, 0, *frame_scenes.scene(0), query_rays, bump_hits, 4, error) ||
            !bump_surface_pass.record(command, 0, *frame_scenes.scene(0), query_rays, bump_hits, bump_surfaces, 4, error) ||
            vkEndCommandBuffer(command) != VK_SUCCESS || vkQueueSubmit(queue, 1, &query_submit, fence) != VK_SUCCESS ||
            vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS ||
            !bump_surfaces.read(0, bump_result, sizeof(bump_result), error))
        { if (error.empty()) error = "deforming ray scene update failed"; break; }
        if (std::abs(bump_result[0].position_valid[2] - .4f) > .001f)
        { error = "deformed BLAS and TLAS did not agree"; break; }
        frame_scenes.destroy();
        bump_scene.destroy();
        level_scene.destroy();
        ok = level_scene.address() == 0 && level_scene.blas_count() == 0 &&
            level_scene.instance_batch_indices().empty() &&
            level_scene.hit_batch_count() == 0 && level_scene.hit_geometry_count() == 0;
    } while (false);
    if (!ok) std::cerr << properties.properties.deviceName << ": " << error << '\n';
    // A failed submission may still own resources; drain the queue before cleanup.
    vkDeviceWaitIdle(device);
    query_pass.destroy();
    bump_query_pass.destroy();
    surface_pass.destroy();
    vsm_allocator.destroy();
    vsm_demand.destroy();
    vsm_static_cache.destroy();
    vsm_resolve.destroy();
    vsm_binning.destroy();
    vsm_invalidation.destroy();
    if (vsm_resolve_view) vkDestroyImageView(device, vsm_resolve_view, nullptr);
    if (vsm_resolve_image) vkDestroyImage(device, vsm_resolve_image, nullptr);
    if (vsm_resolve_memory) vkFreeMemory(device, vsm_resolve_memory, nullptr);
    vsm_atlas.destroy();
    bump_surface_pass.destroy();
    indirect_frames.destroy();
    local_test_generator.destroy();
    local_test_query.destroy();
    local_test_surfaces.destroy();
    local_test_lights.destroy();
    local_test_rays.destroy();
    local_test_hits.destroy();
    indirect_pool.reset();
    if (indirect_image_view) vkDestroyImageView(device, indirect_image_view, nullptr);
    if (indirect_image) vkDestroyImage(device, indirect_image, nullptr);
    if (indirect_image_memory) vkFreeMemory(device, indirect_image_memory, nullptr);
    indirect_surface_pass.destroy();
    indirect_surface_readback.destroy();
    indirect_sun_readback.destroy();
    indirect_radiance_readback.destroy();
    indirect_image_readback.destroy();
    query_surfaces.destroy();
    vsm_needed.destroy(); vsm_table.destroy(); vsm_pages.destroy(); vsm_counters.destroy(); vsm_parameters.destroy();
    vsm_cache_slots.destroy(); vsm_cache_work.destroy();
    vsm_atlas_readback.destroy();
    vsm_resolve_table.destroy(); vsm_resolve_parameters.destroy(); vsm_resolve_readback.destroy();
    vsm_casters.destroy(); vsm_bin_parameters.destroy(); vsm_pairs.destroy();
    vsm_invalidation_parameters.destroy();
    bump_surfaces.destroy();
    visibility_pass.destroy();
    visibility_readback.destroy();
    if (visibility_view) vkDestroyImageView(device, visibility_view, nullptr);
    if (visibility_image) vkDestroyImage(device, visibility_image, nullptr);
    if (visibility_memory) vkFreeMemory(device, visibility_memory, nullptr);
    generated_hit.destroy();
    large_rays.destroy();
    large_hits.destroy();
    generate_pass.destroy();
    indirect_pass.destroy();
    indirect_ray.destroy();
    background_pass.destroy();
    generated_ray.destroy();
    background_ray.destroy();
    alpha_textures.destroy();
    if (alpha_sampler) vkDestroySampler(device, alpha_sampler, nullptr);
    for (uint32_t slot = 0; slot < 3; ++slot)
    {
        if (alpha_views[slot]) vkDestroyImageView(device, alpha_views[slot], nullptr);
        if (alpha_images[slot]) vkDestroyImage(device, alpha_images[slot], nullptr);
        if (alpha_memory[slot]) vkFreeMemory(device, alpha_memory[slot], nullptr);
    }
    alpha_upload.destroy();
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (pool) vkDestroyCommandPool(device, pool, nullptr);
    tlas.destroy();
    blas.destroy();
    instances.destroy();
    query_hits.destroy();
    bump_hits.destroy();
    query_rays.destroy();
    indices.destroy();
    vertices.destroy();
    return ok;
}

int main(int argc, char** argv)
{
    int requested_device = -1;
    const char* shader_directory = nullptr;
    for (int argument = 1; argument < argc; ++argument)
    {
        if (std::string(argv[argument]) == "--device-index")
        {
            if (++argument >= argc)
            {
                std::cerr << "--device-index requires an adapter index\n";
                return 1;
            }
            try
            {
                size_t parsed = 0;
                requested_device = std::stoi(argv[argument], &parsed);
                if (parsed != std::string(argv[argument]).size() || requested_device < 0)
                    throw std::invalid_argument("invalid adapter index");
            }
            catch (const std::exception&)
            {
                std::cerr << "invalid --device-index value\n";
                return 1;
            }
        }
        else if (!shader_directory)
            shader_directory = argv[argument];
        else
        {
            std::cerr << "unexpected argument: " << argv[argument] << '\n';
            return 1;
        }
    }
    uint32_t instance_count = 0;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &instance_count, nullptr) != VK_SUCCESS)
        return 77;
    std::vector<VkExtensionProperties> instance_extensions(instance_count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &instance_count, instance_extensions.data()) != VK_SUCCESS)
        return 77;
    const auto has_properties2 = std::any_of(instance_extensions.begin(), instance_extensions.end(),
        [](const VkExtensionProperties& extension)
        { return std::strcmp(extension.extensionName,
            VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0; });
    if (!has_properties2) return 77;
    const char* requested[]{VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_SURFACE_EXTENSION_NAME};
    const auto has_surface = std::any_of(instance_extensions.begin(), instance_extensions.end(),
        [](const VkExtensionProperties& extension)
        { return std::strcmp(extension.extensionName, VK_KHR_SURFACE_EXTENSION_NAME) == 0; });
    if (!has_surface) return 77;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    uint32_t loader_version = VK_API_VERSION_1_0;
    if (vkEnumerateInstanceVersion(&loader_version) != VK_SUCCESS ||
        loader_version < VK_API_VERSION_1_2)
        return 77;
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    create.pApplicationInfo = &app;
    create.enabledExtensionCount = 2;
    create.ppEnabledExtensionNames = requested;
    VkInstance instance{};
    if (vkCreateInstance(&create, nullptr, &instance) != VK_SUCCESS) return 77;
    auto get_features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2KHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR"));
    if (!get_features2)
    {
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    uint32_t count = 0;
    VkResult status = vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (status != VK_SUCCESS || !count)
    {
        vkDestroyInstance(instance, nullptr);
        return 77;
    }
    std::vector<VkPhysicalDevice> devices(count);
    status = vkEnumeratePhysicalDevices(instance, &count, devices.data());
    if (status != VK_SUCCESS && status != VK_INCOMPLETE)
    {
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    if (requested_device >= 0 && static_cast<uint32_t>(requested_device) >= count)
    {
        std::cerr << "selected Vulkan physical device index is out of range\n";
        vkDestroyInstance(instance, nullptr);
        return 1;
    }
    uint32_t tested_devices = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (requested_device >= 0 && i != static_cast<uint32_t>(requested_device))
            continue;
        uint32_t extension_count = 0;
        if (vkEnumerateDeviceExtensionProperties(devices[i], nullptr, &extension_count, nullptr) != VK_SUCCESS)
        {
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        std::vector<VkExtensionProperties> extensions(extension_count);
        if (vkEnumerateDeviceExtensionProperties(devices[i], nullptr, &extension_count,
                extensions.data()) != VK_SUCCESS)
        {
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(devices[i], &properties);
        const auto rt = probe_ray_tracing_availability(devices[i], extensions,
            properties.apiVersion, get_features2);
        const bool bindless = probe_bindless_texture_availability(devices[i], extensions,
            properties.apiVersion, get_features2);
        if ((rt.ray_query || rt.ray_tracing_pipeline) && !rt.acceleration_structure)
        {
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        std::cout << properties.deviceName << " acceleration=" << rt.acceleration_structure
            << " query=" << rt.ray_query << " pipeline=" << rt.ray_tracing_pipeline
            << " bindless=" << bindless << '\n';
        if (!rt.ray_query || !bindless) continue;
        uint32_t family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &family_count, families.data());
        uint32_t graphics_family = UINT32_MAX;
        for (uint32_t family = 0; family < family_count; ++family)
            if (families[family].queueCount && (families[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
            { graphics_family = family; break; }
        if (graphics_family == UINT32_MAX)
        {
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        PhysicalDevice selected{};
        selected.handle = devices[i];
        selected.properties = properties;
        selected.graphics_present_family = graphics_family;
        selected.ray_tracing = rt;
        selected.bindless_textures = bindless;
        vkGetPhysicalDeviceFeatures(devices[i], &selected.features);
        VkDevice device{};
        VkQueue queue{};
        std::string error;
        if (!create_logical_device(selected, {vkCreateDevice, vkGetDeviceProcAddr},
                device, queue, error, true, app.apiVersion, bindless))
        {
            std::cerr << properties.deviceName << ": " << error << '\n';
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        const auto create_as = vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
        const auto address = vkGetDeviceProcAddr(device,
            properties.apiVersion >= VK_API_VERSION_1_2 ?
                "vkGetBufferDeviceAddress" : "vkGetBufferDeviceAddressKHR");
        if (!create_as || !address || !queue)
        {
            std::cerr << properties.deviceName << ": RT entry points missing createAS=" << bool(create_as)
                << " bufferAddress=" << bool(address) << " queue=" << bool(queue) << '\n';
            vkDestroyDevice(device, nullptr);
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        if (bindless && !exercise_bindless_table(devices[i], device))
        {
            vkDestroyDevice(device, nullptr);
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        if (shader_directory && !exercise_particle_pipelines(devices[i], device, queue,
                graphics_family, properties.limits, shader_directory))
        {
            vkDestroyDevice(device, nullptr);
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        if (!build_triangle_scene(devices[i], device, queue, graphics_family,
                properties.apiVersion))
        {
            vkDestroyDevice(device, nullptr);
            vkDestroyInstance(instance, nullptr);
            return 1;
        }
        ++tested_devices;
        vkDestroyDevice(device, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
    return tested_devices ? 0 : 77;
}
