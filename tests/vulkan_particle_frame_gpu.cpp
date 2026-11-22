#include "xrEngine/stdafx.h"
#include "src/Layers/xrRenderVK/FrameImageGraph.h"
#include "src/Layers/xrRenderVK/FrameImagePool.h"
#include "src/Layers/xrRenderVK/VisibilityPasses.h"
#include "src/Layers/xrRenderVK/LevelModels.h"
#include "src/Layers/xrRenderVK/SkyProbeReducePass.h"
#include "src/Layers/xrRenderVK/SkyProbeBakePass.h"
#include "src/Layers/xrRenderVK/DetailCullPass.h"
#include "src/Layers/xrRenderVK/DetailGeneratePass.h"
#include "src/Layers/xrRenderVK/GpuParticleBuffers.h"
#include "src/Layers/xrRenderVK/GpuParticleCompute.h"
#include "src/Layers/xrRenderVK/GpuParticleFrame.h"
#include "src/Layers/xrRenderVK/GpuParticleFrameInput.h"
#include "src/Layers/xrRenderVK/GpuParticleReadback.h"
#include "src/Layers/xrRenderVK/GpuParticleRuntime.h"
#include "src/Layers/xrRenderVK/GpuParticleGraphics.h"
#include "src/Layers/xrRenderVK/SmokeTrailCompute.h"
#include "src/Layers/xrRenderVK/SmokeTrailDrawPass.h"
#include "src/Layers/xrRenderVK/SmokeTrailNoise.h"
#include "src/Layers/xrRenderVK/GpuParticleEmitters.h"
#include "src/Layers/xrRenderVK/GpuParticleDrawPass.h"
#include "src/Layers/xrRenderVK/ParticleCatalog.h"
#include "src/Layers/xrRenderVK/GpuParticleUniforms.h"
#include "src/Layers/xrRenderVK/GpuParticleUpload.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xray::render;
using namespace xray::render::vulkan;

struct Fixture
{
    VkInstance instance{}; VkDevice device{}; VkCommandPool pool{};
    ~Fixture() { if (device) { vkDeviceWaitIdle(device); if (pool) vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyDevice(device, nullptr); } if (instance) vkDestroyInstance(instance, nullptr); }
};
static void require(bool condition, const std::string& message)
{ if (!condition) throw std::runtime_error(message); }

static void exercise_visibility_raster(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& vk, const VisibilityPasses::Source& source, bool bindless)
{
    std::string error;
    std::vector<uint8_t> shader;
    require(source("vk/visibility_software_raster.cs.spv", shader, error), error);
    require(shader.size() % sizeof(uint32_t) == 0, "invalid raster shader size");
    std::vector<uint32_t> code(shader.size() / sizeof(uint32_t));
    std::memcpy(code.data(), shader.data(), shader.size());
    StorageComputePass raster;
    constexpr auto image = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    constexpr auto storage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    require(raster.initialize(device, vkGetDeviceProcAddr, 4,
        {{0, image}, {1, storage}, {2, storage}, {3, storage}, {4, image}, {5, storage},
            {6, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER}}, sizeof(visibility::SoftwareRasterParameters),
        limits.maxStorageBufferRange, limits.maxUniformBufferRange, VK_NULL_HANDLE,
        code.data(), shader.size(), error), error);

    const visibility::DrawEntry entries[]{{0, 3, 0, 0, 0}, {3, 3, 0, 0, 1, 0}, {6, 3, 0, 0, 1},
        {9, 3, 0, 0, 1}};
    const VkDrawIndexedIndirectCommand commands[]{{3, 1, 0, 0, 0}, {3, 1, 3, 0, 0}, {3, 1, 6, 0, 0},
        {3, 1, 9, 0, 0}};
    const uint32_t indices[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    LevelVertex vertices[12]{};
    for (uint32_t draw = 0; draw < 3; ++draw)
    {
        const float depths[]{0.75f, 0.5f, 0.0625f};
        const float depth = depths[draw];
        const float apex_depth = draw == 1 ? -0.25f : depth;
        const float points[3][3]{{-0.75f, -0.75f, depth}, {0.75f, -0.75f, depth}, {0.f, 0.75f, apex_depth}};
        for (uint32_t vertex = 0; vertex < 3; ++vertex)
        {
            std::copy_n(points[vertex], 3, vertices[3 * draw + vertex].position);
            vertices[3 * draw + vertex].normal[1] = 1.f;
        }
    }
    const float far_points[][3]{{.5f, -.9f, 1.f}, {.89f, -.85f, 1.f}, {.91f, .5f, 1.f}};
    for (uint32_t vertex = 0; vertex < 3; ++vertex)
    {
        std::copy_n(far_points[vertex], 3, vertices[9 + vertex].position);
        vertices[9 + vertex].normal[1] = 1.f;
    }
    constexpr uint32_t visible_draws = 0b1011u;
    constexpr auto host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    BufferResource geometry[4], mask, readback;
    const auto upload = [&](BufferResource& buffer, const void* data, size_t bytes, VkBufferUsageFlags usage) {
        require(buffer.initialize(device, bytes, usage, host, memory, vk, error) && buffer.write(0, data, bytes, error), error);
    };
    upload(geometry[0], entries, sizeof(entries), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    upload(geometry[1], indices, sizeof(indices), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    upload(geometry[2], vertices, sizeof(vertices), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    upload(geometry[3], commands, sizeof(commands), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    upload(mask, &visible_draws, sizeof(visible_draws), VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT);
    constexpr VkDeviceSize raster_bytes = 2 * 8 * 8 * sizeof(uint32_t);
    constexpr VkDeviceSize resolved_bytes = 8 * 8 * 4 * sizeof(uint16_t);
    require(readback.initialize(device, raster_bytes + (bindless ? 2 * resolved_bytes : 0), VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        host, memory, vk, error), error);

    FrameImagePool images;
    require(images.initialize(device, memory, {vkCreateImage, vkDestroyImage, vkGetImageMemoryRequirements,
        vkAllocateMemory, vkFreeMemory, vkBindImageMemory}), "visibility image pool");
    struct Views
    {
        VkDevice device;
        VkBufferView mask{};
        VkImageView images[5]{};
        VkSampler sampler{};
        ~Views()
        {
            if (mask) vkDestroyBufferView(device, mask, nullptr);
            if (sampler) vkDestroySampler(device, sampler, nullptr);
            for (auto image : images)
                if (image) vkDestroyImageView(device, image, nullptr);
        }
    } views{device};
    VkBufferViewCreateInfo mask_info{VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO};
    mask_info.buffer = mask.handle();
    mask_info.format = VK_FORMAT_R32_UINT;
    mask_info.range = sizeof(visible_draws);
    require(vkCreateBufferView(device, &mask_info, nullptr, &views.mask) == VK_SUCCESS, "visibility mask view");
    FrameImageGraph graph;
    ImageStateTracker states;
    VkImage targets[5]{};
    FrameImageGraph::Resource resources[5];
    const size_t target_count = bindless ? 5 : 2;
    constexpr VkImageSubresourceRange color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (size_t i = 0; i < target_count; ++i)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = i < 2 ? VK_FORMAT_R32_UINT : VK_FORMAT_R16G16B16A16_SFLOAT;
        info.extent = {8, 8, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT;
        require(images.acquire(info, targets[i], error), error);
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = targets[i];
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = info.format;
        view_info.subresourceRange = color;
        require(vkCreateImageView(device, &view_info, nullptr, &views.images[i]) == VK_SUCCESS, "visibility image view");
        require(graph.import_image(targets[i], color, resources[i], error) &&
            states.register_image(targets[i], color, ImageUse::Undefined, error), error);
    }
    const auto clear = graph.add_command_pass("clear visibility test", [&](VkCommandBuffer command) {
        VkClearColorValue value{};
        vkCmdClearColorImage(command, targets[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &color);
        value.uint32[0] = 0x3f800000u;
        vkCmdClearColorImage(command, targets[1], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &value, 1, &color);
        if (bindless)
        {
            VkClearColorValue zero{};
            VkClearColorValue material{{0.25f, 0.5f, 1.f, 1.f}};
            vkCmdClearColorImage(command, targets[2], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &color);
            vkCmdClearColorImage(command, targets[3], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &color);
            vkCmdClearColorImage(command, targets[4], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &material, 1, &color);
        }
    });
    for (size_t i = 0; i < target_count; ++i)
        require(graph.write(clear, resources[i], ImageUse::TransferDestination, resources[i], error), error);

    std::vector<VkDescriptorBufferInfo> buffers;
    for (const auto& buffer : geometry)
        buffers.push_back({buffer.handle(), 0, buffer.size()});
    const std::vector<VkDescriptorImageInfo> descriptors{{VK_NULL_HANDLE, views.images[0], VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, views.images[1], VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t phase = 0; phase < 2; ++phase)
    {
        const auto pass = graph.add_checked_command_pass("raster visibility test", [&, phase](VkCommandBuffer cb, std::string& why) {
            visibility::SoftwareRasterParameters params{};
            for (size_t i = 0; i < 16; i += 5) params.camera.view_projection[i] = 1.f;
            params.camera.viewport.extent[0] = params.camera.viewport.extent[1] = 8.f;
            params.write_ids = phase;
            for (uint32_t batch = 0; batch < 2; ++batch)
            {
                params.first_draw = batch;
                params.draw_count = batch ? 3 : 1;
                params.geometry_batch = batch;
                if (!raster.record_mixed(cb, 2 * phase + batch, buffers, descriptors, &params, params.draw_count, 1,
                        why, VK_NULL_HANDLE, views.mask))
                    return false;
            }
            return true;
        });
        for (size_t i = 0; i < 2; ++i)
            require(graph.read_write(pass, resources[i], resources[i], error), error);
    }
    BindlessTextureTable materials;
    VisibilityPasses visibility_passes;
    if (bindless)
    {
        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        require(vkCreateSampler(device, &sampler_info, nullptr, &views.sampler) == VK_SUCCESS, "visibility test sampler");
        require(materials.initialize(device, vkGetDeviceProcAddr, 1, error) &&
            materials.write(0, views.images[4], views.sampler, error, VK_IMAGE_LAYOUT_GENERAL), error);
        require(visibility_passes.initialize(device, vkGetDeviceProcAddr, limits, 1, 2, VK_NULL_HANDLE,
            materials, source, error), error);
        const auto resolve = graph.add_checked_command_pass("resolve visibility test", [&](VkCommandBuffer cb, std::string& why) {
            const VisibilityPasses::Geometry input{buffers[0], buffers[1], buffers[2]};
            const VkDescriptorImageInfo ids{views.sampler, views.images[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            visibility::MaterialResolveParameters params{};
            for (size_t i = 0; i < 16; i += 5) params.camera.view_projection[i] = 1.f;
            params.camera.viewport.extent[0] = params.camera.viewport.extent[1] = 8.f;
            params.material_count = 1;
            for (uint32_t batch = 0; batch < 2; ++batch)
            {
                params.first_draw = batch;
                params.draw_count = batch ? 3 : 1;
                params.geometry_batch = batch;
                if (!visibility_passes.resolve(cb, 0, batch, input, ids, materials, views.images[2], views.images[3],
                        {8, 8}, params, why))
                    return false;
            }
            return true;
        });
        require(graph.read(resolve, resources[0], ImageUse::ComputeSampled, error) &&
            graph.read(resolve, resources[4], ImageUse::StorageRead, error), error);
        for (size_t i = 2; i < 4; ++i)
            require(graph.read_write(resolve, resources[i], resources[i], error), error);
    }
    const auto copy = graph.add_command_pass("read visibility test", [&](VkCommandBuffer cb) {
        for (size_t i = 0; i < (bindless ? 4u : 2u); ++i)
        {
            const auto offset = i < 2 ? i * 8 * 8 * sizeof(uint32_t) : raster_bytes + (i - 2) * resolved_bytes;
            VkBufferImageCopy region{offset, 0, 0,
                {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, {0, 0, 0}, {8, 8, 1}};
            vkCmdCopyImageToBuffer(cb, targets[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(), 1, &region);
        }
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier,
            0, nullptr, 0, nullptr);
    });
    for (size_t i = 0; i < (bindless ? 4u : 2u); ++i)
        require(graph.read(copy, resources[i], ImageUse::TransferSource, error), error);
    VkCommandBuffer cb{};
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    const VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    require(vkAllocateCommandBuffers(device, &allocation, &cb) == VK_SUCCESS &&
        vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS, "visibility test command buffer");
    require(graph.execute(cb, states, {vkCmdPipelineBarrier}, error), error);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "visibility test end");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS,
        "visibility test submit");
    vkFreeCommandBuffers(device, pool, 1, &cb);
    uint32_t pixels[2][64]{};
    require(readback.read(0, pixels, sizeof(pixels), error), error);
    require(pixels[0][36] == visibility::pack_id(1, 0), "nearest unmasked triangle must win across batches");
    float depth{};
    std::memcpy(&depth, &pixels[1][36], sizeof(depth));
    require(std::abs(depth - 0.1875f) < 1e-6f, "visibility depth after near clipping");
    require(pixels[0][20] == visibility::pack_id(0, 0), "near clipping must expose the triangle behind it");
    require(pixels[0][0] == visibility::Background && pixels[1][0] == 0x3f800000u, "visibility background clear");
    require(pixels[0][23] == visibility::pack_id(3, 0) && pixels[1][23] == 0x3f800000u,
        "far-plane triangle must retain its ID at an interior pixel");
    if (bindless)
    {
        uint16_t output[2][64][4]{};
        require(readback.read(raster_bytes, output, sizeof(output), error), error);
        require(output[0][36][0] == 0x3400 && output[0][36][1] == 0x3800 && output[0][36][2] == 0x3c00,
            "visibility resolve material color");
        require(output[0][36][3] == 0x3800 && output[1][36][0] == 0 && output[1][36][1] == 0x3c00 &&
            output[1][36][3] == 0x3a00, "visibility resolve baked sunlight and hemisphere");
        require(output[0][20][3] == 0x3c00 && output[1][20][3] == 0x3c00,
            "visibility resolve unlightmapped material");
        require(output[0][0][0] == 0 && output[1][0][3] == 0, "visibility resolve background");
    }
}

static void exercise_collision_shader(VkPhysicalDevice physical, VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& vk, const char* shader_path)
{
    std::ifstream file(shader_path, std::ios::binary | std::ios::ate);
    const auto size = file.tellg(); require(file && size > 0 && size % 4 == 0, "collision test shader");
    std::vector<uint32_t> code(size_t(size) / 4); file.seekg(0); file.read(reinterpret_cast<char*>(code.data()), size);
    require(bool(file), "collision shader read");
    GpuParticleCollisionInput input;
    input.nodes.resize(1); auto& node = input.nodes[0];
    node.minimum[0] = node.minimum[1] = -1.f; node.minimum[2] = node.maximum[2] = 2.f;
    node.maximum[0] = node.maximum[1] = 1.f; node.escape = 1; node.count = 1;
    input.static_triangles.resize(1);
    auto& tri = input.static_triangles[0];
    tri.a[0] = -1.f; tri.a[1] = -1.f; tri.a[2] = 2.f;
    tri.b[1] = 1.f; tri.b[2] = 2.f;
    tri.c[0] = 1.f; tri.c[1] = -1.f; tri.c[2] = 2.f;
    input.shapes.resize(2); input.objects.resize(2); input.blocks.resize(1);
    for (uint32_t i = 0; i < 2; ++i)
    {
        auto& shape = input.shapes[i]; shape.type = 2;
        shape.data0[0] = float(i * 2); shape.data0[2] = 1.f; shape.data0[3] = .4f;
        auto& object = input.objects[i]; object.first = i; object.count = 1;
        std::memcpy(object.spatial_sphere, shape.data0, sizeof(shape.data0));
        std::memcpy(object.sphere, shape.data0, sizeof(shape.data0));
    }
    input.objects[1].mesh = 1; input.objects[1].row0[0] = input.objects[1].row1[1] = input.objects[1].row2[2] = 1.f;
    input.shapes[1].triangle_count = 1; input.dynamic_triangles.resize(1);
    auto& mesh = input.dynamic_triangles[0];
    mesh.a[0] = 1.75f; mesh.a[1] = -.25f; mesh.a[2] = 1.f;
    mesh.b[0] = 2.f; mesh.b[1] = .25f; mesh.b[2] = 1.f;
    mesh.c[0] = 2.25f; mesh.c[1] = -.25f; mesh.c[2] = 1.f;
    auto& block = input.blocks[0]; block.first = 0; block.count = 2;
    block.minimum[0] = -1.f; block.minimum[1] = -1.f; block.minimum[2] = 0.f;
    block.maximum[0] = 3.f; block.maximum[1] = 1.f; block.maximum[2] = 3.f;
    std::vector<uint8_t> packed; std::string error;
    require(pack_gpu_particle_collision(input, limits.maxStorageBufferRange, packed, error), error);
    BufferResource collision, output;
    require(collision.initialize(device, packed.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error) &&
        collision.write(0, packed.data(), packed.size(), error), error);
    require(output.initialize(device, 8 * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error), error);
    StorageComputePass pass;
    const std::vector<StorageComputePass::Binding> layout{{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}};
    require(pass.initialize(device, vkGetDeviceProcAddr, 1, layout, 0, limits.maxStorageBufferRange,
        limits.maxUniformBufferRange, VK_NULL_HANDLE, code.data(), size_t(size), error), error);
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb{}; require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS, "collision command allocation");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    require(vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS, "collision command begin");
    const std::vector<VkDescriptorBufferInfo> descriptors{{output.handle(), 0, output.size()},
        {collision.handle(), 0, collision.size()}};
    require(pass.record(cb, 0, descriptors, nullptr, 1, 1, error), error);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "collision command end");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS, "collision submit");
    float result[32]{}; require(output.read(0, result, sizeof(result), error), error);
    require(result[0] == 1.f && result[1] == 1.f && result[2] < -.99f &&
        result[4] == 1.f && result[5] == 2.f && result[8] == 1.f && result[9] == 1.f &&
        result[12] == 1.f && result[13] == 2.f && result[16] == 0.f &&
        result[20] == 1.f && result[21] == 0.f && result[22] == -3.f && result[23] == -3.f &&
        result[24] == 1.f && result[25] == 1.f, "GPU collision ray or response results");
    require(result[28] == 255.f && result[29] == 255.f && result[30] == 0.f && result[31] == 123.f,
        "GPU particle animation wrapping");
    vkFreeCommandBuffers(device, pool, 1, &cb);
    std::cout << "GPU particle collision shader passed\n";
    (void)physical;
}

static void exercise_smoke_shader(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& vk, const GpuParticleCompute::Source& source)
{
    std::string error; SmokeTrailCompute smoke;
    require(smoke.initialize(device, vkGetDeviceProcAddr, limits, memory, vk, 1, VK_NULL_HANDLE, source, error), error);
    BufferResource readback;
    require(readback.initialize(device, 96, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error), error);
    SmokeTrailEmitParams emit{}; emit.current_x = 1.f; emit.lifetime = 2.f; emit.count = 2; emit.max_points = 256;
    SmokeTrailSimParams sim{}; sim.dt = .1f; sim.max_points = 256; sim.heat = 1.f;
    SmokeTrailCompactParams compact{}; compact.max_points = 256; compact.subdivisions = 1;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb{}; require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS, "smoke command allocation");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    require(vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS && smoke.record(cb, 0, emit, sim, compact, true, error), error);
    const VkBufferCopy state{0, 0, 16}, args{0, 16, 16}, points{0, 32, 64};
    vkCmdCopyBuffer(cb, smoke.get(SmokeTrailCompute::State).handle(), readback.handle(), 1, &state);
    vkCmdCopyBuffer(cb, smoke.get(SmokeTrailCompute::DrawArguments).handle(), readback.handle(), 1, &args);
    vkCmdCopyBuffer(cb, smoke.get(SmokeTrailCompute::ControlPoints).handle(), readback.handle(), 1, &points);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "smoke command end");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS, "smoke submit");
    uint32_t words[24]{}; require(readback.read(0, words, sizeof(words), error), error);
    float first_x{}, second_x{}, distance{};
    std::memcpy(&first_x, &words[8], 4); std::memcpy(&second_x, &words[16], 4); std::memcpy(&distance, &words[3], 4);
    require(words[0] == 2 && words[1] == 2 && words[2] == 2 && words[4] == 6 && words[5] == 128 &&
        std::abs(first_x - .25f) < .001f && std::abs(second_x - .75f) < .001f && std::abs(distance - .5f) < .001f,
        "smoke trail ordered emission, simulation and compaction");
    vkFreeCommandBuffers(device, pool, 1, &cb);
    emit.previous_x = 10.f;
    emit.current_x = 11.f;
    emit.restart = 1;
    require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS &&
        vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS && smoke.record(cb, 0, emit, sim, compact, false, error), error);
    vkCmdCopyBuffer(cb, smoke.get(SmokeTrailCompute::State).handle(), readback.handle(), 1, &state);
    vkCmdCopyBuffer(cb, smoke.get(SmokeTrailCompute::ControlPoints).handle(), readback.handle(), 1, &points);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "restarted smoke command end");
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS,
        "restarted smoke submit");
    require(readback.read(0, words, sizeof(words), error), error);
    std::memcpy(&first_x, &words[8], 4); std::memcpy(&second_x, &words[16], 4);
    require(words[0] == 2 && words[1] == 2 && words[2] == 2 &&
        std::abs(first_x - 10.25f) < .001f && std::abs(second_x - 10.75f) < .001f,
        "restarted smoke ribbon retained control points from before the gap");
    vkFreeCommandBuffers(device, pool, 1, &cb);
    smoke.destroy();
    FrameImagePool noise_images;
    require(noise_images.initialize(device, memory, {vkCreateImage, vkDestroyImage,
        vkGetImageMemoryRequirements, vkAllocateMemory, vkFreeMemory, vkBindImageMemory}), "smoke noise image pool");
    SmokeTrailNoise noise;
    require(noise.initialize(device, vkGetDeviceProcAddr, limits, memory, vk, 8, 1, VK_NULL_HANDLE,
        source, error, UINT32_MAX, UINT32_MAX, &noise_images), error);
    BufferResource noise_readback;
    require(noise_readback.initialize(device, 8 * 8 * 8 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error), error);
    require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS &&
        noise.record(cb, 0, .25f, 4.f, true, error), error);
    VkBufferCopy noise_copy{0, 0, 8 * 8 * 8 * 4};
    vkCmdCopyBuffer(cb, noise.buffer().handle(), noise_readback.handle(), 1, &noise_copy);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "smoke noise end"); submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS, "smoke noise submit");
    uint32_t voxels[8 * 8 * 8]{}; require(noise_readback.read(0, voxels, sizeof(voxels), error), error);
    bool varied = false;
    for (uint32_t voxel : voxels) { require((voxel >> 24) == 255, "smoke noise missing voxel"); varied |= voxel != voxels[0]; }
    require(varied && voxels[0] != voxels[7 * 8 * 8], "smoke noise did not span all volume slices");
    const uint32_t old_value = voxels[0];
    vkFreeCommandBuffers(device, pool, 1, &cb);
    require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS &&
        noise.record(cb, 0, .75f, 4.f, false, error), error);
    vkCmdCopyBuffer(cb, noise.buffer().handle(), noise_readback.handle(), 1, &noise_copy);
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "smoke noise update end"); submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS &&
        noise_readback.read(0, voxels, sizeof(voxels), error) && voxels[0] != old_value, "smoke noise temporal update");
    vkFreeCommandBuffers(device, pool, 1, &cb); noise.destroy();
    SkyProbeReducePass sky;
    require(sky.initialize(device, vkGetDeviceProcAddr, limits, 1, VK_NULL_HANDLE, source, error), error);
    xray::render::framegraph::SkyProbeLayout layout;
    std::vector<RayQueryInput> generated_rays;
    require(layout.initialize({0, 2, 4}, {2, 4, 6}, 2, 8) &&
        generate_sky_probe_rays(layout, 1, 1, 100, generated_rays) &&
        generated_rays.size() == 6 * SkyProbeRaysPerFace, "sky probe ray layout");
    std::vector<SkyProbeVisibility> grid(layout.count);
    for (size_t i = 0; i < grid.size(); ++i)
        grid[i].faces.fill((i & 1) ? 1.f : 0.f);
    std::array<float, 6> sampled{};
    require(sample_sky_probes(layout, grid, {1.f, 3.f, 5.f}, sampled) &&
        sampled == std::array<float, 6>{.5f, .5f, .5f, .5f, .5f, .5f} &&
        !sample_sky_probes(layout, grid, {-1.f, 3.f, 5.f}, sampled), "sky probe interpolation");
    xray::render::framegraph::SkyProbeLayout translated_grid;
    std::array<float, 3> endpoint{};
    require(translated_grid.initialize({1000000.f, 2.f, 4.f}, {1000000.f, 2.f, 4.f}, .3f, 8) &&
        translated_grid.position(7, endpoint) && sample_sky_probes(translated_grid, grid, endpoint, sampled) &&
        sampled == grid.back().faces, "sky probe interpolation at a rounded grid endpoint");
    const std::vector<SkyProbeVisibility> baked_prefix(grid.begin(), grid.begin() + 2);
    require(sample_sky_probes(layout, baked_prefix, {1.f, 2.f, 4.f}, sampled) &&
        sampled == std::array<float, 6>{.5f, .5f, .5f, .5f, .5f, .5f} &&
        !sample_sky_probes(layout, baked_prefix, {1.f, 3.f, 5.f}, sampled) &&
        sampled == std::array<float, 6>{.5f, .5f, .5f, .5f, .5f, .5f}, "partially baked sky probe interpolation");
    for (size_t i = 0; i < generated_rays.size(); ++i)
    {
        const auto& ray = generated_rays[i];
        const auto face = i / SkyProbeRaysPerFace;
        const float squared_length = ray.direction_max[0] * ray.direction_max[0] +
            ray.direction_max[1] * ray.direction_max[1] + ray.direction_max[2] * ray.direction_max[2];
        require(ray.origin_min[0] == 2.f && ray.origin_min[1] == 2.f && ray.origin_min[2] == 4.f &&
            ray.direction_max[3] == 100.f && std::abs(squared_length - 1.f) < .001f &&
            (ray.direction_max[face / 2] > 0.f) == (face % 2 == 0), "sky probe ray direction");
    }
    const auto host_memory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    BufferResource hits, probes;
    std::array<RayQueryHit, 12> rays{};
    rays[0].found = rays[2].found = rays[3].found = 1;
    require(hits.initialize(device, sizeof(rays), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        host_memory, memory, vk, error) &&
        probes.initialize(device, 2 * sizeof(SkyProbeVisibility), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_memory, memory, vk, error) && hits.write(0, rays.data(), sizeof(rays), error), error);
    require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS &&
        vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS &&
        sky.record(cb, 0, hits, probes, 1, 1, 2, error), error);
    VkMemoryBarrier probe_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    probe_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    probe_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &probe_barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "sky probe reduction end");
    submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
        vkQueueWaitIdle(queue) == VK_SUCCESS, "sky probe reduction submit");
    SkyProbeVisibility sky_result{};
    require(probes.read(sizeof(sky_result), &sky_result, sizeof(sky_result), error) &&
        sky_result.faces == std::array<float, 6>{.5f, 0.f, 1.f, 1.f, 1.f, 1.f}, "sky probe face visibility");
    vkFreeCommandBuffers(device, pool, 1, &cb);
    DetailCullPass grass;
    require(grass.initialize(device, vkGetDeviceProcAddr, limits, 1, VK_NULL_HANDLE, source, error), error);
    BufferResource placements, instances, detail_args;
    const std::array<DetailCullInput, 2> planted{{{2.5f, 0, .5f, 0, 2}, {3.5f, 0, .5f, 0, 1}}};
    require(placements.initialize(device, sizeof(planted), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        host_memory, memory, vk, error) &&
        instances.initialize(device, 2 * sizeof(DetailInstance), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host_memory, memory, vk, error) &&
        detail_args.initialize(device, sizeof(VkDrawIndexedIndirectCommand), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            host_memory, memory, vk, error) && placements.write(0, planted.data(), sizeof(planted), error), error);
    // Uniform clip-space scaling leaves the frustum and world-space radius unchanged.
    const float identity[16]{2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2};
    require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS &&
        vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS &&
        grass.record(cb, 0, placements, instances, detail_args, identity, 2, 12, 1.f, error), error);
    probe_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &probe_barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "detail cull end");
    submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
        vkQueueWaitIdle(queue) == VK_SUCCESS, "detail cull submit");
    VkDrawIndexedIndirectCommand draw{};
    DetailInstance grass_instance{};
    require(detail_args.read(0, &draw, sizeof(draw), error) && draw.indexCount == 12 && draw.instanceCount == 1 &&
        instances.read(0, &grass_instance, sizeof(grass_instance), error) &&
        grass_instance.position_scale[3] == 2.f && grass_instance.rotation[0] == 1.f, "compacted GPU grass");
    vkFreeCommandBuffers(device, pool, 1, &cb);
    DetailGeneratePass generate;
    require(generate.initialize(device, vkGetDeviceProcAddr, limits, 1, VK_NULL_HANDLE, source, error), error);
    const std::array<DetailGenerateSlot, 2> cells{{
        {{0, 0, 0, 2}, {0, 0x3f3f3f00u, 0, 3}, {0x0f0f0f0fu}},
        {{0, 0, 0, 2}, {0, 0x3f3f3f00u, 0, 0}, {0x0f0f0f0fu}}}};
    const DetailGroundTriangle triangles[]{
        {{0, .75f, 0, 0}, {0, .75f, 2, 0}, {2, .75f, 0, 0}},
        {{0, .5f, 0, 0}, {0, 1.5f, 2, 0}, {2, 1.5f, 0, 0}},
        {{0, 2, 0, 0}, {2, 2, 0, 0}, {0, 2, 2, 0}}};
    const std::array<float, 2> scales{2, 3};
    BufferResource cell_data, ground, prototype, generated;
    const auto upload_data = [&](BufferResource& buffer, const void* data, size_t bytes) {
        return buffer.initialize(device, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_memory, memory, vk, error) &&
            buffer.write(0, data, bytes, error);
    };
    require(upload_data(cell_data, cells.data(), sizeof(cells)) && upload_data(ground, triangles, sizeof(triangles)) &&
        upload_data(prototype, scales.data(), sizeof(scales)) &&
        generated.initialize(device, 2 * sizeof(GeneratedDetail), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_memory, memory, vk, error), error);
    require(vkAllocateCommandBuffers(device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS &&
        generate.record(cb, 0, cell_data, ground, prototype, generated, {2, 1, 2.f, 1.f}, error), error);
    require(grass.record(cb, 0, generated, instances, detail_args, identity, 2, 12, 1.f, error, 0), error);
    probe_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 1, &probe_barrier, 0, nullptr, 0, nullptr);
    require(vkEndCommandBuffer(cb) == VK_SUCCESS, "detail generation end");
    submit.pCommandBuffers = &cb;
    require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
        vkQueueWaitIdle(queue) == VK_SUCCESS, "detail generation submit");
    std::array<GeneratedDetail, 2> grass_results{};
    require(generated.read(0, grass_results.data(), sizeof(grass_results), error) && grass_results[0].model == 0 &&
        grass_results[0].position_scale[0] == .5f && grass_results[0].position_scale[1] == 1.f &&
        grass_results[0].position_scale[3] == 2.f && grass_results[1].model == UINT32_MAX, "GPU ground placement");
    require(detail_args.read(0, &draw, sizeof(draw), error) && draw.instanceCount == 1 &&
        instances.read(0, &grass_instance, sizeof(grass_instance), error) &&
        grass_instance.position_scale[1] == 1.f, "GPU generated grass draw arguments");
    vkFreeCommandBuffers(device, pool, 1, &cb);
    std::cout << "GPU smoke trail and detail compute passed\n";
}

int main(int argc, char** argv)
{
    if (argc < 2) { std::cerr << "shader directory is required\n"; return 1; }
    const std::string shader_directory = argv[1];
    const char* collision_shader = nullptr;
    int requested_device = -1;
    for (int argument = 2; argument < argc; ++argument)
    {
        if (std::string(argv[argument]) == "--device-index")
        {
            if (++argument >= argc) { std::cerr << "--device-index requires an adapter index\n"; return 1; }
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
        else if (!collision_shader)
            collision_shader = argv[argument];
        else
        {
            std::cerr << "unexpected argument: " << argv[argument] << '\n';
            return 1;
        }
    }
    Core.Initialize("vulkan_particle_frame_gpu_test", nullptr, false);
    struct CoreGuard { ~CoreGuard() { Core._destroy(); } } core_guard;
    try
    {
        Fixture f;
        uint32_t loader_version = VK_API_VERSION_1_1;
        vkEnumerateInstanceVersion(&loader_version);
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.apiVersion = std::min(loader_version, uint32_t(VK_API_VERSION_1_2));
        VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; instance_info.pApplicationInfo = &app;
        require(vkCreateInstance(&instance_info, nullptr, &f.instance) == VK_SUCCESS, "instance");
        uint32_t count = 0; vkEnumeratePhysicalDevices(f.instance, &count, nullptr);
        require(count != 0, "no Vulkan devices");
        std::vector<VkPhysicalDevice> devices(count); vkEnumeratePhysicalDevices(f.instance, &count, devices.data());
        require(requested_device < 0 || static_cast<uint32_t>(requested_device) < count,
            "selected Vulkan physical device index is out of range");
        VkPhysicalDevice physical{}; uint32_t family = UINT32_MAX; VkPhysicalDeviceProperties props{};
        for (size_t device_index = 0; device_index < devices.size(); ++device_index)
        {
            if (requested_device >= 0 && device_index != static_cast<size_t>(requested_device)) continue;
            const auto candidate = devices[device_index];
            VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(candidate, &p);
            if (p.limits.maxPerStageDescriptorStorageBuffers < 13 || p.limits.maxDescriptorSetStorageBuffers < 13) continue;
            uint32_t n = 0; vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
            std::vector<VkQueueFamilyProperties> queues(n); vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, queues.data());
            for (uint32_t i = 0; i < n; ++i) if (queues[i].queueCount && (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) { physical = candidate; family = i; props = p; break; }
            if (physical) break;
        }
        if (!physical) { std::cout << "no suitable GPU particle device\n"; return 77; }
        std::cout << "GPU particle device: " << props.deviceName << '\n';
        const float priority = 1.f; VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        q.queueFamilyIndex = family; q.queueCount = 1; q.pQueuePriorities = &priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &q;
        VkPhysicalDeviceDescriptorIndexingFeatures indexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
        bool bindless = false;
        if (app.apiVersion >= VK_API_VERSION_1_2 && props.apiVersion >= VK_API_VERSION_1_2)
        {
            VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            features.pNext = &indexing;
            vkGetPhysicalDeviceFeatures2(physical, &features);
            bindless = indexing.runtimeDescriptorArray && indexing.shaderSampledImageArrayNonUniformIndexing &&
                indexing.descriptorBindingSampledImageUpdateAfterBind && indexing.descriptorBindingPartiallyBound;
            if (bindless)
                di.pNext = &indexing;
        }
        require(vkCreateDevice(physical, &di, nullptr, &f.device) == VK_SUCCESS, "device");
        VkQueue queue{}; vkGetDeviceQueue(f.device, family, 0, &queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex = family;
        require(vkCreateCommandPool(f.device, &pi, nullptr, &f.pool) == VK_SUCCESS, "pool");
        VkPhysicalDeviceMemoryProperties memory{}; vkGetPhysicalDeviceMemoryProperties(physical, &memory);
        BufferResourceDispatch vk{vkCreateBuffer, vkDestroyBuffer, vkGetBufferMemoryRequirements, vkAllocateMemory,
            vkFreeMemory, vkBindBufferMemory, vkMapMemory, vkUnmapMemory};
        if (collision_shader) exercise_collision_shader(physical, f.device, queue, f.pool, props.limits, memory, vk, collision_shader);
        std::string error;
        GpuParticleCompute compute;
        const auto source = [&](const std::string& name, std::vector<uint8_t>& bytes, std::string& why) {
            std::ifstream file(shader_directory + "/" + name, std::ios::binary | std::ios::ate);
            const auto size = file.tellg();
            if (!file || size <= 0) { why = "shader file unavailable: " + name; return false; }
            bytes.resize(size_t(size)); file.seekg(0); file.read(reinterpret_cast<char*>(bytes.data()), size);
            if (!file) { why = "shader read failed: " + name; return false; }
            return true;
        };
        exercise_smoke_shader(f.device, queue, f.pool, props.limits, memory, vk, source);
        exercise_visibility_raster(f.device, queue, f.pool, props.limits, memory, vk, source, bindless);
        std::cout << "Visibility material resolve " << (bindless ? "passed" : "skipped (no descriptor indexing)") << '\n';
        require(compute.initialize_from_source(f.device, vkGetDeviceProcAddr, props.limits, 3, VK_NULL_HANDLE, source, error), error);
        GpuParticleBuffers buffers;
        GpuParticleBuffers::Capacity capacity{8, 5, 1, 2, 1, 1, 2, 514, 48};
        require(buffers.initialize(f.device, props.limits, memory, vk, capacity, error), error);
        GpuParticleUniforms uniforms;
        require(uniforms.initialize(f.device, props.limits, memory, vk, 8, error), error);
        GpuParticleUpload upload;
        require(upload.initialize(f.device, vkGetDeviceProcAddr, 16384, memory, vk, error), error);
        GpuParticleFrame frame; require(frame.initialize(f.device, vkGetDeviceProcAddr, error), error);
        GpuParticleReadback readback;
        require(readback.initialize(f.device, vkGetDeviceProcAddr, 2, memory, vk, error), error);
        BufferResource draw;
        require(draw.initialize(f.device, 112 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error), error);
        ParticleEffectDef definition; definition.name = "gpu_test"; definition.max_particles = 2;
        definition.actions = {0, 0, 0, 0};
        GpuParticlePrograms programs; uint32_t program_id{};
        require(programs.intern(definition, program_id, error), error);
        fg::GpuPapiProgram program = programs.programs()[program_id];
        GpuParticleCommands commands(2); GpuParticleCommands::Batch batch;
        require(commands.begin_generation(0, error) && commands.enqueue(0, fg::GpuPapiCommand{}, error) &&
            commands.snapshot(batch, error), error);
        const std::vector<uint32_t> visible_roots{0};
        GpuParticleCollisionInput collision;
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = f.pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
        VkCommandBuffer cb{}; require(vkAllocateCommandBuffers(f.device, &ai, &cb) == VK_SUCCESS, "command buffer");
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        require(vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS, "begin");
        GpuParticleFrameInputResult inputs;
        auto invalid_batch = batch;
        invalid_batch.ranges[0][1] = 2;
        require(!upload_gpu_particle_frame(cb, buffers, upload, programs, invalid_batch, visible_roots, &collision,
            true, true, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, inputs, error),
            "invalid command range accepted");
        require(!upload_gpu_particle_frame(cb, buffers, upload, programs, batch, {0, 0}, &collision,
            true, true, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, inputs, error),
            "duplicate visible root accepted");
        const uint32_t tail_child = capacity.roots * 2;
        invalid_batch = batch;
        invalid_batch.commands[0].emitter = tail_child;
        require(!upload_gpu_particle_frame(cb, buffers, upload, programs, invalid_batch, visible_roots, &collision,
            true, true, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, inputs, error),
            "command accepted for a child slot beyond the root range");
        require(!upload_gpu_particle_frame(cb, buffers, upload, programs, batch, {tail_child}, &collision,
            true, true, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, inputs, error),
            "child slot beyond the root range accepted as a visible root");
        require(upload_gpu_particle_frame(cb, buffers, upload, programs, batch, visible_roots, &collision,
            true, true, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, inputs, error,
            VK_NULL_HANDLE, 0, FrameSubmissionGraph::Queue::Compute,
            FrameSubmissionGraph::Queue::Copy, true), error);
        fg::GpuPapiConstants constants{}; constants.particleCapacity = 8; constants.emitterCapacity = capacity.emitters;
        constants.rootCapacity = capacity.roots;
        constants.actionStateCapacity = 1; constants.commandCount = inputs.command_count;
        constants.programCount = inputs.program_count;
        const float planes[24]{};
        require(frame.record(cb, compute, buffers, uniforms, 0, constants, inputs.pending_roots,
            inputs.visible_roots, planes, true, error,
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, false, nullptr,
            FrameSubmissionGraph::InvalidPass, FrameSubmissionGraph::InvalidPass, 0,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            FrameSubmissionGraph::Queue::Compute, VK_QUEUE_GRAPHICS_BIT,
            {}, {}, &upload, &inputs.uploads, false), error);
        require(readback.record(cb, buffers, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, error,
            FrameSubmissionGraph::Queue::Compute), error);
        VkBufferCopy copy{0, 0, 28 * 16};
        vkCmdCopyBuffer(cb, buffers.get(GpuParticleBuffers::DrawArguments).handle(), draw.handle(), 1, &copy);
        VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        require(vkEndCommandBuffer(cb) == VK_SUCCESS, "end");
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO}; submit.commandBufferCount = 1; submit.pCommandBuffers = &cb;
        require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS, "submit");
        uint32_t draw_words[112 * 4]{};
        GpuParticleReadback::Result result;
        require(readback.read(result, error) && draw.read(0, draw_words, 28 * 16, error), error);
        require(result.statuses[0].generation == 1 && result.statuses[0].serial == 1 && result.errors == 0, "simulation status");
        require(commands.acknowledge(result.statuses, error) && commands.snapshot(batch, error) && batch.commands.empty(), error);
        for (size_t i = 0; i < 28; ++i) require(draw_words[i * 4] == 6 && draw_words[i * 4 + 1] == 0, "draw arguments");
        for (const uint32_t scenario : {0u, 1u, 2u})
        {
            // Test a full parent, then a parent that can emit two particles
            // with zero free child slots and a single free slot beyond the root range.
            fg::GpuPapiEmitter full_emitters[5]{};
            auto& root = full_emitters[0];
            root.active = root.generation = 1;
            root.completedSerial = 1;
            root.flags = GPU_PAPI_PLAYING;
            root.capacity = 2;
            root.count = scenario == 0 ? 2 : 0;
            root.childBirth = 0;
            root.childPlay = root.childDead = root.pad0 = GPU_PAPI_INVALID;
            full_emitters[1].active = 1;
            full_emitters[3].active = 1;
            full_emitters[4].active = scenario == 2 ? 0 : 1;
            fg::GpuPapiProgram source_program = program;
            source_program.actionCount = 1;
            fg::GpuPapiAction source_action{};
            source_action.type = 21;
            source_action.stateIndex = GPU_PAPI_INVALID;
            source_action.p0.y = 100.f;
            fg::GpuPapiCommand tick{};
            tick.type = 6;
            tick.generation = 1;
            tick.serial = 2;
            tick.p0 = 33;
            const uint32_t tick_range[]{0, 1};
            const std::vector<GpuParticleUpload::Update> tick_updates{
                {&buffers.get(GpuParticleBuffers::Programs), 0, &source_program, sizeof(source_program)},
                {&buffers.get(GpuParticleBuffers::Actions), 0, &source_action, sizeof(source_action)},
                {&buffers.get(GpuParticleBuffers::Emitters), 0, full_emitters, sizeof(full_emitters)},
                {&buffers.get(GpuParticleBuffers::Commands), 0, &tick, sizeof(tick)},
                {&buffers.get(GpuParticleBuffers::RootCommands), 0, tick_range, sizeof(tick_range)}};
            require(vkAllocateCommandBuffers(f.device, &ai, &cb) == VK_SUCCESS &&
                vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS, "full emitter tick begin");
            require(upload.record(cb, tick_updates, VK_QUEUE_TRANSFER_BIT, error,
                FrameSubmissionGraph::Queue::Copy), error);
            uniforms.reset();
            require(frame.record(cb, compute, buffers, uniforms, 0, constants, 1, 0, planes, false, error), error);
            require(readback.record(cb, buffers, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, error), error);
            submit.pCommandBuffers = &cb;
            require(vkEndCommandBuffer(cb) == VK_SUCCESS &&
                vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS &&
                vkQueueWaitIdle(queue) == VK_SUCCESS, "full emitter tick submit");
            require(readback.read(result, error), error);
            require(result.statuses[0].serial == 2 && !result.statuses[0].retired &&
                result.statuses[0].playing && result.statuses[0].count == (scenario == 2 ? 4 : 2),
                "child pressure interrupted the root tick or lost an allocatable child");
            require(result.errors == (scenario == 0 ? 0u : 2u), "unexpected child allocation diagnostics");
        }
        uniforms.reset();
        require(buffers.resize_draw_buckets(f.device, 112, props.limits.maxStorageBufferRange, memory, vk, error), error);
        fg::GpuPapiEmitter emitters[3]{}; emitters[0].active = 1; emitters[0].flags = GPU_PAPI_HUD;
        emitters[0].pad0 = 1; emitters[0].count = 1;
        emitters[2] = emitters[0]; emitters[2].program = 1; emitters[2].particleFirst = 1;
        emitters[2].pad0 = GPU_PAPI_INVALID;
        emitters[1] = emitters[0]; emitters[1].pad0 = GPU_PAPI_INVALID; emitters[1].particleFirst = 2;
        program.flags = 1; program.materialID = 0;
        fg::GpuPapiProgram draw_programs[]{program, program}; draw_programs[1].materialID = 1;
        draw_programs[1].flags |= 1u << 10; draw_programs[1].frameCount = 2; draw_programs[1].frameDimX = 2;
        draw_programs[1].frameSize.set(.5f, 1.f);
        fg::GpuPapiParticle particles[3]{};
        // This root is outside the clip volume until projected onto water.
        particles[0].position.set(-.5f, 0.f, 2.f); particles[0].size.set(.5f, .5f, 0.f); particles[0].color = 0xffff0000u;
        particles[1].position.set(.5f, 0.f, .5f); particles[1].size.set(.5f, .5f, 0.f); particles[1].color = 0xff00ff00u;
        particles[1].emitter = 2; particles[1].program = 1; particles[1].frame = 510;
        particles[2] = particles[0]; particles[2].emitter = 1;
        particles[2].position.set(0.f, .6f, .5f); particles[2].color = 0xff0000ffu;
        const uint32_t two_visible_roots[]{0, 2};
        const std::vector<GpuParticleUpload::Update> compact_updates{
            {&buffers.get(GpuParticleBuffers::Programs), 0, draw_programs, sizeof(draw_programs)},
            {&buffers.get(GpuParticleBuffers::Emitters), 0, emitters, sizeof(emitters)},
            {&buffers.get(GpuParticleBuffers::Particles), 0, particles, sizeof(particles)},
            {&buffers.get(GpuParticleBuffers::VisibleRoots), 0, two_visible_roots, sizeof(two_visible_roots)}};
        BufferResource index;
        require(index.initialize(f.device, 12, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error), error);
        require(vkAllocateCommandBuffers(f.device, &ai, &cb) == VK_SUCCESS, "second command buffer");
        require(vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS &&
            upload.record(cb, compact_updates, VK_QUEUE_TRANSFER_BIT, error,
                FrameSubmissionGraph::Queue::Copy), error);
        constants.programCount = 2;
        require(frame.record(cb, compute, buffers, uniforms, 0, constants, 0, 2, planes, false, error), error);
        copy.size = 112 * 16;
        vkCmdCopyBuffer(cb, buffers.get(GpuParticleBuffers::DrawArguments).handle(), draw.handle(), 1, &copy);
        copy.size = 12;
        vkCmdCopyBuffer(cb, buffers.get(GpuParticleBuffers::Indices).handle(), index.handle(), 1, &copy);
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
        require(vkEndCommandBuffer(cb) == VK_SUCCESS, "second end");
        submit.pCommandBuffers = &cb;
        require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS, "second submit");
        uint32_t index_values[]{UINT32_MAX, UINT32_MAX, UINT32_MAX};
        require(draw.read(0, draw_words, sizeof(draw_words), error) && index.read(0, index_values, sizeof(index_values), error), error);
        for (size_t i = 0; i < 112; ++i)
            require(draw_words[i * 4] == 6 && draw_words[i * 4 + 1] == (i == 6 || i == 20 || i == 90 ? 1u : 0u), "compacted draw arguments");
        require(index_values[0] == 0 && index_values[1] == 2 && index_values[2] == 1, "root and child particle indices");
        FrameBufferPool buffer_pool;
        require(buffer_pool.initialize(f.device, memory, vk), "particle frame buffer pool");
        {
            constexpr auto memory_flags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            BufferResource* broad{};
            BufferResource* narrow{};
            const auto broad_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            require(buffer_pool.acquire(256, broad_usage, memory_flags, false, broad, error), error);
            require(buffer_pool.acquire(256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, memory_flags, false, narrow, error), error);
            require(broad != narrow && buffer_pool.release(broad) && buffer_pool.release(narrow), "pool releases");
            BufferResource* reused{};
            require(buffer_pool.acquire(256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, memory_flags, false, reused, error), error);
            require(reused == narrow, "pool prefers exact usage");
            require(buffer_pool.acquire(256, VK_BUFFER_USAGE_TRANSFER_DST_BIT, memory_flags, false, reused, error), error);
            require(reused == broad, "pool reuses available compatible usage");
            require(buffer_pool.release(narrow) && buffer_pool.release(broad), "pool releases reused buffers");
        }
        GpuParticleRuntime runtime;
        require(runtime.initialize(f.device, vkGetDeviceProcAddr, props.limits, memory, vk, capacity,
            1, 16384, VK_NULL_HANDLE, source, error, &buffer_pool), error);
        GpuParticleEmitters emitter_manager(runtime.programs(), runtime.commands(), capacity.roots);
        GpuParticleEmitters::Handle handle{};
        require(emitter_manager.create(definition, handle, error), error);
        require(vkAllocateCommandBuffers(f.device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS, "runtime begin");
        const VkQueueFlags frame_queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
        {
            FrameImagePool image_pool;
            require(image_pool.initialize(f.device, memory, {vkCreateImage, vkDestroyImage,
                vkGetImageMemoryRequirements, vkAllocateMemory, vkFreeMemory, vkBindImageMemory}), "transient image pool");
            FrameImageGraph graph;
            FrameImageGraph::BufferResource transient, written, unused;
            const FrameImageGraph::BufferDescription readback_description{64, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, false};
            require(graph.declare_buffer(readback_description, transient, error) &&
                graph.declare_buffer(128, VK_BUFFER_USAGE_TRANSFER_DST_BIT, unused, error), error);
            VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            image_info.imageType = VK_IMAGE_TYPE_2D;
            image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
            image_info.extent = {4, 4, 1};
            image_info.mipLevels = image_info.arrayLayers = 1;
            image_info.samples = VK_SAMPLE_COUNT_1_BIT;
            image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
            FrameImageGraph::Resource image, cleared;
            require(graph.declare_image(image_info, VK_IMAGE_ASPECT_COLOR_BIT, image, error), error);
            const VkImageSubresourceRange color_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            auto clear = graph.add_command_pass("clear transient image", [&](VkCommandBuffer command) {
                VkClearColorValue red{{1.f, 0.f, 0.f, 1.f}};
                vkCmdClearColorImage(command, graph.image_handle(cleared),
                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &red, 1, &color_range);
            });
            auto fill = graph.add_command_pass("copy transient image to buffer", [&](VkCommandBuffer command) {
                const VkBufferImageCopy region{0, 0, 0, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                    {0, 0, 0}, {4, 4, 1}};
                vkCmdCopyImageToBuffer(command, graph.image_handle(cleared), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    graph.buffer_handle(written), 1, &region);
            });
            auto host = graph.add_pass("read transient buffer", {});
            require(graph.write(clear, image, ImageUse::TransferDestination, cleared, error) &&
                graph.read(fill, cleared, ImageUse::TransferSource, error) &&
                graph.write_buffer(fill, transient, BufferUse::TransferDestination, written, error) &&
                graph.read_buffer(host, written, BufferUse::HostRead, error), error);
            require(graph.materialize_images(image_pool, error), error);
            const VkImage materialized_image = graph.image_handle(image);
            require(graph.materialize_buffers(buffer_pool, error), error);
            const VkBuffer materialized = graph.buffer_handle(written);
            require(materialized && !graph.buffer_handle(unused), "unexpected transient buffer allocation");
            VkCommandBuffer transient_command{};
            require(vkAllocateCommandBuffers(f.device, &ai, &transient_command) == VK_SUCCESS &&
                vkBeginCommandBuffer(transient_command, &begin) == VK_SUCCESS, "transient buffer begin");
            ImageStateTracker states;
            require(graph.execute(transient_command, states, {vkCmdPipelineBarrier}, error,
                frame_queue_flags), error);
            require(!states.has_image(graph.image_handle(image)), "transient image leaked into imported image state");
            require(vkEndCommandBuffer(transient_command) == VK_SUCCESS, "transient buffer end");
            xray::render::framegraph::ResourceRetirement retirement(2);
            graph.retire_allocations(retirement, 1u << 1);
            retirement.complete(0);
            require(retirement.size() == 2 && !graph.buffer_handle(written) && !graph.image_handle(image),
                "transient resource ownership handoff");
            VkImage other_image{};
            require(image_pool.acquire(image_info, other_image, error), error);
            require(other_image != materialized_image, "transient image recycled before frame completion");
            VkSubmitInfo transient_submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            transient_submit.commandBufferCount = 1;
            transient_submit.pCommandBuffers = &transient_command;
            require(vkQueueSubmit(queue, 1, &transient_submit, VK_NULL_HANDLE) == VK_SUCCESS &&
                vkQueueWaitIdle(queue) == VK_SUCCESS, "transient buffer submit");
            vkFreeCommandBuffers(f.device, f.pool, 1, &transient_command);
            retirement.complete(1);
            require(retirement.size() == 0, "transient buffer retirement");
            VkImage reused_image{};
            require(image_pool.acquire(image_info, reused_image, error), error);
            require(reused_image == materialized_image && image_pool.release(reused_image),
                "transient image was not recycled after frame completion");
            require(image_pool.release(other_image), "release second pooled image");
            BufferResource* physical = nullptr;
            require(buffer_pool.acquire(readback_description.size, readback_description.usage,
                readback_description.memory_properties, readback_description.pooled_memory, physical, error), error);
            uint32_t result{};
            require(physical->handle() == materialized &&
                physical->read(0, &result, sizeof(result), error) && result == 0xff0000ffu, "transient image result");
            require(buffer_pool.release(physical), "release transient readback buffer");
        }
        FrameSubmissionGraph runtime_submission_graph;
        const auto runtime_continuation = runtime_submission_graph.add_pass(
            "runtime graphics continuation", FrameSubmissionGraph::Queue::Graphics, family);
        struct GeneratedCommandList
        {
            FrameSubmissionGraph::Pass pass{};
            FrameSubmissionGraph::Queue queue{};
            VkCommandBuffer command{};
        };
        std::vector<GeneratedCommandList> runtime_command_lists;
        const GpuParticleFrame::RequestSubmissionCommandBuffer request_generated =
            [&](FrameSubmissionGraph::Pass pass, FrameSubmissionGraph::Queue command_queue,
                VkCommandBuffer& command_buffer, std::string& request_error) {
                command_buffer = VK_NULL_HANDLE;
                VkCommandBuffer generated{};
                if (vkAllocateCommandBuffers(f.device, &ai, &generated) != VK_SUCCESS)
                { request_error = "could not allocate GPU-particle graph command buffer"; return false; }
                if (vkBeginCommandBuffer(generated, &begin) != VK_SUCCESS)
                { request_error = "could not begin GPU-particle graph command buffer"; return false; }
                runtime_command_lists.push_back({pass, command_queue, generated});
                command_buffer = generated;
                request_error.clear();
                return true;
            };
        require(runtime.record(0, cb, visible_roots, &collision, planes, error,
            frame_queue_flags, false, &runtime_submission_graph,
            FrameSubmissionGraph::InvalidPass, FrameSubmissionGraph::InvalidPass,
            runtime_continuation, 0, true, family, family, family,
            FrameSubmissionGraph::Queue::Compute, frame_queue_flags, request_generated), error);
        for (const GeneratedCommandList& generated : runtime_command_lists)
            require(runtime_submission_graph.depends_on(runtime_continuation, generated.pass, error), error);
        require(std::any_of(runtime_command_lists.begin(), runtime_command_lists.end(),
            [](const GeneratedCommandList& generated) {
                return generated.queue == FrameSubmissionGraph::Queue::Copy;
            }), "GPU-particle runtime omitted its generated Copy upload group");
        require(std::any_of(runtime_command_lists.begin(), runtime_command_lists.end(),
            [](const GeneratedCommandList& generated) {
                return generated.queue == FrameSubmissionGraph::Queue::Compute;
            }), "GPU-particle runtime omitted its generated Compute group");
        require(!runtime.record(0, cb, visible_roots, &collision, planes, error), "busy frame slot accepted");
        uint32_t recorded_bucket = UINT32_MAX;
        require(runtime.draw_bucket(0, 0, 0, 0, 0, false, recorded_bucket) && recorded_bucket == 0 &&
            !runtime.draw_bucket(0, 2, 0, 0, 0, false, recorded_bucket), "recorded root mapping");
        auto another_definition = definition; another_definition.name = "gpu_test_second";
        uint32_t another_program{};
        require(runtime.programs().intern(another_definition, another_program, error) && another_program == 1 &&
            !runtime.draw_bucket(0, 0, 1, 0, 0, false, recorded_bucket), "frame program table must remain stable");
        require(runtime.programs().set_material(0, 0, 0, 0, error), error);
        require(vkEndCommandBuffer(cb) == VK_SUCCESS, "runtime end");
        std::vector<VkCommandBuffer> runtime_submit_commands;
        std::vector<FrameSubmissionGraph::Pass> runtime_submission_order;
        require(runtime_submission_graph.execution_order(runtime_submission_order, error), error);
        for (FrameSubmissionGraph::Pass submission : runtime_submission_order)
        {
            const auto generated = std::find_if(runtime_command_lists.begin(), runtime_command_lists.end(),
                [submission](const GeneratedCommandList& item) { return item.pass == submission; });
            if (generated == runtime_command_lists.end())
                continue; // The final continuation is represented by the game renderer.
            require(vkEndCommandBuffer(generated->command) == VK_SUCCESS,
                "GPU-particle generated command end");
            runtime_submit_commands.push_back(generated->command);
        }
        require(runtime_submit_commands.size() == runtime_command_lists.size(),
            "GPU-particle generated submission order is incomplete");
        submit.commandBufferCount = 1;
        for (VkCommandBuffer generated : runtime_submit_commands)
        {
            submit.pCommandBuffers = &generated;
            require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS,
                "GPU-particle generated queue submit");
            require(vkQueueWaitIdle(queue) == VK_SUCCESS, "GPU-particle generated queue wait");
        }
        require(runtime.submitted(0, error) && runtime.programs().dirty(), "lost program update");
        require(vkQueueWaitIdle(queue) == VK_SUCCESS, "runtime wait");
        require(runtime.complete(0, result, error) && result.statuses[0].serial == 1 && result.errors == 0, error);
        require(!runtime.draw_bucket(0, 0, 0, 0, 0, false, recorded_bucket), "completed frame retained draw mapping");
        fg::GpuPapiStatus snapshot{};
        require(emitter_manager.apply(result.statuses, error) && emitter_manager.snapshot(handle, snapshot), error);
        require(runtime.commands().snapshot(batch, error) && batch.commands.empty(), "runtime command acknowledgement");
        require(runtime.draw_resize_needed(2), "ordered buckets must grow for another program");
        require(runtime.reserve_draw_buckets(f.device, props.limits, memory, vk, 2, error) &&
            !runtime.draw_resize_needed(2), error);
        collision.nodes.resize(1); collision.nodes[0].escape = 1; collision.nodes[0].count = 1;
        collision.static_triangles.resize(1);
        require(runtime.collision_resize_needed(144) && runtime.reserve_collision(f.device, vkGetDeviceProcAddr,
            props.limits, memory, vk, 144, error) && !runtime.collision_resize_needed(144), error);
        require(emitter_manager.destroy(handle, error) && !emitter_manager.snapshot(handle, snapshot), error);
        require(vkAllocateCommandBuffers(f.device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS,
            "retire begin");
        require(runtime.record(0, cb, {}, &collision, planes, error) && vkEndCommandBuffer(cb) == VK_SUCCESS, error);
        submit.pCommandBuffers = &cb;
        require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && runtime.submitted(0, error) &&
            vkQueueWaitIdle(queue) == VK_SUCCESS && runtime.complete(0, result, error) &&
            emitter_manager.apply(result.statuses, error), error);
        require(result.statuses[0].retired == 1, "emitter retirement");
        GpuParticleEmitters::Handle next_handle{};
        require(emitter_manager.create(definition, next_handle, error) && next_handle != handle &&
            !emitter_manager.play(handle, error), "stale emitter handle");
        runtime.destroy();
        if (props.limits.maxPushConstantsSize >= 64)
        {
            VkDescriptorSetLayout material_layout{};
            VkDescriptorSetLayoutBinding material_binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
            VkDescriptorSetLayoutBinding material_bindings[]{material_binding, material_binding};
            material_bindings[1].binding = 36; // material_parallax
            VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout_info.bindingCount = 2; layout_info.pBindings = material_bindings;
            require(vkCreateDescriptorSetLayout(f.device, &layout_info, nullptr, &material_layout) == VK_SUCCESS, "material layout");
            VkDescriptorSetLayout scene_layout{};
            VkDescriptorSetLayoutBinding scene_bindings[]{material_binding, material_binding};
            scene_bindings[1].binding = 2;
            layout_info.bindingCount = 2; layout_info.pBindings = scene_bindings;
            require(vkCreateDescriptorSetLayout(f.device, &layout_info, nullptr, &scene_layout) == VK_SUCCESS, "scene layout");
            VkAttachmentDescription attachments[2]{};
            attachments[0].format = VK_FORMAT_R8G8B8A8_UNORM; attachments[1].format = VK_FORMAT_D32_SFLOAT;
            for (auto& attachment : attachments) { attachment.samples = VK_SAMPLE_COUNT_1_BIT;
                attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
                attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED; }
            attachments[0].finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, depth{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
            VkSubpassDescription subpass{}; subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1; subpass.pColorAttachments = &color; subpass.pDepthStencilAttachment = &depth;
            VkSubpassDependency readback_dependency{};
            readback_dependency.srcSubpass = 0;
            readback_dependency.dstSubpass = VK_SUBPASS_EXTERNAL;
            readback_dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            readback_dependency.dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
            readback_dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            readback_dependency.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rp.attachmentCount = 2; rp.pAttachments = attachments; rp.subpassCount = 1; rp.pSubpasses = &subpass;
            rp.dependencyCount = 1;
            rp.pDependencies = &readback_dependency;
            VkRenderPass render_pass{};
            require(vkCreateRenderPass(f.device, &rp, nullptr, &render_pass) == VK_SUCCESS, "particle render pass");
            GpuParticleGraphics graphics;
            auto mobile_limits = props.limits;
            mobile_limits.maxPushConstantsSize = 128; // Vulkan's guaranteed minimum
            require(graphics.initialize(f.device, vkGetDeviceProcAddr, mobile_limits, memory, vk, material_layout,
                VK_NULL_HANDLE, 1, 3, source, error, VK_SAMPLE_COUNT_1_BIT, render_pass) &&
                graphics.begin_frame(0, error), error);
            SmokeTrailCompute smoke_graph;
            SmokeTrailDrawPass smoke_draw;
            SmokeTrailDrawPass trail_draw;
            SmokeTrailNoise noise_graph;
            require(smoke_graph.initialize(f.device, vkGetDeviceProcAddr, mobile_limits, memory, vk, 1,
                VK_NULL_HANDLE, source, error) && smoke_draw.initialize(f.device, vkGetDeviceProcAddr, mobile_limits,
                memory, vk, render_pass, VK_NULL_HANDLE, 1, source, error) &&
                trail_draw.initialize(f.device, vkGetDeviceProcAddr, mobile_limits, memory, vk, render_pass,
                    VK_NULL_HANDLE, 1, source, error, true) &&
                noise_graph.initialize(f.device, vkGetDeviceProcAddr, mobile_limits, memory, vk, 8, 1,
                    VK_NULL_HANDLE, source, error), error);
            struct Images
            {
                VkDevice device{}; std::vector<VkImage> images; std::vector<VkDeviceMemory> memory;
                std::vector<VkImageView> views; VkSampler sampler{}; VkDescriptorPool pool{}; VkFramebuffer framebuffer{};
                ~Images() { if (framebuffer) vkDestroyFramebuffer(device, framebuffer, nullptr);
                    if (pool) vkDestroyDescriptorPool(device, pool, nullptr);
                    if (sampler) vkDestroySampler(device, sampler, nullptr);
                    for (auto view : views) vkDestroyImageView(device, view, nullptr);
                    for (auto image : images) vkDestroyImage(device, image, nullptr);
                    for (auto allocation : memory) vkFreeMemory(device, allocation, nullptr); }
            } images;
            images.device = f.device;
            const auto create_image = [&](VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage,
                                          VkImageAspectFlags aspect) {
                VkImageCreateInfo ii{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO}; ii.imageType = VK_IMAGE_TYPE_2D;
                ii.format = format; ii.extent = {width, height, 1}; ii.mipLevels = ii.arrayLayers = 1;
                ii.samples = VK_SAMPLE_COUNT_1_BIT; ii.tiling = VK_IMAGE_TILING_OPTIMAL; ii.usage = usage;
                VkImage image{}; require(vkCreateImage(f.device, &ii, nullptr, &image) == VK_SUCCESS, "particle image");
                images.images.push_back(image); VkMemoryRequirements req{}; vkGetImageMemoryRequirements(f.device, image, &req);
                uint32_t type = 0;
                while (type < memory.memoryTypeCount && (!(req.memoryTypeBits & (1u << type)) ||
                    !(memory.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) ++type;
                require(type < memory.memoryTypeCount, "particle image memory type");
                VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mi.allocationSize = req.size; mi.memoryTypeIndex = type;
                VkDeviceMemory allocation{}; require(vkAllocateMemory(f.device, &mi, nullptr, &allocation) == VK_SUCCESS, "particle image memory");
                images.memory.push_back(allocation); require(vkBindImageMemory(f.device, image, allocation, 0) == VK_SUCCESS, "particle image bind");
                VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; vi.image = image;
                vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
                vi.format = format; vi.subresourceRange = {aspect, 0, 1, 0, 1}; VkImageView view{};
                require(vkCreateImageView(f.device, &vi, nullptr, &view) == VK_SUCCESS, "particle image view");
                images.views.push_back(view); return image;
            };
            VkImage color_image = create_image(VK_FORMAT_R8G8B8A8_UNORM, 32, 32,
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            create_image(VK_FORMAT_D32_SFLOAT, 32, 32, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_ASPECT_DEPTH_BIT);
            VkImage texture = create_image(VK_FORMAT_R8G8B8A8_UNORM, 2, 1,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            VkImage trail_texture = create_image(VK_FORMAT_R8G8B8A8_UNORM, 1, 1,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            VkFramebufferCreateInfo fi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO}; fi.renderPass = render_pass;
            fi.attachmentCount = 2; fi.pAttachments = images.views.data(); fi.width = fi.height = 32; fi.layers = 1;
            require(vkCreateFramebuffer(f.device, &fi, nullptr, &images.framebuffer) == VK_SUCCESS, "particle framebuffer");
            VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO}; si.magFilter = si.minFilter = VK_FILTER_NEAREST;
            si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST; si.addressModeU = si.addressModeV = si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            require(vkCreateSampler(f.device, &si, nullptr, &images.sampler) == VK_SUCCESS, "particle sampler");
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
            VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 2;
            dpi.poolSizeCount = 1; dpi.pPoolSizes = &pool_size;
            require(vkCreateDescriptorPool(f.device, &dpi, nullptr, &images.pool) == VK_SUCCESS, "particle material pool");
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = images.pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &material_layout;
            VkDescriptorSet material{}; require(vkAllocateDescriptorSets(f.device, &dai, &material) == VK_SUCCESS, "particle material set");
            VkDescriptorImageInfo dii{images.sampler, images.views[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet dw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; dw.dstSet = material; dw.dstBinding = 0;
            dw.descriptorCount = 1; dw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; dw.pImageInfo = &dii;
            VkWriteDescriptorSet material_writes[]{dw, dw};
            material_writes[1].dstBinding = material_bindings[1].binding;
            vkUpdateDescriptorSets(f.device, 2, material_writes, 0, nullptr);
            dai.pSetLayouts = &scene_layout;
            VkDescriptorSet scene{}; require(vkAllocateDescriptorSets(f.device, &dai, &scene) == VK_SUCCESS, "particle scene set");
            VkWriteDescriptorSet scene_writes[]{dw, dw};
            for (auto& write : scene_writes) write.dstSet = scene;
            scene_writes[1].dstBinding = 2;
            vkUpdateDescriptorSets(f.device, 2, scene_writes, 0, nullptr);
            BufferResource white, image_readback;
            require(white.initialize(f.device, 12, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error), error);
            const uint32_t atlas_pixels[]{0xffffffffu, 0xff0000ffu, 0xff004080u};
            require(white.write(0, atlas_pixels, sizeof(atlas_pixels), error), error);
            require(image_readback.initialize(f.device, 32 * 32 * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, vk, error), error);
            GpuParticleDrawParams draw_params{}; draw_params.camera_top[1] = 1.f; draw_params.camera_right[0] = 1.f;
            draw_params.camera_direction[2] = 1.f; draw_params.eye_position[2] = -1.f; draw_params.bucket = 6;
            draw_params.screen = {1.f / 32, 1.f / 32, 0.f, 0.f};
            draw_params.water_origin = {0.f, 0.f, .5f, 1.f}; draw_params.water_normal = {0.f, 0.f, 1.f, 0.f};
            float material_data[64]{}; material_data[32] = 1.f; material_data[37] = 1.f;
            const float mvp[]{1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            require(vkAllocateCommandBuffers(f.device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS, "particle draw begin");
            VkImageMemoryBarrier image_barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_barrier.srcQueueFamilyIndex = image_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            image_barrier.image = texture; image_barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
            VkBufferImageCopy pixel{}; pixel.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; pixel.imageExtent = {2,1,1};
            vkCmdCopyBufferToImage(cb, white.handle(), texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &pixel);
            image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
            image_barrier.image = trail_texture; image_barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            image_barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            image_barrier.srcAccessMask = 0; image_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
            pixel.bufferOffset = 8; pixel.imageExtent = {1,1,1};
            vkCmdCopyBufferToImage(cb, white.handle(), trail_texture, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &pixel);
            image_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; image_barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            image_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; image_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image_barrier);
            require(noise_graph.record(cb, 0, .25f, 4.f, true, error), error);
            SmokeTrailEmitParams smoke_emit{};
            smoke_emit.previous_y = -.6f; smoke_emit.previous_z = .2f;
            smoke_emit.current_y = -.2f; smoke_emit.current_z = .8f;
            smoke_emit.lifetime = 2.f; smoke_emit.count = 2; smoke_emit.max_points = 256;
            SmokeTrailSimParams smoke_sim{}; smoke_sim.dt = .1f; smoke_sim.heat = 1.f; smoke_sim.max_points = 256;
            SmokeTrailCompactParams smoke_compact{}; smoke_compact.max_points = 256; smoke_compact.subdivisions = 12;
            require(smoke_graph.record(cb, 0, smoke_emit, smoke_sim, smoke_compact, true, error), error);
            VkClearValue clears[2]{}; clears[1].depthStencil = {1.f, 0};
            VkRenderPassBeginInfo rb{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO}; rb.renderPass = render_pass;
            rb.framebuffer = images.framebuffer; rb.renderArea.extent = {32,32}; rb.clearValueCount = 2; rb.pClearValues = clears;
            vkCmdBeginRenderPass(cb, &rb, VK_SUBPASS_CONTENTS_INLINE);
            FrameRecordingContext draw_frame{}; draw_frame.command_buffer = cb; draw_frame.render_pass = render_pass;
            draw_frame.framebuffer = images.framebuffer; draw_frame.extent = {32,32};
            require(graphics.record(draw_frame, buffers, material, "vk/gpu_particle_distortion.ps", 1, false, false,
                draw_params, material_data, mvp, error), error);
            draw_params.bucket = 90;
            draw_params.water_origin[3] = 0.f;
            require(graphics.record(draw_frame, buffers, material, "vk/gpu_particle_world_blended.ps", 1, false, false,
                draw_params, material_data, mvp, error), error);
            draw_params.bucket = 20;
            require(graphics.record(draw_frame, buffers, material, "vk/gpu_particle_hud_distortion.ps", 1, false, false,
                draw_params, material_data, mvp, error, scene_layout, scene), error);
            require(!graphics.record(draw_frame, buffers, material, "vk/gpu_particle_world_blended.ps", 1, false, false,
                draw_params, material_data, mvp, error), "draw capacity not enforced");
            SmokeTrailDrawParams smoke_params{}; smoke_params.texcoord_factor = 1.f; smoke_params.edge_policy = 1;
            smoke_params.sphere_radius = 3.f; smoke_params.turbulence = .00001f; smoke_params.frequency = 2.f;
            const float smoke_mvp[]{100,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
            require(smoke_draw.record(draw_frame, smoke_graph, noise_graph.view(), noise_graph.sampler(), smoke_params, smoke_mvp, error), error);
            std::vector<TrailState::Group> trail_groups(2);
            const auto make_point = [](float x, float y, float distance) {
                TrailControlPoint point{}; point.position[0] = x; point.position[1] = y; point.position[2] = .5f;
                point.direction[1] = .15f; point.age = .5f; point.distance = distance; return point;
            };
            trail_groups[0].points = {make_point(.5f, -.6f, 0.f), make_point(.9f, -.6f, .4f)};
            trail_groups[1].points = {make_point(-.9f, .4f, 0.f), make_point(-.5f, .4f, .4f)};
            trail_groups[0].color = 0xffff0000u; trail_groups[1].color = 0xff00ff00u;
            SmokeTrailDrawParams trail_params{}; trail_params.subdivisions = 12; trail_params.texcoord_factor = 1.f;
            trail_params.edge_policy = 1; trail_params.smoothing = 1; trail_params.use_gpu_state = 0;
            trail_params.camera_z = 3.f; trail_params.camera_facing = 1;
            require(trail_draw.record_trail(draw_frame, trail_groups, trail_params, mvp, images.views[3], images.sampler, error), error);
            vkCmdEndRenderPass(cb);
            VkMemoryBarrier color_barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; color_barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
            color_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &color_barrier, 0, nullptr, 0, nullptr);
            VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; region.imageExtent = {32,32,1};
            vkCmdCopyImageToBuffer(cb, color_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image_readback.handle(), 1, &region);
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
            require(vkEndCommandBuffer(cb) == VK_SUCCESS, "particle draw end"); submit.pCommandBuffers = &cb;
            require(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS && vkQueueWaitIdle(queue) == VK_SUCCESS, "particle draw submit");
            std::vector<uint8_t> pixels(32 * 32 * 4); require(image_readback.read(0, pixels.data(), pixels.size(), error), error);
            const auto pixel_a = (16 * 32 + 7) * 4, pixel_b = (16 * 32 + 24) * 4;
            const auto pixel_child = (6 * 32 + 15) * 4;
            const auto pixel_smoke = (22 * 32 + 16) * 4;
            const auto pixel_trail_a = (26 * 32 + 27) * 4, pixel_trail_b = (9 * 32 + 4) * 4;
            require(pixels[pixel_a] >= 75 && pixels[pixel_a] <= 95 &&
                pixels[pixel_a + 1] >= 75 && pixels[pixel_a + 1] <= 95 &&
                pixels[pixel_b] < 20 && pixels[pixel_b + 1] > 200 &&
                pixels[pixel_child] >= 75 && pixels[pixel_child] <= 95 && pixels[pixel_child + 2] < 20,
                "GPU water projection, world/HUD distortion and particle colors");
            require(pixels[pixel_smoke] > 80 && pixels[pixel_smoke + 1] > 80 && pixels[pixel_smoke + 2] > 80,
                "GPU smoke ribbon draw");
            require(pixels[pixel_trail_a] >= 55 && pixels[pixel_trail_a] <= 70 && pixels[pixel_trail_a + 1] < 10 &&
                pixels[pixel_trail_b] < 10 && pixels[pixel_trail_b + 1] >= 25 && pixels[pixel_trail_b + 1] <= 38,
                "GPU camera-facing textured trail draws");
            trail_draw.destroy(); smoke_draw.destroy(); smoke_graph.destroy(); noise_graph.destroy();
            graphics.destroy();
            vkDestroyFramebuffer(f.device, images.framebuffer, nullptr); images.framebuffer = VK_NULL_HANDLE;
            vkDestroyRenderPass(f.device, render_pass, nullptr);
            vkDestroyDescriptorSetLayout(f.device, scene_layout, nullptr);
            vkDestroyDescriptorSetLayout(f.device, material_layout, nullptr);
            std::cout << props.deviceName << ": GPU particle graphics pipeline passed\n";
        }
        else std::cout << props.deviceName << ": GPU particle graphics pipeline skipped (push constants)\n";
        std::cout << props.deviceName << ": GPU particle frame passed\n";
    }
    catch (const std::exception& ex) { std::cerr << ex.what() << '\n'; return 1; }
}
