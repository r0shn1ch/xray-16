#include "xrEngine/stdafx.h"
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
    smoke.destroy();
    SmokeTrailNoise noise;
    require(noise.initialize(device, vkGetDeviceProcAddr, limits, memory, vk, 8, 1, VK_NULL_HANDLE, source, error), error);
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
    std::cout << "GPU smoke trail compute passed\n";
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
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.apiVersion = VK_API_VERSION_1_1;
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
        require(compute.initialize_from_source(f.device, vkGetDeviceProcAddr, props.limits, 3, VK_NULL_HANDLE, source, error), error);
        GpuParticleBuffers buffers;
        GpuParticleBuffers::Capacity capacity{8, 4, 1, 2, 1, 1, 2, 514, 48};
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
        require(upload_gpu_particle_frame(cb, buffers, upload, programs, batch, visible_roots, &collision,
            true, true, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, inputs, error,
            VK_NULL_HANDLE, 0, FrameSubmissionGraph::Queue::Compute,
            FrameSubmissionGraph::Queue::Copy, true), error);
        fg::GpuPapiConstants constants{}; constants.particleCapacity = 8; constants.emitterCapacity = 4;
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
        GpuParticleRuntime runtime;
        require(runtime.initialize(f.device, vkGetDeviceProcAddr, props.limits, memory, vk, capacity,
            1, 16384, VK_NULL_HANDLE, source, error), error);
        GpuParticleEmitters emitter_manager(runtime.programs(), runtime.commands(), capacity.roots);
        GpuParticleEmitters::Handle handle{};
        require(emitter_manager.create(definition, handle, error), error);
        require(vkAllocateCommandBuffers(f.device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &begin) == VK_SUCCESS, "runtime begin");
        const VkQueueFlags frame_queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
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
            VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            layout_info.bindingCount = 1; layout_info.pBindings = &material_binding;
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
            VkRenderPassCreateInfo rp{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
            rp.attachmentCount = 2; rp.pAttachments = attachments; rp.subpassCount = 1; rp.pSubpasses = &subpass;
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
            VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3};
            VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets = 2;
            dpi.poolSizeCount = 1; dpi.pPoolSizes = &pool_size;
            require(vkCreateDescriptorPool(f.device, &dpi, nullptr, &images.pool) == VK_SUCCESS, "particle material pool");
            VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
            dai.descriptorPool = images.pool; dai.descriptorSetCount = 1; dai.pSetLayouts = &material_layout;
            VkDescriptorSet material{}; require(vkAllocateDescriptorSets(f.device, &dai, &material) == VK_SUCCESS, "particle material set");
            VkDescriptorImageInfo dii{images.sampler, images.views[2], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet dw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; dw.dstSet = material; dw.dstBinding = 0;
            dw.descriptorCount = 1; dw.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; dw.pImageInfo = &dii;
            vkUpdateDescriptorSets(f.device, 1, &dw, 0, nullptr);
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
