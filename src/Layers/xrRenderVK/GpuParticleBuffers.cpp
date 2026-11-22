#include "xrEngine/stdafx.h"
#include "GpuParticleBuffers.h"
#include "gpu_particles/gpu_particle_types.h"
#include <algorithm>

namespace xray::render::vulkan
{
bool GpuParticleBuffers::initialize(VkDevice device, const VkPhysicalDeviceLimits& limits,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
    const Capacity& capacity, std::string& error, FrameBufferPool* pool)
{
    destroy();
    if (!capacity.particles || !capacity.emitters || !capacity.programs || capacity.programs > UINT32_MAX / 28 || !capacity.roots ||
        capacity.roots > capacity.emitters / 2 || capacity.collision_bytes < 48 || capacity.collision_bytes % 4 ||
        capacity.noise_values < 514)
    { error = "invalid GPU particle buffer capacity"; return false; }
    device_ = device;
    memory_ = memory;
    dispatch_ = dispatch;
    pool_ = pool;
    using namespace xray::render::fg;
    const VkDeviceSize sizes[Count]{
        VkDeviceSize(capacity.particles) * sizeof(GpuPapiParticle),
        VkDeviceSize(capacity.emitters) * sizeof(GpuPapiEmitter),
        VkDeviceSize(std::max(capacity.action_states, 1u)) * 16,
        VkDeviceSize(capacity.programs) * sizeof(GpuPapiProgram),
        VkDeviceSize(std::max(capacity.actions, 1u)) * sizeof(GpuPapiAction),
        VkDeviceSize(std::max(capacity.commands, 1u)) * sizeof(GpuPapiCommand),
        VkDeviceSize(capacity.roots) * 8,
        VkDeviceSize(capacity.noise_values) * 16,
        VkDeviceSize(capacity.particles) * 4,
        VkDeviceSize(std::max(capacity.action_states, 1u)) * 4,
        4, VkDeviceSize(capacity.roots) * sizeof(GpuPapiStatus), capacity.collision_bytes,
        VkDeviceSize(capacity.roots) * 4, VkDeviceSize(capacity.particles) * 4,
        VkDeviceSize(capacity.programs) * 28 * 16, VkDeviceSize(capacity.programs) * 28 * 4,
        VkDeviceSize(capacity.programs) * 28 * 4};
    for (size_t i = 0; i < Count; ++i)
    {
        if (!sizes[i] || sizes[i] > limits.maxStorageBufferRange)
        { error = "GPU particle buffer exceeds maxStorageBufferRange"; destroy(); return false; }
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (i == DrawArguments) usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        std::unique_ptr<BufferResource> owned;
        BufferResource* buffer = nullptr;
        if (!create_buffer(sizes[i], usage, true, owned, buffer, error))
        { destroy(); return false; }
        buffers_[i] = buffer;
        owned_buffers_[i] = std::move(owned);
    }
    root_slots_ = capacity.roots;
    error.clear(); return true;
}

bool GpuParticleBuffers::create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool pooled_memory,
    std::unique_ptr<BufferResource>& owned, BufferResource*& result, std::string& error)
{
    result = nullptr;
    // Prefer NVRHI for compatible particle buffers. Preserve shared Vulkan
    // pooling on direct-Vulkan devices; BufferResource handles NVRHI failures.
    const bool use_vulkan_memory_pool = pooled_memory && !dispatch_.nvrhi_device;
    if (pool_)
        return pool_->acquire(size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            use_vulkan_memory_pool, result, error,
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT);
    auto resource = std::make_unique<BufferResource>();
    if (!resource->initialize(device_, size, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            memory_, dispatch_, error, use_vulkan_memory_pool,
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))
        return false;
    result = resource.get();
    owned = std::move(resource);
    error.clear();
    return true;
}

void GpuParticleBuffers::replace_buffer(Resource resource, std::unique_ptr<BufferResource> owned,
    BufferResource* buffer)
{
    if (pool_ && buffers_[resource] && !owned_buffers_[resource])
        pool_->release(buffers_[resource]);
    owned_buffers_[resource].reset();
    owned_buffers_[resource] = std::move(owned);
    buffers_[resource] = buffer;
}

void GpuParticleBuffers::destroy()
{
    for (size_t i = 0; i < Count; ++i)
    {
        if (pool_ && buffers_[i] && !owned_buffers_[i])
            pool_->release(buffers_[i]);
        owned_buffers_[i].reset();
        buffers_[i] = nullptr;
    }
    pool_ = nullptr;
    root_slots_ = 0;
    device_ = VK_NULL_HANDLE;
    memory_ = {};
    dispatch_ = {};
}
bool GpuParticleBuffers::resize_collision(VkDevice device, VkDeviceSize bytes, VkDeviceSize max_range,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error)
{
    if (!get(Collision).handle() || bytes < 48 || bytes % 4 || bytes > max_range)
    { error = "invalid GPU particle collision buffer size"; return false; }
    if (bytes <= get(Collision).size()) { error.clear(); return true; }
    device_ = device;
    memory_ = memory;
    dispatch_ = dispatch;
    std::unique_ptr<BufferResource> owned;
    BufferResource* replacement = nullptr;
    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!create_buffer(bytes, usage, true, owned, replacement, error)) return false;
    replace_buffer(Collision, std::move(owned), replacement); error.clear(); return true;
}
bool GpuParticleBuffers::resize_draw_buckets(VkDevice device, uint32_t buckets, VkDeviceSize max_range,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error)
{
    const VkDeviceSize bytes = VkDeviceSize(buckets) * 16;
    if (!buckets || !get(DrawArguments).handle() || bytes > max_range)
    { error = "GPU particle draw buckets exceed device limits"; return false; }
    if (bytes <= get(DrawArguments).size()) { error.clear(); return true; }
    device_ = device;
    memory_ = memory;
    dispatch_ = dispatch;
    std::unique_ptr<BufferResource> arguments_owned, offsets_owned, cursors_owned;
    BufferResource* arguments = nullptr;
    BufferResource* offsets = nullptr;
    BufferResource* cursors = nullptr;
    const auto usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    const auto discard = [&](BufferResource*& buffer, std::unique_ptr<BufferResource>& owned) {
        if (pool_ && buffer && !owned) pool_->release(buffer);
        owned.reset(); buffer = nullptr;
    };
    if (!create_buffer(bytes, usage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, true,
            arguments_owned, arguments, error) ||
        !create_buffer(VkDeviceSize(buckets) * 4, usage, true, offsets_owned, offsets, error) ||
        !create_buffer(VkDeviceSize(buckets) * 4, usage, true, cursors_owned, cursors, error))
    {
        discard(arguments, arguments_owned);
        discard(offsets, offsets_owned);
        discard(cursors, cursors_owned);
        return false;
    }
    replace_buffer(DrawArguments, std::move(arguments_owned), arguments);
    replace_buffer(BucketOffsets, std::move(offsets_owned), offsets);
    replace_buffer(BucketCursors, std::move(cursors_owned), cursors);
    error.clear(); return true;
}
bool GpuParticleBuffers::descriptors(GpuParticleCompute::Pass pass, VkDescriptorBufferInfo params,
    VkDescriptorBufferInfo visibility, std::vector<VkDescriptorBufferInfo>& output, std::string& error) const
{
    if (!params.buffer || params.range < sizeof(fg::GpuPapiConstants) ||
        (pass == GpuParticleCompute::Pass::Compact && (!visibility.buffer || visibility.range < 6 * 16)))
    { error = "invalid GPU particle uniform buffer"; return false; }
    const Resource init[]{Emitters, ParticleAllocation, StateAllocation, Errors};
    const Resource simulation[]{Particles, Emitters, ActionState, Programs, Actions, Commands, RootCommands,
        Noise, ParticleAllocation, StateAllocation, Errors, Status, Collision};
    const Resource capture_motion[]{Particles, Emitters};
    const Resource compact[]{VisibleRoots, Particles, Emitters, Programs, Indices, DrawArguments, BucketOffsets, BucketCursors};
    const Resource* selected = nullptr;
    size_t count = 0;
    switch (pass)
    {
    case GpuParticleCompute::Pass::Initialize: selected = init; count = std::size(init); break;
    case GpuParticleCompute::Pass::Simulate: selected = simulation; count = std::size(simulation); break;
    case GpuParticleCompute::Pass::Compact: selected = compact; count = std::size(compact); break;
    case GpuParticleCompute::Pass::CaptureMotion: selected = capture_motion; count = std::size(capture_motion); break;
    default: error = "invalid GPU particle compute pass"; return false;
    }
    std::vector<VkDescriptorBufferInfo> result{params};
    if (pass == GpuParticleCompute::Pass::Compact) result.push_back(visibility);
    for (size_t i = 0; i < count; ++i)
    {
        const auto& buffer = get(selected[i]);
        if (!buffer.handle() || !buffer.size())
        { error = "GPU particle buffers are not initialized"; return false; }
        result.push_back({buffer.handle(), 0, buffer.size()});
    }
    output = std::move(result); error.clear(); return true;
}
}
