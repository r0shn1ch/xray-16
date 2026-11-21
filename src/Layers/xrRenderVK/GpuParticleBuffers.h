#pragma once
#include "BufferResource.h"
#include "FrameBufferPool.h"
#include "GpuParticleCompute.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace xray::render::vulkan
{
// Persistent device-local resources shared by the PAPI compute and draw passes.
// Upload, readback and per-dispatch uniforms are owned separately by the caller.
class GpuParticleBuffers
{
public:
    enum Resource : size_t { Particles, Emitters, ActionState, Programs, Actions, Commands,
        RootCommands, Noise, ParticleAllocation, StateAllocation, Errors, Status, Collision,
        VisibleRoots, Indices, DrawArguments, BucketOffsets, BucketCursors, Count };
    struct Capacity
    {
        uint32_t particles{}, emitters{}, action_states{}, programs{}, actions{}, commands{}, roots{}, noise_values{};
        VkDeviceSize collision_bytes{};
    };
    bool initialize(VkDevice device, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        const Capacity& capacity, std::string& error, FrameBufferPool* pool = nullptr);
    // Requires all GPU uses of the previous collision buffer to be complete.
    bool resize_collision(VkDevice device, VkDeviceSize bytes, VkDeviceSize max_range,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error);
    bool resize_draw_buckets(VkDevice device, uint32_t buckets, VkDeviceSize max_range,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch, std::string& error);
    void destroy();
    // Builds descriptor vectors in the exact order declared by each shader.
    // The caller keeps the supplied uniform buffers alive until completion.
    bool descriptors(GpuParticleCompute::Pass pass, VkDescriptorBufferInfo params,
        VkDescriptorBufferInfo visibility, std::vector<VkDescriptorBufferInfo>& output, std::string& error) const;
    const BufferResource& get(Resource resource) const
    { return buffers_[resource] ? *buffers_[resource] : empty_buffers_[resource]; }
private:
    bool create_buffer(VkDeviceSize size, VkBufferUsageFlags usage, bool pooled_memory,
        std::unique_ptr<BufferResource>& owned, BufferResource*& result, std::string& error);
    void replace_buffer(Resource resource, std::unique_ptr<BufferResource> owned, BufferResource* buffer);
    VkDevice device_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    BufferResourceDispatch dispatch_{};
    FrameBufferPool* pool_{};
    std::array<BufferResource*, Count> buffers_{};
    std::array<std::unique_ptr<BufferResource>, Count> owned_buffers_{};
    std::array<BufferResource, Count> empty_buffers_{};
};
}
