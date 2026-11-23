#pragma once
#include "BufferResource.h"
#include <string>

namespace xray::render::vulkan
{
// Host-coherent per-frame uniform arena. Reset only after its frame fence.
class GpuParticleUniforms
{
public:
    bool initialize(VkDevice device, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        uint32_t entries, std::string& error);
    bool append(const void* data, size_t size, VkDescriptorBufferInfo& result, std::string& error);
    void reset() { next_ = 0; }
    void destroy();
private:
    BufferResource buffer_;
    VkDeviceSize stride_{}, next_{}, max_range_{};
};
}
