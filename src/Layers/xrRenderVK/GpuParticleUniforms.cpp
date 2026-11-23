#include "GpuParticleUniforms.h"
#include <algorithm>
#include <limits>

namespace xray::render::vulkan
{
bool GpuParticleUniforms::initialize(VkDevice device, const VkPhysicalDeviceLimits& limits,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
    uint32_t entries, std::string& error)
{
    destroy();
    const VkDeviceSize alignment = std::max<VkDeviceSize>(limits.minUniformBufferOffsetAlignment, 1);
    constexpr VkDeviceSize max_entry = 256; // full existing material constant block
    if (!entries || limits.maxUniformBufferRange < max_entry ||
        alignment > std::numeric_limits<VkDeviceSize>::max() - max_entry + 1)
    { error = "invalid GPU particle uniform configuration"; return false; }
    stride_ = ((max_entry + alignment - 1) / alignment) * alignment;
    if (stride_ > std::numeric_limits<VkDeviceSize>::max() / entries)
    { error = "GPU particle uniform arena overflow"; destroy(); return false; }
    max_range_ = limits.maxUniformBufferRange;
    if (!buffer_.initialize(device, stride_ * entries, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, dispatch, error))
    { destroy(); return false; }
    error.clear(); return true;
}
bool GpuParticleUniforms::append(const void* data, size_t size, VkDescriptorBufferInfo& result, std::string& error)
{
    if (!buffer_.handle() || !data || !size || size > 256 || size > max_range_ ||
        next_ > buffer_.size() || stride_ > buffer_.size() - next_)
    { error = "GPU particle uniform arena exhausted or invalid"; return false; }
    if (!buffer_.write(next_, data, size, error)) return false;
    result = {buffer_.handle(), next_, size}; next_ += stride_; error.clear(); return true;
}
void GpuParticleUniforms::destroy()
{
    buffer_.destroy(); stride_ = next_ = max_range_ = 0;
}
}
