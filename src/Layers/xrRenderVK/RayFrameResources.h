#pragma once
#include "RayGeneratePass.h"
#include "RayVisibilityPass.h"
#include "FrameImagePool.h"

namespace xray::render::vulkan
{
// The image pool, G-buffer views and sampler outlive this object. The owner
// waits for GPU idle before destroying or rebuilding these frame resources.
class RayFrameResources
{
public:
    ~RayFrameResources() { destroy(); }
    RayFrameResources() = default;
    RayFrameResources(const RayFrameResources&) = delete;
    RayFrameResources& operator=(const RayFrameResources&) = delete;
    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
        const std::vector<VkImageView>& depth, const std::vector<VkImageView>& normal,
        VkSampler sampler, const VkPhysicalDeviceProperties& properties,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& buffers,
        const FrameDispatch& images, FrameImagePool& pool, VkPipelineCache cache, std::string& error);
    void destroy();
    RayGeneratePass& generator() { return generator_; }
    RayVisibilityPass& visibility() { return visibility_; }
    BufferResource& rays(uint32_t index) { return rays_.at(index); }
    BufferResource& hits(uint32_t index) { return hits_.at(index); }
    VkImageView view(uint32_t index) const { return index < views_.size() ? views_[index] : VK_NULL_HANDLE; }
    uint32_t count() const { return static_cast<uint32_t>(views_.size()); }
    VkImageLayout layout(uint32_t index) const { return layouts_.at(index); }
    void mark_recorded(uint32_t index) { layouts_.at(index) = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; }
private:
    VkDevice device_{};
    FrameDispatch image_dispatch_{};
    FrameImagePool* pool_{};
    std::vector<BufferResource> rays_, hits_;
    std::vector<VkImage> images_;
    std::vector<VkImageView> views_;
    std::vector<VkImageLayout> layouts_;
    RayGeneratePass generator_;
    RayVisibilityPass visibility_;
};
}
