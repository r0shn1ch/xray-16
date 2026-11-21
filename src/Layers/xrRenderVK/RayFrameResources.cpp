#include "RayFrameResources.h"

namespace xray::render::vulkan
{
bool RayFrameResources::initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
    const std::vector<VkImageView>& depth, const std::vector<VkImageView>& normal,
    VkSampler sampler, const VkPhysicalDeviceProperties& properties,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& buffers,
    const FrameDispatch& images, FrameImagePool& pool, VkPipelineCache cache, std::string& error)
{
    destroy();
    const uint64_t pixels = uint64_t(extent.width) * extent.height;
    const uint64_t query_groups = (pixels + 63) / 64;
    const uint64_t query_x = query_groups < 65535u ? query_groups : 65535u;
    const uint64_t query_y = query_x ? (query_groups + query_x - 1) / query_x : 0;
    if (!physical || !device || !get || !sampler || !pixels || pixels > UINT32_MAX ||
        query_x > properties.limits.maxComputeWorkGroupCount[0] ||
        query_y > properties.limits.maxComputeWorkGroupCount[1] ||
        pixels > properties.limits.maxStorageBufferRange / sizeof(RayQueryInput) ||
        pixels > properties.limits.maxStorageBufferRange / sizeof(RayQueryHit) || depth.empty() || depth.size() > 16 ||
        normal.size() != depth.size() || !images.create_image_view || !images.destroy_image_view ||
        !images.get_format_properties || (extent.width + uint64_t(7)) / 8 > properties.limits.maxComputeWorkGroupCount[0] ||
        (extent.height + uint64_t(7)) / 8 > properties.limits.maxComputeWorkGroupCount[1])
    { error = "ray frame extent exceeds supported buffer or dispatch limits"; return false; }
    VkFormatProperties support{};
    images.get_format_properties(physical, VK_FORMAT_R32_SFLOAT, &support);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((support.optimalTilingFeatures & required) != required)
    { error = "ray visibility requires sampled and storage R32F images"; return false; }
    const uint64_t bytes = pixels * sizeof(RayQueryInput);
    device_ = device; image_dispatch_ = images; pool_ = &pool;
    rays_.resize(depth.size()); hits_.resize(depth.size());
    images_.resize(depth.size()); views_.resize(depth.size());
    layouts_.assign(depth.size(), VK_IMAGE_LAYOUT_UNDEFINED);
    std::vector<BufferResource*> rays, hits;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D; info.format = VK_FORMAT_R32_SFLOAT;
    info.extent = {extent.width, extent.height, 1}; info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT; info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    for (size_t i = 0; i < depth.size(); ++i)
    {
        if (!rays_[i].initialize(device, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, buffers, error) ||
            !hits_[i].initialize(device, pixels * sizeof(RayQueryHit), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, buffers, error) ||
            !pool.acquire(info, images_[i], error)) { destroy(); return false; }
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = images_[i]; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = info.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (images.create_image_view(device, &view, nullptr, &views_[i]) != VK_SUCCESS)
        { error = "ray visibility view creation failed"; destroy(); return false; }
        rays.push_back(&rays_[i]); hits.push_back(&hits_[i]);
    }
    if (!generator_.initialize(device, get, extent, depth, normal, sampler, rays, cache, error) ||
        !visibility_.initialize(device, get, extent, images_, views_, hits, cache, error))
    { destroy(); return false; }
    error.clear(); return true;
}
void RayFrameResources::destroy()
{
    visibility_.destroy(); generator_.destroy();
    if (device_ && image_dispatch_.destroy_image_view)
        for (VkImageView view : views_) if (view) image_dispatch_.destroy_image_view(device_, view, nullptr);
    if (pool_) for (VkImage image : images_) if (image) pool_->release(image);
    views_.clear(); images_.clear(); layouts_.clear(); rays_.clear(); hits_.clear();
    device_ = VK_NULL_HANDLE; pool_ = nullptr; image_dispatch_ = {};
}
}
