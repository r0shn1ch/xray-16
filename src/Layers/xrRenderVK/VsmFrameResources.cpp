#include "VsmFrameResources.h"

namespace xray::render::vulkan
{
bool VsmFrameResources::initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    VkExtent2D extent, const std::vector<VkImageView>& depth, VkSampler depth_sampler, VkFormat shadow_format,
    uint32_t axis, uint32_t levels, uint32_t page_size, uint32_t dynamic_columns,
    uint32_t dynamic_rows, const VkPhysicalDeviceProperties& properties,
    const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& buffers,
    const FrameDispatch& images, FrameImagePool& pool, VkPipelineCache pipeline_cache, std::string& error)
{
    destroy();
    const uint64_t pages = uint64_t(axis) * axis * levels;
    const uint64_t dynamic_pages = uint64_t(dynamic_columns) * dynamic_rows;
    const uint64_t max_range = properties.limits.maxStorageBufferRange;
    if (!physical || !device || !get || !depth_sampler || !extent.width || !extent.height || depth.empty() ||
        depth.size() > 16 || !axis || axis > 32768 || (axis & (axis - 1)) || !levels || levels > 6 ||
        !page_size || !dynamic_columns || !dynamic_rows || pages > UINT32_MAX || dynamic_pages > UINT32_MAX ||
        uint64_t(axis) * levels > UINT32_MAX || 16 + pages * 16 > max_range ||
        dynamic_pages * 16 > max_range || !images.get_format_properties || !images.create_image_view ||
        !images.destroy_image_view || (uint64_t(extent.width) + 7) / 8 > properties.limits.maxComputeWorkGroupCount[0] ||
        (uint64_t(extent.height) + 7) / 8 > properties.limits.maxComputeWorkGroupCount[1])
    { error = "invalid VSM frame resources or device limits"; return false; }
    VkFormatProperties support{};
    images.get_format_properties(physical, VK_FORMAT_R32_SFLOAT, &support);
    const auto features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((support.optimalTilingFeatures & features) != features)
    { error = "VSM visibility requires sampled and storage R32F images"; return false; }
    const auto depth_features = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (shadow_format != VK_FORMAT_D32_SFLOAT && shadow_format != VK_FORMAT_D16_UNORM)
    { error = "VSM atlas format must match the existing shadow pipeline"; return false; }
    images.get_format_properties(physical, shadow_format, &support);
    if ((support.optimalTilingFeatures & depth_features) != depth_features)
    { error = "VSM requires a sampled depth attachment format"; return false; }
    device_ = device; dispatch_ = images; pool_ = &pool;
    const auto fail = [&]() { destroy(); return false; };
    if (!static_atlas.initialize(physical, device, get, shadow_format, page_size, axis, axis * levels,
            properties.limits, memory, images, error, false, &pool) ||
        !dynamic_atlas.initialize(physical, device, get, shadow_format, page_size, dynamic_columns, dynamic_rows,
            properties.limits, memory, images, error, false, &pool)) return fail();
    constexpr VkBufferUsageFlags storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    constexpr VkMemoryPropertyFlags coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!static_slots.initialize(device, pages * 16, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, buffers, error) ||
        !visible_static_table.initialize(device, pages * 4, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, buffers, error) ||
        !visible_dynamic_table.initialize(device, pages * 4, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, buffers, error)) return fail();
    images_.resize(depth.size());
    std::vector<BufferResource*> demand_parameters, needed, static_tables, dynamic_tables, resolve_parameters;
    std::vector<VkImage> outputs; std::vector<VkImageView> views;
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D; info.format = VK_FORMAT_R32_SFLOAT;
    info.extent = {extent.width, extent.height, 1}; info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT; info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    for (auto& image : images_)
    {
        const auto make = [&](BufferResource& buffer, VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags flags) {
            return buffer.initialize(device, size, usage, flags, memory, buffers, error);
        };
        if (!make(image.demand_parameters, sizeof(VsmDemandParameters), storage, coherent) ||
            !make(image.needed, pages * 4, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            !make(image.static_table, pages * 4, storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            !make(image.dynamic_table, pages * 4, storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ||
            !make(image.static_work, 16 + pages * 16, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, coherent) ||
            !make(image.dynamic_pages, dynamic_pages * 16, storage, coherent) ||
            !make(image.dynamic_counts, (levels + 1) * 4, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, coherent) ||
            !make(image.resolve_parameters, sizeof(VsmResolveParameters), storage, coherent) ||
            !pool.acquire(info, image.output, error)) return fail();
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = image.output; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = info.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (images.create_image_view(device, &view, nullptr, &image.output_view) != VK_SUCCESS)
        { error = "VSM visibility image view creation failed"; return fail(); }
        demand_parameters.push_back(&image.demand_parameters); needed.push_back(&image.needed);
        static_tables.push_back(&image.static_table); dynamic_tables.push_back(&image.dynamic_table);
        resolve_parameters.push_back(&image.resolve_parameters); outputs.push_back(image.output); views.push_back(image.output_view);
    }
    const auto count = static_cast<uint32_t>(images_.size());
    if (!demand.initialize(device, get, extent, depth, depth_sampler, demand_parameters, needed, pipeline_cache, error) ||
        !allocator.initialize(device, get, count, max_range, pipeline_cache, error) ||
        !cache.initialize(device, get, count, max_range, pipeline_cache, error) ||
        !resolve.initialize(device, get, extent, depth, static_atlas.view(), dynamic_atlas.view(), depth_sampler,
            static_tables, dynamic_tables, resolve_parameters, outputs, views, pipeline_cache, error)) return fail();
    error.clear(); return true;
}
void VsmFrameResources::destroy()
{
    resolve.destroy(); cache.destroy(); allocator.destroy(); demand.destroy();
    if (device_ && dispatch_.destroy_image_view)
        for (const auto& image : images_) if (image.output_view) dispatch_.destroy_image_view(device_, image.output_view, nullptr);
    if (pool_) for (const auto& image : images_) if (image.output) pool_->release(image.output);
    images_.clear(); visible_static_table.destroy(); visible_dynamic_table.destroy(); static_slots.destroy();
    dynamic_atlas.destroy(); static_atlas.destroy();
    device_ = VK_NULL_HANDLE; dispatch_ = {}; pool_ = nullptr;
}
}
