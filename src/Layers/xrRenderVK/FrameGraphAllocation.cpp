#include "FrameImageGraph.h"
#include "FrameBufferPool.h"
#include "FrameImagePool.h"

namespace xray::render::vulkan
{
namespace
{
struct PooledImage
{
    explicit PooledImage(FrameImagePool& pool) : pool(pool) {}
    ~PooledImage()
    {
        if (view)
            destroy_view(device, view, nullptr);
        if (image)
            pool.release(image);
    }

    FrameImagePool& pool;
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDevice device{};
    PFN_vkDestroyImageView destroy_view{};
};
}

bool FrameImageGraph::materialize_images(FrameImagePool& pool, std::string& error)
{
    return materialize_images([&pool](const VkImageCreateInfo& info, const VkImageSubresourceRange&,
        TransientImage& output, std::string& failure) {
        auto owner = std::make_shared<PooledImage>(pool);
        if (!pool.acquire(info, owner->image, failure))
            return false;
        output.handle = owner->image;
        output.nvrhi_texture = pool.nvrhi_texture(owner->image);
        output.owner = std::move(owner);
        return true;
    }, error);
}

bool FrameImageGraph::materialize_buffers(FrameBufferPool& pool, std::string& error)
{
    return materialize_buffers([&pool](const BufferDescription& description,
        TransientBuffer& output, std::string& failure) {
        vulkan::BufferResource* buffer = nullptr;
        if (!pool.acquire(description.size, description.usage, description.memory_properties,
                description.pooled_memory, buffer, failure, description.shared_queue_flags))
            return false;
        output.owner = std::shared_ptr<void>(buffer, [&pool](void* resource) {
            pool.release(static_cast<vulkan::BufferResource*>(resource));
        });
        output.handle = buffer->handle();
        output.nvrhi_buffer = buffer->nvrhi_buffer();
        output.sharing_mode = buffer->sharing_mode();
        output.queue_families = buffer->sharing_queue_families();
        return true;
    }, error);
}

bool FrameImageGraph::materialize_2d_image_views(FrameImagePool& pool, VkDevice device,
    PFN_vkCreateImageView create_view, PFN_vkDestroyImageView destroy_view, std::string& error)
{
    return materialize_images([&](const VkImageCreateInfo& info, const VkImageSubresourceRange& range,
        TransientImage& output, std::string& failure) {
        if (!device || !create_view || !destroy_view || info.imageType != VK_IMAGE_TYPE_2D)
        {
            failure = "Frame image graph: invalid pooled 2D image view configuration";
            return false;
        }
        auto owner = std::make_shared<PooledImage>(pool);
        owner->device = device;
        owner->destroy_view = destroy_view;
        if (!pool.acquire(info, owner->image, failure))
            return false;
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = owner->image;
        view.viewType = range.layerCount == 1 ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        view.format = info.format;
        view.subresourceRange = range;
        if (create_view(device, &view, nullptr, &owner->view) != VK_SUCCESS)
        {
            owner->view = VK_NULL_HANDLE;
            failure = "Frame image graph: pooled 2D image view creation failed";
            return false;
        }
        output = {owner->image, owner, owner->view, pool.nvrhi_texture(owner->image)};
        return true;
    }, error);
}
}
