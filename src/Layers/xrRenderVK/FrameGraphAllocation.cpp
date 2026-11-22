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
        if (image)
            pool.release(image);
    }

    FrameImagePool& pool;
    VkImage image = VK_NULL_HANDLE;
};
}

bool FrameImageGraph::materialize_images(FrameImagePool& pool, std::string& error)
{
    return materialize_images([&pool](const VkImageCreateInfo& info,
        TransientImage& output, std::string& failure) {
        auto owner = std::make_shared<PooledImage>(pool);
        if (!pool.acquire(info, owner->image, failure))
            return false;
        output.handle = owner->image;
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
        output.sharing_mode = buffer->sharing_mode();
        output.queue_families = buffer->sharing_queue_families();
        return true;
    }, error);
}
}
