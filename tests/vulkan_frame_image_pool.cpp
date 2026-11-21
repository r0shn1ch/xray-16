#include "src/Layers/xrRenderVK/FrameImagePool.h"

#include <cassert>
#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
template <class T> T handle(uintptr_t id)
{
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(id);
    else return static_cast<T>(id);
}
uintptr_t next_image = 10, next_memory = 100;
unsigned created{}, destroyed{}, allocated{}, freed{};
std::vector<std::pair<VkSharingMode, std::vector<uint32_t>>> image_sharing;
VkResult VKAPI_CALL create_image(VkDevice, const VkImageCreateInfo* info, const VkAllocationCallbacks*, VkImage* image)
{
    std::vector<uint32_t> families;
    if (info->pQueueFamilyIndices)
        families.assign(info->pQueueFamilyIndices, info->pQueueFamilyIndices + info->queueFamilyIndexCount);
    image_sharing.emplace_back(info->sharingMode, std::move(families));
    *image = handle<VkImage>(next_image++);
    ++created;
    return VK_SUCCESS;
}
void VKAPI_CALL destroy_image(VkDevice, VkImage, const VkAllocationCallbacks*) { ++destroyed; }
void VKAPI_CALL requirements(VkDevice, VkImage, VkMemoryRequirements* result)
{
    result->size = 4096;
    result->alignment = 256;
    result->memoryTypeBits = 1;
}
VkResult VKAPI_CALL allocate(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory* memory)
{
    *memory = handle<VkDeviceMemory>(next_memory++);
    ++allocated;
    return VK_SUCCESS;
}
void VKAPI_CALL free_memory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*) { ++freed; }
VkResult VKAPI_CALL bind(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
}

int main()
{
    FrameImagePool pool;
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    assert(pool.initialize(handle<VkDevice>(1), memory,
        {create_image, destroy_image, requirements, allocate, free_memory, bind}));
    VkImageCreateInfo description{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    description.imageType = VK_IMAGE_TYPE_2D;
    description.format = VK_FORMAT_R8G8B8A8_UNORM;
    description.extent = {64, 64, 1};
    description.mipLevels = description.arrayLayers = 1;
    description.samples = VK_SAMPLE_COUNT_1_BIT;
    description.tiling = VK_IMAGE_TILING_OPTIMAL;
    description.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    description.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    std::string error;
    VkImage first{}, second{}, reused{}, incompatible{};
    assert(pool.acquire(description, first, error));
    assert(pool.acquire(description, second, error) && second != first);
    assert(created == 2 && allocated == 2);
    assert(!pool.release(handle<VkImage>(999)));
    assert(pool.release(first));
    assert(!pool.release(first));
    assert(pool.acquire(description, reused, error) && reused == first);
    assert(created == 2);
    assert(pool.release(reused));
    description.extent.width = 128;
    assert(pool.acquire(description, incompatible, error) && incompatible != first && incompatible != second);
    assert(created == 3);
    // The unused first image is evicted after four successful frame ticks;
    // the second and the differently sized image are still in use.
    for (int i = 0; i < 3; ++i) pool.tick();
    assert(destroyed == 0 && freed == 0);
    pool.tick();
    assert(destroyed == 1 && freed == 1);
    description.extent.width = 64;
    VkImage new_image{};
    assert(pool.acquire(description, new_image, error) && new_image != first);
    assert(created == 4);

    FrameImageSharing sharing = frame_image_sharing(5, 2);
    assert(sharing.mode == VK_SHARING_MODE_CONCURRENT &&
        sharing.queue_families == std::vector<uint32_t>({2, 5}));
    VkImageCreateInfo concurrent = description;
    sharing.apply(concurrent);
    // Vulkan accepts any ordering, while the pool canonicalizes it for reuse.
    uint32_t reversed_families[]{5, 2};
    concurrent.queueFamilyIndexCount = 2;
    concurrent.pQueueFamilyIndices = reversed_families;
    VkImage concurrent_image{}, concurrent_reused{};
    assert(pool.acquire(concurrent, concurrent_image, error));
    assert(image_sharing.back().first == VK_SHARING_MODE_CONCURRENT &&
        image_sharing.back().second == std::vector<uint32_t>({5, 2}));
    assert(pool.release(concurrent_image));
    concurrent.queueFamilyIndexCount = 2;
    concurrent.pQueueFamilyIndices = sharing.queue_families.data();
    assert(pool.acquire(concurrent, concurrent_reused, error) && concurrent_reused == concurrent_image);
    assert(created == 5);
    assert(pool.release(concurrent_reused));
    uint32_t invalid_families[]{2, 2};
    concurrent.pQueueFamilyIndices = invalid_families;
    assert(!pool.acquire(concurrent, concurrent_reused, error));

    pool.reset();
    assert(destroyed == created && freed == allocated);
}
