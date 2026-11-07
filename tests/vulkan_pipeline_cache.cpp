#include "src/Layers/xrRenderVK/PipelineCache.h"
#include <cassert>
#include <string>
using namespace xray::render::vulkan;
namespace
{
std::vector<uint8_t> driver;
size_t seed_size;
bool reject_seed;
unsigned destroyed;
VKAPI_ATTR VkResult VKAPI_CALL create(VkDevice, const VkPipelineCacheCreateInfo *info,
                                      const VkAllocationCallbacks *, VkPipelineCache *out)
{
    seed_size = info->initialDataSize;
    if (reject_seed && seed_size)
        return VK_ERROR_INITIALIZATION_FAILED;
    *out = (VkPipelineCache)uintptr_t(2);
    return VK_SUCCESS;
}
VKAPI_ATTR void VKAPI_CALL destroy(VkDevice, VkPipelineCache, const VkAllocationCallbacks *)
{
    ++destroyed;
}
VKAPI_ATTR VkResult VKAPI_CALL data(VkDevice, VkPipelineCache, size_t *size, void *out)
{
    if (out)
        std::memcpy(out, driver.data(), driver.size());
    *size = driver.size();
    return VK_SUCCESS;
}
VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL get(VkDevice, const char *name)
{
    if (std::string(name) == "vkCreatePipelineCache")
        return reinterpret_cast<PFN_vkVoidFunction>(create);
    if (std::string(name) == "vkDestroyPipelineCache")
        return reinterpret_cast<PFN_vkVoidFunction>(destroy);
    if (std::string(name) == "vkGetPipelineCacheData")
        return reinterpret_cast<PFN_vkVoidFunction>(data);
    return nullptr;
}
} // namespace
int main()
{
    VkPhysicalDeviceProperties p{};
    p.vendorID = 12;
    p.deviceID = 34;
    p.pipelineCacheUUID[0] = 56;
    driver.resize(40, 0);
    const uint32_t header[]{32, 1, 12, 34};
    std::memcpy(driver.data(), header, 16);
    driver[16] = 56;
    PipelineCache cache;
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, {}));
    assert(seed_size == 0);
    auto disk = cache.snapshot();
    assert(PipelineCache::compatible(disk, p));
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, disk));
    assert(seed_size == 40 && cache.loaded_bytes() == 40 && destroyed == 1);
    auto other = p;
    other.pipelineCacheUUID[0]++;
    assert(!PipelineCache::compatible(disk, other));
    other = p;
    other.deviceID++;
    assert(!PipelineCache::compatible(disk, other));
    other = p;
    other.vendorID++;
    assert(!PipelineCache::compatible(disk, other));
    auto truncated = disk;
    truncated.pop_back();
    assert(!PipelineCache::compatible(truncated, p));
    auto corrupt = disk;
    corrupt.back() ^= 1;
    assert(!PipelineCache::compatible(corrupt, p));
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, corrupt) && seed_size == 0);
    reject_seed = true;
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, disk));
    assert(seed_size == 0 && cache.loaded_bytes() == 0);
}
