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
    p.apiVersion = VK_API_VERSION_1_1;
    p.driverVersion = 78;
    p.pipelineCacheUUID[0] = 56;
    driver.resize(40, 0);
    const uint32_t header[]{32, 1, 12, 34};
    std::memcpy(driver.data(), header, 16);
    driver[16] = 56;
    PipelineCacheFingerprint assets("build-a");
    assets.append("vk\\level.vs.spv", "shader-a", 8);
    const uint64_t shader_fingerprint = assets.value();
    PipelineCacheFingerprint changed_bytes("build-a");
    changed_bytes.append("vk\\level.vs.spv", "shader-b", 8);
    PipelineCacheFingerprint changed_name("build-a");
    changed_name.append("vk\\level.ps.spv", "shader-a", 8);
    PipelineCacheFingerprint changed_build("build-b");
    changed_build.append("vk\\level.vs.spv", "shader-a", 8);
    PipelineCacheFingerprint removed_shader("build-a");
    assert(changed_bytes.value() != shader_fingerprint && changed_name.value() != shader_fingerprint &&
           changed_build.value() != shader_fingerprint && removed_shader.value() != shader_fingerprint);
    PipelineCache cache;
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, {}, shader_fingerprint));
    assert(seed_size == 0);
    auto disk = cache.snapshot(shader_fingerprint);
    assert(PipelineCache::compatible(disk, p, shader_fingerprint));
    assert(!PipelineCache::compatible(disk, p, changed_bytes.value()));
    assert(!PipelineCache::compatible(disk, p, changed_name.value()));
    assert(!PipelineCache::compatible(disk, p, changed_build.value()));
    assert(!PipelineCache::compatible(disk, p, removed_shader.value()));
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, disk, changed_bytes.value()));
    assert(seed_size == 0 && cache.loaded_bytes() == 0);
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, disk, shader_fingerprint));
    assert(seed_size == 40 && cache.loaded_bytes() == 40 && destroyed == 2);
    auto other = p;
    other.driverVersion++;
    assert(!PipelineCache::compatible(disk, other, shader_fingerprint));
    assert(cache.initialize((VkDevice)uintptr_t(1), get, other, disk, shader_fingerprint));
    assert(seed_size == 0 && cache.loaded_bytes() == 0);
    other = p;
    other.apiVersion++;
    assert(!PipelineCache::compatible(disk, other, shader_fingerprint));
    other = p;
    other.pipelineCacheUUID[0]++;
    assert(!PipelineCache::compatible(disk, other, shader_fingerprint));
    other = p;
    other.deviceID++;
    assert(!PipelineCache::compatible(disk, other, shader_fingerprint));
    other = p;
    other.vendorID++;
    assert(!PipelineCache::compatible(disk, other, shader_fingerprint));
    auto truncated = disk;
    truncated.pop_back();
    assert(!PipelineCache::compatible(truncated, p, shader_fingerprint));
    auto corrupt = disk;
    corrupt.back() ^= 1;
    assert(!PipelineCache::compatible(corrupt, p, shader_fingerprint));
    auto old_wrapper = disk;
    std::memcpy(old_wrapper.data(), "XRAYVKC2", 8);
    assert(!PipelineCache::compatible(old_wrapper, p, shader_fingerprint));
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, corrupt, shader_fingerprint) && seed_size == 0);
    reject_seed = true;
    assert(cache.initialize((VkDevice)uintptr_t(1), get, p, disk, shader_fingerprint));
    assert(seed_size == 0 && cache.loaded_bytes() == 0);
}
