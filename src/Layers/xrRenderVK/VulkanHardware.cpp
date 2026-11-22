#include "VulkanHardware.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <vector>

namespace xray::render::vulkan
{
namespace
{
bool supports_extension(const std::vector<VkExtensionProperties> &extensions, const char *name)
{
    return std::any_of(extensions.begin(), extensions.end(), [name](const VkExtensionProperties &extension) {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

bool has_swapchain_extension(VkPhysicalDevice device, PFN_vkEnumerateDeviceExtensionProperties enumerate,
                             std::vector<VkExtensionProperties> &extensions)
{
    uint32_t count = 0;
    if (enumerate(device, nullptr, &count, nullptr) != VK_SUCCESS || count == 0)
        return false;

    extensions.resize(count);
    const VkResult result = enumerate(device, nullptr, &count, extensions.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE)
        return false;
    extensions.resize(count);
    return supports_extension(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
}

int64_t score_device(const VkPhysicalDeviceProperties &properties, uint64_t local_memory_bytes, uint32_t queue_count)
{
    int64_t score = static_cast<int64_t>(std::min<uint64_t>(local_memory_bytes >> 20, 65535));
    switch (properties.deviceType)
    {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score += 1'000'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score += 800'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        score += 400'000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        score += 100'000;
        break;
    default:
        break;
    }
    score += static_cast<int64_t>(properties.limits.maxImageDimension2D);
    score += static_cast<int64_t>(queue_count) * 100;
    return score;
}
} // namespace

bool supports_game_formats(VkPhysicalDevice device, PFN_vkGetPhysicalDeviceFormatProperties get,
                           const VkPhysicalDeviceProperties &properties, std::string &error)
{
    if (!get || properties.limits.maxPushConstantsSize < 256 || properties.limits.maxColorAttachments < 3 ||
        properties.limits.maxImageDimension2D < 1024 ||
        // The material set declares bindings 0..19 for fragment sampling.
        // Pipeline layout creation counts the declared stage bindings, even
        // when a particular variant only samples the first few roles.
        properties.limits.maxPerStageDescriptorSamplers < 20 ||
        properties.limits.maxPerStageDescriptorSampledImages < 20)
    {
        error = "Vulkan gameplay needs 256 push-constant bytes, three color attachments, "
                "1024px textures and 20 fragment sampled images/samplers";
        return false;
    }
    const auto supports = [&](VkFormat format, VkFormatFeatureFlags required) {
        VkFormatProperties available{};
        get(device, format, &available);
        return (available.optimalTilingFeatures & required) == required;
    };
    const auto color = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (!supports(VK_FORMAT_R8G8B8A8_UNORM, color) ||
        (!supports(VK_FORMAT_D32_SFLOAT,
                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) &&
         !supports(VK_FORMAT_D16_UNORM,
                   VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)))
    {
        error = "Vulkan gameplay attachment or sampled depth formats are unavailable";
        return false;
    }
    error.clear();
    return true;
}

RayTracingAvailability probe_ray_tracing_availability(VkPhysicalDevice device,
    const std::vector<VkExtensionProperties>& extensions,
    uint32_t api_version, PFN_vkGetPhysicalDeviceFeatures2KHR get_features2)
{
    RayTracingAvailability result;
    result.vulkan11_dependencies =
        supports_extension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) &&
        supports_extension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) &&
        supports_extension(extensions, VK_KHR_SPIRV_1_4_EXTENSION_NAME) &&
        supports_extension(extensions, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
    const bool acceleration_dependencies = api_version >= VK_API_VERSION_1_2 ||
        (api_version >= VK_API_VERSION_1_1 &&
         supports_extension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) &&
         supports_extension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME));
    const bool shader_dependencies = api_version >= VK_API_VERSION_1_2 ||
        (api_version >= VK_API_VERSION_1_1 &&
         supports_extension(extensions, VK_KHR_SPIRV_1_4_EXTENSION_NAME) &&
         supports_extension(extensions, VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME));
    if (!device || !get_features2 ||
        !supports_extension(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) ||
        !supports_extension(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) ||
        !acceleration_dependencies)
        return result;
    VkPhysicalDeviceBufferDeviceAddressFeatures address{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR pipeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &address;
    address.pNext = &acceleration;
    acceleration.pNext = &query;
    query.pNext = &pipeline;
    get_features2(device, &features);
    result.acceleration_structure = address.bufferDeviceAddress && acceleration.accelerationStructure;
    result.ray_query = result.acceleration_structure && shader_dependencies && query.rayQuery &&
        supports_extension(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);
    result.ray_tracing_pipeline = result.acceleration_structure && shader_dependencies &&
        pipeline.rayTracingPipeline &&
        supports_extension(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    return result;
}

bool probe_bindless_texture_availability(VkPhysicalDevice device,
    const std::vector<VkExtensionProperties>& extensions, uint32_t api_version,
    PFN_vkGetPhysicalDeviceFeatures2KHR get_features2)
{
    if (!device || !get_features2 || api_version < VK_API_VERSION_1_1 ||
        (api_version < VK_API_VERSION_1_2 &&
         !supports_extension(extensions, VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME)))
        return false;
    VkPhysicalDeviceDescriptorIndexingFeatures indexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &indexing;
    get_features2(device, &features);
    return indexing.shaderSampledImageArrayNonUniformIndexing &&
        indexing.runtimeDescriptorArray && indexing.descriptorBindingPartiallyBound &&
        indexing.descriptorBindingSampledImageUpdateAfterBind;
}

bool probe_timeline_semaphore_availability(VkPhysicalDevice device,
    const std::vector<VkExtensionProperties>& extensions, uint32_t api_version,
    PFN_vkGetPhysicalDeviceFeatures2KHR get_features2)
{
    if (!device || !get_features2 || api_version < VK_API_VERSION_1_1 ||
        (api_version < VK_API_VERSION_1_2 &&
            !supports_extension(extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)))
        return false;
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.pNext = &timeline;
    get_features2(device, &features);
    return timeline.timelineSemaphore == VK_TRUE;
}

uint32_t visibility_texture_capacity(const VkPhysicalDeviceProperties& properties,
    const VkPhysicalDeviceDescriptorIndexingProperties& indexing)
{
    // Raster uses four storage buffers, two storage images and one uniform
    // texel buffer; resolve uses three of each plus the sampled ID image.
    constexpr uint32_t storage_buffers = 4, storage_images = 3, other_resources = 7;
    if (properties.limits.maxPerStageDescriptorStorageBuffers < storage_buffers ||
        properties.limits.maxDescriptorSetStorageBuffers < storage_buffers ||
        properties.limits.maxPerStageDescriptorStorageImages < storage_images ||
        properties.limits.maxDescriptorSetStorageImages < storage_images ||
        indexing.maxPerStageDescriptorUpdateAfterBindStorageBuffers < storage_buffers ||
        indexing.maxDescriptorSetUpdateAfterBindStorageBuffers < storage_buffers ||
        indexing.maxPerStageDescriptorUpdateAfterBindStorageImages < storage_images ||
        indexing.maxDescriptorSetUpdateAfterBindStorageImages < storage_images ||
        indexing.maxPerStageDescriptorUpdateAfterBindSampledImages <= 1 ||
        indexing.maxPerStageDescriptorUpdateAfterBindSamplers <= 1 ||
        indexing.maxDescriptorSetUpdateAfterBindSampledImages <= 1 ||
        indexing.maxDescriptorSetUpdateAfterBindSamplers <= 1 ||
        indexing.maxPerStageUpdateAfterBindResources <= other_resources)
        return 0;
    return std::min({indexing.maxPerStageDescriptorUpdateAfterBindSampledImages - 1,
        indexing.maxPerStageDescriptorUpdateAfterBindSamplers - 1,
        indexing.maxDescriptorSetUpdateAfterBindSampledImages - 1,
        indexing.maxDescriptorSetUpdateAfterBindSamplers - 1,
        indexing.maxPerStageUpdateAfterBindResources - other_resources,
        indexing.maxUpdateAfterBindDescriptorsInAllPools});
}

uint32_t ray_query_texture_capacity(const VkPhysicalDeviceProperties& properties,
    const VkPhysicalDeviceDescriptorIndexingProperties& indexing,
    const VkPhysicalDeviceAccelerationStructurePropertiesKHR& acceleration)
{
    constexpr uint32_t storage_buffers = 5;
    constexpr uint32_t other_resources = storage_buffers + 1;
    if (properties.limits.maxPerStageDescriptorStorageBuffers < storage_buffers ||
        properties.limits.maxDescriptorSetStorageBuffers < storage_buffers ||
        indexing.maxPerStageDescriptorUpdateAfterBindStorageBuffers < storage_buffers ||
        indexing.maxDescriptorSetUpdateAfterBindStorageBuffers < storage_buffers ||
        acceleration.maxPerStageDescriptorUpdateAfterBindAccelerationStructures < 1 ||
        acceleration.maxDescriptorSetUpdateAfterBindAccelerationStructures < 1 ||
        indexing.maxPerStageUpdateAfterBindResources <= other_resources)
        return 0;
    return std::min({indexing.maxPerStageDescriptorUpdateAfterBindSampledImages,
        indexing.maxPerStageDescriptorUpdateAfterBindSamplers,
        indexing.maxDescriptorSetUpdateAfterBindSampledImages,
        indexing.maxDescriptorSetUpdateAfterBindSamplers,
        indexing.maxUpdateAfterBindDescriptorsInAllPools,
        indexing.maxPerStageUpdateAfterBindResources - other_resources});
}

bool select_physical_device(VkInstance instance, VkSurfaceKHR surface, const HardwareDispatch &vk,
                            PhysicalDevice &selected, std::string &error)
{
    selected = {};
    if (!vk.enumerate_physical_devices || !vk.get_queue_families || !vk.get_surface_support ||
        !vk.enumerate_device_extensions || !vk.get_physical_properties || !vk.get_physical_features ||
        !vk.get_memory_properties || !vk.get_format_properties)
    {
        error = "required Vulkan hardware procedures are unavailable";
        return false;
    }

    uint32_t physical_count = 0;
    VkResult result = vk.enumerate_physical_devices(instance, &physical_count, nullptr);
    if (result != VK_SUCCESS || physical_count == 0)
    {
        error = "no Vulkan physical device is available";
        return false;
    }

    std::vector<VkPhysicalDevice> devices(physical_count);
    result = vk.enumerate_physical_devices(instance, &physical_count, devices.data());
    if (result != VK_SUCCESS && result != VK_INCOMPLETE)
    {
        error = "could not enumerate Vulkan physical devices";
        return false;
    }

    int64_t best_score = std::numeric_limits<int64_t>::min();
    std::vector<VkExtensionProperties> extensions;
    for (VkPhysicalDevice device : devices)
    {
        if (!has_swapchain_extension(device, vk.enumerate_device_extensions, extensions))
            continue;

        uint32_t family_count = 0;
        vk.get_queue_families(device, &family_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(family_count);
        vk.get_queue_families(device, &family_count, families.data());
        for (uint32_t family = 0; family < family_count; ++family)
        {
            const VkQueueFlags required = VK_QUEUE_GRAPHICS_BIT;
            if (families[family].queueCount == 0 ||
                (families[family].queueFlags & required) != required)
                continue;

            VkBool32 presentation_supported = VK_FALSE;
            if (vk.get_surface_support(device, family, surface, &presentation_supported) != VK_SUCCESS ||
                !presentation_supported)
                continue;

            PhysicalDevice candidate;
            candidate.handle = device;
            candidate.graphics_present_family = family;
            candidate.graphics_queue_flags = families[family].queueFlags;
            // Prefer a second queue in the graphics family to avoid sharing
            // overhead when that family supports compute. Otherwise use a
            // dedicated compute-capable family; cross-family buffers are
            // created with concurrent sharing and the frame submission
            // semaphore carries the producer/consumer dependency.
            if ((families[family].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                families[family].queueCount >= 2)
            {
                candidate.compute_queue_family = family;
                candidate.compute_queue_index = 1;
            }
            if (candidate.compute_queue_family == UINT32_MAX)
            {
                for (uint32_t compute_family = 0; compute_family < family_count; ++compute_family)
                {
                    if (families[compute_family].queueCount &&
                        (families[compute_family].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
                        !(families[compute_family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
                    {
                        candidate.compute_queue_family = compute_family;
                        break;
                    }
                }
            }
            if (candidate.compute_queue_family == UINT32_MAX)
            {
                for (uint32_t compute_family = 0; compute_family < family_count; ++compute_family)
                {
                    if (compute_family != family && families[compute_family].queueCount &&
                        (families[compute_family].queueFlags & VK_QUEUE_COMPUTE_BIT))
                    {
                        candidate.compute_queue_family = compute_family;
                        break;
                    }
                }
            }
            // A single queue that supports both graphics and compute is a
            // valid synchronous fallback. Preserve its family metadata so
            // NVRHI and the frame dependency graph see the actual capability.
            if (candidate.compute_queue_family == UINT32_MAX &&
                (families[family].queueFlags & VK_QUEUE_COMPUTE_BIT))
            {
                candidate.compute_queue_family = family;
                candidate.compute_queue_index = 0;
            }
            if (candidate.compute_queue_family < family_count)
                candidate.compute_queue_flags = families[candidate.compute_queue_family].queueFlags;

            // Frame interpolation's replacement swapchain uses four distinct
            // queues. Reserve the game queue and the regular compute queue,
            // then find distinct present, async-compute and acquire queues.
            // Devices without this topology continue to use the regular
            // swapchain and can still use the renderer's other upscalers.
            struct QueueSlot
            {
                uint32_t family = UINT32_MAX;
                uint32_t index = UINT32_MAX;
            };
            std::vector<std::vector<bool>> occupied(family_count);
            for (uint32_t family_index = 0; family_index < family_count; ++family_index)
                occupied[family_index].resize(families[family_index].queueCount, false);
            occupied[family][0] = true;
            if (candidate.compute_queue_family != UINT32_MAX &&
                candidate.compute_queue_index < occupied[candidate.compute_queue_family].size())
                occupied[candidate.compute_queue_family][candidate.compute_queue_index] = true;

            auto claim_queue = [&](QueueSlot& slot, VkQueueFlags required_flags, bool require_present)
            {
                for (uint32_t queue_family = 0; queue_family < family_count; ++queue_family)
                {
                    if (!families[queue_family].queueCount ||
                        (families[queue_family].queueFlags & required_flags) != required_flags)
                        continue;
                    if (require_present)
                    {
                        VkBool32 supported = VK_FALSE;
                        if (vk.get_surface_support(device, queue_family, surface, &supported) != VK_SUCCESS ||
                            !supported)
                            continue;
                    }
                    for (uint32_t queue_index = 0; queue_index < occupied[queue_family].size(); ++queue_index)
                    {
                        if (occupied[queue_family][queue_index])
                            continue;
                        occupied[queue_family][queue_index] = true;
                        slot = {queue_family, queue_index};
                        return true;
                    }
                }
                return false;
            };

            QueueSlot async_compute{};
            if (candidate.compute_queue_family != UINT32_MAX &&
                !(candidate.compute_queue_family == family && candidate.compute_queue_index == 0))
                async_compute = {candidate.compute_queue_family, candidate.compute_queue_index};
            else if (!claim_queue(async_compute, VK_QUEUE_COMPUTE_BIT, false))
                async_compute = {};
            QueueSlot frame_present{};
            QueueSlot image_acquire{};
            const bool has_frame_generation_queues = async_compute.family != UINT32_MAX &&
                claim_queue(frame_present, 0, true) && claim_queue(image_acquire, 0, false);
            if (has_frame_generation_queues)
            {
                candidate.frame_generation_present_family = frame_present.family;
                candidate.frame_generation_present_index = frame_present.index;
                candidate.frame_generation_compute_family = async_compute.family;
                candidate.frame_generation_compute_index = async_compute.index;
                candidate.frame_generation_acquire_family = image_acquire.family;
                candidate.frame_generation_acquire_index = image_acquire.index;
                candidate.frame_generation_queues = true;
            }
            QueueSlot transfer{};
            const auto claim_transfer_queue = [&](bool dedicated)
            {
                for (uint32_t transfer_family = 0; transfer_family < family_count; ++transfer_family)
                {
                    const VkQueueFlags flags = families[transfer_family].queueFlags;
                    if (!families[transfer_family].queueCount || !(flags & VK_QUEUE_TRANSFER_BIT) ||
                        ((flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0) != dedicated)
                        continue;
                    for (uint32_t queue_index = 0; queue_index < occupied[transfer_family].size(); ++queue_index)
                    {
                        if (occupied[transfer_family][queue_index])
                            continue;
                        occupied[transfer_family][queue_index] = true;
                        transfer = {transfer_family, queue_index};
                        return true;
                    }
                }
                return false;
            };
            if (!claim_transfer_queue(true))
                claim_transfer_queue(false);
            if (transfer.family != UINT32_MAX)
            {
                candidate.transfer_queue_family = transfer.family;
                candidate.transfer_queue_index = transfer.index;
                candidate.transfer_queue_flags = families[transfer.family].queueFlags;
            }
            vk.get_physical_properties(device, &candidate.properties);
            std::string requirements_error;
            if (!supports_game_formats(device, vk.get_format_properties, candidate.properties, requirements_error))
                continue;
            vk.get_physical_features(device, &candidate.features);
            candidate.ray_tracing = probe_ray_tracing_availability(device, extensions,
                candidate.properties.apiVersion,
                vk.get_physical_features2);
            candidate.bindless_textures = probe_bindless_texture_availability(device,
                extensions, candidate.properties.apiVersion, vk.get_physical_features2);
            candidate.timeline_semaphore = probe_timeline_semaphore_availability(device,
                extensions, candidate.properties.apiVersion, vk.get_physical_features2);
            candidate.hdr_metadata = supports_extension(extensions, VK_EXT_HDR_METADATA_EXTENSION_NAME);
            candidate.create_render_pass2_extension = supports_extension(extensions,
                VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
            candidate.depth_stencil_resolve_extension = supports_extension(extensions,
                VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
            if (!candidate.features.independentBlend)
                continue;
            vk.get_memory_properties(device, &candidate.memory);
            for (uint32_t heap = 0; heap < candidate.memory.memoryHeapCount; ++heap)
                if (candidate.memory.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    candidate.local_memory_bytes += candidate.memory.memoryHeaps[heap].size;
            candidate.score =
                score_device(candidate.properties, candidate.local_memory_bytes, families[family].queueCount);
            if (candidate.compute_queue_family != UINT32_MAX)
                candidate.score += 80;
#if defined(XRAY_HAS_FSR3_ANDROID)
            if (candidate.frame_generation_queues)
                candidate.score += 150;
#endif

            if (!selected.handle || candidate.score > best_score)
            {
                selected = candidate;
                best_score = candidate.score;
            }
        }
    }

    if (!selected.handle)
    {
        error = "no Vulkan graphics queue supports the surface and VK_KHR_swapchain";
        return false;
    }
    error.clear();
    return true;
}

bool create_logical_device(const PhysicalDevice &physical_device, const DeviceDispatch &vk, VkDevice &device,
                           VkQueue &graphics_queue, std::string &error, bool enable_ray_tracing,
                           uint32_t instance_api_version, bool enable_bindless_textures,
                           EnabledDeviceInfo* enabled_info, VkQueue* compute_queue,
                           FrameGenerationQueues* frame_generation_queues, VkQueue* transfer_queue)
{
    device = VK_NULL_HANDLE;
    graphics_queue = VK_NULL_HANDLE;
    if (compute_queue) *compute_queue = VK_NULL_HANDLE;
    if (frame_generation_queues) *frame_generation_queues = {};
    if (transfer_queue) *transfer_queue = VK_NULL_HANDLE;
    if (enabled_info) *enabled_info = {};
    if (!physical_device.handle || physical_device.graphics_present_family == UINT32_MAX || !vk.create_device ||
        !vk.get_device_proc)
    {
        error = "Vulkan device selection or creation procedures are invalid";
        return false;
    }

    if (!physical_device.features.independentBlend)
    {
        error = "Vulkan G-buffer requires independentBlend for multiply materials";
        return false;
    }
    if (enable_ray_tracing && (!physical_device.ray_tracing.acceleration_structure ||
        (!physical_device.ray_tracing.ray_query && !physical_device.ray_tracing.ray_tracing_pipeline)))
    {
        error = "selected Vulkan GPU has no enabled ray-query or ray-tracing-pipeline capability";
        return false;
    }
    const uint32_t effective_api_version = std::min(instance_api_version,
        physical_device.properties.apiVersion);
    if (enable_bindless_textures && (!physical_device.bindless_textures ||
        effective_api_version < VK_API_VERSION_1_1))
    {
        error = "bindless textures need Vulkan 1.1 and descriptor indexing features";
        return false;
    }
    if (enable_ray_tracing && (effective_api_version < VK_API_VERSION_1_1 ||
        (effective_api_version < VK_API_VERSION_1_2 &&
         !physical_device.ray_tracing.vulkan11_dependencies)))
    {
        error = "ray tracing needs a Vulkan 1.1 instance with dependency extensions or Vulkan 1.2";
        return false;
    }
    const bool request_compute_queue = compute_queue &&
        physical_device.compute_queue_family != UINT32_MAX;
    const bool request_transfer_queue = transfer_queue &&
        physical_device.transfer_queue_family != UINT32_MAX;
    const bool request_frame_generation_queues = frame_generation_queues &&
        physical_device.frame_generation_queues;
    std::map<uint32_t, uint32_t> requested_queue_counts;
    auto request_queue = [&](uint32_t family, uint32_t index)
    {
        requested_queue_counts[family] = std::max(requested_queue_counts[family], index + 1);
    };
    request_queue(physical_device.graphics_present_family, 0);
    if (request_compute_queue)
        request_queue(physical_device.compute_queue_family, physical_device.compute_queue_index);
    if (request_transfer_queue)
        request_queue(physical_device.transfer_queue_family, physical_device.transfer_queue_index);
    if (request_frame_generation_queues)
    {
        request_queue(physical_device.frame_generation_present_family,
            physical_device.frame_generation_present_index);
        request_queue(physical_device.frame_generation_compute_family,
            physical_device.frame_generation_compute_index);
        request_queue(physical_device.frame_generation_acquire_family,
            physical_device.frame_generation_acquire_index);
    }
    std::vector<std::vector<float>> queue_priorities(requested_queue_counts.size());
    std::vector<VkDeviceQueueCreateInfo> queue_infos(requested_queue_counts.size());
    size_t queue_info_index = 0;
    for (const auto& [family, count] : requested_queue_counts)
    {
        auto& priorities = queue_priorities[queue_info_index];
        priorities.assign(count, 1.0f);
        auto& queue_info = queue_infos[queue_info_index++];
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = count;
        queue_info.pQueuePriorities = priorities.data();
    }
    std::vector<const char*> extensions{VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    if (physical_device.hdr_metadata)
        extensions.push_back(VK_EXT_HDR_METADATA_EXTENSION_NAME);
    if (physical_device.timeline_semaphore && effective_api_version < VK_API_VERSION_1_2)
        extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    if (enable_bindless_textures && effective_api_version < VK_API_VERSION_1_2 &&
        !enable_ray_tracing)
        extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    if (enable_ray_tracing)
    {
        if (effective_api_version < VK_API_VERSION_1_2)
        {
            extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
            extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
            extensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
            extensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
        }
        extensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        extensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        if (physical_device.ray_tracing.ray_query)
            extensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        if (physical_device.ray_tracing.ray_tracing_pipeline)
            extensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    }
    if (effective_api_version >= VK_API_VERSION_1_1 && effective_api_version < VK_API_VERSION_1_2 &&
        physical_device.create_render_pass2_extension && physical_device.depth_stencil_resolve_extension)
    {
        extensions.push_back(VK_KHR_CREATE_RENDERPASS_2_EXTENSION_NAME);
        extensions.push_back(VK_KHR_DEPTH_STENCIL_RESOLVE_EXTENSION_NAME);
    }
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = static_cast<uint32_t>(queue_infos.size());
    device_info.pQueueCreateInfos = queue_infos.data();
    device_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    device_info.ppEnabledExtensionNames = extensions.data();
    // The world sampler enables this only on GPUs advertising the feature.
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.samplerAnisotropy = physical_device.features.samplerAnisotropy;
    enabled_features.independentBlend = VK_TRUE;
    enabled_features.multiDrawIndirect = physical_device.features.multiDrawIndirect;
    enabled_features.geometryShader = physical_device.features.geometryShader;
    enabled_features.tessellationShader = physical_device.features.tessellationShader;
    enabled_features.shaderStorageImageExtendedFormats =
        physical_device.features.shaderStorageImageExtendedFormats;
    device_info.pEnabledFeatures = &enabled_features;
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES};
    VkPhysicalDeviceBufferDeviceAddressFeatures address{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR acceleration{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR query{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR pipeline{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceDescriptorIndexingFeatures indexing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
    if (physical_device.timeline_semaphore)
    {
        timeline.timelineSemaphore = VK_TRUE;
        features2.pNext = &timeline;
    }
    if (enable_ray_tracing)
    {
        features2.features = enabled_features;
        if (physical_device.timeline_semaphore)
            timeline.pNext = &address;
        else
            features2.pNext = &address;
        address.bufferDeviceAddress = VK_TRUE;
        address.pNext = &acceleration;
        acceleration.accelerationStructure = VK_TRUE;
        if (physical_device.ray_tracing.ray_query)
        {
            acceleration.pNext = &query;
            query.rayQuery = VK_TRUE;
        }
        if (physical_device.ray_tracing.ray_tracing_pipeline)
        {
            if (physical_device.ray_tracing.ray_query) query.pNext = &pipeline;
            else acceleration.pNext = &pipeline;
            pipeline.rayTracingPipeline = VK_TRUE;
        }
    }
    if (enable_bindless_textures)
    {
        indexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
        indexing.runtimeDescriptorArray = VK_TRUE;
        indexing.descriptorBindingPartiallyBound = VK_TRUE;
        indexing.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
        if (enable_ray_tracing)
        {
            if (physical_device.ray_tracing.ray_tracing_pipeline) pipeline.pNext = &indexing;
            else if (physical_device.ray_tracing.ray_query) query.pNext = &indexing;
            else acceleration.pNext = &indexing;
        }
        else if (physical_device.timeline_semaphore)
            timeline.pNext = &indexing;
        else
            features2.pNext = &indexing;
    }
    if (enable_ray_tracing || enable_bindless_textures || physical_device.timeline_semaphore)
    {
        features2.features = enabled_features;
        device_info.pEnabledFeatures = nullptr;
        device_info.pNext = &features2;
    }

    if (vk.create_device(physical_device.handle, &device_info, nullptr, &device) != VK_SUCCESS)
    {
        error = "vkCreateDevice failed for the selected Vulkan GPU";
        return false;
    }

    const auto get_queue = reinterpret_cast<PFN_vkGetDeviceQueue>(vk.get_device_proc(device, "vkGetDeviceQueue"));
    if (!get_queue)
    {
        error = "vkGetDeviceQueue is unavailable";
        const auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(vk.get_device_proc(device, "vkDestroyDevice"));
        if (destroy)
            destroy(device, nullptr);
        device = VK_NULL_HANDLE;
        return false;
    }
    get_queue(device, physical_device.graphics_present_family, 0, &graphics_queue);
    if (!graphics_queue)
    {
        error = "the selected Vulkan graphics queue could not be acquired";
        const auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(vk.get_device_proc(device, "vkDestroyDevice"));
        if (destroy)
            destroy(device, nullptr);
        device = VK_NULL_HANDLE;
        return false;
    }
    if (request_compute_queue)
    {
        get_queue(device, physical_device.compute_queue_family, physical_device.compute_queue_index,
            compute_queue);
        if (!*compute_queue)
        {
            error = "the selected Vulkan compute queue could not be acquired";
            const auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(vk.get_device_proc(device, "vkDestroyDevice"));
            if (destroy)
                destroy(device, nullptr);
            device = VK_NULL_HANDLE;
            graphics_queue = VK_NULL_HANDLE;
            return false;
        }
    }
    if (request_transfer_queue)
    {
        get_queue(device, physical_device.transfer_queue_family, physical_device.transfer_queue_index,
            transfer_queue);
        if (!*transfer_queue)
        {
            error = "the selected Vulkan transfer queue could not be acquired";
            const auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(vk.get_device_proc(device, "vkDestroyDevice"));
            if (destroy)
                destroy(device, nullptr);
            device = VK_NULL_HANDLE;
            graphics_queue = VK_NULL_HANDLE;
            if (compute_queue) *compute_queue = VK_NULL_HANDLE;
            return false;
        }
    }
    if (request_frame_generation_queues)
    {
        get_queue(device, physical_device.frame_generation_present_family,
            physical_device.frame_generation_present_index, &frame_generation_queues->present);
        get_queue(device, physical_device.frame_generation_compute_family,
            physical_device.frame_generation_compute_index, &frame_generation_queues->async_compute);
        get_queue(device, physical_device.frame_generation_acquire_family,
            physical_device.frame_generation_acquire_index, &frame_generation_queues->image_acquire);
        if (!frame_generation_queues->present || !frame_generation_queues->async_compute ||
            !frame_generation_queues->image_acquire)
        {
            error = "the selected Vulkan frame-generation queues could not be acquired";
            const auto destroy = reinterpret_cast<PFN_vkDestroyDevice>(vk.get_device_proc(device, "vkDestroyDevice"));
            if (destroy)
                destroy(device, nullptr);
            device = VK_NULL_HANDLE;
            graphics_queue = VK_NULL_HANDLE;
            if (compute_queue) *compute_queue = VK_NULL_HANDLE;
            if (transfer_queue) *transfer_queue = VK_NULL_HANDLE;
            *frame_generation_queues = {};
            return false;
        }
        frame_generation_queues->enabled = true;
    }
    if (enabled_info)
    {
        enabled_info->extensions = extensions;
        enabled_info->buffer_device_address = enable_ray_tracing;
        // The engine currently enables sampled-image update-after-bind only.
        enabled_info->descriptor_binding_uniform_buffer_update_after_bind = false;
        enabled_info->timeline_semaphore = physical_device.timeline_semaphore;
    }
    error.clear();
    return true;
}
} // namespace xray::render::vulkan
