#include "src/Layers/xrRenderVK/VulkanHardware.h"

#include <cassert>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
bool sampled_depth = true;
bool color_attachment = true;
VkPhysicalDeviceFeatures enabled_features{};
bool enabled_rt_address{}, enabled_rt_acceleration{}, enabled_rt_query{}, enabled_rt_pipeline{};
bool enabled_bindless{};
std::vector<std::string> enabled_extensions;
bool rt_address = true, rt_acceleration = true, rt_query = true, rt_pipeline = true;
bool bindless_features = true;

void VKAPI_PTR ray_features(VkPhysicalDevice, VkPhysicalDeviceFeatures2* features)
{
    if (features->pNext &&
        static_cast<VkBaseOutStructure*>(features->pNext)->sType ==
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
    {
        auto* indexing = static_cast<VkPhysicalDeviceDescriptorIndexingFeatures*>(features->pNext);
        indexing->shaderSampledImageArrayNonUniformIndexing = bindless_features;
        indexing->runtimeDescriptorArray = bindless_features;
        indexing->descriptorBindingPartiallyBound = bindless_features;
        indexing->descriptorBindingSampledImageUpdateAfterBind = bindless_features;
        return;
    }
    auto* address = static_cast<VkPhysicalDeviceBufferDeviceAddressFeatures*>(features->pNext);
    auto* acceleration = static_cast<VkPhysicalDeviceAccelerationStructureFeaturesKHR*>(address->pNext);
    auto* query = static_cast<VkPhysicalDeviceRayQueryFeaturesKHR*>(acceleration->pNext);
    auto* pipeline = static_cast<VkPhysicalDeviceRayTracingPipelineFeaturesKHR*>(query->pNext);
    assert(address->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES);
    assert(acceleration->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR);
    assert(query->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR);
    assert(pipeline->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR);
    address->bufferDeviceAddress = rt_address;
    acceleration->accelerationStructure = rt_acceleration;
    query->rayQuery = rt_query;
    pipeline->rayTracingPipeline = rt_pipeline;
}

VkExtensionProperties extension(const char* name)
{
    VkExtensionProperties result{};
    std::strncpy(result.extensionName, name, VK_MAX_EXTENSION_NAME_SIZE - 1);
    return result;
}

VkResult VKAPI_PTR create_device(VkPhysicalDevice, const VkDeviceCreateInfo* info,
    const VkAllocationCallbacks*, VkDevice* output)
{
    enabled_extensions.clear();
    for (uint32_t i = 0; i < info->enabledExtensionCount; ++i)
        enabled_extensions.emplace_back(info->ppEnabledExtensionNames[i]);
    enabled_rt_address = enabled_rt_acceleration = enabled_rt_query = enabled_rt_pipeline = false;
    enabled_bindless = false;
    if (info->pNext)
    {
        assert(info->pEnabledFeatures == nullptr);
        const auto* features = static_cast<const VkPhysicalDeviceFeatures2*>(info->pNext);
        enabled_features = features->features;
        const VkBaseInStructure* next = static_cast<const VkBaseInStructure*>(features->pNext);
        while (next)
        {
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES)
                enabled_rt_address = reinterpret_cast<const VkPhysicalDeviceBufferDeviceAddressFeatures*>(next)
                    ->bufferDeviceAddress;
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR)
                enabled_rt_acceleration = reinterpret_cast<const VkPhysicalDeviceAccelerationStructureFeaturesKHR*>(next)
                    ->accelerationStructure;
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR)
                enabled_rt_query = reinterpret_cast<const VkPhysicalDeviceRayQueryFeaturesKHR*>(next)->rayQuery;
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR)
                enabled_rt_pipeline = reinterpret_cast<const VkPhysicalDeviceRayTracingPipelineFeaturesKHR*>(next)
                    ->rayTracingPipeline;
            if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES)
            {
                const auto* indexing = reinterpret_cast<const VkPhysicalDeviceDescriptorIndexingFeatures*>(next);
                enabled_bindless = indexing->shaderSampledImageArrayNonUniformIndexing &&
                    indexing->runtimeDescriptorArray && indexing->descriptorBindingPartiallyBound &&
                    indexing->descriptorBindingSampledImageUpdateAfterBind;
            }
            next = next->pNext;
        }
    }
    else enabled_features = *info->pEnabledFeatures;
    *output = reinterpret_cast<VkDevice>(uintptr_t(11));
    return VK_SUCCESS;
}
void VKAPI_PTR get_queue(VkDevice, uint32_t, uint32_t, VkQueue* output)
{ *output = reinterpret_cast<VkQueue>(uintptr_t(12)); }
PFN_vkVoidFunction VKAPI_PTR device_proc(VkDevice, const char* name)
{
    if (std::strcmp(name, "vkGetDeviceQueue") == 0)
        return reinterpret_cast<PFN_vkVoidFunction>(get_queue);
    return nullptr;
}

void VKAPI_PTR formats(VkPhysicalDevice, VkFormat format, VkFormatProperties* output)
{
    *output = {};
    if (format == VK_FORMAT_R8G8B8A8_UNORM)
        output->optimalTilingFeatures = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            (color_attachment ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT : 0);
    if (format == VK_FORMAT_D32_SFLOAT || format == VK_FORMAT_D16_UNORM)
        output->optimalTilingFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            (sampled_depth ? VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT : 0);
}
}

int main()
{
    VkPhysicalDeviceProperties properties{};
    properties.limits.maxPushConstantsSize = 256;
    properties.limits.maxColorAttachments = 3;
    properties.limits.maxImageDimension2D = 4096;
    properties.limits.maxPerStageDescriptorSamplers = 20;
    properties.limits.maxPerStageDescriptorSampledImages = 20;
    std::string error;
    assert(supports_game_formats(VK_NULL_HANDLE, formats, properties, error) && error.empty());
    properties.limits.maxPerStageDescriptorSampledImages = 19;
    assert(!supports_game_formats(VK_NULL_HANDLE, formats, properties, error) &&
        error.find("20 fragment sampled images") != std::string::npos);
    properties.limits.maxPerStageDescriptorSampledImages = 20;
    properties.limits.maxPerStageDescriptorSamplers = 19;
    assert(!supports_game_formats(VK_NULL_HANDLE, formats, properties, error));
    properties.limits.maxPerStageDescriptorSamplers = 20;
    properties.limits.maxColorAttachments = 2;
    assert(!supports_game_formats(VK_NULL_HANDLE, formats, properties, error) &&
        error.find("three color attachments") != std::string::npos);
    properties.limits.maxColorAttachments = 3;
    sampled_depth = false;
    assert(!supports_game_formats(VK_NULL_HANDLE, formats, properties, error) && !error.empty());
    sampled_depth = true;
    color_attachment = false;
    assert(!supports_game_formats(VK_NULL_HANDLE, formats, properties, error));
    color_attachment = true;
    properties.limits.maxPushConstantsSize = 128;
    assert(!supports_game_formats(VK_NULL_HANDLE, formats, properties, error));

    const auto fake_gpu = reinterpret_cast<VkPhysicalDevice>(uintptr_t(10));
    std::vector<VkExtensionProperties> rt_extensions{
        extension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME),
        extension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME),
        extension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME),
        extension(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME),
        extension(VK_KHR_SPIRV_1_4_EXTENSION_NAME),
        extension(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME),
        extension(VK_KHR_RAY_QUERY_EXTENSION_NAME),
        extension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME)};
    auto rt = probe_ray_tracing_availability(fake_gpu, rt_extensions, VK_API_VERSION_1_1, ray_features);
    assert(rt.acceleration_structure && rt.ray_query && rt.ray_tracing_pipeline);
    assert(probe_bindless_texture_availability(fake_gpu, rt_extensions,
        VK_API_VERSION_1_1, ray_features));
    bindless_features = false;
    assert(!probe_bindless_texture_availability(fake_gpu, rt_extensions,
        VK_API_VERSION_1_1, ray_features));
    bindless_features = true;
    assert(!probe_bindless_texture_availability(fake_gpu,
        std::vector<VkExtensionProperties>{}, VK_API_VERSION_1_1, ray_features));
    assert(probe_bindless_texture_availability(fake_gpu,
        std::vector<VkExtensionProperties>{}, VK_API_VERSION_1_2, ray_features));
    rt = probe_ray_tracing_availability(fake_gpu, rt_extensions, VK_API_VERSION_1_0, ray_features);
    assert(!rt.acceleration_structure && !rt.ray_query && !rt.ray_tracing_pipeline);
    const auto without = [&](const char* name)
    {
        auto copy = rt_extensions;
        copy.erase(std::remove_if(copy.begin(), copy.end(), [name](const VkExtensionProperties& value)
            { return std::strcmp(value.extensionName, name) == 0; }), copy.end());
        return copy;
    };
    rt = probe_ray_tracing_availability(fake_gpu,
        without(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME), VK_API_VERSION_1_1, ray_features);
    assert(rt.acceleration_structure && rt.ray_query && !rt.ray_tracing_pipeline);
    rt = probe_ray_tracing_availability(fake_gpu,
        without(VK_KHR_SPIRV_1_4_EXTENSION_NAME), VK_API_VERSION_1_1, ray_features);
    assert(rt.acceleration_structure && !rt.ray_query && !rt.ray_tracing_pipeline);
    rt = probe_ray_tracing_availability(fake_gpu,
        without(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME), VK_API_VERSION_1_1, ray_features);
    assert(!rt.acceleration_structure && !rt.ray_query && !rt.ray_tracing_pipeline);
    rt = probe_ray_tracing_availability(fake_gpu,
        without(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME), VK_API_VERSION_1_2, ray_features);
    assert(rt.acceleration_structure && rt.ray_query && rt.ray_tracing_pipeline);
    rt_address = false;
    rt = probe_ray_tracing_availability(fake_gpu, rt_extensions, VK_API_VERSION_1_1, ray_features);
    assert(!rt.acceleration_structure && !rt.ray_query && !rt.ray_tracing_pipeline);
    rt_address = true;
    rt_acceleration = false;
    rt = probe_ray_tracing_availability(fake_gpu, rt_extensions, VK_API_VERSION_1_1, ray_features);
    assert(!rt.acceleration_structure && !rt.ray_query && !rt.ray_tracing_pipeline);
    rt = probe_ray_tracing_availability(fake_gpu, rt_extensions, VK_API_VERSION_1_1, nullptr);
    assert(!rt.acceleration_structure && !rt.ray_query && !rt.ray_tracing_pipeline);

    VkPhysicalDeviceProperties ray_properties{};
    ray_properties.limits.maxPerStageDescriptorStorageBuffers = 5;
    ray_properties.limits.maxDescriptorSetStorageBuffers = 5;
    VkPhysicalDeviceDescriptorIndexingProperties ray_indexing{};
    ray_indexing.maxPerStageDescriptorUpdateAfterBindStorageBuffers = 5;
    ray_indexing.maxDescriptorSetUpdateAfterBindStorageBuffers = 5;
    ray_indexing.maxPerStageDescriptorUpdateAfterBindSampledImages = 64;
    ray_indexing.maxPerStageDescriptorUpdateAfterBindSamplers = 64;
    ray_indexing.maxDescriptorSetUpdateAfterBindSampledImages = 64;
    ray_indexing.maxDescriptorSetUpdateAfterBindSamplers = 64;
    ray_indexing.maxUpdateAfterBindDescriptorsInAllPools = 64;
    ray_indexing.maxPerStageUpdateAfterBindResources = 70;
    VkPhysicalDeviceAccelerationStructurePropertiesKHR ray_acceleration{};
    ray_acceleration.maxPerStageDescriptorUpdateAfterBindAccelerationStructures = 1;
    ray_acceleration.maxDescriptorSetUpdateAfterBindAccelerationStructures = 1;
    assert(ray_query_texture_capacity(ray_properties, ray_indexing, ray_acceleration) == 64);
    ray_indexing.maxPerStageUpdateAfterBindResources = 20;
    assert(ray_query_texture_capacity(ray_properties, ray_indexing, ray_acceleration) == 14);
    ray_indexing.maxDescriptorSetUpdateAfterBindStorageBuffers = 4;
    assert(ray_query_texture_capacity(ray_properties, ray_indexing, ray_acceleration) == 0);
    ray_indexing.maxDescriptorSetUpdateAfterBindStorageBuffers = 5;
    ray_acceleration.maxDescriptorSetUpdateAfterBindAccelerationStructures = 0;
    assert(ray_query_texture_capacity(ray_properties, ray_indexing, ray_acceleration) == 0);

    PhysicalDevice selected;
    selected.handle = fake_gpu;
    selected.graphics_present_family = 0;
    selected.features.independentBlend = VK_TRUE;
    selected.features.geometryShader = VK_TRUE;
    selected.features.tessellationShader = VK_TRUE;
    selected.features.shaderStorageImageExtendedFormats = VK_TRUE;
    VkDevice device{};
    VkQueue queue{};
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error));
    assert(device && queue && enabled_features.independentBlend);
    assert(enabled_features.geometryShader && enabled_features.tessellationShader &&
        enabled_features.shaderStorageImageExtendedFormats);
    selected.features.geometryShader = VK_FALSE;
    selected.features.tessellationShader = VK_FALSE;
    selected.features.shaderStorageImageExtendedFormats = VK_FALSE;
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error));
    assert(!enabled_features.geometryShader && !enabled_features.tessellationShader &&
        !enabled_features.shaderStorageImageExtendedFormats);
    selected.properties.apiVersion = VK_API_VERSION_1_2;
    selected.ray_tracing = {true, true, true};
    selected.bindless_textures = true;
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error,
        true, VK_API_VERSION_1_2, true));
    assert(enabled_rt_address && enabled_rt_acceleration && enabled_rt_query &&
        enabled_rt_pipeline && enabled_bindless);
    assert(enabled_extensions.size() == 5);
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error,
        false, VK_API_VERSION_1_2, true));
    assert(!enabled_rt_address && enabled_bindless && enabled_extensions.size() == 1);
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error,
        false, VK_API_VERSION_1_1, true));
    assert(enabled_bindless && enabled_extensions.size() == 2);
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error,
        true, VK_API_VERSION_1_2));
    assert(enabled_rt_address && enabled_rt_acceleration && enabled_rt_query && enabled_rt_pipeline);
    assert(enabled_extensions.size() == 5);
    selected.ray_tracing.ray_tracing_pipeline = false;
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error,
        true, VK_API_VERSION_1_2));
    assert(enabled_rt_address && enabled_rt_acceleration && enabled_rt_query && !enabled_rt_pipeline);
    assert(enabled_extensions.size() == 4);
    assert(!create_logical_device(selected, {create_device, device_proc}, device, queue, error, true));
    selected.ray_tracing.vulkan11_dependencies = true;
    assert(create_logical_device(selected, {create_device, device_proc}, device, queue, error,
        true, VK_API_VERSION_1_1));
    assert(enabled_extensions.size() == 8);
    selected.ray_tracing.ray_query = false;
    assert(!create_logical_device(selected, {create_device, device_proc}, device, queue, error,
        true, VK_API_VERSION_1_2));
}
