#include "src/Layers/xrRenderVK/MotionVectors.h"
#include "src/Layers/xrRenderVK/ObjectVelocityPass.h"
#include "src/Layers/xrRenderVK/LevelModels.h"
#include "src/Layers/xrRenderVK/ModelGeometry.h"
#include "src/Layers/xrRenderVK/DeferredPass.h"
#include <vulkan/vulkan.h>
#include <array>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
void check(VkResult result)
{
    if (result != VK_SUCCESS) throw std::runtime_error("Vulkan motion result " + std::to_string(result));
}

struct Fixture
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkImage depth{};
    VkImageView depth_view{};
    VkDeviceMemory depth_memory{};
    VkImage cutout_texture{};
    VkImageView cutout_view{};
    VkDeviceMemory cutout_memory{};
    VkSampler cutout_sampler{};
    VkDescriptorSetLayout cutout_layout{};
    VkDescriptorPool cutout_pool{};
    VkDescriptorSet cutout_set{};
    VkDescriptorSetLayout pose_layout{};
    VkDescriptorPool pose_pool{};
    VkDescriptorSet pose_sets[2]{};
    VkBuffer pose_buffers[2]{};
    VkDeviceMemory pose_memories[2]{};
    VkBuffer upload{}, readback{}, vertices{}, previous_vertices{}, indices{}, cutout_upload{}, indirect_commands{};
    VkDeviceMemory upload_memory{}, readback_memory{}, vertex_memory{}, previous_vertex_memory{}, index_memory{},
        cutout_upload_memory{}, indirect_commands_memory{};
    VkPhysicalDeviceMemoryProperties memory{};

    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (pose_pool) vkDestroyDescriptorPool(device, pose_pool, nullptr);
            if (pose_layout) vkDestroyDescriptorSetLayout(device, pose_layout, nullptr);
            for (uint32_t i = 0; i < 2; ++i)
            {
                if (pose_buffers[i]) vkDestroyBuffer(device, pose_buffers[i], nullptr);
                if (pose_memories[i]) vkFreeMemory(device, pose_memories[i], nullptr);
            }
            if (cutout_pool) vkDestroyDescriptorPool(device, cutout_pool, nullptr);
            if (cutout_layout) vkDestroyDescriptorSetLayout(device, cutout_layout, nullptr);
            if (cutout_sampler) vkDestroySampler(device, cutout_sampler, nullptr);
            if (cutout_view) vkDestroyImageView(device, cutout_view, nullptr);
            if (cutout_texture) vkDestroyImage(device, cutout_texture, nullptr);
            if (cutout_memory) vkFreeMemory(device, cutout_memory, nullptr);
            if (cutout_upload) vkDestroyBuffer(device, cutout_upload, nullptr);
            if (cutout_upload_memory) vkFreeMemory(device, cutout_upload_memory, nullptr);
            if (depth_view) vkDestroyImageView(device, depth_view, nullptr);
            if (depth) vkDestroyImage(device, depth, nullptr);
            if (depth_memory) vkFreeMemory(device, depth_memory, nullptr);
            if (upload) vkDestroyBuffer(device, upload, nullptr);
            if (readback) vkDestroyBuffer(device, readback, nullptr);
            if (vertices) vkDestroyBuffer(device, vertices, nullptr);
            if (previous_vertices) vkDestroyBuffer(device, previous_vertices, nullptr);
            if (indices) vkDestroyBuffer(device, indices, nullptr);
            if (indirect_commands) vkDestroyBuffer(device, indirect_commands, nullptr);
            if (upload_memory) vkFreeMemory(device, upload_memory, nullptr);
            if (readback_memory) vkFreeMemory(device, readback_memory, nullptr);
            if (vertex_memory) vkFreeMemory(device, vertex_memory, nullptr);
            if (previous_vertex_memory) vkFreeMemory(device, previous_vertex_memory, nullptr);
            if (index_memory) vkFreeMemory(device, index_memory, nullptr);
            if (indirect_commands_memory) vkFreeMemory(device, indirect_commands_memory, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags)
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & flags) == flags)
            {
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                info.allocationSize = requirements.size;
                info.memoryTypeIndex = i;
                VkDeviceMemory result{};
                check(vkAllocateMemory(device, &info, nullptr, &result));
                return result;
            }
        throw std::runtime_error("motion GPU test memory type unavailable");
    }

    void buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer& result, VkDeviceMemory& allocation)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        check(vkCreateBuffer(device, &info, nullptr, &result));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, result, &requirements);
        allocation = allocate(requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, result, allocation, 0));
    }
};

void image_barrier(VkCommandBuffer command, VkImage image, VkImageAspectFlags aspect,
    VkImageLayout before_layout, VkImageLayout after_layout, VkAccessFlags before_access,
    VkAccessFlags after_access, VkPipelineStageFlags before_stage, VkPipelineStageFlags after_stage)
{
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.srcAccessMask = before_access;
    barrier.dstAccessMask = after_access;
    barrier.oldLayout = before_layout;
    barrier.newLayout = after_layout;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange = {aspect, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, before_stage, after_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}
}

int run_test(int argc, char** argv)
{
    Fixture vk;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &app;
    check(vkCreateInstance(&instance_info, nullptr, &vk.instance));
    uint32_t count{};
    check(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
    if (!count) return 77;
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(vk.instance, &count, devices.data()));
    const int selected = argc > 1 ? std::stoi(argv[1]) : -1;
    const bool fallback = argc > 2 && std::string(argv[2]) == "fallback";
    const bool deform_velocity = argc > 2 && std::string(argv[2]) == "deform";
    const bool wind_velocity = argc > 2 && std::string(argv[2]) == "wind";
    const bool offset_velocity = argc > 2 && std::string(argv[2]) == "offset";
    const bool cutout_velocity = argc > 2 &&
        (std::string(argv[2]) == "cutout" || std::string(argv[2]) == "cutout_solid");
    const bool indirect_velocity = argc > 2 && std::string(argv[2]) == "indirect";
    const bool skinned_cutout = argc > 2 && std::string(argv[2]) == "skinned_cutout";
    const bool solid_cutout = argc > 2 &&
        (std::string(argv[2]) == "cutout_solid" || skinned_cutout);
    const bool blended_skinning = argc > 2 && std::string(argv[2]) == "skinned_blend";
    const bool skinned_velocity = argc > 2 &&
        (std::string(argv[2]) == "skinned" || blended_skinning || skinned_cutout);
    const bool object_velocity = argc > 2 &&
        (std::string(argv[2]) == "object" || deform_velocity || wind_velocity || offset_velocity ||
         cutout_velocity || indirect_velocity || skinned_velocity);
    const bool prefer_reprojected_depth = argc > 2 &&
        (std::string(argv[2]) == "reprojected" || object_velocity);
    const bool rgba = fallback || prefer_reprojected_depth;
    uint32_t family = UINT32_MAX;
    VkQueueFlags selected_family_flags{};
    for (size_t index = 0; index < devices.size(); ++index)
    {
        if (selected >= 0 && size_t(selected) != index) continue;
        VkFormatProperties velocity_support{}, depth_support{};
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(devices[index], &features);
        if (!fallback && !features.shaderStorageImageExtendedFormats) continue;
        vkGetPhysicalDeviceFormatProperties(devices[index], rgba
            ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R16G16_SFLOAT, &velocity_support);
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_D32_SFLOAT, &depth_support);
        if ((velocity_support.optimalTilingFeatures &
                (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
                (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ||
            !(depth_support.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) continue;
        if (object_velocity &&
            (!(velocity_support.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) ||
             !(depth_support.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))) continue;
        uint32_t families{};
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, queues.data());
        for (uint32_t i = 0; i < families; ++i)
            if ((queues[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            {
                vk.physical = devices[index];
                family = i;
                selected_family_flags = queues[i].queueFlags;
                break;
            }
        if (vk.physical) break;
    }
    if (!vk.physical) return 77;
    VkPhysicalDeviceFeatures physical_features{};
    VkPhysicalDeviceProperties physical_properties{};
    vkGetPhysicalDeviceFeatures(vk.physical, &physical_features);
    vkGetPhysicalDeviceProperties(vk.physical, &physical_properties);
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkPhysicalDeviceFeatures enabled_features{};
    enabled_features.shaderStorageImageExtendedFormats = fallback ? VK_FALSE : VK_TRUE;
    enabled_features.multiDrawIndirect = physical_features.multiDrawIndirect;
    device_info.pEnabledFeatures = &enabled_features;
    check(vkCreateDevice(vk.physical, &device_info, nullptr, &vk.device));
    vkGetDeviceQueue(vk.device, family, 0, &vk.queue);
    vkGetPhysicalDeviceMemoryProperties(vk.physical, &vk.memory);
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    check(vkCreateCommandPool(vk.device, &pool_info, nullptr, &vk.pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = vk.pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(vk.device, &command_info, &vk.command));

    VkImageCreateInfo depth_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    depth_info.imageType = VK_IMAGE_TYPE_2D;
    depth_info.format = VK_FORMAT_D32_SFLOAT;
    depth_info.extent = {2, 2, 1};
    depth_info.mipLevels = depth_info.arrayLayers = 1;
    depth_info.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    depth_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
        (object_velocity ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : 0);
    check(vkCreateImage(vk.device, &depth_info, nullptr, &vk.depth));
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(vk.device, vk.depth, &requirements);
    vk.depth_memory = vk.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    check(vkBindImageMemory(vk.device, vk.depth, vk.depth_memory, 0));
    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = vk.depth;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_D32_SFLOAT;
    view_info.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(vk.device, &view_info, nullptr, &vk.depth_view));
    vk.buffer(4 * sizeof(float), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.upload, vk.upload_memory);
    vk.buffer(16 * sizeof(uint16_t), VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.readback, vk.readback_memory);
    const std::array<float, 4> depth{.5f, .5f, 1.f, .5f};
    void* mapped{};
    check(vkMapMemory(vk.device, vk.upload_memory, 0, sizeof(depth), 0, &mapped));
    std::memcpy(mapped, depth.data(), sizeof(depth));
    vkUnmapMemory(vk.device, vk.upload_memory);

    FrameDispatch dispatch{};
    dispatch.get_format_properties = vkGetPhysicalDeviceFormatProperties;
    dispatch.create_image = vkCreateImage;
    dispatch.destroy_image = vkDestroyImage;
    dispatch.get_image_memory_requirements = vkGetImageMemoryRequirements;
    dispatch.allocate_memory = vkAllocateMemory;
    dispatch.free_memory = vkFreeMemory;
    dispatch.bind_image_memory = vkBindImageMemory;
    dispatch.create_image_view = vkCreateImageView;
    dispatch.destroy_image_view = vkDestroyImageView;
    dispatch.cmd_pipeline_barrier = vkCmdPipelineBarrier;
    FrameImagePool image_pool;
    if (!image_pool.initialize(vk.device, vk.memory,
            {vkCreateImage, vkDestroyImage, vkGetImageMemoryRequirements,
             vkAllocateMemory, vkFreeMemory, vkBindImageMemory}))
        throw std::runtime_error("motion image pool initialization failed");
    MotionVectors motion;
    std::string error;
    if (!motion.initialize(vk.physical, vk.device, vkGetDeviceProcAddr, {2, 2}, {vk.depth_view}, {vk.depth},
            vk.memory, dispatch, VK_NULL_HANDLE, !fallback, error, &image_pool,
            prefer_reprojected_depth))
        throw std::runtime_error(error);
    if (motion.format() != (rgba ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R16G16_SFLOAT))
        throw std::runtime_error("motion GPU format selection mismatch");
    if (cutout_velocity || skinned_velocity)
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 1;
        layout_info.pBindings = &binding;
        check(vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &vk.cutout_layout));
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 1;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &vk.cutout_pool));
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = vk.cutout_pool;
        set_info.descriptorSetCount = 1;
        set_info.pSetLayouts = &vk.cutout_layout;
        check(vkAllocateDescriptorSets(vk.device, &set_info, &vk.cutout_set));
        VkImageCreateInfo texture_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        texture_info.imageType = VK_IMAGE_TYPE_2D;
        texture_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        texture_info.extent = {2, 1, 1};
        texture_info.mipLevels = texture_info.arrayLayers = 1;
        texture_info.samples = VK_SAMPLE_COUNT_1_BIT;
        texture_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        texture_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        check(vkCreateImage(vk.device, &texture_info, nullptr, &vk.cutout_texture));
        vkGetImageMemoryRequirements(vk.device, vk.cutout_texture, &requirements);
        vk.cutout_memory = vk.allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindImageMemory(vk.device, vk.cutout_texture, vk.cutout_memory, 0));
        view_info.image = vk.cutout_texture;
        view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        check(vkCreateImageView(vk.device, &view_info, nullptr, &vk.cutout_view));
        VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler_info.magFilter = sampler_info.minFilter = VK_FILTER_NEAREST;
        sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler_info.addressModeU = sampler_info.addressModeV =
            sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler_info.maxLod = 1.f;
        check(vkCreateSampler(vk.device, &sampler_info, nullptr, &vk.cutout_sampler));
        VkDescriptorImageInfo descriptor{vk.cutout_sampler, vk.cutout_view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = vk.cutout_set;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &descriptor;
        vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
        vk.buffer(8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.cutout_upload, vk.cutout_upload_memory);
        const uint8_t texel[]{255, 255, 255, 0, 255, 255, 255, 255};
        void* texel_data{};
        check(vkMapMemory(vk.device, vk.cutout_upload_memory, 0, sizeof(texel), 0, &texel_data));
        std::memcpy(texel_data, texel, sizeof(texel));
        vkUnmapMemory(vk.device, vk.cutout_upload_memory);
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(vk.command, &begin));
        image_barrier(vk.command, vk.cutout_texture, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {2, 1, 1};
        vkCmdCopyBufferToImage(vk.command, vk.cutout_upload, vk.cutout_texture,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        image_barrier(vk.command, vk.cutout_texture, VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
        check(vkEndCommandBuffer(vk.command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &vk.command;
        check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        check(vkResetCommandBuffer(vk.command, 0));
    }
    if (skinned_velocity)
    {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layout_info.bindingCount = 1;
        layout_info.pBindings = &binding;
        check(vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &vk.pose_layout));
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool_info.maxSets = 2;
        pool_info.poolSizeCount = 1;
        pool_info.pPoolSizes = &pool_size;
        check(vkCreateDescriptorPool(vk.device, &pool_info, nullptr, &vk.pose_pool));
        const VkDescriptorSetLayout layouts[]{vk.pose_layout, vk.pose_layout};
        VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_info.descriptorPool = vk.pose_pool;
        set_info.descriptorSetCount = 2;
        set_info.pSetLayouts = layouts;
        check(vkAllocateDescriptorSets(vk.device, &set_info, vk.pose_sets));
        for (uint32_t i = 0; i < 2; ++i)
        {
            std::array<float, 48> matrices{};
            for (uint32_t bone = 0; bone < 3; ++bone)
                matrices[bone * 16] = matrices[bone * 16 + 5] =
                    matrices[bone * 16 + 10] = matrices[bone * 16 + 15] = 1.f;
            if (i == 1) matrices[blended_skinning ? 28 : 12] = blended_skinning ? 1.f : .5f;
            vk.buffer(sizeof(matrices), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                vk.pose_buffers[i], vk.pose_memories[i]);
            void* mapped_pose{};
            check(vkMapMemory(vk.device, vk.pose_memories[i], 0, sizeof(matrices), 0, &mapped_pose));
            std::memcpy(mapped_pose, matrices.data(), sizeof(matrices));
            vkUnmapMemory(vk.device, vk.pose_memories[i]);
            VkDescriptorBufferInfo buffer_info{vk.pose_buffers[i], 0, sizeof(matrices)};
            VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write.dstSet = vk.pose_sets[i];
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write.pBufferInfo = &buffer_info;
            vkUpdateDescriptorSets(vk.device, 1, &write, 0, nullptr);
        }
    }
    ObjectVelocityPass object_pass;
    if (object_velocity)
    {
        if (!motion.color_attachment_ready() ||
            !object_pass.initialize(vk.device, vkGetDeviceProcAddr, {2, 2}, VK_FORMAT_D32_SFLOAT,
                {motion.view(0)}, {vk.depth_view}, VK_NULL_HANDLE, error,
                vk.cutout_layout, vk.pose_layout,
                enabled_features.multiDrawIndirect ? physical_properties.limits.maxDrawIndirectCount : 1u))
            throw std::runtime_error("object velocity initialization failed: " + error);
        std::array<LevelVertex, 6> triangle{};
        const uint32_t base = offset_velocity ? 3 : 0;
        triangle[base].position[0] = -1.f; triangle[base].position[1] = -1.f;
        triangle[base + 1].position[0] = 3.f; triangle[base + 1].position[1] = -1.f;
        triangle[base + 2].position[0] = -1.f; triangle[base + 2].position[1] = 3.f;
        for (uint32_t i = base; i < base + 3; ++i) triangle[i].position[2] = .5f;
        if (wind_velocity)
            for (uint32_t i = base; i < base + 3; ++i)
            { triangle[i].lightmap_uv[0] = 1.f; triangle[i].lightmap_uv[1] = triangle[i].position[1] - 1.f; }
        std::array<ModelVertex, 6> skin_triangle{};
        if (skinned_velocity)
            for (uint32_t i = base; i < base + 3; ++i)
            {
                std::memcpy(skin_triangle[i].position, triangle[i].position, sizeof(triangle[i].position));
                if (blended_skinning)
                {
                    skin_triangle[i].bones[1] = 1;
                    skin_triangle[i].weights[0] = skin_triangle[i].weights[1] = .5f;
                }
            }
        const uint32_t triangle_indices[]{0, 1, 2};
        vk.buffer(skinned_velocity ? sizeof(skin_triangle) : sizeof(triangle),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk.vertices, vk.vertex_memory);
        vk.buffer(sizeof(triangle_indices), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, vk.indices, vk.index_memory);
        void* mapped{};
        check(vkMapMemory(vk.device, vk.vertex_memory, 0,
            skinned_velocity ? sizeof(skin_triangle) : sizeof(triangle), 0, &mapped));
        if (skinned_velocity) std::memcpy(mapped, skin_triangle.data(), sizeof(skin_triangle));
        else std::memcpy(mapped, triangle.data(), sizeof(triangle));
        vkUnmapMemory(vk.device, vk.vertex_memory);
        if (deform_velocity)
        {
            for (uint32_t i = base; i < base + 3; ++i) triangle[i].position[0] += .5f;
            vk.buffer(sizeof(triangle), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                vk.previous_vertices, vk.previous_vertex_memory);
            check(vkMapMemory(vk.device, vk.previous_vertex_memory, 0, sizeof(triangle), 0, &mapped));
            std::memcpy(mapped, triangle.data(), sizeof(triangle));
            vkUnmapMemory(vk.device, vk.previous_vertex_memory);
        }
        check(vkMapMemory(vk.device, vk.index_memory, 0, sizeof(triangle_indices), 0, &mapped));
        std::memcpy(mapped, triangle_indices, sizeof(triangle_indices));
        vkUnmapMemory(vk.device, vk.index_memory);
        if (indirect_velocity)
        {
            const std::array<VkDrawIndexedIndirectCommand, 2> commands{{
                {3, 1, 0, 0, 0},
                {0, 0, 0, 0, 0}}};
            vk.buffer(sizeof(commands), VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                vk.indirect_commands, vk.indirect_commands_memory);
            check(vkMapMemory(vk.device, vk.indirect_commands_memory, 0, sizeof(commands), 0, &mapped));
            std::memcpy(mapped, commands.data(), sizeof(commands));
            vkUnmapMemory(vk.device, vk.indirect_commands_memory);
        }
    }

    std::array<float, 16> identity{};
    identity[0] = identity[5] = identity[10] = identity[15] = 1.f;
    auto shifted = identity, shifted_inverse = identity;
    shifted[12] = .25f;
    shifted_inverse[12] = -.25f;
    auto teleported = identity, teleported_inverse = identity;
    teleported[12] = 2.25f;
    teleported_inverse[12] = -2.25f;
    for (uint32_t iteration = 0; iteration < 4; ++iteration)
    {
        if (iteration == 2) motion.reset_history();
        if (iteration == 1) motion.set_view(identity.data(), identity.data());
        else if (iteration == 3) motion.set_view(teleported.data(), teleported_inverse.data());
        else motion.set_view(shifted.data(), shifted_inverse.data());
        if (iteration == 3 && motion.history_valid())
            throw std::runtime_error("camera cut did not clear motion history");
        check(vkResetCommandBuffer(vk.command, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(vk.command, &begin));
        image_barrier(vk.command, vk.depth, VK_IMAGE_ASPECT_DEPTH_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        region.imageExtent = {2, 2, 1};
        vkCmdCopyBufferToImage(vk.command, vk.upload, vk.depth,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        image_barrier(vk.command, vk.depth, VK_IMAGE_ASPECT_DEPTH_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        FrameRecordingContext frame{};
        frame.command_buffer = vk.command;
        frame.command_queue_flags = selected_family_flags;
        frame.graphics_queue_flags = selected_family_flags;
        std::string record_error;
        if (!motion.record(frame, &record_error))
            throw std::runtime_error("motion record failed: " + record_error);
        if (object_velocity && iteration == 1)
        {
            image_barrier(vk.command, vk.depth, VK_IMAGE_ASPECT_DEPTH_BIT,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT);
            frame.extent = {2, 2};
            auto current = identity, previous = identity;
            if (!deform_velocity && !skinned_velocity && !wind_velocity) previous[12] = .5f;
            if (!object_pass.begin(frame)) throw std::runtime_error("object velocity begin failed");
            const float shifted_uv[8]{1.f, 0.f, 0.f, 1.f, 0.f, 1.f, 0.f, 0.f};
            TreeWindParameters current_wind{}, previous_wind{};
            previous_wind.direction[0] = .5f;
            if (skinned_velocity)
                object_pass.draw_skinned(frame, vk.vertices, vk.indices, 3, 0,
                    current.data(), previous.data(), vk.pose_sets[0], vk.pose_sets[1],
                    skinned_cutout ? vk.cutout_set : VK_NULL_HANDLE, .5f,
                    skinned_cutout ? shifted_uv : nullptr);
            else
                object_pass.draw(frame, vk.vertices,
                    deform_velocity ? vk.previous_vertices : vk.vertices, vk.indices, 3, 0,
                    offset_velocity ? 3 : 0,
                    current.data(), previous.data(), vk.cutout_set, .5f,
                    solid_cutout ? shifted_uv : nullptr,
                    wind_velocity ? &current_wind : nullptr, wind_velocity ? &previous_wind : nullptr,
                    indirect_velocity ? vk.indirect_commands : VK_NULL_HANDLE, 0,
                    indirect_velocity ? 2 : 0);
            object_pass.end(frame);
        }
        image_barrier(vk.command, motion.image(0), VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy output{};
        output.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        output.imageExtent = {2, 2, 1};
        vkCmdCopyImageToBuffer(vk.command, motion.image(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 1, &output);
        check(vkEndCommandBuffer(vk.command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &vk.command;
        check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        check(vkMapMemory(vk.device, vk.readback_memory, 0,
            (rgba ? 16 : 8) * sizeof(uint16_t), 0, &mapped));
        const auto* values = static_cast<const uint16_t*>(mapped);
        const uint32_t stride = rgba ? 4 : 2;
        for (uint32_t pixel = 0; pixel < 4; ++pixel)
        {
            const uint16_t expected_x = iteration == 1 && pixel != 2 ?
                (object_velocity && (!cutout_velocity || solid_cutout) ? 0x3400 : 0x3000) : 0;
            const uint16_t actual_x = values[pixel * stride];
            const bool x_matches = object_velocity && (!cutout_velocity || solid_cutout) &&
                iteration == 1 && pixel != 2 ?
                std::abs(int(actual_x) - int(expected_x)) <= 2 : actual_x == expected_x;
            if (!x_matches || values[pixel * stride + 1] != 0)
                throw std::runtime_error("motion GPU mismatch at iteration " + std::to_string(iteration) +
                    " pixel " + std::to_string(pixel) + ": got " + std::to_string(values[pixel * stride]));
            if (rgba)
            {
                const uint16_t expected_depth = pixel == 2 ? 0x3c00 :
                    (iteration == 1 ? 0x3800 : 0xbc00);
                if (values[pixel * stride + 2] != expected_depth)
                    throw std::runtime_error("motion previous-depth mismatch at iteration " +
                        std::to_string(iteration) + " pixel " + std::to_string(pixel));
            }
        }
        vkUnmapMemory(vk.device, vk.readback_memory);
        if (iteration != 3) motion.commit_view();
    }
    check(vkDeviceWaitIdle(vk.device));
    object_pass.destroy();
    const VkImage reused_image = motion.image(0);
    motion.destroy();
    if (!motion.initialize(vk.physical, vk.device, vkGetDeviceProcAddr, {2, 2}, {vk.depth_view}, {vk.depth},
            vk.memory, dispatch, VK_NULL_HANDLE, !fallback, error, &image_pool,
            prefer_reprojected_depth) ||
        motion.image(0) != reused_image)
        throw std::runtime_error("motion image was not reused after GPU-idle rebuild: " + error);
    motion.destroy();
    image_pool.reset();
    if (indirect_velocity)
        std::printf("%s: indirect object-velocity draw passed (multi-draw limit=%u)\n",
            physical_properties.deviceName, enabled_features.multiDrawIndirect
                ? physical_properties.limits.maxDrawIndirectCount : 1u);
    return 0;
}

int main(int argc, char** argv)
{
    try
    {
        return run_test(argc, argv);
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Vulkan motion GPU test failed: %s\n", error.what());
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::fprintf(stderr, "Vulkan motion GPU test failed with an unknown exception\n");
        return EXIT_FAILURE;
    }
}
