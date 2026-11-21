#include "tests/vulkan_ao_gpu_shader.h"
#include <vulkan/vulkan.h>

#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr uint32_t Side = 64;
constexpr VkDeviceSize ImageBytes = Side * Side * sizeof(float);

void check(VkResult result)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error("AO GPU VkResult=" + std::to_string(result));
}

struct Fixture
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkImage depth{}, output{};
    VkImageView depth_view{}, output_view{};
    VkDeviceMemory depth_memory{}, output_memory{};
    VkBuffer upload{}, readback{}, constants{};
    VkDeviceMemory upload_memory{}, readback_memory{}, constants_memory{};
    VkSampler sampler{};
    VkDescriptorPool descriptor_pool{};
    VkDescriptorSetLayout set_layout{};
    VkPipelineLayout pipeline_layout{};
    VkPipeline pipeline{};
    VkShaderModule module{};
    VkPhysicalDeviceMemoryProperties memory{};

    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
            if (module) vkDestroyShaderModule(device, module, nullptr);
            if (pipeline_layout) vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            if (descriptor_pool) vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
            if (set_layout) vkDestroyDescriptorSetLayout(device, set_layout, nullptr);
            if (sampler) vkDestroySampler(device, sampler, nullptr);
            if (depth_view) vkDestroyImageView(device, depth_view, nullptr);
            if (output_view) vkDestroyImageView(device, output_view, nullptr);
            if (depth) vkDestroyImage(device, depth, nullptr);
            if (output) vkDestroyImage(device, output, nullptr);
            if (depth_memory) vkFreeMemory(device, depth_memory, nullptr);
            if (output_memory) vkFreeMemory(device, output_memory, nullptr);
            for (auto buffer : {upload, readback, constants})
                if (buffer) vkDestroyBuffer(device, buffer, nullptr);
            for (auto allocation : {upload_memory, readback_memory, constants_memory})
                if (allocation) vkFreeMemory(device, allocation, nullptr);
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
        throw std::runtime_error("AO GPU memory type unavailable");
    }

    void make_buffer(VkDeviceSize bytes, VkBufferUsageFlags usage, VkBuffer& buffer, VkDeviceMemory& allocation)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = bytes;
        info.usage = usage;
        check(vkCreateBuffer(device, &info, nullptr, &buffer));
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, buffer, &requirements);
        allocation = allocate(requirements, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, buffer, allocation, 0));
    }

    void make_image(VkImageUsageFlags usage, VkImage& image, VkImageView& view, VkDeviceMemory& allocation)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = VK_FORMAT_R32_SFLOAT;
        info.extent = {Side, Side, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        check(vkCreateImage(device, &info, nullptr, &image));
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, image, &requirements);
        allocation = allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkBindImageMemory(device, image, allocation, 0));
        VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R32_SFLOAT;
        view_info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &view_info, nullptr, &view));
    }
};

void barrier(VkCommandBuffer command, VkImage image, VkImageLayout old_layout, VkImageLayout new_layout,
    VkAccessFlags source, VkAccessFlags destination, VkPipelineStageFlags before, VkPipelineStageFlags after)
{
    VkImageMemoryBarrier info{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    info.srcAccessMask = source;
    info.dstAccessMask = destination;
    info.oldLayout = old_layout;
    info.newLayout = new_layout;
    info.srcQueueFamilyIndex = info.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    info.image = image;
    info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(command, before, after, 0, 0, nullptr, 0, nullptr, 1, &info);
}
}

int main(int argc, char** argv)
{
    Fixture vk;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    check(vkCreateInstance(&instance_info, nullptr, &vk.instance));
    uint32_t count{};
    check(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
    if (!count) return 77;
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(vk.instance, &count, devices.data()));
    const int selected = argc > 1 ? std::stoi(argv[1]) : -1;
    uint32_t family = UINT32_MAX;
    for (size_t index = 0; index < devices.size(); ++index)
    {
        if (selected >= 0 && size_t(selected) != index) continue;
        VkFormatProperties support{};
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_R32_SFLOAT, &support);
        if ((support.optimalTilingFeatures &
                (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
            (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) continue;
        uint32_t families{};
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, queues.data());
        for (uint32_t i = 0; i < families; ++i)
            if (queues[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                vk.physical = devices[index];
                family = i;
                break;
            }
        if (vk.physical) break;
    }
    if (!vk.physical) return 77;
    float priority = 1.f;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
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

    vk.make_image(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        vk.depth, vk.depth_view, vk.depth_memory);
    vk.make_image(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        vk.output, vk.output_view, vk.output_memory);
    vk.make_buffer(ImageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, vk.upload, vk.upload_memory);
    vk.make_buffer(ImageBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, vk.readback, vk.readback_memory);
    vk.make_buffer(44 * sizeof(float), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, vk.constants, vk.constants_memory);
    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = sampler_info.minFilter = VK_FILTER_NEAREST;
    sampler_info.addressModeU = sampler_info.addressModeV = sampler_info.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    check(vkCreateSampler(vk.device, &sampler_info, nullptr, &vk.sampler));

    const VkDescriptorSetLayoutBinding bindings[]{
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 3;
    layout_info.pBindings = bindings;
    check(vkCreateDescriptorSetLayout(vk.device, &layout_info, nullptr, &vk.set_layout));
    const VkDescriptorPoolSize sizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}};
    VkDescriptorPoolCreateInfo descriptor_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    descriptor_info.maxSets = 1;
    descriptor_info.poolSizeCount = 3;
    descriptor_info.pPoolSizes = sizes;
    check(vkCreateDescriptorPool(vk.device, &descriptor_info, nullptr, &vk.descriptor_pool));
    VkDescriptorSetAllocateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set_info.descriptorPool = vk.descriptor_pool;
    set_info.descriptorSetCount = 1;
    set_info.pSetLayouts = &vk.set_layout;
    VkDescriptorSet set{};
    check(vkAllocateDescriptorSets(vk.device, &set_info, &set));
    const VkDescriptorImageInfo depth_info{vk.sampler, vk.depth_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    const VkDescriptorImageInfo output_info{VK_NULL_HANDLE, vk.output_view, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorBufferInfo buffer_info{vk.constants, 0, 44 * sizeof(float)};
    VkWriteDescriptorSet writes[3]{};
    for (auto& write : writes)
    {
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.descriptorCount = 1;
    }
    writes[0].dstBinding = 2;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &depth_info;
    writes[1].dstBinding = 4;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &buffer_info;
    writes[2].dstBinding = 6;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &output_info;
    vkUpdateDescriptorSets(vk.device, 3, writes, 0, nullptr);
    VkPipelineLayoutCreateInfo pipeline_layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vk.set_layout;
    check(vkCreatePipelineLayout(vk.device, &pipeline_layout_info, nullptr, &vk.pipeline_layout));
    VkShaderModuleCreateInfo module_info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    module_info.codeSize = sizeof(ao_gpu_shader::Compute);
    module_info.pCode = ao_gpu_shader::Compute;
    check(vkCreateShaderModule(vk.device, &module_info, nullptr, &vk.module));
    VkComputePipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline_info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = vk.module;
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = vk.pipeline_layout;
    check(vkCreateComputePipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &vk.pipeline));

    std::array<float, 44> uniform{};
    for (int i : {0, 5, 10, 15}) uniform[i] = 1.f;
    uniform[35] = 2.f;
    std::array<float, 4> result{};
    for (uint32_t iteration = 0; iteration < result.size(); ++iteration)
    {
        std::vector<float> input(Side * Side, iteration == 0 ? .5f : .8f);
        for (uint32_t y = 31; y <= 33; ++y)
            for (uint32_t x = 31; x <= 33; ++x)
                input[y * Side + x] = .5f;
        uniform[35] = iteration == 2 ? .75f : iteration == 3 ? 0.f : 2.f;
        void* mapped{};
        check(vkMapMemory(vk.device, vk.upload_memory, 0, ImageBytes, 0, &mapped));
        std::memcpy(mapped, input.data(), ImageBytes);
        vkUnmapMemory(vk.device, vk.upload_memory);
        check(vkMapMemory(vk.device, vk.constants_memory, 0, sizeof(uniform), 0, &mapped));
        std::memcpy(mapped, uniform.data(), sizeof(uniform));
        vkUnmapMemory(vk.device, vk.constants_memory);
        check(vkResetCommandBuffer(vk.command, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(vk.command, &begin));
        barrier(vk.command, vk.depth, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {Side, Side, 1};
        vkCmdCopyBufferToImage(vk.command, vk.upload, vk.depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier(vk.command, vk.depth, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        barrier(vk.command, vk.output, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0,
            VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        vkCmdBindPipeline(vk.command, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipeline);
        vkCmdBindDescriptorSets(vk.command, VK_PIPELINE_BIND_POINT_COMPUTE, vk.pipeline_layout,
            0, 1, &set, 0, nullptr);
        vkCmdDispatch(vk.command, Side / 8, Side / 8, 1);
        barrier(vk.command, vk.output, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        vkCmdCopyImageToBuffer(vk.command, vk.output, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            vk.readback, 1, &copy);
        check(vkEndCommandBuffer(vk.command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &vk.command;
        check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        check(vkMapMemory(vk.device, vk.readback_memory, 0, ImageBytes, 0, &mapped));
        result[iteration] = static_cast<const float*>(mapped)[32 * Side + 32];
        if (std::abs(static_cast<const float*>(mapped)[0] - 1.f) > 1e-4f ||
            static_cast<const float*>(mapped)[1] > 1e-5f)
            throw std::runtime_error("indirect reconstruction plane rejection mismatch");
        vkUnmapMemory(vk.device, vk.readback_memory);
    }
    if (std::abs(result[0] - 1.f) > 1e-4f || result[1] >= .9f || result[1] < .2f ||
        result[2] >= .97f || result[2] <= result[1] || std::abs(result[3] - 1.f) > 1e-4f)
        throw std::runtime_error("AO GPU flat/strong/default/off mismatch: " +
            std::to_string(result[0]) + ", " + std::to_string(result[1]) + ", " +
            std::to_string(result[2]) + ", " + std::to_string(result[3]));
    return 0;
}
