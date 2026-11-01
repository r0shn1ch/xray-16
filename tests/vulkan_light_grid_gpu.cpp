#include "src/Layers/xrRenderVK/LightGrid.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace xray::render::vulkan;
static void check(VkResult value)
{
    if (value != VK_SUCCESS)
        throw std::runtime_error("Vulkan result " + std::to_string(value));
}
int main()
{
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ii.pApplicationInfo = &app;
    VkInstance instance{};
    check(vkCreateInstance(&ii, nullptr, &instance));
    uint32_t n = 0;
    check(vkEnumeratePhysicalDevices(instance, &n, nullptr));
    std::vector<VkPhysicalDevice> physical(n);
    check(vkEnumeratePhysicalDevices(instance, &n, physical.data()));
    if (!n)
        return 77;
    uint32_t family = UINT32_MAX;
    VkPhysicalDevice gpu{};
    for (auto candidate : physical)
    {
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, nullptr);
        std::vector<VkQueueFamilyProperties> queues(n);
        vkGetPhysicalDeviceQueueFamilyProperties(candidate, &n, queues.data());
        for (uint32_t i = 0; i < n; ++i)
            if ((queues[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))
            {
                family = i;
                gpu = candidate;
                break;
            }
        if (gpu)
            break;
    }
    if (!gpu)
        return 77;
    const float priority = 1;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = family;
    qi.queueCount = 1;
    qi.pQueuePriorities = &priority;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1;
    di.pQueueCreateInfos = &qi;
    VkDevice device{};
    check(vkCreateDevice(gpu, &di, nullptr, &device));
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(gpu, &memory);
    const auto allocate = [&](VkMemoryRequirements r, VkMemoryPropertyFlags flags) {
        uint32_t type = 0;
        for (; type < memory.memoryTypeCount; ++type)
            if ((r.memoryTypeBits & (1u << type)) && (memory.memoryTypes[type].propertyFlags & flags) == flags)
                break;
        assert(type < memory.memoryTypeCount);
        VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        a.allocationSize = r.size;
        a.memoryTypeIndex = type;
        VkDeviceMemory result{};
        check(vkAllocateMemory(device, &a, nullptr, &result));
        return result;
    };
    struct Image
    {
        VkImage image;
        VkImageView view;
        VkDeviceMemory memory;
    };
    std::vector<Image> images;
    for (unsigned i = 0; i < 5; ++i)
    {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = VK_FORMAT_R32G32B32A32_SFLOAT;
        ci.extent = {32, 1, 1};
        ci.mipLevels = ci.arrayLayers = 1;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.tiling = i == 4 ? VK_IMAGE_TILING_OPTIMAL : VK_IMAGE_TILING_LINEAR;
        ci.usage = i == 4 ? (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                          : VK_IMAGE_USAGE_SAMPLED_BIT;
        ci.initialLayout = i == 4 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_PREINITIALIZED;
        Image image{};
        check(vkCreateImage(device, &ci, nullptr, &image.image));
        VkMemoryRequirements r{};
        vkGetImageMemoryRequirements(device, image.image, &r);
        image.memory = allocate(r, i == 4 ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                                          : VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindImageMemory(device, image.image, image.memory, 0));
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = image.image;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = ci.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device, &vi, nullptr, &image.view));
        if (i < 4)
        {
            const float pixels[4][4]{{.25f, .4f, .6f, 1}, {.5f, .5f, 1, 1}, {.5f, .5f, .5f, .5f}, {0, 0, 0, 0}};
            VkImageSubresource sub{VK_IMAGE_ASPECT_COLOR_BIT, 0, 0};
            VkSubresourceLayout layout{};
            vkGetImageSubresourceLayout(device, image.image, &sub, &layout);
            void *p{};
            check(vkMapMemory(device, image.memory, 0, VK_WHOLE_SIZE, 0, &p));
            for (unsigned x = 0; x < 32; ++x)
                std::memcpy(static_cast<char *>(p) + layout.offset + x * 16, pixels[i], 16);
            vkUnmapMemory(device, image.memory);
        }
        images.push_back(image);
    }
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    VkSampler sampler{};
    check(vkCreateSampler(device, &sci, nullptr, &sampler));
    VkDescriptorSetLayoutBinding bindings[4]{};
    const uint32_t slots[]{0, 1, 2, 5};
    for (uint32_t i = 0; i < 4; ++i)
        bindings[i] = {slots[i], VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 4;
    sli.pBindings = bindings;
    VkDescriptorSetLayout layout{};
    check(vkCreateDescriptorSetLayout(device, &sli, nullptr, &layout));
    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &ps;
    VkDescriptorPool pool{};
    check(vkCreateDescriptorPool(device, &dpi, nullptr, &pool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &layout;
    VkDescriptorSet set{};
    check(vkAllocateDescriptorSets(device, &dai, &set));
    for (unsigned i = 0; i < 4; ++i)
    {
        const VkDescriptorImageInfo info{sampler, images[i].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w.dstSet = set;
        w.dstBinding = slots[i];
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &info;
        vkUpdateDescriptorSets(device, 1, &w, 0, nullptr);
    }
    VkAttachmentDescription attachment{};
    attachment.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkSubpassDependency dependencies[2]{};
    dependencies[0] = {VK_SUBPASS_EXTERNAL,
                       0,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       0,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       0};
    dependencies[1] = {0,
                       VK_SUBPASS_EXTERNAL,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT,
                       0};
    VkRenderPassCreateInfo rpi{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    rpi.attachmentCount = 1;
    rpi.pAttachments = &attachment;
    rpi.subpassCount = 1;
    rpi.pSubpasses = &subpass;
    rpi.dependencyCount = 2;
    rpi.pDependencies = dependencies;
    VkRenderPass pass{};
    check(vkCreateRenderPass(device, &rpi, nullptr, &pass));
    VkFramebufferCreateInfo fbi{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    fbi.renderPass = pass;
    fbi.attachmentCount = 1;
    fbi.pAttachments = &images[4].view;
    fbi.width = 32;
    fbi.height = fbi.layers = 1;
    VkFramebuffer framebuffer{};
    check(vkCreateFramebuffer(device, &fbi, nullptr, &framebuffer));
    BufferResourceDispatch bd{vkCreateBuffer,   vkDestroyBuffer, vkGetBufferMemoryRequirements,
                              vkAllocateMemory, vkFreeMemory,    vkBindBufferMemory,
                              vkMapMemory,      vkUnmapMemory};
    ScenePassDispatch graphics{};
    std::string error;
    assert(load_scene_pass_dispatch(device, vkGetDeviceProcAddr, graphics, error));
    ShaderModuleDispatch shaders{vkCreateShaderModule, vkDestroyShaderModule};
    LightGrid grid;
    assert(grid.initialize(device, vkGetDeviceProcAddr, pass, layout, memory, bd, graphics, shaders,
                           vkCmdPipelineBarrier, error));
    BufferResource readback;
    assert(readback.initialize(device, 32 * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, bd,
                               error));
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = family;
    VkCommandPool commands{};
    check(vkCreateCommandPool(device, &cpi, nullptr, &commands));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = commands;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(device, &cai, &command));
    std::vector<LocalLightUniform> lights(3);
    std::vector<VkRect2D> rectangles{{{0, 0}, {16, 1}}, {{16, 0}, {16, 1}}, {{0, 0}, {32, 1}}};
    for (unsigned j = 0; j < 3; ++j)
    {
        for (unsigned i = 0; i < 16; i += 5)
            lights[j].inverse_view_projection[i] = 1;
        lights[j].position_range[0] = j == 0 ? -.5f : .5f;
        lights[j].position_range[2] = 2;
        lights[j].position_range[3] = 5;
    }
    lights[0].color_type[0] = 1;
    lights[1].color_type[1] = 1;
    lights[2].shadow_params[2] = 1;
    for (unsigned frame = 0; frame < 4; ++frame)
    {
        check(vkResetCommandPool(device, commands, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin));
        if (frame == 0)
        {
            VkImageMemoryBarrier b[4]{};
            for (unsigned i = 0; i < 4; ++i)
            {
                b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
                b[i].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
                b[i].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                b[i].oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED;
                b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b[i].image = images[i].image;
                b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            }
            vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
                                 nullptr, 0, nullptr, 4, b);
        }
        if (frame == 1)
            lights[0].color_type[0] = .5f;
        if (frame == 2)
            lights.resize(1), rectangles.resize(1);
        if (frame == 3)
        {
            lights.assign(64, {});
            rectangles.assign(64, {{0, 0}, {32, 1}});
            for (auto &light : lights)
            {
                for (unsigned i = 0; i < 16; i += 5)
                    light.inverse_view_projection[i] = 1;
                light.position_range[2] = 2;
                light.position_range[3] = 5;
                light.color_type[0] = .01f;
            }
        }
        FrameRecordingContext context{command, pass, framebuffer, {32, 1}, 0, 0};
        assert(grid.prepare(context, lights, rectangles, error));
        assert(grid.count(0) == (frame == 2 ? 0 : frame == 3 ? 64 : 2));
        assert(grid.contains(0, 2) == (frame == 3));
        VkClearValue clear{};
        VkRenderPassBeginInfo rbi{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        rbi.renderPass = pass;
        rbi.framebuffer = framebuffer;
        rbi.renderArea.extent = {32, 1};
        rbi.clearValueCount = 1;
        rbi.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &rbi, VK_SUBPASS_CONTENTS_INLINE);
        assert(grid.record(context, set));
        vkCmdEndRenderPass(command);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.imageExtent = {32, 1, 1};
        vkCmdCopyImageToBuffer(command, images[4].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(), 1,
                               &copy);
        VkBufferMemoryBarrier rb{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        rb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        rb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        rb.srcQueueFamilyIndex = rb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        rb.buffer = readback.handle();
        rb.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &rb,
                             0, nullptr);
        check(vkEndCommandBuffer(command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(queue));
        float pixels[32][4]{};
        assert(readback.read(0, pixels, sizeof(pixels), error));
        for (unsigned x : {0u, 7u, 15u, 16u, 23u, 31u})
        {
            const bool left = x < 16;
            float dx = (float(x) + .5f) / 16.f - 1.f - (frame == 3 ? 0 : left ? -.5f : .5f);
            const float d2 = dx * dx + 2.25f;
            const float value = (frame == 3 ? .25f * .64f
                                 : left     ? .25f * (frame == 1 ? .5f : 1.f)
                                            : .4f) *
                                (1.f - d2 / 25.f) * 1.5f / std::sqrt(d2);
            for (unsigned channel = 0; channel < 3; ++channel)
            {
                const float expected = frame == 2 ? 0.f : channel == (frame == 3 ? 0u : left ? 0u : 1u) ? value : 0.f;
                if (std::abs(pixels[x][channel] - expected) > .0001f)
                    throw std::runtime_error("GPU light grid pixel mismatch frame=" + std::to_string(frame) +
                                             " x=" + std::to_string(x) + " channel=" + std::to_string(channel) +
                                             " actual=" + std::to_string(pixels[x][channel]) +
                                             " expected=" + std::to_string(expected));
            }
        }
    }
    grid.destroy();
    readback.destroy();
    vkDestroyCommandPool(device, commands, nullptr);
    vkDestroyFramebuffer(device, framebuffer, nullptr);
    vkDestroyRenderPass(device, pass, nullptr);
    vkDestroyDescriptorPool(device, pool, nullptr);
    vkDestroyDescriptorSetLayout(device, layout, nullptr);
    vkDestroySampler(device, sampler, nullptr);
    for (auto image : images)
    {
        vkDestroyImageView(device, image.view, nullptr);
        vkDestroyImage(device, image.image, nullptr);
        vkFreeMemory(device, image.memory, nullptr);
    }
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::cout << "light grid GPU: tile boundaries, summed color, exclusion and frame reuse OK\n";
}
