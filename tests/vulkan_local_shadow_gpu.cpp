#include "../src/Layers/xrRenderVK/LocalShadowTargets.h"
#include "../src/Layers/xrRenderVK/ShadowShaders.h"
#include <cstddef>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace xray::render::vulkan;
static void check(VkResult result)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error("Vulkan result " + std::to_string(result));
}
int main()
{
    VkInstance instance{};
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    check(vkCreateInstance(&instance_info, nullptr, &instance));
    uint32_t count = 1;
    VkPhysicalDevice gpu{};
    check(vkEnumeratePhysicalDevices(instance, &count, &gpu));
    uint32_t families = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, nullptr);
    std::vector<VkQueueFamilyProperties> properties(families);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &families, properties.data());
    uint32_t family = 0;
    while (family < families && !(properties[family].queueFlags & VK_QUEUE_GRAPHICS_BIT))
        ++family;
    if (family == families)
        throw std::runtime_error("no graphics queue");
    float priority = 1;
    VkDeviceQueueCreateInfo queue_info{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    queue_info.queueFamilyIndex = family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;
    VkDeviceCreateInfo device_info{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    VkDevice device{};
    check(vkCreateDevice(gpu, &device_info, nullptr, &device));
    VkQueue queue{};
    vkGetDeviceQueue(device, family, 0, &queue);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(gpu, &memory);
    FrameDispatch dispatch{};
    dispatch.create_image = vkCreateImage;
    dispatch.destroy_image = vkDestroyImage;
    dispatch.get_image_memory_requirements = vkGetImageMemoryRequirements;
    dispatch.allocate_memory = vkAllocateMemory;
    dispatch.free_memory = vkFreeMemory;
    dispatch.bind_image_memory = vkBindImageMemory;
    dispatch.create_image_view = vkCreateImageView;
    dispatch.destroy_image_view = vkDestroyImageView;
    dispatch.create_framebuffer = vkCreateFramebuffer;
    dispatch.destroy_framebuffer = vkDestroyFramebuffer;
    dispatch.create_render_pass = vkCreateRenderPass;
    dispatch.destroy_render_pass = vkDestroyRenderPass;
    dispatch.cmd_begin_render_pass = vkCmdBeginRenderPass;
    dispatch.cmd_end_render_pass = vkCmdEndRenderPass;
    dispatch.cmd_copy_image = vkCmdCopyImage;
    dispatch.cmd_pipeline_barrier = vkCmdPipelineBarrier;
    const BufferResourceDispatch buffers{vkCreateBuffer,   vkDestroyBuffer, vkGetBufferMemoryRequirements,
                                         vkAllocateMemory, vkFreeMemory,    vkBindBufferMemory,
                                         vkMapMemory,      vkUnmapMemory};
    VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pool_info.queueFamilyIndex = family;
    VkCommandPool pool{};
    check(vkCreateCommandPool(device, &pool_info, nullptr, &pool));
    VkCommandBufferAllocateInfo allocation{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocation.commandPool = pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(device, &allocation, &command));
    for (VkFormat format : {VK_FORMAT_D32_SFLOAT, VK_FORMAT_D24_UNORM_S8_UINT})
    {
        LocalShadowTargets targets;
        std::string error;
        if (!targets.initialize(device, format, 2, memory, dispatch, buffers, vkCreateSampler, vkDestroySampler, error))
            throw std::runtime_error(error);
        BufferResource readback;
        if (!readback.initialize(device, 24, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory,
                                 buffers, error))
            throw std::runtime_error(error);
        check(vkResetCommandPool(device, pool, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(command, &begin));
        FrameRecordingContext frame{command, VK_NULL_HANDLE, VK_NULL_HANDLE, {512, 512}, 0, 0}, shadow;
        assert(targets.initialize_layers(frame));
        float matrix[16]{};
        matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1;
        assert(targets.begin_face(frame, 1, 1, shadow, true));
        VkClearAttachment clear{};
        clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        clear.clearValue.depthStencil = {.2f, 0};
        VkClearRect rect{{{0, 0}, {LocalShadowSize, LocalShadowSize}}, 0, 1};
        vkCmdClearAttachments(command, 1, &clear, 1, &rect);
        targets.end(command);
        targets.remember_face(1, 1, matrix);
        assert(targets.cached_face(1, 1, matrix));
        assert(targets.copy_cached_face(frame, 1, 1));
        assert(targets.begin_face(frame, 1, 1, shadow, false, true));
        clear.clearValue.depthStencil.depth = .1f;
        rect.rect.extent = {1, 1};
        vkCmdClearAttachments(command, 1, &clear, 1, &rect);
        targets.end(command);
        // Depth written by dynamic casters must never pollute the shared static cache.
        auto read_depth = [&](uint32_t index, VkDeviceSize offset, uint32_t layer = 7) {
            VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            barrier.image = targets.image(index);
            barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.subresourceRange = {
                VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT |
                                   (format == VK_FORMAT_D24_UNORM_S8_UINT ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)),
                0, 1, layer, 1};
            barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            vkCmdPipelineBarrier(command,
                                 VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
            VkBufferImageCopy copy{};
            copy.bufferOffset = offset;
            copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, layer, 1};
            copy.imageExtent = {1, 1, 1};
            vkCmdCopyImageToBuffer(command, targets.image(index), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readback.handle(), 1, &copy);
        };
        read_depth(0, 0);
        frame.image_index = 1;
        frame.frame_index = 1;
        assert(targets.initialize_layers(frame));
        assert(targets.copy_cached_face(frame, 1, 1));
        assert(targets.begin_face(frame, 1, 1, shadow, false, true));
        targets.end(command);
        read_depth(1, 4);
        // Execute production GPU shadow skinning with one through four bones.
        struct SkinVertex
        {
            float position[3], uv[2];
            uint16_t bones[4];
            float weights[4];
        };
        std::array<SkinVertex, 24> skin_vertices{};
        const float quad[6][2]{{-1, -1}, {1, -1}, {-1, 1}, {-1, 1}, {1, -1}, {1, 1}};
        for (unsigned count = 1; count <= 4; ++count)
            for (unsigned v = 0; v < 6; ++v)
            {
                auto &out = skin_vertices[(count - 1) * 6 + v];
                out.position[0] = quad[v][0];
                out.position[1] = quad[v][1];
                for (unsigned b = 0; b < 4; ++b)
                {
                    out.bones[b] = b;
                    out.weights[b] = b < count ? 1.f / count : 0;
                }
            }
        float bones[64]{};
        for (unsigned b = 0; b < 4; ++b)
        {
            for (unsigned i = 0; i < 16; i += 5)
                bones[b * 16 + i] = 1;
            bones[b * 16 + 14] = .1f * (b + 1);
        }
        BufferResource vertices, pose;
        assert(vertices.initialize(device, sizeof(skin_vertices), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory,
                                   buffers, error));
        assert(vertices.write(0, skin_vertices.data(), sizeof(skin_vertices), error));
        assert(pose.initialize(device, sizeof(bones), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory,
                               buffers, error));
        assert(pose.write(0, bones, sizeof(bones), error));
        VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        VkDescriptorSetLayout layouts[2]{};
        check(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &layouts[0]));
        VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT,
                                             nullptr};
        set_info.bindingCount = 1;
        set_info.pBindings = &binding;
        check(vkCreateDescriptorSetLayout(device, &set_info, nullptr, &layouts[1]));
        VkPushConstantRange push_range{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};
        VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout_info.setLayoutCount = 2;
        layout_info.pSetLayouts = layouts;
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_range;
        VkPipelineLayout layout{};
        check(vkCreatePipelineLayout(device, &layout_info, nullptr, &layout));
        VkDescriptorPoolSize pool_size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo descriptor_pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        descriptor_pool_info.maxSets = 1;
        descriptor_pool_info.poolSizeCount = 1;
        descriptor_pool_info.pPoolSizes = &pool_size;
        VkDescriptorPool descriptor_pool{};
        check(vkCreateDescriptorPool(device, &descriptor_pool_info, nullptr, &descriptor_pool));
        VkDescriptorSetAllocateInfo set_alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        set_alloc.descriptorPool = descriptor_pool;
        set_alloc.descriptorSetCount = 1;
        set_alloc.pSetLayouts = layouts + 1;
        VkDescriptorSet descriptor{};
        check(vkAllocateDescriptorSets(device, &set_alloc, &descriptor));
        VkDescriptorBufferInfo pose_info{pose.handle(), 0, sizeof(bones)};
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = descriptor;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &pose_info;
        vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        auto module = [&](const uint32_t *code, size_t bytes) {
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize = bytes;
            info.pCode = code;
            VkShaderModule result{};
            check(vkCreateShaderModule(device, &info, nullptr, &result));
            return result;
        };
        const auto vs = module(shadow_shaders::SkinVertex, sizeof(shadow_shaders::SkinVertex));
        const auto fs = module(shadow_shaders::Opaque, sizeof(shadow_shaders::Opaque));
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (auto &stage : stages)
            stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vs;
        stages[0].pName = "main";
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = fs;
        stages[1].pName = "main";
        VkVertexInputBindingDescription vertex_binding{0, sizeof(SkinVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attributes[]{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(SkinVertex, position)},
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(SkinVertex, uv)},
            {3, 0, VK_FORMAT_R16G16B16A16_UINT, offsetof(SkinVertex, bones)},
            {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(SkinVertex, weights)}};
        VkPipelineVertexInputStateCreateInfo inputs{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        inputs.vertexBindingDescriptionCount = 1;
        inputs.pVertexBindingDescriptions = &vertex_binding;
        inputs.vertexAttributeDescriptionCount = 4;
        inputs.pVertexAttributeDescriptions = attributes;
        VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport viewport{0, 0, float(LocalShadowSize), float(LocalShadowSize), 0, 1};
        VkRect2D scissor{{0, 0}, {LocalShadowSize, LocalShadowSize}};
        VkPipelineViewportStateCreateInfo viewport_info{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport_info.viewportCount = 1;
        viewport_info.pViewports = &viewport;
        viewport_info.scissorCount = 1;
        viewport_info.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.lineWidth = 1;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
        depth.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.stageCount = 2;
        info.pStages = stages;
        info.pVertexInputState = &inputs;
        info.pInputAssemblyState = &assembly;
        info.pViewportState = &viewport_info;
        info.pRasterizationState = &raster;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depth;
        info.pColorBlendState = &blend;
        info.layout = layout;
        info.renderPass = targets.render_pass();
        VkPipeline pipeline{};
        check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
        for (unsigned c = 0; c < 4; ++c)
        {
            assert(targets.begin_face(frame, 0, c, shadow));
            vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            const auto buffer = vertices.handle();
            const VkDeviceSize offset = c * 6 * sizeof(SkinVertex);
            vkCmdBindVertexBuffers(command, 0, 1, &buffer, &offset);
            vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &descriptor, 0, nullptr);
            vkCmdPushConstants(command, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, matrix);
            vkCmdDraw(command, 6, 1, 0, 0);
            targets.end(command);
            read_depth(1, 8 + c * 4, c);
        }
        VkBufferMemoryBarrier host{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        host.srcQueueFamilyIndex = host.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        host.buffer = readback.handle();
        host.size = 24;
        vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1,
                             &host, 0, nullptr);
        check(vkEndCommandBuffer(command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &command;
        check(vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(queue));
        uint32_t values[6]{};
        assert(readback.read(0, values, 24, error));
        for (unsigned i = 0; i < 6; ++i)
        {
            float actual{};
            if (format == VK_FORMAT_D32_SFLOAT)
                std::memcpy(&actual, &values[i], 4);
            else
                actual = float(values[i] & 0xffffff) / 16777215.f;
            if (std::abs(actual - (i < 2 ? (i ? .2f : .1f) : .05f * float(i))) > .001f)
                throw std::runtime_error("static shadow cache was overwritten by dynamic depth");
        }
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyShaderModule(device, vs, nullptr);
        vkDestroyShaderModule(device, fs, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipelineLayout(device, layout, nullptr);
        for (auto l : layouts)
            vkDestroyDescriptorSetLayout(device, l, nullptr);
        pose.destroy();
        vertices.destroy();
        readback.destroy();
        targets.destroy();
    }
    vkDestroyCommandPool(device, pool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::cout << "local-shadow GPU cache: D32/D24 static preservation + dynamic overlay + 1-4 bone GPU skinning OK\n";
}
