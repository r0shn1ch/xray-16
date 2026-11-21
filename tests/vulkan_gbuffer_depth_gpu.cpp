#include "src/Layers/xrRenderVK/GBufferTargets.h"
#include "tests/vulkan_gbuffer_msaa_shaders.h"

#include <vulkan/vulkan.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
void check(VkResult result)
{
    if (result != VK_SUCCESS)
        throw std::runtime_error("G-buffer depth GPU result " + std::to_string(result));
}

struct Device
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool pool{};
    VkCommandBuffer command{};
    VkBuffer readback{};
    VkDeviceMemory readback_memory{};
    VkPhysicalDeviceMemoryProperties memory{};

    ~Device()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (readback) vkDestroyBuffer(device, readback, nullptr);
            if (readback_memory) vkFreeMemory(device, readback_memory, nullptr);
            if (pool) vkDestroyCommandPool(device, pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    VkDeviceMemory host_memory(VkMemoryRequirements requirements)
    {
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags &
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))
            {
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                info.allocationSize = requirements.size;
                info.memoryTypeIndex = i;
                VkDeviceMemory result{};
                check(vkAllocateMemory(device, &info, nullptr, &result));
                return result;
            }
        throw std::runtime_error("G-buffer depth readback memory unavailable");
    }
};

VkResult VKAPI_PTR fake_image_format(VkPhysicalDevice, VkFormat format, VkImageType, VkImageTiling,
    VkImageUsageFlags, VkImageCreateFlags, VkImageFormatProperties* out)
{
    *out = {};
    out->sampleCounts = format == VK_FORMAT_R8G8B8A8_UNORM ?
        VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_2_BIT;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR fake_create_render_pass2(VkDevice, const VkRenderPassCreateInfo2*,
    const VkAllocationCallbacks*, VkRenderPass*) { return VK_SUCCESS; }

void test_msaa_policy()
{
    VkPhysicalDeviceProperties properties{};
    properties.limits.framebufferColorSampleCounts = properties.limits.framebufferDepthSampleCounts =
        VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT;
    VkPhysicalDeviceDepthStencilResolveProperties resolve{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    resolve.supportedDepthResolveModes = VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    resolve.supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT;
    GBufferMsaaConfig selected; std::string error;
    auto physical = reinterpret_cast<VkPhysicalDevice>(uintptr_t(1));
    if (!negotiate_gbuffer_msaa(physical, properties, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT,
            VK_SAMPLE_COUNT_4_BIT, fake_image_format, resolve, fake_create_render_pass2, selected, error) ||
        selected.samples != VK_SAMPLE_COUNT_2_BIT || selected.depth_resolve != VK_RESOLVE_MODE_MIN_BIT ||
        selected.stencil_resolve != VK_RESOLVE_MODE_NONE)
        throw std::runtime_error("MSAA sample or depth negotiation mismatch: " + error);
    if (!negotiate_gbuffer_msaa(physical, properties, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D24_UNORM_S8_UINT,
            VK_SAMPLE_COUNT_4_BIT, fake_image_format, resolve, fake_create_render_pass2, selected, error) ||
        selected.depth_resolve != VK_RESOLVE_MODE_SAMPLE_ZERO_BIT || selected.stencil_resolve != VK_RESOLVE_MODE_SAMPLE_ZERO_BIT)
        throw std::runtime_error("MSAA depth/stencil resolve negotiation mismatch: " + error);
    resolve.independentResolveNone = VK_TRUE;
    if (!negotiate_gbuffer_msaa(physical, properties, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D24_UNORM_S8_UINT,
            VK_SAMPLE_COUNT_4_BIT, fake_image_format, resolve, fake_create_render_pass2, selected, error) ||
        selected.depth_resolve != VK_RESOLVE_MODE_MIN_BIT || selected.stencil_resolve != VK_RESOLVE_MODE_NONE)
        throw std::runtime_error("MSAA independent depth resolve negotiation mismatch: " + error);
    resolve.supportedDepthResolveModes = 0;
    if (negotiate_gbuffer_msaa(physical, properties, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT,
            VK_SAMPLE_COUNT_4_BIT, fake_image_format, resolve, fake_create_render_pass2, selected, error) ||
        selected.samples != VK_SAMPLE_COUNT_1_BIT)
        throw std::runtime_error("MSAA negotiation did not reject missing depth resolve");
}

void test_msaa_resolve(Device& vk, uint32_t instance_version, const FrameDispatch& dispatch)
{
    VkPhysicalDeviceProperties physical{}; vkGetPhysicalDeviceProperties(vk.physical, &physical);
    if (instance_version < VK_API_VERSION_1_2 || physical.apiVersion < VK_API_VERSION_1_2) return;
    auto create = reinterpret_cast<PFN_vkCreateRenderPass2>(vkGetDeviceProcAddr(vk.device, "vkCreateRenderPass2"));
    if (!create) return;
    VkPhysicalDeviceDepthStencilResolveProperties resolve{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_STENCIL_RESOLVE_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &resolve; vkGetPhysicalDeviceProperties2(vk.physical, &properties);
    GBufferMsaaConfig config; std::string error;
    if (!negotiate_gbuffer_msaa(vk.physical, physical, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT,
            VK_SAMPLE_COUNT_4_BIT, vkGetPhysicalDeviceImageFormatProperties, resolve, create, config, error)) return;
    const VkSampleCountFlagBits samples = config.samples;
    const VkResolveModeFlagBits depth_mode = config.depth_resolve;
    struct Resources
    {
        VkDevice device{}; VkRenderPass pass{}; VkFramebuffer framebuffer{};
        std::array<VkImage, 8> images{}; std::array<VkImageView, 8> views{};
        std::array<VkDeviceMemory, 8> memory{};
        ~Resources()
        {
            if (framebuffer) vkDestroyFramebuffer(device, framebuffer, nullptr);
            for (auto view : views) if (view) vkDestroyImageView(device, view, nullptr);
            for (auto image : images) if (image) vkDestroyImage(device, image, nullptr);
            for (auto allocation : memory) if (allocation) vkFreeMemory(device, allocation, nullptr);
            if (pass) vkDestroyRenderPass(device, pass, nullptr);
        }
    } resources;
    resources.device = vk.device;
    if (!create_msaa_gbuffer_render_pass(vk.device, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM,
            VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT, samples, depth_mode, VK_RESOLVE_MODE_NONE,
            create, resources.pass, error)) throw std::runtime_error(error);
    for (uint32_t i = 0; i < 8; ++i)
    {
        const bool is_depth = i % 4 == 3;
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType = VK_IMAGE_TYPE_2D; image.format = is_depth ? VK_FORMAT_D32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
        image.extent = {1, 1, 1}; image.mipLevels = image.arrayLayers = 1;
        image.samples = i < 4 ? samples : VK_SAMPLE_COUNT_1_BIT; image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = (is_depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) |
            VK_IMAGE_USAGE_SAMPLED_BIT | (i >= 4 ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
        check(vkCreateImage(vk.device, &image, nullptr, &resources.images[i]));
        VkMemoryRequirements req{}; vkGetImageMemoryRequirements(vk.device, resources.images[i], &req);
        uint32_t memory_type = UINT32_MAX;
        for (uint32_t n = 0; n < vk.memory.memoryTypeCount; ++n)
            if ((req.memoryTypeBits & (1u << n)) && (vk.memory.memoryTypes[n].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            { memory_type = n; break; }
        if (memory_type == UINT32_MAX) throw std::runtime_error("MSAA image memory unavailable");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; allocation.allocationSize = req.size;
        allocation.memoryTypeIndex = memory_type;
        check(vkAllocateMemory(vk.device, &allocation, nullptr, &resources.memory[i]));
        check(vkBindImageMemory(vk.device, resources.images[i], resources.memory[i], 0));
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO}; view.image = resources.images[i];
        view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = image.format;
        view.subresourceRange = {VkImageAspectFlags(is_depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
        check(vkCreateImageView(vk.device, &view, nullptr, &resources.views[i]));
    }
    VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer.renderPass = resources.pass; framebuffer.attachmentCount = 8; framebuffer.pAttachments = resources.views.data();
    framebuffer.width = framebuffer.height = framebuffer.layers = 1;
    check(vkCreateFramebuffer(vk.device, &framebuffer, nullptr, &resources.framebuffer));
    check(vkResetCommandBuffer(vk.command, 0));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; check(vkBeginCommandBuffer(vk.command, &begin));
    VkClearValue clear[4]{}; clear[0].color = {{.25f, .5f, .75f, 1.f}}; clear[3].depthStencil = {.375f, 0};
    VkRenderPassBeginInfo render{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render.renderPass = resources.pass; render.framebuffer = resources.framebuffer; render.renderArea.extent = {1, 1};
    render.clearValueCount = 4; render.pClearValues = clear;
    vkCmdBeginRenderPass(vk.command, &render, VK_SUBPASS_CONTENTS_INLINE); vkCmdEndRenderPass(vk.command);
    VkImageMemoryBarrier barriers[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        auto& barrier = barriers[i]; barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = i ? VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT : VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.oldLayout = i ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = resources.images[i ? 7 : 4];
        barrier.subresourceRange = {VkImageAspectFlags(i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
    }
    vkCmdPipelineBarrier(vk.command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    VkBufferImageCopy copy{}; copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}; copy.imageExtent = {1, 1, 1};
    vkCmdCopyImageToBuffer(vk.command, resources.images[4], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.readback, 1, &copy);
    copy.bufferOffset = 4; copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    vkCmdCopyImageToBuffer(vk.command, resources.images[7], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.readback, 1, &copy);
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER}; host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(vk.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(vk.command)); VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1; submit.pCommandBuffers = &vk.command;
    check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE)); check(vkQueueWaitIdle(vk.queue));
    void* data{}; check(vkMapMemory(vk.device, vk.readback_memory, 0, 8, 0, &data));
    const auto* bytes = static_cast<const uint8_t*>(data); float resolved{}; std::memcpy(&resolved, bytes + 4, sizeof(resolved));
    const bool correct = std::abs(int(bytes[0]) - 64) <= 1 && std::abs(int(bytes[1]) - 128) <= 1 &&
        std::abs(int(bytes[2]) - 191) <= 1 && std::abs(resolved - .375f) < 1e-5f;
    vkUnmapMemory(vk.device, vk.readback_memory);
    if (!correct) throw std::runtime_error("multisample color/depth resolve mismatch");
    if (physical.limits.standardSampleLocations)
    {
        struct PipelineResources
        {
            VkDevice device{}; VkShaderModule vertex{}, fragment{}; VkPipelineLayout layout{}; VkPipeline pipeline{};
            ~PipelineResources() {
                if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
                if (layout) vkDestroyPipelineLayout(device, layout, nullptr);
                if (vertex) vkDestroyShaderModule(device, vertex, nullptr);
                if (fragment) vkDestroyShaderModule(device, fragment, nullptr);
            }
        } graphics;
        graphics.device = vk.device;
        VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        module.pCode = gbuffer_msaa_test_shaders::Vertex; module.codeSize = sizeof(gbuffer_msaa_test_shaders::Vertex);
        check(vkCreateShaderModule(vk.device, &module, nullptr, &graphics.vertex));
        module.pCode = gbuffer_msaa_test_shaders::Fragment; module.codeSize = sizeof(gbuffer_msaa_test_shaders::Fragment);
        check(vkCreateShaderModule(vk.device, &module, nullptr, &graphics.fragment));
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        check(vkCreatePipelineLayout(vk.device, &layout, nullptr, &graphics.layout));
        VkPipelineShaderStageCreateInfo stages[2]{};
        for (auto& stage : stages) { stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stage.pName = "main"; }
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = graphics.vertex;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = graphics.fragment;
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkViewport viewport{0, 0, 1, 1, 0, 1}; VkRect2D scissor{{0, 0}, {1, 1}};
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = vp.scissorCount = 1; vp.pViewports = &viewport; vp.pScissors = &scissor;
        VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL; raster.cullMode = VK_CULL_MODE_NONE; raster.lineWidth = 1.f;
        VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = samples;
        VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depth_state.depthTestEnable = depth_state.depthWriteEnable = VK_TRUE; depth_state.depthCompareOp = VK_COMPARE_OP_LESS;
        VkPipelineColorBlendAttachmentState colors[3]{};
        for (auto& color : colors) color.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
            VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blend.attachmentCount = 3; blend.pAttachments = colors;
        VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pipeline.stageCount = 2; pipeline.pStages = stages; pipeline.pVertexInputState = &vi;
        pipeline.pInputAssemblyState = &ia; pipeline.pViewportState = &vp; pipeline.pRasterizationState = &raster;
        pipeline.pMultisampleState = &multisample; pipeline.pDepthStencilState = &depth_state; pipeline.pColorBlendState = &blend;
        pipeline.layout = graphics.layout; pipeline.renderPass = resources.pass;
        check(vkCreateGraphicsPipelines(vk.device, VK_NULL_HANDLE, 1, &pipeline, nullptr, &graphics.pipeline));
        check(vkResetCommandBuffer(vk.command, 0)); check(vkBeginCommandBuffer(vk.command, &begin));
        clear[0].color = {{0, 0, 0, 0}}; clear[3].depthStencil = {1.f, 0};
        vkCmdBeginRenderPass(vk.command, &render, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(vk.command, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics.pipeline);
        vkCmdDraw(vk.command, 6, 1, 0, 0); vkCmdEndRenderPass(vk.command);
        vkCmdPipelineBarrier(vk.command, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
        copy.bufferOffset = 0; copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vkCmdCopyImageToBuffer(vk.command, resources.images[4], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.readback, 1, &copy);
        copy.bufferOffset = 4; copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        vkCmdCopyImageToBuffer(vk.command, resources.images[7], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.readback, 1, &copy);
        vkCmdPipelineBarrier(vk.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
        check(vkEndCommandBuffer(vk.command)); check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        check(vkMapMemory(vk.device, vk.readback_memory, 0, 8, 0, &data));
        bytes = static_cast<const uint8_t*>(data); std::memcpy(&resolved, bytes + 4, sizeof(resolved));
        const bool partial = bytes[0] > 0 && bytes[0] < 255 &&
            (depth_mode != VK_RESOLVE_MODE_MIN_BIT || std::abs(resolved - .25f) < 1e-5f);
        vkUnmapMemory(vk.device, vk.readback_memory);
        if (!partial) throw std::runtime_error("MSAA geometry coverage or depth resolve mismatch");
    }
    GBufferTargets targets;
    if (!targets.initialize(vk.physical, vk.device, {2, 2}, 1, VK_FORMAT_D32_SFLOAT, vk.memory,
            dispatch, vkCreateSampler, vkDestroySampler, error, nullptr, true, &config)) throw std::runtime_error(error);
    if (targets.samples() != samples) throw std::runtime_error("G-buffer multisample count mismatch");
    check(vkResetCommandBuffer(vk.command, 0)); check(vkBeginCommandBuffer(vk.command, &begin));
    FrameRecordingContext frame{vk.command, VK_NULL_HANDLE, VK_NULL_HANDLE, {2, 2}, 0, 0}, geometry;
    if (!targets.begin(frame, geometry)) throw std::runtime_error("multisample G-buffer did not begin");
    VkClearAttachment clear_depth{}; clear_depth.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    clear_depth.clearValue.depthStencil = {.625f, 0};
    VkClearRect rectangle{}; rectangle.rect.extent = {1, 2}; rectangle.layerCount = 1;
    vkCmdClearAttachments(vk.command, 1, &clear_depth, 1, &rectangle);
    targets.end(vk.command);
    if (!targets.copy_depth(frame)) throw std::runtime_error("multisample resolved depth copy failed");
    VkImageMemoryBarrier read{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    read.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; read.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    read.srcQueueFamilyIndex = read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    read.image = targets.sampled_depth_image(0); read.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(vk.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &read);
    VkBufferImageCopy depth_copy{}; depth_copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    depth_copy.imageExtent = {2, 2, 1};
    vkCmdCopyImageToBuffer(vk.command, targets.sampled_depth_image(0), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        vk.readback, 1, &depth_copy);
    vkCmdPipelineBarrier(vk.command, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
    check(vkEndCommandBuffer(vk.command)); check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE)); check(vkQueueWaitIdle(vk.queue));
    check(vkMapMemory(vk.device, vk.readback_memory, 0, 4 * sizeof(float), 0, &data));
    const auto* depths = static_cast<const float*>(data); bool correct_depth = true;
    for (uint32_t i = 0; i < 4; ++i) correct_depth &= std::abs(depths[i] - (i % 2 ? 1.f : .625f)) < 1e-5f;
    vkUnmapMemory(vk.device, vk.readback_memory);
    if (!correct_depth) throw std::runtime_error("multisample G-buffer depth preservation mismatch");
    targets.destroy();
    if (!targets.initialize(vk.physical, vk.device, {2, 2}, 1, VK_FORMAT_D32_SFLOAT,
            vk.memory, dispatch, vkCreateSampler, vkDestroySampler, error,
            nullptr, false, &config))
        throw std::runtime_error("multisample scene depth targets: " + error);
    if (targets.scene_depth_image(0) == targets.graph_attachment_image(0, 3))
        throw std::runtime_error("scene depth aliases sampled G-buffer depth");
    check(vkResetCommandBuffer(vk.command, 0));
    check(vkBeginCommandBuffer(vk.command, &begin));
    frame.graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT;
    if (!targets.begin(frame, geometry))
        throw std::runtime_error("multisample scene depth prepass failed to begin");
    targets.end(vk.command);
    if (!targets.copy_scene_depth(frame))
        throw std::runtime_error("multisample scene depth copy failed");
    check(vkEndCommandBuffer(vk.command));
    check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(vk.queue));
    targets.destroy();
    std::cout << physical.deviceName << ": multisampled G-buffer color and depth resolve passed\n";
}
}

int main(int argc, char** argv)
{
    test_msaa_policy();
    Device vk;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    uint32_t loader_version = VK_API_VERSION_1_0;
    auto enumerate_version = reinterpret_cast<PFN_vkEnumerateInstanceVersion>(vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
    if (enumerate_version) check(enumerate_version(&loader_version));
    application.apiVersion = std::min(loader_version, uint32_t(VK_API_VERSION_1_2));
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
        VkFormatProperties color{}, depth{};
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_R8G8B8A8_UNORM, &color);
        vkGetPhysicalDeviceFormatProperties(devices[index], VK_FORMAT_D32_SFLOAT, &depth);
        const VkFormatFeatureFlags color_needed =
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        const VkFormatFeatureFlags depth_needed =
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((color.optimalTilingFeatures & color_needed) != color_needed ||
            (depth.optimalTilingFeatures & depth_needed) != depth_needed)
            continue;
        uint32_t families{};
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, queues.data());
        for (uint32_t i = 0; i < families; ++i)
            if (queues[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
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
    VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    buffer_info.size = 4 * sizeof(float);
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    check(vkCreateBuffer(vk.device, &buffer_info, nullptr, &vk.readback));
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(vk.device, vk.readback, &requirements);
    vk.readback_memory = vk.host_memory(requirements);
    check(vkBindBufferMemory(vk.device, vk.readback, vk.readback_memory, 0));

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
    dispatch.create_framebuffer = vkCreateFramebuffer;
    dispatch.destroy_framebuffer = vkDestroyFramebuffer;
    dispatch.create_render_pass = vkCreateRenderPass;
    dispatch.destroy_render_pass = vkDestroyRenderPass;
    dispatch.cmd_begin_render_pass = vkCmdBeginRenderPass;
    dispatch.cmd_end_render_pass = vkCmdEndRenderPass;
    dispatch.cmd_pipeline_barrier = vkCmdPipelineBarrier;
    dispatch.cmd_copy_image = vkCmdCopyImage;
    FrameImagePool image_pool;
    if (!image_pool.initialize(vk.device, vk.memory,
            {vkCreateImage, vkDestroyImage, vkGetImageMemoryRequirements,
             vkAllocateMemory, vkFreeMemory, vkBindImageMemory}))
        throw std::runtime_error("G-buffer image pool initialization failed");
    GBufferTargets targets;
    std::string error;
    const auto rebuild = [&] {
        if (!targets.initialize(vk.physical, vk.device, {2, 2}, 1, VK_FORMAT_D32_SFLOAT,
                vk.memory, dispatch, vkCreateSampler, vkDestroySampler, error, &image_pool, true))
            throw std::runtime_error(error);
    };
    rebuild();
    for (float depth : {0.25f, 0.75f})
    {
        check(vkResetCommandBuffer(vk.command, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        check(vkBeginCommandBuffer(vk.command, &begin));
        FrameRecordingContext frame{vk.command, VK_NULL_HANDLE, VK_NULL_HANDLE, {2, 2}, 0, 0};
        FrameRecordingContext geometry;
        if (!targets.begin(frame, geometry))
            throw std::runtime_error("G-buffer render pass did not begin");
        VkClearAttachment clear{};
        clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        clear.clearValue.depthStencil = {depth, 0};
        VkClearRect rect{};
        rect.rect.extent = {1, 2};
        rect.layerCount = 1;
        vkCmdClearAttachments(vk.command, 1, &clear, 1, &rect);
        targets.end(vk.command);
        if (!targets.copy_depth(frame))
            throw std::runtime_error("G-buffer depth graph copy failed");
        VkImageMemoryBarrier read{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        read.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        read.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        read.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        read.srcQueueFamilyIndex = read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        read.image = targets.sampled_depth_image(0);
        read.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(vk.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &read);
        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
        copy.imageExtent = {2, 2, 1};
        vkCmdCopyImageToBuffer(vk.command, targets.sampled_depth_image(0),
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, vk.readback, 1, &copy);
        check(vkEndCommandBuffer(vk.command));
        VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &vk.command;
        check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
        check(vkQueueWaitIdle(vk.queue));
        void* mapped{};
        check(vkMapMemory(vk.device, vk.readback_memory, 0, 4 * sizeof(float), 0, &mapped));
        const auto* values = static_cast<const float*>(mapped);
        for (uint32_t pixel = 0; pixel < 4; ++pixel)
        {
            const float expected = pixel % 2 == 0 ? depth : 1.f;
            if (std::abs(values[pixel] - expected) > 1e-5f)
                throw std::runtime_error("G-buffer depth copy mismatch at pixel " +
                    std::to_string(pixel) + ": got " + std::to_string(values[pixel]));
        }
        vkUnmapMemory(vk.device, vk.readback_memory);
    }
    const VkImage reused = targets.sampled_depth_image(0);
    targets.destroy();
    rebuild();
    if (targets.sampled_depth_image(0) != reused)
        throw std::runtime_error("G-buffer depth image was not reused after GPU-idle rebuild");
    targets.destroy();
    image_pool.reset();
    test_msaa_resolve(vk, application.apiVersion, dispatch);
    return 0;
}
