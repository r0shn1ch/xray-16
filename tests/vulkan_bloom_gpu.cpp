#include "src/Layers/xrRenderVK/FrameImageGraph.h"
#include "src/Layers/xrRenderVK/PostProcessShaders.h"
#include "src/Layers/xrRenderVK/ScreenCopyPass.h"
#include "src/Layers/xrRenderVK/ShaderModule.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cstdint>
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
    if (result != VK_SUCCESS)
        throw std::runtime_error("Vulkan bloom GPU result " + std::to_string(result));
}

struct Fixture
{
    VkInstance instance{};
    VkPhysicalDevice physical{};
    VkDevice device{};
    VkQueue queue{};
    VkCommandPool command_pool{};
    VkPhysicalDeviceMemoryProperties memory{};
    std::vector<VkFramebuffer> framebuffers;
    std::vector<VkRenderPass> passes;
    std::vector<VkImageView> views;
    std::vector<VkImage> images;
    std::vector<VkBuffer> buffers;
    std::vector<VkDeviceMemory> allocations;
    VkSampler sampler{};

    ~Fixture()
    {
        if (device)
        {
            vkDeviceWaitIdle(device);
            if (sampler) vkDestroySampler(device, sampler, nullptr);
            for (auto value : framebuffers) vkDestroyFramebuffer(device, value, nullptr);
            for (auto value : passes) vkDestroyRenderPass(device, value, nullptr);
            for (auto value : views) vkDestroyImageView(device, value, nullptr);
            for (auto value : images) vkDestroyImage(device, value, nullptr);
            for (auto value : buffers) vkDestroyBuffer(device, value, nullptr);
            for (auto value : allocations) vkFreeMemory(device, value, nullptr);
            if (command_pool) vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);
        }
        if (instance) vkDestroyInstance(instance, nullptr);
    }

    VkDeviceMemory allocate(VkMemoryRequirements requirements, VkMemoryPropertyFlags flags)
    {
        for (uint32_t type = 0; type < memory.memoryTypeCount; ++type)
            if ((requirements.memoryTypeBits & (1u << type)) &&
                (memory.memoryTypes[type].propertyFlags & flags) == flags)
            {
                VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                info.allocationSize = requirements.size;
                info.memoryTypeIndex = type;
                VkDeviceMemory allocation{};
                check(vkAllocateMemory(device, &info, nullptr, &allocation));
                allocations.push_back(allocation);
                return allocation;
            }
        throw std::runtime_error("no suitable bloom GPU memory type");
    }

    VkImage image(VkFormat format, VkExtent2D extent, uint32_t mips, VkImageUsageFlags usage)
    {
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.imageType = VK_IMAGE_TYPE_2D;
        info.format = format;
        info.extent = {extent.width, extent.height, 1};
        info.mipLevels = mips;
        info.arrayLayers = 1;
        info.samples = VK_SAMPLE_COUNT_1_BIT;
        info.tiling = VK_IMAGE_TILING_OPTIMAL;
        info.usage = usage;
        VkImage result{};
        check(vkCreateImage(device, &info, nullptr, &result));
        images.push_back(result);
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device, result, &requirements);
        check(vkBindImageMemory(device, result,
            allocate(requirements, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
        return result;
    }

    VkImageView view(VkImage image, VkFormat format, uint32_t mip, uint32_t levels = 1)
    {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = format;
        info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, mip, levels, 0, 1};
        VkImageView result{};
        check(vkCreateImageView(device, &info, nullptr, &result));
        views.push_back(result);
        return result;
    }

    VkBuffer buffer(VkDeviceSize size, VkBufferUsageFlags usage, VkDeviceMemory& allocation)
    {
        VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        info.size = size;
        info.usage = usage;
        VkBuffer result{};
        check(vkCreateBuffer(device, &info, nullptr, &result));
        buffers.push_back(result);
        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device, result, &requirements);
        allocation = allocate(requirements,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        check(vkBindBufferMemory(device, result, allocation, 0));
        return result;
    }

    VkRenderPass pass(VkFormat format, VkAttachmentLoadOp load)
    {
        VkAttachmentDescription attachment{};
        attachment.format = format;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = load;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &reference;
        VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
        info.attachmentCount = 1;
        info.pAttachments = &attachment;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        VkRenderPass result{};
        check(vkCreateRenderPass(device, &info, nullptr, &result));
        passes.push_back(result);
        return result;
    }

    VkFramebuffer framebuffer(VkRenderPass pass, VkImageView view, VkExtent2D extent)
    {
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = pass;
        info.attachmentCount = 1;
        info.pAttachments = &view;
        info.width = extent.width;
        info.height = extent.height;
        info.layers = 1;
        VkFramebuffer result{};
        check(vkCreateFramebuffer(device, &info, nullptr, &result));
        framebuffers.push_back(result);
        return result;
    }
};
}

int main(int argc, char** argv) try
{
    Fixture vk;
    VkApplicationInfo application{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    application.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instance_info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instance_info.pApplicationInfo = &application;
    check(vkCreateInstance(&instance_info, nullptr, &vk.instance));
    uint32_t count = 0;
    check(vkEnumeratePhysicalDevices(vk.instance, &count, nullptr));
    if (!count) return 77;
    std::vector<VkPhysicalDevice> physical(count);
    check(vkEnumeratePhysicalDevices(vk.instance, &count, physical.data()));
    uint32_t family = UINT32_MAX;
    const int requested_gpu = argc > 1 ? std::stoi(argv[1]) : -1;
    for (size_t index = 0; index < physical.size(); ++index)
    {
        if (requested_gpu >= 0 && size_t(requested_gpu) != index) continue;
        auto gpu = physical[index];
        uint32_t queue_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_count, nullptr);
        std::vector<VkQueueFamilyProperties> families(queue_count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &queue_count, families.data());
        VkFormatProperties properties{};
        vkGetPhysicalDeviceFormatProperties(gpu, VK_FORMAT_R16G16B16A16_SFLOAT, &properties);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
            VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if ((properties.optimalTilingFeatures & required) != required) continue;
        vkGetPhysicalDeviceFormatProperties(gpu, VK_FORMAT_R32_SFLOAT, &properties);
        if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) continue;
        for (uint32_t i = 0; i < queue_count; ++i)
            if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                vk.physical = gpu;
                family = i;
                break;
            }
        if (vk.physical) break;
    }
    if (!vk.physical) return 77;
    const float priority = 1.f;
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
    check(vkCreateCommandPool(vk.device, &pool_info, nullptr, &vk.command_pool));
    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = vk.command_pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    VkCommandBuffer command{};
    check(vkAllocateCommandBuffers(vk.device, &command_info, &command));

    const VkExtent2D scene_extent{8, 8}, bloom_extent{4, 4}, scaled_extent{16, 16};
    const auto scene_image = vk.image(VK_FORMAT_R8G8B8A8_UNORM, scene_extent, 1,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    const auto depth_image = vk.image(VK_FORMAT_R32_SFLOAT, scene_extent, 1,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    const auto shaft_depth_image = vk.image(VK_FORMAT_R32_SFLOAT, scene_extent, 1,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    const auto bloom_image = vk.image(VK_FORMAT_R16G16B16A16_SFLOAT, bloom_extent, 2,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    const auto output_image = vk.image(VK_FORMAT_R8G8B8A8_UNORM, scene_extent, 1,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto dof_image = vk.image(VK_FORMAT_R8G8B8A8_UNORM, scene_extent, 1,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto focused_image = vk.image(VK_FORMAT_R8G8B8A8_UNORM, scene_extent, 1,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto shafts_image = vk.image(VK_FORMAT_R8G8B8A8_UNORM, scene_extent, 1,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto linear_image = vk.image(VK_FORMAT_R8G8B8A8_UNORM, scaled_extent, 1,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto cubic_image = vk.image(VK_FORMAT_R8G8B8A8_UNORM, scaled_extent, 1,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    const auto scene_view = vk.view(scene_image, VK_FORMAT_R8G8B8A8_UNORM, 0);
    const auto depth_view = vk.view(depth_image, VK_FORMAT_R32_SFLOAT, 0);
    const auto shaft_depth_view = vk.view(shaft_depth_image, VK_FORMAT_R32_SFLOAT, 0);
    const auto bloom_view0 = vk.view(bloom_image, VK_FORMAT_R16G16B16A16_SFLOAT, 0);
    const auto bloom_view1 = vk.view(bloom_image, VK_FORMAT_R16G16B16A16_SFLOAT, 1);
    const auto full_bloom_view = vk.view(bloom_image, VK_FORMAT_R16G16B16A16_SFLOAT, 0, 2);
    const auto output_view = vk.view(output_image, VK_FORMAT_R8G8B8A8_UNORM, 0);
    const auto dof_view = vk.view(dof_image, VK_FORMAT_R8G8B8A8_UNORM, 0);
    const auto focused_view = vk.view(focused_image, VK_FORMAT_R8G8B8A8_UNORM, 0);
    const auto shafts_view = vk.view(shafts_image, VK_FORMAT_R8G8B8A8_UNORM, 0);
    const auto linear_view = vk.view(linear_image, VK_FORMAT_R8G8B8A8_UNORM, 0);
    const auto cubic_view = vk.view(cubic_image, VK_FORMAT_R8G8B8A8_UNORM, 0);
    const auto bright_render_pass = vk.pass(VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_CLEAR);
    const auto upsample_render_pass = vk.pass(VK_FORMAT_R16G16B16A16_SFLOAT, VK_ATTACHMENT_LOAD_OP_LOAD);
    const auto composite_render_pass = vk.pass(VK_FORMAT_R8G8B8A8_UNORM, VK_ATTACHMENT_LOAD_OP_CLEAR);
    const auto bright_framebuffer = vk.framebuffer(bright_render_pass, bloom_view0, bloom_extent);
    const auto upsample_framebuffer = vk.framebuffer(upsample_render_pass, bloom_view0, bloom_extent);
    const auto composite_framebuffer = vk.framebuffer(composite_render_pass, output_view, scene_extent);
    const auto dof_framebuffer = vk.framebuffer(composite_render_pass, dof_view, scene_extent);
    const auto focused_framebuffer = vk.framebuffer(composite_render_pass, focused_view, scene_extent);
    const auto shafts_framebuffer = vk.framebuffer(composite_render_pass, shafts_view, scene_extent);
    const auto linear_framebuffer = vk.framebuffer(composite_render_pass, linear_view, scaled_extent);
    const auto cubic_framebuffer = vk.framebuffer(composite_render_pass, cubic_view, scaled_extent);
    VkSamplerCreateInfo sampler_info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = sampler_info.addressModeV = sampler_info.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    check(vkCreateSampler(vk.device, &sampler_info, nullptr, &vk.sampler));

    std::string error;
    ShaderModuleDispatch module_dispatch{vkCreateShaderModule, vkDestroyShaderModule};
    ShaderModule vertex, bright_shader, upsample_shader, composite_shader, shafts_shader;
    if (!vertex.initialize(vk.device, module_dispatch, postprocess_shaders::Vertex,
            sizeof(postprocess_shaders::Vertex), error) ||
        !bright_shader.initialize(vk.device, module_dispatch, postprocess_shaders::BrightFragment,
            sizeof(postprocess_shaders::BrightFragment), error) ||
        !upsample_shader.initialize(vk.device, module_dispatch, postprocess_shaders::UpsampleFragment,
            sizeof(postprocess_shaders::UpsampleFragment), error) ||
        !composite_shader.initialize(vk.device, module_dispatch, postprocess_shaders::Fragment,
            sizeof(postprocess_shaders::Fragment), error) ||
        !shafts_shader.initialize(vk.device, module_dispatch, postprocess_shaders::SunShaftsFragment,
            sizeof(postprocess_shaders::SunShaftsFragment), error))
        throw std::runtime_error(error);
    ScreenCopyDispatch dispatch;
    if (!load_screen_copy_dispatch(vk.device, vkGetDeviceProcAddr, dispatch, error))
        throw std::runtime_error(error);
    ScreenCopyPass bright, upsample, composite, dof, focused, shafts, linear_scale, cubic_scale;
    if (!bright.initialize(vk.device, bright_render_pass, scene_view, vk.sampler,
            vertex.handle(), bright_shader.handle(), dispatch, error) ||
        !upsample.initialize(vk.device, upsample_render_pass, bloom_view1, vk.sampler,
            vertex.handle(), upsample_shader.handle(), dispatch, error, true) ||
        !composite.initialize(vk.device, composite_render_pass, scene_view, vk.sampler,
            vertex.handle(), composite_shader.handle(), dispatch, error) ||
        !dof.initialize(vk.device, composite_render_pass, scene_view, vk.sampler,
            vertex.handle(), composite_shader.handle(), dispatch, error) ||
        !focused.initialize(vk.device, composite_render_pass, scene_view, vk.sampler,
            vertex.handle(), composite_shader.handle(), dispatch, error) ||
        !linear_scale.initialize(vk.device, composite_render_pass, scene_view, vk.sampler,
            vertex.handle(), composite_shader.handle(), dispatch, error) ||
        !cubic_scale.initialize(vk.device, composite_render_pass, scene_view, vk.sampler,
            vertex.handle(), composite_shader.handle(), dispatch, error) ||
        !shafts.initialize(vk.device, composite_render_pass, scene_view, vk.sampler,
            vertex.handle(), shafts_shader.handle(), dispatch, error, true))
        throw std::runtime_error(error);
    composite.set_bloom_view(full_bloom_view);
    dof.set_depth_view(depth_view, vk.sampler);
    focused.set_depth_view(depth_view, vk.sampler);
    shafts.set_depth_view(shaft_depth_view, vk.sampler);
    PostProcessConstants shaft_constants;
    shaft_constants.tint[0] = .25f;
    shaft_constants.tint[1] = .05f;
    shaft_constants.tint[2] = 1.f;
    shaft_constants.add[0] = shaft_constants.add[1] = shaft_constants.add[2] = 1.f;
    shaft_constants.effect[0] = 16.f;
    shafts.set_constants(shaft_constants);
    PostProcessConstants constants;
    constants.bloom[1] = 0.5f;
    constants.bloom[2] = 4.f;
    constants.bloom[3] = 0.1f;
    bright.set_constants(constants);
    upsample.set_constants(constants);
    constants.bloom[0] = 1.f;
    composite.set_constants(constants);
    constants.bloom[0] = 0.f;
    constants.tint[3] = .998f;
    constants.add[3] = 1.f;
    dof.set_constants(constants);
    constants.tint[3] = .9998f;
    focused.set_constants(constants);
    PostProcessConstants scale_constants;
    scale_constants.bloom[0] = 0.f;
    linear_scale.set_constants(scale_constants);
    scale_constants.color_map[3] = 2.f;
    cubic_scale.set_constants(scale_constants);

    constexpr VkDeviceSize scene_bytes = 8 * 8 * 4, bloom_bytes = 4 * 4 * 4 * 2;
    VkDeviceMemory upload_memory{}, depth_upload_memory{}, shaft_depth_upload_memory{}, before_memory{}, after_memory{}, output_memory{}, dof_memory{}, focused_memory{}, shafts_memory{}, linear_memory{}, cubic_memory{};
    const auto upload = vk.buffer(scene_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, upload_memory);
    const auto depth_upload = vk.buffer(scene_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, depth_upload_memory);
    const auto shaft_depth_upload = vk.buffer(scene_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, shaft_depth_upload_memory);
    const auto before = vk.buffer(bloom_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, before_memory);
    const auto after = vk.buffer(bloom_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, after_memory);
    const auto output = vk.buffer(scene_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, output_memory);
    const auto dof_readback = vk.buffer(scene_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, dof_memory);
    const auto focused_readback = vk.buffer(scene_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, focused_memory);
    const auto shafts_readback = vk.buffer(scene_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, shafts_memory);
    constexpr VkDeviceSize scaled_bytes = 16 * 16 * 4;
    const auto linear_readback = vk.buffer(scaled_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, linear_memory);
    const auto cubic_readback = vk.buffer(scaled_bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, cubic_memory);
    std::array<uint8_t, scene_bytes> pixels{};
    for (unsigned y = 0; y < 8; ++y)
        for (unsigned x = 0; x < 8; ++x)
        {
            const uint8_t value = x >= 2 && x < 6 && y >= 2 && y < 6 ? 255 : 20;
            const size_t offset = (y * 8 + x) * 4;
            pixels[offset] = pixels[offset + 1] = pixels[offset + 2] = value;
            pixels[offset + 3] = 255;
        }
    void* mapped{};
    check(vkMapMemory(vk.device, upload_memory, 0, scene_bytes, 0, &mapped));
    std::memcpy(mapped, pixels.data(), pixels.size());
    vkUnmapMemory(vk.device, upload_memory);
    std::array<float, 64> depth_pixels{};
    depth_pixels.fill(.9998f);
    check(vkMapMemory(vk.device, depth_upload_memory, 0, scene_bytes, 0, &mapped));
    std::memcpy(mapped, depth_pixels.data(), scene_bytes);
    vkUnmapMemory(vk.device, depth_upload_memory);
    for (unsigned y = 0; y < 2; ++y)
        for (unsigned x = 0; x < 4; ++x)
            depth_pixels[y * 8 + x] = 1.f;
    check(vkMapMemory(vk.device, shaft_depth_upload_memory, 0, scene_bytes, 0, &mapped));
    std::memcpy(mapped, depth_pixels.data(), scene_bytes);
    vkUnmapMemory(vk.device, shaft_depth_upload_memory);

    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    check(vkBeginCommandBuffer(command, &begin));
    ImageStateTracker states;
    const VkImageSubresourceRange scene_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImageSubresourceRange bloom_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 0, 1};
    const VkImageSubresourceRange mip0{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    const VkImageSubresourceRange mip1{VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1};
    if (!states.register_image(scene_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(depth_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(shaft_depth_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(bloom_image, bloom_range, ImageUse::Undefined, error) ||
        !states.register_image(output_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(dof_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(focused_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(linear_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(cubic_image, scene_range, ImageUse::Undefined, error) ||
        !states.register_image(shafts_image, scene_range, ImageUse::Undefined, error) ||
        !states.transition(command, scene_image, ImageUse::TransferDestination,
            {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);
    VkBufferImageCopy upload_region{};
    upload_region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    upload_region.imageExtent = {8, 8, 1};
    vkCmdCopyBufferToImage(command, upload, scene_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload_region);
    if (!states.transition(command, scene_image, ImageUse::Sampled,
            {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);
    if (!states.transition(command, depth_image, ImageUse::TransferDestination,
            {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);
    vkCmdCopyBufferToImage(command, depth_upload, depth_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload_region);
    if (!states.transition(command, depth_image, ImageUse::Sampled,
            {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);
    if (!states.transition(command, shaft_depth_image, ImageUse::TransferDestination,
            {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);
    vkCmdCopyBufferToImage(command, shaft_depth_upload, shaft_depth_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &upload_region);
    if (!states.transition(command, shaft_depth_image, ImageUse::Sampled,
            {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);

    FrameImageGraph graph;
    FrameImageGraph::Resource scene, depth, shaft_depth, bloom0, bloom1, output_resource, dof_resource, focused_resource, shafts_resource, linear_resource, cubic_resource;
    if (!graph.import_image(scene_image, scene_range, scene, error) ||
        !graph.import_image(depth_image, scene_range, depth, error) ||
        !graph.import_image(shaft_depth_image, scene_range, shaft_depth, error) ||
        !graph.import_image(bloom_image, mip0, bloom0, error) ||
        !graph.import_image(bloom_image, mip1, bloom1, error) ||
        !graph.import_image(output_image, scene_range, output_resource, error) ||
        !graph.import_image(dof_image, scene_range, dof_resource, error) ||
        !graph.import_image(focused_image, scene_range, focused_resource, error) ||
        !graph.import_image(linear_image, scene_range, linear_resource, error) ||
        !graph.import_image(cubic_image, scene_range, cubic_resource, error) ||
        !graph.import_image(shafts_image, scene_range, shafts_resource, error))
        throw std::runtime_error(error);
    const auto bright_pass = graph.add_pass("bright", [&] {
        VkClearValue clear{};
        VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        info.renderPass = bright_render_pass;
        info.framebuffer = bright_framebuffer;
        info.renderArea.extent = bloom_extent;
        info.clearValueCount = 1;
        info.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
        bright.record({command, bright_render_pass, bright_framebuffer, bloom_extent, 0, 0});
        vkCmdEndRenderPass(command);
    });
    if (!graph.read(bright_pass, scene, ImageUse::Sampled, error) ||
        !graph.write(bright_pass, bloom0, ImageUse::ColorAttachment, bloom0, error))
        throw std::runtime_error(error);
    const auto downsample_pass = graph.add_pass("downsample", [&] {
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1};
        blit.srcOffsets[1] = {4, 4, 1};
        blit.dstOffsets[1] = {2, 2, 1};
        vkCmdBlitImage(command, bloom_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            bloom_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    });
    if (!graph.read(downsample_pass, bloom0, ImageUse::TransferSource, error) ||
        !graph.write(downsample_pass, bloom1, ImageUse::TransferDestination, bloom1, error))
        throw std::runtime_error(error);
    const auto copy_pixels = [&](VkBuffer target) {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {4, 4, 1};
        vkCmdCopyImageToBuffer(command, bloom_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            target, 1, &region);
    };
    const auto before_pass = graph.add_pass("read bright", [&] { copy_pixels(before); });
    if (!graph.read(before_pass, bloom0, ImageUse::TransferSource, error))
        throw std::runtime_error(error);
    const auto upsample_pass = graph.add_pass("upsample", [&] {
        VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        info.renderPass = upsample_render_pass;
        info.framebuffer = upsample_framebuffer;
        info.renderArea.extent = bloom_extent;
        vkCmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
        upsample.record({command, upsample_render_pass, upsample_framebuffer, bloom_extent, 0, 0});
        vkCmdEndRenderPass(command);
    });
    if (!graph.read(upsample_pass, bloom1, ImageUse::Sampled, error) ||
        !graph.read_write(upsample_pass, bloom0, bloom0, error, ImageUse::ColorAttachment))
        throw std::runtime_error(error);
    const auto after_pass = graph.add_pass("read upsampled", [&] { copy_pixels(after); });
    if (!graph.read(after_pass, bloom0, ImageUse::TransferSource, error))
        throw std::runtime_error(error);
    const auto composite_pass = graph.add_pass("composite", [&] {
        VkClearValue clear{};
        VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        info.renderPass = composite_render_pass;
        info.framebuffer = composite_framebuffer;
        info.renderArea.extent = scene_extent;
        info.clearValueCount = 1;
        info.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
        composite.record({command, composite_render_pass, composite_framebuffer, scene_extent, 0, 0});
        vkCmdEndRenderPass(command);
    });
    if (!graph.read(composite_pass, scene, ImageUse::Sampled, error) ||
        !graph.read(composite_pass, bloom0, ImageUse::Sampled, error) ||
        !graph.read(composite_pass, bloom1, ImageUse::Sampled, error) ||
        !graph.write(composite_pass, output_resource, ImageUse::ColorAttachment,
            output_resource, error))
        throw std::runtime_error(error);
    const auto shafts_pass = graph.add_pass("sun shafts", [&] {
        VkClearValue clear{};
        VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        info.renderPass = composite_render_pass;
        info.framebuffer = shafts_framebuffer;
        info.renderArea.extent = scene_extent;
        info.clearValueCount = 1;
        info.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
        composite.record({command, composite_render_pass, shafts_framebuffer, scene_extent, 0, 0});
        shafts.record({command, composite_render_pass, shafts_framebuffer, scene_extent, 0, 0});
        vkCmdEndRenderPass(command);
    });
    if (!graph.read(shafts_pass, scene, ImageUse::Sampled, error) ||
        !graph.read(shafts_pass, bloom0, ImageUse::Sampled, error) ||
        !graph.read(shafts_pass, bloom1, ImageUse::Sampled, error) ||
        !graph.read(shafts_pass, shaft_depth, ImageUse::Sampled, error) ||
        !graph.write(shafts_pass, shafts_resource, ImageUse::ColorAttachment, shafts_resource, error))
        throw std::runtime_error(error);
    const auto dof_pass = graph.add_pass("depth of field", [&] {
        VkClearValue clear{};
        VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        info.renderPass = composite_render_pass;
        info.framebuffer = dof_framebuffer;
        info.renderArea.extent = scene_extent;
        info.clearValueCount = 1;
        info.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
        dof.record({command, composite_render_pass, dof_framebuffer, scene_extent, 0, 0});
        vkCmdEndRenderPass(command);
    });
    if (!graph.read(dof_pass, scene, ImageUse::Sampled, error) ||
        !graph.read(dof_pass, depth, ImageUse::Sampled, error) ||
        !graph.write(dof_pass, dof_resource, ImageUse::ColorAttachment, dof_resource, error))
        throw std::runtime_error(error);
    const auto focused_pass = graph.add_pass("focused scene", [&] {
        VkClearValue clear{};
        VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        info.renderPass = composite_render_pass;
        info.framebuffer = focused_framebuffer;
        info.renderArea.extent = scene_extent;
        info.clearValueCount = 1;
        info.pClearValues = &clear;
        vkCmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
        focused.record({command, composite_render_pass, focused_framebuffer, scene_extent, 0, 0});
        vkCmdEndRenderPass(command);
    });
    if (!graph.read(focused_pass, scene, ImageUse::Sampled, error) ||
        !graph.read(focused_pass, depth, ImageUse::Sampled, error) ||
        !graph.write(focused_pass, focused_resource, ImageUse::ColorAttachment, focused_resource, error))
        throw std::runtime_error(error);
    const auto scaled_pass = [&](const char* name, ScreenCopyPass& pass, VkFramebuffer framebuffer,
        FrameImageGraph::Resource& target) {
        const auto index = graph.add_pass(name, [&, framebuffer, pass_ptr = &pass] {
            VkClearValue clear{};
            VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
            info.renderPass = composite_render_pass; info.framebuffer = framebuffer;
            info.renderArea.extent = scaled_extent; info.clearValueCount = 1; info.pClearValues = &clear;
            vkCmdBeginRenderPass(command, &info, VK_SUBPASS_CONTENTS_INLINE);
            pass_ptr->record({command, composite_render_pass, framebuffer, scaled_extent, 0, 0});
            vkCmdEndRenderPass(command);
        });
        return graph.read(index, scene, ImageUse::Sampled, error) &&
            graph.write(index, target, ImageUse::ColorAttachment, target, error);
    };
    if (!scaled_pass("bilinear scale", linear_scale, linear_framebuffer, linear_resource) ||
        !scaled_pass("cubic scale", cubic_scale, cubic_framebuffer, cubic_resource)) throw std::runtime_error(error);
    const auto read_scaled = [&](const char* name, VkImage image, VkBuffer buffer, FrameImageGraph::Resource resource) {
        const auto index = graph.add_pass(name, [&, image, buffer] {
            VkBufferImageCopy region{}; region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {16, 16, 1};
            vkCmdCopyImageToBuffer(command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, &region);
        });
        return graph.read(index, resource, ImageUse::TransferSource, error);
    };
    if (!read_scaled("read bilinear", linear_image, linear_readback, linear_resource) ||
        !read_scaled("read cubic", cubic_image, cubic_readback, cubic_resource)) throw std::runtime_error(error);
    const auto output_pass = graph.add_pass("read composite", [&] {
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {8, 8, 1};
        vkCmdCopyImageToBuffer(command, output_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            output, 1, &region);
    });
    if (!graph.read(output_pass, output_resource, ImageUse::TransferSource, error) ||
        !graph.read(graph.add_pass("read sun shafts", [&] {
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {8, 8, 1};
            vkCmdCopyImageToBuffer(command, shafts_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                shafts_readback, 1, &region);
        }), shafts_resource, ImageUse::TransferSource, error) ||
        !graph.read(graph.add_pass("read depth of field", [&] {
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {8, 8, 1};
            vkCmdCopyImageToBuffer(command, dof_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                dof_readback, 1, &region);
        }), dof_resource, ImageUse::TransferSource, error) ||
        !graph.read(graph.add_pass("read focused scene", [&] {
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {8, 8, 1};
            vkCmdCopyImageToBuffer(command, focused_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                focused_readback, 1, &region);
        }), focused_resource, ImageUse::TransferSource, error) ||
        !graph.execute(command, states, {vkCmdPipelineBarrier}, error))
        throw std::runtime_error(error);
    check(vkEndCommandBuffer(command));
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command;
    check(vkQueueSubmit(vk.queue, 1, &submit, VK_NULL_HANDLE));
    check(vkQueueWaitIdle(vk.queue));

    std::array<uint8_t, scaled_bytes> linear_pixels{}, cubic_pixels{};
    check(vkMapMemory(vk.device, linear_memory, 0, scaled_bytes, 0, &mapped));
    std::memcpy(linear_pixels.data(), mapped, scaled_bytes); vkUnmapMemory(vk.device, linear_memory);
    check(vkMapMemory(vk.device, cubic_memory, 0, scaled_bytes, 0, &mapped));
    std::memcpy(cubic_pixels.data(), mapped, scaled_bytes); vkUnmapMemory(vk.device, cubic_memory);
    const auto edge = (7 * 16 + 3) * 4;
    if (cubic_pixels[edge] >= linear_pixels[edge] || cubic_pixels[edge] < 20 ||
        cubic_pixels[(7 * 16 + 4) * 4] <= linear_pixels[(7 * 16 + 4) * 4])
        throw std::runtime_error("cubic upscale did not preserve the edge without overshoot");

    std::array<uint16_t, bloom_bytes / 2> bright_pixels{}, upsampled_pixels{};
    check(vkMapMemory(vk.device, before_memory, 0, bloom_bytes, 0, &mapped));
    std::memcpy(bright_pixels.data(), mapped, bloom_bytes);
    vkUnmapMemory(vk.device, before_memory);
    check(vkMapMemory(vk.device, after_memory, 0, bloom_bytes, 0, &mapped));
    std::memcpy(upsampled_pixels.data(), mapped, bloom_bytes);
    vkUnmapMemory(vk.device, after_memory);
    uint16_t maximum = 0;
    bool increased = false;
    for (size_t i = 0; i < bright_pixels.size(); ++i)
        if (i % 4 != 3)
        {
            maximum = std::max(maximum, bright_pixels[i]);
            increased |= upsampled_pixels[i] > bright_pixels[i];
        }
    if (!maximum || bright_pixels[0] || bright_pixels[1] || bright_pixels[2])
        throw std::runtime_error("bright pass did not separate highlights from dark pixels");
    if (!increased) throw std::runtime_error("bloom upsample did not add light");
    std::array<uint8_t, scene_bytes> composite_pixels{};
    check(vkMapMemory(vk.device, output_memory, 0, scene_bytes, 0, &mapped));
    std::memcpy(composite_pixels.data(), mapped, scene_bytes);
    vkUnmapMemory(vk.device, output_memory);
    if (composite_pixels[(1 * 8 + 1) * 4] <= 20)
        throw std::runtime_error("bloom composite did not light a dark nearby pixel");
    std::array<uint8_t, scene_bytes> shaft_pixels{};
    check(vkMapMemory(vk.device, shafts_memory, 0, scene_bytes, 0, &mapped));
    std::memcpy(shaft_pixels.data(), mapped, scene_bytes);
    vkUnmapMemory(vk.device, shafts_memory);
    const int left_shafts = int(shaft_pixels[(5 * 8 + 1) * 4]) - composite_pixels[(5 * 8 + 1) * 4];
    const int right_shafts = int(shaft_pixels[(5 * 8 + 6) * 4]) - composite_pixels[(5 * 8 + 6) * 4];
    if (left_shafts < 10 || left_shafts <= right_shafts + 5)
        throw std::runtime_error("sun shafts did not follow visible sky and depth occlusion: " +
            std::to_string(shaft_pixels[(5 * 8 + 1) * 4]) + "/" +
            std::to_string(composite_pixels[(5 * 8 + 1) * 4]) + " left, " +
            std::to_string(shaft_pixels[(5 * 8 + 6) * 4]) + "/" +
            std::to_string(composite_pixels[(5 * 8 + 6) * 4]) + " right");
    std::array<uint8_t, scene_bytes> dof_pixels{};
    check(vkMapMemory(vk.device, dof_memory, 0, scene_bytes, 0, &mapped));
    std::memcpy(dof_pixels.data(), mapped, scene_bytes);
    vkUnmapMemory(vk.device, dof_memory);
    if (dof_pixels[(1 * 8 + 1) * 4] <= 20 || dof_pixels[(3 * 8 + 3) * 4] >= 250)
        throw std::runtime_error("depth of field did not spread a bright scene region");
    std::array<uint8_t, scene_bytes> focused_pixels{};
    check(vkMapMemory(vk.device, focused_memory, 0, scene_bytes, 0, &mapped));
    std::memcpy(focused_pixels.data(), mapped, scene_bytes);
    vkUnmapMemory(vk.device, focused_memory);
    if (focused_pixels[(1 * 8 + 1) * 4] != 20 || focused_pixels[(3 * 8 + 3) * 4] != 255)
        throw std::runtime_error("depth of field blurred a focused surface");
}
catch (const std::exception& error)
{
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
