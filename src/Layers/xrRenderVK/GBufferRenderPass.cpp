#include "DeferredPass.h"

namespace xray::render::vulkan
{
bool create_msaa_gbuffer_render_pass(VkDevice device, VkFormat albedo_format, VkFormat normal_format,
    VkFormat surface_format, VkFormat depth_format, VkSampleCountFlagBits samples,
    VkResolveModeFlagBits depth_resolve, VkResolveModeFlagBits stencil_resolve,
    PFN_vkCreateRenderPass2 create_render_pass, VkRenderPass& result, std::string& error)
{
    result = VK_NULL_HANDLE;
    const VkSampleCountFlags valid_samples = VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT |
        VK_SAMPLE_COUNT_8_BIT | VK_SAMPLE_COUNT_16_BIT | VK_SAMPLE_COUNT_32_BIT | VK_SAMPLE_COUNT_64_BIT;
    const uint32_t sample_value = static_cast<uint32_t>(samples);
    const uint32_t depth_value = static_cast<uint32_t>(depth_resolve);
    const uint32_t stencil_value = static_cast<uint32_t>(stencil_resolve);
    constexpr uint32_t valid_resolve = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT | VK_RESOLVE_MODE_AVERAGE_BIT |
        VK_RESOLVE_MODE_MIN_BIT | VK_RESOLVE_MODE_MAX_BIT;
    if (!device || !create_render_pass || albedo_format == VK_FORMAT_UNDEFINED || normal_format == VK_FORMAT_UNDEFINED ||
        surface_format == VK_FORMAT_UNDEFINED || depth_format == VK_FORMAT_UNDEFINED || !(sample_value & valid_samples) ||
        (sample_value & (sample_value - 1)) || !depth_value || !(depth_value & valid_resolve) ||
        (depth_value & (depth_value - 1)) || (stencil_value &&
            (!(stencil_value & (valid_resolve & ~VK_RESOLVE_MODE_AVERAGE_BIT)) || (stencil_value & (stencil_value - 1)))))
    { error = "invalid multisample G-buffer configuration"; return false; }
    const VkFormat formats[]{albedo_format, normal_format, surface_format, depth_format};
    VkAttachmentDescription2 attachments[8]{};
    for (uint32_t i = 0; i < 8; ++i)
    {
        auto& attachment = attachments[i];
        attachment.sType = VK_STRUCTURE_TYPE_ATTACHMENT_DESCRIPTION_2;
        attachment.format = formats[i % 4];
        attachment.samples = i < 4 ? samples : VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = i < 4 ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachment.finalLayout = i == 3 || i == 7 ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    VkAttachmentReference2 colors[3]{}, resolves[3]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        colors[i] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT};
        resolves[i] = {VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, i + 4, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_COLOR_BIT};
    }
    const VkAttachmentReference2 depth{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 3,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT};
    const VkAttachmentReference2 resolved_depth{VK_STRUCTURE_TYPE_ATTACHMENT_REFERENCE_2, nullptr, 7,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, VK_IMAGE_ASPECT_DEPTH_BIT};
    VkSubpassDescriptionDepthStencilResolve depth_info{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_DEPTH_STENCIL_RESOLVE};
    depth_info.depthResolveMode = depth_resolve; depth_info.stencilResolveMode = stencil_resolve;
    depth_info.pDepthStencilResolveAttachment = &resolved_depth;
    VkSubpassDescription2 subpass{VK_STRUCTURE_TYPE_SUBPASS_DESCRIPTION_2};
    subpass.pNext = &depth_info; subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 3; subpass.pColorAttachments = colors; subpass.pResolveAttachments = resolves;
    subpass.pDepthStencilAttachment = &depth;
    VkSubpassDependency2 dependencies[2]{};
    for (auto& dep : dependencies) dep.sType = VK_STRUCTURE_TYPE_SUBPASS_DEPENDENCY_2;
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL; dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    dependencies[1].srcSubpass = 0; dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo2 info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO_2};
    info.attachmentCount = 8; info.pAttachments = attachments;
    info.subpassCount = 1; info.pSubpasses = &subpass; info.dependencyCount = 2; info.pDependencies = dependencies;
    if (create_render_pass(device, &info, nullptr, &result) != VK_SUCCESS)
    { error = "could not create multisample G-buffer render pass"; return false; }
    error.clear(); return true;
}

bool create_gbuffer_render_pass(VkDevice device, VkFormat albedo_format, VkFormat normal_format,
                                VkFormat surface_format, VkFormat depth_format, const FrameDispatch &vk,
                                VkRenderPass &result, std::string &error)
{
    result = VK_NULL_HANDLE;
    if (!device || !vk.create_render_pass || albedo_format == VK_FORMAT_UNDEFINED ||
        normal_format == VK_FORMAT_UNDEFINED || surface_format == VK_FORMAT_UNDEFINED ||
        depth_format == VK_FORMAT_UNDEFINED)
    {
        error = "G-buffer requires color and depth formats and vkCreateRenderPass";
        return false;
    }
    VkAttachmentDescription attachments[4]{};
    for (unsigned i = 0; i < 4; ++i)
    {
        attachments[i].format = i == 0   ? albedo_format
                                : i == 1 ? normal_format
                                : i == 2 ? surface_format
                                         : depth_format;
        attachments[i].samples = VK_SAMPLE_COUNT_1_BIT;
        attachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        // Keep depth alongside albedo and normals so later deferred passes can
        // reconstruct positions and test depth without rerendering geometry.
        attachments[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachments[i].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachments[i].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachments[i].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        attachments[i].finalLayout =
            i < 3 ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    }
    const VkAttachmentReference color[]{{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                                        {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
                                        {2, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}};
    const VkAttachmentReference depth{3, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 3;
    subpass.pColorAttachments = color;
    subpass.pDepthStencilAttachment = &depth;
    const VkSubpassDependency dependencies[]{
        {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0,
         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         VK_DEPENDENCY_BY_REGION_BIT},
        {0, VK_SUBPASS_EXTERNAL,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
             VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
         VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
         0}};
    VkRenderPassCreateInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    info.attachmentCount = 4;
    info.pAttachments = attachments;
    info.subpassCount = 1;
    info.pSubpasses = &subpass;
    info.dependencyCount = 2;
    info.pDependencies = dependencies;
    if (vk.create_render_pass(device, &info, nullptr, &result) != VK_SUCCESS)
    {
        error = "could not create Vulkan G-buffer render pass";
        return false;
    }
    error.clear();
    return true;
}
}
