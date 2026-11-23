#include "VsmAtlas.h"

namespace xray::render::vulkan
{
bool VsmAtlas::initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    VkFormat format, uint32_t page_size, uint32_t columns, uint32_t rows,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const FrameDispatch& dispatch, std::string& error, bool allow_readback, FrameImagePool* image_pool)
{
    destroy();
    if (!physical || !device || !get || (format != VK_FORMAT_D32_SFLOAT && format != VK_FORMAT_D16_UNORM) ||
        !page_size || !columns || !rows ||
        uint64_t(page_size) * columns > UINT32_MAX || uint64_t(page_size) * rows > UINT32_MAX ||
        !dispatch.get_format_properties || !dispatch.create_image || !dispatch.destroy_image ||
        !dispatch.get_image_memory_requirements || !dispatch.allocate_memory || !dispatch.free_memory ||
        !dispatch.bind_image_memory || !dispatch.create_image_view || !dispatch.destroy_image_view ||
        !dispatch.create_framebuffer || !dispatch.destroy_framebuffer || !dispatch.create_render_pass ||
        !dispatch.destroy_render_pass || !dispatch.cmd_begin_render_pass || !dispatch.cmd_end_render_pass ||
        !dispatch.cmd_pipeline_barrier || !dispatch.cmd_clear_attachments)
    { error = "invalid VSM atlas dimensions or Vulkan procedures"; return false; }
    const VkExtent2D extent{page_size * columns, page_size * rows};
    if (extent.width > limits.maxImageDimension2D || extent.height > limits.maxImageDimension2D ||
        extent.width > INT32_MAX || extent.height > INT32_MAX)
    { error = "VSM atlas exceeds maximum image dimensions"; return false; }
    VkFormatProperties support{};
    dispatch.get_format_properties(physical, format, &support);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((support.optimalTilingFeatures & required) != required)
    { error = "VSM depth format is not renderable and sampled"; return false; }
    const auto create_sampler = reinterpret_cast<PFN_vkCreateSampler>(get(device, "vkCreateSampler"));
    destroy_sampler_ = reinterpret_cast<PFN_vkDestroySampler>(get(device, "vkDestroySampler"));
    clear_depth_ = reinterpret_cast<PFN_vkCmdClearDepthStencilImage>(get(device, "vkCmdClearDepthStencilImage"));
    if (!create_sampler || !destroy_sampler_ || !clear_depth_)
    { error = "VSM atlas sampler or clear procedure unavailable"; return false; }
    device_ = device; vk_ = dispatch; image_pool_ = image_pool;
    extent_ = extent; page_size_ = page_size; columns_ = columns; rows_ = rows;
    const auto fail = [&](const char* why) { error = why; destroy(); return false; };
    VkAttachmentDescription depth{};
    depth.format = format; depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD; depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE; depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    const VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{}; subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &reference;
    const VkSubpassDependency dependencies[]{
        {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, 0},
        {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, 0}};
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1; pass.pAttachments = &depth; pass.subpassCount = 1; pass.pSubpasses = &subpass;
    pass.dependencyCount = 2; pass.pDependencies = dependencies;
    if (vk_.create_render_pass(device, &pass, nullptr, &pass_) != VK_SUCCESS) return fail("cannot create VSM atlas render pass");
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D; image.format = format; image.extent = {extent.width, extent.height, 1};
    image.mipLevels = image.arrayLayers = 1; image.samples = VK_SAMPLE_COUNT_1_BIT; image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | (allow_readback ? VK_IMAGE_USAGE_TRANSFER_SRC_BIT : 0);
    if (image_pool_)
    {
        if (!image_pool_->acquire(image, image_, error))
        { error = "VSM atlas image: " + error; destroy(); return false; }
    }
    else
    {
        if (vk_.create_image(device, &image, nullptr, &image_) != VK_SUCCESS) return fail("cannot create VSM atlas image");
        VkMemoryRequirements requirements{}; vk_.get_image_memory_requirements(device, image_, &requirements);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) && (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            { type = i; break; }
        if (type == UINT32_MAX) return fail("VSM atlas device-local memory unavailable");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size; allocation.memoryTypeIndex = type;
        if (vk_.allocate_memory(device, &allocation, nullptr, &memory_) != VK_SUCCESS ||
            vk_.bind_image_memory(device, image_, memory_, 0) != VK_SUCCESS) return fail("cannot bind VSM atlas memory");
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image_; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = format;
    view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (vk_.create_image_view(device, &view, nullptr, &view_) != VK_SUCCESS) return fail("cannot create VSM atlas view");
    VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
    framebuffer.renderPass = pass_; framebuffer.attachmentCount = 1; framebuffer.pAttachments = &view_;
    framebuffer.width = extent.width; framebuffer.height = extent.height; framebuffer.layers = 1;
    if (vk_.create_framebuffer(device, &framebuffer, nullptr, &framebuffer_) != VK_SUCCESS)
        return fail("cannot create VSM atlas framebuffer");
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST; sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (create_sampler(device, &sampler, nullptr, &sampler_) != VK_SUCCESS) return fail("cannot create VSM atlas sampler");
    error.clear(); return true;
}

bool VsmAtlas::initialize_layout(VkCommandBuffer command, std::string& error) const
{
    if (!command || !image_ || initialized_ || !clear_depth_)
    { error = "invalid VSM atlas initial layout request"; return false; }
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED; barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image_; barrier.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vk_.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    const VkClearDepthStencilValue clear{1.f, 0};
    clear_depth_(command, image_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &barrier.subresourceRange);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL; barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vk_.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    error.clear(); return true;
}

bool VsmAtlas::begin_page(const FrameRecordingContext& frame, uint32_t slot, FrameRecordingContext& page_frame,
    std::string& error) const
{
    if (!initialized_ || !frame.command_buffer || slot >= uint64_t(columns_) * rows_)
    { error = "invalid VSM atlas page request"; return false; }
    const VkOffset2D offset{static_cast<int32_t>((slot % columns_) * page_size_),
        static_cast<int32_t>((slot / columns_) * page_size_)};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = pass_; begin.framebuffer = framebuffer_; begin.renderArea = {offset, {page_size_, page_size_}};
    vk_.cmd_begin_render_pass(frame.command_buffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    VkClearAttachment clear{}; clear.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT; clear.clearValue.depthStencil = {1.f, 0};
    const VkClearRect rect{begin.renderArea, 0, 1};
    vk_.cmd_clear_attachments(frame.command_buffer, 1, &clear, 1, &rect);
    page_frame = frame;
    page_frame.render_pass = pass_;
    page_frame.framebuffer = framebuffer_;
    page_frame.extent = {page_size_, page_size_};
    page_frame.viewport_offset = offset;
    error.clear(); return true;
}
void VsmAtlas::end_page(VkCommandBuffer command) const
{ if (command && pass_) vk_.cmd_end_render_pass(command); }
void VsmAtlas::destroy()
{
    if (device_)
    {
        if (framebuffer_) vk_.destroy_framebuffer(device_, framebuffer_, nullptr);
        if (view_) vk_.destroy_image_view(device_, view_, nullptr);
        if (image_)
        {
            if (image_pool_) image_pool_->release(image_);
            else vk_.destroy_image(device_, image_, nullptr);
        }
        if (memory_) vk_.free_memory(device_, memory_, nullptr);
        if (sampler_) destroy_sampler_(device_, sampler_, nullptr);
        if (pass_) vk_.destroy_render_pass(device_, pass_, nullptr);
    }
    device_ = VK_NULL_HANDLE; vk_ = {}; image_pool_ = nullptr;
    image_ = VK_NULL_HANDLE; memory_ = VK_NULL_HANDLE; view_ = VK_NULL_HANDLE;
    framebuffer_ = VK_NULL_HANDLE; pass_ = VK_NULL_HANDLE; sampler_ = VK_NULL_HANDLE; extent_ = {};
    page_size_ = columns_ = rows_ = 0; initialized_ = false; destroy_sampler_ = nullptr; clear_depth_ = nullptr;
}
}
