#include "LocalShadowTargets.h"
#include "../xrRenderGraph/LocalAtlasAllocator.h"

namespace xray::render::vulkan
{
bool LocalShadowTargets::initialize(VkDevice device, VkFormat format, uint32_t count,
                                    const VkPhysicalDeviceMemoryProperties &memory, const FrameDispatch &dispatch,
                                    const BufferResourceDispatch &buffers, PFN_vkCreateSampler create_sampler,
                                    PFN_vkDestroySampler destroy_sampler, std::string &error,
                                    VkDeviceSize uniform_alignment, FrameImagePool* image_pool)
{
    destroy();
    if (!device || format == VK_FORMAT_UNDEFINED || !count || count > 16 || !dispatch.create_image ||
        !dispatch.destroy_image || !dispatch.get_image_memory_requirements || !dispatch.allocate_memory ||
        !dispatch.free_memory || !dispatch.bind_image_memory || !dispatch.create_image_view ||
        !dispatch.destroy_image_view || !dispatch.create_framebuffer || !dispatch.destroy_framebuffer ||
        !dispatch.create_render_pass || !dispatch.destroy_render_pass || !dispatch.cmd_begin_render_pass ||
        !dispatch.cmd_end_render_pass || !create_sampler || !destroy_sampler)
    {
        error = "local shadow targets need complete image procedures";
        return false;
    }
    barrier_aspects_ = VK_IMAGE_ASPECT_DEPTH_BIT;
    if (format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
        format == VK_FORMAT_D16_UNORM_S8_UINT)
        barrier_aspects_ |= VK_IMAGE_ASPECT_STENCIL_BIT;
    device_ = device;
    vk_ = dispatch;
    memory_ = memory;
    buffers_ = buffers;
    image_pool_ = image_pool;
    const VkDeviceSize alignment = uniform_alignment ? uniform_alignment : 1;
    uniform_stride_ = (sizeof(LocalLightUniform) + alignment - 1) / alignment * alignment;
    destroy_sampler_ = destroy_sampler;
    VkAttachmentDescription depth{};
    depth.format = format;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    const VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &reference;
    const VkSubpassDependency dependencies[]{
        {VK_SUBPASS_EXTERNAL, 0, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
             VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
         VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
         VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         VK_DEPENDENCY_BY_REGION_BIT},
        {0, VK_SUBPASS_EXTERNAL,
         (VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT),
         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
         VK_DEPENDENCY_BY_REGION_BIT}};
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1;
    pass.pAttachments = &depth;
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = 2;
    pass.pDependencies = dependencies;
    if (vk_.create_render_pass(device, &pass, nullptr, &pass_) != VK_SUCCESS)
    {
        error = "cannot create local shadow render pass";
        destroy();
        return false;
    }
    depth.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    if (vk_.create_render_pass(device, &pass, nullptr, &tile_pass_) != VK_SUCCESS)
    {
        error = "cannot create local shadow tile pass";
        destroy();
        return false;
    }
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depth.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    if (vk_.create_render_pass(device, &pass, nullptr, &load_pass_) != VK_SUCCESS)
    {
        error = "cannot create local shadow load pass";
        destroy();
        return false;
    }
    depth.initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    if (vk_.create_render_pass(device, &pass, nullptr, &graph_load_pass_) != VK_SUCCESS)
    {
        error = "cannot create graph-managed local shadow load pass";
        destroy();
        return false;
    }
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    if (vk_.create_render_pass(device, &pass, nullptr, &graph_pass_) != VK_SUCCESS)
    {
        error = "cannot create graph-managed local shadow pass";
        destroy();
        return false;
    }
    image_count_ = count;
    cache_index_ = dispatch.cmd_copy_image && dispatch.cmd_pipeline_barrier ? count : UINT32_MAX;
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (create_sampler(device, &sampler, nullptr, &sampler_) != VK_SUCCESS)
    {
        error = "cannot create local shadow sampler";
        destroy();
        return false;
    }
    framegraph::LocalAtlasAllocator atlas;
    for (auto& tile : tiles_)
    {
        const auto node = atlas.Alloc(atlas.LevelOf(LocalShadowSize));
        if (node == UINT32_MAX)
        {
            error = "local shadow atlas cannot fit the configured faces";
            destroy();
            return false;
        }
        uint32_t x{}, y{}, size{};
        atlas.Rect(node, x, y, size);
        tile = {{int32_t(x), int32_t(y)}, {size, size}};
        atlas_extent_.width = std::max(atlas_extent_.width, x + size);
        atlas_extent_.height = std::max(atlas_extent_.height, y + size);
    }
    fixed_tiles_ = tiles_;
    targets_.resize(count + (caching() ? 1 : 0));
    for (Target &target : targets_)
    {
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = format;
        image.extent = {atlas_extent_.width, atlas_extent_.height, 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (image_pool_)
        {
            if (!image_pool_->acquire(image, target.image, error))
            {
                error = "local shadow array: " + error;
                destroy();
                return false;
            }
        }
        else
        {
            if (vk_.create_image(device, &image, nullptr, &target.image) != VK_SUCCESS)
            {
                error = "cannot create local shadow array";
                destroy();
                return false;
            }
            VkMemoryRequirements requirements{};
            vk_.get_image_memory_requirements(device, target.image, &requirements);
            uint32_t type = UINT32_MAX;
            for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
                if ((requirements.memoryTypeBits & (1u << i)) &&
                    (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
                {
                    type = i;
                    break;
                }
            if (type == UINT32_MAX)
            {
                error = "no device-local memory for local shadows";
                destroy();
                return false;
            }
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = type;
            if (vk_.allocate_memory(device, &allocation, nullptr, &target.memory) != VK_SUCCESS ||
                vk_.bind_image_memory(device, target.image, target.memory, 0) != VK_SUCCESS)
            {
                error = "cannot bind local shadow array";
                destroy();
                return false;
            }
        }
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = target.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        view.format = format;
        view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        view.subresourceRange.levelCount = 1;
        view.subresourceRange.layerCount = 1;
        if (vk_.create_image_view(device, &view, nullptr, &target.array_view) != VK_SUCCESS)
        {
            error = "cannot create local shadow array view";
            destroy();
            return false;
        }
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = pass_;
        framebuffer.attachmentCount = 1;
        framebuffer.pAttachments = &target.array_view;
        framebuffer.width = atlas_extent_.width;
        framebuffer.height = atlas_extent_.height;
        framebuffer.layers = 1;
        if (vk_.create_framebuffer(device, &framebuffer, nullptr, &target.framebuffer) != VK_SUCCESS)
        {
            error = "cannot create local shadow atlas framebuffer";
            destroy();
            return false;
        }
        if (!target.constants.initialize(
                device, uniform_stride_ * LocalLightCapacity, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, buffers, error))
        {
            destroy();
            return false;
        }
    }
    error.clear();
    return true;
}

bool LocalShadowTargets::reserve_lights(uint32_t image, size_t count, std::string &error)
{
    if (image >= image_count_)
        return false;
    auto &constants = targets_[image].constants;
    if (count <= constants.size() / uniform_stride_)
        return true;
    const size_t capacity = std::max(count, size_t(constants.size() / uniform_stride_) * 2);
    BufferResource replacement;
    if (!replacement.initialize(device_, uniform_stride_ * capacity, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory_, buffers_, error))
        return false;
    constants = std::move(replacement);
    return true;
}

void LocalShadowTargets::layout_tiles(const std::array<TileRequest, LocalShadowSlots>& requests)
{
    framegraph::LocalAtlasAllocator atlas;
    atlas.Reset(atlas_extent_.width, atlas_extent_.height);
    std::array<uint32_t, LocalShadowSlots> slots{};
    for (uint32_t i = 0; i < slots.size(); ++i)
        slots[i] = i;
    std::sort(slots.begin(), slots.end(), [&](uint32_t a, uint32_t b) {
        if (requests[a].resolution != requests[b].resolution)
            return requests[a].resolution > requests[b].resolution;
        return a < b;
    });

    decltype(tiles_) packed{};
    bool fits = true;
    for (uint32_t slot : slots)
    {
        const auto request = requests[slot];
        if (!request.faces || !request.resolution)
            continue;
        std::array<uint32_t, LocalShadowFaces> nodes{};
        uint32_t level = atlas.LevelOf(request.resolution);
        // Downgrade a light as a unit so point-light faces keep matching resolutions.
        while (level < framegraph::kLocalAtlasLevels && !atlas.AllocateFaces(level, request.faces, nodes))
            ++level;
        if (level == framegraph::kLocalAtlasLevels)
        {
            fits = false;
            break;
        }
        for (uint32_t face = 0; face < request.faces; ++face)
        {
            uint32_t x{}, y{}, size{};
            atlas.Rect(nodes[face], x, y, size);
            packed[slot * LocalShadowFaces + face] = {{int32_t(x), int32_t(y)}, {size, size}};
        }
    }
    if (!fits)
    {
        packed = {};
        for (uint32_t slot = 0; slot < requests.size(); ++slot)
            for (uint32_t face = 0; face < std::min(requests[slot].faces, LocalShadowFaces); ++face)
            {
                auto& tile = packed[slot * LocalShadowFaces + face];
                const auto& reserved = fixed_tiles_[slot * LocalShadowFaces + face];
                if (!requests[slot].resolution)
                    continue;
                const uint32_t size = std::min(requests[slot].resolution, reserved.extent.width);
                tile = {reserved.offset, {size, size}};
            }
    }

    for (size_t i = 0; i < tiles_.size(); ++i)
        if (tiles_[i].offset.x != packed[i].offset.x || tiles_[i].offset.y != packed[i].offset.y ||
            tiles_[i].extent.width != packed[i].extent.width)
            cache_signatures_[i] = 0;
    tiles_ = packed;
}

VkExtent2D LocalShadowTargets::face_extent(uint32_t slot, uint32_t face) const
{
    if (slot >= LocalShadowSlots || face >= LocalShadowFaces)
        return {};
    return tiles_[slot * LocalShadowFaces + face].extent;
}

bool LocalShadowTargets::write_light(uint32_t image, uint32_t index, const LocalLightUniform &value, std::string &error)
{
    if (image >= image_count_ || index >= targets_[image].constants.size() / uniform_stride_)
        return false;
    LocalLightUniform uniform = value;
    uniform.shadow_params[2] = 0.f;
    const float requested_base = value.shadow_params[0];
    if (value.shadow_params[2] > .5f && requested_base >= 0.f &&
        requested_base <= static_cast<float>(tiles_.size() - LocalShadowFaces))
    {
        const uint32_t base = static_cast<uint32_t>(requested_base);
        if (requested_base == static_cast<float>(base) && base % LocalShadowFaces == 0)
        {
            uniform.shadow_params[2] = value.shadow_params[2];
            for (uint32_t face = 0; face < LocalShadowFaces; ++face)
            {
                const auto& tile = tiles_[base + face];
                uniform.shadow_tiles[face][0] = tile.offset.x;
                uniform.shadow_tiles[face][1] = tile.offset.y;
                uniform.shadow_tiles[face][2] = int32_t(tile.extent.width);
                uniform.shadow_tiles[face][3] = 0;
            }
        }
    }
    return targets_[image].constants.write(VkDeviceSize(index) * uniform_stride_, &uniform, sizeof(uniform), error);
}

bool LocalShadowTargets::initialize_layers(const FrameRecordingContext &frame, bool graph_managed_layouts)
{
    if (!frame.command_buffer || frame.image_index >= image_count_)
        return false;
    const auto initialize = [&](Target& target) {
        if (target.initialized)
            return true;
        VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
        begin.renderPass = graph_managed_layouts ? graph_pass_ : pass_;
        begin.framebuffer = target.framebuffer;
        begin.renderArea.extent = atlas_extent_;
        VkClearValue clear{};
        clear.depthStencil = {1.f, 0};
        begin.clearValueCount = 1;
        begin.pClearValues = &clear;
        vk_.cmd_begin_render_pass(frame.command_buffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
        vk_.cmd_end_render_pass(frame.command_buffer);
        target.initialized = true;
        return true;
    };
    if (!initialize(targets_[frame.image_index]))
        return false;
    if (caching() && !initialize(targets_[cache_index_]))
        return false;
    return true;
}

bool LocalShadowTargets::begin_face(const FrameRecordingContext &frame, uint32_t slot, uint32_t face,
                                    FrameRecordingContext &shadow_frame, bool cached, bool preserve,
                                    bool graph_managed_layouts) const
{
    if (!frame.command_buffer || frame.image_index >= image_count_ || slot >= LocalShadowSlots ||
        face >= LocalShadowFaces || (cached && !caching()) || !face_extent(slot, face).width)
        return false;
    const VkFramebuffer framebuffer =
        targets_[cached ? cache_index_ : frame.image_index].framebuffer;
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = graph_managed_layouts ?
        (preserve ? graph_load_pass_ : graph_pass_) : (preserve ? load_pass_ : tile_pass_);
    begin.framebuffer = framebuffer;
    begin.renderArea = tiles_[slot * LocalShadowFaces + face];
    VkClearValue clear{};
    clear.depthStencil = {1.f, 0};
    begin.clearValueCount = 1;
    begin.pClearValues = &clear;
    vk_.cmd_begin_render_pass(frame.command_buffer, &begin, VK_SUBPASS_CONTENTS_INLINE);
    shadow_frame = frame;
    shadow_frame.render_pass = begin.renderPass;
    shadow_frame.framebuffer = framebuffer;
    shadow_frame.extent = begin.renderArea.extent;
    shadow_frame.viewport_offset = begin.renderArea.offset;
    return true;
}

namespace
{
uint64_t shadow_signature(const float *matrix)
{
    const auto *bytes = reinterpret_cast<const uint8_t *>(matrix);
    uint64_t hash = 14695981039346656037ull;
    for (unsigned i = 0; i < 64; ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ull;
    return hash ? hash : 1;
}
} // namespace
bool LocalShadowTargets::cached_face(uint32_t slot, uint32_t face, const float *matrix) const
{
    return caching() && matrix && face_extent(slot, face).width &&
           cache_signatures_[slot * LocalShadowFaces + face] == shadow_signature(matrix);
}
void LocalShadowTargets::remember_face(uint32_t slot, uint32_t face, const float *matrix)
{
    if (matrix && face_extent(slot, face).width)
        cache_signatures_[slot * LocalShadowFaces + face] = shadow_signature(matrix);
}
bool LocalShadowTargets::record_cached_face_copy(VkCommandBuffer command, uint32_t image,
    uint32_t slot, uint32_t face) const
{
    if (!caching() || !command || image >= image_count_ || !face_extent(slot, face).width)
        return false;
    const uint32_t layer = slot * LocalShadowFaces + face;
    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    const auto& tile = tiles_[layer];
    copy.srcOffset = copy.dstOffset = {tile.offset.x, tile.offset.y, 0};
    copy.extent = {tile.extent.width, tile.extent.height, 1};
    vk_.cmd_copy_image(command, targets_[cache_index_].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        targets_[image].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    return true;
}

bool LocalShadowTargets::copy_cached_face(const FrameRecordingContext &frame, uint32_t slot, uint32_t face) const
{
    if (!caching() || !frame.command_buffer || frame.image_index >= image_count_ || !face_extent(slot, face).width)
        return false;
    VkImageMemoryBarrier barriers[2]{};
    for (unsigned i = 0; i < 2; ++i)
    {
        auto &b = barriers[i];
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = targets_[i ? frame.image_index : cache_index_].image;
        b.subresourceRange = {barrier_aspects_, 0, 1, 0, 1};
        b.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        b.newLayout = i ? VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        b.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        b.dstAccessMask = i ? VK_ACCESS_TRANSFER_WRITE_BIT : VK_ACCESS_TRANSFER_READ_BIT;
    }
    vk_.cmd_pipeline_barrier(frame.command_buffer,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, barriers);
    record_cached_face_copy(frame.command_buffer, frame.image_index, slot, face);
    for (unsigned i = 0; i < 2; ++i)
    {
        auto &b = barriers[i];
        b.oldLayout = b.newLayout;
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        b.srcAccessMask = b.dstAccessMask;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                          VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    vk_.cmd_pipeline_barrier(frame.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                 VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0,
                             nullptr, 0, nullptr, 2, barriers);
    return true;
}

void LocalShadowTargets::end(VkCommandBuffer command) const
{
    if (command && pass_)
        vk_.cmd_end_render_pass(command);
}

VkImageView LocalShadowTargets::view(uint32_t image) const
{
    return image < image_count_ ? targets_[image].array_view : VK_NULL_HANDLE;
}
VkBuffer LocalShadowTargets::uniform(uint32_t image) const
{
    return image < image_count_ ? targets_[image].constants.handle() : VK_NULL_HANDLE;
}

void LocalShadowTargets::destroy()
{
    if (device_)
    {
        for (Target &target : targets_)
        {
            target.constants.destroy();
            if (target.framebuffer)
                vk_.destroy_framebuffer(device_, target.framebuffer, nullptr);
            if (target.array_view)
                vk_.destroy_image_view(device_, target.array_view, nullptr);
            if (target.image)
            {
                if (image_pool_) image_pool_->release(target.image);
                else vk_.destroy_image(device_, target.image, nullptr);
            }
            if (target.memory)
                vk_.free_memory(device_, target.memory, nullptr);
        }
        if (sampler_)
            destroy_sampler_(device_, sampler_, nullptr);
        if (pass_)
            vk_.destroy_render_pass(device_, pass_, nullptr);
        if (tile_pass_)
            vk_.destroy_render_pass(device_, tile_pass_, nullptr);
        if (load_pass_)
            vk_.destroy_render_pass(device_, load_pass_, nullptr);
        if (graph_load_pass_)
            vk_.destroy_render_pass(device_, graph_load_pass_, nullptr);
        if (graph_pass_)
            vk_.destroy_render_pass(device_, graph_pass_, nullptr);
    }
    targets_.clear();
    device_ = VK_NULL_HANDLE;
    pass_ = tile_pass_ = load_pass_ = graph_pass_ = graph_load_pass_ = VK_NULL_HANDLE;
    atlas_extent_ = {};
    image_count_ = 0;
    cache_index_ = UINT32_MAX;
    invalidate_cache();
    sampler_ = VK_NULL_HANDLE;
    destroy_sampler_ = nullptr;
    vk_ = {};
    image_pool_ = nullptr;
    uniform_stride_ = sizeof(LocalLightUniform);
}
} // namespace xray::render::vulkan
