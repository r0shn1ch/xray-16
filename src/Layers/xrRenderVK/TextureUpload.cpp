#include "TextureUpload.h"
#include "Common/Platform.hpp"
#if defined(XR_PLATFORM_ANDROID)
// This translation unit does not include the engine precompiled header.
// Keep the crash breadcrumbs available without pulling in engine globals.
extern void android_set_load_context(const char* context);
extern void android_set_vulkan_stage(const char* stage);
#endif

#include <cstring>
#include <exception>
#include <limits>

namespace xray::render::vulkan
{
namespace
{
uint32_t memory_index(const VkPhysicalDeviceMemoryProperties& types, uint32_t allowed,
    VkMemoryPropertyFlags required)
{
    for (uint32_t index = 0; index < types.memoryTypeCount; ++index)
        if ((allowed & (1u << index)) && (types.memoryTypes[index].propertyFlags & required) == required)
            return index;
    return VK_MAX_MEMORY_TYPES;
}

#if defined(XRAY_HAS_NVRHI_VULKAN)
nvrhi::Format nvrhi_format(VkFormat format)
{
    for (uint32_t i = 1; i < static_cast<uint32_t>(nvrhi::Format::COUNT); ++i)
    {
        const auto candidate = static_cast<nvrhi::Format>(i);
        if (nvrhi::vulkan::convertFormat(candidate) == format)
            return candidate;
    }
    return nvrhi::Format::UNKNOWN;
}

VkImage native_image(nvrhi::ITexture* texture)
{
    if (!texture) return VK_NULL_HANDLE;
    const nvrhi::Object object = texture->getNativeObject(nvrhi::ObjectTypes::VK_Image);
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
    return static_cast<VkImage>(object.pointer);
#else
    return static_cast<VkImage>(object.integer);
#endif
}
#endif

VkBuffer native_buffer(nvrhi::IBuffer* buffer)
{
    if (!buffer) return VK_NULL_HANDLE;
    const nvrhi::Object object = buffer->getNativeObject(nvrhi::ObjectTypes::VK_Buffer);
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
    return static_cast<VkBuffer>(object.pointer);
#else
    return static_cast<VkBuffer>(object.integer);
#endif
}

bool nvrhi_buffer_is_host_coherent(VkBuffer buffer,
    VkDevice device, const VkPhysicalDeviceMemoryProperties& memory_types,
    const TextureUploadDispatch& vk)
{
    if (!buffer || !device || !vk.get_buffer_memory_requirements)
        return false;
    VkMemoryRequirements requirements{};
    vk.get_buffer_memory_requirements(device, buffer, &requirements);
    const uint32_t type = memory_index(memory_types, requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    return type != VK_MAX_MEMORY_TYPES &&
        (memory_types.memoryTypes[type].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
}

bool create_nvrhi_staging_buffer(nvrhi::IDevice* nvrhi_device, VkDevice device,
    const VkPhysicalDeviceMemoryProperties& memory_types, const TextureUploadDispatch& vk,
    VkDeviceSize size, const char* debug_name, nvrhi::BufferHandle& handle,
    VkBuffer& buffer, void*& mapped)
{
    handle = nullptr;
    buffer = VK_NULL_HANDLE;
    mapped = nullptr;
    if (!nvrhi_device || !device || !size)
        return false;

    try
    {
        nvrhi::BufferDesc desc{};
        desc.byteSize = size;
        desc.cpuAccess = nvrhi::CpuAccessMode::Write;
        desc.initialState = nvrhi::ResourceStates::CopySource;
        desc.keepInitialState = true;
        desc.debugName = debug_name;
        handle = nvrhi_device->createBuffer(desc);
        buffer = native_buffer(handle);
        if (!handle || !nvrhi_buffer_is_host_coherent(buffer, device, memory_types, vk))
        {
            handle = nullptr;
            buffer = VK_NULL_HANDLE;
            return false;
        }
        mapped = nvrhi_device->mapBuffer(handle, nvrhi::CpuAccessMode::Write);
        if (!mapped)
        {
            handle = nullptr;
            buffer = VK_NULL_HANDLE;
            return false;
        }
        return true;
    }
    catch (...)
    {
        if (handle && mapped)
            nvrhi_device->unmapBuffer(handle);
        handle = nullptr;
        buffer = VK_NULL_HANDLE;
        mapped = nullptr;
        return false;
    }
}

void release_upload(VkDevice device, VkCommandPool pool, const TextureUploadDispatch& vk,
    PendingTextureUpload& upload)
{
    if (upload.fence)
        vk.destroy_fence(device, upload.fence, nullptr);
    if (upload.command)
        vk.free_command_buffers(device, pool, 1, &upload.command);
    if (upload.nvrhi_staging)
        upload.nvrhi_staging = nullptr;
    else if (upload.staging)
        vk.destroy_buffer(device, upload.staging, nullptr);
    if (upload.staging_memory)
        vk.free_memory(device, upload.staging_memory, nullptr);
    upload = {};
}
}

void destroy_texture(VkDevice device, const TextureUploadDispatch& vk, UploadedTexture& texture)
{
    if (texture.view)
        vk.destroy_image_view(device, texture.view, nullptr);
    if (texture.nvrhi_texture)
    {
        texture.nvrhi_texture = nullptr;
    }
    else
    {
        if (texture.image)
            vk.destroy_image(device, texture.image, nullptr);
        if (texture.memory)
            vk.free_memory(device, texture.memory, nullptr);
    }
    texture = {};
}

bool upload_texture(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceMemoryProperties& memory_types, const TextureUploadDispatch& vk,
    const DdsTexture& source, UploadedTexture& result,
    std::vector<PendingTextureUpload>& pending_uploads, ImageStateTracker& image_states,
    std::string& error)
{
    collect_completed_uploads(device, pool, vk, pending_uploads);
    result = {};
    if (source.format == VK_FORMAT_UNDEFINED || source.pixels.empty() || source.copies.empty() || !source.mip_levels ||
        !source.extent.width || !source.extent.height || !source.extent.depth ||
        !source.layers || (source.cube && (source.layers < 6 || source.layers % 6 != 0)) ||
        (source.volume && (source.cube || source.layers != 1)) || (!source.volume && source.extent.depth != 1))
    {
        error = "texture data is empty";
        return false;
    }
    if (!vk.get_image_memory_requirements)
    {
        error = "Vulkan image memory requirements procedure is unavailable";
        return false;
    }

    // Each outstanding copy owns host-visible staging memory and a command
    // buffer. A level can request hundreds of DDS files before the GPU retires
    // them; bound both the number of submissions and their total staging size.
    constexpr size_t max_pending_uploads = 8;
    constexpr VkDeviceSize max_pending_bytes = 32u * 1024u * 1024u;
    VkDeviceSize pending_bytes = 0;
    for (const auto& entry : pending_uploads)
        pending_bytes += entry.staging_bytes;
    const VkDeviceSize next_bytes = source.pixels.size();
    while (!pending_uploads.empty() &&
        (pending_uploads.size() >= max_pending_uploads ||
            next_bytes > max_pending_bytes || pending_bytes > max_pending_bytes - next_bytes))
    {
#if defined(XR_PLATFORM_ANDROID)
        android_set_load_context("vulkan texture: waiting for staging memory");
#endif
        PendingTextureUpload& oldest = pending_uploads.front();
        const VkResult waited = vk.wait_for_fences(device, 1, &oldest.fence, VK_TRUE, UINT64_MAX);
        if (waited != VK_SUCCESS)
        {
            error = "Vulkan texture upload fence wait failed: " + std::to_string(waited);
            return false;
        }
        pending_bytes -= oldest.staging_bytes;
        release_upload(device, pool, vk, oldest);
        pending_uploads.erase(pending_uploads.begin());
    }

    PendingTextureUpload upload;
    upload.staging_bytes = next_bytes;
    auto cleanup = [&]
    {
        release_upload(device, pool, vk, upload);
    };
    auto fail = [&](const char* why)
    {
        error = why;
        cleanup();
        if (result.image)
            image_states.forget_image(result.image);
        destroy_texture(device, vk, result);
        return false;
    };

    void* mapped = nullptr;
    const bool use_nvrhi_staging = create_nvrhi_staging_buffer(vk.nvrhi_device, device, memory_types,
        vk, source.pixels.size(), "xrRenderVK DDS upload staging", upload.nvrhi_staging,
        upload.staging, mapped);
    if (use_nvrhi_staging)
    {
        std::memcpy(mapped, source.pixels.data(), source.pixels.size());
        vk.nvrhi_device->unmapBuffer(upload.nvrhi_staging);
    }
    else
    {
        VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        buffer_info.size = source.pixels.size();
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vk.create_buffer(device, &buffer_info, nullptr, &upload.staging) != VK_SUCCESS)
            return fail("Vulkan staging buffer creation failed");
        VkMemoryRequirements requirements{};
        vk.get_buffer_memory_requirements(device, upload.staging, &requirements);
        const uint32_t staging_type = memory_index(memory_types, requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (staging_type == VK_MAX_MEMORY_TYPES)
            return fail("Vulkan host-visible coherent memory is unavailable");
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = staging_type;
        if (vk.allocate_memory(device, &allocate, nullptr, &upload.staging_memory) != VK_SUCCESS ||
            vk.bind_buffer_memory(device, upload.staging, upload.staging_memory, 0) != VK_SUCCESS)
            return fail("Vulkan staging memory allocation failed");
        if (vk.map_memory(device, upload.staging_memory, 0, source.pixels.size(), 0, &mapped) != VK_SUCCESS)
            return fail("Vulkan staging memory mapping failed");
        std::memcpy(mapped, source.pixels.data(), source.pixels.size());
        vk.unmap_memory(device, upload.staging_memory);
    }

    VkImageCreateInfo image_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image_info.imageType = source.volume ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    image_info.format = source.format;
    image_info.extent = source.extent;
    image_info.mipLevels = source.mip_levels;
    image_info.arrayLayers = source.volume ? 1 : source.layers;
    image_info.flags = source.cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
#if defined(XRAY_HAS_NVRHI_VULKAN)
    const nvrhi::Format format = nvrhi_format(source.format);
    if (vk.nvrhi_device && format != nvrhi::Format::UNKNOWN)
    {
        nvrhi::TextureDesc texture_desc{};
        texture_desc.width = source.extent.width;
        texture_desc.height = source.extent.height;
        texture_desc.depth = source.extent.depth;
        texture_desc.arraySize = source.volume ? 1 : source.layers;
        texture_desc.mipLevels = source.mip_levels;
        texture_desc.sampleCount = 1;
        texture_desc.format = format;
        texture_desc.dimension = source.volume ? nvrhi::TextureDimension::Texture3D :
            source.cube ? (source.layers > 6 ? nvrhi::TextureDimension::TextureCubeArray :
            nvrhi::TextureDimension::TextureCube) :
            source.layers > 1 ? nvrhi::TextureDimension::Texture2DArray : nvrhi::TextureDimension::Texture2D;
        texture_desc.isShaderResource = true;
        texture_desc.debugName = "xrRenderVK game DDS";
        try
        {
            result.nvrhi_texture = vk.nvrhi_device->createTexture(texture_desc);
            if (result.nvrhi_texture)
                result.image = native_image(result.nvrhi_texture);
            if (!result.image)
                result.nvrhi_texture = nullptr;
        }
        catch (const std::exception&)
        {
            result.nvrhi_texture = nullptr;
        }
        catch (...)
        {
            result.nvrhi_texture = nullptr;
        }
    }
#endif
    if (!result.image && vk.create_image(device, &image_info, nullptr, &result.image) != VK_SUCCESS)
        return fail("Vulkan texture image creation failed");
    VkMemoryRequirements requirements{};
    vk.get_image_memory_requirements(device, result.image, &requirements);
    result.required_bytes = requirements.size;
    VkImageSubresourceRange image_range{};
    image_range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    image_range.levelCount = source.mip_levels;
    image_range.layerCount = source.layers;
    if (!image_states.register_image(result.image, image_range, ImageUse::Undefined, error))
    {
        cleanup();
        destroy_texture(device, vk, result);
        return false;
    }
    if (!result.nvrhi_texture)
    {
        VkMemoryAllocateInfo allocate{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        uint32_t image_type = memory_index(memory_types, requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (image_type == VK_MAX_MEMORY_TYPES)
            image_type = memory_index(memory_types, requirements.memoryTypeBits, 0);
        if (image_type == VK_MAX_MEMORY_TYPES)
            return fail("Vulkan texture memory type is unavailable");
        allocate.allocationSize = requirements.size;
        allocate.memoryTypeIndex = image_type;
        if (vk.allocate_memory(device, &allocate, nullptr, &result.memory) != VK_SUCCESS ||
            vk.bind_image_memory(device, result.image, result.memory, 0) != VK_SUCCESS)
            return fail("Vulkan texture image memory allocation failed");
    }

    VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = pool;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    if (vk.allocate_command_buffers(device, &command_info, &upload.command) != VK_SUCCESS)
        return fail("Vulkan upload command allocation failed");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vk.begin_command_buffer(upload.command, &begin) != VK_SUCCESS)
        return fail("Vulkan upload command recording failed");
    const ImageStateDispatch image_state_dispatch{vk.cmd_pipeline_barrier};
#if defined(XR_PLATFORM_ANDROID)
    android_set_load_context("vulkan texture: undefined -> general for transfer");
#endif
    if (!image_states.transition(upload.command, result.image, ImageUse::TransferDestinationGeneral,
            image_state_dispatch, error))
    {
        cleanup();
        image_states.forget_image(result.image);
        destroy_texture(device, vk, result);
        return false;
    }
    vk.cmd_copy_buffer_to_image(upload.command, upload.staging, result.image, VK_IMAGE_LAYOUT_GENERAL,
        static_cast<uint32_t>(source.copies.size()), source.copies.data());
#if defined(XR_PLATFORM_ANDROID)
    android_set_vulkan_stage("texture transfer write to shader read");
    android_set_load_context("vulkan texture: transfer write -> shader read in general layout");
#endif
    // Keep immutable game textures in GENERAL. The affected Adreno driver
    // faults here even for a memory-only barrier. Complete the copy fences
    // before any draw samples the image: either here or at the owner's frame
    // boundary. This makes writes visible to later submissions. Other drivers
    // retain the asynchronous upload and explicit memory barrier.
    if (!vk.fence_after_copy)
    {
        VkMemoryBarrier ready{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        // Immutable textures also feed compute passes, including material resolve.
        vk.cmd_pipeline_barrier(upload.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &ready, 0, nullptr, 0, nullptr);
    }
    const VkResult ended = vk.end_command_buffer(upload.command);
    if (ended != VK_SUCCESS)
    {
        const std::string reason = "Vulkan upload command finalization failed: " + std::to_string(ended);
        return fail(reason.c_str());
    }

    VkImageViewCreateInfo view_info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view_info.image = result.image;
    view_info.viewType = source.volume ? VK_IMAGE_VIEW_TYPE_3D :
        source.cube ? (source.layers > 6 ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE) :
        source.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = source.format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = image_info.mipLevels;
    view_info.subresourceRange.layerCount = source.layers;
    if (vk.create_image_view(device, &view_info, nullptr, &result.view) != VK_SUCCESS)
        return fail("Vulkan texture view creation failed");

    VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vk.create_fence(device, &fence_info, nullptr, &upload.fence) != VK_SUCCESS)
        return fail("Vulkan upload fence creation failed");
    try
    {
        if (pending_uploads.size() == pending_uploads.max_size())
            return fail("Vulkan pending upload queue is full");
        if (pending_uploads.size() == pending_uploads.capacity())
        {
            const size_t capacity = pending_uploads.capacity();
            const size_t next = capacity > pending_uploads.max_size() / 2 ?
                capacity + 1 : capacity ? capacity * 2 : 1;
            pending_uploads.reserve(next);
        }
    }
    catch (...)
    {
        return fail("Vulkan pending upload queue allocation failed");
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &upload.command;
#if defined(XR_PLATFORM_ANDROID)
    android_set_vulkan_stage("texture upload queue submit");
    android_set_load_context("vulkan texture: queue submit");
#endif
    if (vk.queue_submit(queue, 1, &submit, upload.fence) != VK_SUCCESS)
        return fail("Vulkan texture upload submission failed");
#if defined(XR_PLATFORM_ANDROID)
    android_set_vulkan_stage(nullptr);
    android_set_load_context(nullptr);
#endif
    if (vk.fence_after_copy && !vk.defer_fence_wait)
    {
#if defined(XR_PLATFORM_ANDROID)
        android_set_vulkan_stage("texture upload fence wait");
        android_set_load_context("vulkan texture: waiting for copy fence");
#endif
        const VkResult waited = vk.wait_for_fences(device, 1, &upload.fence, VK_TRUE, UINT64_MAX);
        if (waited != VK_SUCCESS)
        {
            error = "Vulkan texture copy fence wait failed: " + std::to_string(waited);
            // A device-lost fence is terminal; otherwise retain the submitted
            // staging resources until the device is torn down.
            if (waited == VK_ERROR_DEVICE_LOST)
                cleanup();
            else
            {
                pending_uploads.push_back(upload);
                upload = {};
            }
            image_states.forget_image(result.image);
            if (waited == VK_ERROR_DEVICE_LOST)
                destroy_texture(device, vk, result);
            return false;
        }
#if defined(XR_PLATFORM_ANDROID)
        android_set_vulkan_stage(nullptr);
        android_set_load_context(nullptr);
#endif
        release_upload(device, pool, vk, upload);
        error.clear();
        return true;
    }
    pending_uploads.push_back(upload);
    upload = {};
    error.clear();
    return true;
}

bool update_texture_pixels(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceMemoryProperties& memory_types, const TextureUploadDispatch& vk,
    VkImage image, uint32_t width, uint32_t height, const uint8_t* rgba,
    PersistentTextureUpload& upload, std::string& error)
{
    if (!device || !queue || !pool || !image || !rgba || !width || !height ||
        width > std::numeric_limits<size_t>::max() / 4 / height ||
        !vk.create_buffer || !vk.destroy_buffer || !vk.get_buffer_memory_requirements ||
        !vk.allocate_memory || !vk.free_memory ||
        !vk.bind_buffer_memory || !vk.map_memory || !vk.unmap_memory ||
        !vk.allocate_command_buffers || !vk.free_command_buffers || !vk.reset_command_buffer ||
        !vk.begin_command_buffer || !vk.end_command_buffer ||
        !vk.cmd_pipeline_barrier || !vk.cmd_copy_buffer_to_image || !vk.create_fence ||
        !vk.destroy_fence || !vk.get_fence_status || !vk.reset_fences || !vk.queue_submit ||
        !vk.wait_for_fences)
    {
        error = "invalid persistent Vulkan texture update";
        return false;
    }
    const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
    constexpr VkDeviceSize max_staging_bytes = 32u * 1024u * 1024u;
    if (bytes > max_staging_bytes)
    {
        error = "persistent Vulkan texture update exceeds the per-image staging budget";
        return false;
    }

    const auto wait_for_previous = [&]() -> bool
    {
        if (!upload.submitted)
            return true;
        VkResult completed = vk.get_fence_status(device, upload.fence);
        if (completed == VK_NOT_READY)
            completed = vk.wait_for_fences(device, 1, &upload.fence, VK_TRUE, UINT64_MAX);
        if (completed != VK_SUCCESS)
        {
            error = "persistent Vulkan texture update fence wait failed: " +
                std::to_string(completed);
            return false;
        }
        upload.submitted = false;
        return true;
    };

    const bool complete = upload.staging && (upload.staging_memory || upload.nvrhi_staging) &&
        upload.mapped && upload.command &&
        upload.fence && upload.capacity >= bytes;
    if (!complete)
    {
        if (!wait_for_previous())
            return false;
        destroy_persistent_texture_upload(device, pool, vk, upload);

        const bool use_nvrhi_staging = create_nvrhi_staging_buffer(vk.nvrhi_device, device, memory_types,
            vk, bytes, "xrRenderVK OGM frame staging", upload.nvrhi_staging,
            upload.staging, upload.mapped);
        if (!use_nvrhi_staging)
        {
            VkBufferCreateInfo buffer_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            buffer_info.size = bytes;
            buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vk.create_buffer(device, &buffer_info, nullptr, &upload.staging) != VK_SUCCESS)
            {
                error = "persistent Vulkan texture staging buffer creation failed";
                destroy_persistent_texture_upload(device, pool, vk, upload);
                return false;
            }
            VkMemoryRequirements requirements{};
            vk.get_buffer_memory_requirements(device, upload.staging, &requirements);
            const uint32_t staging_type = memory_index(memory_types, requirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            if (staging_type == VK_MAX_MEMORY_TYPES)
            {
                error = "persistent Vulkan texture update needs host-visible coherent memory";
                destroy_persistent_texture_upload(device, pool, vk, upload);
                return false;
            }
            VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            allocation.allocationSize = requirements.size;
            allocation.memoryTypeIndex = staging_type;
            if (vk.allocate_memory(device, &allocation, nullptr, &upload.staging_memory) != VK_SUCCESS ||
                vk.bind_buffer_memory(device, upload.staging, upload.staging_memory, 0) != VK_SUCCESS)
            {
                error = "persistent Vulkan texture staging allocation failed";
                destroy_persistent_texture_upload(device, pool, vk, upload);
                return false;
            }
            if (vk.map_memory(device, upload.staging_memory, 0, requirements.size, 0, &upload.mapped) != VK_SUCCESS)
            {
                error = "persistent Vulkan texture staging map failed";
                upload.mapped = nullptr;
                destroy_persistent_texture_upload(device, pool, vk, upload);
                return false;
            }
        }

        VkCommandBufferAllocateInfo command_info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        command_info.commandPool = pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        if (vk.allocate_command_buffers(device, &command_info, &upload.command) != VK_SUCCESS)
        {
            error = "persistent Vulkan texture update command allocation failed";
            destroy_persistent_texture_upload(device, pool, vk, upload);
            return false;
        }
        VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vk.create_fence(device, &fence_info, nullptr, &upload.fence) != VK_SUCCESS)
        {
            error = "persistent Vulkan texture update fence creation failed";
            destroy_persistent_texture_upload(device, pool, vk, upload);
            return false;
        }
        upload.capacity = bytes;
    }

    if (!wait_for_previous())
        return false;
    if (upload.command_recorded)
    {
        const VkResult reset = vk.reset_command_buffer(upload.command, 0);
        if (reset != VK_SUCCESS)
        {
            error = "persistent Vulkan texture update command reset failed: " + std::to_string(reset);
            return false;
        }
        upload.command_recorded = false;
    }
    std::memcpy(upload.mapped, rgba, static_cast<size_t>(bytes));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    upload.command_recorded = true;
    const VkResult began = vk.begin_command_buffer(upload.command, &begin);
    if (began != VK_SUCCESS)
    {
        error = "persistent Vulkan texture update command recording failed: " + std::to_string(began);
        return false;
    }

    VkImageMemoryBarrier to_transfer{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    to_transfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    to_transfer.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_transfer.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    to_transfer.srcQueueFamilyIndex = to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_transfer.image = image;
    to_transfer.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk.cmd_pipeline_barrier(upload.command, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_transfer);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {width, height, 1};
    vk.cmd_copy_buffer_to_image(upload.command, upload.staging, image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    if (!vk.fence_after_copy)
    {
        VkImageMemoryBarrier to_sample{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        to_sample.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_sample.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_sample.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_sample.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        to_sample.srcQueueFamilyIndex = to_sample.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_sample.image = image;
        to_sample.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vk.cmd_pipeline_barrier(upload.command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &to_sample);
    }
    const VkResult ended = vk.end_command_buffer(upload.command);
    if (ended != VK_SUCCESS)
    {
        error = "persistent Vulkan texture update command finalization failed: " + std::to_string(ended);
        return false;
    }
    if (vk.reset_fences(device, 1, &upload.fence) != VK_SUCCESS)
    {
        error = "persistent Vulkan texture update fence reset failed";
        return false;
    }
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &upload.command;
    const VkResult submitted = vk.queue_submit(queue, 1, &submit, upload.fence);
    if (submitted != VK_SUCCESS)
    {
        error = "persistent Vulkan texture update submission failed: " + std::to_string(submitted);
        return false;
    }
    upload.submitted = true;
    if (vk.fence_after_copy)
    {
        const VkResult waited = vk.wait_for_fences(device, 1, &upload.fence, VK_TRUE, UINT64_MAX);
        if (waited != VK_SUCCESS)
        {
            error = "persistent Vulkan texture copy fence wait failed: " + std::to_string(waited);
            return false;
        }
        upload.submitted = false;
    }
    error.clear();
    return true;
}

void destroy_persistent_texture_upload(VkDevice device, VkCommandPool pool,
    const TextureUploadDispatch& vk, PersistentTextureUpload& upload)
{
    const bool nvrhi_owned_staging = bool(upload.nvrhi_staging);
    if (upload.nvrhi_staging)
    {
        if (upload.mapped && vk.nvrhi_device)
            vk.nvrhi_device->unmapBuffer(upload.nvrhi_staging);
        upload.nvrhi_staging = nullptr;
    }
    else if (upload.mapped && upload.staging_memory && vk.unmap_memory)
        vk.unmap_memory(device, upload.staging_memory);
    if (upload.command && vk.free_command_buffers)
        vk.free_command_buffers(device, pool, 1, &upload.command);
    if (upload.fence && vk.destroy_fence)
        vk.destroy_fence(device, upload.fence, nullptr);
    if (!nvrhi_owned_staging && upload.staging && vk.destroy_buffer)
        vk.destroy_buffer(device, upload.staging, nullptr);
    if (upload.staging_memory && vk.free_memory)
        vk.free_memory(device, upload.staging_memory, nullptr);
    upload = {};
}

void collect_completed_uploads(VkDevice device, VkCommandPool pool, const TextureUploadDispatch& vk,
    std::vector<PendingTextureUpload>& pending_uploads)
{
    auto it = pending_uploads.begin();
    while (it != pending_uploads.end())
    {
        if (vk.get_fence_status(device, it->fence) == VK_SUCCESS)
        {
            release_upload(device, pool, vk, *it);
            it = pending_uploads.erase(it);
        }
        else
            ++it;
    }
}

bool wait_for_uploads(VkDevice device, VkCommandPool pool, const TextureUploadDispatch& vk,
    std::vector<PendingTextureUpload>& pending_uploads)
{
    if (pending_uploads.empty()) return true;
    std::vector<VkFence> fences;
    fences.reserve(pending_uploads.size());
    for (const auto& upload : pending_uploads) fences.push_back(upload.fence);
    const VkResult waited = vk.wait_for_fences(device, static_cast<uint32_t>(fences.size()),
        fences.data(), VK_TRUE, UINT64_MAX);
    const bool device_lost = waited == VK_ERROR_DEVICE_LOST;
    if (waited != VK_SUCCESS && !device_lost) return false;
    for (auto& upload : pending_uploads)
        release_upload(device, pool, vk, upload);
    pending_uploads.clear();
    return !device_lost;
}
}
