#pragma once

#include "DdsTexture.h"
#include "ImageStateTracker.h"

#include <nvrhi/vulkan.h>

#include <vector>

namespace xray::render::vulkan
{
struct TextureUploadDispatch
{
    PFN_vkCreateBuffer create_buffer{};
    PFN_vkDestroyBuffer destroy_buffer{};
    PFN_vkGetBufferMemoryRequirements get_buffer_memory_requirements{};
    PFN_vkCreateImage create_image{};
    PFN_vkDestroyImage destroy_image{};
    PFN_vkGetImageMemoryRequirements get_image_memory_requirements{};
    PFN_vkAllocateMemory allocate_memory{};
    PFN_vkFreeMemory free_memory{};
    PFN_vkBindBufferMemory bind_buffer_memory{};
    PFN_vkBindImageMemory bind_image_memory{};
    PFN_vkMapMemory map_memory{};
    PFN_vkUnmapMemory unmap_memory{};
    PFN_vkCreateImageView create_image_view{};
    PFN_vkDestroyImageView destroy_image_view{};
    PFN_vkAllocateCommandBuffers allocate_command_buffers{};
    PFN_vkFreeCommandBuffers free_command_buffers{};
    PFN_vkResetCommandBuffer reset_command_buffer{};
    PFN_vkBeginCommandBuffer begin_command_buffer{};
    PFN_vkEndCommandBuffer end_command_buffer{};
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier{};
    PFN_vkCmdCopyBufferToImage cmd_copy_buffer_to_image{};
    PFN_vkQueueSubmit queue_submit{};
    PFN_vkCreateFence create_fence{};
    PFN_vkDestroyFence destroy_fence{};
    PFN_vkGetFenceStatus get_fence_status{};
    PFN_vkResetFences reset_fences{};
    PFN_vkWaitForFences wait_for_fences{};
    nvrhi::IDevice* nvrhi_device{};
    // Some Adreno drivers fault when recording the post-copy memory barrier.
    // A completed upload fence provides the transfer-to-sampling dependency.
    bool fence_after_copy = false;
    // Owner must finish_uploads before submitting any draw using these images.
    bool defer_fence_wait = false;
};

struct UploadedTexture
{
    VkImage image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    nvrhi::TextureHandle nvrhi_texture;
    VkDeviceSize required_bytes = 0;
};

// Staging resources and command buffers stay alive until the GPU signals the
// submission fence. Keeping these in a queue avoids vkQueueWaitIdle per asset.
struct PendingTextureUpload
{
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    nvrhi::BufferHandle nvrhi_staging;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDeviceSize staging_bytes = 0;
};

// Reused by one persistent dynamic image (currently each OGM frame slot).
// The staging allocation remains mapped and the command buffer/fence are
// recycled only after the previous copy has completed.
struct PersistentTextureUpload
{
    VkBuffer staging = VK_NULL_HANDLE;
    VkDeviceMemory staging_memory = VK_NULL_HANDLE;
    nvrhi::BufferHandle nvrhi_staging;
    void* mapped = nullptr;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0;
    bool command_recorded = false;
    bool submitted = false;
};

bool upload_texture(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceMemoryProperties& memory_types, const TextureUploadDispatch& vk,
    const DdsTexture& source, UploadedTexture& result,
    std::vector<PendingTextureUpload>& pending_uploads, ImageStateTracker& image_states,
    std::string& error);
// Replaces the base level of a persistent, GENERAL-layout UI image. The caller
// must only update an image whose previous frame-slot fence has completed.
bool update_texture_pixels(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceMemoryProperties& memory_types, const TextureUploadDispatch& vk,
    VkImage image, uint32_t width, uint32_t height, const uint8_t* rgba,
    PersistentTextureUpload& upload, std::string& error);
void destroy_persistent_texture_upload(VkDevice device, VkCommandPool pool,
    const TextureUploadDispatch& vk, PersistentTextureUpload& upload);
void collect_completed_uploads(VkDevice device, VkCommandPool pool, const TextureUploadDispatch& vk,
    std::vector<PendingTextureUpload>& pending_uploads);
bool wait_for_uploads(VkDevice device, VkCommandPool pool, const TextureUploadDispatch& vk,
    std::vector<PendingTextureUpload>& pending_uploads);
void destroy_texture(VkDevice device, const TextureUploadDispatch& vk, UploadedTexture& texture);
}
