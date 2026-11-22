#include "FrameImagePool.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <utility>

#if defined(XRAY_HAS_NVRHI_VULKAN)
namespace
{
nvrhi::Format to_nvrhi_format(VkFormat format)
{
    for (uint32_t i = 1; i < static_cast<uint32_t>(nvrhi::Format::COUNT); ++i)
    {
        const auto candidate = static_cast<nvrhi::Format>(i);
        if (nvrhi::vulkan::convertFormat(candidate) == format)
            return candidate;
    }
    return nvrhi::Format::UNKNOWN;
}

}
#endif

namespace xray::render::vulkan
{
FrameImageSharing frame_image_sharing(uint32_t graphics_family, uint32_t compute_family)
{
    FrameImageSharing sharing;
    if (graphics_family == VK_QUEUE_FAMILY_IGNORED || compute_family == VK_QUEUE_FAMILY_IGNORED ||
        graphics_family == compute_family)
        return sharing;
    sharing.mode = VK_SHARING_MODE_CONCURRENT;
    sharing.queue_families = {graphics_family, compute_family};
    std::sort(sharing.queue_families.begin(), sharing.queue_families.end());
    return sharing;
}

FrameImagePool::Descriptor FrameImagePool::describe(const VkImageCreateInfo& info)
{
    Descriptor descriptor{info.flags, info.imageType, info.format, info.extent, info.mipLevels,
        info.arrayLayers, info.samples, info.tiling, info.usage, info.sharingMode, {}, info.initialLayout};
    if (info.sharingMode == VK_SHARING_MODE_CONCURRENT && info.pQueueFamilyIndices)
    {
        descriptor.sharing_queue_families.assign(info.pQueueFamilyIndices,
            info.pQueueFamilyIndices + info.queueFamilyIndexCount);
        std::sort(descriptor.sharing_queue_families.begin(), descriptor.sharing_queue_families.end());
    }
    return descriptor;
}

bool FrameImagePool::initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
    const FrameImagePoolDispatch& dispatch, nvrhi::vulkan::DeviceHandle nvrhi_device)
{
    reset();
    if (!device || !dispatch.create_image || !dispatch.destroy_image || !dispatch.get_requirements ||
        !dispatch.allocate_memory || !dispatch.free_memory || !dispatch.bind_memory)
        return false;
    device_ = device;
    memory_ = memory;
    vk_ = dispatch;
    nvrhi_device_ = std::move(nvrhi_device);
    return true;
}

bool FrameImagePool::acquire(const VkImageCreateInfo& info, VkImage& image, std::string& error)
{
    image = VK_NULL_HANDLE;
    const bool valid_shape =
        (info.imageType == VK_IMAGE_TYPE_1D && info.extent.height == 1 && info.extent.depth == 1) ||
        (info.imageType == VK_IMAGE_TYPE_2D && info.extent.depth == 1) ||
        (info.imageType == VK_IMAGE_TYPE_3D && info.extent.depth && info.arrayLayers == 1);
    if (!device_ || info.sType != VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO || info.pNext ||
        !valid_shape || !info.extent.width || !info.extent.height || !info.mipLevels || !info.arrayLayers ||
        (info.sharingMode == VK_SHARING_MODE_EXCLUSIVE &&
            (info.queueFamilyIndexCount != 0 || info.pQueueFamilyIndices != nullptr)) ||
        (info.sharingMode == VK_SHARING_MODE_CONCURRENT &&
            (info.queueFamilyIndexCount < 2 || !info.pQueueFamilyIndices)) ||
        (info.sharingMode != VK_SHARING_MODE_EXCLUSIVE && info.sharingMode != VK_SHARING_MODE_CONCURRENT))
    {
        error = "frame image pool needs a valid image shape and sharing description";
        return false;
    }
    if (info.sharingMode == VK_SHARING_MODE_CONCURRENT)
    {
        std::vector<uint32_t> families(info.pQueueFamilyIndices,
            info.pQueueFamilyIndices + info.queueFamilyIndexCount);
        std::sort(families.begin(), families.end());
        if (families.front() == VK_QUEUE_FAMILY_IGNORED ||
            std::adjacent_find(families.begin(), families.end()) != families.end())
        {
            error = "frame image pool needs distinct concurrent queue families";
            return false;
        }
    }
    const Descriptor descriptor = describe(info);
    constexpr VkImageUsageFlags reusable_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    Entry* reusable = nullptr;
    for (Entry& entry : entries_)
    {
        if (entry.in_use)
            continue;
        if (entry.descriptor == descriptor)
        {
            reusable = &entry;
            break;
        }
        if (!reusable && descriptor.initial_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
            !(entry.descriptor.usage & ~reusable_usage) &&
            (entry.descriptor.usage & descriptor.usage) == descriptor.usage &&
            entry.descriptor.same_layout(descriptor))
            reusable = &entry;
    }
    if (reusable)
    {
        reusable->in_use = true;
        reusable->last_used = frame_;
        image = reusable->image;
        error.clear();
        return true;
    }

    VkImage created{};
    if (vk_.create_image(device_, &info, nullptr, &created) != VK_SUCCESS)
    {
        error = "frame image pool could not create image";
        return false;
    }
    VkMemoryRequirements requirements{};
    vk_.get_requirements(device_, created, &requirements);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
        if ((requirements.memoryTypeBits & (1u << i)) &&
            (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            type = i;
            break;
        }
    if (type == UINT32_MAX)
    {
        vk_.destroy_image(device_, created, nullptr);
        error = "frame image pool has no device-local memory type";
        return false;
    }
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    VkDeviceMemory memory{};
    VkResult allocated = vk_.allocate_memory(device_, &allocation, nullptr, &memory);
    if (allocated == VK_ERROR_OUT_OF_DEVICE_MEMORY)
    {
        const auto before = entries_.size();
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(), [&](Entry& entry) {
            if (entry.in_use)
                return false;
            destroy(entry);
            return true;
        }), entries_.end());
        if (entries_.size() != before)
            allocated = vk_.allocate_memory(device_, &allocation, nullptr, &memory);
    }
    if (allocated != VK_SUCCESS)
    {
        vk_.destroy_image(device_, created, nullptr);
        error = "frame image pool could not allocate memory";
        return false;
    }
    if (vk_.bind_memory(device_, created, memory, 0) != VK_SUCCESS)
    {
        vk_.free_memory(device_, memory, nullptr);
        vk_.destroy_image(device_, created, nullptr);
        error = "frame image pool could not bind memory";
        return false;
    }
    nvrhi::TextureHandle texture;
#if defined(XRAY_HAS_NVRHI_VULKAN)
    // Wrap the pooled image after checked native allocation. The pool owns
    // the VkImage and its memory; NVRHI holds a non-owning resource handle.
    if (nvrhi_device_ && info.sharingMode == VK_SHARING_MODE_EXCLUSIVE &&
        info.flags == 0 && info.tiling == VK_IMAGE_TILING_OPTIMAL &&
        info.initialLayout == VK_IMAGE_LAYOUT_UNDEFINED && !(info.usage & ~reusable_usage))
    {
        const nvrhi::Format format = to_nvrhi_format(info.format);
        if (format != nvrhi::Format::UNKNOWN)
        {
            nvrhi::TextureDesc texture_desc{};
            texture_desc.width = info.extent.width;
            texture_desc.height = info.extent.height;
            texture_desc.depth = info.extent.depth;
            texture_desc.arraySize = info.arrayLayers;
            texture_desc.mipLevels = info.mipLevels;
            texture_desc.sampleCount = static_cast<uint32_t>(info.samples);
            texture_desc.format = format;
            using Dimension = nvrhi::TextureDimension;
            const bool array = info.arrayLayers > 1;
            if (info.imageType == VK_IMAGE_TYPE_3D)
                texture_desc.dimension = Dimension::Texture3D;
            else if (info.imageType == VK_IMAGE_TYPE_1D)
                texture_desc.dimension = array ? Dimension::Texture1DArray : Dimension::Texture1D;
            else if (info.samples != VK_SAMPLE_COUNT_1_BIT)
                texture_desc.dimension = array ? Dimension::Texture2DMSArray : Dimension::Texture2DMS;
            else
                texture_desc.dimension = array ? Dimension::Texture2DArray : Dimension::Texture2D;
            texture_desc.isShaderResource = (info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) != 0;
            texture_desc.isRenderTarget = (info.usage & (VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)) != 0;
            texture_desc.isUAV = (info.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
            texture_desc.debugName = "Vulkan frame image pool";

            uint64_t native_handle{};
            static_assert(sizeof(created) <= sizeof(native_handle));
            std::memcpy(&native_handle, &created, sizeof(created));
            try
            {
                texture = nvrhi_device_->createHandleForNativeTexture(nvrhi::ObjectTypes::VK_Image,
                    nvrhi::Object(native_handle), texture_desc);
            }
            catch (const std::exception&)
            {
                // Native Vulkan rendering can still use this image.
            }
        }
    }
#endif
    entries_.push_back({descriptor, created, memory, std::move(texture), frame_, true});
    image = created;
    error.clear();
    return true;
}

nvrhi::ITexture* FrameImagePool::nvrhi_texture(VkImage image) const
{
    for (const Entry& entry : entries_)
        if (entry.in_use && entry.image == image)
            return entry.nvrhi_texture.Get();
    return nullptr;
}

bool FrameImagePool::release(VkImage image)
{
    for (Entry& entry : entries_)
        if (entry.image == image && entry.in_use)
        {
            entry.in_use = false;
            entry.last_used = frame_;
            return true;
        }
    return false;
}

void FrameImagePool::destroy(Entry& entry)
{
    entry.nvrhi_texture = nullptr;
    if (entry.image)
        vk_.destroy_image(device_, entry.image, nullptr);
    if (entry.memory)
        vk_.free_memory(device_, entry.memory, nullptr);
}

void FrameImagePool::tick()
{
    ++frame_;
    for (size_t i = 0; i < entries_.size();)
        if (!entries_[i].in_use && frame_ - entries_[i].last_used >= EvictAfterFrames)
        {
            destroy(entries_[i]);
            entries_[i] = entries_.back();
            entries_.pop_back();
        }
        else
            ++i;
}

void FrameImagePool::reset()
{
    if (device_)
        for (Entry& entry : entries_)
            destroy(entry);
    entries_.clear();
    nvrhi_device_ = nullptr;
    device_ = VK_NULL_HANDLE;
    memory_ = {};
    vk_ = {};
    frame_ = 0;
}
}
