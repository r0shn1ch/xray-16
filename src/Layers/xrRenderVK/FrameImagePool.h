#pragma once

#include <vulkan/vulkan.h>
#include <nvrhi/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct FrameImagePoolDispatch
{
    PFN_vkCreateImage create_image{};
    PFN_vkDestroyImage destroy_image{};
    PFN_vkGetImageMemoryRequirements get_requirements{};
    PFN_vkAllocateMemory allocate_memory{};
    PFN_vkFreeMemory free_memory{};
    PFN_vkBindImageMemory bind_memory{};
};

// Resources shared by graphics and compute use concurrent sharing only when
// those roles belong to different Vulkan queue families. Queue indices within
// one family do not require an ownership transfer.
struct FrameImageSharing
{
    VkSharingMode mode = VK_SHARING_MODE_EXCLUSIVE;
    std::vector<uint32_t> queue_families;

    void apply(VkImageCreateInfo& info) const
    {
        info.sharingMode = mode;
        info.queueFamilyIndexCount = static_cast<uint32_t>(queue_families.size());
        info.pQueueFamilyIndices = queue_families.empty() ? nullptr : queue_families.data();
    }
};

FrameImageSharing frame_image_sharing(uint32_t graphics_family, uint32_t compute_family);

// Exact-match Vulkan image pool for renderer-owned offscreen targets.
// release() is called only after all GPU use has completed; views and framebuffers must be
// destroyed before releasing their image. In-use entries are never recycled.
class FrameImagePool
{
public:
    FrameImagePool() = default;
    ~FrameImagePool() { reset(); }
    FrameImagePool(const FrameImagePool&) = delete;
    FrameImagePool& operator=(const FrameImagePool&) = delete;

    bool initialize(VkDevice device, const VkPhysicalDeviceMemoryProperties& memory,
        const FrameImagePoolDispatch& dispatch, nvrhi::vulkan::DeviceHandle nvrhi_device = {});
    bool initialized() const { return device_ != VK_NULL_HANDLE; }
    bool acquire(const VkImageCreateInfo& info, VkImage& image, std::string& error);
    // Borrowed while the image remains acquired; some native images cannot be wrapped.
    nvrhi::ITexture* nvrhi_texture(VkImage image) const;
    bool release(VkImage image);
    void tick();
    void reset();

private:
    struct Descriptor
    {
        VkImageCreateFlags flags{};
        VkImageType type{};
        VkFormat format{};
        VkExtent3D extent{};
        uint32_t mips{}, layers{};
        VkSampleCountFlagBits samples{};
        VkImageTiling tiling{};
        VkImageUsageFlags usage{};
        VkSharingMode sharing{};
        std::vector<uint32_t> sharing_queue_families;
        VkImageLayout initial_layout{};
        bool operator==(const Descriptor& other) const
        {
            return usage == other.usage && same_layout(other);
        }
        bool same_layout(const Descriptor& other) const
        {
            return flags == other.flags && type == other.type && format == other.format &&
                extent.width == other.extent.width && extent.height == other.extent.height &&
                extent.depth == other.extent.depth && mips == other.mips && layers == other.layers &&
                samples == other.samples && tiling == other.tiling &&
                sharing == other.sharing && sharing_queue_families == other.sharing_queue_families &&
                initial_layout == other.initial_layout;
        }
    };
    struct Entry
    {
        Descriptor descriptor{};
        VkImage image{};
        VkDeviceMemory memory{};
        nvrhi::TextureHandle nvrhi_texture;
        uint64_t last_used{};
        bool in_use{};
    };
    static Descriptor describe(const VkImageCreateInfo& info);
    void destroy(Entry& entry);

    VkDevice device_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    FrameImagePoolDispatch vk_{};
    nvrhi::vulkan::DeviceHandle nvrhi_device_;
    std::vector<Entry> entries_;
    uint64_t frame_{};
    static constexpr uint64_t EvictAfterFrames = 4;
};
}
