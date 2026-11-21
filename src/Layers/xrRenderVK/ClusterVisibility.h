#pragma once

#include "BufferResource.h"
#include "FrameContext.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct ClusterCullGroup
{
    float bounds[4]{};
    float error{};
    float padding[3]{};
};
static_assert(sizeof(ClusterCullGroup) == 32);

struct ClusterCullRecord
{
    float bounds[4]{};
    int32_t refined_group{-1};
    int32_t owner_group{-1};
    uint32_t first_index{};
    uint32_t index_count{};
};
static_assert(sizeof(ClusterCullRecord) == 32);

struct ClusterCullSpan
{
    uint32_t group_first{};
    uint32_t group_count{};
    uint32_t cluster_first{};
    uint32_t cluster_count{};
    uint32_t command_first{};
    uint32_t first_index{};
    int32_t vertex_offset{};
};

// Generates camera and sun draw arguments on the GPU. Input ranges are packed
// once per level; each in-flight frame owns a distinct indirect output buffer.
class ClusterVisibility
{
public:
    ClusterVisibility() = default;
    ~ClusterVisibility() { destroy(); }
    ClusterVisibility(const ClusterVisibility&) = delete;
    ClusterVisibility& operator=(const ClusterVisibility&) = delete;
    ClusterVisibility(ClusterVisibility&& other) noexcept;
    ClusterVisibility& operator=(ClusterVisibility&& other) noexcept;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkPipelineCache cache,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& buffers,
        const std::vector<ClusterCullGroup>& groups, const std::vector<ClusterCullRecord>& clusters,
        const std::vector<ClusterCullSpan>& spans, uint32_t command_count,
        VkDeviceSize max_storage_buffer_range, std::string& error);
    bool record(const FrameRecordingContext& frame, const float (&camera_mvp)[16],
        const float (&sun_mvp)[16], const float (&camera)[3], float fov_degrees, uint32_t shadow_height,
        std::string& error, const float* previous_mvp = nullptr,
        VkImage previous_hiz_image = VK_NULL_HANDLE, VkImageView previous_hiz = VK_NULL_HANDLE,
        VkSampler hiz_sampler = VK_NULL_HANDLE, const FrameImageSharing* previous_hiz_sharing = nullptr,
        uint32_t previous_hiz_levels = 0, bool hiz_valid = false) const;
    // Rebuilds the shadow command segment for a local light or virtual page.
    // Call outside the shadow render pass before its indirect draw.
    bool record_shadow_view(const FrameRecordingContext& frame, const float (&shadow_mvp)[16],
        const float (&camera)[3], float fov_degrees, std::string& error) const;
    VkBuffer command_buffer(uint32_t frame_index) const
    { return frame_index < commands_.size() ? commands_[frame_index].handle() : VK_NULL_HANDLE; }
    VkDeviceSize command_offset(uint32_t view, uint32_t command) const
    { return (VkDeviceSize(view) * command_count_ + command) * sizeof(VkDrawIndexedIndirectCommand); }
    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }
    uint64_t bytes() const
    {
        uint64_t result = groups_.size() + clusters_.size();
        for (const BufferResource& output : commands_) result += output.size();
        return result;
    }
    void destroy();

private:
    struct PushConstants
    {
        float mvp[16]{};
        float camera_projection_scale[4]{};
        uint32_t ranges[4]{};
        uint32_t draw[4]{};
        uint32_t quality[4]{};
    };
    static_assert(sizeof(PushConstants) == 128);
    bool record_view(VkCommandBuffer command, VkDescriptorSet set, const float (&mvp)[16],
        const float (&camera)[3], float projection_scale, uint32_t output_view,
        std::string& error, bool hiz_valid = false) const;

    VkDevice device_{};
    BufferResource groups_, clusters_;
    std::array<BufferResource, FrameContext::FramesInFlight> commands_;
    mutable std::array<BufferResource, FrameContext::FramesInFlight> previous_mvp_;
    VkDescriptorSetLayout set_layout_{};
    VkDescriptorPool pool_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    PFN_vkDestroyDescriptorSetLayout destroy_set_layout_{};
    PFN_vkDestroyDescriptorPool destroy_pool_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkDestroyImage destroy_image_{};
    PFN_vkFreeMemory free_memory_{};
    PFN_vkDestroyImageView destroy_image_view_{};
    PFN_vkCmdClearColorImage clear_color_image_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkUpdateDescriptorSets update_sets_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdPipelineBarrier barrier_{};
    VkImage fallback_hiz_image_{};
    VkDeviceMemory fallback_hiz_memory_{};
    VkImageView fallback_hiz_view_{};
    VkSampler fallback_hiz_sampler_{};
    mutable bool fallback_hiz_initialized_{};
    uint32_t command_count_{};
    std::array<VkDescriptorSet, FrameContext::FramesInFlight> sets_{};
    std::vector<ClusterCullSpan> spans_;
    uint32_t group_count_{};
    uint32_t cluster_count_{};
};
} // namespace xray::render::vulkan
