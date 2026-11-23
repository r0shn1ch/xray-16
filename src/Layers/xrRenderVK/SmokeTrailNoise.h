#pragma once
#include "FrameSubmissionGraph.h"
#include "../xrRenderGraph/ResourceRetirement.h"
#include "GpuParticleCompute.h"
#include "GpuParticleUniforms.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace xray::render::vulkan
{
class FrameBufferPool;
class FrameImagePool;
// GPU-generated Perlin volume. With semaphore_handoff, the caller orders
// compute before graphics sampling and waits before reusing the frame slot.
class SmokeTrailNoise
{
public:
    using RequestSubmissionCommandBuffer = std::function<bool(FrameSubmissionGraph::Pass,
        FrameSubmissionGraph::Queue, VkCommandBuffer&, std::string&)>;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        uint32_t size, uint32_t frames, VkPipelineCache cache, const GpuParticleCompute::Source& source,
        std::string& error, uint32_t graphics_queue_family = UINT32_MAX,
        uint32_t compute_queue_family = UINT32_MAX, FrameImagePool* image_pool = nullptr);
    bool record(VkCommandBuffer command, uint32_t slot, float time, float tile_scale, bool first,
        std::string& error, VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
        bool semaphore_handoff = false, FrameSubmissionGraph* submission_graph = nullptr,
        FrameSubmissionGraph::Pass compute_submission_pass = FrameSubmissionGraph::InvalidPass,
        FrameSubmissionGraph::Pass graphics_continuation_pass = FrameSubmissionGraph::InvalidPass,
        uint64_t previous_graphics_submission = 0,
        FrameSubmissionGraph::Queue recording_queue = FrameSubmissionGraph::Queue::Compute,
        VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT,
        RequestSubmissionCommandBuffer request_submission_command_buffer = {},
        FrameBufferPool* buffer_pool = nullptr, framegraph::ResourceRetirement* retirement = nullptr);
    VkImageView view() const { return view_; }
    VkSampler sampler() const { return sampler_; }
    const BufferResource& buffer() const { return buffer_; }
    void destroy(); // Caller waits for GPU idle.
    ~SmokeTrailNoise() { destroy(); }
private:
    VkDevice device_{}; uint32_t size_{};
    FrameImagePool* image_pool_{};
    VkImage image_{}; VkDeviceMemory memory_{}; VkImageView view_{}; VkSampler sampler_{};
    VkSharingMode image_sharing_mode_ = VK_SHARING_MODE_EXCLUSIVE;
    std::vector<uint32_t> image_sharing_queue_families_;
    uint32_t graphics_queue_family_ = VK_QUEUE_FAMILY_IGNORED;
    uint32_t compute_queue_family_ = VK_QUEUE_FAMILY_IGNORED;
    BufferResource buffer_; StorageComputePass compute_;
    std::vector<std::unique_ptr<GpuParticleUniforms>> uniforms_;
    PFN_vkDestroyImage destroy_image_{}; PFN_vkFreeMemory free_memory_{};
    PFN_vkDestroyImageView destroy_view_{}; PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkCmdPipelineBarrier barrier_{}; PFN_vkCmdCopyBufferToImage copy_{};
};
}
