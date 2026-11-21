#pragma once
#include "FrameSubmissionGraph.h"
#include "GpuParticleCompute.h"
#include "GpuParticleUniforms.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace xray::render::vulkan
{
// GPU-generated 4D Perlin volume, sampled as a 3D image by smoke trails. The
// image uses concurrent sharing when its graphics and compute families differ.
// With semaphore_handoff, the caller must order this producer before the later
// graphics sample and wait for a frame slot before reusing it.
class SmokeTrailNoise
{
public:
    using RequestSubmissionCommandBuffer = std::function<bool(FrameSubmissionGraph::Pass,
        FrameSubmissionGraph::Queue, VkCommandBuffer&, std::string&)>;
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        const VkPhysicalDeviceMemoryProperties& memory, const BufferResourceDispatch& dispatch,
        uint32_t size, uint32_t frames, VkPipelineCache cache, const GpuParticleCompute::Source& source,
        std::string& error, uint32_t graphics_queue_family = UINT32_MAX,
        uint32_t compute_queue_family = UINT32_MAX);
    bool record(VkCommandBuffer command, uint32_t slot, float time, float tile_scale, bool first,
        std::string& error, VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
        bool semaphore_handoff = false, FrameSubmissionGraph* submission_graph = nullptr,
        FrameSubmissionGraph::Pass compute_submission_pass = FrameSubmissionGraph::InvalidPass,
        FrameSubmissionGraph::Pass graphics_continuation_pass = FrameSubmissionGraph::InvalidPass,
        uint64_t previous_graphics_submission = 0,
        FrameSubmissionGraph::Queue recording_queue = FrameSubmissionGraph::Queue::Compute,
        VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT,
        RequestSubmissionCommandBuffer request_submission_command_buffer = {});
    VkImageView view() const { return view_; }
    VkSampler sampler() const { return sampler_; }
    VkSharingMode image_sharing_mode() const { return image_sharing_mode_; }
    const std::vector<uint32_t>& image_sharing_queue_families() const { return image_sharing_queue_families_; }
    const BufferResource& buffer() const { return buffer_; }
    void destroy(); // Caller waits for GPU idle.
    ~SmokeTrailNoise() { destroy(); }
private:
    VkDevice device_{}; uint32_t size_{};
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
