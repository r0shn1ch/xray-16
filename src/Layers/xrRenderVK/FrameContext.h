#pragma once

#include "FrameSubmissionGraph.h"
#include "ImageStateTracker.h"
#include "FrameImagePool.h"

#include <vulkan/vulkan.h>

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct FrameDispatch
{
    PFN_vkGetPhysicalDeviceSurfaceCapabilitiesKHR get_surface_capabilities{};
    PFN_vkGetPhysicalDeviceSurfaceFormatsKHR get_surface_formats{};
    PFN_vkCreateSwapchainKHR create_swapchain{};
    PFN_vkDestroySwapchainKHR destroy_swapchain{};
    PFN_vkGetSwapchainImagesKHR get_swapchain_images{};
    PFN_vkAcquireNextImageKHR acquire_next_image{};
    PFN_vkQueuePresentKHR queue_present{};
    PFN_vkCreateImageView create_image_view{};
    PFN_vkDestroyImageView destroy_image_view{};
    PFN_vkGetPhysicalDeviceFormatProperties get_format_properties{};
    PFN_vkGetPhysicalDeviceMemoryProperties get_memory_properties{};
    PFN_vkCreateImage create_image{};
    PFN_vkDestroyImage destroy_image{};
    PFN_vkGetImageMemoryRequirements get_image_memory_requirements{};
    PFN_vkAllocateMemory allocate_memory{};
    PFN_vkFreeMemory free_memory{};
    PFN_vkBindImageMemory bind_image_memory{};
    PFN_vkCreateRenderPass create_render_pass{};
    PFN_vkDestroyRenderPass destroy_render_pass{};
    PFN_vkCreateFramebuffer create_framebuffer{};
    PFN_vkDestroyFramebuffer destroy_framebuffer{};
    PFN_vkCreateCommandPool create_command_pool{};
    PFN_vkDestroyCommandPool destroy_command_pool{};
    PFN_vkAllocateCommandBuffers allocate_command_buffers{};
    PFN_vkFreeCommandBuffers free_command_buffers{};
    PFN_vkResetCommandBuffer reset_command_buffer{};
    PFN_vkBeginCommandBuffer begin_command_buffer{};
    PFN_vkCmdBeginRenderPass cmd_begin_render_pass{};
    PFN_vkCmdEndRenderPass cmd_end_render_pass{};
    PFN_vkCmdClearAttachments cmd_clear_attachments{};
    PFN_vkCmdCopyImageToBuffer cmd_copy_image_to_buffer{};
    PFN_vkCmdCopyImage cmd_copy_image{};
    PFN_vkCmdBlitImage cmd_blit_image{};
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier{};
    PFN_vkEndCommandBuffer end_command_buffer{};
    PFN_vkCreateSemaphore create_semaphore{};
    PFN_vkDestroySemaphore destroy_semaphore{};
    PFN_vkCreateFence create_fence{};
    PFN_vkDestroyFence destroy_fence{};
    PFN_vkWaitForFences wait_for_fences{};
    PFN_vkResetFences reset_fences{};
    PFN_vkQueueSubmit queue_submit{};
    PFN_vkDeviceWaitIdle device_wait_idle{};
    // Timestamp measurement is optional on queues without timestamp support.
    PFN_vkCreateQueryPool create_query_pool{};
    PFN_vkDestroyQueryPool destroy_query_pool{};
    PFN_vkCmdResetQueryPool cmd_reset_query_pool{};
    PFN_vkCmdWriteTimestamp cmd_write_timestamp{};
    PFN_vkGetQueryPoolResults get_query_pool_results{};
    PFN_vkSetHdrMetadataEXT set_hdr_metadata{};
};

// Optional replacement-swapchain provider used by frame interpolation. The
// callback may replace the create/destroy/enumerate/acquire/present functions
// in FrameDispatch and return a swapchain owned by that provider.
struct FrameSwapchainCreateContext
{
    VkInstance instance = VK_NULL_HANDLE;
    PFN_vkGetInstanceProcAddr get_instance_proc{};
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    PFN_vkGetDeviceProcAddr get_device_proc{};
    VkQueue game_queue = VK_NULL_HANDLE;
    uint32_t game_queue_family = UINT32_MAX;
    VkQueueFlags game_queue_flags{};
    VkQueue present_queue = VK_NULL_HANDLE;
    uint32_t present_queue_family = UINT32_MAX;
    VkQueue compute_queue = VK_NULL_HANDLE;
    uint32_t compute_queue_family = UINT32_MAX;
    VkQueue image_acquire_queue = VK_NULL_HANDLE;
    uint32_t image_acquire_queue_family = UINT32_MAX;
    uint32_t api_version = VK_API_VERSION_1_0;
    VkExtent2D scene_extent{};
};
using FrameSwapchainCreate = bool (*)(VkSwapchainCreateInfoKHR& create_info,
    const FrameSwapchainCreateContext& context, FrameDispatch& dispatch,
    VkSwapchainKHR& swapchain, bool& frame_interpolation, void* user_data,
    std::string& error);
using FrameSwapchainDestroyed = void (*)(void* user_data);
using FrameGenerationUiActive = bool (*)(void* user_data);
using FrameGenerationUiRegister = bool (*)(VkImage image, VkFormat format, VkExtent2D extent,
    VkImageUsageFlags usage, void* user_data, std::string& error);

enum class FrameStatus
{
    Presented,
    RecreateRequired
};

bool mark_surface_lost(VkResult result, bool& surface_lost, FrameStatus& status);

struct FrameRecordingContext
{
    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    VkExtent2D extent{};
    uint32_t image_index = 0;
    uint32_t frame_index = 0;
    uint32_t mip_level = 0;
    VkOffset2D viewport_offset{};
    // Set only for an asynchronous compute prepass. Cross-family buffers use
    // concurrent Vulkan sharing and the frame semaphore as the memory dependency.
    VkCommandBuffer compute_command_buffer = VK_NULL_HANDLE;
    bool asynchronous_copy_available = false;
    VkQueueFlags command_queue_flags = VK_QUEUE_GRAPHICS_BIT;
    VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT;
    VkQueueFlags compute_queue_flags = VK_QUEUE_COMPUTE_BIT;
    VkQueueFlags copy_queue_flags = VK_QUEUE_TRANSFER_BIT;
    bool semaphore_handoff = false;
    FrameSubmissionGraph* submission_graph = nullptr;
    FrameSubmissionGraph::Pass compute_submission_pass = FrameSubmissionGraph::InvalidPass;
    FrameSubmissionGraph::Pass graphics_prepass_pass = FrameSubmissionGraph::InvalidPass;
    FrameSubmissionGraph::Pass graphics_continuation_pass = FrameSubmissionGraph::InvalidPass;
    // Last graphics producer that may still own persistent data imported by the
    // current frame graph. Zero means the resource has no submitted predecessor.
    uint64_t previous_graphics_submission{};
    uint32_t graphics_queue_family = VK_QUEUE_FAMILY_IGNORED;
    uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
    uint32_t copy_queue_family = VK_QUEUE_FAMILY_IGNORED;
    FrameSubmissionGraph::Queue command_queue = FrameSubmissionGraph::Queue::Graphics;
    std::function<bool(FrameSubmissionGraph::Pass, FrameSubmissionGraph::Queue,
        VkCommandBuffer&, std::string&)> request_submission_command_buffer;
    // Generated producer for current-frame motion vectors and object velocity.
    // Consumers such as Hi-Z and RTGI must wait for this pass before sampling.
    FrameSubmissionGraph::Pass geometry_motion_submission_pass = FrameSubmissionGraph::InvalidPass;
    // Ray visibility is recorded into the graphics prepass command list.
    // Consumers must not substitute the later RTGI compute submission here.
    FrameSubmissionGraph::Pass* ray_visibility_submission_pass = nullptr;
    // Vulkan barrier dispatch used by frame-local image graphs recorded from
    // this context.
    PFN_vkCmdPipelineBarrier cmd_pipeline_barrier{};
};

// Rebinds a frame context to a command buffer recorded on a specific queue.
// Queue capabilities, family indices, cross-queue state and previous-frame
// producer information survive the transition; a command buffer from an earlier
// stage and its submission-recording callback do not.
inline FrameRecordingContext make_frame_recording_context(const FrameRecordingContext& source,
    VkCommandBuffer command_buffer, VkRenderPass render_pass, VkFramebuffer framebuffer,
    VkExtent2D extent, uint32_t mip_level, VkOffset2D viewport_offset,
    FrameSubmissionGraph::Queue queue)
{
    FrameRecordingContext result = source;
    result.command_buffer = command_buffer;
    result.render_pass = render_pass;
    result.framebuffer = framebuffer;
    result.extent = extent;
    result.mip_level = mip_level;
    result.viewport_offset = viewport_offset;
    result.compute_command_buffer = VK_NULL_HANDLE;
    result.command_queue = queue;
    switch (queue)
    {
    case FrameSubmissionGraph::Queue::Graphics: result.command_queue_flags = result.graphics_queue_flags; break;
    case FrameSubmissionGraph::Queue::Compute: result.command_queue_flags = result.compute_queue_flags; break;
    case FrameSubmissionGraph::Queue::Copy: result.command_queue_flags = result.copy_queue_flags; break;
    }
    result.submission_graph = nullptr;
    result.compute_submission_pass = FrameSubmissionGraph::InvalidPass;
    result.graphics_prepass_pass = FrameSubmissionGraph::InvalidPass;
    result.graphics_continuation_pass = FrameSubmissionGraph::InvalidPass;
    result.request_submission_command_buffer = {};
    result.geometry_motion_submission_pass = FrameSubmissionGraph::InvalidPass;
    result.ray_visibility_submission_pass = nullptr;
    return result;
}

using FrameRecorder = void (*)(const FrameRecordingContext& frame, void* user_data);
using FrameCheckedRecorder = bool (*)(const FrameRecordingContext& frame, void* user_data,
    std::string& error);
using FrameHistoryValid = bool (*)(void* user_data);
using FrameOutputSlot = uint32_t (*)(void* user_data);
// Invoked once for a submitted frame after its fence signals, before its
// per-frame resources are reused. The callback may read coherent GPU output.
using FrameCompleted = bool (*)(uint32_t frame_index, uint32_t image_index, void* user_data,
    std::string& error);
using FrameSubmitted = void (*)(uint32_t frame_index, uint32_t image_index, void* user_data);
using FrameReadbackRecorder = void (*)(VkCommandBuffer command, VkImage image, VkExtent2D extent, void* user_data);
// Records offscreen work before the swapchain render pass. render_pass and
// framebuffer are null; the callback owns any pass it begins and ends.
using FramePrepassRecorder = void (*)(const FrameRecordingContext& frame, void* user_data);
using FrameInterpassRecorder = bool (*)(VkCommandBuffer command, VkImage image,
    uint32_t image_index, void* user_data, std::string& error);
using FrameRecordValidation = bool (*)(void* user_data, std::string& error);

struct FrameSampledImage
{
    VkImage image = VK_NULL_HANDLE;
    VkImageSubresourceRange range{};
    ImageUse use = ImageUse::Undefined;
    FrameImageSharing sharing{};
    FrameSubmissionGraph::Pass producer_pass = FrameSubmissionGraph::InvalidPass;
    // Terminal image state written by producer_pass.
    ImageUse producer_use = ImageUse::Undefined;
};
using FrameSampledImageSink = std::function<bool(const FrameSampledImage&, std::string&)>;
using FrameSampledImageProvider = bool (*)(uint32_t image_index,
    const FrameSampledImageSink& sink, void* user_data, std::string& error);

bool load_frame_dispatch(VkInstance instance, PFN_vkGetInstanceProcAddr get_instance_proc,
    VkDevice device, PFN_vkGetDeviceProcAddr get_device_proc, FrameDispatch& dispatch, std::string& error);

class FrameContext
{
public:
    static constexpr uint32_t FramesInFlight = 2;

    FrameContext() = default;
    ~FrameContext();
    FrameContext(const FrameContext&) = delete;
    FrameContext& operator=(const FrameContext&) = delete;

    // A nonzero scene_extent allocates scene color separately from the swapchain.
    // Scene/interpass callbacks retain that size; the compositor and readback use
    // the surface size. Requires postprocessing to write the presentation image.
    bool initialize(VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface,
        VkQueue queue, uint32_t queue_family, VkExtent2D requested_extent,
        const FrameDispatch& dispatch, std::string& error, bool allow_readback = false,
        bool use_depth = false, bool preserve_prepass_depth = false, bool postprocess = false,
        float timestamp_period_ns = 0.f, uint32_t timestamp_valid_bits = 0,
        VkExtent2D scene_extent = {},
        nvrhi::vulkan::DeviceHandle nvrhi_device = {}, bool async_compute_queue = false,
        VkQueueFlags graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT,
        VkQueueFlags compute_queue_flags = VK_QUEUE_COMPUTE_BIT,
        const FrameSwapchainCreateContext* swapchain_context = nullptr,
        FrameSwapchainCreate swapchain_create = nullptr,
        FrameSwapchainDestroyed swapchain_destroyed = nullptr, void* swapchain_data = nullptr,
        bool async_copy_queue = false, VkQueueFlags copy_queue_flags = VK_QUEUE_TRANSFER_BIT,
        uint32_t compute_queue_family = VK_QUEUE_FAMILY_IGNORED,
        uint32_t copy_queue_family = VK_QUEUE_FAMILY_IGNORED, bool linear_scene = false, bool hdr10 = false);
    // Borrow depth views produced by the offscreen pass. Caller must restore
    // the owned views (empty vector) before destroying the borrowed images.
    bool attach_scene_depth(const std::vector<VkImageView>& views,
        const std::vector<VkImage>& images, std::string& error);
    // Change the sample count used by the scene render passes before the
    // caller recreates the swapchain targets. The count survives recreation.
    bool request_scene_samples(VkSampleCountFlagBits samples, std::string& error);
    bool request_scene_extent(VkExtent2D extent, std::string& error);
    // The G-buffer owner copies depth to these sampled images before the
    // postprocess graph begins. Borrow their handles for graph dependencies.
    bool attach_postprocess_depth(const std::vector<VkImage>& images, std::string& error);
    // Motion vectors are produced during the geometry prepass and sampled by
    // the final postprocess. Caller owns these images through swapchain release.
    bool attach_postprocess_motion(const std::vector<VkImage>& images, std::string& error);
    // Register a shared 1x1 history image before recording begins. The caller
    // owns it and releases it after release_swapchain() and GPU idle.
    bool attach_temporal_exposure(VkImage image, FrameCheckedRecorder recorder, FrameHistoryValid history_valid,
        void* data,
        std::string& error);
    bool attach_temporal_resolve(const std::array<VkImage, 2>& history_images,
        const std::vector<VkImage>& motion_images, FrameCheckedRecorder recorder,
        FrameHistoryValid active, FrameOutputSlot output_slot, void* data,
        std::string& error, VkExtent2D history_extent = {});
    bool attach_fsr3_upscale(const std::vector<VkImage>& output_images,
        const std::vector<VkImage>& motion_images, FrameCheckedRecorder recorder,
        FrameHistoryValid active, void* data, std::string& error);
    bool enable_interpass(std::string& error);
    // Swapchain release/recreation discards pending notifications; callers
    // must invalidate any pending work along with swapchain-sized resources.
    void set_frame_completed_callback(FrameCompleted callback, void* data)
    { m_frame_completed = callback; m_frame_completed_data = data; }
    // Independent observer for systems that need submission and fence events
    // without replacing the existing VSM completion callback.
    void set_frame_observer(FrameSubmitted submitted, FrameCompleted completed, void* data)
    { m_frame_submitted = submitted; m_observer_completed = completed; m_frame_observer_data = data; }
    void set_frame_generation_ui(FrameGenerationUiActive active, FrameGenerationUiRegister register_ui,
        void* data)
    { m_frame_generation_ui_active = active; m_frame_generation_ui_register = register_ui; m_frame_generation_ui_data = data; }
    void clear_depth(VkCommandBuffer command) const;
    void record_distortion_mask(VkCommandBuffer command, uint32_t image_index,
        FrameRecorder recorder, void* data) const;
    bool recreate(VkPhysicalDevice physical_device, VkSurfaceKHR surface,
        VkExtent2D requested_extent, std::string& error);
    bool release_swapchain();
    bool wait_idle();
    bool render_frame(const VkClearColorValue& clear, FrameStatus& status, std::string& error,
        FrameRecorder recorder = nullptr, void* user_data = nullptr,
        FrameReadbackRecorder readback = nullptr, void* readback_data = nullptr,
        FramePrepassRecorder prepass = nullptr, void* prepass_data = nullptr,
        bool clear_target = false, FrameRecorder compositor = nullptr,
        void* compositor_data = nullptr,
        FrameInterpassRecorder interpass = nullptr, void* interpass_data = nullptr,
        FrameRecorder overlay = nullptr, void* overlay_data = nullptr,
        FrameInterpassRecorder final_capture = nullptr, void* final_capture_data = nullptr,
        FrameRecorder final_overlay = nullptr, void* final_overlay_data = nullptr,
        FrameInterpassRecorder refraction_capture = nullptr, void* refraction_capture_data = nullptr,
        FrameRecorder refraction_overlay = nullptr, void* refraction_overlay_data = nullptr,
        FrameRecorder presentation_overlay = nullptr, void* presentation_overlay_data = nullptr,
        FrameInterpassRecorder presentation_capture = nullptr, void* presentation_capture_data = nullptr,
        FrameRecorder presentation_distortion = nullptr, void* presentation_distortion_data = nullptr,
        FrameInterpassRecorder display_capture = nullptr, void* display_capture_data = nullptr,
        FrameRecorder display_grade = nullptr, void* display_grade_data = nullptr,
        bool async_compute_requested = false,
        FrameRecordValidation validate_recording = nullptr, void* validation_data = nullptr,
        FrameSampledImageProvider sampled_images = nullptr, void* sampled_images_data = nullptr);
    void destroy();

    struct CpuTiming
    {
        double fence_ms{}, acquire_ms{}, record_ms{}, submit_ms{}, present_ms{};
    };
    const CpuTiming& cpu_timing() const { return m_cpu_timing; }
    // Fence-completed graphics timeline intervals, including waits for producer
    // queues. These are elapsed intervals, not isolated shader execution costs.
    struct GpuTiming
    {
        double prepass_ms{}, lighting_ms{}, forward_ms{}, postprocess_ms{}, display_ms{};
    };
    const std::optional<GpuTiming>& gpu_timing() const { return m_last_gpu_timing; }
    // Scene callbacks and borrowed depth use extent(); composition/readback use presentation_extent().
    VkExtent2D extent() const { return m_extent; }
    VkExtent2D presentation_extent() const { return m_presentation_extent; }
    VkFormat format() const { return m_format; }
    VkFormat scene_format() const { return m_scene_format; }
    bool hdr10_surface_available() const { return m_hdr10_surface_available; }
    bool hdr10_active() const { return m_color_space == VK_COLOR_SPACE_HDR10_ST2084_EXT; }
    void update_hdr_metadata(float paper_white, float peak, float black);
    VkFormat composite_format() const { return m_composite_format; }
    VkImageView composite_view(uint32_t index) const;
    VkRenderPass display_output_pass() const { return m_display_output_pass; }
    void set_display_output_recorder(FrameRecorder recorder, void* data)
    { m_display_output_recorder = recorder; m_display_output_data = data; }
    VkRenderPass render_pass() const { return m_render_pass; }
    VkRenderPass overlay_render_pass() const { return m_overlay_pass; }
    VkRenderPass composite_render_pass() const { return m_composite_pass; }
    VkRenderPass bloom_render_pass() const { return m_bloom_pass; }
    VkRenderPass bloom_upsample_render_pass() const { return m_bloom_upsample_pass; }
    VkRenderPass presentation_render_pass() const { return m_presentation_pass; }
    VkImageView postprocess_view(size_t index) const { return m_post_views.at(index); }
    VkImageView postprocess_base_view(size_t index) const { return m_post_base_views.at(index); }
    VkImage postprocess_image(size_t index) const { return m_post_images.at(index); }
    uint32_t postprocess_mip_levels() const { return m_post_mip_levels; }
    VkImageView bloom_view(size_t index) const { return m_bloom_views.at(index); }
    VkImageView bloom_level_view(size_t index, size_t mip) const { return m_bloom_level_views.at(index).at(mip); }
    uint32_t bloom_mip_levels() const { return m_bloom_mip_levels; }
    VkFormat depth_format() const { return m_depth_format; }
    VkSampleCountFlagBits scene_samples() const { return m_scene_samples; }
    VkCommandPool command_pool() const { return m_command_pool; }
    size_t image_count() const { return m_images.size(); }
    FrameImagePool& image_pool() { return m_frame_image_pool; }
    const FrameImageSharing& image_sharing() const { return m_frame_image_sharing; }
    bool device_lost() const { return m_device_lost; }
    bool surface_lost() const { return m_surface_lost; }
    std::optional<double> take_gpu_frame_ms()
    {
        auto value = m_last_gpu_ms;
        m_last_gpu_ms.reset();
        return value;
    }

private:
    bool create_swapchain(VkPhysicalDevice physical_device, VkSurfaceKHR surface,
        VkExtent2D requested_extent, std::string& error);
    bool create_render_targets(std::string& error);
    bool create_scene_colors(std::string& error);
    bool create_multisampled_scene_colors(std::string& error);
    bool create_postprocess_targets(std::string& error, VkFormat composite_format = VK_FORMAT_UNDEFINED);
    bool create_commands(std::string& error);
    bool create_sync(std::string& error);
    void destroy_swapchain_resources();

    VkDevice m_device = VK_NULL_HANDLE;
    nvrhi::vulkan::DeviceHandle m_nvrhi_device;
    std::array<std::array<nvrhi::CommandListHandle, 3>, FramesInFlight> m_nvrhi_command_lists{};
    std::array<nvrhi::CommandListHandle, FramesInFlight> m_nvrhi_compute_command_lists{};
    std::array<std::array<std::vector<nvrhi::CommandListHandle>, 3>, FramesInFlight>
        m_nvrhi_frame_graph_command_lists{};
    bool m_async_compute_available = false;
    bool m_async_copy_available = false;
    bool m_async_compute_enabled = false;
    bool m_async_copy_enabled = false;
    uint64_t m_last_graphics_submission{};
    VkQueueFlags m_graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT;
    VkQueueFlags m_compute_queue_flags = VK_QUEUE_COMPUTE_BIT;
    VkQueueFlags m_copy_queue_flags = VK_QUEUE_TRANSFER_BIT;
    uint32_t m_nvrhi_graphics_list_index = 0;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queue_family = UINT32_MAX;
    uint32_t m_compute_queue_family = VK_QUEUE_FAMILY_IGNORED;
    uint32_t m_copy_queue_family = VK_QUEUE_FAMILY_IGNORED;
    FrameImageSharing m_frame_image_sharing{};
    FrameDispatch m_vk{};
    FrameDispatch m_native_vk{};
    FrameSwapchainCreateContext m_swapchain_context{};
    FrameSwapchainCreate m_swapchain_create{};
    FrameSwapchainDestroyed m_swapchain_destroyed{};
    void* m_swapchain_data{};
    FrameGenerationUiActive m_frame_generation_ui_active{};
    FrameGenerationUiRegister m_frame_generation_ui_register{};
    void* m_frame_generation_ui_data{};
    bool m_frame_interpolation_swapchain{};
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    VkRenderPass m_render_pass = VK_NULL_HANDLE;
    VkRenderPass m_overlay_pass = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    VkQueryPool m_timestamp_pool = VK_NULL_HANDLE;
    static constexpr uint32_t TimestampCount = 6;
    float m_timestamp_period_ns{};
    uint32_t m_timestamp_valid_bits{};
    std::array<bool, FramesInFlight> m_timestamp_submitted{};
    std::optional<double> m_last_gpu_ms;
    std::optional<GpuTiming> m_last_gpu_timing;
    CpuTiming m_cpu_timing{};
    VkExtent2D m_extent{};
    VkExtent2D m_presentation_extent{};
    VkExtent2D m_requested_scene_extent{};
    VkExtent2D m_scene_scale_reference{};
    VkSampleCountFlagBits m_scene_samples{VK_SAMPLE_COUNT_1_BIT};
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkFormat m_scene_format = VK_FORMAT_UNDEFINED;
    VkFormat m_composite_format = VK_FORMAT_UNDEFINED;
    bool m_linear_scene{};
    VkColorSpaceKHR m_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    bool m_hdr10_surface_available{};
    bool m_hdr10_requested{};
    float m_hdr_metadata_peak{-1.f};
    float m_hdr_metadata_black{-1.f};
    VkFormat m_depth_format = VK_FORMAT_UNDEFINED;
    VkPhysicalDeviceMemoryProperties m_memory_properties{};
    bool m_allow_readback = false;
    bool m_preserve_prepass_depth = false;
    bool m_postprocess = false;
    bool m_scene_depth_attached = false;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_image_views;
    std::vector<VkImage> m_scene_images;
    std::vector<VkImageView> m_scene_views;
    std::vector<VkImage> m_multisampled_scene_images;
    std::vector<VkImageView> m_multisampled_scene_views;
    std::vector<VkImage> m_depth_images;
    std::vector<VkImage> m_scene_depth_images;
    std::vector<VkDeviceMemory> m_depth_memories;
    std::vector<VkImageView> m_depth_views;
    std::vector<VkFramebuffer> m_framebuffers;
    VkRenderPass m_composite_pass = VK_NULL_HANDLE;
    VkRenderPass m_bloom_pass = VK_NULL_HANDLE;
    VkRenderPass m_bloom_upsample_pass = VK_NULL_HANDLE;
    VkRenderPass m_presentation_pass = VK_NULL_HANDLE;
    VkRenderPass m_ui_clear_pass = VK_NULL_HANDLE;
    std::vector<VkImage> m_post_images;
    FrameImagePool m_frame_image_pool;
    std::vector<VkImage> m_post_depth_images;
    std::vector<VkImage> m_post_motion_images;
    ImageStateTracker m_post_image_states;
    VkImage m_exposure_image = VK_NULL_HANDLE;
    FrameCheckedRecorder m_exposure_recorder{};
    FrameHistoryValid m_exposure_history_valid{};
    void* m_exposure_data{};
    std::array<VkImage, 2> m_temporal_images{};
    VkExtent2D m_temporal_extent{};
    std::vector<VkImage> m_temporal_motion_images;
    FrameCheckedRecorder m_temporal_recorder{};
    FrameHistoryValid m_temporal_active{};
    FrameOutputSlot m_temporal_output_slot{};
    void* m_temporal_data{};
    std::vector<VkImage> m_fsr3_output_images;
    std::vector<VkImage> m_fsr3_motion_images;
    FrameCheckedRecorder m_fsr3_recorder{};
    FrameHistoryValid m_fsr3_active{};
    void* m_fsr3_data{};
    uint32_t m_post_mip_levels{1};
    uint32_t m_bloom_mip_levels{1};
    VkExtent2D m_bloom_extent{};
    VkFormat m_bloom_format = VK_FORMAT_UNDEFINED;
    std::vector<VkImageView> m_post_views;
    std::vector<VkImageView> m_post_base_views;
    std::vector<VkImage> m_bloom_images;
    std::vector<VkImageView> m_bloom_views;
    std::vector<VkImageView> m_bloom_target_views;
    std::vector<VkFramebuffer> m_bloom_framebuffers;
    std::vector<std::vector<VkImageView>> m_bloom_level_views;
    std::vector<std::vector<VkFramebuffer>> m_bloom_upsample_framebuffers;
    std::vector<VkFramebuffer> m_composite_framebuffers;
    std::vector<VkImage> m_composite_images;
    std::vector<VkImageView> m_composite_views;
    VkRenderPass m_display_output_pass{};
    std::vector<VkFramebuffer> m_display_output_framebuffers;
    FrameRecorder m_display_output_recorder{};
    void* m_display_output_data{};
    std::vector<VkImage> m_frame_generation_ui_images;
    std::vector<VkImageView> m_frame_generation_ui_views;
    std::vector<VkFramebuffer> m_frame_generation_ui_framebuffers;
    std::vector<VkCommandBuffer> m_commands;
    std::vector<VkSemaphore> m_render_finished;
    std::array<VkSemaphore, FramesInFlight> m_image_available{};
    std::array<VkFence, FramesInFlight> m_frame_fences{};
    std::array<bool, FramesInFlight> m_completion_pending{};
    std::array<uint32_t, FramesInFlight> m_completion_images{};
    FrameCompleted m_frame_completed{};
    void* m_frame_completed_data{};
    FrameSubmitted m_frame_submitted{};
    FrameCompleted m_observer_completed{};
    void* m_frame_observer_data{};
    std::vector<VkFence> m_image_fences;
    uint32_t m_current_frame{};
    bool m_device_lost{};
    bool m_submission_failed{};
    bool m_surface_lost{};
};
}
