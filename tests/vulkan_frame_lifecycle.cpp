#include "src/Layers/xrRenderVK/FrameContext.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <type_traits>
#include <vector>
#include <map>

using namespace xray::render::vulkan;

namespace
{
template <class T> T handle(uintptr_t id)
{
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(id);
    else return static_cast<T>(id);
}
uintptr_t next_handle = 100;
std::map<VkImage, VkExtent3D> allocated_images;
std::map<VkImage, VkFormat> allocated_formats;
std::map<VkImageView, VkImage> viewed_images;
unsigned allocated_memory{}, freed_memory{};
unsigned created_frame_color_images{};
bool fail_multisample_image{};
unsigned multisample_images_created{};
VkResult swapchain_create_result = VK_SUCCESS;
VkResult surface_capabilities_result = VK_SUCCESS;
unsigned blit_count{};
VkOffset3D temporal_source_extent{}, temporal_destination_extent{};
int fail_image_after = -1;
bool float_bloom_supported = true;
VkImage last_copy_source{};
VkImage last_copy_target{};
VkExtent3D last_copy_extent{};
std::vector<VkImageMemoryBarrier> image_barriers;
std::vector<VkExtent2D> framebuffer_extents;
unsigned idle_calls{}, swapchain_creates{}, swapchain_destroys{}, submissions{}, presents{},
    fences_created{}, fences_destroyed{}, semaphores_created{}, semaphores_destroyed{},
    query_creates{}, query_destroys{}, timestamp_writes{}, fence_waits{};
VkResult next_acquire = VK_SUCCESS, next_present = VK_SUCCESS;
VkResult next_idle = VK_SUCCESS, next_submit = VK_SUCCESS;
VkSwapchainKHR active_swapchain = VK_NULL_HANDLE;
unsigned frame_generation_ui_registrations{};
bool frame_generation_ui_registered_after_overlay{};
VkResult VKAPI_PTR capabilities(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* out)
{
    if (surface_capabilities_result != VK_SUCCESS)
        return surface_capabilities_result;
    *out = {};
    out->minImageCount = 2;
    out->maxImageCount = 2;
    out->currentExtent = {UINT32_MAX, UINT32_MAX};
    out->minImageExtent = {1, 1};
    out->maxImageExtent = {4096, 4096};
    out->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    out->supportedCompositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    return VK_SUCCESS;
}
VkResult VKAPI_PTR formats(VkPhysicalDevice, VkSurfaceKHR, uint32_t* count, VkSurfaceFormatKHR* out)
{
    *count = 1;
    if (out) *out = {VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR};
    return VK_SUCCESS;
}
VkResult VKAPI_PTR create_chain(VkDevice, const VkSwapchainCreateInfoKHR* info,
    const VkAllocationCallbacks*, VkSwapchainKHR* out)
{
    assert(!active_swapchain && info->imageExtent.width && info->imageExtent.height);
    if (swapchain_create_result != VK_SUCCESS)
        return swapchain_create_result;
    active_swapchain = *out = handle<VkSwapchainKHR>(++next_handle);
    ++swapchain_creates;
    return VK_SUCCESS;
}
void VKAPI_PTR destroy_chain(VkDevice, VkSwapchainKHR chain, const VkAllocationCallbacks*)
{
    assert(chain == active_swapchain && idle_calls >= swapchain_destroys + 1);
    active_swapchain = VK_NULL_HANDLE;
    ++swapchain_destroys;
}
VkResult VKAPI_PTR images(VkDevice, VkSwapchainKHR chain, uint32_t* count, VkImage* out)
{
    assert(chain == active_swapchain);
    *count = 2;
    if (out) { out[0] = handle<VkImage>(10); out[1] = handle<VkImage>(11); }
    return VK_SUCCESS;
}
VkResult VKAPI_PTR view(VkDevice, const VkImageViewCreateInfo* info, const VkAllocationCallbacks*, VkImageView* out)
{ if (allocated_formats.count(info->image)) assert(info->format == allocated_formats.at(info->image));
  *out=handle<VkImageView>(++next_handle); viewed_images[*out]=info->image; return VK_SUCCESS; }
void VKAPI_PTR destroy_view(VkDevice, VkImageView view, const VkAllocationCallbacks*)
{ assert(viewed_images.erase(view)==1); }
VkResult VKAPI_PTR pass(VkDevice, const VkRenderPassCreateInfo*, const VkAllocationCallbacks*, VkRenderPass* out)
{ *out = handle<VkRenderPass>(++next_handle); return VK_SUCCESS; }
void VKAPI_PTR destroy_pass(VkDevice, VkRenderPass, const VkAllocationCallbacks*) {}
VkResult VKAPI_PTR framebuffer(VkDevice, const VkFramebufferCreateInfo* info, const VkAllocationCallbacks*, VkFramebuffer* out)
{
    framebuffer_extents.push_back({info->width, info->height});
    const auto image = viewed_images.find(info->pAttachments[0]);
    if (image != viewed_images.end() && allocated_images.count(image->second)) {
        const auto extent = allocated_images.at(image->second);
        assert(info->width <= extent.width && info->height <= extent.height);
    }
    *out = handle<VkFramebuffer>(++next_handle); return VK_SUCCESS;
}
void VKAPI_PTR destroy_framebuffer(VkDevice, VkFramebuffer, const VkAllocationCallbacks*) {}
VkResult VKAPI_PTR pool(VkDevice, const VkCommandPoolCreateInfo*, const VkAllocationCallbacks*, VkCommandPool* out)
{ *out = handle<VkCommandPool>(++next_handle); return VK_SUCCESS; }
void VKAPI_PTR destroy_pool(VkDevice, VkCommandPool, const VkAllocationCallbacks*) {}
VkResult VKAPI_PTR commands(VkDevice, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* out)
{ for (uint32_t i=0; i<info->commandBufferCount; ++i) out[i]=handle<VkCommandBuffer>(++next_handle); return VK_SUCCESS; }
void VKAPI_PTR free_commands(VkDevice, VkCommandPool, uint32_t, const VkCommandBuffer*) {}
VkResult VKAPI_PTR semaphore(VkDevice, const VkSemaphoreCreateInfo*, const VkAllocationCallbacks*, VkSemaphore* out)
{ *out=handle<VkSemaphore>(++next_handle); ++semaphores_created; return VK_SUCCESS; }
void VKAPI_PTR destroy_semaphore(VkDevice, VkSemaphore, const VkAllocationCallbacks*)
{ ++semaphores_destroyed; }
VkResult VKAPI_PTR fence(VkDevice, const VkFenceCreateInfo*, const VkAllocationCallbacks*, VkFence* out)
{ *out=handle<VkFence>(++next_handle); ++fences_created; return VK_SUCCESS; }
void VKAPI_PTR destroy_fence(VkDevice, VkFence, const VkAllocationCallbacks*) { ++fences_destroyed; }
VkResult VKAPI_PTR wait_fences(VkDevice, uint32_t count, const VkFence* fences, VkBool32, uint64_t)
{ assert(count == 1 && *fences); ++fence_waits; return VK_SUCCESS; }
VkResult VKAPI_PTR reset_fences(VkDevice, uint32_t count, const VkFence* fences)
{ assert(count == 1 && *fences); return VK_SUCCESS; }
VkResult VKAPI_PTR acquire(VkDevice, VkSwapchainKHR chain, uint64_t, VkSemaphore semaphore, VkFence, uint32_t* index)
{ assert(chain == active_swapchain && semaphore); *index = 0; const auto result=next_acquire; next_acquire=VK_SUCCESS; return result; }
VkResult VKAPI_PTR submit(VkQueue, uint32_t, const VkSubmitInfo* info, VkFence frame_fence)
{ assert(info->waitSemaphoreCount == 1 && info->signalSemaphoreCount == 1 && frame_fence);
  ++submissions; const auto result=next_submit; next_submit=VK_SUCCESS; return result; }
VkResult VKAPI_PTR present(VkQueue, const VkPresentInfoKHR* info)
{ assert(info->swapchainCount == 1 && *info->pSwapchains == active_swapchain); ++presents;
  const auto result=next_present; next_present=VK_SUCCESS; return result; }
VkResult VKAPI_PTR idle(VkDevice) { ++idle_calls; return next_idle; }
VkResult VKAPI_PTR create_queries(VkDevice, const VkQueryPoolCreateInfo* info,
    const VkAllocationCallbacks*, VkQueryPool* out)
{ assert(info->queryType == VK_QUERY_TYPE_TIMESTAMP && info->queryCount == 4);
  *out = handle<VkQueryPool>(++next_handle); ++query_creates; return VK_SUCCESS; }
void VKAPI_PTR destroy_queries(VkDevice, VkQueryPool, const VkAllocationCallbacks*) { ++query_destroys; }
void VKAPI_PTR reset_queries(VkCommandBuffer, VkQueryPool, uint32_t, uint32_t count)
{ assert(count == 2); }
void VKAPI_PTR write_timestamp(VkCommandBuffer, VkPipelineStageFlagBits, VkQueryPool, uint32_t)
{ ++timestamp_writes; }
VkResult VKAPI_PTR query_results(VkDevice, VkQueryPool, uint32_t, uint32_t count,
    size_t, void* output, VkDeviceSize, VkQueryResultFlags)
{ assert(count == 2); auto* ticks = static_cast<uint64_t*>(output); ticks[0] = 100; ticks[1] = 500100; return VK_SUCCESS; }
VkResult VKAPI_PTR success_command(VkCommandBuffer, VkCommandBufferResetFlags) { return VK_SUCCESS; }
VkResult VKAPI_PTR begin(VkCommandBuffer, const VkCommandBufferBeginInfo*) { return VK_SUCCESS; }
VkResult VKAPI_PTR end(VkCommandBuffer) { return VK_SUCCESS; }
void VKAPI_PTR begin_pass(VkCommandBuffer, const VkRenderPassBeginInfo*, VkSubpassContents) {}
void VKAPI_PTR end_pass(VkCommandBuffer) {}
void VKAPI_PTR unused() {}
void VKAPI_PTR depth_formats(VkPhysicalDevice, VkFormat format, VkFormatProperties *out)
{ out->optimalTilingFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
    VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT |
    VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  if (format == VK_FORMAT_R16G16B16A16_SFLOAT && !float_bloom_supported)
      out->optimalTilingFeatures &= ~VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT; }
void VKAPI_PTR memory_types(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *out)
{ *out = {}; out->memoryTypeCount = 1; out->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; }
VkResult VKAPI_PTR depth_image(VkDevice, const VkImageCreateInfo *info, const VkAllocationCallbacks *, VkImage *out)
{
    if (info->samples != VK_SAMPLE_COUNT_1_BIT)
    {
        if (fail_multisample_image)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        ++multisample_images_created;
    }
    if (fail_image_after == 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (fail_image_after > 0) --fail_image_after;
    *out = handle<VkImage>(++next_handle);
    if (info->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ++created_frame_color_images;
    allocated_images[*out]=info->extent;
    allocated_formats[*out]=info->format; return VK_SUCCESS;
}
void VKAPI_PTR destroy_depth_image(VkDevice, VkImage image, const VkAllocationCallbacks *)
{ assert(allocated_images.erase(image)==1 && allocated_formats.erase(image)==1); }
void VKAPI_PTR depth_requirements(VkDevice, VkImage, VkMemoryRequirements *out)
{ *out = {}; out->size = 4096; out->alignment = 16; out->memoryTypeBits = 1; }
VkResult VKAPI_PTR depth_allocate(VkDevice, const VkMemoryAllocateInfo *, const VkAllocationCallbacks *, VkDeviceMemory *out)
{ *out = handle<VkDeviceMemory>(++next_handle); ++allocated_memory; return VK_SUCCESS; }
void VKAPI_PTR depth_free(VkDevice, VkDeviceMemory, const VkAllocationCallbacks *) { ++freed_memory; }
VkResult VKAPI_PTR depth_bind(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
std::vector<int> pass_order;
struct TemporalTestState { int stage; bool active; uint32_t output; };
bool temporal_record(const FrameRecordingContext&, void* data, std::string& error)
{ pass_order.push_back(static_cast<TemporalTestState*>(data)->stage); error.clear(); return true; }
bool temporal_active(void* data)
{ return static_cast<TemporalTestState*>(data)->active; }
uint32_t temporal_output(void* data)
{ return static_cast<TemporalTestState*>(data)->output; }
void record_stage(const FrameRecordingContext &, void *stage)
{ pass_order.push_back(*static_cast<int *>(stage)); }
bool capture_stage(VkCommandBuffer, VkImage, uint32_t, void *stage, std::string& error)
{ pass_order.push_back(*static_cast<int *>(stage)); error.clear(); return true; }
void VKAPI_PTR copy_image(VkCommandBuffer, VkImage source, VkImageLayout, VkImage target, VkImageLayout,
    uint32_t count, const VkImageCopy* copies)
{ assert(count==1); last_copy_source=source; last_copy_target=target; last_copy_extent=copies[0].extent; }
void VKAPI_PTR blit_image(VkCommandBuffer, VkImage source, VkImageLayout, VkImage target, VkImageLayout,
    uint32_t count, const VkImageBlit* regions, VkFilter filter)
{ assert(count==1);
  if (filter==VK_FILTER_NEAREST)
  { assert(source!=target && regions[0].srcSubresource.mipLevel==0 &&
      regions[0].dstSubresource.mipLevel==0);
    temporal_source_extent = regions[0].srcOffsets[1]; temporal_destination_extent = regions[0].dstOffsets[1]; return; }
  assert(source==target && filter==VK_FILTER_LINEAR);
  assert(regions[0].dstSubresource.mipLevel==regions[0].srcSubresource.mipLevel+1); ++blit_count; }
void VKAPI_PTR barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t count,
    const VkImageMemoryBarrier* images)
{ image_barriers.insert(image_barriers.end(), images, images + count); }
bool frame_interpolation_swapchain(VkSwapchainCreateInfoKHR& info,
    const FrameSwapchainCreateContext& context, FrameDispatch& dispatch, VkSwapchainKHR& swapchain,
    bool& frame_interpolation, void*, std::string& error)
{
    if (!dispatch.create_swapchain || dispatch.create_swapchain(context.device, &info, nullptr,
            &swapchain) != VK_SUCCESS)
    {
        error = "test frame-interpolation swapchain creation failed";
        return false;
    }
    frame_interpolation = true;
    error.clear();
    return true;
}
bool frame_generation_ui_active(void*) { return true; }
bool register_frame_generation_ui(VkImage image, VkFormat format, VkExtent2D extent,
    VkImageUsageFlags usage, void*, std::string& error)
{
    ++frame_generation_ui_registrations;
    frame_generation_ui_registered_after_overlay = false;
    if (!image || format == VK_FORMAT_UNDEFINED || !extent.width || !extent.height ||
        !(usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || !allocated_images.count(image))
    {
        error = "test frame-generation UI image metadata is invalid";
        return false;
    }
    frame_generation_ui_registered_after_overlay =
        std::find(pass_order.begin(), pass_order.end(), 20) != pass_order.end();
    if (!frame_generation_ui_registered_after_overlay)
    {
        error = "frame-generation UI was registered before its overlay pass";
        return false;
    }
    error.clear();
    return true;
}
void scaled_scene(const FrameRecordingContext& context, void*)
{ assert(context.extent.width==320 && context.extent.height==180); pass_order.push_back(1); }
VkExtent2D expected_presentation;
VkRenderPass expected_bloom_pass = VK_NULL_HANDLE;
VkRenderPass expected_upsample_pass = VK_NULL_HANDLE;
std::vector<uint32_t> upsample_levels;
void scaled_output(const FrameRecordingContext& context, void*)
{
  if (context.render_pass == expected_bloom_pass)
  { assert(context.extent.width==160 && context.extent.height==90); pass_order.push_back(15); }
  else if (context.render_pass == expected_upsample_pass)
  { assert(context.mip_level < 3);
    assert(context.extent.width == (160u >> context.mip_level));
    upsample_levels.push_back(context.mip_level); }
  else
  { assert(context.extent.width==expected_presentation.width && context.extent.height==expected_presentation.height); pass_order.push_back(10); }
}
void scaled_readback(VkCommandBuffer, VkImage image, VkExtent2D extent, void*)
{ assert(image==handle<VkImage>(10) || image==handle<VkImage>(11));
  assert(extent.width==expected_presentation.width && extent.height==expected_presentation.height); pass_order.push_back(50); }
VkRenderPass expected_ui_pass;
void native_ui(const FrameRecordingContext& context, void*)
{ assert(context.render_pass==expected_ui_pass); assert(context.extent.width==expected_presentation.width);
  assert(context.extent.height==expected_presentation.height); pass_order.push_back(20); }
bool native_ui_capture(VkCommandBuffer, VkImage image, uint32_t, void*, std::string& error)
{ assert(image==handle<VkImage>(10) || image==handle<VkImage>(11)); pass_order.push_back(30); error.clear(); return true; }
bool rejected_ui_capture(VkCommandBuffer, VkImage, uint32_t, void*, std::string& error)
{ error = "intentional UI capture failure"; return false; }
bool rejected_recording(void*, std::string& error)
{ error = "intentional draw recording failure"; return false; }
void native_ui_distortion(const FrameRecordingContext& context, void*)
{ assert(context.render_pass==expected_ui_pass && context.extent.width==expected_presentation.width);
  pass_order.push_back(40); }
bool display_capture(VkCommandBuffer, VkImage image, uint32_t, void*, std::string& error)
{ assert(image==handle<VkImage>(10) || image==handle<VkImage>(11)); pass_order.push_back(60); error.clear(); return true; }
void display_grade(const FrameRecordingContext& context, void*)
{ assert(context.render_pass==expected_ui_pass && context.extent.width==expected_presentation.width);
  pass_order.push_back(70); }
FrameDispatch dispatch()
{
    FrameDispatch d{};
    d.get_surface_capabilities=capabilities; d.get_surface_formats=formats;
    d.create_swapchain=create_chain; d.destroy_swapchain=destroy_chain;
    d.get_swapchain_images=images; d.acquire_next_image=acquire; d.queue_present=present;
    d.create_image_view=view; d.destroy_image_view=destroy_view;
    d.create_render_pass=pass; d.destroy_render_pass=destroy_pass;
    d.create_framebuffer=framebuffer; d.destroy_framebuffer=destroy_framebuffer;
    d.create_command_pool=pool; d.destroy_command_pool=destroy_pool;
    d.allocate_command_buffers=commands; d.free_command_buffers=free_commands;
    d.reset_command_buffer=success_command; d.begin_command_buffer=begin; d.end_command_buffer=end;
    d.cmd_begin_render_pass=begin_pass; d.cmd_end_render_pass=end_pass;
    d.create_semaphore=semaphore; d.destroy_semaphore=destroy_semaphore;
    d.create_fence=fence; d.destroy_fence=destroy_fence;
    d.wait_for_fences=wait_fences; d.reset_fences=reset_fences;
    d.queue_submit=submit; d.device_wait_idle=idle;
    d.create_query_pool=create_queries; d.destroy_query_pool=destroy_queries;
    d.cmd_reset_query_pool=reset_queries; d.cmd_write_timestamp=write_timestamp;
    d.get_query_pool_results=query_results;
    // Not called by a color-only, no-readback fixture, but mandatory in the
    // dispatch contract for game frames with depth and screenshot support.
    d.get_format_properties=reinterpret_cast<decltype(d.get_format_properties)>(unused);
    d.get_memory_properties=reinterpret_cast<decltype(d.get_memory_properties)>(unused);
    d.create_image=reinterpret_cast<decltype(d.create_image)>(unused);
    d.destroy_image=reinterpret_cast<decltype(d.destroy_image)>(unused);
    d.get_image_memory_requirements=reinterpret_cast<decltype(d.get_image_memory_requirements)>(unused);
    d.allocate_memory=reinterpret_cast<decltype(d.allocate_memory)>(unused);
    d.free_memory=reinterpret_cast<decltype(d.free_memory)>(unused);
    d.bind_image_memory=reinterpret_cast<decltype(d.bind_image_memory)>(unused);
    d.cmd_clear_attachments=reinterpret_cast<decltype(d.cmd_clear_attachments)>(unused);
    d.cmd_copy_image_to_buffer=reinterpret_cast<decltype(d.cmd_copy_image_to_buffer)>(unused);
    d.cmd_copy_image=reinterpret_cast<decltype(d.cmd_copy_image)>(unused);
    d.cmd_pipeline_barrier=reinterpret_cast<decltype(d.cmd_pipeline_barrier)>(unused);
    return d;
}
}

int main()
{
    bool surface_lost = false;
    FrameStatus status = FrameStatus::Presented;
    assert(!mark_surface_lost(VK_SUCCESS, surface_lost, status));
    assert(!surface_lost && status == FrameStatus::Presented);

    assert(mark_surface_lost(VK_ERROR_SURFACE_LOST_KHR, surface_lost, status));
    assert(surface_lost && status == FrameStatus::RecreateRequired);

    status = FrameStatus::Presented;
    assert(mark_surface_lost(VK_ERROR_SURFACE_LOST_KHR, surface_lost, status));
    assert(surface_lost && status == FrameStatus::RecreateRequired);

    FrameContext frame;
    std::string error;
    const VkClearColorValue clear{};
    const auto physical=handle<VkPhysicalDevice>(1);
    const auto device=handle<VkDevice>(2);
    const auto surface=handle<VkSurfaceKHR>(3);
    const auto queue=handle<VkQueue>(4);
    assert(frame.initialize(physical, device, surface, queue, 0, {640, 480}, dispatch(), error,
        false, false, false, false, 1.f, 64));
    assert(frame.image_count() == 2 && swapchain_creates == 1);
    assert(frame.render_frame(clear, status, error) && status == FrameStatus::Presented);
    assert(submissions == 1 && presents == 1);
    next_acquire = VK_ERROR_OUT_OF_DATE_KHR;
    assert(frame.render_frame(clear, status, error) && status == FrameStatus::RecreateRequired);
    assert(submissions == 1); // No submission or unsignaled fence after failed acquire.
    assert(frame.recreate(physical, surface, {720, 1280}, error));
    assert(swapchain_destroys == 1 && swapchain_creates == 2);
    assert(frame.extent().width == 720 && frame.extent().height == 1280);
    next_acquire = VK_SUBOPTIMAL_KHR;
    assert(frame.render_frame(clear, status, error) && status == FrameStatus::Presented);
    next_acquire = VK_SUCCESS;
    next_present = VK_SUBOPTIMAL_KHR;
    assert(frame.render_frame(clear, status, error) && status == FrameStatus::Presented);
    next_present = VK_ERROR_SURFACE_LOST_KHR;
    assert(frame.render_frame(clear, status, error) && frame.surface_lost());
    assert(frame.take_gpu_frame_ms().value_or(-1.) == .5);
    assert(frame.recreate(physical, surface, {720, 1280}, error) && !frame.surface_lost());
    next_acquire = VK_ERROR_DEVICE_LOST;
    assert(!frame.render_frame(clear, status, error) && frame.device_lost());
    next_idle = VK_ERROR_DEVICE_LOST;
    frame.destroy();
    next_idle = VK_SUCCESS;
    next_acquire = VK_SUCCESS;
    assert(frame.initialize(physical, device, surface, queue, 0, {640, 480}, dispatch(), error,
        false, false, false, false, 1.f, 64));
    assert(!frame.device_lost() && frame.render_frame(clear, status, error));
    struct Completion { uint32_t count{}, frame{}, image{}; } completion;
    struct Observer { uint32_t submitted{}, completed{}, slot{}, image{}; } observer;
    frame.set_frame_completed_callback([](uint32_t slot, uint32_t image, void* user, std::string&) {
        auto& result = *static_cast<Completion*>(user);
        ++result.count; result.frame = slot; result.image = image; return true;
    }, &completion);
    frame.set_frame_observer([](uint32_t slot, uint32_t image, void* user) {
        auto& result = *static_cast<Observer*>(user);
        ++result.submitted; result.slot = slot; result.image = image;
    }, [](uint32_t, uint32_t, void* user, std::string&) {
        ++static_cast<Observer*>(user)->completed; return true;
    }, &observer);
    assert(frame.render_frame(clear, status, error));
    assert(frame.render_frame(clear, status, error));
    assert(completion.count == 0 && observer.submitted == 2 && observer.completed == 0);
    assert(frame.render_frame(clear, status, error));
    assert(completion.count == 1 && completion.frame == 1 && completion.image < frame.image_count() &&
        observer.submitted == 3 && observer.completed == 1 && observer.image < frame.image_count());
    frame.set_frame_completed_callback(nullptr, nullptr);
    frame.set_frame_observer(nullptr, nullptr, nullptr);
    frame.destroy();
    auto world_dispatch = dispatch();
    world_dispatch.get_format_properties = depth_formats;
    world_dispatch.get_memory_properties = memory_types;
    world_dispatch.create_image = depth_image;
    world_dispatch.destroy_image = destroy_depth_image;
    world_dispatch.get_image_memory_requirements = depth_requirements;
    world_dispatch.allocate_memory = depth_allocate;
    world_dispatch.free_memory = depth_free;
    world_dispatch.bind_image_memory = depth_bind;
    assert(frame.initialize(physical, device, surface, queue, 0, {640, 480}, world_dispatch, error,
        true, true, true));
    assert(frame.attach_scene_depth({handle<VkImageView>(801), handle<VkImageView>(802)},
        {handle<VkImage>(901), handle<VkImage>(902)}, error));
    assert(frame.enable_interpass(error));
    int opaque = 1, opaque_copy = 2, transparent = 3, world_copy = 4, distortion_and_hud = 5,
        ui_copy = 6, ui_distortion = 7;
    assert(frame.render_frame(clear, status, error, record_stage, &opaque, nullptr, nullptr,
        nullptr, nullptr, false, nullptr, nullptr, capture_stage, &opaque_copy, record_stage, &transparent,
        capture_stage, &ui_copy, record_stage, &ui_distortion,
        capture_stage, &world_copy, record_stage, &distortion_and_hud));
    assert((pass_order == std::vector<int>{1, 2, 3, 4, 5, 6, 7}));
    pass_order.clear();
    // Frames without refractive particles retain the original two passes.
    assert(frame.render_frame(clear, status, error, record_stage, &opaque, nullptr, nullptr,
        nullptr, nullptr, false, nullptr, nullptr, capture_stage, &opaque_copy, record_stage, &transparent));
    assert((pass_order == std::vector<int>{1, 2, 3}));
    frame.destroy();
    // A scene-sized color target must not shrink the presentation or screenshots.
    world_dispatch.cmd_copy_image=copy_image;
    world_dispatch.cmd_blit_image=blit_image;
    world_dispatch.cmd_pipeline_barrier=barrier;
    framebuffer_extents.clear();
    assert(frame.initialize(physical, device, surface, queue, 0, {640, 360}, world_dispatch, error,
        true, false, false, true, 0.f, 0, {320, 180}, {}, false,
        VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT, VK_QUEUE_COMPUTE_BIT));
    std::vector<VkImage> sampled_depth;
    for (size_t i = 0; i < frame.image_count(); ++i)
        sampled_depth.push_back(handle<VkImage>(900 + i));
    assert(!frame.attach_postprocess_depth({}, error));
    assert(frame.attach_postprocess_depth(sampled_depth, error));
    assert(!frame.attach_postprocess_depth(sampled_depth, error));
    assert(frame.extent().width==320 && frame.presentation_extent().width==640);
    assert(framebuffer_extents.size()==12);
    assert(framebuffer_extents[0].width==320 && framebuffer_extents[2].width==160 &&
        framebuffer_extents[6].width==640);
    expected_bloom_pass=frame.bloom_render_pass();
    expected_upsample_pass=frame.bloom_upsample_render_pass();
    expected_presentation={640,360};
    pass_order.clear();
    const bool scaled_frame_rendered = frame.render_frame(clear,status,error,scaled_scene,nullptr,
        scaled_readback,nullptr,nullptr,nullptr,false,scaled_output);
    if (!scaled_frame_rendered) std::fprintf(stderr, "scaled postprocess frame failed: %s\n", error.c_str());
    assert(scaled_frame_rendered);
    assert((pass_order==std::vector<int>{1,15,10,50}));
    assert((upsample_levels==std::vector<uint32_t>{2,1,0}));
    assert(allocated_images.count(last_copy_source) && last_copy_extent.width==320 && last_copy_extent.height==180);
    assert(blit_count > 0);
    const VkImage post_image = last_copy_target;
    const VkImage bloom_image = viewed_images.at(frame.bloom_view(0));
    assert(bloom_image != post_image);
    assert(allocated_images.at(bloom_image).width == 160 && allocated_images.at(bloom_image).height == 90);
    assert(allocated_formats.at(bloom_image) == VK_FORMAT_R16G16B16A16_SFLOAT);
    bool first_post_write = false;
    bool first_bloom_write = false, bloom_sampled = false;
    for (const auto& entry : image_barriers)
    {
        if (entry.image == post_image && entry.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            first_post_write |= entry.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED;
        if (entry.image == bloom_image && entry.newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
            first_bloom_write |= entry.oldLayout == VK_IMAGE_LAYOUT_UNDEFINED;
        if (entry.image == bloom_image && entry.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            bloom_sampled = true;
    }
    assert(first_post_write && first_bloom_write && bloom_sampled);
    image_barriers.clear();
    const bool reused_frame_rendered = frame.render_frame(clear,status,error,scaled_scene,nullptr,
        scaled_readback,nullptr,nullptr,nullptr,false,scaled_output);
    if (!reused_frame_rendered) std::fprintf(stderr, "reused postprocess frame failed: %s\n", error.c_str());
    assert(reused_frame_rendered);
    bool reused_post_image = false;
    bool reused_bloom_image = false;
    for (const auto& entry : image_barriers)
    {
        if (entry.image == post_image && entry.newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
            reused_post_image |= entry.oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL &&
                entry.srcAccessMask == VK_ACCESS_SHADER_READ_BIT;
        if (entry.image == bloom_image && entry.newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
            reused_bloom_image |= entry.oldLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }
    assert(reused_post_image && reused_bloom_image);
    image_barriers.clear();
    TemporalTestState temporal{25, true, 0};
    const std::array<VkImage, 2> histories{handle<VkImage>(11000), handle<VkImage>(11001)};
    const std::vector<VkImage> motion_images{handle<VkImage>(11002), handle<VkImage>(11003)};
    assert(frame.attach_temporal_resolve(histories, motion_images, temporal_record,
        temporal_active, temporal_output, &temporal, error, {640, 360}));
    assert(!frame.attach_temporal_resolve(histories, motion_images, temporal_record,
        temporal_active, temporal_output, &temporal, error));
    pass_order.clear();
    const bool temporal_frame_rendered = frame.render_frame(clear,status,error,scaled_scene,nullptr,
        scaled_readback,nullptr,nullptr,nullptr,false,scaled_output);
    if (!temporal_frame_rendered) std::fprintf(stderr, "temporal postprocess frame failed: %s\n", error.c_str());
    assert(temporal_frame_rendered);
    assert((pass_order==std::vector<int>{1,25,15,10,50}));
    bool history_written = false, history_copied = false, output_sampled = false;
    for (const auto& entry : image_barriers)
    {
        if (entry.image == histories[0] && entry.newLayout == VK_IMAGE_LAYOUT_GENERAL)
            history_written = true;
        if (entry.image == histories[0] && entry.newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            history_copied = true;
        if (entry.image == histories[0] && entry.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            output_sampled = true;
    }
    assert(history_written && history_copied && output_sampled);
    assert(temporal_source_extent.x == 640 && temporal_source_extent.y == 360 &&
        temporal_destination_extent.x == 320 && temporal_destination_extent.y == 180);
    temporal.output = 1;
    pass_order.clear();
    image_barriers.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output));
    assert((pass_order==std::vector<int>{1,25,15,10,50}));
    bool prior_history_sampled = false;
    for (const auto& entry : image_barriers)
        if (entry.image == histories[0] && entry.newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            prior_history_sampled = true;
    assert(prior_history_sampled);
    temporal.active = false;
    pass_order.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output));
    assert((pass_order==std::vector<int>{1,15,10,50}));
    assert(frame.recreate(physical,surface,{1280,720},error));
    sampled_depth.clear();
    for (size_t i = 0; i < frame.image_count(); ++i)
        sampled_depth.push_back(handle<VkImage>(910 + i));
    assert(frame.attach_postprocess_depth(sampled_depth, error));
    assert(frame.extent().width==320 && frame.presentation_extent().width==1280);
    expected_presentation={1280,720};
    expected_bloom_pass=frame.bloom_render_pass();
    expected_upsample_pass=frame.bloom_upsample_render_pass();
    expected_ui_pass=frame.presentation_render_pass();
    assert(expected_ui_pass);
    pass_order.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, // world capture/overlay
        nullptr,nullptr, nullptr,nullptr, // scene UI capture/overlay
        nullptr,nullptr, nullptr,nullptr, // particle refraction capture/overlay
        native_ui,nullptr, native_ui_capture,nullptr, native_ui_distortion,nullptr,
        display_capture,nullptr, display_grade,nullptr));
    assert((pass_order==std::vector<int>{1,15,10,20,30,40,60,70,50}));
    // Display correction also works without any UI distortion batches.
    pass_order.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
        nullptr,nullptr, nullptr,nullptr, native_ui,nullptr,
        nullptr,nullptr, nullptr,nullptr, display_capture,nullptr, display_grade,nullptr));
    assert((pass_order==std::vector<int>{1,15,10,20,60,70,50}));
    // Neutral correction needs neither a full-resolution snapshot nor a draw.
    pass_order.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
        nullptr,nullptr, nullptr,nullptr, native_ui,nullptr));
    assert((pass_order==std::vector<int>{1,15,10,20,50}));
    // A capture without its distortion draw must fail before submitting a frame.
    const auto submitted_before=submissions;
    assert(!frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
        nullptr,nullptr, nullptr,nullptr, native_ui,nullptr, native_ui_capture,nullptr));
    assert(submissions==submitted_before);
    assert(!frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
        nullptr,nullptr, nullptr,nullptr, native_ui,nullptr,
        nullptr,nullptr, nullptr,nullptr, display_capture,nullptr));
    assert(submissions==submitted_before);
    // A failed image capture stops graph recording before any queue submit.
    assert(!frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
        nullptr,nullptr, nullptr,nullptr, native_ui,nullptr,
        rejected_ui_capture,nullptr, native_ui_distortion,nullptr));
    assert(error.find("intentional UI capture failure") != std::string::npos);
    assert(submissions==submitted_before);
    const auto waits_after_capture_failure = fence_waits;
    assert(!frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output));
    assert(error.find("recreate the swapchain") != std::string::npos);
    assert(fence_waits==waits_after_capture_failure && submissions==submitted_before);
    // Gameplay releases framebuffers before borrowed G-buffer depth and then
    // rebuilds the surface. Exercise that order repeatedly with a fixed scene
    // size and changing presentation size, including the final UI/grade pass.
    const auto live_images = allocated_images.size();
    const auto live_memory = allocated_memory - freed_memory;
    const auto frame_color_images_created_before_resizes = created_frame_color_images;
    size_t pooled_images = 0;
    unsigned pooled_memory = 0;
    for (unsigned cycle = 0; cycle < 12; ++cycle)
    {
        assert(frame.release_swapchain());
        // Views are rebuilt on resize, while exact-match postprocess and bloom
        // images stay in the frame image pool until reuse or eviction.
        assert(viewed_images.empty());
        if (cycle == 0)
        {
            pooled_images = allocated_images.size();
            pooled_memory = allocated_memory - freed_memory;
            assert(pooled_images > 0 && pooled_images <= live_images);
        }
        assert(allocated_images.size() == pooled_images &&
            allocated_memory - freed_memory == pooled_memory);
        expected_presentation = cycle % 2 ? VkExtent2D{1280,720} : VkExtent2D{640,360};
        assert(frame.recreate(physical,surface,expected_presentation,error));
        assert(frame.extent().width==320 && frame.extent().height==180);
        assert(allocated_images.size()==live_images && allocated_memory-freed_memory==live_memory);
        assert(created_frame_color_images == frame_color_images_created_before_resizes);
        expected_ui_pass=frame.presentation_render_pass();
        expected_bloom_pass=frame.bloom_render_pass();
        expected_upsample_pass=frame.bloom_upsample_render_pass();
        assert(expected_ui_pass);
        pass_order.clear();
        assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
            nullptr,nullptr,false,scaled_output,nullptr,
            nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
            nullptr,nullptr, nullptr,nullptr, native_ui,nullptr,
            native_ui_capture,nullptr, native_ui_distortion,nullptr,
            display_capture,nullptr, display_grade,nullptr));
        assert((pass_order==std::vector<int>{1,15,10,20,30,40,60,70,50}));
        if (cycle == 0)
        {
            const auto submitted_before_validation = submissions;
            assert(!frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
                nullptr,nullptr,false,scaled_output,nullptr,
                nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
                nullptr,nullptr, nullptr,nullptr, native_ui,nullptr,
                native_ui_capture,nullptr, native_ui_distortion,nullptr,
                display_capture,nullptr, display_grade,nullptr,
                false, rejected_recording, nullptr));
            assert(error.find("intentional draw recording failure") != std::string::npos);
            assert(submissions == submitted_before_validation);
        }
    }
    // A rejected submit leaves an acquired image and a reset fence. A second
    // render must fail before waiting on that fence or acquiring another image.
    uint32_t failed_submit_notifications = 0;
    frame.set_frame_observer([](uint32_t, uint32_t, void* user) {
        ++*static_cast<uint32_t*>(user);
    }, nullptr, &failed_submit_notifications);
    next_submit = VK_ERROR_UNKNOWN;
    assert(!frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output));
    assert(error == "vkQueueSubmit failed" && !frame.device_lost() && failed_submit_notifications == 0);
    const auto waits_after_failure = fence_waits;
    const auto submits_after_failure = submissions;
    assert(!frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output));
    assert(error.find("recreate the swapchain") != std::string::npos);
    assert(fence_waits == waits_after_failure && submissions == submits_after_failure);
    frame.set_frame_observer(nullptr, nullptr, nullptr);
    assert(frame.recreate(physical,surface,{640,360},error));
    expected_presentation = {640,360};
    expected_bloom_pass=frame.bloom_render_pass();
    expected_upsample_pass=frame.bloom_upsample_render_pass();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output));
    frame.destroy();
    assert(allocated_images.empty() && viewed_images.empty() && allocated_memory == freed_memory);
    const VkExtent2D extent_before_msaa{640, 360};
    assert(frame.initialize(physical, device, surface, queue, 0, extent_before_msaa, world_dispatch, error,
        true, true, false, true, 0.f, 0, {320, 180}));
    assert(frame.request_scene_samples(VK_SAMPLE_COUNT_4_BIT, error));
    fail_multisample_image = true;
    assert(!frame.recreate(physical, surface, extent_before_msaa, error));
    fail_multisample_image = false;
    assert(frame.request_scene_samples(VK_SAMPLE_COUNT_1_BIT, error));
    assert(frame.recreate(physical, surface, extent_before_msaa, error));
    const auto multisample_count = multisample_images_created;
    assert(frame.request_scene_samples(VK_SAMPLE_COUNT_2_BIT, error));
    assert(frame.recreate(physical, surface, extent_before_msaa, error));
    assert(frame.scene_samples() == VK_SAMPLE_COUNT_2_BIT && multisample_images_created > multisample_count);
    assert(frame.request_scene_samples(VK_SAMPLE_COUNT_4_BIT, error));
    assert(frame.recreate(physical, surface, extent_before_msaa, error));
    assert(frame.scene_samples() == VK_SAMPLE_COUNT_4_BIT);
    assert(frame.request_scene_samples(VK_SAMPLE_COUNT_1_BIT, error));
    assert(frame.recreate(physical, surface, extent_before_msaa, error));
    assert(frame.scene_samples() == VK_SAMPLE_COUNT_1_BIT);
    surface_capabilities_result = VK_ERROR_SURFACE_LOST_KHR;
    assert(!frame.recreate(physical, surface, extent_before_msaa, error));
    assert(frame.surface_lost() && !frame.device_lost());
    surface_capabilities_result = VK_SUCCESS;
    assert(frame.recreate(physical, surface, extent_before_msaa, error));
    swapchain_create_result = VK_ERROR_SURFACE_LOST_KHR;
    assert(!frame.recreate(physical, surface, extent_before_msaa, error));
    assert(frame.surface_lost() && !frame.device_lost());
    swapchain_create_result = VK_SUCCESS;
    assert(frame.recreate(physical, surface, extent_before_msaa, error));
    assert(!frame.surface_lost());
    swapchain_create_result = VK_ERROR_DEVICE_LOST;
    assert(!frame.recreate(physical, surface, extent_before_msaa, error));
    assert(frame.device_lost() && !frame.surface_lost());
    swapchain_create_result = VK_SUCCESS;
    frame.destroy();
    assert(allocated_images.empty() && viewed_images.empty() && allocated_memory==freed_memory);
    // Frame-generation UI is registered only after the display graph declares
    // and records its final transfer-source read.
    FrameSwapchainCreateContext frame_interpolation_context{};
    frame_interpolation_context.physical_device = physical;
    frame_interpolation_context.device = device;
    frame_interpolation_context.game_queue = queue;
    frame_interpolation_context.game_queue_family = 0;
    frame_interpolation_context.game_queue_flags = VK_QUEUE_GRAPHICS_BIT;
    frame_interpolation_context.present_queue = queue;
    frame_interpolation_context.present_queue_family = 0;
    FrameContext interpolation_frame;
    const bool interpolation_frame_initialized = interpolation_frame.initialize(
        physical,device,surface,queue,0,{640,360},world_dispatch,error,
        true,false,false,true,0.f,0,{320,180},{},false,
        VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,VK_QUEUE_COMPUTE_BIT,
        &frame_interpolation_context,frame_interpolation_swapchain);
    if (!interpolation_frame_initialized)
        std::fprintf(stderr, "frame-interpolation UI frame initialization failed: %s\n", error.c_str());
    assert(interpolation_frame_initialized);
    interpolation_frame.set_frame_generation_ui(frame_generation_ui_active,
        register_frame_generation_ui,nullptr);
    expected_presentation={640,360};
    expected_ui_pass=interpolation_frame.presentation_render_pass();
    expected_bloom_pass=interpolation_frame.bloom_render_pass();
    expected_upsample_pass=interpolation_frame.bloom_upsample_render_pass();
    frame_generation_ui_registrations=0;
    frame_generation_ui_registered_after_overlay=false;
    pass_order.clear();
    image_barriers.clear();
    const bool frame_interpolation_rendered = interpolation_frame.render_frame(
        clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        nullptr,nullptr,nullptr,nullptr,native_ui,nullptr);
    if (!frame_interpolation_rendered)
        std::fprintf(stderr, "frame-interpolation UI frame failed: %s\n", error.c_str());
    assert(frame_interpolation_rendered);
    assert(frame_generation_ui_registrations==1 && frame_generation_ui_registered_after_overlay);
    assert((pass_order==std::vector<int>{1,15,10,20,50}));
    interpolation_frame.destroy();
    assert(allocated_images.empty() && viewed_images.empty() && allocated_memory==freed_memory);
    // Failure after the first scene image must release all partially built resources.
    fail_image_after=1;
    assert(!frame.initialize(physical,device,surface,queue,0,{640,360},world_dispatch,error,
        true,false,false,true,0.f,0,{320,180}));
    assert(allocated_images.empty() && viewed_images.empty() && allocated_memory==freed_memory);
    fail_image_after=-1;
    // A failure while creating the first bloom image must release the scene
    // colors, postprocess image, views, memory, and render passes as well.
    fail_image_after=3;
    assert(!frame.initialize(physical,device,surface,queue,0,{640,360},world_dispatch,error,
        true,false,false,true,0.f,0,{320,180}));
    assert(error.find("bloom image") != std::string::npos);
    assert(allocated_images.empty() && viewed_images.empty() && allocated_memory==freed_memory);
    fail_image_after=-1;
    assert(!frame.initialize(physical,device,surface,queue,0,{640,360},world_dispatch,error,
        true,false,false,false,0.f,0,{320,180}));
    float_bloom_supported=false;
    assert(frame.initialize(physical,device,surface,queue,0,{640,360},world_dispatch,error,
        true,false,false,true,0.f,0,{320,180}));
    assert(allocated_formats.at(viewed_images.at(frame.bloom_view(0))) == VK_FORMAT_B8G8R8A8_UNORM);
    frame.destroy();
    float_bloom_supported=true;
    assert(swapchain_creates == swapchain_destroys && fences_created == fences_destroyed &&
        semaphores_created == semaphores_destroyed && query_creates == query_destroys && timestamp_writes >= 4);
}
