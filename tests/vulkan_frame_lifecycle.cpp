#include "src/Layers/xrRenderVK/FrameContext.h"

#include <cassert>
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
std::map<VkImageView, VkImage> viewed_images;
unsigned allocated_memory{}, freed_memory{};
int fail_image_after = -1;
VkImage last_copy_source{};
VkExtent3D last_copy_extent{};
std::vector<VkExtent2D> framebuffer_extents;
unsigned idle_calls{}, swapchain_creates{}, swapchain_destroys{}, submissions{}, presents{},
    fences_created{}, fences_destroyed{}, semaphores_created{}, semaphores_destroyed{},
    query_creates{}, query_destroys{}, timestamp_writes{};
VkResult next_acquire = VK_SUCCESS, next_present = VK_SUCCESS;
VkResult next_idle = VK_SUCCESS;
VkSwapchainKHR active_swapchain = VK_NULL_HANDLE;
VkResult VKAPI_PTR capabilities(VkPhysicalDevice, VkSurfaceKHR, VkSurfaceCapabilitiesKHR* out)
{
    *out = {};
    out->minImageCount = 2;
    out->maxImageCount = 2;
    out->currentExtent = {UINT32_MAX, UINT32_MAX};
    out->minImageExtent = {1, 1};
    out->maxImageExtent = {4096, 4096};
    out->supportedUsageFlags = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
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
{ *out=handle<VkImageView>(++next_handle); viewed_images[*out]=info->image; return VK_SUCCESS; }
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
{ assert(count == 1 && *fences); return VK_SUCCESS; }
VkResult VKAPI_PTR reset_fences(VkDevice, uint32_t count, const VkFence* fences)
{ assert(count == 1 && *fences); return VK_SUCCESS; }
VkResult VKAPI_PTR acquire(VkDevice, VkSwapchainKHR chain, uint64_t, VkSemaphore semaphore, VkFence, uint32_t* index)
{ assert(chain == active_swapchain && semaphore); *index = 0; const auto result=next_acquire; next_acquire=VK_SUCCESS; return result; }
VkResult VKAPI_PTR submit(VkQueue, uint32_t, const VkSubmitInfo* info, VkFence frame_fence)
{ assert(info->waitSemaphoreCount == 1 && info->signalSemaphoreCount == 1 && frame_fence); ++submissions; return VK_SUCCESS; }
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
void VKAPI_PTR depth_formats(VkPhysicalDevice, VkFormat, VkFormatProperties *out)
{ out->optimalTilingFeatures = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT; }
void VKAPI_PTR memory_types(VkPhysicalDevice, VkPhysicalDeviceMemoryProperties *out)
{ *out = {}; out->memoryTypeCount = 1; out->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT; }
VkResult VKAPI_PTR depth_image(VkDevice, const VkImageCreateInfo *info, const VkAllocationCallbacks *, VkImage *out)
{
    if (fail_image_after == 0) return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (fail_image_after > 0) --fail_image_after;
    *out = handle<VkImage>(++next_handle); allocated_images[*out]=info->extent; return VK_SUCCESS;
}
void VKAPI_PTR destroy_depth_image(VkDevice, VkImage image, const VkAllocationCallbacks *) { assert(allocated_images.erase(image)==1); }
void VKAPI_PTR depth_requirements(VkDevice, VkImage, VkMemoryRequirements *out)
{ *out = {}; out->size = 4096; out->alignment = 16; out->memoryTypeBits = 1; }
VkResult VKAPI_PTR depth_allocate(VkDevice, const VkMemoryAllocateInfo *, const VkAllocationCallbacks *, VkDeviceMemory *out)
{ *out = handle<VkDeviceMemory>(++next_handle); ++allocated_memory; return VK_SUCCESS; }
void VKAPI_PTR depth_free(VkDevice, VkDeviceMemory, const VkAllocationCallbacks *) { ++freed_memory; }
VkResult VKAPI_PTR depth_bind(VkDevice, VkImage, VkDeviceMemory, VkDeviceSize) { return VK_SUCCESS; }
std::vector<int> pass_order;
void record_stage(const FrameRecordingContext &, void *stage)
{ pass_order.push_back(*static_cast<int *>(stage)); }
void capture_stage(VkCommandBuffer, VkImage, uint32_t, void *stage)
{ pass_order.push_back(*static_cast<int *>(stage)); }
void VKAPI_PTR copy_image(VkCommandBuffer, VkImage source, VkImageLayout, VkImage, VkImageLayout,
    uint32_t count, const VkImageCopy* copies)
{ assert(count==1); last_copy_source=source; last_copy_extent=copies[0].extent; }
void VKAPI_PTR barrier(VkCommandBuffer, VkPipelineStageFlags, VkPipelineStageFlags, VkDependencyFlags,
    uint32_t, const VkMemoryBarrier*, uint32_t, const VkBufferMemoryBarrier*, uint32_t, const VkImageMemoryBarrier*) {}
void scaled_scene(const FrameRecordingContext& context, void*)
{ assert(context.extent.width==320 && context.extent.height==180); pass_order.push_back(1); }
VkExtent2D expected_presentation;
void scaled_output(const FrameRecordingContext& context, void*)
{ assert(context.extent.width==expected_presentation.width && context.extent.height==expected_presentation.height); pass_order.push_back(10); }
void scaled_readback(VkCommandBuffer, VkImage image, VkExtent2D extent, void*)
{ assert(image==handle<VkImage>(10) || image==handle<VkImage>(11));
  assert(extent.width==expected_presentation.width && extent.height==expected_presentation.height); pass_order.push_back(50); }
VkRenderPass expected_ui_pass;
void native_ui(const FrameRecordingContext& context, void*)
{ assert(context.render_pass==expected_ui_pass); assert(context.extent.width==expected_presentation.width);
  assert(context.extent.height==expected_presentation.height); pass_order.push_back(20); }
void native_ui_capture(VkCommandBuffer, VkImage image, uint32_t, void*)
{ assert(image==handle<VkImage>(10) || image==handle<VkImage>(11)); pass_order.push_back(30); }
void native_ui_distortion(const FrameRecordingContext& context, void*)
{ assert(context.render_pass==expected_ui_pass && context.extent.width==expected_presentation.width);
  pass_order.push_back(40); }
void display_capture(VkCommandBuffer, VkImage image, uint32_t, void*)
{ assert(image==handle<VkImage>(10) || image==handle<VkImage>(11)); pass_order.push_back(60); }
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
    assert(frame.attach_scene_depth({handle<VkImageView>(801), handle<VkImageView>(802)}, error));
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
    world_dispatch.cmd_pipeline_barrier=barrier;
    framebuffer_extents.clear();
    assert(frame.initialize(physical, device, surface, queue, 0, {640, 360}, world_dispatch, error,
        true, false, false, true, 0.f, 0, {320, 180}));
    assert(frame.extent().width==320 && frame.presentation_extent().width==640);
    assert(framebuffer_extents.size()==4);
    assert(framebuffer_extents[0].width==320 && framebuffer_extents[2].width==640);
    expected_presentation={640,360};
    pass_order.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output));
    assert((pass_order==std::vector<int>{1,10,50}));
    assert(allocated_images.count(last_copy_source) && last_copy_extent.width==320 && last_copy_extent.height==180);
    assert(frame.recreate(physical,surface,{1280,720},error));
    assert(frame.extent().width==320 && frame.presentation_extent().width==1280);
    expected_presentation={1280,720};
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
    assert((pass_order==std::vector<int>{1,10,20,30,40,60,70,50}));
    // Display correction also works without any UI distortion batches.
    pass_order.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
        nullptr,nullptr, nullptr,nullptr, native_ui,nullptr,
        nullptr,nullptr, nullptr,nullptr, display_capture,nullptr, display_grade,nullptr));
    assert((pass_order==std::vector<int>{1,10,20,60,70,50}));
    // Neutral correction needs neither a full-resolution snapshot nor a draw.
    pass_order.clear();
    assert(frame.render_frame(clear,status,error,scaled_scene,nullptr,scaled_readback,nullptr,
        nullptr,nullptr,false,scaled_output,nullptr,
        nullptr,nullptr, nullptr,nullptr, nullptr,nullptr, nullptr,nullptr,
        nullptr,nullptr, nullptr,nullptr, native_ui,nullptr));
    assert((pass_order==std::vector<int>{1,10,20,50}));
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
    frame.destroy();
    assert(allocated_images.empty() && viewed_images.empty() && allocated_memory==freed_memory);
    // Failure after the first scene image must release all partially built resources.
    fail_image_after=1;
    assert(!frame.initialize(physical,device,surface,queue,0,{640,360},world_dispatch,error,
        true,false,false,true,0.f,0,{320,180}));
    assert(allocated_images.empty() && viewed_images.empty() && allocated_memory==freed_memory);
    fail_image_after=-1;
    assert(!frame.initialize(physical,device,surface,queue,0,{640,360},world_dispatch,error,
        true,false,false,false,0.f,0,{320,180}));
    assert(swapchain_creates == swapchain_destroys && fences_created == fences_destroyed &&
        semaphores_created == semaphores_destroyed && query_creates == query_destroys && timestamp_writes >= 4);
}
