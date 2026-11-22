#include "xrEngine/stdafx.h"
#include "VulkanGameDevice.h"
#include "ShadowStability.h"
#include "ParticleVisual.h"
#include "VulkanRainRender.h"
#include "VulkanThunderboltRender.h"
#include "SceneShaders.h"
#include "ShaderAssetFingerprint.h"
#include "EmbeddedShaderFingerprint.h"
#include "ShaderModule.h"
#include "PostProcessShaders.h"
#include "TransparentDrawOrder.h"
#include "VulkanGameLightingMath.h"
#include "VulkanSettings.h"
#include "TemporalJitter.h"
#include "xrEngine/x_ray.h"
#include "xrEngine/IGame_Persistent.h"
#include "xrEngine/IGame_Level.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <cstring>
#if defined(XR_PLATFORM_ANDROID)
#include <SDL_syswm.h>
#include <android/native_window.h>
#endif

namespace xray::render::vulkan
{
bool shader_asset_fingerprint(uint64_t &result, std::string &error)
{
    FS_FileSet files;
    // Android indexes bundled engine shaders under an overlay directory.
    // file_list on $game_shaders$ only walks the game's physical directory,
    // which can be empty even though r_open resolves shaders from the overlay.
    if (FS.path_exist("$game_shaders$"))
    {
        const FS_Path *shader_path = FS.get_path("$game_shaders$");
        if (shader_path && shader_path->m_Overlay)
            FS.file_list(files, shader_path->m_Overlay, FS_ListFiles);
        FS.file_list(files, "$game_shaders$", FS_ListFiles);
    }
    PipelineCacheFingerprint fingerprint(xrCore::GetBuildCommit());
    append_embedded_shader_fingerprint(fingerprint);
    size_t bytes_hashed = 0;
    size_t count = 0;
    const auto is_shader_asset = [](std::string_view name)
    {
        const size_t dot = name.find_last_of('.');
        if (dot == std::string_view::npos) return false;
        std::string extension(name.substr(dot));
        std::transform(extension.begin(), extension.end(), extension.begin(),
            [](unsigned char value) { return static_cast<char>(std::tolower(value)); });
        return extension == ".spv" || extension == ".vs" || extension == ".hs" ||
            extension == ".ds" || extension == ".gs" || extension == ".ps" ||
            extension == ".cs" || extension == ".h" || extension == ".hlsli" ||
            extension == ".inc" || extension == ".inl" || extension == ".s";
    };
    for (const auto &file : files)
    {
        const std::string_view name(file.name.c_str());
        if (!is_shader_asset(name))
            continue;
        if (++count > 4096)
        {
            error = "too many game shader assets for pipeline cache validation";
            return false;
        }
        IReader *reader = FS.r_open("$game_shaders$", file.name.c_str());
        if (!reader)
        {
            error = "could not hash game shader asset: " + std::string(name);
            return false;
        }
        const size_t size = reader->length();
        if (size > PipelineCache::MaximumBytes - bytes_hashed)
        {
            FS.r_close(reader);
            error = "precompiled game shaders exceed pipeline cache validation limit";
            return false;
        }
        fingerprint.append(name, reader->pointer(), size);
        bytes_hashed += size;
        FS.r_close(reader);
    }
    result = fingerprint.value();
    error.clear();
    return true;
}

namespace
{

template <typename T> T proc(VkDevice device, PFN_vkGetDeviceProcAddr get, const char *name)
{
    return reinterpret_cast<T>(get(device, name));
}

VkExtent2D scaled_scene_extent(VkExtent2D display_extent, uint32_t upscaler, uint32_t percent)
{
    if (!display_extent.width || !display_extent.height || percent == 100 || (!percent && upscaler == 0))
        return display_extent;
    const double scale = percent ? double(std::clamp(percent, 50u, 100u)) / 100.0 : upscaler == 1 ? .75 : 2.0 / 3.0;
    const auto scaled = [scale](uint32_t value)
    {
        if (value < 2) return value;
        uint32_t result = static_cast<uint32_t>(std::floor(value * scale));
        result = std::max(2u, result & ~1u);
        return std::min(value, result);
    };
    return {scaled(display_extent.width), scaled(display_extent.height)};
}

uint32_t rain_shadow_target_size(const VkPhysicalDeviceLimits& limits)
{
    const uint32_t device_limit = std::max(limits.maxImageDimension2D, 64u);
    const uint32_t requested = static_cast<uint32_t>(std::clamp(dynamic_wet_surfaces_resolution, 64, 2048));
    return std::min(requested, device_limit);
}

#if defined(XR_PLATFORM_ANDROID)
bool set_android_buffer_extent(SDL_Window *window, VkExtent2D extent, std::string &error)
{
    SDL_SysWMinfo info{};
    SDL_VERSION(&info.version);
    if (!window || !SDL_GetWindowWMInfo(window, &info) || info.subsystem != SDL_SYSWM_ANDROID ||
        !info.info.android.window)
    {
        error = "Android Vulkan window has no native surface";
        return false;
    }
    if (ANativeWindow_setBuffersGeometry(info.info.android.window, static_cast<int32_t>(extent.width),
                                         static_cast<int32_t>(extent.height), 0) != 0)
    {
        error = "could not set Android Vulkan buffer resolution";
        return false;
    }
    return true;
}

bool android_presentation_extent(SDL_Window *window, VkExtent2D &extent, std::string &error)
{
    if (!set_android_buffer_extent(window, {}, error))
        return false;
    SDL_SysWMinfo info{};
    SDL_VERSION(&info.version);
    if (!SDL_GetWindowWMInfo(window, &info) || !info.info.android.window)
        return false;
    const int width = ANativeWindow_getWidth(info.info.android.window);
    const int height = ANativeWindow_getHeight(info.info.android.window);
    if (width <= 0 || height <= 0)
    {
        error = "Android Vulkan surface has an invalid native resolution";
        return false;
    }
    extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
    return true;
}
#endif

} // namespace

void VulkanGameDevice::collect_retired(uint32_t slot)
{
    textures_.set_retirement_safe(true);
    retired_.complete(slot);
    textures_.set_retirement_safe(false);
}
bool VulkanGameDevice::wait_idle()
{
    if (window_.device() && !window_.frame().device_lost() && !window_.frame().wait_idle() &&
        !window_.frame().device_lost())
        return false;
    if (gpu_particles_ && !window_.frame().device_lost())
        for (uint32_t slot = 0; slot < gpu_particle_pending_.size(); ++slot)
        {
            std::string ignored;
            gpu_particles_completed(slot, 0, this, ignored);
        }
    textures_.set_retirement_safe(true);
    retired_.drain();
    textures_.set_retirement_safe(false);
    return true;
}

bool VulkanGameDevice::record_pre_geometry(const FrameRecordingContext& frame, void* user, std::string& error)
{
    auto& owner = *static_cast<VulkanGameDevice*>(user);
    FrameRecordingContext compute_frame = frame;
    compute_frame.graphics_queue_flags = frame.command_queue_flags;
    if (frame.compute_command_buffer)
    {
        compute_frame.command_buffer = frame.compute_command_buffer;
        compute_frame.command_queue_flags = frame.compute_queue_flags;
        compute_frame.command_queue = FrameSubmissionGraph::Queue::Compute;
    }
    if (owner.smoke_compute_ && owner.smoke_noise_ && !owner.smoke_failed_)
    {
        if (frame.frame_index >= owner.smoke_frames_.size()) { error = "invalid smoke trail frame slot"; return false; }
        auto prepared = owner.smoke_controller_.prepare(Device.dwFrame, Device.fTimeDelta, Device.fTimeGlobal);
        if (!owner.smoke_noise_->record(compute_frame.command_buffer, frame.frame_index,
                Device.fTimeGlobal, 4.f, !owner.smoke_initialized_, error,
                compute_frame.command_queue_flags, frame.semaphore_handoff,
                compute_frame.submission_graph, compute_frame.compute_submission_pass,
                compute_frame.graphics_continuation_pass, compute_frame.previous_graphics_submission,
                compute_frame.command_queue, frame.command_queue_flags,
                compute_frame.request_submission_command_buffer,
                owner.frame_buffer_pool_.initialized() ? &owner.frame_buffer_pool_ : nullptr, &owner.retired_) ||
            !owner.smoke_compute_->record(compute_frame.command_buffer, frame.frame_index, prepared.emit, prepared.simulation,
                prepared.compact, !owner.smoke_initialized_, error, compute_frame.command_queue_flags,
                compute_frame.semaphore_handoff,
                compute_frame.submission_graph, compute_frame.compute_submission_pass,
                compute_frame.graphics_continuation_pass, compute_frame.previous_graphics_submission,
                compute_frame.graphics_queue_family, compute_frame.compute_queue_family,
                compute_frame.copy_queue_family, compute_frame.command_queue,
                frame.command_queue_flags, compute_frame.request_submission_command_buffer))
        {
            owner.smoke_failed_ = true;
            Msg("! [renderer-vulkan] smoke trail compute disabled: %s", error.c_str());
            return false; // The caller discards this partially recorded command buffer.
        }
        owner.smoke_frames_[frame.frame_index] = prepared;
        owner.smoke_recorded_[frame.frame_index] = true;
    }
    return record_gpu_particles(compute_frame, user, error);
}

bool VulkanGameDevice::record_gpu_particles(const FrameRecordingContext& frame, void* user, std::string& error)
{
    auto& owner = *static_cast<VulkanGameDevice*>(user);
    if (!owner.gpu_particles_ || owner.gpu_particle_failed_ || owner.gpu_particles_->programs().programs().empty())
        return true;
    std::vector<uint32_t> visible;
    for (const auto& draw : owner.particle_draws_)
    {
        if (draw.visual->getType() == 8)
            static_cast<VulkanParticleEffect*>(draw.visual)->append_gpu_visible(visible);
        else if (draw.visual->getType() == 9)
            static_cast<VulkanParticleGroup*>(draw.visual)->append_gpu_visible(visible);
    }
    const auto& programs = owner.gpu_particles_->programs().programs();
    const bool needs_collision = std::any_of(programs.begin(), programs.end(),
        [](const fg::GpuPapiProgram& program) { return (program.flags & (1u << 16)) != 0; });
    const bool needs_dynamic = std::any_of(programs.begin(), programs.end(),
        [](const fg::GpuPapiProgram& program) { return (program.flags & ((1u << 16) | (1u << 19))) == ((1u << 16) | (1u << 19)); });
    bool collision_changed = false;
    if (needs_collision && !owner.gpu_collision_.update(collision_changed, error))
    {
        Msg("! [renderer-vulkan] GPU particle level collision unavailable: %s", error.c_str());
        owner.disable_gpu_particles(); error.clear(); return true;
    }
    if (collision_changed) owner.gpu_collision_uploaded_ = false;
    bool dynamic_changed = false;
    if (!owner.gpu_collision_.update_dynamic(needs_dynamic, dynamic_changed, error))
    {
        Msg("! [renderer-vulkan] GPU particle object collision unavailable: %s", error.c_str());
        owner.disable_gpu_particles(); error.clear(); return true;
    }
    if (dynamic_changed) owner.gpu_collision_uploaded_ = false;
    const GpuParticleCollisionInput empty_collision{};
    const auto& collision = needs_collision ? owner.gpu_collision_.data() : empty_collision;
    const uint64_t collision_bytes = 48u + uint64_t(collision.nodes.size()) * sizeof(GpuParticleCollisionNode) +
        (uint64_t(collision.static_triangles.size()) + collision.dynamic_triangles.size()) * sizeof(GpuParticleCollisionTriangle) +
        uint64_t(collision.shapes.size()) * sizeof(GpuParticleCollisionShape) +
        uint64_t(collision.objects.size()) * sizeof(GpuParticleCollisionObject) +
        uint64_t(collision.blocks.size()) * sizeof(GpuParticleCollisionNode);
    const bool resize_collision = owner.gpu_particles_->collision_resize_needed(collision_bytes);
    const bool resize_draw = owner.gpu_particles_->draw_resize_needed(static_cast<uint32_t>(visible.size()));
    if (resize_collision || resize_draw)
    {
        if (!owner.window_.frame().wait_idle())
        { error = "GPU particle collision resize could not wait for the device"; owner.disable_gpu_particles(); return false; }
        for (uint32_t i = 0; i < owner.gpu_particle_pending_.size(); ++i)
            gpu_particles_completed(i, 0, &owner, error);
        if (owner.gpu_particle_failed_) { error.clear(); return true; }
        const auto& physical = owner.window_.physical();
        if ((resize_collision && !owner.gpu_particles_->reserve_collision(owner.window_.device(), owner.window_.device_proc(),
                physical.properties.limits, physical.memory, owner.buffer_upload_.buffer, collision_bytes, error)) ||
            (resize_draw && !owner.gpu_particles_->reserve_draw_buckets(owner.window_.device(), physical.properties.limits,
                physical.memory, owner.buffer_upload_.buffer, static_cast<uint32_t>(visible.size()), error)))
        {
            Msg("! [renderer-vulkan] GPU particle collision resize: %s", error.c_str());
            owner.disable_gpu_particles(); error.clear(); return true;
        }
        if (resize_collision) owner.gpu_collision_uploaded_ = false;
    }
    const float conservative_planes[24]{};
    const bool upload_collision = !owner.gpu_collision_uploaded_;
    if (!owner.gpu_particles_->record(frame.frame_index, frame.command_buffer, visible,
            upload_collision ? &collision : nullptr, conservative_planes, error,
            frame.command_queue_flags, frame.semaphore_handoff, frame.submission_graph,
            frame.compute_submission_pass, frame.graphics_prepass_pass,
            frame.graphics_continuation_pass,
            frame.previous_graphics_submission, frame.asynchronous_copy_available,
            frame.graphics_queue_family, frame.compute_queue_family, frame.copy_queue_family,
            frame.command_queue, frame.graphics_queue_flags,
            frame.request_submission_command_buffer))
    {
        Msg("! [renderer-vulkan] GPU particle simulation disabled: %s", error.c_str());
        owner.disable_gpu_particles(); error.clear(); return true;
    }
    if (owner.gpu_particle_graphics_ && !owner.gpu_particle_graphics_failed_ &&
        !owner.gpu_particle_graphics_->begin_frame(frame.frame_index, error))
    {
        Msg("! [renderer-vulkan] GPU particle graphics disabled: %s", error.c_str());
        owner.gpu_particle_graphics_failed_ = true; error.clear();
    }
    owner.gpu_particle_recorded_[frame.frame_index] = true;
    owner.gpu_collision_recorded_[frame.frame_index] = upload_collision;
    error.clear(); return true;
}
void VulkanGameDevice::gpu_particles_submitted(uint32_t slot, uint32_t, void* user)
{
    auto& owner = *static_cast<VulkanGameDevice*>(user);
    if (slot < owner.sky_probe_recorded_.size())
    {
        if (owner.sky_probe_recorded_[slot])
        {
            owner.sky_probe_next_ += owner.sky_probe_recorded_[slot];
            if (owner.sky_probe_next_ == owner.sky_probe_layout_.count)
                owner.sky_probe_last_slot_ = slot;
        }
        owner.sky_probe_recorded_[slot] = 0;
    }
    if (slot < owner.smoke_recorded_.size() && owner.smoke_recorded_[slot])
    {
        owner.smoke_controller_.commit(owner.smoke_frames_[slot]);
        owner.smoke_initialized_ = true;
        owner.smoke_recorded_[slot] = false;
    }
    if (!owner.gpu_particles_ || slot >= owner.gpu_particle_recorded_.size() || !owner.gpu_particle_recorded_[slot]) return;
    std::string error;
    if (owner.gpu_particles_->submitted(slot, error))
    {
        owner.gpu_particle_pending_[slot] = true;
        if (owner.gpu_collision_recorded_[slot]) owner.gpu_collision_uploaded_ = true;
    }
    else { owner.disable_gpu_particles(); Msg("! [renderer-vulkan] GPU particle submission: %s", error.c_str()); }
    owner.gpu_particle_recorded_[slot] = false;
    owner.gpu_collision_recorded_[slot] = false;
}
void VulkanGameDevice::disable_gpu_particles()
{
    gpu_particle_failed_ = true;
    if (gpu_particle_emitters_)
        gpu_particle_emitters_->disable();
}

bool VulkanGameDevice::gpu_particles_completed(uint32_t slot, uint32_t, void* user, std::string& error)
{
    auto& owner = *static_cast<VulkanGameDevice*>(user);
    if (slot == owner.sky_probe_last_slot_)
    {
        owner.sky_probe_last_slot_ = UINT32_MAX;
        owner.sky_probe_cpu_.resize(owner.sky_probe_layout_.count);
        if (!owner.sky_probe_results_.read(0, owner.sky_probe_cpu_.data(),
                owner.sky_probe_cpu_.size() * sizeof(SkyProbeVisibility), error))
        {
            owner.sky_probe_cpu_.clear();
            Msg("! [renderer-vulkan] sky probe readback: %s", error.c_str());
            error.clear();
        }
    }
    if (!owner.gpu_particles_ || slot >= owner.gpu_particle_pending_.size() || !owner.gpu_particle_pending_[slot])
        return true;
    GpuParticleReadback::Result result;
    if (!owner.gpu_particles_->complete(slot, result, error) ||
        !owner.gpu_particle_emitters_->apply(result.statuses, error))
    {
        Msg("! [renderer-vulkan] GPU particle readback disabled: %s", error.c_str());
        owner.disable_gpu_particles(); error.clear();
    }
    else if (result.errors)
        Msg("! [renderer-vulkan] GPU particle simulation flags=0x%x", result.errors);
    owner.gpu_particle_pending_[slot] = false;
    return true;
}

bool VulkanGameDevice::initialize(SDL_Window *window, VkExtent2D extent, std::string &error)
{
    destroy();
    fsr3_upscaler_.request_frame_generation(frame_generation != 0 && spatial_upscaler == 3);
    VkExtent2D scene_extent{};
#if defined(XR_PLATFORM_ANDROID)
    if (!extent.width || !extent.height)
        return false;
    scaled_android_window_ = window;
    requested_android_extent_ = extent;
    scene_extent = scaled_scene_extent(extent, spatial_upscaler, render_scale_percent);
    if (!android_presentation_extent(window, extent, error))
        return false;
#endif
    readback_enabled_ = window_.initialize(window, extent, true, error, true, true, true, scene_extent,
#if defined(XR_PLATFORM_ANDROID) && defined(XRAY_HAS_FSR3_ANDROID)
        frame_generation != 0 && spatial_upscaler == 3 ? Fsr3Upscaler::swapchain_create_callback : nullptr,
        frame_generation != 0 && spatial_upscaler == 3 ? Fsr3Upscaler::swapchain_destroyed_callback : nullptr,
        &fsr3_upscaler_);
#else
        nullptr, nullptr, nullptr);
#endif
    if (!readback_enabled_)
    {
        destroy();
        return false;
    }
#if defined(XR_PLATFORM_ANDROID) && defined(XRAY_HAS_FSR3_ANDROID)
    frame_generation_available = window_.frame_generation_queues().enabled && !window_.frame().hdr10_active() ? 1 : 0;
    if (!window_.frame().hdr10_active() && frame_generation != 0 && spatial_upscaler == 3 &&
        !fsr3_upscaler_.frame_generation_active())
    {
        frame_generation_available = 0;
        frame_generation = 0;
        Msg("! [renderer-vulkan] FSR frame generation requested but unavailable; queue roles or surface usage missing");
    }
#else
    frame_generation_available = 0;
#endif
    const VkDevice device = window_.device();
    const auto get = window_.device_proc();
    const auto &physical = window_.physical();
    auto &frame = window_.frame();
    frame.set_frame_generation_ui(Fsr3Upscaler::frame_generation_ui_active_callback,
        Fsr3Upscaler::register_frame_generation_ui_callback, &fsr3_upscaler_);
#if defined(XR_PLATFORM_ANDROID)
    Msg("[renderer-vulkan] resolution scene=%ux%u presentation=%ux%u", frame.extent().width,
        frame.extent().height, frame.presentation_extent().width, frame.presentation_extent().height);
#endif
    Msg("[renderer-vulkan] device.gpu name='%s' driver=0x%x api=%u.%u push-constant-limit=%u",
        physical.properties.deviceName, physical.properties.driverVersion,
        VK_VERSION_MAJOR(physical.properties.apiVersion), VK_VERSION_MINOR(physical.properties.apiVersion),
        physical.properties.limits.maxPushConstantsSize);
    if (!load_frame_dispatch(window_.instance(), window_.instance_proc(), device, get, frame_dispatch_, error))
        goto failed;

#define LOAD_TEXTURE(field, name) texture_dispatch_.field = proc<decltype(texture_dispatch_.field)>(device, get, name)
    LOAD_TEXTURE(create_buffer, "vkCreateBuffer");
    LOAD_TEXTURE(destroy_buffer, "vkDestroyBuffer");
    LOAD_TEXTURE(get_buffer_memory_requirements, "vkGetBufferMemoryRequirements");
    LOAD_TEXTURE(create_image, "vkCreateImage");
    LOAD_TEXTURE(destroy_image, "vkDestroyImage");
    LOAD_TEXTURE(get_image_memory_requirements, "vkGetImageMemoryRequirements");
    LOAD_TEXTURE(allocate_memory, "vkAllocateMemory");
    LOAD_TEXTURE(free_memory, "vkFreeMemory");
    LOAD_TEXTURE(bind_buffer_memory, "vkBindBufferMemory");
    LOAD_TEXTURE(bind_image_memory, "vkBindImageMemory");
    LOAD_TEXTURE(map_memory, "vkMapMemory");
    LOAD_TEXTURE(unmap_memory, "vkUnmapMemory");
    LOAD_TEXTURE(create_image_view, "vkCreateImageView");
    LOAD_TEXTURE(destroy_image_view, "vkDestroyImageView");
    LOAD_TEXTURE(allocate_command_buffers, "vkAllocateCommandBuffers");
    LOAD_TEXTURE(free_command_buffers, "vkFreeCommandBuffers");
    LOAD_TEXTURE(reset_command_buffer, "vkResetCommandBuffer");
    LOAD_TEXTURE(begin_command_buffer, "vkBeginCommandBuffer");
    LOAD_TEXTURE(end_command_buffer, "vkEndCommandBuffer");
    LOAD_TEXTURE(cmd_pipeline_barrier, "vkCmdPipelineBarrier");
    LOAD_TEXTURE(cmd_copy_buffer_to_image, "vkCmdCopyBufferToImage");
    LOAD_TEXTURE(queue_submit, "vkQueueSubmit");
    LOAD_TEXTURE(create_fence, "vkCreateFence");
    LOAD_TEXTURE(destroy_fence, "vkDestroyFence");
    LOAD_TEXTURE(get_fence_status, "vkGetFenceStatus");
    LOAD_TEXTURE(reset_fences, "vkResetFences");
    LOAD_TEXTURE(wait_for_fences, "vkWaitForFences");
#undef LOAD_TEXTURE
    texture_dispatch_.nvrhi_device = window_.nvrhi_device().Get();
    texture_dispatch_.fence_after_copy = physical.properties.vendorID == 0x5143; // Qualcomm

    buffer_upload_.buffer = {texture_dispatch_.create_buffer,
                             texture_dispatch_.destroy_buffer,
                             texture_dispatch_.get_buffer_memory_requirements,
                             texture_dispatch_.allocate_memory,
                             texture_dispatch_.free_memory,
                             texture_dispatch_.bind_buffer_memory,
                             texture_dispatch_.map_memory,
                             texture_dispatch_.unmap_memory};
    buffer_upload_.buffer.nvrhi_device = window_.nvrhi_device().Get();
    buffer_upload_.buffer.graphics_queue_family = physical.graphics_present_family;
    buffer_upload_.buffer.compute_queue_family = physical.compute_queue_family;
    buffer_upload_.buffer.transfer_queue_family = physical.transfer_queue_family;
    buffer_upload_.allocate_command_buffers = texture_dispatch_.allocate_command_buffers;
    buffer_upload_.free_command_buffers = texture_dispatch_.free_command_buffers;
    buffer_upload_.begin_command_buffer = texture_dispatch_.begin_command_buffer;
    buffer_upload_.end_command_buffer = texture_dispatch_.end_command_buffer;
    buffer_upload_.cmd_copy_buffer = proc<PFN_vkCmdCopyBuffer>(device, get, "vkCmdCopyBuffer");
    buffer_upload_.cmd_pipeline_barrier = texture_dispatch_.cmd_pipeline_barrier;
    buffer_upload_.queue_submit = texture_dispatch_.queue_submit;
    buffer_upload_.create_fence = texture_dispatch_.create_fence;
    buffer_upload_.destroy_fence = texture_dispatch_.destroy_fence;
    buffer_upload_.get_fence_status = texture_dispatch_.get_fence_status;
    buffer_upload_.wait_for_fences = texture_dispatch_.wait_for_fences;

    if (!frame_buffer_pool_.initialize(device, physical.memory, buffer_upload_.buffer))
    {
        error = "Vulkan frame buffer pool initialization failed";
        goto failed;
    }

    {
        const auto create_sampler = proc<PFN_vkCreateSampler>(device, get, "vkCreateSampler");
        const auto destroy_sampler = proc<PFN_vkDestroySampler>(device, get, "vkDestroySampler");
        create_sampler_ = create_sampler;
        destroy_sampler_ = destroy_sampler;
        ScenePassDispatch scene_dispatch;
        if (!load_scene_pass_dispatch(device, get, scene_dispatch, error) || !create_sampler || !destroy_sampler ||
            !buffer_upload_.cmd_copy_buffer)
        {
            if (error.empty())
                error = "Vulkan game procedures are incomplete";
            goto failed;
        }
        if (physical.features.tessellationShader)
            scene_dispatch.game_stage_flags |= VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
                VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
        scene_dispatch.max_draw_indirect_count = std::max(physical.properties.limits.maxDrawIndirectCount, 1u);
        scene_dispatch.multi_draw_indirect = physical.features.multiDrawIndirect;
        if (physical.features.geometryShader)
            scene_dispatch.game_stage_flags |= VK_SHADER_STAGE_GEOMETRY_BIT;
        std::vector<uint8_t> cached;
        uint64_t shader_fingerprint = 0;
        std::string fingerprint_error;
        const bool fingerprint_valid = shader_asset_fingerprint(shader_fingerprint, fingerprint_error);
        if (!fingerprint_valid)
            Msg("! [renderer-vulkan] pipeline cache reuse disabled: %s", fingerprint_error.c_str());
        string_path cache_path;
        if (fingerprint_valid && FS.path_exist("$app_data_root$") &&
            FS.exist(cache_path, "$app_data_root$", "vulkan-pipelines.cache"))
            if (IReader *reader = FS.r_open(cache_path))
            {
                if (reader->length() <= PipelineCache::MaximumDiskBytes)
                    cached.assign(static_cast<const uint8_t *>(reader->pointer()),
                                  static_cast<const uint8_t *>(reader->pointer()) + reader->length());
                FS.r_close(reader);
            }
        if (pipeline_cache_.initialize(device, get, physical.properties, cached, shader_fingerprint))
        {
            scene_dispatch.pipeline_cache = pipeline_cache_.handle();
            Msg("[renderer-vulkan] pipeline.cache loaded-bytes=%zu", pipeline_cache_.loaded_bytes());
        }
        if (!initialize_gbuffer_targets(error))
            goto failed;
        scene_dispatch.gbuffer_samples = targets_.samples();
        {
            std::vector<VkImage> sampled_depth;
            for (uint32_t i = 0; i < frame.image_count(); ++i)
                sampled_depth.push_back(targets_.sampled_depth_image(i));
            if (!frame.attach_postprocess_depth(sampled_depth, error))
                goto failed;
        }
        FrameImagePool* image_pool = frame.image_pool().initialized() ? &frame.image_pool() : nullptr;
        if (!sun_shadows_.initialize(device, frame.depth_format(), static_cast<uint32_t>(frame.image_count()),
                                     physical.memory, frame_dispatch_, buffer_upload_.buffer, create_sampler,
                                     destroy_sampler, error, image_pool))
            goto failed;
        if (!rain_shadows_.initialize(device, frame.depth_format(), static_cast<uint32_t>(frame.image_count()),
                                      physical.memory, frame_dispatch_, buffer_upload_.buffer, create_sampler,
                                      destroy_sampler, error, image_pool,
                                      rain_shadow_target_size(physical.properties.limits)))
            goto failed;
        if (!local_shadows_.initialize(device, frame.depth_format(), static_cast<uint32_t>(frame.image_count()),
                                       physical.memory, frame_dispatch_, buffer_upload_.buffer, create_sampler,
                                       destroy_sampler, error,
                                       physical.properties.limits.minUniformBufferOffsetAlignment, image_pool))
            goto failed;
        local_sets_.resize(frame.image_count());
        {
            std::vector<VkImageView> depth_views;
            std::vector<VkImage> depth_images;
            for (uint32_t i = 0; i < frame.image_count(); ++i)
            {
                depth_views.push_back(targets_.scene_depth_view(i));
                depth_images.push_back(targets_.scene_depth_image(i));
            }
            if (!window_.frame().attach_scene_depth(depth_views, depth_images, error))
                goto failed;
        }
        if (!window_.frame().enable_interpass(error) ||
            !water_targets_.initialize(physical.handle, device, frame.scene_format(), frame.extent(),
                                       static_cast<uint32_t>(frame.image_count()), physical.memory, frame_dispatch_,
                                       buffer_upload_.buffer, error, false, image_pool))
            goto failed;
        const ShaderModuleDispatch shaders{proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
                                           proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")};
        if (physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT)
        {
            std::vector<VkImageView> depth_views;
            std::vector<VkImage> depth_images;
            for (uint32_t i = 0; i < frame.image_count(); ++i)
            {
                depth_views.push_back(targets_.sampled_depth_view(i));
                depth_images.push_back(targets_.sampled_depth_image(i));
            }
            if (!motion_.initialize(physical.handle, device, get, frame.extent(), depth_views, depth_images, physical.memory,
                    frame_dispatch_, pipeline_cache_.handle(),
                    physical.features.shaderStorageImageExtendedFormats, error, &frame.image_pool(), true,
                    &targets_.image_sharing()))
            {
                Msg("! [renderer-vulkan] motion vectors unavailable: %s", error.c_str());
                error.clear();
            }
            if (motion_.ready())
            {
                std::vector<VkImageView> motion_views;
                std::vector<VkImage> motion_images;
                for (uint32_t i = 0; i < frame.image_count(); ++i)
                {
                    motion_views.push_back(motion_.view(i));
                    motion_images.push_back(motion_.image(i));
                }
                if (!hiz_.initialize(physical.handle, device, get, frame.extent(), depth_views, depth_images,
                        motion_views, motion_images, physical.memory, frame_dispatch_, pipeline_cache_.handle(),
                        error, &frame.image_pool(), &targets_.image_sharing()))
                {
                    Msg("! [renderer-vulkan] Hi-Z unavailable: %s", error.c_str());
                    error.clear();
                }
            }
        }
        frame_.set_hiz_pyramid(&hiz_);
        frame_.set_motion_vectors(&motion_);
        configure_engine_shader_resources(device, shaders, shader_resources_,
            physical.features.geometryShader, physical.features.tessellationShader);
        DeferredShaderFactory deferred_factory;
        Msg("[renderer-vulkan] device.deferred begin");
        if (!deferred_factory.create(device, shaders, scene_dispatch, targets_.render_pass(), frame.render_pass(),
                                     deferred_, shader_resources_, error, sun_shadows_.render_pass(),
                                     local_shadows_.render_pass()) ||
            !targets_.bind_lighting(deferred_, error))
            goto failed;
        // enable_interpass() creates a second, compatible render pass. Register it
        // on first startup as well as on swapchain recreation: transparent
        // geometry and UI are recorded there when the level uses water.
        deferred_.rebind_compatible_render_passes(targets_.render_pass(), frame.render_pass(),
                                                  sun_shadows_.render_pass(), local_shadows_.render_pass(),
                                                  frame.overlay_render_pass(), rain_shadows_.render_pass());
        deferred_.set_local_shadow_graph_passes(local_shadows_.graph_render_pass(),
            local_shadows_.graph_load_render_pass(), local_shadows_.load_render_pass());
        if (motion_.color_attachment_ready())
        {
            std::vector<VkImageView> velocity_views, geometry_depth_views;
            for (uint32_t i = 0; i < frame.image_count(); ++i)
            {
                velocity_views.push_back(motion_.view(i));
                geometry_depth_views.push_back(targets_.depth_view(i));
            }
            if (!object_velocity_.initialize(device, get, frame.extent(), frame.depth_format(),
                    velocity_views, geometry_depth_views, pipeline_cache_.handle(), error,
                    deferred_.material_layout(), deferred_.pose_layout(),
                    physical.features.multiDrawIndirect
                        ? std::max(physical.properties.limits.maxDrawIndirectCount, 1u) : 1u))
            {
                Msg("! [renderer-vulkan] object velocity unavailable: %s", error.c_str());
                error.clear();
            }
        }
        frame_.set_object_velocity_recorder(object_velocity_.ready() ? record_object_velocity : nullptr, this);
        if ((physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT) &&
            !light_grid_.initialize(device, get, frame.render_pass(), deferred_.gbuffer_layout(), physical.memory,
                                    buffer_upload_.buffer, scene_dispatch, shaders,
                                    frame_dispatch_.cmd_pipeline_barrier, error))
        {
            Msg("! [renderer-vulkan] light grid unavailable; using individual lights: %s", error.c_str());
            error.clear();
        }
        Msg("[renderer-vulkan] device.deferred ready");
        targets_.bind_sun_shadow(deferred_, sun_shadows_);
        targets_.bind_rain_shadow(deferred_, rain_shadows_);
        if (Core.Params && strstr(Core.Params, "-vk-vsm") &&
            (physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT) && !prepare_vsm_frames(error))
        {
            Msg("! [renderer-vulkan] VSM resources unavailable: %s", error.c_str());
            error.clear();
        }
        if (!create_forward_targets(error))
            goto failed;
        ShaderModule scene_vertex, scene_fragment, ui_vertex, ui_fragment;
        if (!scene_vertex.initialize(device, shaders, scene_shaders::SceneVertex, sizeof(scene_shaders::SceneVertex), error) ||
            !scene_fragment.initialize(device, shaders, scene_shaders::SceneFragment, sizeof(scene_shaders::SceneFragment), error) ||
            !ui_vertex.initialize(device, shaders, scene_shaders::UiVertex, sizeof(scene_shaders::UiVertex), error) ||
            !ui_fragment.initialize(device, shaders, scene_shaders::UiFragment, sizeof(scene_shaders::UiFragment), error) ||
            !ui_pass_.initialize(device, frame.render_pass(), scene_vertex.handle(), scene_fragment.handle(), ui_vertex.handle(), ui_fragment.handle(),
                                 scene_dispatch, error, true, frame.presentation_render_pass()) ||
            !textures_.initialize(device, window_.queue(), frame.command_pool(), physical.memory, physical.features.textureCompressionBC,
                physical.features, physical.properties,
                float(texture_anisotropy),
                texture_dispatch_,
                                  create_sampler, destroy_sampler, frame_dispatch_.device_wait_idle, error))
            goto failed;
        Msg("[renderer-vulkan] texture.compression bc=%d etc2=%d astc=%d",
            physical.features.textureCompressionBC ? 1 : 0,
            physical.features.textureCompressionETC2 ? 1 : 0,
            physical.features.textureCompressionASTC_LDR ? 1 : 0);
        for (uint32_t i = 0; i < frame.image_count(); ++i)
            if (!deferred_.water_set(i, water_targets_.refraction(i), water_targets_.distortion_field(i),
                                     targets_.sampled_depth_view(i), textures_.sampler(), targets_.sampler(),
                                     water_targets_.uniform(i), error))
                goto failed;
        ui_pass_.rebind_render_pass(frame.render_pass(), frame.overlay_render_pass(), frame.presentation_render_pass());
        if (!create_ui_scene_sets(error) || !create_postprocess(error))
            goto failed;
        Msg("[renderer-vulkan] device.passes ready water-overlay=%d images=%zu",
            frame.overlay_render_pass() != VK_NULL_HANDLE, frame.image_count());
    }
    deferred_.set_game_pipeline_request(
        [this](const std::string &vs, const std::string &ps, SurfaceMode mode, bool hud, bool skinned,
               const std::vector<std::string> &defines,
               const VkSpecializationInfo *vertex_specialization,
               const VkSpecializationInfo *hull_specialization,
               const VkSpecializationInfo *domain_specialization,
               const VkSpecializationInfo *geometry_specialization,
               const VkSpecializationInfo *pixel_specialization,
               uint64_t &variant_hash, std::string &reason) {
            return request_shader_pair(vs, ps, mode, hud, skinned, reason, defines, &variant_hash,
                vertex_specialization, hull_specialization, domain_specialization,
                geometry_specialization, pixel_specialization);
        });
    ui_.configure(
        device, physical.memory, buffer_upload_.buffer, ui_pass_,
        {[this](VkDescriptorSet set) { return textures_.retain_ui(set, ui_pass_); },
         [this](VkDescriptorSet set) { retired_.defer([this, set] { textures_.release_ui(set, ui_pass_); }); }});
    deferred_.set_frame_ready([this](uint32_t slot) { collect_retired(slot); });
    if (Core.Params && strstr(Core.Params, "-vk-gpu-particles-draw") &&
        (physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT))
    {
        auto runtime = std::make_unique<GpuParticleRuntime>();
        constexpr uint32_t root_slots = 512;
        constexpr uint32_t child_slots = 1024;
        const GpuParticleBuffers::Capacity capacity{32768, 2 * std::max(root_slots, child_slots),
            8192, 512, 8192, 8192, root_slots, 514, 48};
        if (runtime->initialize(device, get, physical.properties.limits, physical.memory, buffer_upload_.buffer,
                capacity, FrameContext::FramesInFlight, 8 * 1024 * 1024, pipeline_cache_.handle(),
                read_engine_shader, error,
                frame_buffer_pool_.initialized() ? &frame_buffer_pool_ : nullptr))
        {
            gpu_particle_emitters_ = std::make_unique<GpuParticleEmitters>(runtime->programs(), runtime->commands(), capacity.roots);
            gpu_particles_ = std::move(runtime);
            auto graphics = std::make_unique<GpuParticleGraphics>();
            if (graphics->initialize(device, get, physical.properties.limits, physical.memory, buffer_upload_.buffer,
                    deferred_.material_layout(), pipeline_cache_.handle(), FrameContext::FramesInFlight, 512,
                    read_engine_shader, error, targets_.samples(), object_velocity_.render_pass()))
                gpu_particle_graphics_ = std::move(graphics);
            else
            {
                Msg("! [renderer-vulkan] GPU particle graphics unavailable: %s", error.c_str());
                error.clear();
            }
        }
        else
        {
            Msg("! [renderer-vulkan] GPU particle simulation unavailable: %s", error.c_str());
            error.clear();
        }
    }
    trail_enabled_ = Core.Params && strstr(Core.Params, "-vk-trails");
    if (Core.Params && strstr(Core.Params, "-vk-smoke-trails") && (physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT))
    {
        auto compute = std::make_unique<SmokeTrailCompute>();
        auto noise = std::make_unique<SmokeTrailNoise>();
        if (compute->initialize(device, get, physical.properties.limits, physical.memory, buffer_upload_.buffer,
                FrameContext::FramesInFlight, pipeline_cache_.handle(), read_engine_shader, error,
                frame_buffer_pool_.initialized() ? &frame_buffer_pool_ : nullptr) &&
            noise->initialize(device, get, physical.properties.limits, physical.memory, buffer_upload_.buffer,
                32, FrameContext::FramesInFlight, pipeline_cache_.handle(), read_engine_shader, error,
                physical.graphics_present_family, physical.compute_queue_family,
                &window_.frame().image_pool()))
        {
            smoke_compute_ = std::move(compute); smoke_noise_ = std::move(noise);
            Msg("[renderer-vulkan] experimental GPU smoke trails enabled");
        }
        else { Msg("! [renderer-vulkan] GPU smoke trails unavailable: %s", error.c_str()); error.clear(); }
    }
    if (gpu_particles_ || smoke_compute_)
    {
        frame_.set_compute_recorder(record_pre_geometry, this);
        window_.frame().set_frame_observer(gpu_particles_submitted, gpu_particles_completed, this);
    }
    error.clear();
    return true;
failed:
    Msg("! [renderer-vulkan] device.initialize failed: %s", error.c_str());
    destroy();
    return false;
}

bool VulkanGameDevice::reload_game_shaders(std::string &error)
{
    const VkDevice device = window_.device();
    if (!device || !wait_idle())
    {
        error = "Vulkan shader reload needs an idle gameplay device";
        return false;
    }
    const auto get = window_.device_proc();
    const ShaderModuleDispatch shaders{proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
                                       proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")};
    DeferredShaderFactory factory;
    shader_resources_.clear();
    if (gpu_particle_graphics_)
    {
        gpu_particle_graphics_->clear_pipelines();
        gpu_particle_graphics_failed_ = false;
    }
    return factory.reload_game_pipelines(device, shaders, deferred_, shader_resources_, error);
}

bool VulkanGameDevice::create_ui_scene_sets(std::string &error)
{
    const auto &frame = window_.frame();
    const bool native_ui = frame.presentation_render_pass() != VK_NULL_HANDLE;
    if (native_ui && !ui_snapshot_targets_.initialize(window_.physical().handle, window_.device(),
            frame.composite_format(), frame.presentation_extent(), static_cast<uint32_t>(frame.image_count()),
            window_.physical().memory, frame_dispatch_, buffer_upload_.buffer, error, true))
        return false;
    ui_scene_sets_.resize(window_.frame().image_count());
    ui_depth_sets_.resize(window_.frame().image_count());
    for (uint32_t i = 0; i < ui_scene_sets_.size(); ++i)
        if (!ui_pass_.create_ui_texture_set(
                native_ui ? ui_snapshot_targets_.refraction(i) : water_targets_.refraction(i),
                targets_.sampler(), ui_scene_sets_[i],
                                            error) ||
            !ui_pass_.create_ui_texture_set(targets_.sampled_depth_view(i), targets_.sampler(), ui_depth_sets_[i],
                                            error, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
            return false;
    return true;
}

void VulkanGameDevice::release_ui_scene_sets()
{
    window_.frame().set_display_output_recorder(nullptr, nullptr);
    hdr_output_passes_.clear();
    display_grade_passes_.clear();
    for (auto &set : ui_scene_sets_)
        ui_pass_.release_ui_texture_set(set);
    ui_scene_sets_.clear();
    for (auto &set : ui_depth_sets_)
        ui_pass_.release_ui_texture_set(set);
    ui_depth_sets_.clear();
    ui_snapshot_targets_.destroy();
}

bool VulkanGameDevice::create_forward_targets(std::string &error)
{
    release_forward_targets();
    const auto &frame = window_.frame();
    forward_buffers_.resize(frame.image_count());
    for (uint32_t i = 0; i < frame.image_count(); ++i)
        if (!forward_buffers_[i].initialize(window_.device(), sizeof(ForwardLightUniform),
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                            window_.physical().memory, buffer_upload_.buffer, error) ||
            !deferred_.forward_set(i, forward_buffers_[i].handle(), error))
        {
            release_forward_targets();
            return false;
        }
    return true;
}

void VulkanGameDevice::release_forward_targets()
{
    deferred_.release_forward_sets();
    forward_buffers_.clear();
}

bool VulkanGameDevice::create_postprocess(std::string &error)
{
    postprocess_passes_.clear();
    temporal_upscale_passes_.clear();
    fsr3_upscale_passes_.clear();
    fsr3_upscaler_.reset_for_reinitialize();
    temporal_upscale_ = false;
    sun_shaft_passes_.clear();
    bloom_bright_passes_.clear();
    bloom_upsample_passes_.clear();
    display_grade_passes_.clear();
    exposure_.destroy();
    temporal_.destroy();
    auto &frame = window_.frame();
    frame.set_display_output_recorder(nullptr, nullptr);
    hdr_output_passes_.clear();
    const auto device = window_.device();
    const auto get = window_.device_proc();
    const auto &physical = window_.physical();
    if ((physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT) && frame.postprocess_mip_levels() > 1)
    {
        std::vector<VkImageView> scene_views;
        for (size_t i = 0; i < frame.image_count(); ++i)
            scene_views.push_back(frame.postprocess_view(i));
        if (!exposure_.initialize(physical.handle, device, get, scene_views, physical.memory,
                frame_dispatch_, pipeline_cache_.handle(), error, &frame.image_pool(), frame.image_sharing()))
        {
            Msg("! [renderer-vulkan] temporal exposure unavailable: %s", error.c_str());
            error.clear();
        }
        else if (!frame.attach_temporal_exposure(exposure_.image(),
                TemporalExposure::record_callback, TemporalExposure::history_valid_callback,
                &exposure_, error))
            return false;
    }
#if defined(XRAY_HAS_FSR3_ANDROID)
    if (fsr3_upscaler_.frame_generation_active() &&
        (spatial_upscaler != 3 || !motion_.ready()))
    {
        error = "FSR frame generation requires the FSR 3.1 upscaler and motion vectors";
        return false;
    }
    if (spatial_upscaler == 3 && motion_.ready())
    {
        std::vector<VkImage> color_images, depth_images, motion_images;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
        {
            color_images.push_back(frame.postprocess_image(i));
            depth_images.push_back(targets_.sampled_depth_image(i));
            motion_images.push_back(motion_.image(i));
        }
        if (fsr3_upscaler_.initialize(window_.instance(), window_.instance_proc(),
                physical.handle, device, get, physical.properties.apiVersion,
                frame_dispatch_.get_format_properties, frame.extent(), frame.presentation_extent(),
                frame.scene_format(), frame.scene_format(), frame.depth_format(), motion_.format(),
                frame.postprocess_mip_levels(), color_images, depth_images, motion_images,
                frame.image_pool(), error, frame.image_sharing()))
        {
            std::vector<VkImage> output_images;
            for (uint32_t i = 0; i < frame.image_count(); ++i)
                output_images.push_back(fsr3_upscaler_.output_image(i));
            if (!frame.attach_fsr3_upscale(output_images, motion_images,
                    Fsr3Upscaler::record_callback, Fsr3Upscaler::active_callback,
                    &fsr3_upscaler_, error))
            {
                if (fsr3_upscaler_.frame_generation_active())
                    return false;
                Msg("! [renderer-vulkan] FSR3 frame graph unavailable; using temporal fallback: %s",
                    error.c_str());
                fsr3_upscaler_.destroy();
                error.clear();
            }
            else
                Msg("[renderer-vulkan] fsr3.upscaler.ready render=%ux%u display=%ux%u",
                    frame.extent().width, frame.extent().height,
                    frame.presentation_extent().width, frame.presentation_extent().height);
        }
        else
        {
            if (fsr3_upscaler_.frame_generation_active())
                return false;
            Msg("! [renderer-vulkan] FSR3 unavailable; using temporal fallback: %s", error.c_str());
            error.clear();
        }
    }
#endif
    if (motion_.ready() && frame.postprocess_mip_levels() > 1)
    {
        std::vector<VkImageView> scene_views, motion_views, depth_views;
        std::vector<VkImage> motion_images;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
        {
            scene_views.push_back(frame.postprocess_base_view(i));
            motion_views.push_back(motion_.view(i));
            depth_views.push_back(targets_.sampled_depth_view(i));
            motion_images.push_back(motion_.image(i));
        }
        const VkExtent2D scene_extent = frame.extent(), output_extent = frame.presentation_extent();
        const bool upscale = ((spatial_upscaler == 2 ||
            (spatial_upscaler == 3 && !fsr3_upscaler_.ready()) ||
            (Core.Params && strstr(Core.Params, "-vk-temporal-upscale")))) &&
            (scene_extent.width < output_extent.width || scene_extent.height < output_extent.height);
        VkExtent2D history_extent = upscale ? output_extent : scene_extent;
        bool ready = temporal_.initialize(physical.handle, device, get, history_extent, scene_views,
                motion_views, depth_views, physical.memory, frame_dispatch_, pipeline_cache_.handle(),
                error, &frame.image_pool(), motion_.format() == VK_FORMAT_R16G16B16A16_SFLOAT, false,
                upscale, frame.image_sharing());
        if (!ready && upscale)
        {
            Msg("! [renderer-vulkan] temporal upscaling unavailable, trying scene-resolution TAA: %s", error.c_str());
            history_extent = scene_extent;
            ready = temporal_.initialize(physical.handle, device, get, history_extent, scene_views,
                motion_views, depth_views, physical.memory, frame_dispatch_, pipeline_cache_.handle(),
                error, &frame.image_pool(), motion_.format() == VK_FORMAT_R16G16B16A16_SFLOAT,
                false, false, frame.image_sharing());
        }
        temporal_upscale_ = ready && upscale && history_extent.width == output_extent.width && history_extent.height == output_extent.height;
        if (!ready)
        {
            Msg("! [renderer-vulkan] temporal resolve unavailable: %s", error.c_str());
            error.clear();
        }
        else if (!frame.attach_temporal_resolve({temporal_.image(0), temporal_.image(1)},
                motion_images, TemporalResolve::record_callback, TemporalResolve::active_callback,
                TemporalResolve::output_slot_callback, &temporal_, error, history_extent))
            return false;
    }
    if (motion_.ready())
    {
        std::vector<VkImage> motion_images;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
            motion_images.push_back(motion_.image(i));
        if (!frame.attach_postprocess_motion(motion_images, error))
            return false;
    }
    ScreenCopyDispatch dispatch;
    ShaderModuleDispatch modules{proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
                                 proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")};
    ShaderModule vertex, fragment, bright_fragment, upsample_fragment, sun_shaft_fragment;
    if (!load_screen_copy_dispatch(device, get, dispatch, error) || !dispatch.cmd_push_constants ||
        !vertex.initialize(device, modules, postprocess_shaders::Vertex, sizeof(postprocess_shaders::Vertex), error) ||
        !fragment.initialize(device, modules, postprocess_shaders::Fragment, sizeof(postprocess_shaders::Fragment),
                             error) ||
        !bright_fragment.initialize(device, modules, postprocess_shaders::BrightFragment,
            sizeof(postprocess_shaders::BrightFragment), error) ||
        !upsample_fragment.initialize(device, modules, postprocess_shaders::UpsampleFragment,
            sizeof(postprocess_shaders::UpsampleFragment), error) ||
        !sun_shaft_fragment.initialize(device, modules, postprocess_shaders::SunShaftsFragment,
            sizeof(postprocess_shaders::SunShaftsFragment), error))
    {
        if (error.empty())
            error = "Vulkan postprocess requires push constants";
        return false;
    }
    ShaderModule hdr_fragment;
    if (frame.display_output_pass())
    {
        std::vector<uint8_t> bytes;
        if (!read_engine_shader("vk/hdr10_output.ps.spv", bytes, error) ||
            !hdr_fragment.initialize_bytes(device, modules, bytes.data(), bytes.size(), error))
            return false;
    }
    for (size_t i = 0; i < frame.image_count(); ++i)
    {
        auto pass = std::make_unique<ScreenCopyPass>();
        if (!pass->initialize(device, frame.composite_render_pass(), frame.postprocess_view(i), textures_.sampler(),
                              vertex.handle(), fragment.handle(), dispatch, error))
        {
            postprocess_passes_.clear();
            return false;
        }
        pass->set_color_maps(color_map_views_[0], color_map_views_[1]);
        pass->set_bloom_view(frame.bloom_view(i));
        pass->set_depth_view(targets_.sampled_depth_view(i), targets_.sampler());
        if (motion_.ready()) pass->set_motion_view(motion_.view(static_cast<uint32_t>(i)));
        if (exposure_.ready())
            pass->set_exposure_view(exposure_.view(), exposure_.sampler());
        postprocess_passes_.push_back(std::move(pass));
        if (fsr3_upscaler_.ready())
        {
            auto fsr3_pass = std::make_unique<ScreenCopyPass>();
            if (!fsr3_pass->initialize(device, frame.composite_render_pass(),
                    fsr3_upscaler_.output_view(static_cast<uint32_t>(i)), textures_.sampler(),
                    vertex.handle(), fragment.handle(), dispatch, error))
                return false;
            fsr3_pass->set_color_maps(color_map_views_[0], color_map_views_[1]);
            fsr3_pass->set_bloom_view(frame.bloom_view(i));
            fsr3_pass->set_depth_view(targets_.sampled_depth_view(static_cast<uint32_t>(i)), targets_.sampler());
            if (motion_.ready()) fsr3_pass->set_motion_view(motion_.view(static_cast<uint32_t>(i)));
            if (exposure_.ready()) fsr3_pass->set_exposure_view(exposure_.view(), exposure_.sampler());
            fsr3_upscale_passes_.push_back(std::move(fsr3_pass));
        }
        if (temporal_upscale_)
        {
            temporal_upscale_passes_.emplace_back();
            for (uint32_t slot = 0; slot < 2; ++slot)
            {
                auto highres = std::make_unique<ScreenCopyPass>();
                if (!highres->initialize(device, frame.composite_render_pass(), temporal_.view(slot), textures_.sampler(),
                        vertex.handle(), fragment.handle(), dispatch, error)) return false;
                highres->set_color_maps(color_map_views_[0], color_map_views_[1]);
                highres->set_bloom_view(frame.bloom_view(i));
                highres->set_depth_view(targets_.sampled_depth_view(i), targets_.sampler());
                if (motion_.ready()) highres->set_motion_view(motion_.view(static_cast<uint32_t>(i)));
                if (exposure_.ready()) highres->set_exposure_view(exposure_.view(), exposure_.sampler());
                temporal_upscale_passes_.back()[slot] = std::move(highres);
            }
        }
        auto sun_shafts = std::make_unique<ScreenCopyPass>();
        if (!sun_shafts->initialize(device, frame.composite_render_pass(), frame.postprocess_view(i),
                textures_.sampler(), vertex.handle(), sun_shaft_fragment.handle(), dispatch, error, true))
            return false;
        sun_shafts->set_depth_view(targets_.sampled_depth_view(i), targets_.sampler());
        sun_shaft_passes_.push_back(std::move(sun_shafts));
        auto bright = std::make_unique<ScreenCopyPass>();
        if (!bright->initialize(device, frame.bloom_render_pass(), frame.postprocess_base_view(i),
                textures_.sampler(), vertex.handle(), bright_fragment.handle(), dispatch, error))
        {
            postprocess_passes_.clear();
            bloom_bright_passes_.clear();
            return false;
        }
        bloom_bright_passes_.push_back(std::move(bright));
        bloom_upsample_passes_.emplace_back();
        for (uint32_t mip = 0; mip + 1 < frame.bloom_mip_levels(); ++mip)
        {
            auto upsample = std::make_unique<ScreenCopyPass>();
            if (!upsample->initialize(device, frame.bloom_upsample_render_pass(),
                    frame.bloom_level_view(i, mip + 1), textures_.sampler(), vertex.handle(),
                    upsample_fragment.handle(), dispatch, error, true))
                return false;
            bloom_upsample_passes_.back().push_back(std::move(upsample));
        }
        if (frame.presentation_render_pass())
        {
            auto display = std::make_unique<ScreenCopyPass>();
            if (!display->initialize(device, frame.presentation_render_pass(), ui_snapshot_targets_.refraction(i),
                    textures_.sampler(), vertex.handle(), fragment.handle(), dispatch, error))
                return false;
            display->set_bloom_view(frame.bloom_view(i));
            display_grade_passes_.push_back(std::move(display));
        }
        if (hdr_fragment.handle())
        {
            auto output = std::make_unique<ScreenCopyPass>();
            if (!output->initialize(device, frame.display_output_pass(), frame.composite_view(static_cast<uint32_t>(i)),
                    textures_.sampler(), vertex.handle(), hdr_fragment.handle(), dispatch, error))
                return false;
            hdr_output_passes_.push_back(std::move(output));
        }
    }
    if (!hdr_output_passes_.empty())
        frame.set_display_output_recorder(record_hdr_output, this);
    return true;
}

void VulkanGameDevice::record_hdr_output(const FrameRecordingContext& frame, void* user)
{
    auto& self = *static_cast<VulkanGameDevice*>(user);
    auto& pass = *self.hdr_output_passes_[frame.image_index];
    self.window_.frame().update_hdr_metadata(hdr_paper_white_nits, hdr_peak_nits, hdr_black_nits);
    pass.set_hdr_calibration(hdr_paper_white_nits, hdr_peak_nits, hdr_black_nits, hdr_calibration_pattern != 0);
    pass.record(frame);
}

void VulkanGameDevice::set_postprocess(const PostProcessConstants &params, std::string first, std::string second)
{
    postprocess_params_ = params;
    postprocess_params_.grade[3] = tonemap_enabled ? tonemap_amount : 0.f;
    postprocess_params_.color_map[2] = tonemap_middlegray;
    color_map_names_[0] = std::move(first);
    color_map_names_[1] = std::move(second);
}

bool VulkanGameDevice::refresh_color_maps(std::string &error)
{
    if (color_map_names_[0] == active_map_names_[0] && color_map_names_[1] == active_map_names_[1])
        return true;
    // Descriptor writes and texture eviction must wait for previous submissions.
    if (!wait_idle())
    {
        error = "could not wait for previous frames before changing postprocess color maps";
        return false;
    }

    VkImageView replacements[2]{};
    for (size_t i = 0; i < 2; ++i)
        if (!color_map_names_[i].empty() && !textures_.environment(color_map_names_[i], replacements[i], error))
        {
            for (auto view : replacements)
                if (view)
                    textures_.release_environment(view);
            return false;
        }
    if (!textures_.finish_uploads())
    {
        for (auto view : replacements)
            if (view)
                textures_.release_environment(view);
        error = "could not upload Vulkan postprocess color maps";
        return false;
    }
    for (auto &pass : postprocess_passes_)
        pass->set_color_maps(replacements[0], replacements[1]);
    for (auto& pair : temporal_upscale_passes_)
        for (auto& pass : pair) pass->set_color_maps(replacements[0], replacements[1]);
    for (auto& pass : fsr3_upscale_passes_)
        pass->set_color_maps(replacements[0], replacements[1]);
    for (auto view : color_map_views_)
        if (view)
            textures_.release_environment(view);
    std::copy_n(replacements, 2, color_map_views_);
    std::copy_n(color_map_names_, 2, active_map_names_);
    return true;
}

void VulkanGameDevice::detach_color_maps_for_asset_reload()
{
    for (auto &pass : postprocess_passes_)
        pass->set_color_maps(VK_NULL_HANDLE, VK_NULL_HANDLE);
    for (auto &pair : temporal_upscale_passes_)
        for (auto &pass : pair)
            pass->set_color_maps(VK_NULL_HANDLE, VK_NULL_HANDLE);
    for (auto &pass : fsr3_upscale_passes_)
        pass->set_color_maps(VK_NULL_HANDLE, VK_NULL_HANDLE);
    for (auto view : color_map_views_)
        if (view)
            textures_.release_environment(view);
    color_map_views_[0] = color_map_views_[1] = VK_NULL_HANDLE;
    active_map_names_[0].clear();
    active_map_names_[1].clear();
}

bool VulkanGameDevice::refresh_color_maps_after_asset_reload(std::string &error)
{
    return refresh_color_maps(error);
}

bool VulkanGameDevice::update_rain_shadow_resolution(std::string& error)
{
    if (!window_.device() || !window_.frame().image_count())
    {
        error = "rain shadow resize requires an initialized Vulkan frame";
        return false;
    }
    const auto& physical = window_.physical();
    const uint32_t requested = rain_shadow_target_size(physical.properties.limits);
    if (requested == rain_shadows_.size())
    {
        error.clear();
        return true;
    }
    if (!wait_idle())
    {
        error = "could not wait for Vulkan frames before rain shadow resize";
        return false;
    }
    FrameImagePool* image_pool = window_.frame().image_pool().initialized() ?
        &window_.frame().image_pool() : nullptr;
    SunShadowTargets replacement;
    if (!replacement.initialize(window_.device(), window_.frame().depth_format(),
            static_cast<uint32_t>(window_.frame().image_count()), physical.memory, frame_dispatch_,
            buffer_upload_.buffer, create_sampler_, destroy_sampler_, error, image_pool, requested))
    {
        Msg("! [renderer-vulkan] rain shadow resolution %u unavailable; retaining %u: %s",
            requested, rain_shadows_.size(), error.c_str());
        dynamic_wet_surfaces_resolution = static_cast<int>(rain_shadows_.size());
        error.clear();
        return true;
    }
    rain_shadows_.swap(replacement);
    deferred_.set_rain_shadow_pass(rain_shadows_.render_pass());
    targets_.bind_rain_shadow(deferred_, rain_shadows_);
    Msg("[renderer-vulkan] rain shadow resolution=%u", rain_shadows_.size());
    error.clear();
    return true;
}

void VulkanGameDevice::queue_model(GpuModel &model, IKinematics *skeleton, const float (&mvp)[16], bool hud,
                                   float sort_distance, const void *instance, float lod, const Fmatrix *world,
                                   const float *hemi_faces, bool shadow_only, const Fsphere *shadow_bounds,
                                   float opacity)
{
    ModelDraw draw;
    draw.model = &model;
    if (shadow_bounds && std::isfinite(shadow_bounds->P.x) && std::isfinite(shadow_bounds->P.y) &&
        std::isfinite(shadow_bounds->P.z) && std::isfinite(shadow_bounds->R) && shadow_bounds->R > 0.f)
    {
        draw.shadow_sphere = {shadow_bounds->P.x, shadow_bounds->P.y, shadow_bounds->P.z, shadow_bounds->R};
        draw.has_shadow_sphere = true;
    }
    draw.instance = instance;
    draw.skeleton = skeleton;
    std::copy_n(mvp, 16, draw.mvp.data());
    if (!hud && temporal_jitter_active_)
        apply_temporal_jitter(draw.mvp.data(), temporal_jitter_ndc_);
    const Fmatrix &rotation = world ? *world : Fidentity;
    std::memcpy(draw.world.data(), &rotation, sizeof(rotation));
    if (!hud && !shadow_only)
        draw.has_previous_mvp = object_motion_.stage(
            {ObjectMotionHistory::Kind::Model, instance, &model, 0},
            draw.mvp, draw.previous_mvp, &draw.world, &draw.moved);
    draw.normal_rows = {rotation.i.x, rotation.j.x, rotation.k.x, -1.f,         rotation.i.y, rotation.j.y,
                        rotation.k.y, 0.f,          rotation.i.z, rotation.j.z, rotation.k.z, 0.f};
    if (hemi_faces && std::all_of(hemi_faces, hemi_faces + 6, [](float value) { return std::isfinite(value); }))
    {
        draw.normal_rows[3] = std::clamp(hemi_faces[VulkanShaderHemiFaceOrder[0]], 0.f, 1.f);
        draw.normal_rows[7] = std::clamp(hemi_faces[VulkanShaderHemiFaceOrder[1]], 0.f, 1.f);
        draw.normal_rows[11] = std::clamp(hemi_faces[VulkanShaderHemiFaceOrder[2]], 0.f, 1.f);
        for (unsigned face = 0; face < 3; ++face)
            draw.hemi_negative[face] = std::clamp(hemi_faces[VulkanShaderHemiFaceOrder[face + 3]], 0.f, 1.f);
    }
    draw.sort_distance = std::isfinite(sort_distance) ? std::max(sort_distance, 0.f) : 0.f;
    draw.lod = lod;
    draw.opacity = std::clamp(opacity, 0.f, 1.f);
    draw.hud = hud;
    if (!shadow_only)
        model_draws_.push_back(draw);
    if (!hud)
    {
        const auto range = shadow_draw_lookup_.equal_range(draw.model);
        for (auto it = range.first; it != range.second; ++it)
        {
            const auto &other = shadow_model_draws_[it->second];
            if (other.skeleton == draw.skeleton && other.world == draw.world)
                return;
        }
        shadow_draw_lookup_.emplace(draw.model, shadow_model_draws_.size());
        shadow_model_draws_.push_back(draw);
    }
}

void VulkanGameDevice::queue_particle(IRenderVisual *visual, const float (&mvp)[16], const Fvector &right,
                                      const Fvector &up, bool hud, float distance)
{
    ParticleDraw draw;
    draw.visual = visual;
    std::copy_n(mvp, 16, draw.mvp.data());
    if (!hud && temporal_jitter_active_)
        apply_temporal_jitter(draw.mvp.data(), temporal_jitter_ndc_);
    draw.right = right;
    draw.up = up;
    draw.hud = hud;
    draw.sort_distance = std::isfinite(distance) ? std::max(distance, 0.f) : 0.f;
    particle_draws_.push_back(draw);
}

void VulkanGameDevice::discard_model_draws(const void *instance)
{
    object_motion_.forget_instance(instance);
    particle_draws_.erase(std::remove_if(particle_draws_.begin(), particle_draws_.end(),
                                         [instance](const ParticleDraw &draw) { return draw.visual == instance; }),
                          particle_draws_.end());
    model_draws_.erase(std::remove_if(model_draws_.begin(), model_draws_.end(),
                                      [instance](const ModelDraw &draw) { return draw.instance == instance; }),
                       model_draws_.end());
    shadow_model_draws_.erase(std::remove_if(shadow_model_draws_.begin(), shadow_model_draws_.end(),
                                             [instance](const ModelDraw &draw) { return draw.instance == instance; }),
                              shadow_model_draws_.end());
    shadow_draw_lookup_.clear();
    for (size_t i = 0; i < shadow_model_draws_.size(); ++i)
        shadow_draw_lookup_.emplace(shadow_model_draws_[i].model, i);
    const size_t previous_level_count = level_draws_.size();
    level_draws_.erase(std::remove_if(level_draws_.begin(), level_draws_.end(),
                                      [instance](const LevelDraw &draw) { return draw.instance == instance; }),
                       level_draws_.end());
    if (level_draws_.size() == previous_level_count)
        return;
    // Erasure shifts vector indices. Rebuild before another submission can
    // consult the deduplication table in this same frame.
    level_draw_lookup_.clear();
    for (size_t index = 0; index < level_draws_.size(); ++index)
        level_draw_lookup_.emplace(level_draws_[index].index, index);
}

void VulkanGameDevice::queue_level_visual(uint32_t index, const float (&mvp)[16], bool hud, float sort_distance,
                                          const void *instance, float lod, const Fmatrix* world)
{
    std::array<float, 16> render_mvp{};
    std::copy_n(mvp, 16, render_mvp.begin());
    if (!hud && temporal_jitter_active_)
        apply_temporal_jitter(render_mvp.data(), temporal_jitter_ndc_);
    const auto existing = level_draw_lookup_.equal_range(index);
    for (auto it = existing.first; it != existing.second; ++it)
    {
        const LevelDraw &draw = level_draws_[it->second];
        if (draw.hud == hud && draw.instance == instance && draw.mvp == render_mvp)
            return;
    }

    LevelDraw draw;
    draw.index = index;
    draw.instance = instance;
    draw.mvp = render_mvp;
    if (!hud)
    {
        ObjectMotionHistory::Matrix world_matrix{};
        if (world) std::memcpy(world_matrix.data(), world, sizeof(world_matrix));
        draw.has_previous_mvp = object_motion_.stage(
            {ObjectMotionHistory::Kind::LevelVisual, instance, nullptr, index},
            draw.mvp, draw.previous_mvp, &world_matrix, &draw.moved);
    }
    draw.sort_distance = std::isfinite(sort_distance) ? std::max(sort_distance, 0.f) : 0.f;
    draw.lod = lod;
    draw.hud = hud;
    level_draws_.push_back(draw);
    level_draw_lookup_.emplace(index, level_draws_.size() - 1);
}

void VulkanGameDevice::begin_frame()
{
    if (frame_buffer_pool_.initialized())
        frame_buffer_pool_.tick();
    object_motion_.begin_frame();
    const bool fsr3_selected = spatial_upscaler == 3 && fsr3_upscaler_.ready();
    temporal_jitter_active_ = fsr3_selected ||
        ((antialiasing == 2 || spatial_upscaler == 2) && temporal_.ready());
    temporal_jitter_ndc_ = {};
    if (fsr3_selected)
    {
        float jitter_x = 0.f, jitter_y = 0.f;
        const VkExtent2D extent = window_.frame().extent();
        if (extent.width && extent.height &&
            fsr3_upscaler_.jitter_offset(temporal_sample_index_, jitter_x, jitter_y))
        {
            temporal_jitter_ndc_ = {2.f * jitter_x / float(extent.width),
                -2.f * jitter_y / float(extent.height)};
            fsr3_upscaler_.set_jitter(jitter_x, jitter_y);
        }
        else
            temporal_jitter_active_ = false;
    }
    else if (temporal_jitter_active_)
        temporal_jitter_ndc_ = temporal_jitter_ndc(temporal_sample_index_, window_.frame().extent());
    temporal_.set_jitter(temporal_jitter_ndc_[0] * .5f, -temporal_jitter_ndc_[1] * .5f);
    discard_scene_draws();
    light_snapshots_.clear();
    ui_.reset_frame();
    weather_set_ = VK_NULL_HANDLE;
}

VkDescriptorSet VulkanGameDevice::glow_texture(const std::string &name)
{
    if (name.empty())
        return VK_NULL_HANDLE;
    const auto found = glow_textures_.find(name);
    if (found != glow_textures_.end())
        return found->second;
    VkDescriptorSet set = VK_NULL_HANDLE;
    std::string error;
    if (!textures_.ui(name, ui_pass_, set, error))
    {
        Msg("! [renderer-vulkan] glow texture '%s': %s", name.c_str(), error.c_str());
        return VK_NULL_HANDLE;
    }
    glow_textures_.emplace(name, set);
    return set;
}

void VulkanGameDevice::release_trail_texture()
{
    if (trail_texture_) textures_.release_environment(trail_texture_);
    trail_texture_ = VK_NULL_HANDLE;
}

VkImageView VulkanGameDevice::light_projection(const std::string &name)
{
    if (failed_light_projections_.count(name))
        return light_projection("");
    auto found = light_projections_.find(name);
    if (found != light_projections_.end())
        return found->second;
    VkImageView view{};
    std::string error;
    if (!textures_.projection_texture(name, view, error))
    {
        Msg("! [renderer-vulkan] spot texture '%s': %s", name.c_str(), error.c_str());
        failed_light_projections_.insert(name);
        return light_projection("");
    }
    light_projections_.emplace(name, view);
    return view;
}

void VulkanGameDevice::release_level_glows()
{
    for (const auto &entry : light_projections_)
        if (!entry.first.empty() && entry.second)
            textures_.release_environment(entry.second);
    local_shadows_.invalidate_cache();
    light_projections_.clear();
    indirect_projections_.clear();
    failed_light_projections_.clear();
    local_projection_views_.clear();
    local_bound_projections_.clear();

    if (glow_textures_.empty())
        return;
    R_ASSERT2(wait_idle(), "Vulkan glow descriptors require idle GPU frames");
    for (const auto &[name, set] : glow_textures_)
        textures_.release_ui(set, ui_pass_);
    glow_textures_.clear();
}

void VulkanGameDevice::queue_weather(VkDescriptorSet set, const WeatherLighting &lighting, const float (&fog_color)[3],
                                     float fog_near, float fog_far)
{
    weather_set_ = set;
    weather_lighting_ = lighting;
    std::copy_n(fog_color, 3, forward_lighting_.fog_color_start);
    forward_lighting_.fog_color_start[3] = fog_near;
    forward_lighting_.fog_end_camera[0] = fog_far;
}

void VulkanGameDevice::queue_rain(VulkanRainRender &rain)
{
    rain_draws_.push_back(&rain);
}

void VulkanGameDevice::queue_thunderbolt(VulkanThunderboltRender &bolt)
{
    thunderbolt_draws_.push_back(&bolt);
}

void VulkanGameDevice::discard_rain(const VulkanRainRender *rain)
{
    rain_draws_.erase(std::remove(rain_draws_.begin(), rain_draws_.end(), rain), rain_draws_.end());
}

void VulkanGameDevice::discard_thunderbolt(const VulkanThunderboltRender *bolt)
{
    thunderbolt_draws_.erase(std::remove(thunderbolt_draws_.begin(), thunderbolt_draws_.end(), bolt),
                             thunderbolt_draws_.end());
}

bool VulkanGameDevice::take_screenshot(std::vector<uint8_t> &pixels, VkExtent2D &extent, VkFormat &format)
{
    if (!screenshot_ready_)
        return false;
    screenshot_ready_ = false;
    extent = screenshot_extent_;
    format = window_.frame().format();
    pixels.resize(size_t(extent.width) * extent.height * 4);
    std::string error;
    if (!screenshot_buffer_.read(0, pixels.data(), pixels.size(), error))
    {
        Msg("! [renderer-vulkan] screenshot: %s", error.c_str());
        return false;
    }
    return true;
}

void VulkanGameDevice::discard_scene_draws()
{
    model_draws_.clear();
    shadow_model_draws_.clear();
    shadow_draw_lookup_.clear();
    particle_draws_.clear();
    rain_draws_.clear();
    thunderbolt_draws_.clear();
    level_draws_.clear();
    level_draw_lookup_.clear();
    current_level_ = nullptr;
}

bool VulkanGameDevice::validate_frame_recording(void *user, std::string& error)
{
    const auto& owner = *static_cast<VulkanGameDevice*>(user);
    if (!owner.level_recorded_)
        error = owner.level_error_.empty() ? "Vulkan level visual recording failed" : owner.level_error_;
    else if (!owner.models_recorded_)
        error = owner.model_error_.empty() ? "Vulkan model/shadow recording failed" : owner.model_error_;
    else if (!owner.local_lights_recorded_)
        error = owner.model_error_.empty() ? "Vulkan local light recording failed" : owner.model_error_;
    else if (!owner.ui_recorded_)
        error = owner.ui_error_.empty() ? "Vulkan UI recording failed" : owner.ui_error_;
    else
        return true;
    return false;
}

bool VulkanGameDevice::render(const GpuLevel &level, const float (&mvp)[16], const DeferredLight &light,
                              FrameStatus &status, std::string &error, bool render_world, bool clear_target,
                              const DeferredEnvironment *environment)
{
    float render_mvp[16];
    std::copy_n(mvp, 16, render_mvp);
    if (render_world && temporal_jitter_active_)
        apply_temporal_jitter(render_mvp, temporal_jitter_ndc_);
    ui_recorded_ = true;
    models_recorded_ = true;
    local_lights_recorded_ = true;
    level_recorded_ = true;
    level_error_.clear();
    current_level_ = &level;
    std::copy_n(render_mvp, 16, scene_mvp_.begin());
    ui_error_.clear();
    model_error_.clear();
    if (!update_rain_shadow_resolution(error))
        return false;
    if (!refresh_color_maps(error))
        return false;
    auto frame_postprocess = postprocess_params_;
    const bool native_ui = window_.frame().presentation_render_pass() != VK_NULL_HANDLE;
    bool grade_display = false;
    if (native_ui)
    {
        PostProcessConstants display;
        display.bloom[0] = 0.f;
        for (size_t i = 0; i < 3; ++i)
        {
            display.grade[i] = frame_postprocess.grade[i];
            grade_display |= display.grade[i] != 1.f;
            frame_postprocess.grade[i] = 1.f;
        }
        for (auto &pass : display_grade_passes_)
            pass->set_constants(display);
    }
    // Resolve the saved console value each frame. Menus and loading screens
    // bypass AA, and changing AA needs neither vid_restart nor a new pipeline.
    const bool fsr3_active = render_world && temporal_jitter_active_ &&
        spatial_upscaler == 3 && fsr3_upscaler_.ready();
    fsr3_upscaler_.set_active(fsr3_active);
    temporal_.set_active(render_world && temporal_jitter_active_ && !fsr3_active);
    frame_postprocess.color_map[3] = render_world && !temporal_jitter_active_ &&
        (antialiasing == 1 || antialiasing == 2 || spatial_upscaler == 2 || spatial_upscaler == 3) ? 1.f : 0.f;
    const VkExtent2D scene_size = window_.frame().extent();
    const VkExtent2D output_size = window_.frame().presentation_extent();
    if (render_world && spatial_upscaler == 1 && !(temporal_upscale_ && temporal_.active()) &&
        (scene_size.width < output_size.width || scene_size.height < output_size.height))
        frame_postprocess.color_map[3] += 2.f;
    if (render_world && motion_.ready())
        frame_postprocess.color_map[3] += 0.5f * std::clamp(motion_blur_strength, 0.f, 1.f);
    const float bloom_delta = std::isfinite(Device.fTimeDelta) ? std::clamp(Device.fTimeDelta, 0.f, 1.f) : 0.f;
    const float legacy_bloom_target = std::clamp(legacy_bloom_speed * bloom_delta, 0.f, 4.f);
    legacy_bloom_factor_ = 0.9f * legacy_bloom_factor_ + 0.1f * legacy_bloom_target;
    frame_postprocess.bloom[0] = render_world ? bloom_intensity * legacy_bloom_factor_ : 0.f;
    // The legacy RGB-sum cutoff is relative to its stock value in the HDR bloom prefilter.
    frame_postprocess.bloom[1] = std::clamp(bloom_threshold +
        (legacy_bloom_threshold - default_legacy_bloom_threshold), 0.f, 4.f);
    frame_postprocess.bloom[2] = bloom_mip;
    frame_postprocess.bloom[3] = bloom_fast ? -bloom_knee : bloom_knee;
    if (render_world && dof_strength > 0.f)
    {
        Fmatrix projection;
        std::memcpy(&projection, render_mvp, sizeof(projection));
        Fvector focus_point;
        focus_point.mad(Device.vCameraPosition, Device.vCameraDirection, std::max(dof_focus, .2f));
        Fvector4 projected;
        projection.transform(projected, focus_point);
        frame_postprocess.tint[3] = std::clamp(projected.z, 0.f, 1.f);
    }
    frame_postprocess.add[3] = render_world ? dof_strength : 0.f;
    sun_shafts_active_ = false;
    constexpr float minimum_sun_shafts_intensity = 0.0001f;
    const float environment_sun_shafts_intensity = g_pGamePersistent ?
        std::clamp(g_pGamePersistent->Environment().CurrentEnv.m_fSunShaftsIntensity, 0.f, 1.f) : 0.f;
    const float sun_shaft_intensity = std::clamp(sun_shafts_strength, 0.f, 1.f) *
        environment_sun_shafts_intensity;
    if (render_world && sun_shaft_intensity >= minimum_sun_shafts_intensity &&
        std::max({light.color[0], light.color[1], light.color[2]}) > 0.001f)
    {
        Fvector sun_direction;
        sun_direction.set(-light.direction_ambient[0], -light.direction_ambient[1],
            -light.direction_ambient[2]);
        sun_direction.normalize_safe();
        Fvector sun_point;
        sun_point.mad(Device.vCameraPosition, sun_direction, 100.f);
        Fmatrix projection;
        std::memcpy(&projection, render_mvp, sizeof(projection));
        const float clip_w = sun_point.x * projection._14 + sun_point.y * projection._24 +
            sun_point.z * projection._34 + projection._44;
        if (clip_w > 0.001f)
        {
            Fvector4 projected;
            projection.transform(projected, sun_point);
            PostProcessConstants shafts{};
            shafts.tint[0] = 0.5f * (projected.x + 1.f);
            shafts.tint[1] = 0.5f * (1.f - projected.y);
            shafts.tint[2] = sun_shaft_intensity;
            const uint32_t shaft_quality = std::min<uint32_t>(quality_preset,
                static_cast<uint32_t>(std::size(PresetSunShaftSamples) - 1));
            shafts.effect[0] = static_cast<float>(PresetSunShaftSamples[shaft_quality]);
            for (int channel = 0; channel < 3; ++channel)
                shafts.add[channel] = std::clamp(light.color[channel], 0.f, 2.f);
            sun_shafts_active_ = shafts.tint[0] >= -.5f && shafts.tint[0] <= 1.5f &&
                shafts.tint[1] >= -.5f && shafts.tint[1] <= 1.5f;
            for (auto& pass : sun_shaft_passes_)
                pass->set_constants(shafts);
        }
    }
    frame_postprocess.gray_weights[3] =
        exposure_.ready() && render_world && tonemap_enabled ? 1.f : 0.f;
    exposure_.set_parameters(tonemap_middlegray, tonemap_low_luminance,
        Device.fTimeDelta, tonemap_adaptation_speed);
    for (auto &pass : postprocess_passes_)
        pass->set_constants(frame_postprocess);
    for (auto& pair : temporal_upscale_passes_)
        for (auto& pass : pair) pass->set_constants(frame_postprocess);
    for (auto& pass : fsr3_upscale_passes_)
        pass->set_constants(frame_postprocess);
    for (auto &pass : bloom_bright_passes_)
        pass->set_constants(frame_postprocess);
    for (auto &image_passes : bloom_upsample_passes_)
        for (auto &pass : image_passes)
            pass->set_constants(frame_postprocess);
    const float weather_blend = weather_lighting_.light.color[3];
    const float cloud_params = weather_lighting_.light.grade[3];
    weather_lighting_.light = light;
    weather_lighting_.light.color[3] = weather_blend;
    weather_lighting_.light.grade[3] = cloud_params;
    last_ui_draw_calls_ = ui_.draw_calls();
    last_ui_triangles_ = ui_.triangles();
    if (Device.dwPrecacheFrame &&
        (Device.dwPrecacheFrame == Device.dwPrecacheTotal || Device.dwPrecacheFrame % 10 == 0))
        Msg("[renderer-vulkan] frame.begin precache=%u world=%d clear=%d water=%d ui-draws=%u level-draws=%zu "
            "model-draws=%zu particles=%zu ambient=%.3f sun=(%.3f,%.3f,%.3f)",
            Device.dwPrecacheFrame, render_world ? 1 : 0, clear_target ? 1 : 0, level.has_water() ? 1 : 0,
            last_ui_draw_calls_, level_draws_.size(), model_draws_.size(), particle_draws_.size(),
            light.direction_ambient[3], light.color[0], light.color[1], light.color[2]);
    screenshot_ready_ = false;
    if (screenshot_requested_ && readback_enabled_)
    {
        screenshot_extent_ = window_.frame().presentation_extent();
        const VkDeviceSize bytes = VkDeviceSize(screenshot_extent_.width) * screenshot_extent_.height * 4;
        if (screenshot_buffer_.size() < bytes)
        {
            if (!wait_idle() || !screenshot_buffer_.initialize(
                                    window_.device(), bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                    window_.physical().memory, buffer_upload_.buffer, error))
                return false;
        }
    }
    SunShadowUniform sun_uniform{};
    sun_uniform.fog_plane[3] = 1.f;
    if (render_world && environment)
    {
        const float fog_near = environment->fog_near;
        const float fog_far = environment->fog_far;
        if (std::isfinite(fog_near) && std::isfinite(fog_far) && fog_far > fog_near + 0.01f)
        {
            const Fmatrix &matrix = Device.mFullTransform;
            float plane[4]{-(matrix._14 + matrix._13), -(matrix._24 + matrix._23),
                           -(matrix._34 + matrix._33), -(matrix._44 + matrix._43)};
            const float normal_length = std::sqrt(plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
            if (std::isfinite(normal_length) && normal_length > 1e-6f)
            {
                const float inverse_range = 1.f / (fog_far - fog_near);
                const float plane_scale = -1.f / normal_length;
                for (float &coefficient : plane)
                    coefficient *= plane_scale;
                sun_uniform.fog_plane[0] = -plane[0] * inverse_range;
                sun_uniform.fog_plane[1] = -plane[1] * inverse_range;
                sun_uniform.fog_plane[2] = -plane[2] * inverse_range;
                sun_uniform.fog_plane[3] = 1.f - (plane[3] - fog_near) * inverse_range;
                for (unsigned channel = 0; channel < 3; ++channel)
                    sun_uniform.fog_color[channel] = std::isfinite(environment->fog_color[channel])
                        ? environment->fog_color[channel] : 0.f;
            }
        }
    }
    std::copy_n(sun_uniform.fog_plane, 4, forward_lighting_.fog_plane);
    std::copy_n(sun_uniform.fog_color, 3, forward_lighting_.fog_color_start);
    TreeWindParameters tree_wind{};
    if (g_pGamePersistent)
    {
        const auto& env = g_pGamePersistent->Environment().CurrentEnv;
        constexpr float inverse_turn = 1.f / 6.283185307179586f;
        if (std::isfinite(env.m_fTreeRotation) && std::abs(env.m_fTreeRotation) > 1e-5f &&
            std::isfinite(env.m_fTreeAmplitude) && std::isfinite(env.m_fTreeSpeed) &&
            std::isfinite(env.m_fTreeWave.x) && std::isfinite(env.m_fTreeWave.y) &&
            std::isfinite(env.m_fTreeWave.z) && std::isfinite(Device.fTimeGlobal))
        {
            const float rotation = 6.283185307179586f * Device.fTimeGlobal / env.m_fTreeRotation;
            tree_wind.direction[0] = std::sin(rotation) * env.m_fTreeAmplitude;
            tree_wind.direction[1] = std::cos(rotation) * env.m_fTreeAmplitude;
            tree_wind.wave[0] = env.m_fTreeWave.x * inverse_turn;
            tree_wind.wave[1] = env.m_fTreeWave.y * inverse_turn;
            tree_wind.wave[2] = env.m_fTreeWave.z * inverse_turn;
            tree_wind.wave[3] = Device.fTimeGlobal * env.m_fTreeSpeed * inverse_turn;
        }
    }
    deferred_.set_tree_wind(tree_wind);
    current_tree_wind_ = tree_wind;
    sun_uniform.options[3] = render_world ? ao_strength : 0.f;
    const uint32_t ao_quality = std::min<uint32_t>(quality_preset,
        static_cast<uint32_t>(std::size(PresetAODirections) - 1));
    sun_uniform.ao_settings[0] = static_cast<float>(PresetAODirections[ao_quality]);
    if (render_world && Core.Params)
    {
        // The old argument remains accepted for existing custom launchers.
        const char *mode = strstr(Core.Params, "-vk-light-debug=");
        if (mode)
        {
            const char *value = mode + sizeof("-vk-light-debug=") - 1;
            char *end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end != value && parsed >= 1 && parsed <= 9 &&
                (*end == '\0' || std::isspace(static_cast<unsigned char>(*end))))
                sun_uniform.options[1] = static_cast<float>(parsed);
        }
        else if (strstr(Core.Params, "-vk-light-diagnostic"))
            sun_uniform.options[1] = 1.f;
    }
    for (int channel = 0; channel < 3; ++channel)
    {
        const auto valid = [](float value) { return std::isfinite(value) ? std::max(value, 0.f) : 0.f; };
        const float ambient =
            environment ? valid(environment->ambient_color[channel]) : valid(light.direction_ambient[3]);
        // R2 passes twice the weather ambient to hmodel.
        sun_uniform.ambient_color[channel] = std::max(ambient * 2.f * ambient_scale, .001f);
        sun_uniform.hemi_color[channel] = environment ? valid(environment->hemi_color[channel]) : valid(light.grade[3]);
        sun_uniform.hemi_color[channel] *= hemisphere_scale;
    }
    if (sun_uniform.options[1] > .5f && (Device.dwFrame < 3 || Device.dwFrame % 300 == 0))
        Msg("[renderer-vulkan] lighting.diagnostic frame=%u mode=%u "
            "panels=1:base,sun,baked,albedo;6:env,hemi-one,sun-one,both;7:old-flat "
            "shadow-size=%u bias=%.4f "
            "forward=single-base-only quad=opaque-only",
            Device.dwFrame, unsigned(sun_uniform.options[1]), SunShadowTargets::Size, sun_uniform.options[0]);
    if (render_world && Device.dwFrame % 300 == 0)
        Msg("[renderer-vulkan] lighting.rgb frame=%u ambient=(%.4f,%.4f,%.4f) hemi=(%.4f,%.4f,%.4f) "
            "sun=(%.4f,%.4f,%.4f) combine=%s diagnostic=%.0f",
            Device.dwFrame, sun_uniform.ambient_color[0], sun_uniform.ambient_color[1], sun_uniform.ambient_color[2],
            sun_uniform.hemi_color[0], sun_uniform.hemi_color[1], sun_uniform.hemi_color[2], light.color[0],
            light.color[1], light.color[2], weather_set_ ? "weather" : "plain", sun_uniform.options[1]);
    WaterSceneUniform water_uniform{};
    bool vsm_ready = false;
    {
        Fmatrix camera, inverse, sun_view, sun_projection, sun_vp;
        std::memcpy(&camera, render_mvp, sizeof(camera));
        inverse.invert_44(camera);
        std::memcpy(sun_uniform.inverse_view_projection, &inverse, sizeof(inverse));
        std::memcpy(water_uniform.view_projection, &camera, sizeof(camera));
        std::memcpy(water_uniform.inverse_view_projection, &inverse, sizeof(inverse));
        std::memcpy(forward_lighting_.inverse_view_projection, &inverse, sizeof(inverse));
        std::copy_n(sun_uniform.fog_plane, 4, water_uniform.fog_plane);
        std::copy_n(sun_uniform.fog_color, 3, water_uniform.fog_color);
        forward_lighting_.fog_end_camera[1] = Device.vCameraPosition.x;
        forward_lighting_.fog_end_camera[2] = Device.vCameraPosition.y;
        forward_lighting_.fog_end_camera[3] = Device.vCameraPosition.z;
        water_uniform.camera_position[0] = Device.vCameraPosition.x;
        water_uniform.camera_position[1] = Device.vCameraPosition.y;
        water_uniform.camera_position[2] = Device.vCameraPosition.z;
        water_uniform.camera_position[3] = 1.f;
        Fvector direction;
        direction.set(light.direction_ambient[0], light.direction_ambient[1], light.direction_ambient[2]);
        if (direction.square_magnitude() < 0.0001f)
            direction.set(0.f, -1.f, 0.f);
        direction.normalize_safe();
        Fvector up;
        up.set(0.f, 1.f, 0.f);
        if (std::abs(direction.y) > .95f)
            up.set(0.f, 0.f, 1.f);
        Fvector origin;
        origin.mad(Device.vCameraPosition, direction, -140.f);
        sun_view.build_camera_dir(origin, direction, up);
        if (render_world && vsm_frames_.count())
        {
            const float light_direction[3]{direction.x, direction.y, direction.z};
            float stable_direction[3]{};
            if (!stabilize_vsm_direction(light_direction, stable_direction))
                std::copy_n(light_direction, 3, stable_direction);
            Fvector stable_sun, stable_up, light_origin;
            stable_sun.set(stable_direction[0], stable_direction[1], stable_direction[2]);
            stable_up.set(0.f, 1.f, 0.f);
            if (std::abs(stable_sun.y) > .95f) stable_up.set(0.f, 0.f, 1.f);
            light_origin.set(0.f, 0.f, 0.f);
            Fmatrix stable_light;
            stable_light.build_camera_dir(light_origin, stable_sun, stable_up);
            float light_matrix[16];
            std::memcpy(light_matrix, &stable_light, sizeof(light_matrix));
            const float camera_world[3]{Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z};
            const float camera_depth = light_matrix[2] * camera_world[0] + light_matrix[6] * camera_world[1] +
                light_matrix[10] * camera_world[2];
            const float near_depth = std::floor(camera_depth / 64.f) * 64.f - 256.f;
            std::string vsm_error;
            vsm_ready = build_vsm_clipmap(sun_uniform.inverse_view_projection, light_matrix, camera_world,
                8, 3, 32.f, near_depth, near_depth + 512.f, vsm_clipmap_, vsm_error);
            if (!vsm_ready && Device.dwFrame % 300 == 0)
                Msg("! [renderer-vulkan] VSM clipmap unavailable: %s", vsm_error.c_str());
        }
        const float shadow_span = sun_shadow_span(quality_preset);
        sun_projection.build_projection_ortho(shadow_span, shadow_span, 1.f, 320.f);
        sun_uniform.options[2] = .08f;
#if defined(XR_PLATFORM_ANDROID)
        if (Device.dwFrame % 300 == 0)
            Msg("[renderer-vulkan] shadow.range frame=%u preset=%u span=%.1fm texel=%.4fm", Device.dwFrame,
                quality_preset, shadow_span, shadow_span / SunShadowTargets::Size);
#endif
        sun_vp.mul(sun_projection, sun_view);
        stabilize_shadow_projection(reinterpret_cast<float *>(&sun_vp), SunShadowTargets::Size);
        std::memcpy(sun_uniform.sun_view_projection, &sun_vp, sizeof(sun_vp));

        // Rain falls vertically. A world-aligned orthographic map plus texel
        // snapping keeps its roof edges stationary as the camera pans.
        const float rain_falloff_near = std::clamp(dynamic_wet_surfaces_near, 5.f, 70.f);
        const float rain_falloff_far = std::max(std::clamp(dynamic_wet_surfaces_far, 20.f, 100.f),
            rain_falloff_near + 0.01f);
        constexpr float rain_camera_height = 150.f;
        constexpr float rain_depth = 300.f;
        const float vertical_tangent = std::tan(deg2rad(std::clamp(Device.fFOV, 20.f, 140.f)) * .5f);
        const float horizontal_tangent = vertical_tangent * std::max(Device.fASPECT, .1f);
        const float radius = rain_falloff_far * .5f *
            (1.f + vertical_tangent * vertical_tangent + horizontal_tangent * horizontal_tangent);
        const float rain_span = std::clamp(2.f * radius, 40.f, 600.f);
        Fvector rain_direction, rain_up, rain_origin;
        rain_direction.set(0.f, -1.f, 0.f);
        rain_up.set(0.f, 0.f, 1.f);
        rain_origin.set(Device.vCameraPosition.x, Device.vCameraPosition.y + rain_camera_height,
            Device.vCameraPosition.z);
        Fmatrix rain_view, rain_projection, rain_vp;
        rain_view.build_camera_dir(rain_origin, rain_direction, rain_up);
        rain_projection.build_projection_ortho(rain_span, rain_span, .1f, rain_depth);
        rain_vp.mul(rain_projection, rain_view);
        stabilize_shadow_projection(reinterpret_cast<float *>(&rain_vp), rain_shadows_.size());
        std::memcpy(sun_uniform.rain_view_projection, &rain_vp, sizeof(rain_vp));
        const float rain_density = render_world && dynamic_wet_surfaces && g_pGamePersistent ?
            std::clamp(g_pGamePersistent->Environment().CurrentEnv.rain_density, 0.f, 1.f) : 0.f;
        sun_uniform.rain_params[0] = rain_density;
        sun_uniform.rain_params[1] = rain_falloff_near;
        sun_uniform.rain_params[2] = rain_falloff_far;
        sun_uniform.rain_params[3] = std::fmod(std::max(Device.fTimeGlobal, 0.f), 4096.f);
    }
    const bool sun_visible = light.color[0] > 1e-4f || light.color[1] > 1e-4f || light.color[2] > 1e-4f;
    const bool indirect_active = render_world && rtgi_strength > 0.f && indirect_surfaces_ && indirect_frames_.output_view(0);
    const bool ray_active = render_world && (sun_visible || indirect_active) && ray_queries_ && ray_queries_->ready() &&
        level.ray_scene_address() && ray_frames_.count() == window_.frame().image_count();
    vsm_ray_visibility_ = ray_active && sun_visible;
    sun_uniform.ambient_color[3] = ray_active && sun_visible ? 1.f : 0.f;
    sun_uniform.hemi_color[3] = ray_active && indirect_active ? rtgi_strength : 0.f;
    indirect_recorded_image_ = UINT32_MAX;
    frame_.set_ray_recorder(ray_active ? record_ray_visibility : nullptr, this);
    frame_.set_vsm_recorder(vsm_ready ? record_vsm_requests : nullptr, this);
    render_lights(render_mvp, render_world, light, sun_uniform);
    indirect_projections_.clear();
    if (indirect_active && indirect_frames_.projection_capacity())
        for (const auto& candidate : light_candidates_)
        {
            const auto& source = *candidate.light;
            if (!source.indirect || source.type != VulkanLightType::Spot || source.texture.empty()) continue;
            const auto found = std::find_if(indirect_projections_.begin(), indirect_projections_.end(),
                [&](const auto& item) { return item.first == source.texture; });
            if (found != indirect_projections_.end()) continue;
            if (indirect_projections_.size() >= indirect_frames_.projection_capacity())
            {
                if (Device.dwFrame % 300 == 0)
                    Msg("! [renderer-vulkan] indirect spotlight texture limit reached; excess lights use untextured GI");
                break;
            }
            indirect_projections_.emplace_back(source.texture, light_projection(source.texture));
        }
    motion_.set_view(render_mvp, sun_uniform.inverse_view_projection);
    if (!motion_.history_valid())
    {
        hiz_.reset_history();
        hiz_unsupported_dynamic_previous_ = false;
        temporal_.reset_history();
        fsr3_upscaler_.reset_history();
        indirect_temporal_.reset_history();
        indirect_frames_.reset_reservoir_history();
        temporal_sample_index_ = 0;
    }
    deferred_.set_forward_weather(render_world ? weather_set_ : VK_NULL_HANDLE);
    if (render_world && Device.dwFrame % 300 == 0)
        Msg("[renderer-vulkan] shadow.casters visible-models=%zu caster-models=%zu", model_draws_.size(),
            shadow_model_draws_.size());
    frame_.set_shadow_recorder(record_shadow_models, this);
    frame_.set_light_grid_recorder(prepare_light_grid, this);
    bool particle_refraction = false;
    bool world_refraction = false;
    for (const auto &particle : particle_draws_)
    {
        const bool active = particle.visual->getType() == 8
            ? static_cast<VulkanParticleEffect *>(particle.visual)->has_active_distortion()
            : static_cast<VulkanParticleGroup *>(particle.visual)->has_distortion();
        particle_refraction |= active;
        world_refraction |= active && !particle.hud;
        if (world_refraction)
            break;
    }
    frame_.set_distortion_recorder(render_world && world_refraction ? record_distortion : nullptr, this);
    // Resolve GPU program materials before finishing this frame's texture uploads.
    // The compute recorder runs later and uploads the resulting program table.
    if (gpu_particles_ && !gpu_particle_failed_ && gpu_particle_catalog_ &&
        !gpu_particle_materials_.sync(gpu_particles_->programs(), *gpu_particle_catalog_, textures_, deferred_, error))
    {
        Msg("! [renderer-vulkan] GPU particle materials unavailable: %s", error.c_str());
        disable_gpu_particles();
        error.clear();
    }
    if (trail_enabled_ && !trail_failed_ && !trail_texture_ && pSettings)
    {
        const char* name = pSettings->line_exist("bullet_manager", "tracer_texture")
            ? pSettings->r_string("bullet_manager", "tracer_texture") : "fx\\fx_tracer_weapons";
        if (!textures_.projection_texture(name, trail_texture_, error))
        {
            Msg("! [renderer-vulkan] trail texture '%s' unavailable: %s", name, error.c_str());
            trail_failed_ = true; error.clear();
        }
    }
    // All gameplay texture requests for this frame have now been gathered.
    // The completed copy fences provide the Adreno transfer visibility before
    // any rendering command can sample a newly loaded image.
    if (!textures_.finish_uploads())
    {
        error = "Vulkan gameplay texture batch did not complete";
        reset_required_ = true;
        return false;
    }
    bool unsupported_dynamic_occluders = render_world && !scene_visibility_;
    const bool gpu_particle_draw_active = gpu_particle_graphics_ && !gpu_particle_graphics_failed_;
    for (const ParticleDraw& draw : particle_draws_)
        if (!draw.hud && (draw.visual->getType() == 8
                ? !static_cast<VulkanParticleEffect *>(draw.visual)->supports_dynamic_occluder_mask(
                    false, gpu_particle_draw_active)
                : draw.visual->getType() == 9 &&
                    !static_cast<VulkanParticleGroup *>(draw.visual)->supports_dynamic_occluder_mask(
                        false, gpu_particle_draw_active)))
        {
            unsupported_dynamic_occluders = true;
            break;
        }
    if (!unsupported_dynamic_occluders && render_world && scene_visibility_ && current_level_)
        for (const LevelDraw& draw : level_draws_)
            if (!draw.hud && !current_level_->supports_dynamic_occluder_mask(draw.index))
            {
                unsupported_dynamic_occluders = true;
                break;
            }
    const bool hiz_dynamic_mask_enabled = render_world && scene_visibility_ &&
        !hiz_unsupported_dynamic_previous_ &&
        !unsupported_dynamic_occluders && motion_.ready() &&
        motion_.format() == VK_FORMAT_R16G16B16A16_SFLOAT && object_velocity_.ready() &&
        object_velocity_.lod_ready();
    hiz_dynamic_mask_recorded_ = hiz_dynamic_mask_enabled;
    frame_.set_hiz_dynamic_mask_enabled(hiz_dynamic_mask_enabled);
    if (fsr3_upscaler_.ready())
    {
        const float far_plane = g_pGamePersistent ? g_pGamePersistent->Environment().CurrentEnv.far_plane : 500.f;
        fsr3_upscaler_.set_frame(Device.dwFrame, Device.fTimeDelta * 1000.f, VIEWPORT_NEAR,
            far_plane, Device.fFOV * (3.14159265358979323846f / 180.f));
    }
    if (!frame_.render(window_.frame(), targets_, level, deferred_, render_mvp, light, status, error, record_ui, this,
                       record_hud, this, record_models, this, record_transparent, this,
                       scene_visibility_ ? record_level_visuals : nullptr, this, render_world, clear_target,
                       render_world ? weather_set_ : VK_NULL_HANDLE, &weather_lighting_,
                       screenshot_requested_ && readback_enabled_ ? record_readback : nullptr, this, record_postprocess,
                       this, &sun_shadows_, &sun_uniform, &local_shadows_, &local_uniforms_, record_local_lights, this,
                       render_world && (level.has_water() || particle_refraction) ? &water_targets_ : nullptr,
                       &rain_shadows_, &water_uniform, ui_.has_distortion() ? capture_ui : nullptr, this,
                       ui_.has_distortion() ? record_ui_distortion : nullptr, this,
                       grade_display ? capture_ui : nullptr, this,
                       grade_display ? record_display_grade : nullptr, this,
                       validate_frame_recording, this))
    {
        hiz_.reset_history();
        fsr3_upscaler_.reset_history();
        hiz_unsupported_dynamic_previous_ = false;
        smoke_recorded_ = {};
        sky_probe_recorded_ = {};
        if (gpu_particles_)
            for (uint32_t slot = 0; slot < gpu_particle_recorded_.size(); ++slot)
                if (gpu_particle_recorded_[slot])
                {
                    std::string ignored;
                    gpu_particles_->cancel(slot, ignored);
                    gpu_particle_recorded_[slot] = false;
                    gpu_collision_recorded_[slot] = false;
                }
        if (!window_.frame().device_lost())
            reset_required_ = true;
        model_draws_.clear();
        shadow_model_draws_.clear();
        shadow_draw_lookup_.clear();
        particle_draws_.clear();
        level_draws_.clear();
        level_draw_lookup_.clear();
        current_level_ = nullptr;
        ui_.reset_frame();
        return false;
    }
    if (!render_world)
    {
        hiz_.reset_history();
        hiz_unsupported_dynamic_previous_ = false;
    }
    else
        hiz_unsupported_dynamic_previous_ = unsupported_dynamic_occluders || !hiz_dynamic_mask_recorded_;
    if (indirect_recorded_image_ != UINT32_MAX)
    {
        indirect_frames_.mark_output_recorded(indirect_recorded_image_);
        indirect_frames_.commit_reservoir_history();
        if (indirect_temporal_.ready()) indirect_temporal_.commit_standalone();
    }
    else
    {
        indirect_temporal_.reset_history();
        indirect_frames_.reset_reservoir_history();
    }
    if (Device.dwFrame % 300 == 0)
    {
        const auto &timing = window_.frame().cpu_timing();
        Msg("[renderer-vulkan] frame.cpu frame=%u fence=%.3f acquire=%.3f record=%.3f submit=%.3f present=%.3f",
            Device.dwFrame, timing.fence_ms, timing.acquire_ms, timing.record_ms, timing.submit_ms, timing.present_ms);
    }
    if (status == FrameStatus::RecreateRequired)
    {
        reset_required_ = true;
        object_motion_.reset();
        motion_.reset_history();
        indirect_temporal_.reset_history();
        indirect_frames_.reset_reservoir_history();
        exposure_.reset_history();
        temporal_.reset_history();
        fsr3_upscaler_.reset_history();
        temporal_sample_index_ = 0;
    }
    else
    {
        if (render_world)
        {
            motion_.commit_view();
            previous_tree_wind_ = current_tree_wind_;
            object_motion_.commit_frame();
        }
        else
        {
            motion_.reset_history();
            object_motion_.reset();
        }
        if (render_world && tonemap_enabled)
            exposure_.commit_frame();
        else
            exposure_.reset_history();
        if (render_world && temporal_jitter_active_)
        {
            if (temporal_.active())
                temporal_.commit_frame();
            if (fsr3_upscaler_.active())
                fsr3_upscaler_.commit_frame();
            ++temporal_sample_index_;
        }
        else
        {
            temporal_.reset_history();
            fsr3_upscaler_.reset_history();
            temporal_sample_index_ = 0;
        }
        if (screenshot_requested_ && readback_enabled_)
            screenshot_ready_ = wait_idle();
    }
    if (status == FrameStatus::Presented)
        screenshot_requested_ = false;
    if (Device.dwPrecacheFrame == 1)
        save_pipeline_cache();
    model_draws_.clear();
    shadow_model_draws_.clear();
    shadow_draw_lookup_.clear();
    particle_draws_.clear();
    level_draws_.clear();
    level_draw_lookup_.clear();
    current_level_ = nullptr;
    ui_.reset_frame();
    return true;
}

void VulkanGameDevice::record_sky_probe_slice(const FrameRecordingContext& frame, const LevelRayScene& scene)
{
    if (!sky_probe_textures_ || sky_probe_failed_ || !g_pGameLevel ||
        frame.frame_index >= sky_probe_recorded_.size() || sky_probe_recorded_[frame.frame_index])
        return;
    std::string error;
    if (!sky_probe_bake_)
    {
        const Fbox& bounds = g_pGameLevel->ObjectSpace.GetBoundingVolume();
        const auto& physical = window_.physical();
        auto pass = std::make_unique<SkyProbeBakePass>();
        if (!sky_probe_layout_.initialize({bounds.vMin.x, bounds.vMin.y, bounds.vMin.z},
                {bounds.vMax.x, bounds.vMax.y, bounds.vMax.z}, 4.f, 65536) ||
            !pass->initialize(window_.device(), window_.device_proc(), physical.properties.limits,
                physical.memory, buffer_upload_.buffer, FrameContext::FramesInFlight,
                pipeline_cache_.handle(), *sky_probe_textures_, read_engine_shader, error) ||
            !sky_probe_results_.initialize(window_.device(),
                VkDeviceSize(sky_probe_layout_.count) * sizeof(SkyProbeVisibility),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                physical.memory, buffer_upload_.buffer, error))
        {
            sky_probe_failed_ = true;
            Msg("! [renderer-vulkan] sky probe initialization: %s", error.empty() ? "invalid level bounds" : error.c_str());
            return;
        }
        sky_probe_bake_ = std::move(pass);
    }
    const uint32_t count = std::min(SkyProbeBakePass::ProbesPerSlice, sky_probe_layout_.count - sky_probe_next_);
    if (!count)
        return;
    if (!sky_probe_bake_->record(frame.command_buffer, frame.frame_index, scene, sky_probe_layout_,
            sky_probe_results_, sky_probe_next_, count, 500.f, error))
    {
        sky_probe_failed_ = true;
        Msg("! [renderer-vulkan] sky probe recording: %s", error.c_str());
        return;
    }
    if (sky_probe_next_ + count == sky_probe_layout_.count)
    {
        VkMemoryBarrier host_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        host_read.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        host_read.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        frame_dispatch_.cmd_pipeline_barrier(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_read, 0, nullptr, 0, nullptr);
    }
    sky_probe_recorded_[frame.frame_index] = count;
}

bool VulkanGameDevice::record_ray_visibility(const FrameRecordingContext& frame, const GpuLevel& level,
    const SunShadowUniform& shadow, const DeferredLight& light, void* user)
{
    auto& device = *static_cast<VulkanGameDevice*>(user);
    if (!device.ray_queries_ || frame.image_index >= device.ray_frames_.count()) return false;
    const float direction[]{-light.direction_ambient[0], -light.direction_ambient[1],
        -light.direction_ambient[2], 500.f};
    const uint64_t pixels = uint64_t(frame.extent.width) * frame.extent.height;
    std::string error;
    auto& rays = device.ray_frames_.rays(frame.image_index);
    auto& hits = device.ray_frames_.hits(frame.image_index);
    static_assert(FrameContext::FramesInFlight == LevelRayScene::FrameSlots);
    const LevelRayScene* ray_scene = &level.ray_scene();
    std::vector<TextureMatrixSet> model_matrices;
    if (device.model_ray_preparer_ && frame.frame_index < LevelRayScene::FrameSlots)
    {
        auto batches = level.ray_geometry_batches();
        for (size_t i = 0; i < batches.size(); ++i)
            if (!batches[i].geometries.empty()) batches[i].external_blas = level.ray_scene().batch_blas_address(i);
        std::vector<const ModelDraw*> draws;
        for (const auto& draw : device.shadow_model_draws_)
            if (!draw.hud && draw.model && draw.model->ray_geometry_enabled()) draws.push_back(&draw);
        std::sort(draws.begin(), draws.end(), [](const ModelDraw* a, const ModelDraw* b) {
            if (a->model != b->model) return std::less<GpuModel*>{}(a->model, b->model);
            if (a->instance != b->instance) return std::less<const void*>{}(a->instance, b->instance);
            return a->world < b->world;
        });
        bool refit_tlas = false;
        for (const auto* draw : draws)
        {
            VkTransformMatrixKHR world{};
            for (unsigned row = 0; row < 3; ++row)
                for (unsigned column = 0; column < 4; ++column)
                    world.matrix[row][column] = draw->world[column * 4 + row];
            RayGeometryBatch batch;
            std::vector<TextureMatrixSet> matrices;
            std::string issue;
            bool updated = false;
            if (!device.model_ray_preparer_(*draw->model, draw->skeleton, frame.frame_index,
                    frame.command_buffer, world, batch, matrices, updated, issue))
            {
                if (Device.dwFrame % 300 == 0)
                    Msg("! [renderer-vulkan] model ray preparation skipped: %s", issue.c_str());
                continue;
            }
            if (batch.geometries.empty()) continue;
            refit_tlas |= updated;
            model_matrices.insert(model_matrices.end(), matrices.begin(), matrices.end());
            batches.push_back(std::move(batch));
        }
        if (!model_matrices.empty())
        {
            uint64_t revision = 1469598103934665603ull;
            const auto mix = [&](uint64_t value) { revision = (revision ^ value) * 1099511628211ull; };
            for (const auto& batch : batches)
            {
                mix(batch.external_blas); mix(batch.geometries.size());
                for (size_t i = 0; i < batch.geometries.size(); ++i)
                {
                    const auto& triangles = batch.geometries[i].geometry.triangles;
                    mix(triangles.vertexData.deviceAddress); mix(triangles.indexData.deviceAddress);
                    mix(triangles.vertexStride); mix(triangles.maxVertex); mix(batch.geometries[i].flags);
                    uint32_t cutoff_bits{};
                    std::memcpy(&cutoff_bits, &batch.alpha_cutoffs[i], sizeof(cutoff_bits));
                    mix(cutoff_bits);
                    mix(batch.primitive_counts[i]); mix(batch.material_ids[i]); mix(batch.texture_slots[i]);
                    mix(batch.diffuse_texture_slots[i]); mix(batch.pbr_texture_slots[i]);
                    mix(batch.metallic_texture_slots[i]); mix(batch.roughness_texture_slots[i]);
                    mix(batch.ao_texture_slots[i]);
                    mix(batch.normal_texture_slots[i]); mix(batch.normal_aux_texture_slots[i]);
                    if (i < batch.terrain_texture_slots.size())
                        for (uint32_t slot : batch.terrain_texture_slots[i]) mix(slot);
                    if (i < batch.terrain_scales.size())
                    {
                        uint32_t scale_bits{};
                        std::memcpy(&scale_bits, &batch.terrain_scales[i], sizeof(scale_bits));
                        mix(scale_bits);
                    }
                    if (i < batch.parallax_heights.size())
                    {
                        uint32_t height_bits{};
                        std::memcpy(&height_bits, &batch.parallax_heights[i], sizeof(height_bits));
                        mix(height_bits);
                    }
                    if (!batch.material_flags.empty()) mix(batch.material_flags[i]);
                }
            }
            const auto& window = device.window_;
            auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                window.instance_proc()(window.instance(), "vkGetPhysicalDeviceProperties2"));
            if (get_properties)
            {
                VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
                VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
                properties.pNext = &acceleration;
                get_properties(window.physical().handle, &properties);
                auto dispatch = load_acceleration_structure_dispatch(window.device(), window.device_proc(),
                    window.physical().properties.apiVersion >= VK_API_VERSION_1_2);
                std::vector<PendingBufferUpload> pending;
                if (device.model_ray_scenes_.prepare(frame.frame_index, revision, frame.command_buffer,
                        window.device(), window.queue(), window.frame().command_pool(), window.physical().memory,
                        device.buffer_upload_, pending, batches, dispatch, acceleration, error, refit_tlas))
                    ray_scene = device.model_ray_scenes_.scene(frame.frame_index);
                else if (Device.dwFrame % 300 == 0)
                    Msg("! [renderer-vulkan] model ray scene unavailable: %s", error.c_str());
            }
        }
    }
    std::vector<RayTextureTransform> transforms;
    transforms.reserve(level.ray_scene().hit_geometry_count());
    for (const auto& batch : level.ray_texture_matrices())
        for (const auto& matrix : batch)
        {
            RayTextureTransform transform;
            if (matrix.enabled[0])
            {
                const auto rows = matrix.values[0].evaluate(Device.fTimeGlobal,
                    reinterpret_cast<const float*>(&Device.mView));
                std::copy(rows[0].begin(), rows[0].end(), transform.row0);
                std::copy(rows[1].begin(), rows[1].end(), transform.row1);
            }
            if (matrix.enabled[1])
            {
                const auto rows = matrix.values[1].evaluate(Device.fTimeGlobal,
                    reinterpret_cast<const float*>(&Device.mView));
                std::copy(rows[0].begin(), rows[0].end(), transform.detail_row0);
                std::copy(rows[1].begin(), rows[1].end(), transform.detail_row1);
            }
            transforms.push_back(transform);
        }
    if (ray_scene != &level.ray_scene() && device.sky_probe_textures_ &&
        !device.sky_probe_failed_ && !level.ray_scene().write_texture_transforms(frame.frame_index, transforms, error))
    {
        device.sky_probe_failed_ = true;
        Msg("! [renderer-vulkan] sky probe texture transforms: %s", error.c_str());
    }
    if (ray_scene != &level.ray_scene())
        for (const auto& matrix : model_matrices)
        {
            RayTextureTransform transform;
            if (matrix.enabled[0])
            {
                const auto rows = matrix.values[0].evaluate(Device.fTimeGlobal,
                    reinterpret_cast<const float*>(&Device.mView));
                std::copy(rows[0].begin(), rows[0].end(), transform.row0);
                std::copy(rows[1].begin(), rows[1].end(), transform.row1);
            }
            if (matrix.enabled[1])
            {
                const auto rows = matrix.values[1].evaluate(Device.fTimeGlobal,
                    reinterpret_cast<const float*>(&Device.mView));
                std::copy(rows[0].begin(), rows[0].end(), transform.detail_row0);
                std::copy(rows[1].begin(), rows[1].end(), transform.detail_row1);
            }
            transforms.push_back(transform);
        }
    if (!ray_scene->write_texture_transforms(frame.frame_index, transforms, error) ||
        !device.ray_frames_.generator().record(frame, shadow.inverse_view_projection, direction) ||
        !device.ray_queries_->record(frame.command_buffer, frame.frame_index, *ray_scene, rays, hits,
            static_cast<uint32_t>(pixels), error) ||
        !device.ray_frames_.visibility().record(frame, device.ray_frames_.layout(frame.image_index)))
    {
        Msg("! [renderer-vulkan] RT visibility recording failed: %s", error.c_str());
        return false;
    }
    device.record_sky_probe_slice(frame, level.ray_scene());
    device.ray_frames_.mark_recorded(frame.image_index);
    device.deferred_.bind_ray_visibility(device.targets_.lighting_set(frame.image_index),
        device.ray_frames_.view(frame.image_index), device.targets_.sampler());
    if (shadow.hemi_color[3] > 0.f && device.indirect_surfaces_ &&
        device.indirect_frames_.output_view(frame.image_index))
    {
        std::vector<IndirectLocalLight> indirect_lights;
        indirect_lights.reserve(std::min<size_t>(device.light_candidates_.size(), LocalLightCapacity));
        for (const auto& candidate : device.light_candidates_)
        {
            const auto& source = *candidate.light;
            if (!source.indirect || indirect_lights.size() >= LocalLightCapacity) continue;
            IndirectLocalLight local{};
            for (unsigned axis = 0; axis < 3; ++axis)
            {
                local.position_range[axis] = source.position[axis];
                local.direction_cone[axis] = source.direction[axis];
                local.color_type[axis] = std::isfinite(source.color[axis]) ? std::max(source.color[axis], 0.f) : 0.f;
            }
            local.position_range[3] = std::clamp(source.range, .2f, 1000.f);
            local.direction_cone[3] = std::cos(std::clamp(source.cone, .15f, 3.f) * .5f);
            local.color_type[3] = (source.type == VulkanLightType::Spot ? IndirectLocalLightSpotFlag : 0.f) +
                (source.shadow ? IndirectLocalLightShadowFlag : 0.f);
            std::copy(source.right.begin(), source.right.end(), local.spot_right);
            local.spot_right[3] = std::tan(std::clamp(source.cone, .15f, 3.f) * .5f);
            if (source.type == VulkanLightType::Spot && !source.texture.empty())
            {
                const auto found = std::find_if(device.indirect_projections_.begin(), device.indirect_projections_.end(),
                    [&](const auto& item) { return item.first == source.texture; });
                if (found != device.indirect_projections_.end())
                    local.projection_slot = static_cast<uint32_t>(found - device.indirect_projections_.begin());
            }
            indirect_lights.push_back(local);
        }
        std::vector<VkImageView> projection_views;
        for (const auto& projection : device.indirect_projections_) projection_views.push_back(projection.second);
        const VkImageView motion_view = device.motion_.ready() ? device.motion_.view(frame.image_index) :
            device.targets_.normal_view(frame.image_index);
        const bool reservoir_history_valid = device.motion_.ready() && device.motion_.history_valid() &&
            device.indirect_temporal_.ready() && device.indirect_temporal_.history_valid();
        const bool motion_history_valid = device.motion_.ready() && device.motion_.history_valid();
        if (!device.indirect_frames_.record(frame, *ray_scene, *device.indirect_surfaces_,
                shadow.inverse_view_projection, 100.f, Device.dwFrame, direction, shadow.hemi_color,
                light.color, indirect_lights, error, projection_views, device.textures_.sampler(),
                motion_view, motion_history_valid, reservoir_history_valid,
                device.motion_.ready() ? device.motion_.image(frame.image_index) : VK_NULL_HANDLE))
        {
            Msg("! [renderer-vulkan] indirect ray recording failed: %s", error.c_str());
            return false;
        }
        VkImageView indirect_view = device.indirect_frames_.output_view(frame.image_index);
        if (device.indirect_temporal_.ready())
        {
            if (!device.indirect_temporal_.record_standalone(frame))
            { Msg("! [renderer-vulkan] indirect temporal resolve failed"); return false; }
            indirect_view = device.indirect_temporal_.view(device.indirect_temporal_.output_slot());
        }
        device.deferred_.bind_indirect_light(device.targets_.lighting_set(frame.image_index),
            indirect_view, device.targets_.sampler());
        device.indirect_recorded_image_ = frame.image_index;
    }
    return true;
}

bool VulkanGameDevice::prepare_ray_frames(std::string& error)
{
    auto& frame = window_.frame();
    const auto& physical = window_.physical();
    std::vector<VkImageView> depth, normal;
    for (uint32_t i = 0; i < frame.image_count(); ++i)
    {
        depth.push_back(targets_.sampled_depth_view(i));
        normal.push_back(targets_.normal_view(i));
    }
    return ray_frames_.initialize(physical.handle, window_.device(), window_.device_proc(), frame.extent(),
        depth, normal, targets_.sampler(), physical.properties, physical.memory, buffer_upload_.buffer,
        frame_dispatch_, frame.image_pool(), pipeline_cache_.handle(), error);
}

bool VulkanGameDevice::prepare_vsm_frames(std::string& error)
{
    auto& frame = window_.frame();
    const auto& physical = window_.physical();
    std::vector<VkImageView> depth;
    for (uint32_t i = 0; i < frame.image_count(); ++i)
        depth.push_back(targets_.sampled_depth_view(i));
    // Bound physical atlas memory on Android as well as on desktop.
    if (!vsm_frames_.initialize(physical.handle, window_.device(), window_.device_proc(), frame.extent(),
            depth, targets_.sampler(), frame.depth_format(), 8, 3, 64, 8, 8, physical.properties,
            physical.memory, buffer_upload_.buffer, frame_dispatch_, frame.image_pool(), pipeline_cache_.handle(), error))
        return false;
    deferred_.set_vsm_shadow_pass(vsm_frames_.static_atlas.render_pass(), vsm_frames_.dynamic_atlas.render_pass());
    frame.set_frame_completed_callback(vsm_frame_completed, this);
    vsm_cache_initialized_ = false;
    return true;
}

bool VulkanGameDevice::vsm_frame_completed(uint32_t frame, uint32_t image, void* user, std::string& error)
{
    auto& owner = *static_cast<VulkanGameDevice*>(user);
    if (owner.vsm_work_state_ != VsmWorkState::Waiting || frame != owner.vsm_pending_frame_ ||
        image != owner.vsm_pending_image_) return true;
    auto& resources = owner.vsm_frames_;
    auto& source = resources.image(image);
    if (!read_vsm_dynamic_jobs(source.dynamic_counts, source.dynamic_pages, owner.vsm_pending_cache_.axis,
            owner.vsm_pending_cache_.levels, static_cast<uint32_t>(resources.dynamic_atlas.page_count()),
            owner.vsm_dynamic_jobs_, error)) return false;
    resources.static_atlas.mark_initialized();
    resources.dynamic_atlas.mark_initialized();
    owner.vsm_work_state_ = VsmWorkState::Ready;
    return true;
}

bool VulkanGameDevice::record_vsm_dynamic_page(const FrameRecordingContext& frame, const float* vp,
    const VsmPageJob& job, void* user, std::string& error)
{
    auto& owner = *static_cast<VulkanGameDevice*>(user);
    VsmLevelPageContext trees{owner.current_level_, &owner.deferred_, false, true};
    return record_vsm_level_page(frame, vp, job, &trees, error) &&
        record_vsm_shadow_models(frame, vp, job, user, error);
}

bool VulkanGameDevice::record_vsm_requests(const FrameRecordingContext& frame, void* user)
{
    auto& owner = *static_cast<VulkanGameDevice*>(user);
    if (frame.image_index >= owner.vsm_frames_.count()) return false;
    auto& resources = owner.vsm_frames_;
    auto& image = resources.image(frame.image_index);
    std::string error;
    if (owner.vsm_work_state_ == VsmWorkState::Ready)
    {
        auto& previous = resources.image(owner.vsm_pending_image_);
        if (owner.current_level_ == owner.vsm_pending_level_)
        {
            VsmLevelPageContext level{owner.current_level_, &owner.deferred_, true, false};
            if (!resources.cache.render_and_publish(resources.static_atlas, frame,
                    owner.vsm_pending_clipmap_.projection, owner.vsm_pending_cache_, previous.needed,
                    previous.static_table, resources.static_slots, previous.static_work,
                    record_vsm_level_page, &level, error, prepare_vsm_level_page) ||
                !record_vsm_pages(resources.dynamic_atlas, frame, owner.vsm_pending_clipmap_.projection,
                    owner.vsm_dynamic_jobs_, record_vsm_dynamic_page, &owner, error))
            {
                Msg("! [renderer-vulkan] VSM page rendering failed: %s", error.c_str());
                return false;
            }
            VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            before.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
            owner.frame_dispatch_.cmd_pipeline_barrier(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
            const VkBufferCopy copy{0, 0, resources.visible_static_table.size()};
            owner.buffer_upload_.cmd_copy_buffer(frame.command_buffer, previous.static_table.handle(),
                resources.visible_static_table.handle(), 1, &copy);
            owner.buffer_upload_.cmd_copy_buffer(frame.command_buffer, previous.dynamic_table.handle(),
                resources.visible_dynamic_table.handle(), 1, &copy);
            VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
            after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            owner.frame_dispatch_.cmd_pipeline_barrier(frame.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
            owner.vsm_visible_clipmap_ = owner.vsm_pending_clipmap_;
            owner.vsm_visible_level_ = owner.current_level_;
            owner.vsm_visible_valid_ = true;
        }
        else owner.vsm_visible_valid_ = false;
        owner.vsm_work_state_ = VsmWorkState::Idle;
        owner.vsm_dynamic_jobs_.clear();
    }
    if (owner.vsm_visible_valid_ && owner.current_level_ == owner.vsm_visible_level_)
    {
        auto resolve = owner.vsm_visible_clipmap_.resolve;
        std::copy_n(owner.vsm_clipmap_.resolve.inverse_view_projection, 16, resolve.inverse_view_projection);
        resolve.page_size = 64; resolve.static_columns = 8; resolve.dynamic_columns = 8;
        resolve.static_slots = static_cast<uint32_t>(resources.static_atlas.page_count());
        resolve.dynamic_slots = static_cast<uint32_t>(resources.dynamic_atlas.page_count());
        resolve.dynamic_enabled = 1; resolve.bias = .0015f;
        if (!resources.resolve.record(frame, resolve, image.output_layout, error, UINT32_MAX,
                &resources.visible_static_table, &resources.visible_dynamic_table,
                owner.vsm_ray_visibility_ ? owner.ray_frames_.view(frame.image_index) : VK_NULL_HANDLE,
                owner.vsm_ray_visibility_ ? owner.targets_.sampler() : VK_NULL_HANDLE) ||
            !owner.sun_shadows_.set_visibility_enabled(frame.image_index, true, error))
        {
            Msg("! [renderer-vulkan] VSM visibility resolve failed: %s", error.c_str());
            return false;
        }
        image.output_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        owner.deferred_.bind_ray_visibility(owner.targets_.lighting_set(frame.image_index), image.output_view,
            owner.targets_.sampler());
    }
    if (owner.vsm_work_state_ == VsmWorkState::Waiting) return true;
    auto params = owner.vsm_clipmap_.cache;
    params.frame = Device.dwFrame; params.refresh_interval = 120; params.budget = 24;
    const auto& previous = owner.vsm_previous_clipmap_.projection;
    const auto& current = owner.vsm_clipmap_.projection;
    params.force_dirty = !owner.vsm_previous_valid_ || owner.current_level_ != owner.vsm_previous_level_ ||
        std::memcmp(previous.light_view, current.light_view, sizeof(current.light_view)) != 0 ||
        previous.near_depth != current.near_depth || previous.far_depth != current.far_depth;
    if ((!owner.vsm_cache_initialized_ &&
            (!resources.static_atlas.initialize_layout(frame.command_buffer, error) ||
             !resources.dynamic_atlas.initialize_layout(frame.command_buffer, error) ||
             !resources.cache.reset(frame.command_buffer, resources.static_slots, error))) ||
        !resources.demand.record(frame, owner.vsm_clipmap_.demand, error) ||
        !resources.cache.prepare(frame.command_buffer, frame.frame_index, params, image.needed, image.static_table,
            resources.static_slots, image.static_work, error) ||
        !resources.allocator.record(frame.command_buffer, frame.frame_index, params.axis, params.levels,
            static_cast<uint32_t>(resources.dynamic_atlas.page_count()), image.needed, image.dynamic_table,
            image.dynamic_pages, image.dynamic_counts, error))
    {
        Msg("! [renderer-vulkan] VSM page preparation failed: %s", error.c_str());
        return false;
    }
    owner.vsm_cache_initialized_ = true;
    owner.vsm_pending_frame_ = frame.frame_index;
    owner.vsm_pending_image_ = frame.image_index;
    owner.vsm_pending_level_ = owner.current_level_;
    owner.vsm_pending_clipmap_ = owner.vsm_clipmap_;
    owner.vsm_pending_cache_ = params;
    owner.vsm_previous_clipmap_ = owner.vsm_clipmap_;
    owner.vsm_previous_level_ = owner.current_level_;
    owner.vsm_previous_valid_ = true;
    owner.vsm_work_state_ = VsmWorkState::Waiting;
    return true;
}

void VulkanGameDevice::release_vsm_frames()
{
    frame_.set_vsm_recorder(nullptr, nullptr);
    window_.frame().set_frame_completed_callback(nullptr, nullptr);
    deferred_.set_vsm_shadow_pass(VK_NULL_HANDLE);
    vsm_frames_.destroy();
    vsm_cache_initialized_ = false;
    vsm_work_state_ = VsmWorkState::Idle;
    vsm_previous_valid_ = false;
    vsm_visible_valid_ = false;
    vsm_visible_level_ = nullptr;
    vsm_pending_level_ = vsm_previous_level_ = nullptr;
    vsm_dynamic_jobs_.clear();
}

bool VulkanGameDevice::prepare_indirect_frames(RaySurfacePass& surfaces,
    BindlessTextureTable& alpha_textures, std::string& error)
{
    release_indirect_frames();
    auto& frame = window_.frame();
    const auto& physical = window_.physical();
    if (!physical.features.shaderStorageImageExtendedFormats)
    { error = "indirect output requires extended storage image formats"; return false; }
    std::vector<VkImageView> depth, normal, albedo, surface;
    std::vector<VkImage> depth_images, normal_images, albedo_images, surface_images;
    for (uint32_t i = 0; i < frame.image_count(); ++i)
    {
        depth.push_back(targets_.sampled_depth_view(i));
        normal.push_back(targets_.normal_view(i));
        depth_images.push_back(targets_.sampled_depth_image(i));
        normal_images.push_back(targets_.normal_image(i));
        albedo.push_back(targets_.albedo_view(i));
        surface.push_back(targets_.surface_view(i));
        albedo_images.push_back(targets_.albedo_image(i));
        surface_images.push_back(targets_.surface_image(i));
    }
    const VkExtent2D full = frame.extent();
    VkExtent2D extent{full.width / 2 + full.width % 2, full.height / 2 + full.height % 2};
#if defined(XR_PLATFORM_ANDROID)
    // Bound per-swapchain-image ray working sets on mobile before allocating.
    constexpr uint64_t indirect_budget = 64ull * 1024 * 1024;
    const uint64_t reservoir_history_bytes = sizeof(RestirGIReservoir) * 2;
    const uint64_t history_bytes_per_swap_image =
        (reservoir_history_bytes + frame.image_count() - 1) / frame.image_count();
    const uint64_t bytes_per_pixel = sizeof(RayQueryInput) * 3 + sizeof(RayQueryHit) * 3 +
        sizeof(RaySurface) + sizeof(float) * 8 + 8 + history_bytes_per_swap_image;
    while ((extent.width > 1 || extent.height > 1) &&
        uint64_t(extent.width) * extent.height * frame.image_count() * bytes_per_pixel > indirect_budget)
    {
        extent.width = extent.width / 2 + extent.width % 2;
        extent.height = extent.height / 2 + extent.height % 2;
    }
#endif
    bool prepared = false;
    while (extent.width && extent.height)
    {
        if (indirect_frames_.initialize(window_.device(), window_.device_proc(), extent, depth, normal,
                targets_.sampler(), physical.properties, physical.memory, buffer_upload_.buffer,
                alpha_textures, pipeline_cache_.handle(), error, depth_images, normal_images,
                albedo, surface, albedo_images, surface_images, &targets_.image_sharing()))
        {
            if (indirect_frames_.prepare_output(physical.handle, window_.device(), window_.device_proc(),
                    frame_dispatch_, frame.image_pool(), pipeline_cache_.handle(), error))
                prepared = true;
            else
                indirect_frames_.destroy();
            break;
        }
        indirect_frames_.destroy();
        if (extent.width == 1 && extent.height == 1) break;
        extent.width = extent.width / 2 + extent.width % 2;
        extent.height = extent.height / 2 + extent.height % 2;
    }
    if (!prepared) return false;
    if (!indirect_frames_.projection_capacity())
        Msg("! [renderer-vulkan] indirect spotlight textures unavailable; using untextured GI");
    if (motion_.ready())
    {
        std::vector<VkImageView> scene_views, motion_views;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
        {
            scene_views.push_back(indirect_frames_.output_view(i));
            motion_views.push_back(motion_.view(i));
        }
        std::string temporal_error;
        if (indirect_temporal_.initialize(physical.handle, window_.device(), window_.device_proc(), extent,
                scene_views, motion_views, depth, physical.memory, frame_dispatch_, pipeline_cache_.handle(),
                temporal_error, &frame.image_pool(), motion_.format() == VK_FORMAT_R16G16B16A16_SFLOAT, true))
            indirect_temporal_.set_active(true);
        else
            Msg("! [renderer-vulkan] indirect temporal resolve unavailable: %s", temporal_error.c_str());
    }
    indirect_surfaces_ = &surfaces;
    indirect_alpha_textures_ = &alpha_textures;
    rtgi_available = 1;
    error.clear();
    return true;
}

void VulkanGameDevice::release_indirect_frames()
{
    for (uint32_t i = 0; i < ray_frames_.count(); ++i)
        deferred_.bind_indirect_light(targets_.lighting_set(i), targets_.sampled_depth_view(i), targets_.sampler());
    indirect_temporal_.destroy();
    indirect_frames_.destroy();
    indirect_surfaces_ = nullptr;
    indirect_alpha_textures_ = nullptr;
    indirect_recorded_image_ = UINT32_MAX;
    rtgi_available = 0;
}

void VulkanGameDevice::release_ray_frames()
{
    release_indirect_frames();
    model_ray_scenes_.destroy();
    ray_queries_ = nullptr;
    sky_probe_textures_ = nullptr;
    sky_probe_bake_.reset();
    sky_probe_results_.destroy();
    sky_probe_layout_ = {};
    sky_probe_next_ = 0;
    sky_probe_last_slot_ = UINT32_MAX;
    sky_probe_cpu_.clear();
    sky_probe_failed_ = false;
    sky_probe_recorded_ = {};
    frame_.set_ray_recorder(nullptr, nullptr);
    // Callers wait for the GPU before release. Restore valid descriptors before
    // destroying the visibility views, including when a level is unloaded.
    for (uint32_t i = 0; i < ray_frames_.count(); ++i)
        deferred_.bind_ray_visibility(targets_.lighting_set(i), targets_.sampled_depth_view(i), targets_.sampler());
    ray_frames_.destroy();
}

GBufferMsaaConfig VulkanGameDevice::negotiate_requested_msaa()
{
    if (gbuffer_msaa_samples != 2 && gbuffer_msaa_samples != 4)
        return {};
    const auto& physical = window_.physical();
    const auto& frame = window_.frame();
    GBufferMsaaConfig negotiated{};
    std::string error;
    const auto requested = gbuffer_msaa_samples == 4 ? VK_SAMPLE_COUNT_4_BIT : VK_SAMPLE_COUNT_2_BIT;
    if (window_.gbuffer_msaa_resolve_available() &&
        negotiate_gbuffer_msaa(physical.handle, physical.properties, VK_FORMAT_R8G8B8A8_UNORM,
            frame.depth_format(), requested, window_.image_format_properties(),
            window_.depth_stencil_resolve_properties(), window_.create_render_pass2(), negotiated, error,
            frame.scene_format()))
        return negotiated;
    Msg("! [renderer-vulkan] requested %ux G-buffer MSAA unavailable; using single-sample rendering: %s",
        gbuffer_msaa_samples, error.empty() ? "render-pass depth resolve unsupported" : error.c_str());
    return {};
}

bool VulkanGameDevice::initialize_gbuffer_targets(std::string& error, const GBufferMsaaConfig* preselected)
{
    gbuffer_msaa_request_ = gbuffer_msaa_samples;
    const auto& physical = window_.physical();
    auto& frame = window_.frame();
    const auto negotiated = gbuffer_msaa_configured_ ? gbuffer_msaa_config_ :
        preselected ? *preselected : negotiate_requested_msaa();
    const auto* msaa = negotiated.samples != VK_SAMPLE_COUNT_1_BIT ? &negotiated : nullptr;
    const VkSampleCountFlagBits requested_scene_samples = msaa ? msaa->samples : VK_SAMPLE_COUNT_1_BIT;
    const VkExtent2D presentation_extent = frame.presentation_extent();
    if (frame.scene_samples() != requested_scene_samples &&
        !window_.recreate_frame(presentation_extent, requested_scene_samples, error))
    {
        if (gbuffer_msaa_configured_ || !msaa || frame.device_lost() || frame.surface_lost())
            return false;
        Msg("! [renderer-vulkan] %ux MSAA scene creation failed; retrying single-sample: %s",
            gbuffer_msaa_samples, error.c_str());
        msaa = nullptr;
        if (!window_.recreate_frame(presentation_extent, VK_SAMPLE_COUNT_1_BIT, error))
            return false;
    }

    const FrameImageSharing sharing = frame_image_sharing(physical.graphics_present_family,
        physical.compute_queue_family);
    const auto initialize = [&](const GBufferMsaaConfig* config)
    {
        return targets_.initialize(physical.handle, window_.device(), frame.extent(),
            static_cast<uint32_t>(frame.image_count()), frame.depth_format(), physical.memory,
            frame_dispatch_, create_sampler_, destroy_sampler_, error, &frame.image_pool(), false, config,
            &sharing);
    };
    if (!initialize(msaa))
    {
        if (gbuffer_msaa_configured_ || !msaa)
            return false;
        Msg("! [renderer-vulkan] %ux G-buffer MSAA target creation failed; retrying single-sample: %s",
            gbuffer_msaa_samples, error.c_str());
        error.clear();
        msaa = nullptr;
        if (frame.scene_samples() != VK_SAMPLE_COUNT_1_BIT &&
            !window_.recreate_frame(frame.presentation_extent(), VK_SAMPLE_COUNT_1_BIT, error))
            return false;
        if (!initialize(nullptr))
            return false;
    }
    if (!gbuffer_msaa_configured_)
    {
        gbuffer_msaa_config_ = msaa ? negotiated : GBufferMsaaConfig{};
        gbuffer_msaa_configured_ = true;
    }
    Msg("[renderer-vulkan] gbuffer.samples=%u", static_cast<uint32_t>(targets_.samples()));
    error.clear();
    return true;
}

bool VulkanGameDevice::recreate_swapchain(VkExtent2D extent, std::string &error, bool recreate_surface)
{
    if (!window_.device() || !extent.width || !extent.height)
    {
        error = "Vulkan swapchain reset requires a live device and nonzero drawable extent";
        return false;
    }
    if (!wait_idle())
    {
        error = "could not wait for Vulkan device before renderer reset";
        return false;
    }

    const auto previous_samples = targets_.samples();
    if (gbuffer_msaa_request_ != gbuffer_msaa_samples)
        gbuffer_msaa_configured_ = false;
    if (gpu_particle_graphics_) gpu_particle_graphics_->clear_pipelines();
    smoke_draw_.reset();
    trail_draw_.reset();
    const bool replace_surface = recreate_surface || window_.frame().surface_lost();
    // The existing frame supplies the attachment formats for a same-surface reset.
    // A replacement surface is negotiated after SDL has created it.
    const bool preselect_msaa = !replace_surface && !gbuffer_msaa_configured_;
    auto selected_msaa = preselect_msaa ? negotiate_requested_msaa() : GBufferMsaaConfig{};
    release_visibility_gbuffer();
    for (auto& target : visibility_targets_)
        release_visibility_targets(target);
    const bool had_ray_frames = ray_frames_.count() != 0;
    const bool had_vsm_frames = vsm_frames_.count() != 0;
    RayQueryBatch* previous_ray_queries = ray_queries_;
    const auto* previous_sky_probe_textures = sky_probe_textures_;
    RaySurfacePass* previous_indirect_surfaces = indirect_surfaces_;
    BindlessTextureTable* previous_indirect_alpha = indirect_alpha_textures_;
    targets_.release_lighting(deferred_);
    release_vsm_frames();
    release_ray_frames();
    deferred_.release_water_sets();
    release_ui_scene_sets();
    for (auto &image : local_sets_)
        for (auto &set : image)
            deferred_.release_gbuffer(set);
    local_sets_.clear();
    local_descriptor_buffers_.clear();
    local_bound_projections_.clear();
    release_forward_targets();
    postprocess_passes_.clear();
    temporal_upscale_passes_.clear(); temporal_upscale_ = false;
    fsr3_upscale_passes_.clear();
    sun_shaft_passes_.clear();
    bloom_bright_passes_.clear();
    bloom_upsample_passes_.clear();
    if (!window_.frame().release_swapchain())
    {
        error = "could not release Vulkan framebuffers before scene depth";
        return false;
    }
    exposure_.destroy();
    temporal_.destroy();
    fsr3_upscaler_.reset_for_reinitialize();
    temporal_sample_index_ = 0;
    temporal_jitter_active_ = false;
    hiz_.destroy();
    object_velocity_.destroy();
    motion_.destroy();
    targets_.destroy();
    rain_shadows_.destroy();
    water_targets_.destroy();
#if defined(XR_PLATFORM_ANDROID)
    if (!android_presentation_extent(scaled_android_window_, extent, error))
        return false;
    const VkExtent2D scene_extent = scaled_scene_extent(extent, spatial_upscaler, render_scale_percent);
    const auto recreate = [&] { return replace_surface ? window_.recreate_surface(extent, scene_extent, error) :
        window_.recreate_frame(extent, scene_extent, error); };
#else
    const auto recreate = [&] { return replace_surface ? window_.recreate_surface(extent, error) :
        window_.recreate_frame(extent, error); };
#endif
    if (preselect_msaa && !window_.frame().request_scene_samples(selected_msaa.samples, error))
        return false;
    if (!recreate())
    {
        if (!preselect_msaa || selected_msaa.samples == VK_SAMPLE_COUNT_1_BIT ||
            window_.frame().device_lost() || window_.frame().surface_lost())
            return false;
        Msg("! [renderer-vulkan] %ux MSAA scene creation failed; retrying single-sample: %s",
            gbuffer_msaa_samples, error.c_str());
        selected_msaa = {};
        if (!window_.frame().request_scene_samples(selected_msaa.samples, error) || !recreate())
            return false;
    }

    auto &frame = window_.frame();
    const auto &physical = window_.physical();
    if (!initialize_gbuffer_targets(error, preselect_msaa ? &selected_msaa : nullptr))
        return false;
    frame.set_frame_generation_ui(Fsr3Upscaler::frame_generation_ui_active_callback,
        Fsr3Upscaler::register_frame_generation_ui_callback, &fsr3_upscaler_);
    FrameImagePool* image_pool = frame.image_pool().initialized() ? &frame.image_pool() : nullptr;
    if (gpu_particles_ || smoke_compute_) frame.set_frame_observer(gpu_particles_submitted, gpu_particles_completed, this);
#if defined(XR_PLATFORM_ANDROID)
    Msg("[renderer-vulkan] resolution reset scene=%ux%u presentation=%ux%u", frame.extent().width,
        frame.extent().height, frame.presentation_extent().width, frame.presentation_extent().height);
#endif
    if (gpu_particle_graphics_)
    {
        gpu_particle_graphics_->rebind_gbuffer_samples(targets_.samples());
        gpu_particle_graphics_failed_ = false;
    }
    if (had_ray_frames && !prepare_ray_frames(error))
    {
        Msg("! [renderer-vulkan] RT frame resources unavailable after resize: %s", error.c_str());
        error.clear();
    }
    else if (had_ray_frames)
        set_ray_queries(previous_ray_queries, previous_sky_probe_textures);
    {
        std::vector<VkImage> sampled_depth;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
            sampled_depth.push_back(targets_.sampled_depth_image(i));
        if (!frame.attach_postprocess_depth(sampled_depth, error))
            return false;
    }
    if (physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT)
    {
        std::vector<VkImageView> depth_views;
        std::vector<VkImage> depth_images;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
        {
            depth_views.push_back(targets_.sampled_depth_view(i));
            depth_images.push_back(targets_.sampled_depth_image(i));
        }
        if (!motion_.initialize(physical.handle, window_.device(), window_.device_proc(), frame.extent(),
                depth_views, depth_images, physical.memory, frame_dispatch_, pipeline_cache_.handle(),
                physical.features.shaderStorageImageExtendedFormats, error, &frame.image_pool(), true,
                &targets_.image_sharing()))
        {
            Msg("! [renderer-vulkan] motion vectors unavailable after resize: %s", error.c_str());
            error.clear();
        }
        if (motion_.ready())
        {
            std::vector<VkImageView> motion_views;
            std::vector<VkImage> motion_images;
            for (uint32_t i = 0; i < frame.image_count(); ++i)
            {
                motion_views.push_back(motion_.view(i));
                motion_images.push_back(motion_.image(i));
            }
            if (!hiz_.initialize(physical.handle, window_.device(), window_.device_proc(), frame.extent(),
                    depth_views, depth_images, motion_views, motion_images, physical.memory, frame_dispatch_,
                    pipeline_cache_.handle(), error, &frame.image_pool(), &targets_.image_sharing()))
            {
                Msg("! [renderer-vulkan] Hi-Z unavailable after resize: %s", error.c_str());
                error.clear();
            }
        }
    }
    if (ray_queries_ && previous_indirect_surfaces && previous_indirect_alpha &&
        !prepare_indirect_frames(*previous_indirect_surfaces, *previous_indirect_alpha, error))
    {
        Msg("! [renderer-vulkan] indirect frame resources unavailable after resize: %s", error.c_str());
        error.clear();
    }
    sun_shadows_.destroy();
    if (!sun_shadows_.initialize(window_.device(), frame.depth_format(), static_cast<uint32_t>(frame.image_count()),
                                 physical.memory, frame_dispatch_, buffer_upload_.buffer, create_sampler_,
                                 destroy_sampler_, error, image_pool))
        return false;
    if (!rain_shadows_.initialize(window_.device(), frame.depth_format(), static_cast<uint32_t>(frame.image_count()),
                                  physical.memory, frame_dispatch_, buffer_upload_.buffer, create_sampler_,
                                  destroy_sampler_, error, image_pool,
                                  rain_shadow_target_size(physical.properties.limits)))
        return false;
    local_shadows_.destroy();
    if (!local_shadows_.initialize(window_.device(), frame.depth_format(), static_cast<uint32_t>(frame.image_count()),
                                   physical.memory, frame_dispatch_, buffer_upload_.buffer, create_sampler_,
                                    destroy_sampler_, error, physical.properties.limits.minUniformBufferOffsetAlignment,
                                    image_pool))
        return false;
    local_sets_.resize(frame.image_count());
    {
        std::vector<VkImageView> depth_views;
        std::vector<VkImage> depth_images;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
        {
            depth_views.push_back(targets_.scene_depth_view(i));
            depth_images.push_back(targets_.scene_depth_image(i));
        }
        if (!frame.attach_scene_depth(depth_views, depth_images, error))
            return false;
    }
    if (!frame.enable_interpass(error) ||
        !water_targets_.initialize(physical.handle, window_.device(), frame.scene_format(), frame.extent(),
                                   static_cast<uint32_t>(frame.image_count()), physical.memory, frame_dispatch_,
                                   buffer_upload_.buffer, error, false, image_pool))
        return false;
    deferred_.rebind_compatible_render_passes(targets_.render_pass(), frame.render_pass(), sun_shadows_.render_pass(),
                                              local_shadows_.render_pass(), frame.overlay_render_pass(),
                                              rain_shadows_.render_pass());
    deferred_.set_local_shadow_graph_passes(local_shadows_.graph_render_pass(),
        local_shadows_.graph_load_render_pass(), local_shadows_.load_render_pass());
    if (previous_samples != targets_.samples())
    {
        const auto device = window_.device();
        const auto get = window_.device_proc();
        const ShaderModuleDispatch shaders{
            proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")};
        DeferredShaderFactory factory;
        if (!factory.rebuild_scene(device, shaders, targets_.samples(), targets_.render_pass(), frame.render_pass(),
                deferred_, shader_resources_, error, local_shadows_.render_pass()))
            return false;
        ShaderModule scene_vertex, scene_fragment, ui_vertex, ui_fragment;
        if (!scene_vertex.initialize(device, shaders, scene_shaders::SceneVertex, sizeof(scene_shaders::SceneVertex), error) ||
            !scene_fragment.initialize(device, shaders, scene_shaders::SceneFragment, sizeof(scene_shaders::SceneFragment), error) ||
            !ui_vertex.initialize(device, shaders, scene_shaders::UiVertex, sizeof(scene_shaders::UiVertex), error) ||
            !ui_fragment.initialize(device, shaders, scene_shaders::UiFragment, sizeof(scene_shaders::UiFragment), error) ||
            !ui_pass_.rebuild_scene_pipelines(frame.render_pass(), scene_vertex.handle(), scene_fragment.handle(),
                ui_vertex.handle(), ui_fragment.handle(), targets_.samples(), true, error))
            return false;
        light_grid_.destroy();
        if (physical.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT)
        {
            ScenePassDispatch dispatch;
            if (!load_scene_pass_dispatch(device, get, dispatch, error))
                return false;
            dispatch.gbuffer_samples = targets_.samples();
            dispatch.pipeline_cache = pipeline_cache_.handle();
            if (!light_grid_.initialize(device, get, frame.render_pass(), deferred_.gbuffer_layout(), physical.memory,
                    buffer_upload_.buffer, dispatch, shaders, frame_dispatch_.cmd_pipeline_barrier, error))
            {
                Msg("! [renderer-vulkan] light grid unavailable after MSAA change: %s", error.c_str());
                error.clear();
            }
        }
    }
    if (motion_.color_attachment_ready())
    {
        std::vector<VkImageView> velocity_views, geometry_depth_views;
        for (uint32_t i = 0; i < frame.image_count(); ++i)
        {
            velocity_views.push_back(motion_.view(i));
            geometry_depth_views.push_back(targets_.depth_view(i));
        }
        if (!object_velocity_.initialize(window_.device(), window_.device_proc(), frame.extent(),
                frame.depth_format(), velocity_views, geometry_depth_views,
                pipeline_cache_.handle(), error, deferred_.material_layout(), deferred_.pose_layout(),
                physical.features.multiDrawIndirect
                    ? std::max(physical.properties.limits.maxDrawIndirectCount, 1u) : 1u))
        {
            Msg("! [renderer-vulkan] object velocity unavailable after resize: %s", error.c_str());
            error.clear();
        }
    }
    frame_.set_object_velocity_recorder(object_velocity_.ready() ? record_object_velocity : nullptr, this);
    ui_pass_.rebind_render_pass(frame.render_pass(), frame.overlay_render_pass(), frame.presentation_render_pass());
    if (!create_ui_scene_sets(error))
        return false;
    if (!targets_.bind_lighting(deferred_, error))
        return false;
    targets_.bind_sun_shadow(deferred_, sun_shadows_);
    targets_.bind_rain_shadow(deferred_, rain_shadows_);
    if (had_vsm_frames && !prepare_vsm_frames(error))
    {
        Msg("! [renderer-vulkan] VSM resources unavailable after resize: %s", error.c_str());
        error.clear();
    }
    for (uint32_t i = 0; i < frame.image_count(); ++i)
        if (!deferred_.water_set(i, water_targets_.refraction(i), water_targets_.distortion_field(i),
                                 targets_.sampled_depth_view(i), textures_.sampler(), targets_.sampler(),
                                 water_targets_.uniform(i), error))
            return false;
    if (!create_forward_targets(error))
        return false;
    if (!create_postprocess(error))
        return false;

    reset_required_ = false;
    error.clear();
    return true;
}

void VulkanGameDevice::set_level_visibility(const VisibilityPasses* passes, const BindlessTextureTable* materials)
{
    visibility_passes_ = passes;
    visibility_materials_ = materials;
    frame_.set_visibility_recorder(passes && materials ? record_level_visibility : nullptr, this);
    frame_.set_visibility_gbuffer_recorder(passes && materials ? draw_level_visibility : nullptr);
    if (!passes)
    {
        release_visibility_gbuffer();
        for (size_t slot = 0; slot < visibility_draw_masks_.size(); ++slot)
        {
            if (visibility_mask_views_[slot])
                destroy_visibility_mask_view_(window_.device(), visibility_mask_views_[slot], nullptr);
            visibility_mask_views_[slot] = VK_NULL_HANDLE;
            visibility_draw_masks_[slot].destroy();
        }
        for (auto& target : visibility_targets_)
            release_visibility_targets(target);
        if (visibility_sampler_)
            destroy_sampler_(window_.device(), visibility_sampler_, nullptr);
        visibility_sampler_ = VK_NULL_HANDLE;
    }
}

void VulkanGameDevice::release_visibility_targets(VisibilityPasses::Targets& targets)
{
    for (auto* target : {&targets.ids, &targets.depth, &targets.base_color, &targets.normals})
    {
        if (target->view)
            frame_dispatch_.destroy_image_view(window_.device(), target->view, nullptr);
        if (target->image)
            window_.frame().image_pool().release(target->image);
    }
    targets = {};
}

bool VulkanGameDevice::ensure_visibility_targets(uint32_t slot, VkExtent2D extent, std::string& error)
{
    auto& targets = visibility_targets_[slot];
    if (targets.extent.width == extent.width && targets.extent.height == extent.height && targets.ids.view)
        return true;
    const auto device = window_.device();
    const auto physical = window_.physical().handle;
    auto& pool = window_.frame().image_pool();
    release_visibility_targets(targets);
    constexpr auto features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    const auto supports = [&](VkFormat format, VkFormatFeatureFlags required) {
        VkFormatProperties properties{};
        frame_dispatch_.get_format_properties(physical, format, &properties);
        return (properties.optimalTilingFeatures & required) == required;
    };
    if (!supports(VK_FORMAT_R32_UINT, features | VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT) ||
        !supports(VK_FORMAT_R16G16B16A16_SFLOAT, features))
    {
        error = "visibility image formats unavailable";
        return false;
    }
    if (!visibility_sampler_)
    {
        VkSamplerCreateInfo sampler{};
        sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        sampler.minFilter = sampler.magFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (create_sampler_(device, &sampler, nullptr, &visibility_sampler_) != VK_SUCCESS)
        {
            visibility_sampler_ = VK_NULL_HANDLE;
            error = "visibility sampler creation failed";
            return false;
        }
    }
    const auto create = [&](VisibilityPasses::Target& target, VkFormat format) {
        VkImageCreateInfo image{};
        image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = format;
        image.extent = {extent.width, extent.height, 1};
        image.mipLevels = image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (!pool.acquire(image, target.image, error))
            return false;
        VkImageViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image = target.image;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (frame_dispatch_.create_image_view(device, &view, nullptr, &target.view) == VK_SUCCESS)
            return true;
        target.view = VK_NULL_HANDLE;
        error = "visibility image view creation failed";
        return false;
    };
    if (!create(targets.ids, VK_FORMAT_R32_UINT) || !create(targets.depth, VK_FORMAT_R32_UINT) ||
        !create(targets.base_color, VK_FORMAT_R16G16B16A16_SFLOAT) || !create(targets.normals, VK_FORMAT_R16G16B16A16_SFLOAT))
    {
        release_visibility_targets(targets);
        return false;
    }
    targets.extent = extent;
    targets.id_sampler = visibility_sampler_;
    return true;
}

void VulkanGameDevice::record_level_visibility(const FrameRecordingContext& frame, const GpuLevel& level,
    const float (&mvp)[16], void* user)
{
    auto& self = *static_cast<VulkanGameDevice*>(user);
    if (frame.frame_index >= self.visibility_targets_.size())
        return;
    self.visibility_recorded_[frame.frame_index] = false;
    if (!self.scene_visibility_ || self.targets_.samples() != VK_SAMPLE_COUNT_1_BIT || !frame.extent.width || !frame.extent.height ||
        frame.viewport_offset.x || frame.viewport_offset.y)
        return;
    std::string error;
    const auto commands = level.visibility_camera_commands(frame.frame_index);
    const auto draw_count = commands.range / sizeof(VkDrawIndexedIndirectCommand);
    const auto words = (draw_count + 31) / 32;
    const auto& physical = self.window_.physical();
    if (!words || words > physical.properties.limits.maxTexelBufferElements)
        return;
    std::vector<uint32_t> selected(static_cast<size_t>(words));
    for (const auto& draw : self.level_draws_)
        if (!draw.hud && !draw.instance && std::equal(draw.mvp.begin(), draw.mvp.end(), mvp))
            level.mark_visibility_draws(draw.index, draw.lod, selected.data(), selected.size());
    if (std::none_of(selected.begin(), selected.end(), [](uint32_t word) { return word != 0; }))
        return;
    auto& mask = self.visibility_draw_masks_[frame.frame_index];
    auto& mask_view = self.visibility_mask_views_[frame.frame_index];
    const auto bytes = selected.size() * sizeof(uint32_t);
    if (!mask_view)
    {
        const auto device = self.window_.device();
        const auto get = self.window_.device_proc();
        const auto create_view = proc<PFN_vkCreateBufferView>(device, get, "vkCreateBufferView");
        self.destroy_visibility_mask_view_ = proc<PFN_vkDestroyBufferView>(device, get, "vkDestroyBufferView");
        VkFormatProperties format{};
        self.frame_dispatch_.get_format_properties(physical.handle, VK_FORMAT_R32_UINT, &format);
        if (!create_view || !self.destroy_visibility_mask_view_ ||
            !(format.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT) ||
            !mask.initialize(device, bytes, VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, physical.memory, self.buffer_upload_.buffer, error))
            return;
        VkBufferViewCreateInfo view{};
        view.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
        view.buffer = mask.handle();
        view.format = VK_FORMAT_R32_UINT;
        view.range = bytes;
        if (create_view(device, &view, nullptr, &mask_view) != VK_SUCCESS)
        {
            mask_view = VK_NULL_HANDLE;
            mask.destroy();
            return;
        }
    }
    if (!mask.write(0, selected.data(), bytes, error))
    {
        Msg("! [renderer-vulkan] visibility draw mask unavailable: %s", error.c_str());
        return;
    }
    visibility::Camera camera{};
    std::memcpy(camera.view_projection, mvp, sizeof(mvp));
    camera.viewport.extent[0] = static_cast<float>(frame.extent.width);
    camera.viewport.extent[1] = static_cast<float>(frame.extent.height);
    if (!self.initialize_visibility_gbuffer(error) || !self.ensure_visibility_targets(frame.frame_index, frame.extent, error) ||
        !self.visibility_passes_->record(frame, level, self.visibility_targets_[frame.frame_index], camera,
            *self.visibility_materials_, mask_view, error))
    {
        Msg("! [renderer-vulkan] visibility buffer recording disabled: %s", error.c_str());
        self.frame_.set_visibility_recorder(nullptr, nullptr);
        self.frame_.set_visibility_gbuffer_recorder(nullptr);
        return;
    }
    self.visibility_recorded_[frame.frame_index] = true;
}

bool VulkanGameDevice::initialize_visibility_gbuffer(std::string& error)
{
    if (visibility_gbuffer_pipeline_)
        return true;
    const auto device = window_.device();
    const auto get = window_.device_proc();
    auto& vk = visibility_graphics_;
    if (!load_screen_copy_dispatch(device, get, vk, error))
        return false;
    const auto fail = [&](const char* message) {
        if (error.empty())
            error = message;
        release_visibility_gbuffer();
        return false;
    };
    std::array<VkDescriptorSetLayoutBinding, 3> bindings{};
    for (uint32_t i = 0; i < bindings.size(); ++i)
        bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo layout{};
    layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout.bindingCount = static_cast<uint32_t>(bindings.size());
    layout.pBindings = bindings.data();
    if (vk.create_descriptor_set_layout(device, &layout, nullptr, &visibility_descriptor_layout_) != VK_SUCCESS)
        return fail("visibility G-buffer descriptor layout creation failed");

    constexpr uint32_t slots = FrameContext::FramesInFlight;
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, slots * 3};
    VkDescriptorPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool.maxSets = slots;
    pool.poolSizeCount = 1;
    pool.pPoolSizes = &size;
    if (vk.create_descriptor_pool(device, &pool, nullptr, &visibility_descriptor_pool_) != VK_SUCCESS)
        return fail("visibility G-buffer descriptor pool creation failed");
    std::array<VkDescriptorSetLayout, slots> layouts;
    layouts.fill(visibility_descriptor_layout_);
    VkDescriptorSetAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocation.descriptorPool = visibility_descriptor_pool_;
    allocation.descriptorSetCount = slots;
    allocation.pSetLayouts = layouts.data();
    if (vk.allocate_descriptor_sets(device, &allocation, visibility_descriptors_.data()) != VK_SUCCESS)
        return fail("visibility G-buffer descriptor allocation failed");
    VkPipelineLayoutCreateInfo pipeline_layout{};
    pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout.setLayoutCount = 1;
    pipeline_layout.pSetLayouts = &visibility_descriptor_layout_;
    if (vk.create_pipeline_layout(device, &pipeline_layout, nullptr, &visibility_gbuffer_layout_) != VK_SUCCESS)
        return fail("visibility G-buffer pipeline layout creation failed");

    const ShaderModuleDispatch modules{proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
        proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")};
    const auto load = [&](const char* name, ShaderModule& shader) {
        std::vector<uint8_t> bytes;
        return read_engine_shader(name, bytes, error) &&
            shader.initialize_bytes(device, modules, bytes.data(), bytes.size(), error);
    };
    ShaderModule vertex, fragment;
    if (!load("vk/visibility_gbuffer.vs.spv", vertex) || !load("vk/visibility_gbuffer.ps.spv", fragment) ||
        !deferred_.create_visibility_gbuffer_pipeline(targets_.render_pass(), visibility_gbuffer_layout_,
            vertex.handle(), fragment.handle(), visibility_gbuffer_pipeline_, error))
        return fail("visibility G-buffer pipeline creation failed");
    return true;
}

void VulkanGameDevice::draw_level_visibility(const FrameRecordingContext& frame, void* user)
{
    auto& self = *static_cast<VulkanGameDevice*>(user);
    if (frame.frame_index >= self.visibility_recorded_.size() || !self.visibility_recorded_[frame.frame_index])
        return;
    self.visibility_recorded_[frame.frame_index] = false;
    const auto& targets = self.visibility_targets_[frame.frame_index];
    if (targets.extent.width != frame.extent.width || targets.extent.height != frame.extent.height)
        return;

    const auto set = self.visibility_descriptors_[frame.frame_index];
    const VkDescriptorImageInfo images[]{
        {targets.id_sampler, targets.depth.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {targets.id_sampler, targets.base_color.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {targets.id_sampler, targets.normals.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    std::array<VkWriteDescriptorSet, 3> writes{};
    for (uint32_t i = 0; i < writes.size(); ++i)
    {
        auto& write = writes[i];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = i;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = &images[i];
    }
    self.visibility_graphics_.update_descriptor_sets(self.window_.device(),
        static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    self.deferred_.draw_visibility_gbuffer(frame, self.visibility_gbuffer_pipeline_, self.visibility_gbuffer_layout_, set);
}

void VulkanGameDevice::release_visibility_gbuffer()
{
    const auto device = window_.device();
    const auto& vk = visibility_graphics_;
    if (visibility_gbuffer_pipeline_)
        vk.destroy_pipeline(device, visibility_gbuffer_pipeline_, nullptr);
    if (visibility_gbuffer_layout_)
        vk.destroy_pipeline_layout(device, visibility_gbuffer_layout_, nullptr);
    if (visibility_descriptor_pool_)
        vk.destroy_descriptor_pool(device, visibility_descriptor_pool_, nullptr);
    if (visibility_descriptor_layout_)
        vk.destroy_descriptor_set_layout(device, visibility_descriptor_layout_, nullptr);
    visibility_gbuffer_pipeline_ = VK_NULL_HANDLE;
    visibility_gbuffer_layout_ = VK_NULL_HANDLE;
    visibility_descriptor_pool_ = VK_NULL_HANDLE;
    visibility_descriptor_layout_ = VK_NULL_HANDLE;
    visibility_descriptors_ = {};
    visibility_recorded_ = {};
}

bool VulkanGameDevice::prepare_for_reset(std::string &error)
{
    if (!window_.device())
    {
        error = "Vulkan renderer reset requires a live device";
        return false;
    }
    if (!wait_idle())
    {
        error = "could not wait for Vulkan device before renderer reset";
        return false;
    }

    // A reset starts between frames. Drop commands accumulated for a frame
    // that will no longer be submitted, while keeping device-owned UI buffers.
    model_draws_.clear();
    shadow_model_draws_.clear();
    shadow_draw_lookup_.clear();
    particle_draws_.clear();
    level_draws_.clear();
    level_draw_lookup_.clear();
    current_level_ = nullptr;
    motion_.reset_history();
    object_motion_.reset();
    exposure_.reset_history();
    temporal_.reset_history();
    fsr3_upscaler_.reset_history();
    temporal_sample_index_ = 0;
    ui_.setup_states();
    error.clear();
    return true;
}

void VulkanGameDevice::save_pipeline_cache()
{
    if (!pipeline_cache_.handle() || !FS.path_exist("$app_data_root$"))
        return;
    uint64_t shader_fingerprint = 0;
    std::string fingerprint_error;
    if (!shader_asset_fingerprint(shader_fingerprint, fingerprint_error))
    {
        Msg("! [renderer-vulkan] pipeline cache save skipped: %s", fingerprint_error.c_str());
        return;
    }
    const auto bytes = pipeline_cache_.snapshot(shader_fingerprint);
    if (bytes.empty())
        return;
    string_path destination;
    FS.update_path(destination, "$app_data_root$", "vulkan-pipelines.cache");
    if (IWriter *writer = FS.w_open(destination))
    {
        writer->w(bytes.data(), bytes.size());
        FS.w_close(writer);
        if (FS.file_length(destination) == static_cast<int>(bytes.size()))
            Msg("[renderer-vulkan] pipeline.cache saved-bytes=%zu", bytes.size());
        else
            Msg("! [renderer-vulkan] pipeline cache write verification failed: %s", destination);
    }
}

void VulkanGameDevice::destroy()
{
    frame_generation_available = 0;
    Msg("[renderer-vulkan] teardown game device begin");
    if (window_.device())
        R_ASSERT2(wait_idle(), "Vulkan shutdown requires completed frames");
    set_level_visibility(nullptr, nullptr);
    frame_.set_compute_recorder(nullptr, nullptr);
    window_.frame().set_frame_observer(nullptr, nullptr, nullptr);
    smoke_draw_.reset(); smoke_noise_.reset(); smoke_compute_.reset();
    trail_draw_.reset(); release_trail_texture(); trail_manager_.clear(); trail_enabled_ = trail_failed_ = false;
    smoke_controller_.reset(); smoke_recorded_ = {}; smoke_initialized_ = smoke_failed_ = false;
    gpu_particle_graphics_.reset(); gpu_particle_graphics_failed_ = false;
    gpu_particle_materials_.clear(textures_, deferred_);
    gpu_particle_catalog_.reset();
    gpu_particle_emitters_.reset();
    gpu_particles_.reset();
    frame_buffer_pool_.reset();
    gpu_particle_recorded_ = {}; gpu_particle_pending_ = {}; gpu_particle_failed_ = false;
    gpu_collision_recorded_ = {}; gpu_collision_uploaded_ = false; gpu_collision_.clear();
    Msg("[renderer-vulkan] teardown game device idle");
    save_pipeline_cache();
    ui_.DestroyUIGeom();
    R_ASSERT2(wait_idle(), "Vulkan UI shutdown requires completed descriptor leases");
    screenshot_buffer_.destroy();
    screenshot_requested_ = screenshot_ready_ = readback_enabled_ = false;
    model_draws_.clear();
    shadow_model_draws_.clear();
    shadow_draw_lookup_.clear();
    particle_draws_.clear();
    level_draws_.clear();
    level_draw_lookup_.clear();
    current_level_ = nullptr;
    scene_visibility_ = false;
    postprocess_passes_.clear();
    temporal_upscale_passes_.clear(); temporal_upscale_ = false;
    fsr3_upscale_passes_.clear();
    sun_shaft_passes_.clear();
    bloom_bright_passes_.clear();
    bloom_upsample_passes_.clear();
    for (auto view : color_map_views_)
        if (view)
            textures_.release_environment(view);
    color_map_views_[0] = color_map_views_[1] = VK_NULL_HANDLE;
    active_map_names_[0].clear();
    active_map_names_[1].clear();
    targets_.release_lighting(deferred_);
    release_vsm_frames();
    release_ray_frames();
    deferred_.release_water_sets();
    release_ui_scene_sets();
    release_forward_targets();
    release_level_glows();
    Msg("[renderer-vulkan] teardown textures begin");
    textures_.destroy();
    Msg("[renderer-vulkan] teardown textures end");
    if (window_.device())
        window_.frame().release_swapchain();
    exposure_.destroy();
    temporal_.destroy();
    fsr3_upscaler_.destroy();
    temporal_sample_index_ = 0;
    temporal_jitter_active_ = false;
    hiz_.destroy();
    object_velocity_.destroy();
    motion_.destroy();
    targets_.destroy();
    gbuffer_msaa_config_ = {};
    gbuffer_msaa_configured_ = false;
    water_targets_.destroy();
    for (auto &image : local_sets_)
        for (auto &set : image)
            deferred_.release_gbuffer(set);
    local_sets_.clear();
    local_descriptor_buffers_.clear();
    local_bound_projections_.clear();
    local_shadows_.destroy();
    rain_shadows_.destroy();
    sun_shadows_.destroy();
    ui_pass_.destroy();
    light_grid_.destroy();
    deferred_.destroy();
    pipeline_cache_.destroy();
    shader_resources_.destroy();
    Msg("[renderer-vulkan] teardown window begin");
    window_.destroy();
#if defined(XR_PLATFORM_ANDROID)
    if (scaled_android_window_)
    {
        std::string ignored;
        if (!set_android_buffer_extent(scaled_android_window_, {}, ignored))
            Msg("! [renderer-vulkan] restore Android buffer geometry: %s", ignored.c_str());
        scaled_android_window_ = nullptr;
        requested_android_extent_ = {};
    }
#endif
    Msg("[renderer-vulkan] teardown window end");
    frame_dispatch_ = {};
    texture_dispatch_ = {};
    buffer_upload_ = {};
    create_sampler_ = nullptr;
    destroy_sampler_ = nullptr;
    reset_required_ = false;
}
} // namespace xray::render::vulkan
