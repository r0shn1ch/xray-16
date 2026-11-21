#include "xrEngine/stdafx.h"
#include "VulkanVideoTexture.h"
#if defined(ANDROID)
#include "AndroidMediaVideoDecoder.h"
#endif
#include "ScenePass.h"
#include "xrEngine/device.h"

#include <vector>

namespace xray::render::vulkan
{
VulkanVideoTexture::~VulkanVideoTexture()
{
    for (VkDescriptorSet& descriptor : descriptors_)
        if (descriptor) textures_.release_ui(descriptor, pass_);
}

bool VulkanVideoTexture::load(const char* path, u32 time, std::string& error)
{
    bool avi = false;
#if defined(ANDROID)
    bool android_media = false;
    string_path android_path;
    xr_strcpy(android_path, path);
    const char* android_extension = strext(android_path);
    if (android_extension && (xr_stricmp(android_extension, ".mp4") == 0 ||
            xr_stricmp(android_extension, ".m4v") == 0 || xr_stricmp(android_extension, ".mov") == 0 ||
            xr_stricmp(android_extension, ".3gp") == 0 || xr_stricmp(android_extension, ".mkv") == 0))
    {
        android_surface_ = std::make_unique<AndroidMediaVideoDecoder>();
        if (!android_surface_->load(path, extent_.width, extent_.height, error))
        {
            android_surface_.reset();
            return false;
        }
        android_media = true;
    }
#endif
#if defined(XR_PLATFORM_WINDOWS)
    string_path mutable_path;
    xr_strcpy(mutable_path, path);
    const char* extension = strext(mutable_path);
    avi = extension && xr_stricmp(extension, ".avi") == 0;
    if (avi)
    {
        avi_surface_ = std::make_unique<CAviPlayerCustom>();
        if (!avi_surface_->Load(mutable_path))
        {
            error = "could not decode AVI video";
            avi_surface_.reset();
            return false;
        }
        extent_ = {avi_surface_->m_dwWidth, avi_surface_->m_dwHeight};
    }
#endif
    if (!avi
#if defined(ANDROID)
        && !android_media
#endif
    )
    {
        if (!surface_.Load(path)) { error = "could not decode OGM video"; return false; }
        surface_.ForceSoftwareRGB();
        extent_ = {surface_.Width(true), surface_.Height(true)};
    }
    if (!extent_.width || !extent_.height || extent_.width > 8192 || extent_.height > 8192)
    { error = "video dimensions are invalid"; return false; }
    if (uint64_t(extent_.width) * extent_.height * 4 > 32ull * 1024 * 1024)
    { error = "video frame exceeds the persistent UI texture budget"; return false; }
    const size_t pixel_count = size_t(extent_.width) * extent_.height;
    decoded_pixels_.resize(pixel_count);
    static_assert(sizeof(u32) == 4);
    const auto* blank = reinterpret_cast<const uint8_t*>(decoded_pixels_.data());
    for (VkDescriptorSet& descriptor : descriptors_)
        if (!textures_.ui_pixels(blank, extent_.width, extent_.height,
                pass_, descriptor, error, clamp_))
        {
            for (VkDescriptorSet& created : descriptors_)
                if (created)
                {
                    textures_.release_ui(created, pass_);
                    created = VK_NULL_HANDLE;
                }
            return false;
        }
    if (!avi
#if defined(ANDROID)
        && !android_media
#endif
    )
    {
        const bool stop_at_end = strstr(path, "intro\\") || strstr(path, "outro\\") ||
            strstr(path, "intro/") || strstr(path, "outro/");
        surface_.Play(!stop_at_end, time);
    }
#if defined(ANDROID)
    else if (android_media)
    {
        const bool stop_at_end = strstr(path, "intro\\") || strstr(path, "outro\\") ||
            strstr(path, "intro/") || strstr(path, "outro/");
        android_surface_->play(!stop_at_end, time);
    }
#endif
    return true;
}

void VulkanVideoTexture::play(bool looped, u32 time)
{
#if defined(ANDROID)
    if (android_surface_)
    {
        sync_time_ = time;
        last_update_frame_ = UINT32_MAX;
        android_surface_->play(looped, time);
        return;
    }
#endif
#if defined(XR_PLATFORM_WINDOWS)
    if (avi_surface_) return;
#endif
    sync_time_ = time;
    last_update_frame_ = UINT32_MAX;
    surface_.Play(looped, time);
}

void VulkanVideoTexture::stop()
{
#if defined(ANDROID)
    if (android_surface_) { android_surface_->stop(); return; }
#endif
#if defined(XR_PLATFORM_WINDOWS)
    if (avi_surface_) return;
#endif
    surface_.Stop();
}

bool VulkanVideoTexture::playing()
{
#if defined(ANDROID)
    if (android_surface_) return android_surface_->playing();
#endif
#if defined(XR_PLATFORM_WINDOWS)
    if (avi_surface_) return false;
#endif
    return surface_.IsPlaying();
}

void VulkanVideoTexture::update(u32 time, uint32_t frame_slot)
{
    if (frame_slot >= descriptors_.size() || !descriptors_[frame_slot] ||
        (last_update_frame_ == Device.dwFrame && last_update_slot_ == frame_slot))
        return;
    last_update_frame_ = Device.dwFrame;
    last_update_slot_ = frame_slot;
    bool decoded_new_frame = false;
#if defined(ANDROID)
    if (android_surface_)
        decoded_new_frame = android_surface_->update(
            sync_time_ == 0xFFFFFFFF ? time : sync_time_, decoded_pixels_);
#endif
#if defined(XR_PLATFORM_WINDOWS)
    if (avi_surface_ && avi_surface_->NeedUpdate())
    {
        u8* frame = nullptr;
        if (avi_surface_->GetFrame(&frame) && frame)
        {
            // VFW returns 32-bit BI_RGB pixels (B,G,R,X in memory). Convert
            // them to the RGBA layout used by the Vulkan UI upload path.
            auto* rgba = reinterpret_cast<uint8_t*>(decoded_pixels_.data());
            const auto* bgra = reinterpret_cast<const uint8_t*>(frame);
            for (size_t i = 0; i < decoded_pixels_.size(); ++i)
            {
                rgba[i * 4] = bgra[i * 4 + 2];
                rgba[i * 4 + 1] = bgra[i * 4 + 1];
                rgba[i * 4 + 2] = bgra[i * 4];
                rgba[i * 4 + 3] = 255;
            }
            decoded_new_frame = true;
        }
    }
#endif
    if (!decoded_new_frame
#if defined(ANDROID)
        && !android_surface_
#endif
        && surface_.Valid() &&
        surface_.Update(sync_time_ == 0xFFFFFFFF ? time : sync_time_))
    {
        int decoded = 0;
        surface_.DecompressFrame(decoded_pixels_.data(), 0, decoded);
        if (decoded == static_cast<int>(decoded_pixels_.size()))
        {
            // Convert the decoder's packed 0xAARRGGBB words in place to RGBA
            // bytes. This keeps one decoded-frame cache instead of retaining a
            // second full-size CPU copy for the Vulkan upload path.
            auto* rgba = reinterpret_cast<uint8_t*>(decoded_pixels_.data());
            for (size_t i = 0; i < decoded_pixels_.size(); ++i)
            {
                const u32 pixel = decoded_pixels_[i];
                rgba[i * 4] = uint8_t(pixel >> 16);
                rgba[i * 4 + 1] = uint8_t(pixel >> 8);
                rgba[i * 4 + 2] = uint8_t(pixel);
                rgba[i * 4 + 3] = uint8_t(pixel >> 24);
            }
            decoded_new_frame = true;
        }
    }
    if (decoded_new_frame)
    {
        if (content_generation_ == UINT64_MAX)
        {
            slot_generations_.fill(0);
            content_generation_ = 1;
        }
        else
            ++content_generation_;
    }

    // Each frame slot owns a separate image. If playback did not produce a
    // new decoded frame this call, bring the selected slot up to the latest
    // cached frame so lower-frame-rate movies do not alternate stale images.
    if (decoded_pixels_.empty() || slot_generations_[frame_slot] == content_generation_)
        return;
    std::string error;
    if (!textures_.update_ui_pixels(descriptors_[frame_slot],
            reinterpret_cast<const uint8_t*>(decoded_pixels_.data()), extent_.width,
            extent_.height, error))
        Msg("! [renderer-vulkan] OGM frame: %s", error.c_str());
    else
        slot_generations_[frame_slot] = content_generation_;
}
}
