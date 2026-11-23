#pragma once

#include "FrameContext.h"
#include "GameTextureFactory.h"
#include "xrEngine/xrTheora_Surface.h"
#if defined(ANDROID)
#include "AndroidMediaVideoDecoder.h"
#endif

#include <array>
#include <memory>
#include <vector>

#if defined(XR_PLATFORM_WINDOWS)
#include "xrEngine/tntQAVI.h"
#endif

namespace xray::render::vulkan
{
class VulkanVideoTexture
{
public:
    VulkanVideoTexture(GameTextureFactory& textures, ScenePass& pass, bool clamp = true)
        : textures_(textures), pass_(pass), clamp_(clamp) {}
    ~VulkanVideoTexture();
    bool load(const char* path, u32 time, std::string& error);
    void update(u32 time, uint32_t frame_slot);
    void sync(u32 time) { sync_time_ = time; last_update_frame_ = UINT32_MAX; }
    void play(bool looped, u32 time);
    void stop();
    bool playing();
    VkDescriptorSet descriptor(uint32_t frame_slot) const
    { return frame_slot < descriptors_.size() ? descriptors_[frame_slot] : VK_NULL_HANDLE; }
    VkDescriptorSet descriptor() const { return descriptors_[0]; }
    VkExtent2D extent() const { return extent_; }

private:
    GameTextureFactory& textures_;
    ScenePass& pass_;
    CTheoraSurface surface_;
#if defined(ANDROID)
    std::unique_ptr<AndroidMediaVideoDecoder> android_surface_;
#endif
#if defined(XR_PLATFORM_WINDOWS)
    std::unique_ptr<CAviPlayerCustom> avi_surface_;
#endif
    std::array<VkDescriptorSet, FrameContext::FramesInFlight> descriptors_{};
    std::array<uint64_t, FrameContext::FramesInFlight> slot_generations_{};
    std::vector<u32> decoded_pixels_;
    VkExtent2D extent_{};
    u32 sync_time_{0xFFFFFFFF};
    u32 last_update_frame_{UINT32_MAX};
    uint32_t last_update_slot_{UINT32_MAX};
    uint64_t content_generation_{1};
    bool clamp_{true};
};
}
