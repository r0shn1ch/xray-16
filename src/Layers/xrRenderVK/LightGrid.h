#pragma once
#include "BufferResource.h"
#include "ScenePass.h"
#include "ShaderModule.h"
#include "LocalShadowTargets.h"
#include <array>

namespace xray::render::vulkan
{
// Point lights without shadows share one G-buffer read per pixel. Compute
// builds conservative tile lists; shadowed/projected lights retain their path.
class LightGrid
{
  public:
    static constexpr uint32_t TileSize = XRAY_VK_LIGHT_TILE_SIZE;
    static_assert(LocalLightCapacity <= 64);
    ~LightGrid()
    {
        destroy();
    }
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkRenderPass pass, VkDescriptorSetLayout gbuffer,
                    const VkPhysicalDeviceMemoryProperties &memory, const BufferResourceDispatch &buffers,
                    const ScenePassDispatch &graphics, const ShaderModuleDispatch &shaders,
                    PFN_vkCmdPipelineBarrier barrier, std::string &error);
    bool prepare(const FrameRecordingContext &frame, const std::vector<LocalLightUniform> &lights,
                 const std::vector<VkRect2D> &scissors, std::string &error);
    bool record(const FrameRecordingContext &frame, VkDescriptorSet gbuffer) const;
    bool contains(uint32_t image, uint32_t light) const;
    uint32_t count(uint32_t image) const;
    void discard_frame(uint32_t image)
    {
        if (image < images_.size())
        {
            images_[image].count = 0;
            images_[image].mask = 0;
        }
    }
    void destroy();

  private:
    struct Image
    {
        BufferResource lights, tiles;
        VkDescriptorSet set{};
        VkExtent2D extent{};
        uint64_t mask{};
        uint32_t count{};
    };
    VkDevice device_{};
    VkDescriptorSetLayout set_layout_{};
    VkPipelineLayout layout_{};
    VkDescriptorPool pool_{};
    VkPipeline cull_{}, shade_{};
    PFN_vkCmdDispatch dispatch_{};
    PFN_vkCmdPipelineBarrier barrier_{};
    ScenePassDispatch vk_{};
    VkPhysicalDeviceMemoryProperties memory_{};
    BufferResourceDispatch buffers_{};
    std::array<Image, 16> images_{};
};
} // namespace xray::render::vulkan
