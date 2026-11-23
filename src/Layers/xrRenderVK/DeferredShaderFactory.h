#pragma once

#include "DeferredPass.h"
#include "ShaderModule.h"
#include "GameShaderResources.h"

namespace xray::render::vulkan
{
bool read_engine_shader(const std::string& name, std::vector<uint8_t>& bytes, std::string& error);
void configure_engine_shader_resources(VkDevice device, const ShaderModuleDispatch &dispatch,
    GameShaderResources &resources, bool geometry_supported, bool tessellation_supported);
// Owns modules after pipeline creation, including the failure path.
class DeferredShaderFactory
{
  public:
    bool create(VkDevice device, const ShaderModuleDispatch &shader_dispatch, const ScenePassDispatch &pass_dispatch, VkRenderPass geometry_pass,
                VkRenderPass light_pass, DeferredPass &pass, GameShaderResources &resources, std::string &error,
                VkRenderPass shadow_pass = VK_NULL_HANDLE,
                VkRenderPass local_shadow_pass = VK_NULL_HANDLE);
    // The caller waits for submitted frames before swapping pipelines.
    bool reload_game_pipelines(VkDevice device, const ShaderModuleDispatch &shader_dispatch, DeferredPass &pass, GameShaderResources &resources,
                               std::string &error);
    bool rebuild_scene(VkDevice device, const ShaderModuleDispatch& shader_dispatch,
        VkSampleCountFlagBits samples, VkRenderPass geometry_pass, VkRenderPass light_pass,
        DeferredPass& pass, GameShaderResources& resources, std::string& error,
        VkRenderPass local_shadow_pass);

private:
    bool build(VkDevice device, const ShaderModuleDispatch& shader_dispatch, const ScenePassDispatch* initial_dispatch,
        VkSampleCountFlagBits samples, VkRenderPass geometry_pass, VkRenderPass light_pass,
        DeferredPass& pass, GameShaderResources& resources, std::string& error,
        VkRenderPass shadow_pass, VkRenderPass local_shadow_pass);
};
} // namespace xray::render::vulkan
