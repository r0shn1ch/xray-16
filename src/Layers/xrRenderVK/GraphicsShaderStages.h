#pragma once

#include <vulkan/vulkan.h>

#include <array>
#include <cstdint>
#include <string>

namespace xray::render::vulkan
{
struct GraphicsShaderModules
{
    VkShaderModule vertex{}, hull{}, domain{}, geometry{}, fragment{};
};

struct GraphicsShaderStages
{
    std::array<VkPipelineShaderStageCreateInfo, 5> stages{};
    uint32_t count{};
    bool tessellation{};
};

inline bool build_graphics_shader_stages(const GraphicsShaderModules& modules,
    GraphicsShaderStages& result, std::string& error)
{
    result = {};
    if (!modules.vertex || !modules.fragment || bool(modules.hull) != bool(modules.domain))
    {
        error = "graphics pipeline needs vertex/fragment and paired hull/domain shaders";
        return false;
    }
    const auto add = [&](VkShaderStageFlagBits stage, VkShaderModule module) {
        if (!module) return;
        auto& item = result.stages[result.count++];
        item.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        item.stage = stage;
        item.module = module;
        item.pName = "main";
    };
    add(VK_SHADER_STAGE_VERTEX_BIT, modules.vertex);
    add(VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT, modules.hull);
    add(VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, modules.domain);
    add(VK_SHADER_STAGE_GEOMETRY_BIT, modules.geometry);
    add(VK_SHADER_STAGE_FRAGMENT_BIT, modules.fragment);
    result.tessellation = modules.hull != VK_NULL_HANDLE;
    error.clear();
    return true;
}
}
