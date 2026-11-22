#include "xrEngine/stdafx.h"
#include "DeferredPass.h"
#include "GraphicsShaderStages.h"
#include "SunShadowTargets.h"
#include "ShaderMaterialLibrary.h"
#include "ModelGeometry.h"
#include "ModelPoseConstants.h"
#include "GeometryDepth.h"
#include "DetailInstance.h"
#include "VulkanSettings.h"
#include "xrEngine/device.h"
#if defined(XR_PLATFORM_ANDROID)
#include "xrEngine/x_ray.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>

namespace xray::render::vulkan
{
namespace
{
enum class PipelineFlag : uint32_t
{
    None = 0,
    Geometry = 1u << 0,
    GBuffer = 1u << 1,
    Transparent = 1u << 2,
    Hud = 1u << 3,
    Skinned = 1u << 4,
    Shadow = 1u << 5,
    Additive = 1u << 6,
    AlphaAdditive = 1u << 7,
    Multiply = 1u << 8,
    Multiply2x = 1u << 9,
    ParticleSet = 1u << 10,
    WaterDepthWrite = 1u << 11,
    Wallmark = 1u << 12,
    LegacyMultiply = 1u << 13,
    ReplaceNoDepth = 1u << 14,
    HudSetNoDepthWrite = 1u << 15,
    Cutout = 1u << 16,
    Instanced = 1u << 17,
};
constexpr PipelineFlag operator|(PipelineFlag a, PipelineFlag b)
{
    return PipelineFlag(uint32_t(a) | uint32_t(b));
}
constexpr bool has(PipelineFlag value, PipelineFlag flag)
{
    return (uint32_t(value) & uint32_t(flag)) != 0;
}

bool make_pipeline(VkDevice device, VkRenderPass pass, VkPipelineLayout layout, VkShaderModule vertex,
                   VkShaderModule fragment, PipelineFlag options, const ScenePassDispatch &vk, VkPipeline &pipeline,
                    std::string &error, const char *label,
                    const DeferredPass::GamePipelineStages &game_stages = {})
{
    const bool geometry_input = has(options, PipelineFlag::Geometry);
    const bool gbuffer = has(options, PipelineFlag::GBuffer);
    const bool transparent = has(options, PipelineFlag::Transparent);
    const bool hud = has(options, PipelineFlag::Hud);
    const bool skinned = has(options, PipelineFlag::Skinned);
    const bool shadow = has(options, PipelineFlag::Shadow);
    const bool additive = has(options, PipelineFlag::Additive);
    const bool alpha_additive = has(options, PipelineFlag::AlphaAdditive);
    const bool multiply = has(options, PipelineFlag::Multiply);
    const bool multiply_2x = has(options, PipelineFlag::Multiply2x);
    const bool particle_set = has(options, PipelineFlag::ParticleSet);
    const bool water_depth_write = has(options, PipelineFlag::WaterDepthWrite);
    const bool wallmark = has(options, PipelineFlag::Wallmark);
    const bool legacy_multiply = has(options, PipelineFlag::LegacyMultiply);
    const bool replace_no_depth = has(options, PipelineFlag::ReplaceNoDepth);
    const bool hud_set_no_depth_write = has(options, PipelineFlag::HudSetNoDepthWrite);
    const bool cutout = has(options, PipelineFlag::Cutout);

    GraphicsShaderStages shader_stages;
    if (!build_graphics_shader_stages({vertex, game_stages.hull, game_stages.domain,
            game_stages.geometry, fragment}, shader_stages, error))
        return false;
    for (uint32_t index = 0; index < shader_stages.count; ++index)
    {
        VkPipelineShaderStageCreateInfo& stage = shader_stages.stages[index];
        switch (stage.stage)
        {
        case VK_SHADER_STAGE_VERTEX_BIT: stage.pSpecializationInfo = game_stages.vertex_specialization; break;
        case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT: stage.pSpecializationInfo = game_stages.hull_specialization; break;
        case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT: stage.pSpecializationInfo = game_stages.domain_specialization; break;
        case VK_SHADER_STAGE_GEOMETRY_BIT: stage.pSpecializationInfo = game_stages.geometry_specialization; break;
        case VK_SHADER_STAGE_FRAGMENT_BIT: stage.pSpecializationInfo = game_stages.fragment_specialization; break;
        default: break;
        }
    }
    const VkVertexInputBindingDescription binding{
        0, static_cast<uint32_t>(skinned ? sizeof(ModelVertex) : sizeof(LevelVertex)), VK_VERTEX_INPUT_RATE_VERTEX};
    const VkVertexInputAttributeDescription attributes[]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LevelVertex, normal)},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, uv)},
        {3, 0, VK_FORMAT_R16G16B16A16_UINT, offsetof(ModelVertex, bones)},
        {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(ModelVertex, weights)},
        {5, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, lightmap_uv)},
        {6, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(LevelVertex, baked)}};
    const VkVertexInputAttributeDescription skinned_attributes[]{
        attributes[0],
        attributes[1],
        attributes[2],
        attributes[3],
        attributes[4],
        {7, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ModelVertex, tangent)},
        {8, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ModelVertex, bitangent)}};
    const VkVertexInputAttributeDescription static_attributes[]{
        attributes[0],
        attributes[1],
        attributes[2],
        attributes[5],
        attributes[6],
        {7, 0, VK_FORMAT_R8G8B8A8_SNORM, offsetof(LevelVertex, tangent)},
        {8, 0, VK_FORMAT_R8G8B8A8_SNORM, offsetof(LevelVertex, bitangent)},
        {11, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, lightmap_uv)}};
    const bool instanced = has(options, PipelineFlag::Instanced);
    const VkVertexInputBindingDescription instance_bindings[]{
        binding, {1, sizeof(DetailInstance), VK_VERTEX_INPUT_RATE_INSTANCE}};
    const VkVertexInputAttributeDescription instance_attributes[]{
        static_attributes[0],
        static_attributes[1],
        static_attributes[2],
        static_attributes[3],
        static_attributes[4],
        static_attributes[5],
        static_attributes[6],
        static_attributes[7],
        {9, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(DetailInstance, position_scale)},
        {10, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(DetailInstance, rotation)}};
    VkPipelineVertexInputStateCreateInfo inputs{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    inputs.vertexBindingDescriptionCount = geometry_input ? (instanced ? 2 : 1) : 0;
    inputs.pVertexBindingDescriptions = geometry_input ? (instanced ? instance_bindings : &binding) : nullptr;
    inputs.vertexAttributeDescriptionCount = geometry_input ? (instanced ? 10 : skinned ? 7 : 8) : 0;
    inputs.pVertexAttributeDescriptions = geometry_input ? (instanced ? instance_attributes
                                                            : skinned ? skinned_attributes
                                                                      : static_attributes)
                                                         : nullptr;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = shader_stages.tessellation ? VK_PRIMITIVE_TOPOLOGY_PATCH_LIST :
        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineTessellationStateCreateInfo tessellation{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
    tessellation.patchControlPoints = 3;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    // Depth precision on shallow terrain slopes otherwise produces long,
    // repeating self-shadow bands. Apply bias while writing shadow depth,
    // in addition to the small receiver bias used by the lighting shader.
    const auto bias = geometry_depth_bias(shadow, wallmark, transparent && geometry_input && !water_depth_write);
    raster.depthBiasEnable = bias.enabled;
    raster.depthBiasConstantFactor = bias.constant;
    raster.depthBiasSlopeFactor = bias.slope;
    VkPipelineMultisampleStateCreateInfo multi{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multi.rasterizationSamples = shadow ? VK_SAMPLE_COUNT_1_BIT : vk.gbuffer_samples;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    // The present pass has its own cleared depth attachment. HUD geometry
    // draws after world lighting, so depth testing here orders HUD surfaces
    // against one another without occluding the HUD with world geometry.
    depth.depthTestEnable =
        (gbuffer || hud || transparent || shadow) && (!additive && !alpha_additive || transparent || hud) ? VK_TRUE
                                                                                                          : VK_FALSE;
    depth.depthWriteEnable = !legacy_multiply && ((gbuffer && !transparent) || (hud && !hud_set_no_depth_write) ||
                                                  shadow || particle_set || water_depth_write)
                                 ? VK_TRUE
                                 : VK_FALSE;
    // Decal polygons lie on top of the level surface. Strict LESS rejects
    // coplanar pixels and makes papers/wallmarks blink as the view moves.
    // R_Backend uses LESSEQUAL for ordinary geometry, including alpha decals.
    // Equality must survive the forward pass that loads the floor's depth.
    depth.depthCompareOp = geometry_depth_compare(shadow);
    VkPipelineColorBlendAttachmentState attachments[3]{};
    for (auto &attachment : attachments)
    {
        attachment.colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        attachment.blendEnable = transparent || hud ? VK_TRUE : VK_FALSE;
        attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        attachment.colorBlendOp = VK_BLEND_OP_ADD;
        attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        attachment.alphaBlendOp = VK_BLEND_OP_ADD;
        if (additive)
        {
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
            attachment.blendEnable = VK_TRUE;
            attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        }
        if (alpha_additive)
        {
            attachment.blendEnable = VK_TRUE;
            attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
            attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        }
        if (multiply || multiply_2x)
        {
            attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
            attachment.blendEnable = VK_TRUE;
            attachment.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
            attachment.dstColorBlendFactor = multiply_2x ? VK_BLEND_FACTOR_SRC_COLOR : VK_BLEND_FACTOR_ZERO;
        }
        if (particle_set || replace_no_depth)
            attachment.blendEnable = VK_FALSE;
    }
    if (legacy_multiply)
    {
        attachments[0].blendEnable = VK_TRUE;
        attachments[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
        attachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
        attachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR;
        attachments[1].colorWriteMask = 0;
        attachments[1].blendEnable = VK_FALSE;
        attachments[2].colorWriteMask = 0;
        attachments[2].blendEnable = VK_FALSE;
    }
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = shadow ? 0 : gbuffer ? 3 : 1;
    blend.pAttachments = attachments;
    const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.stageCount = shader_stages.count;
    info.pStages = shader_stages.stages.data();
    info.pVertexInputState = &inputs;
    info.pInputAssemblyState = &assembly;
    info.pTessellationState = shader_stages.tessellation ? &tessellation : nullptr;
    info.pViewportState = &viewport;
    info.pRasterizationState = &raster;
    info.pMultisampleState = &multi;
    info.pDepthStencilState = &depth;
    info.pColorBlendState = &blend;
    info.pDynamicState = &dynamic;
    info.layout = layout;
    info.renderPass = pass;
#if defined(XR_PLATFORM_ANDROID)
    char context[160];
    std::snprintf(context, sizeof(context), "vulkan pipeline: %s", label ? label : "unknown");
    android_set_load_context(context);
#endif
    const VkResult result = vk.create_graphics_pipelines(device, vk.pipeline_cache, 1, &info, nullptr, &pipeline);
#if defined(XR_PLATFORM_ANDROID)
    android_set_load_context("vulkan pipeline: created");
#endif
    if (result != VK_SUCCESS)
    {
        error = std::string("vkCreateGraphicsPipelines(") + label +
                ") failed: VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }
    return true;
}

void viewport_scissor(const FrameRecordingContext &frame, const ScenePassDispatch &vk)
{
    const VkViewport viewport{float(frame.viewport_offset.x), float(frame.viewport_offset.y),
        float(frame.extent.width), float(frame.extent.height), 0, 1};
    const VkRect2D scissor{frame.viewport_offset, frame.extent};
    vk.cmd_set_viewport(frame.command_buffer, 0, 1, &viewport);
    vk.cmd_set_scissor(frame.command_buffer, 0, 1, &scissor);
}
} // namespace

bool DeferredPass::create_visibility_gbuffer_pipeline(VkRenderPass render_pass, VkPipelineLayout layout,
    VkShaderModule vertex, VkShaderModule fragment, VkPipeline& pipeline, std::string& error) const
{
    return make_pipeline(device_, render_pass, layout, vertex, fragment, PipelineFlag::GBuffer,
        vk_, pipeline, error, "visibility G-buffer resolve");
}

void DeferredPass::draw_visibility_gbuffer(const FrameRecordingContext& frame, VkPipeline pipeline,
    VkPipelineLayout layout, VkDescriptorSet sources) const
{
    viewport_scissor(frame, vk_);
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &sources, 0, nullptr);
    vk_.cmd_draw(frame.command_buffer, 3, 1, 0, 0);
}

DeferredLight make_environment_deferred_light(const DeferredEnvironment &environment)
{
    const auto safe_nonnegative = [](float value) { return std::isfinite(value) ? std::max(value, 0.f) : 0.f; };
    float direction[3]{environment.sun_direction[0], environment.sun_direction[1], environment.sun_direction[2]};
    const float magnitude =
        std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
    if (std::isfinite(magnitude) && magnitude > 1e-6f)
        for (float &value : direction)
            value /= magnitude;
    else
    {
        direction[0] = 0.f;
        direction[1] = -1.f;
        direction[2] = 0.f;
    }

    const auto luminance = [&](const float(&color)[3]) {
        return 0.2126f * safe_nonnegative(color[0]) + 0.7152f * safe_nonnegative(color[1]) +
               0.0722f * safe_nonnegative(color[2]);
    };
    // The static hemisphere channel already records visibility. Keep its
    // weather intensity separate from ambient so indoor lightmaps can supply
    // the actual indirect contribution instead of multiplying a small ambient.
    const float ambient = luminance(environment.ambient_color);
    return {{direction[0], direction[1], direction[2], ambient},
            {safe_nonnegative(environment.sun_color[0]), safe_nonnegative(environment.sun_color[1]),
             safe_nonnegative(environment.sun_color[2]), 0.f},
            {1.f, 1.f, 1.f, luminance(environment.hemi_color)}};
}

bool DeferredPass::initialize(VkDevice device, VkRenderPass geometry_pass, VkRenderPass light_pass,
                              VkShaderModule geometry_vertex, VkShaderModule geometry_fragment,
                              VkShaderModule alpha_test_fragment, VkShaderModule transparent_fragment,
                              VkShaderModule light_vertex, VkShaderModule light_fragment,
                              VkShaderModule weather_fragment, const ScenePassDispatch &dispatch, std::string &error,
                              VkRenderPass shadow_pass, VkShaderModule shadow_vertex, VkShaderModule shadow_opaque,
                              VkShaderModule shadow_cutout, VkRenderPass local_shadow_pass,
                              VkShaderModule local_light_fragment, VkShaderModule water_fragment,
                              VkShaderModule shadow_skin, VkShaderModule multisampled_light_fragment,
                              VkShaderModule multisampled_weather_fragment,
                              VkShaderModule multisampled_local_light_fragment)
{
    destroy();
    if (!device || !geometry_pass || !light_pass || !geometry_vertex || !geometry_fragment || !alpha_test_fragment ||
        !transparent_fragment || !light_vertex || !light_fragment || !weather_fragment ||
        !dispatch.create_pipeline_layout || !dispatch.destroy_pipeline_layout || !dispatch.create_graphics_pipelines ||
        !dispatch.destroy_pipeline || !dispatch.create_descriptor_set_layout ||
        !dispatch.destroy_descriptor_set_layout || !dispatch.create_descriptor_pool ||
        !dispatch.destroy_descriptor_pool || !dispatch.allocate_descriptor_sets || !dispatch.update_descriptor_sets ||
        !dispatch.cmd_bind_pipeline || !dispatch.cmd_set_viewport || !dispatch.cmd_set_scissor ||
        !dispatch.cmd_bind_vertex_buffers || !dispatch.cmd_bind_index_buffer || !dispatch.cmd_bind_descriptor_sets ||
        !dispatch.cmd_push_constants || !dispatch.cmd_draw_indexed || !dispatch.cmd_draw)
    {
        error = "deferred pass requires Vulkan shaders and procedures";
        return false;
    }
    constexpr VkShaderStageFlags base_stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    constexpr VkShaderStageFlags optional_stages = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT |
        VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT | VK_SHADER_STAGE_GEOMETRY_BIT;
    if ((dispatch.game_stage_flags & base_stages) != base_stages ||
        (dispatch.game_stage_flags & ~(base_stages | optional_stages)) ||
        bool(dispatch.game_stage_flags & VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT) !=
            bool(dispatch.game_stage_flags & VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT))
    {
        error = "invalid Vulkan game shader stage mask";
        return false;
    }
    error.clear();
    device_ = device;
    geometry_pass_ = geometry_pass;
    light_pass_ = light_pass;
    shadow_pass_ = shadow_pass;
    local_shadow_pass_ = local_shadow_pass;
    vk_ = dispatch;
    game_stage_flags_ = dispatch.game_stage_flags;
#define VK_STEP(label, expression)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        const VkResult step_result = (expression);                                                                     \
        if (step_result != VK_SUCCESS)                                                                                 \
        {                                                                                                              \
            error = std::string(label) + " failed: VkResult=" + std::to_string(static_cast<int>(step_result));         \
            goto failed;                                                                                               \
        }                                                                                                              \
    } while (false)
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    const VkDescriptorSetLayoutBinding gbuffer_bindings[]{
        bindings[0],
        bindings[1],
        bindings[2],
        bindings[3],
        {4, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {10, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutBinding weather_bindings[]{
        bindings[0],
        bindings[1],
        bindings[2],
        bindings[3],
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {5, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    const VkDescriptorSetLayoutBinding pose_binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, game_stage_flags_,
                                                    nullptr};
    const VkDescriptorSetLayoutBinding forward_binding{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                                       game_stage_flags_, nullptr};
    const VkDescriptorSetLayoutBinding local_bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    const VkDescriptorSetLayoutBinding water_bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo descriptor{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    std::array<VkDescriptorSetLayoutBinding, SurfaceMaterialTextureCount> material_bindings{};
    for (uint32_t i = 0; i < material_bindings.size(); ++i)
        material_bindings[i] = {i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    for (auto &binding : material_bindings) binding.stageFlags = game_stage_flags_;
    for (auto &binding : weather_bindings) binding.stageFlags = game_stage_flags_;
    descriptor.bindingCount = static_cast<uint32_t>(material_bindings.size());
    descriptor.pBindings = material_bindings.data();
    VK_STEP("vkCreateDescriptorSetLayout(material)",
            vk_.create_descriptor_set_layout(device, &descriptor, nullptr, &material_layout_));
    descriptor.bindingCount = 1;
    descriptor.pBindings = &pose_binding;
    VK_STEP("vkCreateDescriptorSetLayout(pose)",
            vk_.create_descriptor_set_layout(device, &descriptor, nullptr, &pose_layout_));
    descriptor.pBindings = gbuffer_bindings;
    descriptor.bindingCount = 11;
    VK_STEP("vkCreateDescriptorSetLayout(gbuffer)",
            vk_.create_descriptor_set_layout(device, &descriptor, nullptr, &gbuffer_layout_));
    descriptor.pBindings = weather_bindings;
    descriptor.bindingCount = 6;
    VK_STEP("vkCreateDescriptorSetLayout(weather)",
            vk_.create_descriptor_set_layout(device, &descriptor, nullptr, &weather_set_layout_));
    descriptor.pBindings = local_bindings;
    descriptor.bindingCount = 3;
    VK_STEP("vkCreateDescriptorSetLayout(local light)",
            vk_.create_descriptor_set_layout(device, &descriptor, nullptr, &local_set_layout_));
    descriptor.pBindings = water_bindings;
    descriptor.bindingCount = 4;
    VK_STEP("vkCreateDescriptorSetLayout(water)",
            vk_.create_descriptor_set_layout(device, &descriptor, nullptr, &water_set_layout_));
    descriptor.pBindings = &forward_binding;
    descriptor.bindingCount = 1;
    VK_STEP("vkCreateDescriptorSetLayout(forward)",
            vk_.create_descriptor_set_layout(device, &descriptor, nullptr, &forward_layout_));
    {
        const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 393216},
                                           {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4096},
                                           {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}};
        VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pool.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pool.maxSets = 8192;
        pool.poolSizeCount = 3;
        pool.pPoolSizes = sizes;
        VK_STEP("vkCreateDescriptorPool(deferred)", vk_.create_descriptor_pool(device, &pool, nullptr, &pool_));
    }
    {
        // Material UV transforms occupy bytes 128..239 in the fragment stage.
        const VkPushConstantRange camera{game_stage_flags_, 0,
                                         sizeof(float) * 64};
        const VkPushConstantRange light{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(DeferredLight)};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.pushConstantRangeCount = 1;
        const VkDescriptorSetLayout forward_sets[]{material_layout_, forward_layout_, weather_set_layout_};
        layout.setLayoutCount = 3;
        layout.pSetLayouts = forward_sets;
        layout.pPushConstantRanges = &camera;
        VK_STEP("vkCreatePipelineLayout(geometry)",
                vk_.create_pipeline_layout(device, &layout, nullptr, &geometry_layout_));
        const VkDescriptorSetLayout skinned_sets[]{material_layout_, pose_layout_, forward_layout_,
                                                   weather_set_layout_};
        layout.setLayoutCount = 4;
        layout.pSetLayouts = skinned_sets;
        VK_STEP("vkCreatePipelineLayout(skinned)",
                vk_.create_pipeline_layout(device, &layout, nullptr, &skinned_layout_));
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &gbuffer_layout_;
        layout.pPushConstantRanges = &light;
        VK_STEP("vkCreatePipelineLayout(light)", vk_.create_pipeline_layout(device, &layout, nullptr, &light_layout_));
        const VkDescriptorSetLayout weather_sets[]{gbuffer_layout_, weather_set_layout_};
        const VkPushConstantRange weather_range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(WeatherLighting)};
        layout.setLayoutCount = 2;
        layout.pSetLayouts = weather_sets;
        layout.pPushConstantRanges = &weather_range;
        VK_STEP("vkCreatePipelineLayout(weather)",
                vk_.create_pipeline_layout(device, &layout, nullptr, &weather_layout_));
        const VkDescriptorSetLayout local_sets[]{gbuffer_layout_, local_set_layout_};
        layout.pSetLayouts = local_sets;
        layout.pushConstantRangeCount = 0;
        layout.pPushConstantRanges = nullptr;
        VK_STEP("vkCreatePipelineLayout(local light)",
                vk_.create_pipeline_layout(device, &layout, nullptr, &local_light_layout_));
        const VkDescriptorSetLayout water_sets[]{material_layout_, water_set_layout_};
        const VkPushConstantRange water_ranges[]{{VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float) * 16},
                                                 {VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(float) * 16, sizeof(float) * 4}};
        layout.pSetLayouts = water_sets;
        layout.pushConstantRangeCount = 2;
        layout.pPushConstantRanges = water_ranges;
        VK_STEP("vkCreatePipelineLayout(water)", vk_.create_pipeline_layout(device, &layout, nullptr, &water_layout_));
    }
    if (!rebuild_scene_pipelines(geometry_pass, light_pass, dispatch.gbuffer_samples,
            geometry_vertex, geometry_fragment, alpha_test_fragment, transparent_fragment,
            light_vertex, light_fragment, weather_fragment, local_light_fragment, water_fragment,
            multisampled_light_fragment, multisampled_weather_fragment, multisampled_local_light_fragment, error))
        goto failed;
    if (shadow_pass &&
        (!shadow_vertex || !shadow_opaque || !shadow_cutout ||
         !make_pipeline(device, shadow_pass, geometry_layout_, shadow_vertex, shadow_opaque,
                        PipelineFlag::Geometry | PipelineFlag::Shadow, vk_, shadow_opaque_, error, "shadow opaque") ||
         !make_pipeline(device, shadow_pass, geometry_layout_, shadow_vertex, shadow_cutout,
                        PipelineFlag::Geometry | PipelineFlag::Shadow, vk_, shadow_cutout_, error, "shadow cutout")))
        goto failed;
    if (shadow_pass && shadow_skin &&
        (!make_pipeline(device, shadow_pass, skinned_layout_, shadow_skin, shadow_opaque,
                        PipelineFlag::Geometry | PipelineFlag::Shadow | PipelineFlag::Skinned, vk_, shadow_skin_opaque_,
                        error, "GPU skinned shadow") ||
         !make_pipeline(device, shadow_pass, skinned_layout_, shadow_skin, shadow_cutout,
                        PipelineFlag::Geometry | PipelineFlag::Shadow | PipelineFlag::Skinned, vk_, shadow_skin_cutout_,
                        error, "GPU skinned cutout shadow")))
    {
        destroy();
        return false;
    }
    error.clear();
    return true;
failed:
    if (error.empty())
        error = "missing shader or render pass for Vulkan deferred pipeline";
    destroy();
    return false;
#undef VK_STEP
}

bool DeferredPass::rebuild_scene_pipelines(VkRenderPass geometry_pass, VkRenderPass light_pass, VkSampleCountFlagBits samples,
        VkShaderModule geometry_vertex, VkShaderModule geometry_fragment, VkShaderModule alpha_test_fragment,
        VkShaderModule transparent_fragment, VkShaderModule light_vertex, VkShaderModule light_fragment,
        VkShaderModule weather_fragment, VkShaderModule local_light_fragment, VkShaderModule water_fragment,
        VkShaderModule multisampled_light_fragment, VkShaderModule multisampled_weather_fragment,
        VkShaderModule multisampled_local_light_fragment, std::string& error)
{
    const VkDevice device = device_;
    const VkRenderPass shadow_pass = shadow_pass_, local_shadow_pass = local_shadow_pass_;
    auto dispatch = vk_;
    dispatch.gbuffer_samples = samples;
    struct Pipelines
    {
        VkPipeline geometry_{};
        VkPipeline alpha_test_{};
        VkPipeline transparent_{};
        VkPipeline hud_{};
        VkPipeline lighting_{};
        VkPipeline weather_pipeline_{};
        VkPipeline multisampled_lighting_{};
        VkPipeline multisampled_weather_pipeline_{};
        VkPipeline local_light_pipeline_{};
        VkPipeline multisampled_local_light_pipeline_{};
        VkPipeline water_pipeline_{};
    } next;
    const auto release = [&]
    {
        if (next.geometry_) vk_.destroy_pipeline(device_, next.geometry_, nullptr);
        if (next.alpha_test_) vk_.destroy_pipeline(device_, next.alpha_test_, nullptr);
        if (next.transparent_) vk_.destroy_pipeline(device_, next.transparent_, nullptr);
        if (next.hud_) vk_.destroy_pipeline(device_, next.hud_, nullptr);
        if (next.lighting_) vk_.destroy_pipeline(device_, next.lighting_, nullptr);
        if (next.weather_pipeline_) vk_.destroy_pipeline(device_, next.weather_pipeline_, nullptr);
        if (next.multisampled_lighting_) vk_.destroy_pipeline(device_, next.multisampled_lighting_, nullptr);
        if (next.multisampled_weather_pipeline_) vk_.destroy_pipeline(device_, next.multisampled_weather_pipeline_, nullptr);
        if (next.local_light_pipeline_) vk_.destroy_pipeline(device_, next.local_light_pipeline_, nullptr);
        if (next.multisampled_local_light_pipeline_) vk_.destroy_pipeline(device_, next.multisampled_local_light_pipeline_, nullptr);
        if (next.water_pipeline_) vk_.destroy_pipeline(device_, next.water_pipeline_, nullptr);
    };
    if (!make_pipeline(device, geometry_pass, geometry_layout_, geometry_vertex, geometry_fragment,
                       PipelineFlag::Geometry | PipelineFlag::GBuffer, dispatch, next.geometry_, error, "geometry") ||
        !make_pipeline(device, geometry_pass, geometry_layout_, geometry_vertex, alpha_test_fragment,
                       PipelineFlag::Geometry | PipelineFlag::GBuffer | PipelineFlag::Cutout, dispatch, next.alpha_test_, error,
                       "alpha test") ||
        !make_pipeline(device, light_pass, geometry_layout_, geometry_vertex, transparent_fragment,
                       PipelineFlag::Geometry | PipelineFlag::Transparent, dispatch, next.transparent_, error, "transparent") ||
        !make_pipeline(device, light_pass, geometry_layout_, geometry_vertex, transparent_fragment,
                       PipelineFlag::Geometry | PipelineFlag::Hud, dispatch, next.hud_, error, "HUD") ||
        !make_pipeline(device, light_pass, light_layout_, light_vertex, light_fragment, PipelineFlag::None, dispatch,
                       next.lighting_, error, "lighting") ||
        !make_pipeline(device, light_pass, weather_layout_, light_vertex, weather_fragment, PipelineFlag::None, dispatch,
                       next.weather_pipeline_, error, "weather"))
        goto failed;
    if (multisampled_light_fragment && multisampled_weather_fragment &&
        (!make_pipeline(device, light_pass, light_layout_, light_vertex, multisampled_light_fragment,
                        PipelineFlag::None, dispatch, next.multisampled_lighting_, error, "multisample lighting") ||
         !make_pipeline(device, light_pass, weather_layout_, light_vertex, multisampled_weather_fragment,
                        PipelineFlag::None, dispatch, next.multisampled_weather_pipeline_, error, "multisample weather")))
        goto failed;
    if (local_shadow_pass &&
        (!shadow_pass || !local_light_fragment ||
         !make_pipeline(device, light_pass, local_light_layout_, light_vertex, local_light_fragment,
                        PipelineFlag::Additive, dispatch, next.local_light_pipeline_, error, "local light")))
        goto failed;
    if (local_shadow_pass && multisampled_local_light_fragment &&
        !make_pipeline(device, light_pass, local_light_layout_, light_vertex, multisampled_local_light_fragment,
                       PipelineFlag::Additive, dispatch, next.multisampled_local_light_pipeline_, error,
                       "multisample local light"))
        goto failed;
    if (water_fragment &&
        !make_pipeline(device, light_pass, water_layout_, geometry_vertex, water_fragment,
                       PipelineFlag::Geometry | PipelineFlag::Transparent | PipelineFlag::WaterDepthWrite, dispatch,
                       next.water_pipeline_, error, "water"))
        goto failed;
    std::swap(geometry_, next.geometry_);
    std::swap(alpha_test_, next.alpha_test_);
    std::swap(transparent_, next.transparent_);
    std::swap(hud_, next.hud_);
    std::swap(lighting_, next.lighting_);
    std::swap(weather_pipeline_, next.weather_pipeline_);
    std::swap(multisampled_lighting_, next.multisampled_lighting_);
    std::swap(multisampled_weather_pipeline_, next.multisampled_weather_pipeline_);
    std::swap(local_light_pipeline_, next.local_light_pipeline_);
    std::swap(multisampled_local_light_pipeline_, next.multisampled_local_light_pipeline_);
    std::swap(water_pipeline_, next.water_pipeline_);
    release();
    geometry_pass_ = geometry_pass;
    light_pass_ = light_pass;
    vk_.gbuffer_samples = samples;
    error.clear();
    return true;
failed:
    release();
    if (error.empty())
        error = "missing shader or render pass for Vulkan deferred pipeline";
    return false;
}

bool DeferredPass::allocate(VkDescriptorSetLayout layout, VkImageView first, VkImageView second, VkSampler sampler,
                            VkDescriptorSet &set, std::string &error)
{
    set = VK_NULL_HANDLE;
    if (!pool_ || !first || !sampler || (layout == gbuffer_layout_ && !second))
    {
        error = "missing deferred texture or descriptor pool";
        return false;
    }
    VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    info.descriptorPool = pool_;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &layout;
    if (vk_.allocate_descriptor_sets(device_, &info, &set) != VK_SUCCESS)
    {
        error = "deferred texture descriptor pool is exhausted";
        return false;
    }
    const VkImageLayout image_layout =
        layout == material_layout_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    const VkDescriptorImageInfo images[]{{sampler, first, image_layout}, {sampler, second, image_layout}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < (second ? 2u : 1u); ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &images[i];
    }
    vk_.update_descriptor_sets(device_, second ? 2 : 1, writes, 0, nullptr);
    if (layout == material_layout_)
    {
        std::array<ShaderImageShape, SurfaceMaterialTextureCount> shapes{};
        shapes.fill({ShaderImageDimension::TwoD, false, false});
        material_image_shapes_[set] = shapes;
        std::array<VkImageView, SurfaceMaterialBaseTextureCount> base{};
        base.fill(first);
        if (second)
            base[1] = second;
        update_terrain_material(set, base, sampler);
        std::array<VkImageView, SurfaceMaterialExtraTextureCount> fallback{};
        fallback.fill(first);
        update_surface_textures(set, fallback, sampler);
    }
    error.clear();
    return true;
}

void DeferredPass::set_material_image_shapes(VkDescriptorSet set,
    const std::array<ShaderImageShape, SurfaceMaterialTextureCount>& shapes)
{
    if (set)
        material_image_shapes_[set] = shapes;
}

bool DeferredPass::material_image_shapes_match(VkDescriptorSet set,
    const std::array<ShaderImageShape, SurfaceMaterialTextureCount>& expected,
    const std::array<uint32_t, SurfaceMaterialTextureCount>& counts) const
{
    const auto found = material_image_shapes_.find(set);
    if (found == material_image_shapes_.end())
        return false;
    for (size_t binding = 0; binding < expected.size(); ++binding)
    {
        const ShaderImageShape& shader_shape = expected[binding];
        if (shader_shape.dimension == ShaderImageDimension::Unknown)
            continue;
        const size_t count = std::max<size_t>(counts[binding], 1);
        if (count > expected.size() - binding)
            return false;
        for (size_t element = 0; element < count; ++element)
        {
            const ShaderImageShape& view_shape = found->second[binding + element];
            if (shader_shape.dimension != view_shape.dimension ||
                shader_shape.arrayed != view_shape.arrayed ||
                shader_shape.multisampled != view_shape.multisampled)
                return false;
        }
    }
    return true;
}

bool DeferredPass::write_material_descriptor_set(VkDescriptorSet target,
    const std::array<uint32_t, SurfaceMaterialTextureCount>& counts,
    const std::array<VkDescriptorImageInfo, SurfaceMaterialTextureCount>& images) const
{
    if (!target || !vk_.update_descriptor_sets)
        return false;
    size_t image_count = 0;
    for (uint32_t binding = 0; binding < counts.size(); ++binding)
    {
        const uint32_t count = std::max(counts[binding], 1u);
        if (count > counts.size() - binding)
            return false;
        image_count += count;
        for (uint32_t element = 0; element < count; ++element)
            if (!images[binding + element].imageView || !images[binding + element].sampler)
                return false;
    }
    std::vector<VkDescriptorImageInfo> write_images;
    std::vector<VkWriteDescriptorSet> writes;
    write_images.reserve(image_count);
    writes.reserve(counts.size());
    for (uint32_t binding = 0; binding < counts.size(); ++binding)
    {
        const uint32_t count = std::max(counts[binding], 1u);
        const size_t first_image = write_images.size();
        for (uint32_t element = 0; element < count; ++element)
            write_images.push_back(images[binding + element]);
        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.dstSet = target;
        write.dstBinding = binding;
        write.descriptorCount = count;
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = write_images.data() + first_image;
        writes.push_back(write);
    }
    vk_.update_descriptor_sets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}

void DeferredPass::refresh_material_descriptor_sets(VkDescriptorSet set)
{
    const auto images = material_image_descriptors_.find(set);
    const auto variants = material_descriptor_variants_.find(set);
    if (images == material_image_descriptors_.end() || variants == material_descriptor_variants_.end())
        return;
    for (const auto& [layout, variant] : variants->second)
        write_material_descriptor_set(variant.set, variant.counts, images->second);
}

VkDescriptorSet DeferredPass::material_descriptor_set(VkDescriptorSet set, const GamePipeline* pipeline) const
{
    if (!pipeline || !pipeline->material_layout)
        return set;
    const auto images = material_image_descriptors_.find(set);
    if (images == material_image_descriptors_.end() || !pool_ || !vk_.allocate_descriptor_sets)
        return VK_NULL_HANDLE;
    auto& variants = material_descriptor_variants_[set];
    if (const auto found = variants.find(pipeline->material_layout); found != variants.end())
        return found->second.set;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &pipeline->material_layout;
    VkDescriptorSet result{};
    if (vk_.allocate_descriptor_sets(device_, &allocate, &result) != VK_SUCCESS || !result)
        return VK_NULL_HANDLE;
    if (!write_material_descriptor_set(result, pipeline->material_descriptor_counts, images->second))
    {
        if (vk_.free_descriptor_sets)
            vk_.free_descriptor_sets(device_, pool_, 1, &result);
        return VK_NULL_HANDLE;
    }
    variants.emplace(pipeline->material_layout,
        MaterialDescriptorVariant{result, pipeline->material_descriptor_counts});
    return result;
}

bool DeferredPass::weather_image_shapes_match(VkDescriptorSet set,
    const std::array<ShaderImageShape, 6>& expected, const std::array<uint32_t, 6>& counts) const
{
    const auto found = weather_image_shapes_.find(set);
    if (found == weather_image_shapes_.end())
        return false;
    for (size_t binding = 0; binding < expected.size(); ++binding)
    {
        const ShaderImageShape& shader_shape = expected[binding];
        if (shader_shape.dimension == ShaderImageDimension::Unknown)
            continue;
        const size_t count = std::max<size_t>(counts[binding], 1);
        if (count > expected.size() - binding)
            return false;
        for (size_t element = 0; element < count; ++element)
        {
            const ShaderImageShape& view_shape = found->second[binding + element];
            if (view_shape.dimension == ShaderImageDimension::Unknown ||
                shader_shape.dimension != view_shape.dimension ||
                shader_shape.arrayed != view_shape.arrayed ||
                shader_shape.multisampled != view_shape.multisampled)
                return false;
        }
    }
    return true;
}

bool DeferredPass::write_weather_descriptor_set(VkDescriptorSet target, const std::array<uint32_t, 6>& counts,
    const std::array<VkDescriptorImageInfo, 6>& images) const
{
    if (!target || !vk_.update_descriptor_sets)
        return false;
    for (uint32_t binding = 0; binding < counts.size(); ++binding)
    {
        const uint32_t count = std::max(counts[binding], 1u);
        if (count > counts.size() - binding)
            return false;
        for (uint32_t element = 0; element < count; ++element)
            if (!images[binding + element].imageView || !images[binding + element].sampler)
                return false;
    }
    std::array<VkWriteDescriptorSet, 6> writes{};
    for (uint32_t binding = 0; binding < counts.size(); ++binding)
    {
        VkWriteDescriptorSet& write = writes[binding];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = target;
        write.dstBinding = binding;
        write.descriptorCount = std::max(counts[binding], 1u);
        write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write.pImageInfo = images.data() + binding;
    }
    vk_.update_descriptor_sets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    return true;
}

VkDescriptorSet DeferredPass::weather_descriptor_set(VkDescriptorSet set, const GamePipeline* pipeline) const
{
    if (!pipeline || !pipeline->weather_layout)
        return set;
    const auto images = weather_image_descriptors_.find(set);
    if (images == weather_image_descriptors_.end() || !pool_ || !vk_.allocate_descriptor_sets ||
        !weather_image_shapes_match(set, pipeline->weather_image_shapes, pipeline->weather_descriptor_counts))
        return VK_NULL_HANDLE;
    auto& variants = weather_descriptor_variants_[set];
    if (const auto found = variants.find(pipeline->weather_layout); found != variants.end())
        return found->second;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &pipeline->weather_layout;
    VkDescriptorSet result{};
    if (vk_.allocate_descriptor_sets(device_, &allocate, &result) != VK_SUCCESS || !result)
        return VK_NULL_HANDLE;
    if (!write_weather_descriptor_set(result, pipeline->weather_descriptor_counts, images->second))
    {
        if (vk_.free_descriptor_sets)
            vk_.free_descriptor_sets(device_, pool_, 1, &result);
        return VK_NULL_HANDLE;
    }
    variants.emplace(pipeline->weather_layout, result);
    return result;
}

void DeferredPass::release_material_descriptor_sets(VkDescriptorSet set)
{
    const auto found = material_descriptor_variants_.find(set);
    if (found != material_descriptor_variants_.end() && device_ && pool_ && vk_.free_descriptor_sets)
        for (const auto& [layout, variant] : found->second)
            if (variant.set)
                vk_.free_descriptor_sets(device_, pool_, 1, &variant.set);
    if (found != material_descriptor_variants_.end())
        material_descriptor_variants_.erase(found);
    const auto weather = weather_descriptor_variants_.find(set);
    if (weather != weather_descriptor_variants_.end() && device_ && pool_ && vk_.free_descriptor_sets)
        for (const auto& [layout, variant] : weather->second)
            if (variant)
                vk_.free_descriptor_sets(device_, pool_, 1, &variant);
    if (weather != weather_descriptor_variants_.end())
        weather_descriptor_variants_.erase(weather);
}

void DeferredPass::release_all_material_descriptor_sets()
{
    while (!material_descriptor_variants_.empty() || !weather_descriptor_variants_.empty())
    {
        if (!material_descriptor_variants_.empty())
            release_material_descriptor_sets(material_descriptor_variants_.begin()->first);
        else
            release_material_descriptor_sets(weather_descriptor_variants_.begin()->first);
    }
}

void DeferredPass::destroy_game_pipeline(GamePipeline& pipeline)
{
    if (!device_)
        return;
    if (pipeline.handle && vk_.destroy_pipeline)
        vk_.destroy_pipeline(device_, pipeline.handle, nullptr);
    if (pipeline.layout && vk_.destroy_pipeline_layout)
        vk_.destroy_pipeline_layout(device_, pipeline.layout, nullptr);
    if (pipeline.material_layout && vk_.destroy_descriptor_set_layout)
        vk_.destroy_descriptor_set_layout(device_, pipeline.material_layout, nullptr);
    if (pipeline.weather_layout && vk_.destroy_descriptor_set_layout)
        vk_.destroy_descriptor_set_layout(device_, pipeline.weather_layout, nullptr);
    pipeline = {};
}

bool DeferredPass::material(VkImageView albedo, VkSampler sampler, VkDescriptorSet &set, std::string &error)
{
    return allocate(material_layout_, albedo, VK_NULL_HANDLE, sampler, set, error);
}

bool DeferredPass::lightmapped_material(VkImageView albedo, VkImageView lightmap, VkSampler sampler,
                                        VkDescriptorSet &set, std::string &error, VkSampler lightmap_sampler)
{
    if (!albedo || !lightmap)
    {
        error = "lightmapped material needs both textures";
        return false;
    }
    if (!allocate(material_layout_, albedo, lightmap, sampler, set, error))
        return false;
    std::array<VkImageView, SurfaceMaterialBaseTextureCount> views;
    views.fill(albedo);
    views[1] = lightmap;
    update_terrain_material(set, views, sampler, lightmap_sampler);
    return true;
}

bool DeferredPass::terrain_material(
    const std::array<VkImageView, SurfaceMaterialBaseTextureCount> &views, VkSampler sampler, VkDescriptorSet &set,
                                    std::string &error, VkSampler lightmap_sampler)
{
    if (std::any_of(views.begin(), views.end(), [](VkImageView view) { return !view; }))
    {
        error = "terrain material needs all seven textures";
        return false;
    }
    if (!allocate(material_layout_, views[0], views[1], sampler, set, error))
        return false;
    update_terrain_material(set, views, sampler, lightmap_sampler);
    return true;
}

void DeferredPass::push_surface(const FrameRecordingContext &frame, VkPipelineLayout layout, VkDescriptorSet material,
                                uint32_t offset) const
{
    const auto found = surface_parameters_.find(material);
    const SurfaceParameters empty{};
    SurfaceParameters parameters = found == surface_parameters_.end() ? empty : found->second;
    uint32_t map_flags = static_cast<uint32_t>(std::max(parameters.external_normal, 0.f) + 0.5f);
    if (steep_parallax)
        map_flags |= SurfaceMapSteepParallax;
    else
        map_flags &= ~SurfaceMapSteepParallax;
    parameters.external_normal = static_cast<float>(map_flags);
    command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_,
                        offset, sizeof(parameters), &parameters);
}

void DeferredPass::update_surface_textures(VkDescriptorSet set,
                                           const std::array<VkImageView, SurfaceMaterialExtraTextureCount> &views,
                                           VkSampler sampler)
{
    if (!device_ || !set)
        return;
    std::array<VkDescriptorImageInfo, SurfaceMaterialExtraTextureCount> images{};
    std::array<VkWriteDescriptorSet, SurfaceMaterialExtraTextureCount> writes{};
    for (uint32_t i = 0; i < views.size(); ++i)
    {
        images[i] = {sampler, views[i], VK_IMAGE_LAYOUT_GENERAL};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i + 7;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &images[i];
    }
    vk_.update_descriptor_sets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    auto& material_images = material_image_descriptors_[set];
    for (uint32_t i = 0; i < views.size(); ++i)
        material_images[i + 7] = images[i];
    refresh_material_descriptor_sets(set);
}

const DeferredPass::GamePipeline *DeferredPass::game_pipeline(const char *vertex_name, const char *fragment_name,
    uint64_t variant_hash) const
{
    if (!vertex_name || !fragment_name || !*vertex_name || !*fragment_name)
        return nullptr;
    if (const auto *cached = pipeline_lookup_cache_.find(vertex_name, fragment_name, variant_hash))
        return cached;
    std::string key(vertex_name);
    key.push_back('\0');
    key += fragment_name;
    key.push_back('\0');
    key += std::to_string(variant_hash);
    const auto found = game_pipelines_.find(key);
    if (found == game_pipelines_.end())
        return nullptr;
    pipeline_lookup_cache_.remember(vertex_name, fragment_name, &found->second, variant_hash);
    return &found->second;
}

bool DeferredPass::has_game_pipeline(const std::string &vertex_name, const std::string &fragment_name,
    uint64_t variant_hash) const
{
    return game_pipeline(vertex_name.c_str(), fragment_name.c_str(), variant_hash) != nullptr;
}

bool DeferredPass::require_game_pipeline(const char *vertex_name, const char *fragment_name, SurfaceMode mode, bool hud,
                                         bool skinned, std::string &error, uint64_t variant_hash) const
{
    const auto *pair = game_pipeline(vertex_name, fragment_name, variant_hash);
    if (!pair)
        error = "unknown Vulkan SVS/SPS pair: " + std::string(vertex_name ? vertex_name : "<null>") + " / " +
                (fragment_name ? fragment_name : "<null>");
    else if (pair->mode != mode || pair->hud != hud || pair->skinned != skinned)
        error = "incompatible Vulkan SVS/SPS pass parameters: " + std::string(vertex_name) + " / " + fragment_name;
    else
    {
        error.clear();
        return true;
    }
    return false;
}

void DeferredPass::abort_game_pipeline_reload()
{
    for (auto& [name, pipeline] : pending_game_pipelines_)
        destroy_game_pipeline(pipeline);
    pending_game_pipelines_.clear();
    reloading_game_pipelines_ = false;
}

void DeferredPass::begin_game_pipeline_reload()
{
    abort_game_pipeline_reload();
    reloading_game_pipelines_ = true;
}

void DeferredPass::commit_game_pipeline_reload()
{
    if (!reloading_game_pipelines_)
        return;
    pipeline_lookup_cache_.clear();
    release_all_material_descriptor_sets();
    for (auto& [name, pipeline] : game_pipelines_)
        destroy_game_pipeline(pipeline);
    game_pipelines_ = std::move(pending_game_pipelines_);
    pending_game_pipelines_.clear();
    reloading_game_pipelines_ = false;
}

bool DeferredPass::create_game_pipeline(const std::string &vertex_name, const std::string &fragment_name,
                                         VkShaderModule vertex, VkShaderModule fragment, SurfaceMode mode, bool hud,
                                         std::string &error, bool skinned)
{
    return create_game_pipeline(vertex_name, fragment_name, vertex, fragment, mode, hud, error,
        skinned, GamePipelineStages{});
}

bool DeferredPass::create_game_pipeline(const std::string &vertex_name, const std::string &fragment_name,
                                        VkShaderModule vertex, VkShaderModule fragment, SurfaceMode mode, bool hud,
                                         std::string &error, bool skinned, const GamePipelineStages &stages)
{
    const auto has_suffix = [](const std::string &name, const char *suffix) {
        const size_t length = std::strlen(suffix);
        return name.size() > length && name.compare(name.size() - length, length, suffix) == 0;
    };
    if (!has_suffix(vertex_name, ".vs") || !has_suffix(fragment_name, ".ps"))
    {
        error = "invalid Vulkan SVS/SPS shader stages: " + vertex_name + " / " + fragment_name;
        return false;
    }
    if (!device_ || vertex_name.empty() || fragment_name.empty() || !vertex || !fragment ||
        (hud && mode == SurfaceMode::AlphaTest))
    {
        error = "invalid Vulkan game shader pair or pass state: " + vertex_name + " / " + fragment_name;
        return false;
    }
    if (bool(stages.hull) != bool(stages.domain) ||
        bool(stages.hull) != !stages.hull_name.empty() ||
        bool(stages.domain) != !stages.domain_name.empty() ||
        bool(stages.geometry) != !stages.geometry_name.empty() ||
        (!stages.hull_name.empty() && !has_suffix(stages.hull_name, ".hs")) ||
        (!stages.domain_name.empty() && !has_suffix(stages.domain_name, ".ds")) ||
        (!stages.geometry_name.empty() && !has_suffix(stages.geometry_name, ".gs")))
    {
        error = "invalid Vulkan optional game shader stages: " + vertex_name + " / " + fragment_name;
        return false;
    }
    if ((stages.hull && (game_stage_flags_ &
            (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) !=
            (VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT)) ||
        (stages.geometry && !(game_stage_flags_ & VK_SHADER_STAGE_GEOMETRY_BIT)))
    {
        error = "Vulkan game shader stages are missing from the pipeline layout";
        return false;
    }
    std::string key = vertex_name;
    key.push_back('\0');
    key += fragment_name;
    key.push_back('\0');
    key += std::to_string(stages.shader_variant_hash);
    auto &registry = reloading_game_pipelines_ ? pending_game_pipelines_ : game_pipelines_;
    const auto existing = registry.find(key);
    if (existing != registry.end())
    {
        if (existing->second.mode != mode || existing->second.hud != hud || existing->second.skinned != skinned ||
            existing->second.hull_name != stages.hull_name ||
            existing->second.domain_name != stages.domain_name ||
            existing->second.geometry_name != stages.geometry_name ||
            existing->second.specialization_hash != stages.specialization_hash ||
            existing->second.shader_fingerprint != stages.shader_fingerprint ||
            existing->second.shader_variant_hash != stages.shader_variant_hash ||
            existing->second.material_descriptor_counts != stages.material_descriptor_counts ||
            existing->second.weather_descriptor_counts != stages.weather_descriptor_counts)
        {
            error = "Vulkan game shader pair requested with incompatible pass state: " + vertex_name + " / " +
                    fragment_name;
            return false;
        }
        error.clear();
        return true;
    }
    VkPipeline pipeline{};
    VkDescriptorSetLayout custom_material_layout{}, custom_weather_layout{};
    VkPipelineLayout custom_pipeline_layout{};
    VkPipelineLayout game_layout = skinned ? skinned_layout_ : geometry_layout_;
    bool has_material_arrays = false, has_weather_arrays = false;
    for (uint32_t binding = 0; binding < stages.material_descriptor_counts.size(); ++binding)
    {
        const uint32_t count = stages.material_descriptor_counts[binding];
        if (!count || count > stages.material_descriptor_counts.size() - binding)
        {
            error = "invalid Vulkan game material descriptor count at binding " + std::to_string(binding);
            return false;
        }
        has_material_arrays |= count > 1;
    }
    for (uint32_t binding = 0; binding < stages.weather_descriptor_counts.size(); ++binding)
    {
        const uint32_t count = stages.weather_descriptor_counts[binding];
        if (!count || count > stages.weather_descriptor_counts.size() - binding)
        {
            error = "invalid Vulkan game weather descriptor count at binding " + std::to_string(binding);
            return false;
        }
        has_weather_arrays |= count > 1;
    }
    if (has_material_arrays || has_weather_arrays)
    {
        if (!vk_.free_descriptor_sets)
        {
            error = "Vulkan game descriptor arrays require vkFreeDescriptorSets";
            return false;
        }
        if (has_material_arrays)
        {
            std::array<VkDescriptorSetLayoutBinding, SurfaceMaterialTextureCount> bindings{};
            for (uint32_t binding = 0; binding < bindings.size(); ++binding)
                bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    stages.material_descriptor_counts[binding], game_stage_flags_, nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            descriptor_layout.bindingCount = static_cast<uint32_t>(bindings.size());
            descriptor_layout.pBindings = bindings.data();
            if (!vk_.create_descriptor_set_layout ||
                vk_.create_descriptor_set_layout(device_, &descriptor_layout, nullptr,
                    &custom_material_layout) != VK_SUCCESS)
            {
                error = "failed to create Vulkan game material array descriptor layout";
                return false;
            }
        }
        if (has_weather_arrays)
        {
            std::array<VkDescriptorSetLayoutBinding, 6> bindings{};
            for (uint32_t binding = 0; binding < bindings.size(); ++binding)
                bindings[binding] = {binding, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    stages.weather_descriptor_counts[binding], game_stage_flags_, nullptr};
            VkDescriptorSetLayoutCreateInfo descriptor_layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
            descriptor_layout.bindingCount = static_cast<uint32_t>(bindings.size());
            descriptor_layout.pBindings = bindings.data();
            if (!vk_.create_descriptor_set_layout ||
                vk_.create_descriptor_set_layout(device_, &descriptor_layout, nullptr,
                    &custom_weather_layout) != VK_SUCCESS)
            {
                if (custom_material_layout && vk_.destroy_descriptor_set_layout)
                    vk_.destroy_descriptor_set_layout(device_, custom_material_layout, nullptr);
                error = "failed to create Vulkan game weather array descriptor layout";
                return false;
            }
        }
        const VkDescriptorSetLayout material_set = custom_material_layout ? custom_material_layout : material_layout_;
        const VkDescriptorSetLayout weather_set = custom_weather_layout ? custom_weather_layout : weather_set_layout_;
        const VkDescriptorSetLayout forward_sets[]{material_set, forward_layout_, weather_set};
        const VkDescriptorSetLayout skinned_sets[]{material_set, pose_layout_, forward_layout_, weather_set};
        const VkPushConstantRange camera{game_stage_flags_, 0, sizeof(float) * 64};
        VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pipeline_layout.setLayoutCount = skinned ? 4 : 3;
        pipeline_layout.pSetLayouts = skinned ? skinned_sets : forward_sets;
        pipeline_layout.pushConstantRangeCount = 1;
        pipeline_layout.pPushConstantRanges = &camera;
        if (!vk_.create_pipeline_layout ||
            vk_.create_pipeline_layout(device_, &pipeline_layout, nullptr, &custom_pipeline_layout) != VK_SUCCESS)
        {
            if (vk_.destroy_descriptor_set_layout)
            {
                if (custom_weather_layout)
                    vk_.destroy_descriptor_set_layout(device_, custom_weather_layout, nullptr);
                if (custom_material_layout)
                    vk_.destroy_descriptor_set_layout(device_, custom_material_layout, nullptr);
            }
            error = "failed to create Vulkan game descriptor array pipeline layout";
            return false;
        }
        game_layout = custom_pipeline_layout;
    }
    const bool transparent = mode == SurfaceMode::Transparent;
    const bool particle_additive = (fragment_name.find("particle_") != std::string::npos ||
                                    fragment_name.find("world_set_") != std::string::npos ||
                                    fragment_name.find("model_set_") != std::string::npos) &&
                                   fragment_name.find("_additive.ps") != std::string::npos;
    const bool particle_alpha_add = fragment_name.find("particle_") != std::string::npos &&
                                    fragment_name.find("_alpha_add.ps") != std::string::npos;
    const bool glow_alpha_add = fragment_name == "vk\\glow_alpha_add.ps" ||
                                fragment_name == "vk\\model_set_alpha_add.ps" ||
                                fragment_name == "vk\\hud_model_set_alpha_add.ps";
    const bool particle_multiply = (fragment_name.find("particle_") != std::string::npos ||
                                    fragment_name.find("world_set_") != std::string::npos ||
                                    fragment_name.find("model_set_") != std::string::npos) &&
                                   fragment_name.find("_multiply.ps") != std::string::npos;
    const bool particle_multiply_2x = (fragment_name.find("particle_") != std::string::npos ||
                                       fragment_name.find("world_set_") != std::string::npos ||
                                       fragment_name.find("model_set_") != std::string::npos) &&
                                      fragment_name.find("_multiply_2x.ps") != std::string::npos;
    const bool wallmark_multiply_2x = fragment_name == "vk\\wallmark_multiply_2x.ps" ||
                                      fragment_name == "vk\\model_set_wallmark_multiply_2x.ps" ||
                                      fragment_name == "vk\\hud_model_set_wallmark_multiply_2x.ps";
    const bool particle_set =
        fragment_name.find("particle_") != std::string::npos && fragment_name.find("_set.ps") != std::string::npos;
    const bool world_set_depth_write =
        fragment_name == "vk\\world_set_set.ps" || fragment_name == "vk\\level_priority_opaque.ps" ||
        fragment_name == "vk\\level_priority_lightmap.ps" || fragment_name == "vk\\object_priority_opaque.ps" ||
        fragment_name == "vk\\level_priority_detail.ps" ||
        fragment_name == "vk\\level_priority_detail_lightmap.ps" ||
        fragment_name == "vk\\skinned_priority_opaque.ps" || fragment_name == "vk\\model_set_set.ps" ||
        fragment_name == "vk\\hud_model_set_set.ps";
    const bool world_set_replace =
        fragment_name == "vk\\world_set_set_2r.ps" || fragment_name == "vk\\level_priority_opaque.ps" ||
        fragment_name == "vk\\level_priority_lightmap.ps" || fragment_name == "vk\\object_priority_opaque.ps" ||
        fragment_name == "vk\\level_priority_detail.ps" ||
        fragment_name == "vk\\level_priority_detail_lightmap.ps" ||
        fragment_name == "vk\\skinned_priority_opaque.ps" || fragment_name == "vk\\model_set_set_2r.ps" ||
        fragment_name == "vk\\hud_model_set_set_2r.ps";
    const bool hud_set_no_depth_write =
        hud &&
        ((fragment_name.find("vk\\hud_model_set_") == 0 && !world_set_depth_write) ||
         (fragment_name.find("vk\\hud_particle_screen_") == 0 && fragment_name != "vk\\hud_particle_screen_set.ps"));
    const bool legacy_multiply = fragment_name == "vk\\level_laemb_multiply.ps";
    if (!make_pipeline(
            device_, transparent || hud ? light_pass_ : geometry_pass_, game_layout,
            vertex, fragment,
            (vertex_name == "vk\\level_detail_instanced.vs" ? PipelineFlag::Instanced : PipelineFlag::None) |
                PipelineFlag::Geometry | (!transparent && !hud ? PipelineFlag::GBuffer : PipelineFlag::None) |
                (transparent ? PipelineFlag::Transparent : PipelineFlag::None) |
                (hud ? PipelineFlag::Hud : PipelineFlag::None) |
                (skinned ? PipelineFlag::Skinned : PipelineFlag::None) |
                (particle_additive ? PipelineFlag::Additive : PipelineFlag::None) |
                (particle_alpha_add || glow_alpha_add ? PipelineFlag::AlphaAdditive : PipelineFlag::None) |
                (particle_multiply ? PipelineFlag::Multiply : PipelineFlag::None) |
                (particle_multiply_2x || wallmark_multiply_2x ? PipelineFlag::Multiply2x : PipelineFlag::None) |
                (particle_set || world_set_depth_write ? PipelineFlag::ParticleSet : PipelineFlag::None) |
                (wallmark_multiply_2x ? PipelineFlag::Wallmark : PipelineFlag::None) |
                (legacy_multiply ? PipelineFlag::LegacyMultiply : PipelineFlag::None) |
                (world_set_replace ? PipelineFlag::ReplaceNoDepth : PipelineFlag::None) |
                (hud_set_no_depth_write ? PipelineFlag::HudSetNoDepthWrite : PipelineFlag::None) |
                (mode == SurfaceMode::AlphaTest ? PipelineFlag::Cutout : PipelineFlag::None),
            vk_, pipeline, error, "game shader pair", stages))
    {
        if (custom_pipeline_layout && vk_.destroy_pipeline_layout)
            vk_.destroy_pipeline_layout(device_, custom_pipeline_layout, nullptr);
        if (custom_material_layout && vk_.destroy_descriptor_set_layout)
            vk_.destroy_descriptor_set_layout(device_, custom_material_layout, nullptr);
        if (custom_weather_layout && vk_.destroy_descriptor_set_layout)
            vk_.destroy_descriptor_set_layout(device_, custom_weather_layout, nullptr);
        error += ": " + vertex_name + " / " + fragment_name;
        return false;
    }
    GamePipeline game_pipeline;
    game_pipeline.handle = pipeline;
    game_pipeline.layout = custom_pipeline_layout;
    game_pipeline.material_layout = custom_material_layout;
    game_pipeline.weather_layout = custom_weather_layout;
    game_pipeline.mode = mode;
    game_pipeline.hud = hud;
    game_pipeline.skinned = skinned;
    game_pipeline.hull_name = stages.hull_name;
    game_pipeline.domain_name = stages.domain_name;
    game_pipeline.geometry_name = stages.geometry_name;
    game_pipeline.specialization_hash = stages.specialization_hash;
    game_pipeline.shader_fingerprint = stages.shader_fingerprint;
    game_pipeline.shader_variant_hash = stages.shader_variant_hash;
    game_pipeline.material_image_shapes = stages.material_image_shapes;
    game_pipeline.material_descriptor_counts = stages.material_descriptor_counts;
    game_pipeline.weather_image_shapes = stages.weather_image_shapes;
    game_pipeline.weather_descriptor_counts = stages.weather_descriptor_counts;
    registry.emplace(std::move(key), std::move(game_pipeline));
    error.clear();
    return true;
}

bool DeferredPass::pose_descriptor(VkBuffer pose, VkDeviceSize bytes, VkDescriptorSet &result, std::string &error)
{
    result = VK_NULL_HANDLE;
    if (!device_ || !pose_layout_ || !pool_ || !pose || !bytes)
    {
        error = "invalid Vulkan skeletal pose buffer";
        return false;
    }
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &pose_layout_;
    if (vk_.allocate_descriptor_sets(device_, &allocate, &result) != VK_SUCCESS)
    {
        error = "Vulkan skeletal pose descriptor pool is exhausted";
        return false;
    }
    const VkDescriptorBufferInfo buffer{pose, 0, bytes};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = result;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &buffer;
    vk_.update_descriptor_sets(device_, 1, &write, 0, nullptr);
    error.clear();
    return true;
}

void DeferredPass::release_pose_descriptor(VkDescriptorSet &set)
{
    if (device_ && pool_ && set && vk_.free_descriptor_sets)
        vk_.free_descriptor_sets(device_, pool_, 1, &set);
    set = VK_NULL_HANDLE;
}

void DeferredPass::push_texture_matrices(const FrameRecordingContext &frame, VkPipelineLayout layout,
                                         const TextureMatrixSet *matrices, const float *normal_rows, bool tree_wave) const
{
    // Seven float4 rows: base UV, detail UV, environment UV/direction.
    float rows[28]{};
    rows[0] = rows[5] = rows[8] = rows[13] = rows[16] = rows[21] = rows[26] = 1.f;
    if (matrices)
        for (size_t slot = 0; slot < 3; ++slot)
        {
            if (!matrices->enabled[slot])
                continue;
            const auto result =
                matrices->values[slot].evaluate(Device.fTimeGlobal, reinterpret_cast<const float *>(&Device.mView));
            const size_t first = slot == 0 ? 0 : slot == 1 ? 2 : 4;
            for (size_t row = 0; row < (slot == 2 ? 3u : 2u); ++row)
                std::copy(result[row].begin(), result[row].end(), rows + (first + row) * 4);
            if (slot == 2 && matrices->values[slot].mode == 2)
                rows[27] = 1.f; // sphere reflection uses the world normal
        }
    // UV matrix rows use x, y and w (the coordinate's z is always zero).
    // Preserve the transform, including its translation in w, and carry
    // packed model hemi pairs in the unused z components of base U/V and detail U.
    for (unsigned axis = 0; axis < 3; ++axis)
        rows[axis * 4 + 2] =
            pack_model_hemi(normal_rows ? normal_rows[axis * 4 + 3] : -1.f, model_hemi_negative_[axis]);
    if (tree_wave) { rows[2] = tree_wind_.direction[0]; rows[6] = tree_wind_.direction[1]; }
    command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_,
                        128, sizeof(rows), rows);
    auto rotation = model_rotation_quaternion(normal_rows);
    if (tree_wave) std::copy_n(tree_wind_.wave, 4, rotation.begin());
    command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_,
                        240, sizeof(rotation), rotation.data());
}

bool DeferredPass::record_skinned(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                                  uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material_set,
                                  VkDescriptorSet pose_set, SurfaceMode mode, bool hud, uint32_t first_index,
                                  const char *vertex_name, const char *fragment_name, float alpha_ref,
                                  const float *normal_rows, float detail_scale, const float *environment_camera,
                                  const TextureMatrixSet *matrices, uint64_t variant_hash) const
{
    command_cache_.invalidate();
    const auto *pair = game_pipeline(vertex_name, fragment_name, variant_hash);
    if (!pair || !pair->skinned || pair->mode != mode || pair->hud != hud ||
        (hud || mode == SurfaceMode::Transparent ? !lighting_compatible(frame) : frame.render_pass != geometry_pass_) ||
        !frame.command_buffer || !frame.extent.width || !frame.extent.height || !vertices || !indices || !index_count ||
        !material_set || !pose_set || !material_image_shapes_match(
            material_set, pair->material_image_shapes, pair->material_descriptor_counts) ||
        (mode == SurfaceMode::Transparent && !hud &&
            (frame.image_index >= forward_sets_.size() || !forward_sets_[frame.image_index])))
        return false;
    const VkPipelineLayout layout = pair->layout ? pair->layout : skinned_layout_;
    const VkDescriptorSet draw_material_set = material_descriptor_set(material_set, pair);
    if (!draw_material_set)
        return false;
    if (pair->weather_layout && (!forward_weather_ ||
            !weather_image_shapes_match(forward_weather_, pair->weather_image_shapes,
                pair->weather_descriptor_counts)))
        return false;
    const VkDescriptorSet draw_weather_set = weather_descriptor_set(forward_weather_, pair);
    if (pair->weather_layout && !draw_weather_set)
        return false;
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pair->handle);
    viewport_scissor(frame, vk_);
    const VkDeviceSize offset = 0;
    vk_.cmd_bind_vertex_buffers(frame.command_buffer, 0, 1, &vertices, &offset);
    vk_.cmd_bind_index_buffer(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    const VkDescriptorSet sets[]{draw_material_set, pose_set};
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 2, sets, 0,
                                 nullptr);
    command_cache_.push(vk_, frame.command_buffer, layout,
                        game_stage_flags_, 0, sizeof(mvp), mvp);
    const bool opaque_environment =
        mode == SurfaceMode::Opaque && fragment_name && std::strstr(fragment_name, "skinned_environment_opaque.ps");
    push_surface(frame, layout, material_set, opaque_environment ? 96 : 80);
    if (opaque_environment)
    {
        const float origin[4]{};
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, 64, sizeof(origin),
                            environment_camera ? environment_camera : origin);
    }
    else
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, 68, sizeof(detail_scale),
                            &detail_scale);
    if (mode == SurfaceMode::AlphaTest ||
        (fragment_name &&
         (std::strstr(fragment_name, "world_set_blended.ps") || std::strstr(fragment_name, "model_set_blended.ps") ||
          std::strstr(fragment_name, "world_set_set_2r.ps") || std::strstr(fragment_name, "model_set_set_2r.ps") ||
          std::strstr(fragment_name, "world_set_blend_2r.ps") || std::strstr(fragment_name, "model_set_blend_2r.ps") ||
          std::strstr(fragment_name, "world_set_blend_4r.ps") || std::strstr(fragment_name, "model_set_blend_4r.ps") ||
          std::strstr(fragment_name, "model_set_alpha_add.ps") ||
          std::strstr(fragment_name, "model_set_fog_blended.ps") ||
          std::strstr(fragment_name, "skinned_model_set_fog_blended.ps") ||
          std::strstr(fragment_name, "object_wallmark_blended.ps") ||
          std::strstr(fragment_name, "glow_alpha_add.ps"))))
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, sizeof(mvp), sizeof(alpha_ref),
                            &alpha_ref);
    if (mode == SurfaceMode::Transparent && !hud)
    {
        vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, 1,
                                     &forward_sets_[frame.image_index], 0, nullptr);
        if (draw_weather_set)
            vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 3, 1,
                                         &draw_weather_set, 0, nullptr);
    }
    push_texture_matrices(frame, layout, matrices);
    if (mode != SurfaceMode::Transparent && !opaque_environment)
    {
        const bool hud_environment = hud && fragment_name && std::strstr(fragment_name, "hud_model_environment.ps");
        const Fvector& source = hud_environment ? Device.vCameraDirection : Device.vCameraPosition;
        const float camera[4]{source.x, source.y, source.z, hud_environment ? 0.f : 1.f};
        command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_, 112, sizeof(camera), camera);
    }
    vk_.cmd_draw_indexed(frame.command_buffer, index_count, 1, first_index, 0, 0);
    ++draw_calls_;
    triangles_ += index_count / 3;
    return true;
}

void DeferredPass::update_material(VkDescriptorSet set, VkImageView view, VkSampler sampler)
{
    const VkDescriptorImageInfo image{sampler, view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    vk_.update_descriptor_sets(device_, 1, &write, 0, nullptr);
    std::array<VkImageView, SurfaceMaterialBaseTextureCount> base{};
    base.fill(view);
    update_terrain_material(set, base, sampler);
    std::array<VkImageView, SurfaceMaterialExtraTextureCount> fallback{};
    fallback.fill(view);
    update_surface_textures(set, fallback, sampler);
}

void DeferredPass::update_lightmapped_material(VkDescriptorSet set, VkImageView albedo, VkImageView lightmap,
                                               VkSampler sampler, VkSampler lightmap_sampler)
{
    if (!device_ || !set || !albedo || !lightmap)
        return;
    const VkDescriptorImageInfo images[]{
        {sampler, albedo, VK_IMAGE_LAYOUT_GENERAL},
        {lightmap_sampler ? lightmap_sampler : sampler, lightmap, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &images[i];
    }
    vk_.update_descriptor_sets(device_, 2, writes, 0, nullptr);
    std::array<VkImageView, SurfaceMaterialBaseTextureCount> base{};
    base.fill(albedo);
    base[1] = lightmap;
    update_terrain_material(set, base, sampler, lightmap_sampler);
    std::array<VkImageView, SurfaceMaterialExtraTextureCount> fallback{};
    fallback.fill(albedo);
    update_surface_textures(set, fallback, sampler);
}

void DeferredPass::update_terrain_material(
    VkDescriptorSet set, const std::array<VkImageView, SurfaceMaterialBaseTextureCount> &views,
                                           VkSampler sampler, VkSampler lightmap_sampler)
{
    if (!device_ || !set)
        return;
    std::array<VkDescriptorImageInfo, 7> images{};
    std::array<VkWriteDescriptorSet, 7> writes{};
    for (uint32_t i = 0; i < views.size(); ++i)
    {
        images[i] = {i == 1 && lightmap_sampler ? lightmap_sampler : sampler, views[i], VK_IMAGE_LAYOUT_GENERAL};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &images[i];
    }
    vk_.update_descriptor_sets(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    auto& material_images = material_image_descriptors_[set];
    for (uint32_t i = 0; i < views.size(); ++i)
        material_images[i] = images[i];
    refresh_material_descriptor_sets(set);
}

bool DeferredPass::gbuffer(VkImageView albedo, VkImageView normal, VkImageView depth, VkImageView surface,
                           VkSampler sampler, VkDescriptorSet &set, std::string &error)
{
    if (!depth || !surface || !allocate(gbuffer_layout_, albedo, normal, sampler, set, error))
        return false;
    const VkDescriptorImageInfo images[]{{sampler, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                         {sampler, surface, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                         {sampler, depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                         {sampler, normal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    const uint32_t slots[]{2, 5, 8, 9};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = slots[i];
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &images[i];
    }
    vk_.update_descriptor_sets(device_, 4, writes, 0, nullptr);
    // Keep the additional descriptor valid until an RT visibility image is bound.
    bind_ray_visibility(set, depth, sampler);
    bind_indirect_light(set, depth, sampler);
    bind_rain_shadow(set, depth, sampler);
    return true;
}

bool DeferredPass::gbuffer_multisampled(VkImageView albedo, VkImageView normal, VkImageView depth,
                                        VkImageView surface, VkImageView resolved_depth,
                                        VkImageView resolved_normal, VkSampler sampler,
                                        VkDescriptorSet &set, std::string &error)
{
    if (!albedo || !normal || !depth || !surface || !resolved_depth || !resolved_normal || !sampler ||
        !allocate(gbuffer_layout_, albedo, normal, sampler, set, error))
        return false;
    const VkDescriptorImageInfo images[]{{sampler, depth, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
                                         {sampler, surface, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                         {sampler, resolved_depth, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                                         {sampler, resolved_normal, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    const uint32_t slots[]{2, 5, 8, 9};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = slots[i];
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &images[i];
    }
    vk_.update_descriptor_sets(device_, 4, writes, 0, nullptr);
    bind_ray_visibility(set, resolved_depth, sampler);
    bind_indirect_light(set, resolved_depth, sampler);
    bind_rain_shadow(set, resolved_depth, sampler);
    multisampled_gbuffer_sets_.insert(set);
    error.clear();
    return true;
}

void DeferredPass::pair_gbuffer_sets(VkDescriptorSet resolved_set, VkDescriptorSet multisampled_set)
{
    if (resolved_set && multisampled_gbuffer_sets_.count(multisampled_set))
        gbuffer_msaa_pairs_[resolved_set] = multisampled_set;
}

void DeferredPass::bind_ray_visibility(VkDescriptorSet set, VkImageView view, VkSampler sampler)
{
    if (!set || !view || !sampler) return;
    const VkDescriptorImageInfo image{sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2]{};
    uint32_t count = 1;
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 6;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &image;
    if (const auto pair = gbuffer_msaa_pairs_.find(set); pair != gbuffer_msaa_pairs_.end())
    {
        writes[1] = writes[0];
        writes[1].dstSet = pair->second;
        count = 2;
    }
    vk_.update_descriptor_sets(device_, count, writes, 0, nullptr);
}

void DeferredPass::bind_indirect_light(VkDescriptorSet set, VkImageView view, VkSampler sampler)
{
    if (!set || !view || !sampler) return;
    const VkDescriptorImageInfo image{sampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2]{};
    uint32_t count = 1;
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 7;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &image;
    if (const auto pair = gbuffer_msaa_pairs_.find(set); pair != gbuffer_msaa_pairs_.end())
    {
        writes[1] = writes[0];
        writes[1].dstSet = pair->second;
        count = 2;
    }
    vk_.update_descriptor_sets(device_, count, writes, 0, nullptr);
}

void DeferredPass::bind_sun_shadow(VkDescriptorSet set, VkImageView view, VkSampler sampler, VkBuffer uniform)
{
    if (!set || !view || !sampler || !uniform)
        return;
    const VkDescriptorImageInfo image{sampler, view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo buffer{uniform, 0, sizeof(SunShadowUniform)};
    VkWriteDescriptorSet writes[4]{};
    uint32_t count = 2;
    for (uint32_t i = 0; i < count; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].descriptorCount = 1;
    }
    writes[0].dstBinding = 3;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &image;
    writes[1].dstBinding = 4;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &buffer;
    if (const auto pair = gbuffer_msaa_pairs_.find(set); pair != gbuffer_msaa_pairs_.end())
    {
        writes[2] = writes[0];
        writes[2].dstSet = pair->second;
        writes[3] = writes[1];
        writes[3].dstSet = pair->second;
        count = 4;
    }
    vk_.update_descriptor_sets(device_, count, writes, 0, nullptr);
}

void DeferredPass::bind_rain_shadow(VkDescriptorSet set, VkImageView view, VkSampler sampler)
{
    if (!set || !view || !sampler)
        return;
    const VkDescriptorImageInfo image{sampler, view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[2]{};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 10;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &image;
    uint32_t count = 1;
    if (const auto pair = gbuffer_msaa_pairs_.find(set); pair != gbuffer_msaa_pairs_.end())
    {
        writes[1] = writes[0];
        writes[1].dstSet = pair->second;
        count = 2;
    }
    vk_.update_descriptor_sets(device_, count, writes, 0, nullptr);
}

bool DeferredPass::local_light_set(VkImageView shadow_array, VkSampler sampler, VkBuffer uniform, VkDeviceSize offset,
                                   VkDeviceSize range, VkDescriptorSet &set, std::string &error, VkImageView projection)
{
    set = VK_NULL_HANDLE;
    if (!pool_ || !shadow_array || !sampler || !uniform || !range || !projection)
    {
        error = "local light needs shadow array, projection texture and uniform buffer";
        return false;
    }
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = pool_;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &local_set_layout_;
    if (vk_.allocate_descriptor_sets(device_, &allocation, &set) != VK_SUCCESS)
    {
        error = "local light descriptor pool exhausted";
        return false;
    }
    const VkDescriptorImageInfo image{sampler, shadow_array, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    const VkDescriptorBufferInfo buffer{uniform, offset, range};
    VkWriteDescriptorSet writes[2]{};
    for (auto &write : writes)
    {
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.descriptorCount = 1;
    }
    writes[0].dstBinding = 0;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &image;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[1].pBufferInfo = &buffer;
    vk_.update_descriptor_sets(device_, 2, writes, 0, nullptr);
    if (projection)
        update_light_projection(set, projection, sampler);
    error.clear();
    return true;
}

void DeferredPass::update_light_projection(VkDescriptorSet set, VkImageView projection, VkSampler sampler)
{
    const VkDescriptorImageInfo image{sampler, projection, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 2;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &image;
    vk_.update_descriptor_sets(device_, 1, &write, 0, nullptr);
}

bool DeferredPass::water_set(uint32_t image_index, VkImageView refraction, VkImageView distortion_field, VkImageView depth,
                             VkSampler sampler, VkSampler depth_sampler, VkBuffer scene_uniform, std::string &error)
{
    if (!pool_ || !refraction || !distortion_field || !depth || !sampler || !depth_sampler || !scene_uniform ||
        image_index >= 16)
    {
        error = "water needs a scene snapshot, distortion field and opaque depth";
        return false;
    }
    if (water_sets_.size() <= image_index)
        water_sets_.resize(image_index + 1);
    if (water_sets_[image_index])
        release_gbuffer(water_sets_[image_index]);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = pool_;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &water_set_layout_;
    VkDescriptorSet &set = water_sets_[image_index];
    if (vk_.allocate_descriptor_sets(device_, &allocation, &set) != VK_SUCCESS)
    {
        error = "water descriptor pool exhausted";
        return false;
    }
    const VkImageView views[]{refraction, distortion_field, depth};
    VkDescriptorImageInfo images[3]{};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 3; ++i)
    {
        images[i] = {i == 2 ? depth_sampler : sampler, views[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].pImageInfo = &images[i];
    }
    const VkDescriptorBufferInfo buffer{scene_uniform, 0, sizeof(float) * 36};
    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = set;
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writes[3].pBufferInfo = &buffer;
    vk_.update_descriptor_sets(device_, 4, writes, 0, nullptr);
    error.clear();
    return true;
}

void DeferredPass::release_water_sets()
{
    for (auto &set : water_sets_)
        release_gbuffer(set);
    water_sets_.clear();
}

bool DeferredPass::forward_set(uint32_t image, VkBuffer buffer, std::string &error)
{
    if (!pool_ || !forward_layout_ || !buffer || image >= 16)
    {
        error = "invalid forward light uniform";
        return false;
    }
    if (forward_sets_.size() <= image)
        forward_sets_.resize(image + 1);
    release_gbuffer(forward_sets_[image]);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = pool_;
    allocation.descriptorSetCount = 1;
    allocation.pSetLayouts = &forward_layout_;
    VkDescriptorSet &set = forward_sets_[image];
    if (vk_.allocate_descriptor_sets(device_, &allocation, &set) != VK_SUCCESS)
    {
        error = "forward light descriptor pool exhausted";
        return false;
    }
    const VkDescriptorBufferInfo info{buffer, 0, sizeof(ForwardLightUniform)};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &info;
    vk_.update_descriptor_sets(device_, 1, &write, 0, nullptr);
    error.clear();
    return true;
}

void DeferredPass::release_forward_sets()
{
    for (auto &set : forward_sets_)
        release_gbuffer(set);
    forward_sets_.clear();
}

bool DeferredPass::weather_set(VkImageView sky_a, VkImageView sky_b, VkImageView clouds_a, VkImageView clouds_b,
                               VkImageView environment_a, VkImageView environment_b, VkSampler sampler,
                               VkDescriptorSet &set, const std::array<ShaderImageShape, 6>& image_shapes,
                               std::string &error)
{
    set = VK_NULL_HANDLE;
    if (!pool_ || !sampler || !sky_a || !sky_b || !clouds_a || !clouds_b || !environment_a || !environment_b)
    {
        error = "weather requires sky, environment and cloud textures";
        return false;
    }
    VkDescriptorSetAllocateInfo info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    info.descriptorPool = pool_;
    info.descriptorSetCount = 1;
    info.pSetLayouts = &weather_set_layout_;
    if (vk_.allocate_descriptor_sets(device_, &info, &set) != VK_SUCCESS)
    {
        error = "weather descriptor pool is exhausted";
        return false;
    }
    const VkImageView views[]{sky_a, sky_b, clouds_a, clouds_b, environment_a, environment_b};
    std::array<VkDescriptorImageInfo, 6> images{};
    VkWriteDescriptorSet writes[6]{};
    for (uint32_t i = 0; i < 6; ++i)
    {
        images[i] = {sampler, views[i], VK_IMAGE_LAYOUT_GENERAL};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[i].descriptorCount = 1;
        writes[i].pImageInfo = &images[i];
    }
    vk_.update_descriptor_sets(device_, 6, writes, 0, nullptr);
    weather_image_descriptors_[set] = images;
    weather_image_shapes_[set] = image_shapes;
    error.clear();
    return true;
}

void DeferredPass::release_gbuffer(VkDescriptorSet &set)
{
    release_material_descriptor_sets(set);
    surface_parameters_.erase(set);
    material_image_shapes_.erase(set);
    material_image_descriptors_.erase(set);
    weather_image_descriptors_.erase(set);
    weather_image_shapes_.erase(set);
    VkDescriptorSet paired{};
    if (const auto found = gbuffer_msaa_pairs_.find(set); found != gbuffer_msaa_pairs_.end())
    {
        paired = found->second;
        multisampled_gbuffer_sets_.erase(paired);
        gbuffer_msaa_pairs_.erase(found);
    }
    else if (multisampled_gbuffer_sets_.erase(set))
    {
        for (auto it = gbuffer_msaa_pairs_.begin(); it != gbuffer_msaa_pairs_.end(); ++it)
            if (it->second == set)
            {
                paired = it->first;
                gbuffer_msaa_pairs_.erase(it);
                break;
            }
    }
    if (device_ && pool_ && set && vk_.free_descriptor_sets)
    {
        VkDescriptorSet sets[]{set, paired};
        vk_.free_descriptor_sets(device_, pool_, paired ? 2 : 1, sets);
    }
    set = VK_NULL_HANDLE;
}

bool DeferredPass::record_geometry(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                                   uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material_set,
                                   SurfaceMode mode, uint32_t first_index, const char *vertex_name,
                                   const char *fragment_name, float alpha_ref, const float *normal_rows,
                                   float terrain_scale, const float *environment_camera,
                                   const float *environment_factor, int32_t vertex_base,
                                   const TextureMatrixSet *matrices, VkBuffer instances, uint32_t instance_count,
                                   VkBuffer indirect_commands, VkDeviceSize indirect_offset,
                                   uint32_t indirect_draw_count, uint64_t variant_hash) const
{
    const bool instanced = vertex_name && std::strcmp(vertex_name, "vk\\level_detail_instanced.vs") == 0;
    if (instanced != bool(instances) || !instance_count)
        return false;
    if ((indirect_commands && (!vk_.cmd_draw_indexed_indirect || !indirect_draw_count)) ||
        (!indirect_commands && indirect_draw_count))
        return false;
    if (mode == SurfaceMode::Transparent)
        return false;
    VkPipeline pipeline = mode == SurfaceMode::AlphaTest ? alpha_test_ : geometry_;
    const GamePipeline *pair = nullptr;
    if (vertex_name || fragment_name)
    {
        pair = game_pipeline(vertex_name, fragment_name, variant_hash);
        if (!pair || pair->mode != mode || pair->hud || pair->skinned)
            return false;
        pipeline = pair->handle;
    }
    if (!pipeline || frame.render_pass != geometry_pass_ || !frame.command_buffer || !frame.extent.width ||
        !frame.extent.height || !vertices || !indices || (!index_count && !indirect_commands) || !material_set)
        return false;
    if (pair && !material_image_shapes_match(
            material_set, pair->material_image_shapes, pair->material_descriptor_counts))
        return false;
    const VkPipelineLayout layout = pair && pair->layout ? pair->layout : geometry_layout_;
    const VkDescriptorSet draw_material_set = material_descriptor_set(material_set, pair);
    if (!draw_material_set)
        return false;
    command_cache_.bind(vk_, frame, pipeline, layout, vertices, indices, draw_material_set);
    command_cache_.push(vk_, frame.command_buffer, layout,
                        game_stage_flags_, 0, sizeof(mvp), mvp);
    const bool environment_variant = fragment_name && (std::strstr(fragment_name, "level_environment.ps") ||
                                                       std::strstr(fragment_name, "object_environment_opaque.ps") ||
                                                       std::strstr(fragment_name, "level_laemb_base.ps"));
    push_surface(frame, layout, material_set, environment_variant ? 96 : 80);
    if (mode == SurfaceMode::AlphaTest)
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, sizeof(mvp), sizeof(alpha_ref),
                            &alpha_ref);
    if (!environment_variant)
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, sizeof(mvp) + sizeof(alpha_ref),
                            sizeof(terrain_scale), &terrain_scale);
    if (fragment_name && (std::strstr(fragment_name, "level_environment.ps") ||
                          std::strstr(fragment_name, "object_environment_opaque.ps") ||
                          std::strstr(fragment_name, "level_laemb_base.ps")))
    {
        const float zero_camera[4]{};
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, 64, sizeof(zero_camera),
                            environment_camera ? environment_camera : zero_camera);
        if (environment_factor)
            command_cache_.push(vk_, frame.command_buffer, layout,
                                game_stage_flags_, 80, sizeof(float) * 4,
                                environment_factor);
    }
    push_texture_matrices(frame, layout, matrices, normal_rows,
        vertex_name && std::strstr(vertex_name, "level_tree_w.vs"));
    if (!environment_variant)
    {
        const float camera[4]{Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z, 1.f};
        command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_, 112, sizeof(camera), camera);
    }
    if (instances)
    {
        const VkDeviceSize offset = 0;
        vk_.cmd_bind_vertex_buffers(frame.command_buffer, 1, 1, &instances, &offset);
    }
    if (indirect_commands)
    {
        const uint32_t max_count = vk_.multi_draw_indirect ? std::max(vk_.max_draw_indirect_count, 1u) : 1u;
        for (uint32_t first = 0; first < indirect_draw_count;)
        {
            const uint32_t count = std::min(indirect_draw_count - first, max_count);
            vk_.cmd_draw_indexed_indirect(frame.command_buffer, indirect_commands,
                indirect_offset + VkDeviceSize(first) * sizeof(VkDrawIndexedIndirectCommand), count,
                sizeof(VkDrawIndexedIndirectCommand));
            first += count;
        }
    }
    else
        vk_.cmd_draw_indexed(frame.command_buffer, index_count, instance_count, first_index, vertex_base, 0);
    ++draw_calls_;
    if (!indirect_commands)
        triangles_ += index_count / 3 * instance_count;
    return true;
}

bool DeferredPass::record_sun_shadow(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                                     uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material,
                                     bool alpha_test, uint32_t first_index, float alpha_ref, int32_t vertex_base,
                                     VkDescriptorSet pose, bool tree_wave, const TextureMatrixSet* matrices,
                                     VkBuffer indirect_commands, VkDeviceSize indirect_offset,
                                     uint32_t indirect_draw_count) const
{
    const VkPipeline pipeline = pose ? (alpha_test ? shadow_skin_cutout_ : shadow_skin_opaque_)
                                     : (alpha_test ? shadow_cutout_ : shadow_opaque_);
    const auto layout = pose ? skinned_layout_ : geometry_layout_;
    if (!pipeline || !frame.command_buffer ||
        (frame.render_pass != shadow_pass_ && frame.render_pass != local_shadow_pass_ &&
         frame.render_pass != local_shadow_load_pass_ && frame.render_pass != local_shadow_graph_pass_ &&
         frame.render_pass != local_shadow_graph_load_pass_ &&
         frame.render_pass != rain_shadow_pass_ &&
         frame.render_pass != vsm_shadow_pass_ && frame.render_pass != vsm_dynamic_shadow_pass_) || !vertices || !indices ||
        !material || (!index_count && !indirect_commands) ||
        (indirect_commands && (!vk_.cmd_draw_indexed_indirect || !indirect_draw_count)) ||
        (!indirect_commands && indirect_draw_count))
        return false;
    command_cache_.bind(vk_, frame, pipeline, layout, vertices, indices, material);
    command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_, 0,
                        sizeof(mvp), mvp);
    push_texture_matrices(frame, layout, matrices, nullptr, tree_wave && !pose);
    if (!pose && !tree_wave)
    {
        const float zero[4]{};
        command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_, 136, sizeof(float), zero);
        command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_, 152, sizeof(float), zero);
        command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_, 240, sizeof(zero), zero);
    }
    if (alpha_test)
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, sizeof(mvp), sizeof(alpha_ref),
                            &alpha_ref);
    if (pose)
        vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1, &pose, 0,
                                     nullptr);
    if (indirect_commands)
    {
        const uint32_t max_count = vk_.multi_draw_indirect ? std::max(vk_.max_draw_indirect_count, 1u) : 1u;
        for (uint32_t first = 0; first < indirect_draw_count;)
        {
            const uint32_t count = std::min(indirect_draw_count - first, max_count);
            vk_.cmd_draw_indexed_indirect(frame.command_buffer, indirect_commands,
                indirect_offset + VkDeviceSize(first) * sizeof(VkDrawIndexedIndirectCommand), count,
                sizeof(VkDrawIndexedIndirectCommand));
            first += count;
        }
    }
    else
        vk_.cmd_draw_indexed(frame.command_buffer, index_count, 1, first_index, vertex_base, 0);
    ++draw_calls_;
    if (!indirect_commands)
        triangles_ += index_count / 3;
    return true;
}

bool DeferredPass::record_transparent(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                                      uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material_set,
                                      uint32_t first_index, const char *vertex_name, const char *fragment_name,
                                      const char **failure, float alpha_ref, const float *normal_rows,
                                      float detail_scale, int32_t vertex_base, const TextureMatrixSet *matrices,
                                      uint64_t variant_hash) const
{
    const auto reject = [failure](const char *reason) {
        if (failure)
            *failure = reason;
        return false;
    };
    VkPipeline pipeline = transparent_;
    const GamePipeline *pair = nullptr;
    if (vertex_name || fragment_name)
    {
        pair = game_pipeline(vertex_name, fragment_name, variant_hash);
        if (!pair || pair->mode != SurfaceMode::Transparent || pair->hud || pair->skinned)
            return reject("incompatible shader pair");
        if (!material_image_shapes_match(material_set, pair->material_image_shapes,
                pair->material_descriptor_counts))
            return reject("material image view does not match shader image dimension");
        pipeline = pair->handle;
    }
    if (!pipeline)
        return reject("missing pipeline");
    if (!lighting_compatible(frame))
        return reject("incompatible render pass");
    if (!frame.command_buffer)
        return reject("missing command buffer");
    if (!frame.extent.width || !frame.extent.height)
        return reject("empty frame extent");
    if (!vertices)
        return reject("missing vertex buffer");
    if (!indices)
        return reject("missing index buffer");
    if (!index_count)
        return reject("empty index window");
    if (!material_set)
        return reject("missing material descriptor");
    // A rejected draw must not leave half-recorded commands in the frame.
    // Check the forward set before binding the pipeline or issuing writes.
    if ((vertex_name || fragment_name) &&
        (frame.image_index >= forward_sets_.size() || !forward_sets_[frame.image_index]))
        return reject("missing forward lighting descriptor");
    if (pair && pair->weather_layout && (!forward_weather_ ||
            !weather_image_shapes_match(forward_weather_, pair->weather_image_shapes,
                pair->weather_descriptor_counts)))
        return reject("weather image views do not match shader descriptor array");
    const VkPipelineLayout layout = pair && pair->layout ? pair->layout : geometry_layout_;
    const VkDescriptorSet draw_material_set = material_descriptor_set(material_set, pair);
    if (!draw_material_set)
        return reject("failed to allocate material descriptor array set");
    const VkDescriptorSet draw_weather_set = weather_descriptor_set(forward_weather_, pair);
    if (pair && pair->weather_layout && !draw_weather_set)
        return reject("failed to allocate weather descriptor array set");
    command_cache_.bind(vk_, frame, pipeline, layout, vertices, indices, draw_material_set);
    command_cache_.push(vk_, frame.command_buffer, layout,
                        game_stage_flags_, 0, sizeof(mvp), mvp);
    push_surface(frame, layout, material_set);
    command_cache_.push(vk_, frame.command_buffer, layout,
                        game_stage_flags_, 68, sizeof(detail_scale),
                        &detail_scale);
    if (fragment_name &&
        ((std::strstr(fragment_name, "particle_") && std::strstr(fragment_name, "_set.ps") &&
          !std::strstr(fragment_name, "particle_screen_set.ps")) ||
         std::strstr(fragment_name, "world_set_blended.ps") || std::strstr(fragment_name, "model_set_blended.ps") ||
         std::strstr(fragment_name, "world_set_set_2r.ps") || std::strstr(fragment_name, "model_set_set_2r.ps") ||
         std::strstr(fragment_name, "world_set_blend_2r.ps") || std::strstr(fragment_name, "model_set_blend_2r.ps") ||
         std::strstr(fragment_name, "world_set_blend_4r.ps") || std::strstr(fragment_name, "model_set_blend_4r.ps") ||
         std::strstr(fragment_name, "model_set_alpha_add.ps") ||
         std::strstr(fragment_name, "model_set_fog_blended.ps") ||
         std::strstr(fragment_name, "skinned_model_set_fog_blended.ps") ||
         std::strstr(fragment_name, "object_wallmark_blended.ps") ||
         std::strstr(fragment_name, "level_fog.ps") ||
         std::strstr(fragment_name, "glow_alpha_add.ps")))
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, sizeof(mvp), sizeof(alpha_ref),
                            &alpha_ref);
    if (vertex_name || fragment_name)
    {
        vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1,
                                     &forward_sets_[frame.image_index], 0, nullptr);
        if (draw_weather_set)
            vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, 1,
                                         &draw_weather_set, 0, nullptr);
    }
    push_texture_matrices(frame, layout, matrices, normal_rows);
    vk_.cmd_draw_indexed(frame.command_buffer, index_count, 1, first_index, vertex_base, 0);
    ++draw_calls_;
    triangles_ += index_count / 3;
    return true;
}

bool DeferredPass::record_hud(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                              uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material_set,
                              uint32_t first_index, const char *vertex_name, const char *fragment_name, float alpha_ref,
                              const float *normal_rows, int32_t vertex_base, const TextureMatrixSet *matrices,
                              uint64_t variant_hash) const
{
    command_cache_.invalidate();
    VkPipeline pipeline = hud_;
    const GamePipeline *pair = nullptr;
    if (vertex_name || fragment_name)
    {
        pair = game_pipeline(vertex_name, fragment_name, variant_hash);
        if (!pair || !pair->hud || pair->skinned)
            return false;
        if (!material_image_shapes_match(material_set, pair->material_image_shapes,
                pair->material_descriptor_counts))
            return false;
        pipeline = pair->handle;
    }
    if (!pipeline || !lighting_compatible(frame) || !frame.command_buffer || !frame.extent.width ||
        !frame.extent.height || !vertices || !indices || !index_count || !material_set)
        return false;
    const VkPipelineLayout layout = pair && pair->layout ? pair->layout : geometry_layout_;
    const VkDescriptorSet draw_material_set = material_descriptor_set(material_set, pair);
    if (!draw_material_set)
        return false;
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    viewport_scissor(frame, vk_);
    const VkDeviceSize offset = 0;
    vk_.cmd_bind_vertex_buffers(frame.command_buffer, 0, 1, &vertices, &offset);
    vk_.cmd_bind_index_buffer(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                                 &draw_material_set, 0, nullptr);
    command_cache_.push(vk_, frame.command_buffer, layout,
                        game_stage_flags_, 0, sizeof(mvp), mvp);
    if (fragment_name &&
        ((std::strstr(fragment_name, "particle_") && std::strstr(fragment_name, "_set.ps") &&
          !std::strstr(fragment_name, "particle_screen_set.ps")) ||
         std::strstr(fragment_name, "model_set_blended.ps") || std::strstr(fragment_name, "model_set_set_2r.ps") ||
         std::strstr(fragment_name, "model_set_blend_2r.ps") || std::strstr(fragment_name, "model_set_blend_4r.ps") ||
         std::strstr(fragment_name, "model_set_alpha_add.ps") || std::strstr(fragment_name, "world_set_blended.ps") ||
         std::strstr(fragment_name, "glow_alpha_add.ps") ||
         std::strstr(fragment_name, "hud_particle_screen_blended.ps") ||
         std::strstr(fragment_name, "hud_particle_screen_alpha_add.ps")))
        command_cache_.push(vk_, frame.command_buffer, layout,
                            game_stage_flags_, sizeof(mvp), sizeof(alpha_ref),
                            &alpha_ref);
    if (fragment_name && std::strstr(fragment_name, "hud_model_environment.ps"))
    {
        const float camera[4]{Device.vCameraDirection.x, Device.vCameraDirection.y, Device.vCameraDirection.z, 0.f};
        command_cache_.push(vk_, frame.command_buffer, layout, game_stage_flags_, 112, sizeof(camera), camera);
    }
    push_texture_matrices(frame, layout, matrices, normal_rows);
    vk_.cmd_draw_indexed(frame.command_buffer, index_count, 1, first_index, vertex_base, 0);
    ++draw_calls_;
    triangles_ += index_count / 3;
    return true;
}

bool DeferredPass::record_lighting(const FrameRecordingContext &frame, VkDescriptorSet gbuffer_set,
                                   const DeferredLight &light, VkDescriptorSet weather_set,
                                   const WeatherLighting *weather) const
{
    command_cache_.invalidate();
    if (!lighting_ || frame.render_pass != light_pass_ || !frame.command_buffer || !frame.extent.width ||
        !frame.extent.height || !gbuffer_set || !vk_.cmd_draw)
        return false;
    const bool is_multisampled = multisampled_gbuffer_sets_.count(gbuffer_set) != 0;
    if (is_multisampled && (!multisampled_lighting_ ||
        (weather_set && weather && !multisampled_weather_pipeline_)))
        return false;
    const bool has_weather = weather_set && weather &&
        (is_multisampled ? multisampled_weather_pipeline_ : weather_pipeline_);
    const VkPipelineLayout layout = has_weather ? weather_layout_ : light_layout_;
    const VkPipeline pipeline = has_weather ?
        (is_multisampled ? multisampled_weather_pipeline_ : weather_pipeline_) :
        (is_multisampled && multisampled_lighting_ ? multisampled_lighting_ : lighting_);
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          pipeline);
    viewport_scissor(frame, vk_);
    const VkDescriptorSet sets[]{gbuffer_set, weather_set};
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, has_weather ? 2 : 1,
                                 sets, 0, nullptr);
    command_cache_.push(vk_, frame.command_buffer, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                        has_weather ? sizeof(WeatherLighting) : sizeof(light),
                        has_weather ? static_cast<const void *>(weather) : static_cast<const void *>(&light));
    vk_.cmd_draw(frame.command_buffer, 3, 1, 0, 0);
    ++draw_calls_;
    ++triangles_;
    return true;
}

bool DeferredPass::record_local_light(const FrameRecordingContext &frame, VkDescriptorSet gbuffer_set,
                                      VkDescriptorSet local_set, const VkRect2D &scissor) const
{
    command_cache_.invalidate();
    const bool is_multisampled = multisampled_gbuffer_sets_.count(gbuffer_set) != 0;
    const VkPipeline pipeline = is_multisampled ? multisampled_local_light_pipeline_ : local_light_pipeline_;
    if (!pipeline || frame.render_pass != light_pass_ || !frame.command_buffer || !gbuffer_set || !local_set)
        return false;
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    viewport_scissor(frame, vk_);
    vk_.cmd_set_scissor(frame.command_buffer, 0, 1, &scissor);
    const VkDescriptorSet sets[]{gbuffer_set, local_set};
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, local_light_layout_, 0, 2, sets,
                                 0, nullptr);
    vk_.cmd_draw(frame.command_buffer, 3, 1, 0, 0);
    ++draw_calls_;
    ++triangles_;
    return true;
}

bool DeferredPass::initialize_distortion(VkShaderModule vertex, VkShaderModule fragment, VkShaderModule screen_vertex,
                                          VkShaderModule composite, std::string &error)
{
    if (!device_ || !water_layout_ || !vertex || !fragment)
    {
        error = "distortion requires initialized scene descriptors and shader stages";
        return false;
    }
    return make_pipeline(device_, light_pass_, water_layout_, vertex, fragment,
                         PipelineFlag::Geometry | PipelineFlag::Transparent, vk_, distortion_pipeline_, error,
                         "particle distortion") &&
           make_pipeline(device_, light_pass_, water_layout_, screen_vertex, composite,
                         PipelineFlag::None, vk_, distortion_composite_, error, "distortion composite");
}

bool DeferredPass::composite_distortion(const FrameRecordingContext &frame) const
{
    if (!distortion_composite_ || !lighting_compatible(frame) || frame.image_index >= water_sets_.size())
        return false;
    command_cache_.invalidate();
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, distortion_composite_);
    viewport_scissor(frame, vk_);
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, water_layout_, 1, 1,
                                 &water_sets_[frame.image_index], 0, nullptr);
    vk_.cmd_draw(frame.command_buffer, 3, 1, 0, 0);
    ++draw_calls_;
    ++triangles_;
    return true;
}

bool DeferredPass::record_distortion(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                                     uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material, bool field) const
{
    command_cache_.invalidate();
    if (!distortion_pipeline_ || !lighting_compatible(frame) || !frame.command_buffer ||
        frame.image_index >= water_sets_.size() || !water_sets_[frame.image_index] || !vertices || !indices ||
        !index_count || !material || !frame.extent.width || !frame.extent.height)
        return false;
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, distortion_pipeline_);
    viewport_scissor(frame, vk_);
    const VkDeviceSize offset = 0;
    vk_.cmd_bind_vertex_buffers(frame.command_buffer, 0, 1, &vertices, &offset);
    vk_.cmd_bind_index_buffer(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    const VkDescriptorSet sets[]{material, water_sets_[frame.image_index]};
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, water_layout_, 0, 2, sets, 0,
                                 nullptr);
    command_cache_.push(vk_, frame.command_buffer, water_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mvp), mvp);
    const float params[]{0.f, 1.f / frame.extent.width, 1.f / frame.extent.height, field ? 1.f : 0.f};
    command_cache_.push(vk_, frame.command_buffer, water_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(mvp),
                        sizeof(params), params);
    vk_.cmd_draw_indexed(frame.command_buffer, index_count, 1, 0, 0, 0);
    ++draw_calls_;
    triangles_ += index_count / 3;
    return true;
}

bool DeferredPass::record_water(const FrameRecordingContext &frame, VkBuffer vertices, VkBuffer indices,
                                uint32_t index_count, const float (&mvp)[16], VkDescriptorSet material,
                                uint32_t first_index, float time, float opacity, int32_t vertex_base) const
{
    command_cache_.invalidate();
    if (!water_pipeline_ || !lighting_compatible(frame) || !frame.command_buffer ||
        frame.image_index >= water_sets_.size() || !water_sets_[frame.image_index] || !vertices || !indices ||
        !index_count || !material || !frame.extent.width || !frame.extent.height)
        return false;
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, water_pipeline_);
    viewport_scissor(frame, vk_);
    const VkDeviceSize offset = 0;
    vk_.cmd_bind_vertex_buffers(frame.command_buffer, 0, 1, &vertices, &offset);
    vk_.cmd_bind_index_buffer(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    const VkDescriptorSet sets[]{material, water_sets_[frame.image_index]};
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, water_layout_, 0, 2, sets, 0,
                                 nullptr);
    command_cache_.push(vk_, frame.command_buffer, water_layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(mvp), mvp);
    const float params[]{time, 1.f / frame.extent.width, 1.f / frame.extent.height, opacity};
    command_cache_.push(vk_, frame.command_buffer, water_layout_, VK_SHADER_STAGE_FRAGMENT_BIT, sizeof(mvp),
                        sizeof(params), params);
    vk_.cmd_draw_indexed(frame.command_buffer, index_count, 1, first_index, vertex_base, 0);
    ++draw_calls_;
    triangles_ += index_count / 3;
    return true;
}

void DeferredPass::destroy()
{
    pipeline_lookup_cache_.clear();
    command_cache_.invalidate();
    game_pipeline_request_ = {};
    release_forward_sets();
    release_water_sets();
    abort_game_pipeline_reload();
    release_all_material_descriptor_sets();
    if (device_ && vk_.destroy_pipeline)
    {
        for (auto& [name, pipeline] : game_pipelines_)
            destroy_game_pipeline(pipeline);
        if (geometry_)
            vk_.destroy_pipeline(device_, geometry_, nullptr);
        if (alpha_test_)
            vk_.destroy_pipeline(device_, alpha_test_, nullptr);
        if (transparent_)
            vk_.destroy_pipeline(device_, transparent_, nullptr);
        if (hud_)
            vk_.destroy_pipeline(device_, hud_, nullptr);
        if (lighting_)
            vk_.destroy_pipeline(device_, lighting_, nullptr);
        if (weather_pipeline_)
            vk_.destroy_pipeline(device_, weather_pipeline_, nullptr);
        if (multisampled_lighting_)
            vk_.destroy_pipeline(device_, multisampled_lighting_, nullptr);
        if (multisampled_weather_pipeline_)
            vk_.destroy_pipeline(device_, multisampled_weather_pipeline_, nullptr);
        if (multisampled_local_light_pipeline_)
            vk_.destroy_pipeline(device_, multisampled_local_light_pipeline_, nullptr);
        if (shadow_skin_opaque_)
            vk_.destroy_pipeline(device_, shadow_skin_opaque_, nullptr);
        if (shadow_skin_cutout_)
            vk_.destroy_pipeline(device_, shadow_skin_cutout_, nullptr);
        if (shadow_opaque_)
            vk_.destroy_pipeline(device_, shadow_opaque_, nullptr);
        if (shadow_cutout_)
            vk_.destroy_pipeline(device_, shadow_cutout_, nullptr);
        if (local_light_pipeline_)
            vk_.destroy_pipeline(device_, local_light_pipeline_, nullptr);
        if (water_pipeline_)
            vk_.destroy_pipeline(device_, water_pipeline_, nullptr);
        if (distortion_composite_)
            vk_.destroy_pipeline(device_, distortion_composite_, nullptr);
        if (distortion_pipeline_)
            vk_.destroy_pipeline(device_, distortion_pipeline_, nullptr);
    }
    game_pipelines_.clear();
    surface_parameters_.clear();
    material_image_shapes_.clear();
    material_image_descriptors_.clear();
    material_descriptor_variants_.clear();
    weather_image_descriptors_.clear();
    weather_image_shapes_.clear();
    weather_descriptor_variants_.clear();
    if (device_ && vk_.destroy_pipeline_layout)
    {
        if (geometry_layout_)
            vk_.destroy_pipeline_layout(device_, geometry_layout_, nullptr);
        if (skinned_layout_)
            vk_.destroy_pipeline_layout(device_, skinned_layout_, nullptr);
        if (light_layout_)
            vk_.destroy_pipeline_layout(device_, light_layout_, nullptr);
        if (weather_layout_)
            vk_.destroy_pipeline_layout(device_, weather_layout_, nullptr);
        if (local_light_layout_)
            vk_.destroy_pipeline_layout(device_, local_light_layout_, nullptr);
        if (water_layout_)
            vk_.destroy_pipeline_layout(device_, water_layout_, nullptr);
    }
    if (device_ && vk_.destroy_descriptor_pool && pool_)
        vk_.destroy_descriptor_pool(device_, pool_, nullptr);
    if (device_ && vk_.destroy_descriptor_set_layout)
    {
        if (material_layout_)
            vk_.destroy_descriptor_set_layout(device_, material_layout_, nullptr);
        if (pose_layout_)
            vk_.destroy_descriptor_set_layout(device_, pose_layout_, nullptr);
        if (gbuffer_layout_)
            vk_.destroy_descriptor_set_layout(device_, gbuffer_layout_, nullptr);
        if (weather_set_layout_)
            vk_.destroy_descriptor_set_layout(device_, weather_set_layout_, nullptr);
        if (local_set_layout_)
            vk_.destroy_descriptor_set_layout(device_, local_set_layout_, nullptr);
        if (water_set_layout_)
            vk_.destroy_descriptor_set_layout(device_, water_set_layout_, nullptr);
        if (forward_layout_)
            vk_.destroy_descriptor_set_layout(device_, forward_layout_, nullptr);
    }
    device_ = VK_NULL_HANDLE;
    geometry_pass_ = light_pass_ = shadow_pass_ = local_shadow_pass_ = local_shadow_load_pass_ = local_shadow_graph_pass_ =
        local_shadow_graph_load_pass_ = vsm_shadow_pass_ = overlay_pass_ = VK_NULL_HANDLE;
    vsm_dynamic_shadow_pass_ = VK_NULL_HANDLE;
    rain_shadow_pass_ = VK_NULL_HANDLE;
    geometry_ = alpha_test_ = transparent_ = hud_ = lighting_ = weather_pipeline_ = VK_NULL_HANDLE;
    multisampled_lighting_ = multisampled_weather_pipeline_ = multisampled_local_light_pipeline_ = VK_NULL_HANDLE;
    shadow_skin_opaque_ = shadow_skin_cutout_ = VK_NULL_HANDLE;
    shadow_opaque_ = shadow_cutout_ = VK_NULL_HANDLE;
    local_light_pipeline_ = VK_NULL_HANDLE;
    water_pipeline_ = VK_NULL_HANDLE;
    distortion_pipeline_ = distortion_composite_ = VK_NULL_HANDLE;
    geometry_layout_ = skinned_layout_ = light_layout_ = weather_layout_ = VK_NULL_HANDLE;
    local_light_layout_ = VK_NULL_HANDLE;
    water_layout_ = VK_NULL_HANDLE;
    material_layout_ = pose_layout_ = gbuffer_layout_ = weather_set_layout_ = VK_NULL_HANDLE;
    local_set_layout_ = VK_NULL_HANDLE;
    water_set_layout_ = VK_NULL_HANDLE;
    forward_layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    multisampled_gbuffer_sets_.clear();
    gbuffer_msaa_pairs_.clear();
    vk_ = {};
}
} // namespace xray::render::vulkan
