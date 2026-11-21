#include "GpuParticleDrawPass.h"

namespace xray::render::vulkan
{
bool GpuParticleDrawPass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
    VkRenderPass render_pass, VkDescriptorSetLayout material_layout, VkDescriptorSetLayout particle_layout,
    VkShaderModule vertex, VkShaderModule fragment, VkPipelineCache cache, uint32_t mode,
    bool hud, bool gbuffer, std::string& error, VkDescriptorSetLayout scene_layout,
    VkSampleCountFlagBits gbuffer_samples, bool velocity, bool transparent_velocity)
{
    destroy();
    const uint32_t sample_count = static_cast<uint32_t>(gbuffer_samples);
    const bool valid_gbuffer_samples = sample_count && !(sample_count & (sample_count - 1)) && sample_count <= 64;
    if (!device || !get || !render_pass || !material_layout || !particle_layout || !vertex || !fragment ||
        mode > 5 || (gbuffer && (hud || mode != 0)) || limits.maxPushConstantsSize < 128 ||
        (velocity && (hud || gbuffer || scene_layout)) ||
        (scene_layout && limits.maxBoundDescriptorSets < 3) ||
        (gbuffer && !valid_gbuffer_samples))
    { error = "invalid GPU particle graphics pipeline configuration"; return false; }
#define GPU_PROC(type, name) reinterpret_cast<type>(get(device, name))
    auto create_layout = GPU_PROC(PFN_vkCreatePipelineLayout, "vkCreatePipelineLayout");
    auto create_pipeline = GPU_PROC(PFN_vkCreateGraphicsPipelines, "vkCreateGraphicsPipelines");
    destroy_layout_ = GPU_PROC(PFN_vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
    destroy_pipeline_ = GPU_PROC(PFN_vkDestroyPipeline, "vkDestroyPipeline");
    bind_pipeline_ = GPU_PROC(PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
    bind_sets_ = GPU_PROC(PFN_vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    viewport_ = GPU_PROC(PFN_vkCmdSetViewport, "vkCmdSetViewport");
    scissor_ = GPU_PROC(PFN_vkCmdSetScissor, "vkCmdSetScissor");
    push_ = GPU_PROC(PFN_vkCmdPushConstants, "vkCmdPushConstants");
    draw_ = GPU_PROC(PFN_vkCmdDrawIndirect, "vkCmdDrawIndirect");
#undef GPU_PROC
    if (!create_layout || !create_pipeline || !destroy_layout_ || !destroy_pipeline_ || !bind_pipeline_ ||
        !bind_sets_ || !viewport_ || !scissor_ || !push_ || !draw_)
    { error = "GPU particle graphics procedures unavailable"; destroy(); return false; }
    device_ = device; render_pass_ = render_pass; has_scene_ = scene_layout != VK_NULL_HANDLE;
    velocity_ = velocity;
    // Both the G-buffer and forward scene passes use the negotiated scene
    // sample count. The render pass determines whether this pipeline writes
    // one or three color attachments; the sample count is shared.
    samples_ = gbuffer_samples;
    const VkDescriptorSetLayout sets[]{material_layout, particle_layout, scene_layout};
    const VkPushConstantRange push_range{VK_SHADER_STAGE_VERTEX_BIT, 0, 128};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = has_scene_ ? 3 : 2; li.pSetLayouts = sets; li.pushConstantRangeCount = 1; li.pPushConstantRanges = &push_range;
    if (create_layout(device, &li, nullptr, &layout_) != VK_SUCCESS)
    { error = "GPU particle graphics pipeline layout failed"; destroy(); return false; }
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = vertex; stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = fragment; stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vs.viewportCount = vs.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = samples_;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = !velocity && (gbuffer || hud || mode == 0);
    ds.depthCompareOp = velocity && !transparent_velocity ? VK_COMPARE_OP_EQUAL : VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    attachment.blendEnable = !velocity && !gbuffer && mode != 0;
    attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.colorBlendOp = attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    if (mode == 2) { attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        attachment.colorWriteMask &= ~VK_COLOR_COMPONENT_A_BIT; }
    if (mode == 3 || mode == 4) { attachment.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
        attachment.dstColorBlendFactor = mode == 4 ? VK_BLEND_FACTOR_SRC_COLOR : VK_BLEND_FACTOR_ZERO;
        attachment.colorWriteMask &= ~VK_COLOR_COMPONENT_A_BIT; }
    if (mode == 5) { attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA; attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; }
    const VkPipelineColorBlendAttachmentState attachments[]{attachment, attachment, attachment};
    VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    bs.attachmentCount = gbuffer ? 3 : 1; bs.pAttachments = attachments;
    const VkDynamicState dynamic_states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2; dy.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    ci.stageCount = 2; ci.pStages = stages; ci.pVertexInputState = &vi; ci.pInputAssemblyState = &ia;
    ci.pViewportState = &vs; ci.pRasterizationState = &rs; ci.pMultisampleState = &ms;
    ci.pDepthStencilState = &ds; ci.pColorBlendState = &bs; ci.pDynamicState = &dy;
    ci.layout = layout_; ci.renderPass = render_pass;
    if (create_pipeline(device, cache, 1, &ci, nullptr, &pipeline_) != VK_SUCCESS)
    { error = "GPU particle graphics pipeline creation failed"; destroy(); return false; }
    error.clear(); return true;
}
bool GpuParticleDrawPass::record(const FrameRecordingContext& frame, const GpuParticleDrawBindings& bindings,
    const GpuParticleBuffers& buffers, VkDescriptorSet material, VkDescriptorBufferInfo camera,
    VkDescriptorBufferInfo material_params, uint32_t descriptor_slot, uint32_t bucket,
    const void* constants, std::string& error, VkDescriptorSet scene) const
{
    const auto& indirect = buffers.get(GpuParticleBuffers::DrawArguments);
    const VkDeviceSize offset = VkDeviceSize(bucket) * sizeof(VkDrawIndirectCommand);
    if (!pipeline_ || !layout_ || !frame.command_buffer || (!velocity_ && frame.render_pass != render_pass_) || !frame.extent.width ||
        !frame.extent.height || !material || !constants || !indirect.handle() || offset > indirect.size() ||
        sizeof(VkDrawIndirectCommand) > indirect.size() - offset || (has_scene_ && !scene))
    { error = "invalid GPU particle indirect draw"; return false; }
    if (!bindings.bind(frame.command_buffer, layout_, descriptor_slot, buffers, camera, material_params, error)) return false;
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    VkViewport viewport{float(frame.viewport_offset.x), float(frame.viewport_offset.y), float(frame.extent.width),
        float(frame.extent.height), 0.f, 1.f};
    VkRect2D scissor{frame.viewport_offset, frame.extent};
    viewport_(frame.command_buffer, 0, 1, &viewport); scissor_(frame.command_buffer, 0, 1, &scissor);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &material, 0, nullptr);
    if (has_scene_) bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 2, 1, &scene, 0, nullptr);
    push_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, 128, constants);
    draw_(frame.command_buffer, indirect.handle(), offset, 1, sizeof(VkDrawIndirectCommand));
    error.clear(); return true;
}
void GpuParticleDrawPass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    device_ = VK_NULL_HANDLE; render_pass_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; layout_ = VK_NULL_HANDLE;
    has_scene_ = false; velocity_ = false; samples_ = VK_SAMPLE_COUNT_1_BIT;
}
}
