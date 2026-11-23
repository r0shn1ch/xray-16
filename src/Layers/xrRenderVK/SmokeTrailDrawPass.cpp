#include "xrEngine/stdafx.h"
#include "SmokeTrailDrawPass.h"
#include "ShaderModule.h"

namespace xray::render::vulkan
{
bool SmokeTrailDrawPass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch, VkRenderPass render_pass, VkPipelineCache cache,
    uint32_t frames, const GpuParticleCompute::Source& source, std::string& error, bool trail_mode,
    VkSampleCountFlagBits scene_samples)
{
    destroy();
    const uint32_t sample_count = static_cast<uint32_t>(scene_samples);
    if (!device || !get || !render_pass || !source || !frames || frames > 5 || !sample_count ||
        (sample_count & (sample_count - 1)) || sample_count > 64 || limits.maxPushConstantsSize < 64 ||
        limits.maxPerStageDescriptorStorageBuffers < (trail_mode ? 1u : 2u) ||
        limits.maxPerStageDescriptorSampledImages < 1 || limits.maxPerStageDescriptorSamplers < 1 ||
        limits.maxPerStageDescriptorUniformBuffers < 1 || limits.maxPerStageResources < (trail_mode ? 2u : 4u) ||
        (trail_mode && (limits.maxStorageBufferRange < TrailState::MaxPoints * sizeof(TrailControlPoint) ||
            limits.minStorageBufferOffsetAlignment > (UINT64_MAX - TrailState::MaxPoints * sizeof(TrailControlPoint)) / MaxTrailGroups)))
    { error = "device limits do not support smoke trail drawing"; return false; }
#define SMOKE_PROC(type, name) reinterpret_cast<type>(get(device, name))
    auto create_set_layout = SMOKE_PROC(PFN_vkCreateDescriptorSetLayout, "vkCreateDescriptorSetLayout");
    auto create_pool = SMOKE_PROC(PFN_vkCreateDescriptorPool, "vkCreateDescriptorPool");
    auto allocate = SMOKE_PROC(PFN_vkAllocateDescriptorSets, "vkAllocateDescriptorSets");
    auto create_layout = SMOKE_PROC(PFN_vkCreatePipelineLayout, "vkCreatePipelineLayout");
    auto create_pipeline = SMOKE_PROC(PFN_vkCreateGraphicsPipelines, "vkCreateGraphicsPipelines");
    destroy_set_layout_ = SMOKE_PROC(PFN_vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
    destroy_pool_ = SMOKE_PROC(PFN_vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
    destroy_layout_ = SMOKE_PROC(PFN_vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
    destroy_pipeline_ = SMOKE_PROC(PFN_vkDestroyPipeline, "vkDestroyPipeline");
    update_ = SMOKE_PROC(PFN_vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
    bind_pipeline_ = SMOKE_PROC(PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
    bind_sets_ = SMOKE_PROC(PFN_vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    viewport_ = SMOKE_PROC(PFN_vkCmdSetViewport, "vkCmdSetViewport");
    scissor_ = SMOKE_PROC(PFN_vkCmdSetScissor, "vkCmdSetScissor");
    push_ = SMOKE_PROC(PFN_vkCmdPushConstants, "vkCmdPushConstants");
    draw_ = SMOKE_PROC(PFN_vkCmdDrawIndirect, "vkCmdDrawIndirect");
    draw_direct_ = SMOKE_PROC(PFN_vkCmdDraw, "vkCmdDraw");
    ShaderModuleDispatch shaders{SMOKE_PROC(PFN_vkCreateShaderModule, "vkCreateShaderModule"),
        SMOKE_PROC(PFN_vkDestroyShaderModule, "vkDestroyShaderModule")};
#undef SMOKE_PROC
    if (!create_set_layout || !create_pool || !allocate || !create_layout || !create_pipeline ||
        !destroy_set_layout_ || !destroy_pool_ || !destroy_layout_ || !destroy_pipeline_ || !update_ ||
        !bind_pipeline_ || !bind_sets_ || !viewport_ || !scissor_ || !push_ || !draw_ || !draw_direct_ || !shaders.create || !shaders.destroy)
    { error = "smoke trail graphics procedures unavailable"; destroy(); return false; }
    device_ = device; render_pass_ = render_pass; trail_mode_ = trail_mode;
    VkDescriptorSetLayoutBinding bindings[4]{};
    bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[2] = {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    bindings[3] = {3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
    if (trail_mode)
    {
        bindings[1] = {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr};
        bindings[2] = {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
    }
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount = trail_mode ? 3 : 4; si.pBindings = bindings;
    if (create_set_layout(device, &si, nullptr, &set_layout_) != VK_SUCCESS)
    { error = "smoke trail descriptor layout failed"; destroy(); return false; }
    const uint32_t set_count = frames * (trail_mode ? MaxTrailGroups : 1);
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, set_count * (trail_mode ? 1u : 2u)},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, set_count}, {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set_count}};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = set_count; pi.poolSizeCount = 3; pi.pPoolSizes = sizes;
    if (create_pool(device, &pi, nullptr, &pool_) != VK_SUCCESS)
    { error = "smoke trail descriptor pool failed"; destroy(); return false; }
    sets_.resize(set_count); std::vector<VkDescriptorSetLayout> layouts(set_count, set_layout_);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_; ai.descriptorSetCount = set_count; ai.pSetLayouts = layouts.data();
    if (allocate(device, &ai, sets_.data()) != VK_SUCCESS)
    { error = "smoke trail descriptors failed"; destroy(); return false; }
    const VkPushConstantRange push_range{VK_SHADER_STAGE_VERTEX_BIT, 0, 64};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &set_layout_; li.pushConstantRangeCount = 1; li.pPushConstantRanges = &push_range;
    if (create_layout(device, &li, nullptr, &layout_) != VK_SUCCESS)
    { error = "smoke trail pipeline layout failed"; destroy(); return false; }
    const char* names[]{trail_mode ? "vk/trail.vs.spv" : "vk/smoke_trail.vs.spv",
        trail_mode ? "vk/trail.ps.spv" : "vk/smoke_trail.ps.spv"};
    ShaderModule modules[2];
    for (uint32_t i = 0; i < 2; ++i)
    {
        std::vector<uint8_t> bytes;
        if (!source(names[i], bytes, error) || !has_spirv_entry(bytes.data(), bytes.size(), i ? 4 : 0, "main") ||
            !modules[i].initialize_bytes(device, shaders, bytes.data(), bytes.size(), error))
        { if (error.empty()) error = "invalid smoke trail graphics shader"; destroy(); return false; }
    }
    VkPipelineShaderStageCreateInfo stages[2]{};
    for (auto& stage : stages) { stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO; stage.pName = "main"; }
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT; stages[0].module = modules[0].handle();
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT; stages[1].module = modules[1].handle();
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO}; ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vs{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO}; vs.viewportCount = vs.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL; rs.cullMode = VK_CULL_MODE_NONE; rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE; rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO}; ms.rasterizationSamples = scene_samples;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = VK_TRUE; ds.depthWriteEnable = VK_FALSE; ds.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    attachment.blendEnable = VK_TRUE; attachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    attachment.colorBlendOp = attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO}; bs.attachmentCount = 1; bs.pAttachments = &attachment;
    const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO}; dy.dynamicStateCount = 2; dy.pDynamicStates = states;
    VkGraphicsPipelineCreateInfo ci{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO}; ci.stageCount = 2; ci.pStages = stages;
    ci.pVertexInputState = &vi; ci.pInputAssemblyState = &ia; ci.pViewportState = &vs; ci.pRasterizationState = &rs;
    ci.pMultisampleState = &ms; ci.pDepthStencilState = &ds; ci.pColorBlendState = &bs; ci.pDynamicState = &dy;
    ci.layout = layout_; ci.renderPass = render_pass;
    if (create_pipeline(device, cache, 1, &ci, nullptr, &pipeline_) != VK_SUCCESS)
    { error = "smoke trail graphics pipeline failed"; destroy(); return false; }
    for (uint32_t i = 0; i < frames; ++i)
    {
        auto uniform = std::make_unique<GpuParticleUniforms>();
        if (!uniform->initialize(device, limits, memory, dispatch, trail_mode ? MaxTrailGroups : 1, error))
        { destroy(); return false; }
        uniforms_.push_back(std::move(uniform));
        if (trail_mode)
        {
            trail_alignment_ = std::max<VkDeviceSize>(limits.minStorageBufferOffsetAlignment, 1);
            const VkDeviceSize capacity = TrailState::MaxPoints * sizeof(TrailControlPoint) + MaxTrailGroups * trail_alignment_;
            auto buffer = std::make_unique<BufferResource>();
            if (!buffer->initialize(device, capacity, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, memory, dispatch, error))
            { destroy(); return false; }
            trail_buffers_.push_back(std::move(buffer));
        }
    }
    error.clear(); return true;
}
bool SmokeTrailDrawPass::record(const FrameRecordingContext& frame, const SmokeTrailCompute& smoke,
    VkImageView noise, VkSampler sampler, const SmokeTrailDrawParams& params, const float* mvp, std::string& error)
{
    if (trail_mode_ || !pipeline_ || !frame.command_buffer || frame.render_pass != render_pass_ || frame.frame_index >= sets_.size() ||
        !frame.extent.width || !frame.extent.height || !noise || !sampler || !mvp || !params.use_gpu_state ||
        !smoke.get(SmokeTrailCompute::DrawArguments).handle())
    { error = "invalid smoke trail draw"; return false; }
    auto& uniform = *uniforms_[frame.frame_index]; uniform.reset();
    VkDescriptorBufferInfo infos[3]{{smoke.get(SmokeTrailCompute::ControlPoints).handle(), 0, smoke.get(SmokeTrailCompute::ControlPoints).size()},
        {smoke.get(SmokeTrailCompute::State).handle(), 0, smoke.get(SmokeTrailCompute::State).size()}, {}};
    if (!infos[0].buffer || !infos[1].buffer || !uniform.append(&params, sizeof(params), infos[2], error)) return false;
    VkDescriptorImageInfo image{sampler, noise, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[4]{};
    for (uint32_t i = 0; i < 4; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = sets_[frame.frame_index];
        writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = i == 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
            i == 2 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        if (i == 3) writes[i].pImageInfo = &image; else writes[i].pBufferInfo = &infos[i];
    }
    update_(device_, 4, writes, 0, nullptr);
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    VkViewport viewport{float(frame.viewport_offset.x), float(frame.viewport_offset.y), float(frame.extent.width), float(frame.extent.height), 0.f, 1.f};
    VkRect2D scissor{frame.viewport_offset, frame.extent};
    viewport_(frame.command_buffer, 0, 1, &viewport); scissor_(frame.command_buffer, 0, 1, &scissor);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &sets_[frame.frame_index], 0, nullptr);
    push_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, mvp);
    draw_(frame.command_buffer, smoke.get(SmokeTrailCompute::DrawArguments).handle(), 0, 1, sizeof(VkDrawIndirectCommand));
    error.clear(); return true;
}
bool SmokeTrailDrawPass::record_trail(const FrameRecordingContext& frame, const std::vector<TrailState::Group>& groups,
    const SmokeTrailDrawParams& params, const float* mvp, VkImageView texture, VkSampler sampler, std::string& error)
{
    if (!trail_mode_ || !pipeline_ || !frame.command_buffer || frame.render_pass != render_pass_ ||
        frame.frame_index >= uniforms_.size() || frame.frame_index >= trail_buffers_.size() || !mvp || !texture || !sampler ||
        !frame.extent.width || !frame.extent.height || !params.subdivisions || params.subdivisions > 256 ||
        params.smoothing > 1 || params.edge_policy > 1 || params.uv_policy > 2)
    { error = "invalid trail draw"; return false; }
    size_t point_count = 0, draws = 0;
    for (const auto& group : groups)
    {
        point_count += group.points.size();
        if (group.points.size() >= 2) ++draws;
        if (point_count > TrailState::MaxPoints || draws > MaxTrailGroups)
        { error = "trail point or group capacity exceeded"; return false; }
    }
    auto& uniform = *uniforms_[frame.frame_index]; uniform.reset();
    auto& buffer = *trail_buffers_[frame.frame_index];
    VkDeviceSize offset = 0; uint32_t draw_index = 0;
    VkViewport viewport{float(frame.viewport_offset.x), float(frame.viewport_offset.y), float(frame.extent.width), float(frame.extent.height), 0.f, 1.f};
    VkRect2D scissor{frame.viewport_offset, frame.extent};
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    viewport_(frame.command_buffer, 0, 1, &viewport); scissor_(frame.command_buffer, 0, 1, &scissor);
    push_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT, 0, 64, mvp);
    for (const auto& group : groups)
    {
        if (group.points.size() < 2) continue;
        const VkDeviceSize size = group.points.size() * sizeof(TrailControlPoint);
        offset = ((offset + trail_alignment_ - 1) / trail_alignment_) * trail_alignment_;
        if (offset > buffer.size() || size > buffer.size() - offset || !buffer.write(offset, group.points.data(), size, error))
        { if (error.empty()) error = "trail upload buffer exhausted"; return false; }
        SmokeTrailDrawParams constants = params;
        constants.point_count = static_cast<uint32_t>(group.points.size());
        constants.material_id = group.color;
        constants.total_distance = group.points.back().distance; constants.use_gpu_state = 0;
        VkDescriptorBufferInfo infos[2]{{buffer.handle(), offset, size}, {}};
        if (!uniform.append(&constants, sizeof(constants), infos[1], error)) return false;
        const VkDescriptorSet set = sets_[frame.frame_index * MaxTrailGroups + draw_index++];
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t i = 0; i < 2; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = set; writes[i].dstBinding = i;
            writes[i].descriptorCount = 1; writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        VkDescriptorImageInfo image{sampler, texture, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[2].dstSet = set;
        writes[2].dstBinding = 2; writes[2].descriptorCount = 1;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[2].pImageInfo = &image;
        update_(device_, 3, writes, 0, nullptr);
        bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &set, 0, nullptr);
        draw_direct_(frame.command_buffer, (constants.point_count - 1) * constants.subdivisions * 6, 1, 0, 0);
        offset += size;
    }
    error.clear(); return true;
}
void SmokeTrailDrawPass::destroy()
{
    uniforms_.clear();
    trail_buffers_.clear();
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE; render_pass_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE; pool_ = VK_NULL_HANDLE; set_layout_ = VK_NULL_HANDLE; sets_.clear();
    trail_mode_ = false; trail_alignment_ = 0;
}
}
