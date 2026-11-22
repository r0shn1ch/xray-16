#include "ObjectVelocityPass.h"
#include "LevelModels.h"
#include "ModelGeometry.h"
#include "DeferredPass.h"
#include "ObjectVelocityShaders.h"
#include "ShaderModule.h"
#include "DetailInstance.h"
#include <algorithm>
#include <cstddef>
#include <cstring>

namespace xray::render::vulkan
{
namespace
{
template <typename T> T procedure(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{ return reinterpret_cast<T>(get(device, name)); }
}

bool ObjectVelocityPass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
    VkFormat depth_format, const std::vector<VkImageView>& velocity_views,
    const std::vector<VkImageView>& depth_views, VkPipelineCache cache, std::string& error,
    VkDescriptorSetLayout material_layout, VkDescriptorSetLayout pose_layout,
    uint32_t max_draw_indirect_count)
{
    destroy();
    if (!device || !get || !extent.width || !extent.height ||
        depth_format == VK_FORMAT_UNDEFINED || velocity_views.empty() ||
        velocity_views.size() != depth_views.size() || velocity_views.size() > 16 ||
        std::any_of(velocity_views.begin(), velocity_views.end(), [](VkImageView view) { return !view; }) ||
        std::any_of(depth_views.begin(), depth_views.end(), [](VkImageView view) { return !view; }))
    { error = "object velocity requires matching color and depth views"; return false; }
    device_ = device;
    max_draw_indirect_count_ = std::max(max_draw_indirect_count, 1u);
    auto create_pass = procedure<PFN_vkCreateRenderPass>(device, get, "vkCreateRenderPass");
    destroy_pass_ = procedure<PFN_vkDestroyRenderPass>(device, get, "vkDestroyRenderPass");
    auto create_framebuffer = procedure<PFN_vkCreateFramebuffer>(device, get, "vkCreateFramebuffer");
    destroy_framebuffer_ = procedure<PFN_vkDestroyFramebuffer>(device, get, "vkDestroyFramebuffer");
    auto create_layout = procedure<PFN_vkCreatePipelineLayout>(device, get, "vkCreatePipelineLayout");
    destroy_layout_ = procedure<PFN_vkDestroyPipelineLayout>(device, get, "vkDestroyPipelineLayout");
    auto create_pipeline = procedure<PFN_vkCreateGraphicsPipelines>(device, get, "vkCreateGraphicsPipelines");
    destroy_pipeline_ = procedure<PFN_vkDestroyPipeline>(device, get, "vkDestroyPipeline");
    begin_pass_ = procedure<PFN_vkCmdBeginRenderPass>(device, get, "vkCmdBeginRenderPass");
    end_pass_ = procedure<PFN_vkCmdEndRenderPass>(device, get, "vkCmdEndRenderPass");
    bind_pipeline_ = procedure<PFN_vkCmdBindPipeline>(device, get, "vkCmdBindPipeline");
    bind_sets_ = procedure<PFN_vkCmdBindDescriptorSets>(device, get, "vkCmdBindDescriptorSets");
    set_viewport_ = procedure<PFN_vkCmdSetViewport>(device, get, "vkCmdSetViewport");
    set_scissor_ = procedure<PFN_vkCmdSetScissor>(device, get, "vkCmdSetScissor");
    bind_vertices_ = procedure<PFN_vkCmdBindVertexBuffers>(device, get, "vkCmdBindVertexBuffers");
    bind_indices_ = procedure<PFN_vkCmdBindIndexBuffer>(device, get, "vkCmdBindIndexBuffer");
    push_constants_ = procedure<PFN_vkCmdPushConstants>(device, get, "vkCmdPushConstants");
    draw_indexed_ = procedure<PFN_vkCmdDrawIndexed>(device, get, "vkCmdDrawIndexed");
    draw_indexed_indirect_ = procedure<PFN_vkCmdDrawIndexedIndirect>(device, get, "vkCmdDrawIndexedIndirect");
    const auto fail = [&](const char* reason) {
        error = reason;
        destroy();
        return false;
    };
    if (!create_pass || !destroy_pass_ || !create_framebuffer || !destroy_framebuffer_ ||
        !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ ||
        !begin_pass_ || !end_pass_ || !bind_pipeline_ || !bind_sets_ || !set_viewport_ || !set_scissor_ ||
        !bind_vertices_ || !bind_indices_ || !push_constants_ || !draw_indexed_ || !draw_indexed_indirect_)
        return fail("object velocity Vulkan procedures unavailable");
    extent_ = extent;

    VkAttachmentDescription attachments[2]{};
    attachments[0].format = VK_FORMAT_R16G16B16A16_SFLOAT;
    attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[0].initialLayout = attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    attachments[1].format = depth_format;
    attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
    attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachments[1].initialLayout = attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    const VkAttachmentReference color{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    const VkAttachmentReference depth{1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color;
    subpass.pDepthStencilAttachment = &depth;
    const VkSubpassDependency dependencies[]{
        {VK_SUBPASS_EXTERNAL, 0,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT},
        {0, VK_SUBPASS_EXTERNAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT, VK_DEPENDENCY_BY_REGION_BIT}};
    VkRenderPassCreateInfo pass_info{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass_info.attachmentCount = 2;
    pass_info.pAttachments = attachments;
    pass_info.subpassCount = 1;
    pass_info.pSubpasses = &subpass;
    pass_info.dependencyCount = 2;
    pass_info.pDependencies = dependencies;
    if (create_pass(device_, &pass_info, nullptr, &pass_) != VK_SUCCESS)
        return fail("object velocity render pass creation failed");
    for (size_t i = 0; i < velocity_views.size(); ++i)
    {
        const VkImageView views[]{velocity_views[i], depth_views[i]};
        VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        info.renderPass = pass_;
        info.attachmentCount = 2;
        info.pAttachments = views;
        info.width = extent.width;
        info.height = extent.height;
        info.layers = 1;
        VkFramebuffer framebuffer{};
        if (create_framebuffer(device_, &info, nullptr, &framebuffer) != VK_SUCCESS)
            return fail("object velocity framebuffer creation failed");
        framebuffers_.push_back(framebuffer);
    }
    if (pose_layout && !material_layout)
        return fail("object velocity skinning needs a material layout for descriptor set zero");
    const VkPushConstantRange ranges[]{
        {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 224}};
    const VkDescriptorSetLayout set_layouts[]{material_layout, pose_layout, pose_layout};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = pose_layout ? 3 : material_layout ? 1 : 0;
    layout_info.pSetLayouts = layout_info.setLayoutCount ? set_layouts : nullptr;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = ranges;
    if (create_layout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("object velocity pipeline layout creation failed");
    ShaderModule vertex, detail_vertex, skinned_vertex, lod_vertex, particle_velocity_vertex, particle_velocity_fragment,
        fragment, cutout_fragment, transparent_fragment, lod_fragment;
    const ShaderModuleDispatch shader_dispatch{
        procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
        procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")};
    if (!vertex.initialize(device_, shader_dispatch, object_velocity_shaders::Vertex,
            sizeof(object_velocity_shaders::Vertex), error) ||
        !fragment.initialize(device_, shader_dispatch, object_velocity_shaders::Fragment,
            sizeof(object_velocity_shaders::Fragment), error) ||
        !detail_vertex.initialize(device_, shader_dispatch, object_velocity_shaders::DetailVertex,
            sizeof(object_velocity_shaders::DetailVertex), error))
    { const std::string reason = "object velocity shader: " + error; return fail(reason.c_str()); }
    if (material_layout && !cutout_fragment.initialize(device_, shader_dispatch,
            object_velocity_shaders::CutoutFragment, sizeof(object_velocity_shaders::CutoutFragment), error))
    { const std::string reason = "object velocity cutout shader: " + error; return fail(reason.c_str()); }
    if (material_layout && !transparent_fragment.initialize(device_, shader_dispatch,
            object_velocity_shaders::TransparentFragment, sizeof(object_velocity_shaders::TransparentFragment), error))
    { const std::string reason = "object velocity transparent shader: " + error; return fail(reason.c_str()); }
    if (material_layout &&
        (!particle_velocity_vertex.initialize(device_, shader_dispatch, object_velocity_shaders::ParticleVelocityVertex,
             sizeof(object_velocity_shaders::ParticleVelocityVertex), error) ||
         !particle_velocity_fragment.initialize(device_, shader_dispatch, object_velocity_shaders::ParticleVelocityFragment,
             sizeof(object_velocity_shaders::ParticleVelocityFragment), error)))
    { const std::string reason = "object velocity particle shader: " + error; return fail(reason.c_str()); }
    if (material_layout &&
        (!lod_vertex.initialize(device_, shader_dispatch, object_velocity_shaders::LodVertex,
             sizeof(object_velocity_shaders::LodVertex), error) ||
         !lod_fragment.initialize(device_, shader_dispatch, object_velocity_shaders::LodFragment,
             sizeof(object_velocity_shaders::LodFragment), error)))
    { const std::string reason = "object velocity LOD shader: " + error; return fail(reason.c_str()); }
    if (pose_layout && !skinned_vertex.initialize(device_, shader_dispatch,
            object_velocity_shaders::SkinnedVertex, sizeof(object_velocity_shaders::SkinnedVertex), error))
    { const std::string reason = "object velocity skinning shader: " + error; return fail(reason.c_str()); }
    VkPipelineShaderStageCreateInfo stages[]{
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
            VK_SHADER_STAGE_VERTEX_BIT, vertex.handle(), "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
            VK_SHADER_STAGE_FRAGMENT_BIT, fragment.handle(), "main", nullptr}};
    const VkVertexInputBindingDescription bindings[]{
        {0, sizeof(LevelVertex), VK_VERTEX_INPUT_RATE_VERTEX},
        {1, sizeof(LevelVertex), VK_VERTEX_INPUT_RATE_VERTEX}};
    const VkVertexInputAttributeDescription attributes[]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {1, 1, VK_FORMAT_R32G32B32_SFLOAT, 0},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, uv)},
        {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, lightmap_uv)}};
    VkPipelineVertexInputStateCreateInfo input{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    input.vertexBindingDescriptionCount = 2;
    input.vertexAttributeDescriptionCount = 4;
    input.pVertexBindingDescriptions = bindings;
    input.pVertexAttributeDescriptions = attributes;
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth_state{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth_state.depthTestEnable = VK_TRUE;
    depth_state.depthWriteEnable = VK_FALSE;
    depth_state.depthCompareOp = VK_COMPARE_OP_EQUAL;
    VkPipelineColorBlendAttachmentState color_state{};
    color_state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &color_state;
    const VkDynamicState dynamic_states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo pipeline_info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = stages;
    pipeline_info.pVertexInputState = &input;
    pipeline_info.pInputAssemblyState = &assembly;
    pipeline_info.pViewportState = &viewport;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState = &multisample;
    pipeline_info.pDepthStencilState = &depth_state;
    pipeline_info.pColorBlendState = &blend;
    pipeline_info.pDynamicState = &dynamic;
    pipeline_info.layout = layout_;
    pipeline_info.renderPass = pass_;
    if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS)
        return fail("object velocity graphics pipeline creation failed");
    const VkVertexInputBindingDescription detail_bindings[]{
        {0, sizeof(LevelVertex), VK_VERTEX_INPUT_RATE_VERTEX},
        {1, sizeof(DetailInstance), VK_VERTEX_INPUT_RATE_INSTANCE}};
    const VkVertexInputAttributeDescription detail_attributes[]{
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LevelVertex, position)},
        {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, uv)},
        {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, lightmap_uv)},
        {4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(DetailInstance, position_scale)},
        {5, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(DetailInstance, rotation)}};
    input.pVertexBindingDescriptions = detail_bindings;
    input.pVertexAttributeDescriptions = detail_attributes;
    input.vertexAttributeDescriptionCount = 5;
    stages[0].module = detail_vertex.handle();
    if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &detail_pipeline_) != VK_SUCCESS)
        return fail("object velocity detail pipeline creation failed");
    if (material_layout)
    {
        stages[1].module = cutout_fragment.handle();
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &detail_cutout_pipeline_) != VK_SUCCESS)
            return fail("object velocity detail cutout pipeline creation failed");
        stages[1].module = transparent_fragment.handle();
        depth_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &detail_transparent_pipeline_) != VK_SUCCESS)
            return fail("object velocity detail transparent pipeline creation failed");
        depth_state.depthCompareOp = VK_COMPARE_OP_EQUAL;
    }
    input.pVertexBindingDescriptions = bindings;
    input.pVertexAttributeDescriptions = attributes;
    input.vertexAttributeDescriptionCount = 4;
    stages[0].module = vertex.handle();
    if (material_layout)
    {
        stages[1].module = cutout_fragment.handle();
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &cutout_pipeline_) != VK_SUCCESS)
            return fail("object velocity cutout pipeline creation failed");

        const VkVertexInputBindingDescription particle_bindings[]{
            {0, sizeof(LevelVertex), VK_VERTEX_INPUT_RATE_VERTEX},
            {1, sizeof(LevelVertex), VK_VERTEX_INPUT_RATE_VERTEX}};
        const VkVertexInputAttributeDescription particle_attributes[]{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LevelVertex, position)},
            {1, 1, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LevelVertex, position)},
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, uv)},
            {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(LevelVertex, baked)},
            {5, 1, VK_FORMAT_R32_SFLOAT, offsetof(LevelVertex, normal)}};
        input.vertexBindingDescriptionCount = 2;
        input.pVertexBindingDescriptions = particle_bindings;
        input.vertexAttributeDescriptionCount = 5;
        input.pVertexAttributeDescriptions = particle_attributes;
        stages[0].module = particle_velocity_vertex.handle();
        stages[1].module = particle_velocity_fragment.handle();
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &particle_velocity_pipeline_) != VK_SUCCESS)
            return fail("object velocity particle pipeline creation failed");
        depth_state.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr,
                &particle_transparent_velocity_pipeline_) != VK_SUCCESS)
            return fail("object velocity transparent particle pipeline creation failed");
        depth_state.depthCompareOp = VK_COMPARE_OP_EQUAL;

        const VkVertexInputBindingDescription lod_binding{0, sizeof(LevelVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription lod_attributes[]{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LevelVertex, position)},
            {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(LevelVertex, normal)},
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, uv)},
            {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(LevelVertex, lightmap_uv)}};
        input.vertexBindingDescriptionCount = 1;
        input.pVertexBindingDescriptions = &lod_binding;
        input.vertexAttributeDescriptionCount = 4;
        input.pVertexAttributeDescriptions = lod_attributes;
        stages[0].module = lod_vertex.handle();
        stages[1].module = lod_fragment.handle();
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &lod_pipeline_) != VK_SUCCESS)
            return fail("object velocity LOD graphics pipeline creation failed");
    }
    if (pose_layout)
    {
        const VkVertexInputBindingDescription skin_binding{
            0, sizeof(ModelVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription skin_attributes[]{
            {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(ModelVertex, position)},
            {2, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ModelVertex, uv)},
            {3, 0, VK_FORMAT_R16G16B16A16_UINT, offsetof(ModelVertex, bones)},
            {4, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(ModelVertex, weights)}};
        input.vertexBindingDescriptionCount = 1;
        input.pVertexBindingDescriptions = &skin_binding;
        input.vertexAttributeDescriptionCount = 4;
        input.pVertexAttributeDescriptions = skin_attributes;
        stages[0].module = skinned_vertex.handle();
        stages[1].module = fragment.handle();
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &skinned_pipeline_) != VK_SUCCESS)
            return fail("object velocity skinning pipeline creation failed");
        stages[1].module = cutout_fragment.handle();
        if (create_pipeline(device_, cache, 1, &pipeline_info, nullptr, &skinned_cutout_pipeline_) != VK_SUCCESS)
            return fail("object velocity skinning cutout pipeline creation failed");
    }
    error.clear();
    return true;
}

bool ObjectVelocityPass::begin(const FrameRecordingContext& frame) const
{
    if (!ready() || !frame.command_buffer || frame.image_index >= framebuffers_.size() ||
        frame.extent.width != extent_.width || frame.extent.height != extent_.height)
        return false;
    VkRenderPassBeginInfo info{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    info.renderPass = pass_;
    info.framebuffer = framebuffers_[frame.image_index];
    info.renderArea.extent = extent_;
    begin_pass_(frame.command_buffer, &info, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.f, 0.f, float(extent_.width), float(extent_.height), 0.f, 1.f};
    const VkRect2D scissor{{0, 0}, extent_};
    set_viewport_(frame.command_buffer, 0, 1, &viewport);
    set_scissor_(frame.command_buffer, 0, 1, &scissor);
    return true;
}

void ObjectVelocityPass::draw(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer previous_vertices,
    VkBuffer indices,
    uint32_t index_count, uint32_t first_index, int32_t vertex_offset, const float* current_mvp,
    const float* previous_mvp, VkDescriptorSet material, float alpha_ref,
    const float* base_rows, const TreeWindParameters* current_wind, const TreeWindParameters* previous_wind,
    VkBuffer indirect_commands, VkDeviceSize indirect_offset, uint32_t indirect_draw_count) const
{
    const bool indirect = indirect_commands != VK_NULL_HANDLE;
    if (!ready() || !frame.command_buffer || !vertices || !previous_vertices || !indices ||
        (!index_count && !indirect) || !current_mvp || !previous_mvp ||
        (material && !cutout_pipeline_) ||
        (indirect && (!indirect_draw_count || indirect_offset % sizeof(VkDrawIndexedIndirectCommand))) ||
        (!indirect && (indirect_offset || indirect_draw_count))) return;
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        material ? cutout_pipeline_ : pipeline_);
    float matrices[32];
    std::memcpy(matrices, current_mvp, sizeof(float) * 16);
    std::memcpy(matrices + 16, previous_mvp, sizeof(float) * 16);
    const VkBuffer vertex_buffers[]{vertices, previous_vertices};
    const VkDeviceSize offsets[]{0, 0};
    bind_vertices_(frame.command_buffer, 0, 2, vertex_buffers, offsets);
    bind_indices_(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(matrices), matrices);
    float wind[12]{};
    if (current_wind && previous_wind)
    {
        std::copy_n(current_wind->wave, 4, wind);
        std::copy_n(previous_wind->wave, 4, wind + 4);
        std::copy_n(current_wind->direction, 2, wind + 8);
        std::copy_n(previous_wind->direction, 2, wind + 10);
    }
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        176, sizeof(wind), wind);
    if (material)
    {
        struct CutoutParams { float alpha; float padding[3]; float base_u[4]; float base_v[4]; };
        static_assert(sizeof(CutoutParams) == 48);
        CutoutParams params{};
        params.alpha = alpha_ref;
        params.base_u[0] = params.base_v[1] = 1.f;
        if (base_rows)
        {
            std::memcpy(params.base_u, base_rows, sizeof(float) * 4);
            std::memcpy(params.base_v, base_rows + 4, sizeof(float) * 4);
        }
        bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 0, 1, &material, 0, nullptr);
        push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            128, sizeof(params), &params);
    }
    if (indirect)
    {
        for (uint32_t first = 0; first < indirect_draw_count;)
        {
            const uint32_t count = std::min(indirect_draw_count - first, max_draw_indirect_count_);
            draw_indexed_indirect_(frame.command_buffer, indirect_commands,
                indirect_offset + VkDeviceSize(first) * sizeof(VkDrawIndexedIndirectCommand),
                count, sizeof(VkDrawIndexedIndirectCommand));
            first += count;
        }
    }
    else
        draw_indexed_(frame.command_buffer, index_count, 1, first_index, vertex_offset, 0);
}

bool ObjectVelocityPass::draw_details(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer indices,
    uint32_t index_count, VkBuffer instances, uint32_t instance_count, const float* current_mvp,
    const float* previous_mvp, const DetailWindParameters& current_wind, const DetailWindParameters& previous_wind,
    VkDescriptorSet material, float alpha_ref, const float* base_rows, VkBuffer indirect_commands,
    bool transparent, VkDeviceSize instance_offset) const
{
    const auto pipeline = transparent ? detail_transparent_pipeline_ : material ? detail_cutout_pipeline_ : detail_pipeline_;
    if (!pipeline || !frame.command_buffer || !vertices || !indices || !index_count || !instances ||
        !instance_count || !current_mvp || !previous_mvp || (transparent && !material) ||
        instance_offset % sizeof(DetailInstance) || (indirect_commands && instance_offset)) return false;
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    const VkBuffer buffers[]{vertices, instances};
    const VkDeviceSize offsets[]{0, instance_offset};
    bind_vertices_(frame.command_buffer, 0, 2, buffers, offsets);
    bind_indices_(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    float data[56]{};
    std::copy_n(current_mvp, 16, data);
    std::copy_n(previous_mvp, 16, data + 16);
    data[32] = alpha_ref;
    data[36] = data[41] = 1.f;
    if (base_rows) std::copy_n(base_rows, 8, data + 36);
    std::copy_n(current_wind.directions, 4, data + 44);
    std::copy_n(previous_wind.directions, 4, data + 48);
    data[52] = current_wind.phase; data[53] = previous_wind.phase;
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(data), data);
    if (material)
        bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &material, 0, nullptr);
    if (indirect_commands)
        draw_indexed_indirect_(frame.command_buffer, indirect_commands, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
    else
        draw_indexed_(frame.command_buffer, index_count, instance_count, 0, 0, 0);
    return true;
}

void ObjectVelocityPass::draw_skinned(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer indices,
    uint32_t index_count, uint32_t first_index, const float* current_mvp,
    const float* previous_mvp, VkDescriptorSet current_pose, VkDescriptorSet previous_pose,
    VkDescriptorSet material, float alpha_ref, const float* base_rows) const
{
    if (!ready() || !skinned_pipeline_ || !frame.command_buffer || !vertices || !indices || !index_count ||
        !current_mvp || !previous_mvp || !current_pose || !previous_pose ||
        (material && !skinned_cutout_pipeline_)) return;
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
        material ? skinned_cutout_pipeline_ : skinned_pipeline_);
    float matrices[32];
    std::memcpy(matrices, current_mvp, sizeof(float) * 16);
    std::memcpy(matrices + 16, previous_mvp, sizeof(float) * 16);
    const VkDeviceSize offset = 0;
    bind_vertices_(frame.command_buffer, 0, 1, &vertices, &offset);
    bind_indices_(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    const VkDescriptorSet poses[]{current_pose, previous_pose};
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 1, 2, poses, 0, nullptr);
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(matrices), matrices);
    if (material)
    {
        struct CutoutParams { float alpha; float padding[3]; float base_u[4]; float base_v[4]; };
        static_assert(sizeof(CutoutParams) == 48);
        CutoutParams params{};
        params.alpha = alpha_ref;
        params.base_u[0] = params.base_v[1] = 1.f;
        if (base_rows)
        {
            std::memcpy(params.base_u, base_rows, sizeof(float) * 4);
            std::memcpy(params.base_v, base_rows + 4, sizeof(float) * 4);
        }
        bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
            layout_, 0, 1, &material, 0, nullptr);
        push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            128, sizeof(params), &params);
    }
    draw_indexed_(frame.command_buffer, index_count, 1, first_index, 0, 0);
}

void ObjectVelocityPass::draw_lod(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer indices,
    uint32_t index_count, uint32_t first_index, const float* mvp, VkDescriptorSet material,
    float alpha_ref, float lod_blend) const
{
    if (!ready() || !lod_pipeline_ || !frame.command_buffer || !vertices || !indices || !index_count ||
        !mvp || !material)
        return;
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, lod_pipeline_);
    const VkDeviceSize offset = 0;
    bind_vertices_(frame.command_buffer, 0, 1, &vertices, &offset);
    bind_indices_(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &material, 0, nullptr);
    float matrices[32]{};
    std::memcpy(matrices, mvp, sizeof(float) * 16);
    std::memcpy(matrices + 16, mvp, sizeof(float) * 16);
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(matrices), matrices);
    const float parameters[]{alpha_ref, lod_blend};
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        128, sizeof(parameters), parameters);
    draw_indexed_(frame.command_buffer, index_count, 1, first_index, 0, 0);
}

void ObjectVelocityPass::draw_particle_velocity(const FrameRecordingContext& frame, VkBuffer vertices,
    VkBuffer previous_vertices, VkBuffer indices, uint32_t index_count, const float* current_mvp,
    const float* previous_mvp, VkDescriptorSet material, float alpha_ref, const float* base_rows,
    bool transparent) const
{
    const VkPipeline particle_pipeline = transparent ? particle_transparent_velocity_pipeline_ : particle_velocity_pipeline_;
    if (!ready() || !particle_pipeline || !frame.command_buffer || !vertices || !previous_vertices ||
        !indices || !index_count || !current_mvp || !previous_mvp || !material)
        return;
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, particle_pipeline);
    const VkBuffer vertex_buffers[]{vertices, previous_vertices};
    const VkDeviceSize offsets[]{0, 0};
    bind_vertices_(frame.command_buffer, 0, 2, vertex_buffers, offsets);
    bind_indices_(frame.command_buffer, indices, 0, VK_INDEX_TYPE_UINT32);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &material, 0, nullptr);
    float matrices[32];
    std::memcpy(matrices, current_mvp, sizeof(float) * 16);
    std::memcpy(matrices + 16, previous_mvp, sizeof(float) * 16);
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
        0, sizeof(matrices), matrices);
    struct CutoutParams { float alpha; float padding[3]; float base_u[4]; float base_v[4]; };
    static_assert(sizeof(CutoutParams) == 48);
    CutoutParams params{};
    params.alpha = alpha_ref;
    params.padding[0] = transparent ? 1.f : 0.f;
    params.base_u[0] = params.base_v[1] = 1.f;
    if (base_rows)
    {
        std::memcpy(params.base_u, base_rows, sizeof(float) * 4);
        std::memcpy(params.base_v, base_rows + 4, sizeof(float) * 4);
    }
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 128, sizeof(params), &params);
    draw_indexed_(frame.command_buffer, index_count, 1, 0, 0, 0);
}

void ObjectVelocityPass::end(const FrameRecordingContext& frame) const
{ if (ready() && frame.command_buffer) end_pass_(frame.command_buffer); }

void ObjectVelocityPass::destroy()
{
    if (device_)
    {
        if (detail_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, detail_pipeline_, nullptr);
        if (detail_cutout_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, detail_cutout_pipeline_, nullptr);
        if (detail_transparent_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, detail_transparent_pipeline_, nullptr);
        if (skinned_cutout_pipeline_ && destroy_pipeline_)
            destroy_pipeline_(device_, skinned_cutout_pipeline_, nullptr);
        if (skinned_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, skinned_pipeline_, nullptr);
        if (particle_velocity_pipeline_ && destroy_pipeline_)
            destroy_pipeline_(device_, particle_velocity_pipeline_, nullptr);
        if (particle_transparent_velocity_pipeline_ && destroy_pipeline_)
            destroy_pipeline_(device_, particle_transparent_velocity_pipeline_, nullptr);
        if (lod_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, lod_pipeline_, nullptr);
        if (cutout_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, cutout_pipeline_, nullptr);
        if (pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
        if (layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
        for (VkFramebuffer framebuffer : framebuffers_)
            if (framebuffer && destroy_framebuffer_) destroy_framebuffer_(device_, framebuffer, nullptr);
        if (pass_ && destroy_pass_) destroy_pass_(device_, pass_, nullptr);
    }
    device_ = VK_NULL_HANDLE;
    extent_ = {};
    max_draw_indirect_count_ = 1;
    pass_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    cutout_pipeline_ = VK_NULL_HANDLE;
    detail_pipeline_ = detail_cutout_pipeline_ = VK_NULL_HANDLE;
    detail_transparent_pipeline_ = VK_NULL_HANDLE;
    lod_pipeline_ = VK_NULL_HANDLE;
    particle_velocity_pipeline_ = VK_NULL_HANDLE;
    particle_transparent_velocity_pipeline_ = VK_NULL_HANDLE;
    skinned_pipeline_ = VK_NULL_HANDLE;
    skinned_cutout_pipeline_ = VK_NULL_HANDLE;
    framebuffers_.clear();
}
}
