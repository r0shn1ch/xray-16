#include "src/Layers/xrRenderVK/ScreenCopyPass.h"

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace
{
template <typename T>
T fake_handle(uintptr_t value)
{
    if constexpr (std::is_pointer_v<T>)
        return reinterpret_cast<T>(value);
    else
        return static_cast<T>(value);
}

uint32_t created_layouts = 0;
uint32_t destroyed_layouts = 0;
uint32_t created_pools = 0;
uint32_t destroyed_pools = 0;
uint32_t created_pipeline_layouts = 0;
uint32_t destroyed_pipeline_layouts = 0;
uint32_t created_pipelines = 0;
uint32_t destroyed_pipelines = 0;
uint32_t descriptor_updates = 0;
uint32_t color_map_updates = 0;
uint32_t draw_calls = 0;
uint32_t pushed_constants = 0;
float expected_aa = 0.f;
int expected_hdr_mode = -1;
VkResult pipeline_result = VK_SUCCESS;
bool expected_additive = false;

VkResult VKAPI_CALL create_descriptor_set_layout(VkDevice, const VkDescriptorSetLayoutCreateInfo* info,
    const VkAllocationCallbacks*, VkDescriptorSetLayout* layout)
{
    assert(info->bindingCount == 8);
    assert(info->pBindings[0].binding == 0);
    assert(info->pBindings[0].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    assert(info->pBindings[1].binding == 1);
    assert(info->pBindings[1].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER);
    assert(info->pBindings[2].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    assert(info->pBindings[3].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    assert(info->pBindings[4].binding == 4);
    assert(info->pBindings[4].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    assert(info->pBindings[5].binding == 5);
    assert(info->pBindings[5].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    assert(info->pBindings[6].binding == 6);
    assert(info->pBindings[6].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    assert(info->pBindings[7].binding == 7);
    assert(info->pBindings[7].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    *layout = fake_handle<VkDescriptorSetLayout>(1);
    ++created_layouts;
    return VK_SUCCESS;
}

void VKAPI_CALL destroy_descriptor_set_layout(VkDevice, VkDescriptorSetLayout,
    const VkAllocationCallbacks*)
{
    ++destroyed_layouts;
}

VkResult VKAPI_CALL create_descriptor_pool(VkDevice, const VkDescriptorPoolCreateInfo* info,
    const VkAllocationCallbacks*, VkDescriptorPool* pool)
{
    assert(info->maxSets == 1);
    assert(info->poolSizeCount == 3);
    assert(info->pPoolSizes[2].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    assert(info->pPoolSizes[2].descriptorCount == 2);
    *pool = fake_handle<VkDescriptorPool>(2);
    ++created_pools;
    return VK_SUCCESS;
}

void VKAPI_CALL destroy_descriptor_pool(VkDevice, VkDescriptorPool, const VkAllocationCallbacks*)
{
    ++destroyed_pools;
}

VkResult VKAPI_CALL allocate_descriptor_sets(VkDevice, const VkDescriptorSetAllocateInfo* info,
    VkDescriptorSet* set)
{
    assert(info->descriptorSetCount == 1);
    *set = fake_handle<VkDescriptorSet>(3);
    return VK_SUCCESS;
}

void VKAPI_CALL update_descriptor_sets(VkDevice, uint32_t count, const VkWriteDescriptorSet* writes,
    uint32_t, const VkCopyDescriptorSet*)
{
    assert(count == 8 || count == 2 || count == 1);
    if (count == 1)
    {
        assert(writes[0].dstBinding == 4 || writes[0].dstBinding == 5 || writes[0].dstBinding == 6);
        assert(writes[0].pImageInfo->imageView ==
            fake_handle<VkImageView>(writes[0].dstBinding == 4 ? 17 : writes[0].dstBinding == 5 ? 18 : 20));
        if (writes[0].dstBinding == 5 || writes[0].dstBinding == 6)
            assert(writes[0].pImageInfo->sampler == fake_handle<VkSampler>(writes[0].dstBinding == 5 ? 19 : 21));
        if (writes[0].dstBinding == 7)
            assert(writes[0].pImageInfo->imageView == fake_handle<VkImageView>(4));
        assert(writes[0].pImageInfo->imageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        ++descriptor_updates;
        return;
    }
    if (count == 2)
    {
        assert(writes[0].dstBinding == 2);
        assert(writes[0].pImageInfo->imageView == fake_handle<VkImageView>(15));
        assert(writes[1].dstBinding == 3);
        if (color_map_updates == 0)
        {
            assert(writes[0].pImageInfo->imageLayout == VK_IMAGE_LAYOUT_GENERAL);
            assert(writes[1].pImageInfo->imageView == fake_handle<VkImageView>(16));
            assert(writes[1].pImageInfo->imageLayout == VK_IMAGE_LAYOUT_GENERAL);
        }
        else
        {
            assert(writes[0].pImageInfo->imageLayout == VK_IMAGE_LAYOUT_GENERAL);
            assert(writes[1].pImageInfo->imageView == fake_handle<VkImageView>(4));
            assert(writes[1].pImageInfo->imageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        ++color_map_updates;
        ++descriptor_updates;
        return;
    }
    assert(writes[0].dstBinding == 0);
    assert(writes[0].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    assert(writes[0].pImageInfo->imageView == fake_handle<VkImageView>(4));
    assert(writes[0].pImageInfo->imageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    assert(writes[1].dstBinding == 1);
    assert(writes[1].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER);
    assert(writes[1].pImageInfo->sampler == fake_handle<VkSampler>(5));
    assert(writes[2].dstBinding == 2);
    assert(writes[3].dstBinding == 3);
    assert(writes[4].dstBinding == 4);
    assert(writes[5].dstBinding == 5);
    assert(writes[5].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    assert(writes[5].pImageInfo->imageView == fake_handle<VkImageView>(4));
    assert(writes[6].dstBinding == 6);
    assert(writes[6].descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    assert(writes[6].pImageInfo->imageView == fake_handle<VkImageView>(4));
    assert(writes[7].dstBinding == 7);
    assert(writes[7].descriptorType == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
    assert(writes[7].pImageInfo->imageView == fake_handle<VkImageView>(4));
    assert(writes[7].pImageInfo->imageLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    ++descriptor_updates;
}

VkResult VKAPI_CALL create_pipeline_layout(VkDevice, const VkPipelineLayoutCreateInfo* info,
    const VkAllocationCallbacks*, VkPipelineLayout* layout)
{
    assert(info->setLayoutCount == 1);
    assert(info->pSetLayouts[0] == fake_handle<VkDescriptorSetLayout>(1));
    assert(info->pushConstantRangeCount == 1);
    assert(info->pPushConstantRanges[0].size == sizeof(xray::render::vulkan::PostProcessConstants));
    *layout = fake_handle<VkPipelineLayout>(6);
    ++created_pipeline_layouts;
    return VK_SUCCESS;
}

void VKAPI_CALL destroy_pipeline_layout(VkDevice, VkPipelineLayout, const VkAllocationCallbacks*)
{
    ++destroyed_pipeline_layouts;
}

VkResult VKAPI_CALL create_graphics_pipelines(VkDevice, VkPipelineCache, uint32_t count,
    const VkGraphicsPipelineCreateInfo* info, const VkAllocationCallbacks*, VkPipeline* pipeline)
{
    assert(count == 1);
    assert(info->stageCount == 2);
    assert(info->pStages[0].stage == VK_SHADER_STAGE_VERTEX_BIT);
    assert(info->pStages[0].module == fake_handle<VkShaderModule>(7));
    assert(info->pStages[1].stage == VK_SHADER_STAGE_FRAGMENT_BIT);
    assert(info->pStages[1].module == fake_handle<VkShaderModule>(8));
    assert(info->pVertexInputState->vertexBindingDescriptionCount == 0);
    assert(info->pVertexInputState->vertexAttributeDescriptionCount == 0);
    assert(info->pInputAssemblyState->topology == VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    assert(info->pViewportState->viewportCount == 1);
    assert(info->pViewportState->scissorCount == 1);
    assert(info->pRasterizationState->cullMode == VK_CULL_MODE_NONE);
    assert(info->pColorBlendState->attachmentCount == 1);
    assert(bool(info->pColorBlendState->pAttachments[0].blendEnable) == expected_additive);
    if (expected_additive)
    {
        assert(info->pColorBlendState->pAttachments[0].srcColorBlendFactor == VK_BLEND_FACTOR_ONE);
        assert(info->pColorBlendState->pAttachments[0].dstColorBlendFactor == VK_BLEND_FACTOR_ONE);
    }
    assert(info->pDynamicState->dynamicStateCount == 2);
    assert(info->pDynamicState->pDynamicStates[0] == VK_DYNAMIC_STATE_VIEWPORT);
    assert(info->pDynamicState->pDynamicStates[1] == VK_DYNAMIC_STATE_SCISSOR);
    assert(info->renderPass == fake_handle<VkRenderPass>(9));
    if (pipeline_result == VK_SUCCESS)
    {
        *pipeline = fake_handle<VkPipeline>(10);
        ++created_pipelines;
    }
    return pipeline_result;
}

void VKAPI_CALL destroy_pipeline(VkDevice, VkPipeline, const VkAllocationCallbacks*)
{
    ++destroyed_pipelines;
}

void VKAPI_CALL cmd_bind_pipeline(VkCommandBuffer, VkPipelineBindPoint, VkPipeline pipeline)
{
    assert(pipeline == fake_handle<VkPipeline>(10));
}

void VKAPI_CALL cmd_bind_descriptor_sets(VkCommandBuffer, VkPipelineBindPoint, VkPipelineLayout layout,
    uint32_t first_set, uint32_t count, const VkDescriptorSet* sets, uint32_t, const uint32_t*)
{
    assert(layout == fake_handle<VkPipelineLayout>(6));
    assert(first_set == 0 && count == 1);
    assert(sets[0] == fake_handle<VkDescriptorSet>(3));
}

void VKAPI_CALL cmd_set_viewport(VkCommandBuffer, uint32_t first, uint32_t count, const VkViewport* viewport)
{
    assert(first == 0 && count == 1);
    assert(viewport->x == 0.0f && viewport->y == 0.0f);
    assert(viewport->width == 640.0f && viewport->height == 480.0f);
}

void VKAPI_CALL cmd_set_scissor(VkCommandBuffer, uint32_t first, uint32_t count, const VkRect2D* scissor)
{
    assert(first == 0 && count == 1);
    assert(scissor->offset.x == 0 && scissor->offset.y == 0);
    assert(scissor->extent.width == 640 && scissor->extent.height == 480);
}

void VKAPI_CALL cmd_draw(VkCommandBuffer, uint32_t vertex_count, uint32_t instance_count,
    uint32_t first_vertex, uint32_t first_instance)
{
    assert(vertex_count == 3 && instance_count == 1);
    assert(first_vertex == 0 && first_instance == 0);
    ++draw_calls;
}

void VKAPI_CALL cmd_push_constants(VkCommandBuffer, VkPipelineLayout, VkShaderStageFlags stage,
    uint32_t offset, uint32_t size, const void* data)
{
    assert(stage == VK_SHADER_STAGE_FRAGMENT_BIT && offset == 0);
    assert(size == sizeof(xray::render::vulkan::PostProcessConstants));
    assert(static_cast<const xray::render::vulkan::PostProcessConstants*>(data)->effect[0] == 2.f);
    assert(static_cast<const xray::render::vulkan::PostProcessConstants*>(data)->color_map[3] == expected_aa);
    if (expected_hdr_mode >= 0)
    {
        const auto& grade = static_cast<const xray::render::vulkan::PostProcessConstants*>(data)->grade;
        assert(grade[0] == 203.f && grade[1] == 1000.f && grade[2] == .05f && grade[3] == float(expected_hdr_mode));
    }
    ++pushed_constants;
}

xray::render::vulkan::ScreenCopyDispatch make_dispatch()
{
    return {create_descriptor_set_layout, destroy_descriptor_set_layout, create_descriptor_pool,
        destroy_descriptor_pool, allocate_descriptor_sets, update_descriptor_sets, create_pipeline_layout,
        destroy_pipeline_layout, create_graphics_pipelines, destroy_pipeline, cmd_bind_pipeline,
        cmd_bind_descriptor_sets, cmd_set_viewport, cmd_set_scissor, cmd_draw, cmd_push_constants};
}
}

int main()
{
    const VkDevice device = fake_handle<VkDevice>(11);
    const auto dispatch = make_dispatch();
    std::string error;
    xray::render::vulkan::ScreenCopyPass pass;
    assert(pass.initialize(device, fake_handle<VkRenderPass>(9), fake_handle<VkImageView>(4),
        fake_handle<VkSampler>(5), fake_handle<VkShaderModule>(7), fake_handle<VkShaderModule>(8),
        dispatch, error));
    assert(error.empty());
    assert(descriptor_updates == 1);
    pass.set_color_maps(fake_handle<VkImageView>(15), fake_handle<VkImageView>(16));
    assert(descriptor_updates == 2);
    pass.set_color_maps(fake_handle<VkImageView>(15), VK_NULL_HANDLE);
    assert(descriptor_updates == 3);
    pass.set_bloom_view(fake_handle<VkImageView>(17));
    assert(descriptor_updates == 4);
    pass.set_exposure_view(fake_handle<VkImageView>(18), fake_handle<VkSampler>(19));
    assert(descriptor_updates == 5);
    pass.set_depth_view(fake_handle<VkImageView>(20), fake_handle<VkSampler>(21));
    assert(descriptor_updates == 6);
    xray::render::vulkan::PostProcessConstants params;
    params.effect[0] = 2.f;
    pass.set_constants(params);

    const xray::render::vulkan::FrameRecordingContext frame{fake_handle<VkCommandBuffer>(12),
        fake_handle<VkRenderPass>(9), fake_handle<VkFramebuffer>(13), {640, 480}, 0, 0};
    pass.record(frame);
    assert(draw_calls == 1);
    assert(pushed_constants == 1);
    // Toggle AA without allocating descriptors or recreating a pipeline.
    expected_aa = params.color_map[3] = 1.f;
    pass.set_constants(params);
    pass.record(frame);
    expected_aa = params.color_map[3] = 0.f;
    pass.set_constants(params);
    pass.record(frame);
    assert(pushed_constants == 3 && created_pipelines == 1 && descriptor_updates == 6);
    pass.record({fake_handle<VkCommandBuffer>(12), VK_NULL_HANDLE, VK_NULL_HANDLE, {0, 480}, 0, 0});
    pass.record({fake_handle<VkCommandBuffer>(12), fake_handle<VkRenderPass>(14), VK_NULL_HANDLE, {640, 480}, 0, 0});
    assert(draw_calls == 3);
    for (expected_hdr_mode = 0; expected_hdr_mode <= 2; ++expected_hdr_mode)
    {
        pass.set_hdr_calibration(203.f, 1000.f, .05f, expected_hdr_mode);
        pass.record(frame);
    }
    assert(draw_calls == 6 && pushed_constants == 6 && created_pipelines == 1 && descriptor_updates == 6);
    pass.destroy();
    assert(destroyed_pipelines == 1);
    assert(destroyed_pipeline_layouts == 1);
    assert(destroyed_pools == 1);
    assert(destroyed_layouts == 1);

    pipeline_result = VK_ERROR_INITIALIZATION_FAILED;
    assert(!pass.initialize(device, fake_handle<VkRenderPass>(9), fake_handle<VkImageView>(4),
        fake_handle<VkSampler>(5), fake_handle<VkShaderModule>(7), fake_handle<VkShaderModule>(8),
        dispatch, error));
    assert(!error.empty());
    assert(destroyed_pipeline_layouts == 2);
    assert(destroyed_pools == 2);
    assert(destroyed_layouts == 2);
    pipeline_result = VK_SUCCESS;
    expected_additive = true;
    assert(pass.initialize(device, fake_handle<VkRenderPass>(9), fake_handle<VkImageView>(4),
        fake_handle<VkSampler>(5), fake_handle<VkShaderModule>(7), fake_handle<VkShaderModule>(8),
        dispatch, error, true));
    pass.destroy();
    assert(created_pipelines == destroyed_pipelines);
}
