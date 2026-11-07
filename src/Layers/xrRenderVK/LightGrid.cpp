#include "LightGrid.h"
#include "LightGridShaders.h"
#include "DeferredShaders.h"
#include <cstring>
#include <algorithm>

namespace xray::render::vulkan
{
namespace
{
struct alignas(16) PointRecord
{
    float position[4], color[4];
    int32_t rectangle[4];
};
struct alignas(16) LightData
{
    float inverse[16];
    uint32_t shape[4];
    PointRecord points[LocalLightCapacity];
};
static_assert(offsetof(LightData, points) == 80 && sizeof(PointRecord) == 48);
} // namespace
bool LightGrid::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkRenderPass pass,
                           VkDescriptorSetLayout gbuffer, const VkPhysicalDeviceMemoryProperties &memory,
                           const BufferResourceDispatch &buffers, const ScenePassDispatch &graphics,
                           const ShaderModuleDispatch &shaders, PFN_vkCmdPipelineBarrier barrier, std::string &error)
{
    destroy();
    device_ = device;
    vk_ = graphics;
    memory_ = memory;
    buffers_ = buffers;
    barrier_ = barrier;
    dispatch_ = reinterpret_cast<PFN_vkCmdDispatch>(get(device, "vkCmdDispatch"));
    auto create_compute = reinterpret_cast<PFN_vkCreateComputePipelines>(get(device, "vkCreateComputePipelines"));
    const auto fail = [&](const char *reason) {
        error = reason;
        destroy();
        return false;
    };
    if (!dispatch_ || !create_compute || !barrier_)
        return fail("light grid compute procedures unavailable");
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo si{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    si.bindingCount = 2;
    si.pBindings = bindings;
    if (vk_.create_descriptor_set_layout(device_, &si, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("light grid set layout failed");
    const VkDescriptorSetLayout layouts[]{gbuffer, set_layout_};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 2;
    li.pSetLayouts = layouts;
    if (vk_.create_pipeline_layout(device_, &li, nullptr, &layout_) != VK_SUCCESS)
        return fail("light grid pipeline layout failed");
    const VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
    VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pi.maxSets = 16;
    pi.poolSizeCount = 1;
    pi.pPoolSizes = &size;
    if (vk_.create_descriptor_pool(device_, &pi, nullptr, &pool_) != VK_SUCCESS)
        return fail("light grid descriptor pool failed");
    ShaderModule compute, vertex, fragment;
    if (!compute.initialize(device, shaders, light_grid_shaders::Cull, sizeof(light_grid_shaders::Cull), error) ||
        !vertex.initialize(device, shaders, deferred_shaders::LightVertex, sizeof(deferred_shaders::LightVertex),
                           error) ||
        !fragment.initialize(device, shaders, light_grid_shaders::Shade, sizeof(light_grid_shaders::Shade), error))
    {
        destroy();
        return false;
    }
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                nullptr,
                0,
                VK_SHADER_STAGE_COMPUTE_BIT,
                compute.handle(),
                "main",
                nullptr};
    ci.layout = layout_;
    if (create_compute(device_, VK_NULL_HANDLE, 1, &ci, nullptr, &cull_) != VK_SUCCESS)
        return fail("light grid compute pipeline failed");
    const VkPipelineShaderStageCreateInfo stages[]{{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                                    VK_SHADER_STAGE_VERTEX_BIT, vertex.handle(), "main", nullptr},
                                                   {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                                                    VK_SHADER_STAGE_FRAGMENT_BIT, fragment.handle(), "main", nullptr}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState attachment{};
    attachment.blendEnable = VK_TRUE;
    attachment.srcColorBlendFactor = attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.srcAlphaBlendFactor = attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    attachment.colorBlendOp = attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    attachment.colorWriteMask = 15;
    VkPipelineColorBlendStateCreateInfo blend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    blend.attachmentCount = 1;
    blend.pAttachments = &attachment;
    const VkDynamicState states[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.stageCount = 2;
    gp.pStages = stages;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pColorBlendState = &blend;
    gp.pDynamicState = &dynamic;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    gp.pDepthStencilState = &depth;
    gp.layout = layout_;
    gp.renderPass = pass;
    if (vk_.create_graphics_pipelines(device_, VK_NULL_HANDLE, 1, &gp, nullptr, &shade_) != VK_SUCCESS)
        return fail("light grid shade pipeline failed");
    error.clear();
    return true;
}
bool LightGrid::prepare(const FrameRecordingContext &frame, const std::vector<LocalLightUniform> &lights,
                        const std::vector<VkRect2D> &scissors, std::string &error)
{
    if (frame.image_index >= images_.size() || lights.size() != scissors.size())
    {
        error = "invalid light grid input";
        return false;
    }
    auto &image = images_[frame.image_index];
    image.count = 0;
    image.mask = 0;
    if (!device_)
        return true;
    LightData data{};
    const uint32_t tiles_x = (frame.extent.width + TileSize - 1) / TileSize,
                   tiles_y = (frame.extent.height + TileSize - 1) / TileSize;
    for (size_t i = 0; i < lights.size() && i < LocalLightCapacity; ++i)
    {
        const auto &light = lights[i];
        if (light.color_type[3] >= .5f || light.shadow_params[2] >= .5f || light.debug_mode[0] > .5f)
            continue;
        auto &point = data.points[image.count++];
        std::copy_n(light.position_range, 4, point.position);
        std::copy_n(light.color_type, 4, point.color);
        const auto &rect = scissors[i];
        point.rectangle[0] = rect.offset.x;
        point.rectangle[1] = rect.offset.y;
        point.rectangle[2] = rect.offset.x + rect.extent.width;
        point.rectangle[3] = rect.offset.y + rect.extent.height;
        image.mask |= uint64_t{1} << i;
    }
    if (image.count < 2)
    {
        image.count = 0;
        image.mask = 0;
        return true;
    }
    std::copy_n(lights[0].inverse_view_projection, 16, data.inverse);
    data.shape[0] = frame.extent.width;
    data.shape[1] = frame.extent.height;
    data.shape[2] = tiles_x;
    data.shape[3] = image.count;
    const VkDeviceSize tile_bytes = VkDeviceSize(tiles_x) * tiles_y * (LocalLightCapacity + 1) * sizeof(uint32_t);
    bool changed = false;
    if (!image.lights.handle())
    {
        if (!image.lights.initialize(device_, sizeof(data), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     memory_, buffers_, error))
            return false;
        changed = true;
    }
    if (image.tiles.size() < tile_bytes)
    {
        if (!image.tiles.initialize(device_, tile_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory_, buffers_, error))
            return false;
        changed = true;
    }
    if (!image.set)
    {
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool_;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &set_layout_;
        if (vk_.allocate_descriptor_sets(device_, &allocation, &image.set) != VK_SUCCESS)
        {
            error = "light grid set allocation failed";
            return false;
        }
        changed = true;
    }
    if (changed)
    {
        const VkDescriptorBufferInfo infos[]{{image.lights.handle(), 0, sizeof(data)},
                                             {image.tiles.handle(), 0, image.tiles.size()}};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t i = 0; i < 2; ++i)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = image.set;
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &infos[i];
        }
        vk_.update_descriptor_sets(device_, 2, writes, 0, nullptr);
    }
    if (!image.lights.write(0, &data, sizeof(data), error))
        return false;
    image.extent = frame.extent;
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, cull_);
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 1, 1, &image.set, 0,
                                 nullptr);
    dispatch_(frame.command_buffer, tiles_x, tiles_y, 1);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = image.tiles.handle();
    barrier.size = VK_WHOLE_SIZE;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0,
             nullptr, 1, &barrier, 0, nullptr);
    return true;
}
bool LightGrid::record(const FrameRecordingContext &frame, VkDescriptorSet gbuffer) const
{
    if (!count(frame.image_index))
        return true;
    const auto &image = images_[frame.image_index];
    if (image.extent.width != frame.extent.width || image.extent.height != frame.extent.height)
        return false;
    vk_.cmd_bind_pipeline(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, shade_);
    const VkViewport viewport{0, 0, float(frame.extent.width), float(frame.extent.height), 0, 1};
    const VkRect2D scissor{{0, 0}, frame.extent};
    vk_.cmd_set_viewport(frame.command_buffer, 0, 1, &viewport);
    vk_.cmd_set_scissor(frame.command_buffer, 0, 1, &scissor);
    const VkDescriptorSet sets[]{gbuffer, image.set};
    vk_.cmd_bind_descriptor_sets(frame.command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 2, sets, 0,
                                 nullptr);
    vk_.cmd_draw(frame.command_buffer, 3, 1, 0, 0);
    return true;
}
bool LightGrid::contains(uint32_t image, uint32_t light) const
{
    return image < images_.size() && light < LocalLightCapacity && (images_[image].mask & (uint64_t{1} << light));
}
uint32_t LightGrid::count(uint32_t image) const
{
    return image < images_.size() ? images_[image].count : 0;
}
void LightGrid::destroy()
{
    if (!device_)
        return;
    for (auto &image : images_)
    {
        image.lights.destroy();
        image.tiles.destroy();
        image = {};
    }
    if (shade_)
        vk_.destroy_pipeline(device_, shade_, nullptr);
    if (cull_)
        vk_.destroy_pipeline(device_, cull_, nullptr);
    if (pool_)
        vk_.destroy_descriptor_pool(device_, pool_, nullptr);
    if (layout_)
        vk_.destroy_pipeline_layout(device_, layout_, nullptr);
    if (set_layout_)
        vk_.destroy_descriptor_set_layout(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE;
    shade_ = cull_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
}
} // namespace xray::render::vulkan
