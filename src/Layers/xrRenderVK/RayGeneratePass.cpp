#include "RayGeneratePass.h"
#include "RayGenerateShaders.h"
#include "ComputePipeline.h"
#include "ShaderModule.h"
#include <algorithm>
#include <array>
#include <cstring>

namespace xray::render::vulkan
{
namespace {
template<class T> T proc(VkDevice d, PFN_vkGetDeviceProcAddr get, const char* n)
{ return reinterpret_cast<T>(get(d, n)); }
}
bool RayGeneratePass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
    const std::vector<VkImageView>& depth, const std::vector<VkImageView>& normal,
    VkSampler sampler, const std::vector<BufferResource*>& outputs, VkPipelineCache cache, std::string& error,
    Mode mode, const std::vector<BufferResource*>& primary_surface_outputs)
{
    destroy();
    const uint64_t pixels = uint64_t(extent.width) * extent.height;
    if (!device || !get || !sampler || !pixels || pixels > UINT32_MAX || depth.empty() ||
        depth.size() > 16 || normal.size() != depth.size() || outputs.size() != depth.size() ||
        (mode == Mode::Indirect && primary_surface_outputs.size() != depth.size()) ||
        (mode != Mode::Indirect && !primary_surface_outputs.empty()) ||
        (extent.width + uint64_t(7)) / 8 > 65535 || (extent.height + uint64_t(7)) / 8 > 65535)
    { error = "invalid ray generation extent or resources"; return false; }
    for (size_t i = 0; i < depth.size(); ++i)
    {
        if (!depth[i] || !normal[i] || !outputs[i] || !outputs[i]->handle() ||
            outputs[i]->size() < pixels * sizeof(RayQueryInput))
        { error = "ray generation input or output is unavailable"; return false; }
        if (mode == Mode::Indirect && (!primary_surface_outputs[i] || !primary_surface_outputs[i]->handle() ||
            primary_surface_outputs[i]->size() < pixels * sizeof(uint32_t) * 2))
        { error = "indirect primary-surface output is unavailable"; return false; }
    }
    device_ = device;
#define LOAD(field, type, name) field = proc<type>(device, get, name)
    auto create_set_layout = proc<PFN_vkCreateDescriptorSetLayout>(device, get, "vkCreateDescriptorSetLayout");
    LOAD(destroy_set_layout_, PFN_vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
    auto create_pool = proc<PFN_vkCreateDescriptorPool>(device, get, "vkCreateDescriptorPool");
    LOAD(destroy_pool_, PFN_vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
    auto allocate = proc<PFN_vkAllocateDescriptorSets>(device, get, "vkAllocateDescriptorSets");
    auto update = proc<PFN_vkUpdateDescriptorSets>(device, get, "vkUpdateDescriptorSets");
    auto create_layout = proc<PFN_vkCreatePipelineLayout>(device, get, "vkCreatePipelineLayout");
    LOAD(destroy_layout_, PFN_vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
    auto create_pipeline = proc<PFN_vkCreateComputePipelines>(device, get, "vkCreateComputePipelines");
    LOAD(destroy_pipeline_, PFN_vkDestroyPipeline, "vkDestroyPipeline");
    LOAD(bind_pipeline_, PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
    LOAD(bind_sets_, PFN_vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    LOAD(push_constants_, PFN_vkCmdPushConstants, "vkCmdPushConstants");
    LOAD(dispatch_, PFN_vkCmdDispatch, "vkCmdDispatch");
    LOAD(barrier_, PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
#undef LOAD
    const auto fail = [&](const char* why) { error = why; destroy(); return false; };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate ||
        !update || !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ ||
        !bind_pipeline_ || !bind_sets_ || !push_constants_ || !dispatch_ || !barrier_)
        return fail("ray generation Vulkan procedures unavailable");
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    const uint32_t binding_count = mode == Mode::Indirect ? 4 : 3;
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = binding_count; set_info.pBindings = bindings;
    if (create_set_layout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("ray generation descriptor layout creation failed");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 80};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1; layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1; layout_info.pPushConstantRanges = &range;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("ray generation pipeline layout creation failed");
    uint32_t count = static_cast<uint32_t>(depth.size());
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count * (mode == Mode::Indirect ? 2u : 1u)}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = count; pool_info.poolSizeCount = 2; pool_info.pPoolSizes = sizes;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("ray generation descriptor pool creation failed");
    sets_.resize(count);
    std::vector<VkDescriptorSetLayout> layouts(count, set_layout_);
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = pool_; alloc.descriptorSetCount = count; alloc.pSetLayouts = layouts.data();
    if (allocate(device, &alloc, sets_.data()) != VK_SUCCESS)
        return fail("ray generation descriptor allocation failed");
    for (uint32_t i = 0; i < count; ++i)
    {
        const VkDescriptorImageInfo images[]{{sampler, depth[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {sampler, normal[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
        const VkDescriptorBufferInfo buffers[]{
            {outputs[i]->handle(), 0, pixels * sizeof(RayQueryInput)},
            {mode == Mode::Indirect ? primary_surface_outputs[i]->handle() : VK_NULL_HANDLE,
                0, mode == Mode::Indirect ? pixels * sizeof(uint32_t) * 2 : 0}};
        VkWriteDescriptorSet writes[4]{};
        for (uint32_t j = 0; j < binding_count; ++j)
        { writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[j].dstSet = sets_[i];
          writes[j].dstBinding = j; writes[j].descriptorCount = 1;
          writes[j].descriptorType = j == 2 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; }
        if (binding_count == 4) writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pImageInfo = &images[0]; writes[1].pImageInfo = &images[1];
        writes[2].pBufferInfo = &buffers[0];
        if (binding_count == 4) writes[3].pBufferInfo = &buffers[1];
        update(device, binding_count, writes, 0, nullptr);
    }
    ShaderModule shader;
    const uint32_t* code = mode == Mode::Indirect ? ray_generate_shaders::Indirect : ray_generate_shaders::Generate;
    const size_t code_size = mode == Mode::Indirect ? sizeof(ray_generate_shaders::Indirect) : sizeof(ray_generate_shaders::Generate);
    if (!shader.initialize(device, {proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            code, code_size, error) ||
        !create_compute_pipeline(device, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error)) { destroy(); return false; }
    extent_ = extent;
    error.clear(); return true;
}
bool RayGeneratePass::record(const FrameRecordingContext& frame, const float* inverse, const float* direction) const
{
    if (!pipeline_ || !frame.command_buffer || frame.image_index >= sets_.size() || !inverse || !direction ||
        frame.extent.width != extent_.width || frame.extent.height != extent_.height) return false;
    std::array<float, 20> constants{};
    std::copy_n(inverse, 16, constants.begin()); std::copy_n(direction, 4, constants.begin() + 16);
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets_[frame.image_index], 0, nullptr);
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), constants.data());
    dispatch_(frame.command_buffer, (extent_.width + 7) / 8, (extent_.height + 7) / 8, 1);
    VkMemoryBarrier sync{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    sync.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; sync.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &sync, 0, nullptr, 0, nullptr);
    return true;
}
void RayGeneratePass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE; extent_ = {}; set_layout_ = VK_NULL_HANDLE; pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; sets_.clear();
}
}
