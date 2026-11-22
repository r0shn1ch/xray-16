#include "VsmDemandPass.h"
#include "VsmDemandShaders.h"
#include "ComputePipeline.h"
#include "ShaderModule.h"
#include <algorithm>
#include <cmath>

namespace xray::render::vulkan
{
namespace {
template<class T> T proc(VkDevice d, PFN_vkGetDeviceProcAddr get, const char* n)
{ return reinterpret_cast<T>(get(d, n)); }
}
bool VsmDemandPass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
    const std::vector<VkImageView>& depths, VkSampler sampler,
    const std::vector<BufferResource*>& parameters, const std::vector<BufferResource*>& needed,
    VkPipelineCache cache, std::string& error)
{
    destroy();
    if (!device || !get || !sampler || !extent.width || !extent.height || depths.empty() || depths.size() > 16 ||
        depths.size() != parameters.size() || depths.size() != needed.size() ||
        (uint64_t(extent.width) + 7) / 8 > 65535 || (uint64_t(extent.height) + 7) / 8 > 65535)
    { error = "invalid VSM demand extent or resources"; return false; }
    for (size_t i = 0; i < depths.size(); ++i)
        if (!depths[i] || !parameters[i] || !parameters[i]->handle() ||
            parameters[i]->size() < sizeof(VsmDemandParameters) || !needed[i] || !needed[i]->handle())
        { error = "VSM demand input or output is unavailable"; return false; }
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
    LOAD(dispatch_, PFN_vkCmdDispatch, "vkCmdDispatch");
    LOAD(fill_, PFN_vkCmdFillBuffer, "vkCmdFillBuffer");
    LOAD(barrier_, PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
#undef LOAD
    const auto fail = [&](const char* why) { error = why; destroy(); return false; };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate ||
        !update || !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ ||
        !bind_pipeline_ || !bind_sets_ || !dispatch_ || !fill_ || !barrier_)
        return fail("VSM demand Vulkan procedures unavailable");
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 3; set_info.pBindings = bindings;
    if (create_set_layout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("VSM demand descriptor layout creation failed");
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1; layout_info.pSetLayouts = &set_layout_;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("VSM demand pipeline layout creation failed");
    const uint32_t count = static_cast<uint32_t>(depths.size());
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count * 2}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = count; pool_info.poolSizeCount = 2; pool_info.pPoolSizes = sizes;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("VSM demand descriptor pool creation failed");
    sets_.resize(count);
    std::vector<VkDescriptorSetLayout> layouts(count, set_layout_);
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = pool_; alloc.descriptorSetCount = count; alloc.pSetLayouts = layouts.data();
    if (allocate(device, &alloc, sets_.data()) != VK_SUCCESS)
        return fail("VSM demand descriptor allocation failed");
    for (uint32_t i = 0; i < count; ++i)
    {
        const VkDescriptorImageInfo image{sampler, depths[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorBufferInfo buffers[]{{parameters[i]->handle(), 0, sizeof(VsmDemandParameters)},
            {needed[i]->handle(), 0, needed[i]->size()}};
        VkWriteDescriptorSet writes[3]{};
        for (uint32_t j = 0; j < 3; ++j)
        {
            writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[j].dstSet = sets_[i];
            writes[j].dstBinding = j; writes[j].descriptorCount = 1;
            writes[j].descriptorType = j == 0 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        writes[0].pImageInfo = &image; writes[1].pBufferInfo = &buffers[0]; writes[2].pBufferInfo = &buffers[1];
        update(device, 3, writes, 0, nullptr);
    }
    ShaderModule shader;
    if (!shader.initialize(device, {proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            vsm_demand_shaders::Mark, sizeof(vsm_demand_shaders::Mark), error) ||
        !create_compute_pipeline(device, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error)) { destroy(); return false; }
    extent_ = extent; parameters_ = parameters; needed_ = needed;
    error.clear(); return true;
}
bool VsmDemandPass::record(const FrameRecordingContext& frame, const VsmDemandParameters& params, std::string& error) const
{
    if (!pipeline_ || !frame.command_buffer || frame.image_index >= sets_.size() ||
        frame.extent.width != extent_.width || frame.extent.height != extent_.height ||
        !params.axis || params.axis > 65535 || !params.levels || params.levels > 6 ||
        !params.step || params.step > 65535 || params.lod_bias >= params.levels ||
        !std::isfinite(params.near_depth) || !std::isfinite(params.inverse_depth_range) ||
        params.inverse_depth_range < 0.f)
    { error = "invalid VSM demand recording request"; return false; }
    const uint64_t pages = uint64_t(params.axis) * params.axis * params.levels;
    if (pages > UINT32_MAX || needed_[frame.image_index]->size() < pages * 4 ||
        !std::all_of(params.inverse_view_projection, params.inverse_view_projection + 16,
            [](float value) { return std::isfinite(value); }) ||
        !std::all_of(params.light_view, params.light_view + 16,
            [](float value) { return std::isfinite(value); }))
    { error = "invalid VSM demand buffers or matrices"; return false; }
    for (uint32_t i = 0; i < params.levels; ++i)
        if (!std::isfinite(params.level[i][0]) || !std::isfinite(params.level[i][1]) ||
            !std::isfinite(params.level[i][2]) || params.level[i][2] <= 0.f)
        { error = "invalid VSM clipmap level"; return false; }
    if (!parameters_[frame.image_index]->write(0, &params, sizeof(params), error)) return false;
    fill_(frame.command_buffer, needed_[frame.image_index]->handle(), 0, pages * 4, 0);
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1,
        &sets_[frame.image_index], 0, nullptr);
    const uint64_t stride = uint64_t(params.step) * 8;
    dispatch_(frame.command_buffer, static_cast<uint32_t>((extent_.width + stride - 1) / stride),
        static_cast<uint32_t>((extent_.height + stride - 1) / stride), 1);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    error.clear(); return true;
}
void VsmDemandPass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE; extent_ = {}; set_layout_ = VK_NULL_HANDLE; pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; sets_.clear(); parameters_.clear(); needed_.clear();
}
}
