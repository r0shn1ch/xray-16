#include "IndirectSunPass.h"
#include "IndirectSunShaders.h"
#include "ComputePipeline.h"
#include "ShaderModule.h"
#include <cstring>

namespace xray::render::vulkan
{
namespace {
template<class T> T proc(VkDevice d, PFN_vkGetDeviceProcAddr get, const char* name)
{ return reinterpret_cast<T>(get(d, name)); }
}
bool IndirectSunPass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t slots,
    VkDeviceSize max_storage_range, VkPipelineCache cache, std::string& error)
{
    destroy();
    if (!device || !get || !slots || slots > 16 || !max_storage_range)
    { error = "invalid indirect sun pass configuration"; return false; }
    device_ = device;
    max_storage_range_ = max_storage_range;
#define LOAD(field, type, name) field = proc<type>(device, get, name)
    auto create_set = proc<PFN_vkCreateDescriptorSetLayout>(device, get, "vkCreateDescriptorSetLayout");
    LOAD(destroy_set_layout_, PFN_vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
    auto create_pool = proc<PFN_vkCreateDescriptorPool>(device, get, "vkCreateDescriptorPool");
    LOAD(destroy_pool_, PFN_vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
    auto allocate = proc<PFN_vkAllocateDescriptorSets>(device, get, "vkAllocateDescriptorSets");
    LOAD(update_sets_, PFN_vkUpdateDescriptorSets, "vkUpdateDescriptorSets");
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
    auto fail = [&](const char* why) { error = why; destroy(); return false; };
    if (!create_set || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate || !update_sets_ ||
        !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ || !bind_pipeline_ ||
        !bind_sets_ || !push_constants_ || !dispatch_ || !barrier_)
        return fail("indirect sun Vulkan procedures unavailable");
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 2; set_info.pBindings = bindings;
    if (create_set(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("indirect sun descriptor layout creation failed");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1; layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1; layout_info.pPushConstantRanges = &range;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("indirect sun pipeline layout creation failed");
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, slots * 2};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = slots; pool_info.poolSizeCount = 1; pool_info.pPoolSizes = &size;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("indirect sun descriptor pool creation failed");
    sets_.resize(slots);
    std::vector<VkDescriptorSetLayout> layouts(slots, set_layout_);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = pool_; allocation.descriptorSetCount = slots; allocation.pSetLayouts = layouts.data();
    if (allocate(device, &allocation, sets_.data()) != VK_SUCCESS)
        return fail("indirect sun descriptor allocation failed");
    ShaderModule shader;
    if (!shader.initialize(device, {proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            indirect_sun_shaders::Generate, sizeof(indirect_sun_shaders::Generate), error) ||
        !create_compute_pipeline(device, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error))
    { destroy(); return false; }
    error.clear(); return true;
}

bool IndirectSunPass::record(VkCommandBuffer command, uint32_t slot, const BufferResource& surfaces,
    const BufferResource& rays, uint32_t count, const float* direction, std::string& error) const
{
    const VkDeviceSize surface_bytes = VkDeviceSize(count) * sizeof(RaySurface);
    const VkDeviceSize ray_bytes = VkDeviceSize(count) * sizeof(RayQueryInput);
    if (!pipeline_ || !command || slot >= sets_.size() || !direction || !surfaces.handle() || !rays.handle() ||
        surface_bytes > surfaces.size() || ray_bytes > rays.size() || surface_bytes > max_storage_range_ ||
        ray_bytes > max_storage_range_ || (uint64_t(count) + 63) / 64 > uint64_t(65535) * 65535)
    { error = "invalid indirect sun buffers or dispatch"; return false; }
    if (!count) { error.clear(); return true; }
    const VkDescriptorBufferInfo infos[]{{surfaces.handle(), 0, surface_bytes}, {rays.handle(), 0, ray_bytes}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = sets_[slot];
        writes[i].dstBinding = i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
    }
    update_sets_(device_, 2, writes, 0, nullptr);
    bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets_[slot], 0, nullptr);
    uint32_t groups = static_cast<uint32_t>((uint64_t(count) + 63) / 64);
    uint32_t groups_x = groups < 65535u ? groups : 65535u;
    struct Constants { float direction[4]; uint32_t count, groups_x, pad[2]; } constants{};
    std::memcpy(constants.direction, direction, sizeof(constants.direction));
    constants.count = count; constants.groups_x = groups_x;
    push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    dispatch_(command, groups_x, (groups + groups_x - 1) / groups_x, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier_(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    error.clear(); return true;
}
void IndirectSunPass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE; set_layout_ = VK_NULL_HANDLE; max_storage_range_ = 0; sets_.clear();
}
}
