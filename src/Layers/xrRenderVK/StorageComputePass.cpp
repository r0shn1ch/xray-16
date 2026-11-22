#include "StorageComputePass.h"
#include "ComputePipeline.h"
#include "ShaderModule.h"
#include <algorithm>

namespace xray::render::vulkan
{
namespace {
template<class T> T proc(VkDevice d, PFN_vkGetDeviceProcAddr get, const char* name)
{ return reinterpret_cast<T>(get(d, name)); }
}
bool StorageComputePass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t slots,
    uint32_t bindings, uint32_t push_size, VkDeviceSize max_storage_range, VkPipelineCache cache,
    const uint32_t* code, size_t code_size, std::string& error)
{
    if (!bindings || bindings > 8) { destroy(); error = "invalid storage compute configuration"; return false; }
    std::vector<Binding> layout;
    for (uint32_t i = 0; i < bindings; ++i) layout.push_back({i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
    return initialize(device, get, slots, layout, push_size, max_storage_range, 0, cache, code, code_size, error);
}
bool StorageComputePass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, uint32_t slots,
    const std::vector<Binding>& bindings, uint32_t push_size, VkDeviceSize max_storage_range,
    VkDeviceSize max_uniform_range, VkPipelineCache cache, const uint32_t* code, size_t code_size,
    std::string& error, VkDescriptorSetLayout external_layout)
{
    destroy();
    if (!device || !get || !slots || slots > 1024 || bindings.empty() || bindings.size() > 32 ||
        push_size > 128 || push_size % 4 || !max_storage_range || !code || !code_size)
    { error = "invalid storage compute configuration"; return false; }
    uint32_t storage_count = 0, uniform_count = 0, sampled_image_count = 0, storage_image_count = 0;
    uint32_t texel_count = 0;
    for (size_t i = 0; i < bindings.size(); ++i)
    {
        if (bindings[i].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ++storage_count;
        else if (bindings[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER && max_uniform_range) ++uniform_count;
        else if (bindings[i].type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) ++sampled_image_count;
        else if (bindings[i].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) ++storage_image_count;
        else if (bindings[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER && !texel_count) ++texel_count;
        else { error = "invalid storage compute descriptor type"; return false; }
        for (size_t j = 0; j < i; ++j)
            if (bindings[j].number == bindings[i].number)
            { error = "duplicate storage compute binding"; return false; }
    }
    device_ = device; bindings_ = bindings; push_size_ = push_size; external_layout_ = external_layout != VK_NULL_HANDLE;
    max_storage_range_ = max_storage_range; max_uniform_range_ = max_uniform_range;
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
    LOAD(dispatch_indirect_, PFN_vkCmdDispatchIndirect, "vkCmdDispatchIndirect");
    LOAD(barrier_, PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
#undef LOAD
    auto fail = [&](const char* why) { error = why; destroy(); return false; };
    if (!create_set || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate || !update_sets_ ||
        !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ || !bind_pipeline_ ||
        !bind_sets_ || (push_size && !push_constants_) || !dispatch_ || !barrier_)
        return fail("storage compute Vulkan procedures unavailable");
    std::vector<VkDescriptorSetLayoutBinding> set_bindings(bindings.size());
    for (size_t i = 0; i < bindings.size(); ++i)
        set_bindings[i] = {bindings[i].number, bindings[i].type, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = static_cast<uint32_t>(bindings.size()); set_info.pBindings = set_bindings.data();
    if (create_set(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("storage compute descriptor layout creation failed");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    const VkDescriptorSetLayout layouts_for_pipeline[]{set_layout_, external_layout};
    layout_info.setLayoutCount = external_layout_ ? 2 : 1; layout_info.pSetLayouts = layouts_for_pipeline;
    layout_info.pushConstantRangeCount = push_size ? 1 : 0; layout_info.pPushConstantRanges = push_size ? &range : nullptr;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("storage compute pipeline layout creation failed");
    std::vector<VkDescriptorPoolSize> sizes;
    if (storage_count) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, slots * storage_count});
    if (uniform_count) sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, slots * uniform_count});
    if (sampled_image_count) sizes.push_back({VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, slots * sampled_image_count});
    if (storage_image_count) sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, slots * storage_image_count});
    if (texel_count) sizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, slots * texel_count});
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = slots; pool_info.poolSizeCount = static_cast<uint32_t>(sizes.size()); pool_info.pPoolSizes = sizes.data();
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("storage compute descriptor pool creation failed");
    sets_.resize(slots);
    std::vector<VkDescriptorSetLayout> layouts(slots, set_layout_);
    VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocation.descriptorPool = pool_; allocation.descriptorSetCount = slots; allocation.pSetLayouts = layouts.data();
    if (allocate(device, &allocation, sets_.data()) != VK_SUCCESS)
        return fail("storage compute descriptor allocation failed");
    ShaderModule shader;
    if (!shader.initialize(device, {proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")}, code, code_size, error) ||
        !create_compute_pipeline(device, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error))
    { destroy(); return false; }
    error.clear(); return true;
}
bool StorageComputePass::record(VkCommandBuffer command, uint32_t slot,
    const std::vector<VkDescriptorBufferInfo>& buffers, const void* constants,
    uint32_t groups_x, uint32_t groups_y, std::string& error, VkDescriptorSet external_set) const
{
    return record_mixed(command, slot, buffers, {}, constants, groups_x, groups_y, error, external_set);
}

bool StorageComputePass::record_mixed(VkCommandBuffer command, uint32_t slot,
    const std::vector<VkDescriptorBufferInfo>& buffers, const std::vector<VkDescriptorImageInfo>& images,
    const void* constants, uint32_t groups_x, uint32_t groups_y, std::string& error,
    VkDescriptorSet external_set, VkBufferView uniform_texels,
    VkBuffer indirect_arguments, VkDeviceSize indirect_offset) const
{
    const size_t expected_buffers = static_cast<size_t>(std::count_if(bindings_.begin(), bindings_.end(),
        [](const Binding& binding) { return binding.type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
            binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER; }));
    const bool expects_texels = std::any_of(bindings_.begin(), bindings_.end(), [](const Binding& binding) {
        return binding.type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    });
    const size_t expected_images = bindings_.size() - expected_buffers - size_t(expects_texels);
    if (!pipeline_ || !command || slot >= sets_.size() || buffers.size() != expected_buffers ||
        images.size() != expected_images || expects_texels != (uniform_texels != VK_NULL_HANDLE) || (push_size_ && !constants) ||
        !groups_x || !groups_y || groups_x > 65535 || groups_y > 65535 ||
        (indirect_arguments && (!dispatch_indirect_ || indirect_offset % 4)) ||
        (!indirect_arguments && indirect_offset) || external_layout_ != (external_set != VK_NULL_HANDLE))
    { error = "invalid storage compute dispatch"; return false; }
    std::vector<VkWriteDescriptorSet> writes(bindings_.size());
    size_t buffer_index = 0, image_index = 0;
    for (size_t i = 0; i < bindings_.size(); ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = sets_[slot];
        writes[i].dstBinding = bindings_[i].number; writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings_[i].type;
        if (bindings_[i].type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER ||
            bindings_[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        {
            const VkDeviceSize limit = bindings_[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ?
                max_uniform_range_ : max_storage_range_;
            if (!buffers[buffer_index].buffer || !buffers[buffer_index].range || buffers[buffer_index].range > limit)
            {
                error = "invalid storage compute buffer range at binding " +
                    std::to_string(bindings_[i].number) + " (" +
                    std::to_string(buffers[buffer_index].range) + " bytes; limit " +
                    std::to_string(limit) + ")";
                return false;
            }
            writes[i].pBufferInfo = &buffers[buffer_index++];
        }
        else if (bindings_[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER)
            writes[i].pTexelBufferView = &uniform_texels;
        else
        {
            const auto& image = images[image_index++];
            const bool storage = bindings_[i].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            if (!image.imageView || (storage ? image.imageLayout != VK_IMAGE_LAYOUT_GENERAL :
                !image.sampler || image.imageLayout == VK_IMAGE_LAYOUT_UNDEFINED))
            { error = "invalid storage compute image descriptor"; return false; }
            writes[i].pImageInfo = &image;
        }
    }
    update_sets_(device_, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets_[slot], 0, nullptr);
    if (external_layout_) bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 1, 1, &external_set, 0, nullptr);
    if (push_size_) push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size_, constants);
    if (indirect_arguments)
        dispatch_indirect_(command, indirect_arguments, indirect_offset);
    else
        dispatch_(command, groups_x, groups_y, 1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    barrier_(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
    error.clear(); return true;
}
bool StorageComputePass::record_indirect(VkCommandBuffer command, uint32_t slot,
    const std::vector<VkDescriptorBufferInfo>& buffers, const void* constants,
    const BufferResource& arguments, VkDeviceSize offset, std::string& error) const
{
    if (!arguments.handle() || offset > arguments.size() ||
        sizeof(VkDispatchIndirectCommand) > arguments.size() - offset)
    { error = "invalid storage compute indirect range"; return false; }
    return record_mixed(command, slot, buffers, {}, constants, 1, 1, error,
        VK_NULL_HANDLE, VK_NULL_HANDLE, arguments.handle(), offset);
}

void StorageComputePass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE; set_layout_ = VK_NULL_HANDLE; max_storage_range_ = max_uniform_range_ = 0;
    bindings_.clear(); push_size_ = 0; external_layout_ = false; sets_.clear();
}
}
