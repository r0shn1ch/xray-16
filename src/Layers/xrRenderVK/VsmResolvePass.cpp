#include "VsmResolvePass.h"
#include "VsmResolveShaders.h"
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
bool VsmResolvePass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
    const std::vector<VkImageView>& depths, VkImageView static_atlas, VkImageView dynamic_atlas,
    VkSampler sampler, const std::vector<BufferResource*>& static_tables,
    const std::vector<BufferResource*>& dynamic_tables, const std::vector<BufferResource*>& parameters,
    const std::vector<VkImage>& images, const std::vector<VkImageView>& views,
    VkPipelineCache cache, std::string& error)
{
    destroy();
    if (!device || !get || !sampler || !static_atlas || !dynamic_atlas || !extent.width || !extent.height ||
        depths.empty() || depths.size() > 16 || depths.size() != static_tables.size() ||
        depths.size() != dynamic_tables.size() || depths.size() != parameters.size() ||
        depths.size() != images.size() || depths.size() != views.size() ||
        (uint64_t(extent.width) + 7) / 8 > 65535 || (uint64_t(extent.height) + 7) / 8 > 65535)
    { error = "invalid VSM resolve extent or resources"; return false; }
    for (size_t i = 0; i < depths.size(); ++i)
        if (!depths[i] || !images[i] || !views[i] || !static_tables[i] || !static_tables[i]->handle() ||
            !dynamic_tables[i] || !dynamic_tables[i]->handle() || !parameters[i] || !parameters[i]->handle() ||
            parameters[i]->size() < sizeof(VsmResolveParameters))
        { error = "VSM resolve input or output unavailable"; return false; }
    device_ = device;
#define LOAD(field, type, name) field = proc<type>(device, get, name)
    auto create_set_layout = proc<PFN_vkCreateDescriptorSetLayout>(device, get, "vkCreateDescriptorSetLayout");
    LOAD(destroy_set_layout_, PFN_vkDestroyDescriptorSetLayout, "vkDestroyDescriptorSetLayout");
    auto create_pool = proc<PFN_vkCreateDescriptorPool>(device, get, "vkCreateDescriptorPool");
    LOAD(destroy_pool_, PFN_vkDestroyDescriptorPool, "vkDestroyDescriptorPool");
    auto allocate = proc<PFN_vkAllocateDescriptorSets>(device, get, "vkAllocateDescriptorSets");
    auto update = proc<PFN_vkUpdateDescriptorSets>(device, get, "vkUpdateDescriptorSets");
    update_sets_ = update;
    auto create_layout = proc<PFN_vkCreatePipelineLayout>(device, get, "vkCreatePipelineLayout");
    LOAD(destroy_layout_, PFN_vkDestroyPipelineLayout, "vkDestroyPipelineLayout");
    auto create_pipeline = proc<PFN_vkCreateComputePipelines>(device, get, "vkCreateComputePipelines");
    LOAD(destroy_pipeline_, PFN_vkDestroyPipeline, "vkDestroyPipeline");
    LOAD(bind_pipeline_, PFN_vkCmdBindPipeline, "vkCmdBindPipeline");
    LOAD(bind_sets_, PFN_vkCmdBindDescriptorSets, "vkCmdBindDescriptorSets");
    LOAD(dispatch_, PFN_vkCmdDispatch, "vkCmdDispatch");
    LOAD(barrier_, PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
#undef LOAD
    const auto fail = [&](const char* why) { error = why; destroy(); return false; };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate ||
        !update || !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ ||
        !bind_pipeline_ || !bind_sets_ || !dispatch_ || !barrier_)
        return fail("VSM resolve Vulkan procedures unavailable");
    VkDescriptorSetLayoutBinding bindings[8]{};
    for (uint32_t i = 0; i < 8; ++i)
    {
        bindings[i].binding = i; bindings[i].descriptorCount = 1; bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[i].descriptorType = i < 3 || i == 7 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
            i < 6 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 8; set_info.pBindings = bindings;
    if (create_set_layout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("VSM resolve descriptor layout creation failed");
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1; layout_info.pSetLayouts = &set_layout_;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("VSM resolve pipeline layout creation failed");
    const uint32_t count = static_cast<uint32_t>(depths.size());
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count * 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count * 3}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, count}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = count; pool_info.poolSizeCount = 3; pool_info.pPoolSizes = sizes;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("VSM resolve descriptor pool creation failed");
    sets_.resize(count);
    std::vector<VkDescriptorSetLayout> layouts(count, set_layout_);
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = pool_; alloc.descriptorSetCount = count; alloc.pSetLayouts = layouts.data();
    if (allocate(device, &alloc, sets_.data()) != VK_SUCCESS)
        return fail("VSM resolve descriptor allocation failed");
    for (uint32_t i = 0; i < count; ++i)
    {
        const VkDescriptorImageInfo sampled[]{{sampler, depths[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            {sampler, static_atlas, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL},
            {sampler, dynamic_atlas, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL}};
        const VkDescriptorBufferInfo buffers[]{{static_tables[i]->handle(), 0, static_tables[i]->size()},
            {dynamic_tables[i]->handle(), 0, dynamic_tables[i]->size()}, {parameters[i]->handle(), 0, sizeof(VsmResolveParameters)}};
        const VkDescriptorImageInfo output{VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[8]{};
        for (uint32_t j = 0; j < 8; ++j)
        {
            writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[j].dstSet = sets_[i];
            writes[j].dstBinding = j; writes[j].descriptorCount = 1; writes[j].descriptorType = bindings[j].descriptorType;
            if (j < 3) writes[j].pImageInfo = &sampled[j];
            else if (j < 6) writes[j].pBufferInfo = &buffers[j - 3];
            else if (j == 6) writes[j].pImageInfo = &output;
            else writes[j].pImageInfo = &sampled[0];
        }
        update(device, 8, writes, 0, nullptr);
    }
    ShaderModule shader;
    if (!shader.initialize(device, {proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            vsm_resolve_shaders::Resolve, sizeof(vsm_resolve_shaders::Resolve), error) ||
        !create_compute_pipeline(device, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error)) { destroy(); return false; }
    extent_ = extent; images_ = images; fallback_views_ = depths; fallback_sampler_ = sampler;
    static_tables_ = static_tables; dynamic_tables_ = dynamic_tables; parameters_ = parameters;
    error.clear(); return true;
}

bool VsmResolvePass::record(const FrameRecordingContext& frame, const VsmResolveParameters& params,
    VkImageLayout old_layout, std::string& error, uint32_t table_image,
    const BufferResource* static_override, const BufferResource* dynamic_override,
    VkImageView ray_view, VkSampler ray_sampler) const
{
    if (table_image == UINT32_MAX) table_image = frame.image_index;
    if (!pipeline_ || !frame.command_buffer || frame.image_index >= sets_.size() ||
        table_image >= static_tables_.size() || !update_sets_ || bool(ray_view) != bool(ray_sampler) ||
        frame.extent.width != extent_.width || frame.extent.height != extent_.height ||
        (old_layout != VK_IMAGE_LAYOUT_UNDEFINED && old_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
        !params.axis || params.axis > 65535 || !params.levels || params.levels > 6 || !params.page_size ||
        !params.static_columns || params.page_size > INT32_MAX || (params.dynamic_enabled && !params.dynamic_columns) ||
        !std::isfinite(params.near_depth) || !std::isfinite(params.inverse_depth_range) || params.inverse_depth_range <= 0.f ||
        !std::isfinite(params.bias) || params.bias < 0.f)
    { error = "invalid VSM resolve recording request"; return false; }
    const auto atlas_fits = [](uint32_t size, uint32_t columns, uint32_t slots) {
        return uint64_t(size) * columns <= INT32_MAX &&
            uint64_t(size) * ((uint64_t(slots) + columns - 1) / columns) <= INT32_MAX;
    };
    if (!atlas_fits(params.page_size, params.static_columns, params.static_slots) ||
        (params.dynamic_enabled && !atlas_fits(params.page_size, params.dynamic_columns, params.dynamic_slots)))
    { error = "VSM resolve atlas dimensions overflow"; return false; }
    const auto* static_table = static_override ? static_override : static_tables_[table_image];
    const auto* dynamic_table = dynamic_override ? dynamic_override : dynamic_tables_[table_image];
    const uint64_t count = uint64_t(params.axis) * params.axis * params.levels;
    if (count > UINT32_MAX || !static_table->handle() || !dynamic_table->handle() ||
        static_table->size() < count * 4 || dynamic_table->size() < count * 4 ||
        !std::all_of(params.inverse_view_projection, params.inverse_view_projection + 16, [](float v) { return std::isfinite(v); }) ||
        !std::all_of(params.light_view, params.light_view + 16, [](float v) { return std::isfinite(v); }))
    { error = "invalid VSM resolve tables or matrices"; return false; }
    for (uint32_t i = 0; i < params.levels; ++i)
        if (!std::isfinite(params.level[i][0]) || !std::isfinite(params.level[i][1]) ||
            !std::isfinite(params.level[i][2]) || params.level[i][2] <= 0.f)
        { error = "invalid VSM resolve clipmap level"; return false; }
    auto effective_params = params;
    effective_params.reserved = ray_view ? 1.f : 0.f;
    if (!parameters_[frame.image_index]->write(0, &effective_params, sizeof(effective_params), error)) return false;
    // The descriptor belongs to the acquired image and is only updated after
    // its fence has completed. Table data may come from an earlier image.
    const VkDescriptorBufferInfo tables[]{{static_table->handle(), 0, static_table->size()},
        {dynamic_table->handle(), 0, dynamic_table->size()}};
    const VkDescriptorImageInfo ray{ray_view ? ray_sampler : fallback_sampler_,
        ray_view ? ray_view : fallback_views_[frame.image_index], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 2; ++i)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[i].dstSet = sets_[frame.image_index];
        writes[i].dstBinding = 3 + i; writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &tables[i];
    }
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[2].dstSet = sets_[frame.image_index];
    writes[2].dstBinding = 7; writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; writes[2].pImageInfo = &ray;
    update_sets_(device_, 3, writes, 0, nullptr);
    VkImageMemoryBarrier image{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image.srcAccessMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_SHADER_READ_BIT;
    image.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; image.oldLayout = old_layout; image.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image.srcQueueFamilyIndex = image.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image.image = images_[frame.image_index]; image.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier_(frame.command_buffer, old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image);
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets_[frame.image_index], 0, nullptr);
    dispatch_(frame.command_buffer, (extent_.width + 7) / 8, (extent_.height + 7) / 8, 1);
    image.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; image.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image.oldLayout = VK_IMAGE_LAYOUT_GENERAL; image.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkMemoryBarrier tables_read{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    tables_read.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    tables_read.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &tables_read, 0, nullptr, 1, &image);
    error.clear(); return true;
}
void VsmResolvePass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE; extent_ = {}; set_layout_ = VK_NULL_HANDLE; pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; sets_.clear(); images_.clear();
    static_tables_.clear(); dynamic_tables_.clear(); parameters_.clear();
    fallback_views_.clear(); fallback_sampler_ = VK_NULL_HANDLE;
    update_sets_ = nullptr;
}
}
