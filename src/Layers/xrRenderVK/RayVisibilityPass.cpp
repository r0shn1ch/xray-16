#include "RayVisibilityPass.h"
#include "RayVisibilityShaders.h"
#include "IndirectResolveShaders.h"
#include "ComputePipeline.h"
#include "ShaderModule.h"

namespace xray::render::vulkan
{
namespace {
template<class T> T proc(VkDevice d, PFN_vkGetDeviceProcAddr get, const char* n)
{ return reinterpret_cast<T>(get(d, n)); }
}
bool RayVisibilityPass::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
    const std::vector<VkImage>& images, const std::vector<VkImageView>& views,
    const std::vector<BufferResource*>& hits, VkPipelineCache cache, std::string& error, Mode mode)
{
    destroy();
    const uint64_t pixels = uint64_t(extent.width) * extent.height;
    const size_t element_size = mode == Mode::IndirectRadiance ? sizeof(float) * 4 : sizeof(RayQueryHit);
    if (!device || !get || !pixels || pixels > UINT32_MAX || images.empty() || images.size() > 16 ||
        images.size() != views.size() || images.size() != hits.size() ||
        (uint64_t(extent.width) + 7) / 8 > 65535 || (uint64_t(extent.height) + 7) / 8 > 65535)
    { error = "invalid ray visibility extent or resources"; return false; }
    for (size_t i = 0; i < images.size(); ++i)
        if (!images[i] || !views[i] || !hits[i] || !hits[i]->handle() ||
            hits[i]->size() < pixels * element_size)
        { error = "ray visibility input or output is unavailable"; return false; }
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
    LOAD(barrier_, PFN_vkCmdPipelineBarrier, "vkCmdPipelineBarrier");
#undef LOAD
    const auto fail = [&](const char* why) { error = why; destroy(); return false; };
    if (!create_set_layout || !destroy_set_layout_ || !create_pool || !destroy_pool_ || !allocate ||
        !update || !create_layout || !destroy_layout_ || !create_pipeline || !destroy_pipeline_ ||
        !bind_pipeline_ || !bind_sets_ || !dispatch_ || !barrier_)
        return fail("ray visibility Vulkan procedures unavailable");
    const VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 2; set_info.pBindings = bindings;
    if (create_set_layout(device, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("ray visibility descriptor layout creation failed");
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1; layout_info.pSetLayouts = &set_layout_;
    if (create_layout(device, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("ray visibility pipeline layout creation failed");
    uint32_t count = static_cast<uint32_t>(images.size());
    const VkDescriptorPoolSize sizes[]{{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, count}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = count; pool_info.poolSizeCount = 2; pool_info.pPoolSizes = sizes;
    if (create_pool(device, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("ray visibility descriptor pool creation failed");
    sets_.resize(count);
    std::vector<VkDescriptorSetLayout> layouts(count, set_layout_);
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = pool_; alloc.descriptorSetCount = count; alloc.pSetLayouts = layouts.data();
    if (allocate(device, &alloc, sets_.data()) != VK_SUCCESS)
        return fail("ray visibility descriptor allocation failed");
    for (uint32_t i = 0; i < count; ++i)
    {
        const VkDescriptorBufferInfo buffer{hits[i]->handle(), 0, pixels * element_size};
        const VkDescriptorImageInfo image{VK_NULL_HANDLE, views[i], VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[2]{};
        for (uint32_t j = 0; j < 2; ++j)
        { writes[j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; writes[j].dstSet = sets_[i];
          writes[j].dstBinding = j; writes[j].descriptorCount = 1; }
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[0].pBufferInfo = &buffer;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[1].pImageInfo = &image;
        update(device, 2, writes, 0, nullptr);
    }
    ShaderModule shader;
    const uint32_t* code = mode == Mode::IndirectRadiance ? indirect_resolve_shaders::Resolve : ray_visibility_shaders::Resolve;
    const size_t code_size = mode == Mode::IndirectRadiance ? sizeof(indirect_resolve_shaders::Resolve) : sizeof(ray_visibility_shaders::Resolve);
    if (!shader.initialize(device, {proc<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            proc<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            code, code_size, error) ||
        !create_compute_pipeline(device, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error)) { destroy(); return false; }
    images_ = images; extent_ = extent; error.clear(); return true;
}
bool RayVisibilityPass::record(const FrameRecordingContext& frame, VkImageLayout old_layout) const
{
    if (!pipeline_ || !frame.command_buffer || frame.image_index >= sets_.size() ||
        (old_layout != VK_IMAGE_LAYOUT_UNDEFINED && old_layout != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) ||
        frame.extent.width != extent_.width || frame.extent.height != extent_.height) return false;
    VkImageMemoryBarrier image{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    image.srcAccessMask = old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? 0 : VK_ACCESS_SHADER_READ_BIT;
    image.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    image.oldLayout = old_layout; image.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    image.srcQueueFamilyIndex = image.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    image.image = images_[frame.image_index];
    image.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    barrier_(frame.command_buffer, old_layout == VK_IMAGE_LAYOUT_UNDEFINED ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT :
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &image);
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &sets_[frame.image_index], 0, nullptr);
    dispatch_(frame.command_buffer, (extent_.width + 7) / 8, (extent_.height + 7) / 8, 1);
    image.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; image.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    image.oldLayout = VK_IMAGE_LAYOUT_GENERAL; image.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &image);
    return true;
}
void RayVisibilityPass::destroy()
{
    if (device_ && pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
    if (device_ && layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
    if (device_ && pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
    if (device_ && set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
    device_ = VK_NULL_HANDLE; extent_ = {}; set_layout_ = VK_NULL_HANDLE; pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE; pipeline_ = VK_NULL_HANDLE; sets_.clear(); images_.clear();
}
}
