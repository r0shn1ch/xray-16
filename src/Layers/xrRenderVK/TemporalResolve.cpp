#include "TemporalResolve.h"
#include "TemporalResolveShaders.h"
#include "ComputePipeline.h"
#include "ShaderModule.h"
#include <algorithm>
#include <cmath>
#include <utility>

namespace xray::render::vulkan
{
namespace
{
template <typename T> T procedure(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{ return reinterpret_cast<T>(get(device, name)); }
struct Parameters { float weight; uint32_t valid; uint32_t reprojected_depth; float current_x, current_y, previous_x, previous_y; };
static_assert(sizeof(Parameters) == 28);
}

bool TemporalResolve::create_target(Target& target,
    const VkPhysicalDeviceMemoryProperties& memory, std::string& error)
{
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    info.extent = {extent_.width, extent_.height, 1};
    info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sharing_.apply(info);
    if (image_pool_)
    {
        if (!image_pool_->acquire(info, target.image, error)) return false;
    }
    else
    {
        if (images_.create_image(device_, &info, nullptr, &target.image) != VK_SUCCESS)
        { error = "temporal resolve image creation failed"; return false; }
        VkMemoryRequirements requirements{};
        images_.get_image_memory_requirements(device_, target.image, &requirements);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            { type = i; break; }
        if (type == UINT32_MAX) { error = "temporal resolve device memory unavailable"; return false; }
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (images_.allocate_memory(device_, &allocation, nullptr, &target.memory) != VK_SUCCESS ||
            images_.bind_image_memory(device_, target.image, target.memory, 0) != VK_SUCCESS)
        { error = "temporal resolve image allocation failed"; return false; }
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = target.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = info.format;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (images_.create_image_view(device_, &view, nullptr, &target.view) != VK_SUCCESS)
    { error = "temporal resolve image view creation failed"; return false; }
    return true;
}

bool TemporalResolve::initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    VkExtent2D extent, const std::vector<VkImageView>& scene_views,
    const std::vector<VkImageView>& motion_views,
    const std::vector<VkImageView>& depth_views,
    const VkPhysicalDeviceMemoryProperties& memory, const FrameDispatch& images,
    VkPipelineCache cache, std::string& error, FrameImagePool* image_pool, bool reprojected_depth,
    bool indirect, bool upscale, FrameImageSharing sharing)
{
    destroy();
    reprojected_depth_ = reprojected_depth;
    indirect_ = indirect;
    upscale_ = upscale;
    sharing_ = std::move(sharing);
    if (!physical || !device || !get || !extent.width || !extent.height || scene_views.empty() ||
        scene_views.size() != motion_views.size() || scene_views.size() != depth_views.size() ||
        scene_views.size() > 16 ||
        std::any_of(scene_views.begin(), scene_views.end(), [](VkImageView view) { return !view; }) ||
        std::any_of(motion_views.begin(), motion_views.end(), [](VkImageView view) { return !view; }) ||
        std::any_of(depth_views.begin(), depth_views.end(), [](VkImageView view) { return !view; }) ||
        !images.get_format_properties || !images.create_image || !images.destroy_image ||
        !images.get_image_memory_requirements || !images.allocate_memory || !images.free_memory ||
        !images.bind_image_memory || !images.create_image_view || !images.destroy_image_view)
    { error = "temporal resolve requires scene, motion, depth and image procedures"; return false; }
    VkFormatProperties support{};
    images.get_format_properties(physical, VK_FORMAT_R16G16B16A16_SFLOAT, &support);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | (indirect ? 0 : VK_FORMAT_FEATURE_BLIT_SRC_BIT);
    if ((support.optimalTilingFeatures & required) != required)
    { error = "RGBA16F temporal history format features unavailable"; return false; }
    device_ = device;
    extent_ = extent;
    images_ = images;
    image_pool_ = image_pool;
    auto create_sampler = procedure<PFN_vkCreateSampler>(device, get, "vkCreateSampler");
    destroy_sampler_ = procedure<PFN_vkDestroySampler>(device, get, "vkDestroySampler");
    auto create_set_layout = procedure<PFN_vkCreateDescriptorSetLayout>(device, get, "vkCreateDescriptorSetLayout");
    destroy_set_layout_ = procedure<PFN_vkDestroyDescriptorSetLayout>(device, get, "vkDestroyDescriptorSetLayout");
    auto create_pool = procedure<PFN_vkCreateDescriptorPool>(device, get, "vkCreateDescriptorPool");
    destroy_pool_ = procedure<PFN_vkDestroyDescriptorPool>(device, get, "vkDestroyDescriptorPool");
    auto allocate_sets = procedure<PFN_vkAllocateDescriptorSets>(device, get, "vkAllocateDescriptorSets");
    auto update_sets = procedure<PFN_vkUpdateDescriptorSets>(device, get, "vkUpdateDescriptorSets");
    auto create_layout = procedure<PFN_vkCreatePipelineLayout>(device, get, "vkCreatePipelineLayout");
    destroy_layout_ = procedure<PFN_vkDestroyPipelineLayout>(device, get, "vkDestroyPipelineLayout");
    auto create_pipeline = procedure<PFN_vkCreateComputePipelines>(device, get, "vkCreateComputePipelines");
    destroy_pipeline_ = procedure<PFN_vkDestroyPipeline>(device, get, "vkDestroyPipeline");
    bind_pipeline_ = procedure<PFN_vkCmdBindPipeline>(device, get, "vkCmdBindPipeline");
    bind_sets_ = procedure<PFN_vkCmdBindDescriptorSets>(device, get, "vkCmdBindDescriptorSets");
    push_constants_ = procedure<PFN_vkCmdPushConstants>(device, get, "vkCmdPushConstants");
    dispatch_ = procedure<PFN_vkCmdDispatch>(device, get, "vkCmdDispatch");
    barrier_ = procedure<PFN_vkCmdPipelineBarrier>(device, get, "vkCmdPipelineBarrier");
    const auto fail = [&](const char* message) { error = message; destroy(); return false; };
    if (!create_sampler || !destroy_sampler_ || !create_set_layout || !destroy_set_layout_ ||
        !create_pool || !destroy_pool_ || !allocate_sets || !update_sets || !create_layout ||
        !destroy_layout_ || !create_pipeline || !destroy_pipeline_ || !bind_pipeline_ ||
        !bind_sets_ || !push_constants_ || !dispatch_ || !barrier_)
        return fail("temporal resolve compute procedures unavailable");
    for (Target& target : targets_)
        if (!create_target(target, memory, error)) { destroy(); return false; }
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = VK_LOD_CLAMP_NONE;
    if (create_sampler(device_, &sampler, nullptr, &sampler_) != VK_SUCCESS)
        return fail("temporal resolve sampler creation failed");
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 5;
    set_info.pBindings = bindings;
    if (create_set_layout(device_, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("temporal resolve descriptor layout creation failed");
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Parameters)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    if (create_layout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("temporal resolve pipeline layout creation failed");
    const uint32_t count = uint32_t(scene_views.size() * 2);
    const VkDescriptorPoolSize sizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count * 4},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, count}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = count;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = sizes;
    if (create_pool(device_, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("temporal resolve descriptor pool creation failed");
    ShaderModule shader;
    if (!shader.initialize(device_, {procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            temporal_resolve_shaders::Resolve, sizeof(temporal_resolve_shaders::Resolve), error))
    { destroy(); return false; }
    if (!create_compute_pipeline(device_, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error))
    { destroy(); return false; }
    sets_.resize(count);
    std::vector<VkDescriptorSetLayout> layouts(count, set_layout_);
    VkDescriptorSetAllocateInfo sets_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sets_info.descriptorPool = pool_;
    sets_info.descriptorSetCount = count;
    sets_info.pSetLayouts = layouts.data();
    if (allocate_sets(device_, &sets_info, sets_.data()) != VK_SUCCESS)
        return fail("temporal resolve descriptor allocation failed");
    for (uint32_t image = 0; image < scene_views.size(); ++image)
        for (uint32_t output = 0; output < 2; ++output)
        {
            const VkDescriptorImageInfo infos[]{
                {sampler_, scene_views[image], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {sampler_, motion_views[image], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {sampler_, targets_[output ^ 1].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
                {VK_NULL_HANDLE, targets_[output].view, VK_IMAGE_LAYOUT_GENERAL},
                {sampler_, depth_views[image], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
            VkWriteDescriptorSet writes[5]{};
            for (uint32_t binding = 0; binding < 5; ++binding)
            {
                writes[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[binding].dstSet = sets_[image * 2 + output];
                writes[binding].dstBinding = binding;
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType = binding == 3 ?
                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                writes[binding].pImageInfo = &infos[binding];
            }
            update_sets(device_, 5, writes, 0, nullptr);
        }
    reset_history();
    output_slot_ = 0;
    error.clear();
    return true;
}

void TemporalResolve::set_history_weight(float weight)
{ history_weight_ = std::isfinite(weight) ? std::clamp(weight, 0.f, .95f) : .9f; }
void TemporalResolve::set_jitter(float x, float y)
{ current_jitter_ = {std::isfinite(x) ? x : 0.f, std::isfinite(y) ? y : 0.f}; }

bool TemporalResolve::record(const FrameRecordingContext& frame)
{
    if (!ready() || !active_ || !frame.command_buffer || frame.image_index * 2 + output_slot_ >= sets_.size())
        return false;
    const Parameters params{history_weight_, history_valid_ ? 1u : 0u,
        (reprojected_depth_ ? 1u : 0u) | (indirect_ ? 2u : 0u) | (upscale_ ? 4u : 0u),
        current_jitter_[0], current_jitter_[1], previous_jitter_[0], previous_jitter_[1]};
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    const VkDescriptorSet set = sets_[frame.image_index * 2 + output_slot_];
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &set, 0, nullptr);
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
    dispatch_(frame.command_buffer, (extent_.width + 7) / 8, (extent_.height + 7) / 8, 1);
    recorded_ = true;
    return true;
}

bool TemporalResolve::record_standalone(const FrameRecordingContext& frame)
{
    if (!ready() || !active_ || !frame.command_buffer || frame.image_index * 2 + output_slot_ >= sets_.size())
        return false;
    VkImageMemoryBarrier images[2]{};
    for (uint32_t slot = 0; slot < 2; ++slot)
    {
        auto& image = images[slot];
        image.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        image.srcAccessMask = layouts_initialized_ ? VK_ACCESS_SHADER_READ_BIT : 0;
        image.dstAccessMask = slot == output_slot_ ? VK_ACCESS_SHADER_WRITE_BIT : VK_ACCESS_SHADER_READ_BIT;
        image.oldLayout = layouts_initialized_ ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
        image.newLayout = slot == output_slot_ ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image.srcQueueFamilyIndex = image.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        image.image = targets_[slot].image;
        image.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    barrier_(frame.command_buffer, layouts_initialized_ ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 2, images);
    if (!record(frame)) return false;
    auto& output = images[output_slot_];
    output.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    output.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    output.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    output.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &output);
    return true;
}

bool TemporalResolve::record_callback(const FrameRecordingContext& frame, void* user, std::string& error)
{
    if (!user || !static_cast<TemporalResolve*>(user)->record(frame))
    {
        error = "Temporal resolve has invalid frame inputs";
        return false;
    }
    error.clear();
    return true;
}
bool TemporalResolve::history_valid_callback(void* user)
{ return user && static_cast<TemporalResolve*>(user)->history_valid(); }
bool TemporalResolve::active_callback(void* user)
{ return user && static_cast<TemporalResolve*>(user)->active_; }
uint32_t TemporalResolve::output_slot_callback(void* user)
{ return user ? static_cast<TemporalResolve*>(user)->output_slot() : 0; }

void TemporalResolve::destroy()
{
    if (device_)
    {
        if (pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
        if (pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
        if (layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
        if (set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
        if (sampler_ && destroy_sampler_) destroy_sampler_(device_, sampler_, nullptr);
        for (Target& target : targets_)
        {
            if (target.view && images_.destroy_image_view)
                images_.destroy_image_view(device_, target.view, nullptr);
            if (target.image)
            {
                if (image_pool_) image_pool_->release(target.image);
                else if (images_.destroy_image) images_.destroy_image(device_, target.image, nullptr);
            }
            if (target.memory && images_.free_memory)
                images_.free_memory(device_, target.memory, nullptr);
        }
    }
    targets_ = {};
    sets_.clear();
    device_ = VK_NULL_HANDLE;
    extent_ = {};
    images_ = {};
    image_pool_ = nullptr;
    sharing_ = {};
    sampler_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    output_slot_ = 0;
    active_ = false;
    reprojected_depth_ = false;
    indirect_ = false;
    upscale_ = false;
    current_jitter_ = {}; previous_jitter_ = {};
    layouts_initialized_ = false;
    reset_history();
}
}
