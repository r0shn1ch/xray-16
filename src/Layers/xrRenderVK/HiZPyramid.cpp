#include "HiZPyramid.h"
#include "HiZShaders.h"
#include "ComputePipeline.h"
#include "FrameImageGraph.h"
#include "ShaderModule.h"
#include <algorithm>

namespace xray::render::vulkan
{
namespace
{
template <typename T> T procedure(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{
    return reinterpret_cast<T>(get(device, name));
}
uint32_t half(uint32_t size) { return std::max(1u, size / 2); }
}

bool HiZPyramid::initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    VkExtent2D scene_extent, const std::vector<VkImageView>& depth_views,
    const std::vector<VkImage>& depth_images,
    const std::vector<VkImageView>& motion_views, const std::vector<VkImage>& motion_images,
    const VkPhysicalDeviceMemoryProperties& memory, const FrameDispatch& images,
    VkPipelineCache cache, std::string& error, FrameImagePool* image_pool,
    const FrameImageSharing* sharing)
{
    destroy();
    if (!physical || !device || !get || !scene_extent.width || !scene_extent.height ||
        depth_views.empty() || depth_views.size() > 16 || depth_images.size() != depth_views.size() ||
        motion_views.size() != depth_views.size() || motion_images.size() != depth_views.size() ||
        std::any_of(depth_images.begin(), depth_images.end(), [](VkImage image) { return !image; }) ||
        std::any_of(depth_views.begin(), depth_views.end(), [](VkImageView view) { return !view; }) ||
        std::any_of(motion_images.begin(), motion_images.end(), [](VkImage image) { return !image; }) ||
        std::any_of(motion_views.begin(), motion_views.end(), [](VkImageView view) { return !view; }) ||
        !images.get_format_properties || !images.create_image || !images.destroy_image ||
        !images.get_image_memory_requirements || !images.allocate_memory || !images.free_memory ||
        !images.bind_image_memory || !images.create_image_view || !images.destroy_image_view ||
        !images.cmd_pipeline_barrier)
    {
        error = "Hi-Z requires depth views and Vulkan image procedures";
        return false;
    }
    VkFormatProperties support{};
    images.get_format_properties(physical, VK_FORMAT_R32_SFLOAT, &support);
    if ((support.optimalTilingFeatures &
            (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
        (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
    {
        error = "R32F storage and sampling are unavailable for Hi-Z";
        return false;
    }
    device_ = device;
    depth_images_ = depth_images;
    motion_images_ = motion_images;
    images_ = images;
    image_pool_ = image_pool;
    sharing_ = sharing ? *sharing : FrameImageSharing{};
    memory_ = memory;
    extent_ = {half(scene_extent.width), half(scene_extent.height)};
    for (VkExtent2D size = extent_;; size = {half(size.width), half(size.height)})
    {
        ++levels_;
        if (size.width == 1 && size.height == 1) break;
    }
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
    const auto fail = [&](const char* message) {
        error = message;
        destroy();
        return false;
    };
    if (!create_sampler || !destroy_sampler_ || !create_set_layout || !destroy_set_layout_ ||
        !create_pool || !destroy_pool_ || !allocate_sets || !update_sets || !create_layout ||
        !destroy_layout_ || !create_pipeline || !destroy_pipeline_ || !bind_pipeline_ ||
        !bind_sets_ || !push_constants_ || !dispatch_)
        return fail("Hi-Z compute procedures unavailable");
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (create_sampler(device_, &sampler, nullptr, &sampler_) != VK_SUCCESS)
        return fail("Hi-Z sampler creation failed");
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 3;
    set_info.pBindings = bindings;
    if (create_set_layout(device_, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("Hi-Z descriptor layout creation failed");
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout_;
    const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;
    if (create_layout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("Hi-Z pipeline layout creation failed");
    const uint32_t set_count = static_cast<uint32_t>(depth_views.size()) * levels_;
    const VkDescriptorPoolSize sizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, set_count * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, set_count}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = set_count;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = sizes;
    if (create_pool(device_, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("Hi-Z descriptor pool creation failed");
    ShaderModule shader;
    if (!shader.initialize(device_, {procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            hiz_shaders::Build, sizeof(hiz_shaders::Build), error))
    {
        destroy();
        return false;
    }
    if (!create_compute_pipeline(device_, cache, layout_, shader.handle(), "main", create_pipeline,
            destroy_pipeline_, pipeline_, error))
    {
        destroy();
        return false;
    }
    targets_.resize(depth_views.size());
    for (size_t i = 0; i < depth_views.size(); ++i)
    {
        if (!create_target(targets_[i], error))
        {
            destroy();
            return false;
        }
        targets_[i].sets.resize(levels_);
        std::vector<VkDescriptorSetLayout> layouts(levels_, set_layout_);
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool_;
        allocation.descriptorSetCount = levels_;
        allocation.pSetLayouts = layouts.data();
        if (allocate_sets(device_, &allocation, targets_[i].sets.data()) != VK_SUCCESS)
            return fail("Hi-Z descriptor allocation failed");
        for (uint32_t mip = 0; mip < levels_; ++mip)
        {
            const VkDescriptorImageInfo input{sampler_, mip ? targets_[i].mips[mip - 1] : depth_views[i],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            const VkDescriptorImageInfo output{VK_NULL_HANDLE, targets_[i].mips[mip], VK_IMAGE_LAYOUT_GENERAL};
            const VkDescriptorImageInfo dynamic_mask{sampler_, motion_views[i],
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            VkWriteDescriptorSet writes[3]{};
            for (VkWriteDescriptorSet& write : writes)
            {
                write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                write.dstSet = targets_[i].sets[mip];
                write.descriptorCount = 1;
            }
            writes[0].dstBinding = 0;
            writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[0].pImageInfo = &input;
            writes[1].dstBinding = 1;
            writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[1].pImageInfo = &output;
            writes[2].dstBinding = 2;
            writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[2].pImageInfo = &dynamic_mask;
            update_sets(device_, 3, writes, 0, nullptr);
        }
    }
    error.clear();
    return true;
}

bool HiZPyramid::create_target(Target& target, std::string& error)
{
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = VK_FORMAT_R32_SFLOAT;
    image.extent = {extent_.width, extent_.height, 1};
    image.mipLevels = levels_;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sharing_.apply(image);
    if (image_pool_)
    {
        if (!image_pool_->acquire(image, target.image, error))
        {
            error = "Hi-Z image: " + error;
            return false;
        }
    }
    else if (images_.create_image(device_, &image, nullptr, &target.image) != VK_SUCCESS)
    {
        error = "Hi-Z image creation failed";
        return false;
    }
    if (!image_pool_)
    {
        VkMemoryRequirements requirements{};
        images_.get_image_memory_requirements(device_, target.image, &requirements);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory_.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory_.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            {
                type = i;
                break;
            }
        if (type == UINT32_MAX)
        {
            error = "Hi-Z needs device-local image memory";
            return false;
        }
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (images_.allocate_memory(device_, &allocation, nullptr, &target.memory) != VK_SUCCESS ||
            images_.bind_image_memory(device_, target.image, target.memory, 0) != VK_SUCCESS)
        {
            error = "Hi-Z image memory allocation failed";
            return false;
        }
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = target.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R32_SFLOAT;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = levels_;
    view.subresourceRange.layerCount = 1;
    if (images_.create_image_view(device_, &view, nullptr, &target.view) != VK_SUCCESS)
    {
        error = "Hi-Z full image view creation failed";
        return false;
    }
    view.subresourceRange.levelCount = 1;
    for (uint32_t mip = 0; mip < levels_; ++mip)
    {
        view.subresourceRange.baseMipLevel = mip;
        VkImageView single{};
        if (images_.create_image_view(device_, &view, nullptr, &single) != VK_SUCCESS)
        {
            error = "Hi-Z mip view creation failed";
            return false;
        }
        target.mips.push_back(single);
    }
    return true;
}

bool HiZPyramid::record(const FrameRecordingContext& frame, bool reject_dynamic_occluders)
{
    pending_image_index_ = UINT32_MAX;
    if (!ready() || !frame.command_buffer || frame.image_index >= targets_.size()) return false;
    Target& target = targets_[frame.image_index];
    const bool use_generated_compute = frame.compute_command_buffer && frame.submission_graph &&
        frame.compute_submission_pass != FrameSubmissionGraph::InvalidPass &&
        frame.graphics_prepass_pass != FrameSubmissionGraph::InvalidPass &&
        frame.graphics_continuation_pass != FrameSubmissionGraph::InvalidPass &&
        frame.request_submission_command_buffer;
    const FrameSubmissionGraph::Queue compute_queue = use_generated_compute ?
        FrameSubmissionGraph::Queue::Compute : frame.command_queue;
    const ImageUse final_output_use = use_generated_compute ? ImageUse::ComputeSampled :
        ImageUse::SampledGraphicsCompute;
    std::string error;
    ImageStateTracker states;
    const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (!states.register_image(depth_images_[frame.image_index], depth_range,
            ImageUse::SampledGraphicsCompute, error))
        return false;
    const VkImageSubresourceRange motion_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (reject_dynamic_occluders &&
        !states.register_image(motion_images_[frame.image_index], motion_range,
            ImageUse::SampledGraphicsCompute, error))
        return false;
    const VkImageSubresourceRange whole{VK_IMAGE_ASPECT_COLOR_BIT, 0, levels_, 0, 1};
    if (!states.register_image(target.image, whole,
            target.initialized ? final_output_use : ImageUse::Undefined, error))
        return false;
    FrameImageGraph graph(compute_queue);
    FrameImageGraph::Resource depth;
    if (!graph.import_image(depth_images_[frame.image_index], depth_range, depth, error,
            FrameSubmissionGraph::Queue::Graphics, 0, sharing_.mode,
            sharing_.queue_families, ImageUse::SampledGraphicsCompute,
            frame.graphics_prepass_pass))
        return false;
    FrameImageGraph::Resource motion;
    if (!graph.import_image(motion_images_[frame.image_index], motion_range, motion, error,
            FrameSubmissionGraph::Queue::Graphics, 0,
            sharing_.mode, sharing_.queue_families,
            ImageUse::SampledGraphicsCompute, frame.geometry_motion_submission_pass))
        return false;
    const uint64_t previous_submission = target.initialized ? frame.previous_graphics_submission : 0;
    // FrameContext waits the acquired swapchain image's fence before this
    // target is reused. A zero cross-frame token is therefore valid here: the
    // fence has already completed the previous graphics continuation, which
    // itself waits for this target's compute producer.
    std::vector<FrameImageGraph::Resource> mips;
    mips.reserve(levels_);
    for (uint32_t mip = 0; mip < levels_; ++mip)
    {
        const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, mip, 1, 0, 1};
        FrameImageGraph::Resource resource;
        if (!graph.import_image(target.image, range, resource, error,
                target.initialized ? FrameSubmissionGraph::Queue::Graphics : compute_queue,
                previous_submission, sharing_.mode, sharing_.queue_families,
                target.initialized ? final_output_use : ImageUse::Undefined))
            return false;
        mips.push_back(resource);
    }
    VkExtent2D size = extent_;
    for (uint32_t mip = 0; mip < levels_; ++mip)
    {
        const auto build = graph.add_command_pass("Hi-Z reduction", compute_queue, [&, mip, size](VkCommandBuffer command) {
            bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
            bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1,
                &target.sets[mip], 0, nullptr);
            const uint32_t mask_dynamic = reject_dynamic_occluders && mip == 0 ? 1u : 0u;
            push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                sizeof(mask_dynamic), &mask_dynamic);
            dispatch_(command, (size.width + 7) / 8, (size.height + 7) / 8, 1);
        });
        if (!graph.read(build, mip ? mips[mip - 1] : depth, ImageUse::ComputeSampled, error))
            return false;
        if (reject_dynamic_occluders && mip == 0 &&
            !graph.read(build, motion, ImageUse::ComputeSampled, error))
            return false;
        if (!graph.write(build, mips[mip], ImageUse::StorageWrite, mips[mip], error))
            return false;
        size = {half(size.width), half(size.height)};
    }
    const auto finish = graph.add_pass("Hi-Z ready for sampling", compute_queue, [] {});
    for (const auto mip : mips)
        if (!graph.read(finish, mip, final_output_use, error))
            return false;
    if (use_generated_compute)
    {
        std::vector<FrameImageGraph::QueueGroup> groups;
        std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
        const auto queue_family = [&frame](FrameSubmissionGraph::Queue queue) {
            if (queue == FrameSubmissionGraph::Queue::Compute) return frame.compute_queue_family;
            if (queue == FrameSubmissionGraph::Queue::Graphics) return frame.graphics_queue_family;
            return frame.copy_queue_family;
        };
        if (!graph.generate_submission_groups(*frame.submission_graph, queue_family,
                groups, submission_for_pass, error))
            return false;
        FrameImageGraph::ExecutionState execution;
        if (!graph.begin_execution(execution, error))
            return false;
        for (const FrameImageGraph::QueueGroup& group : groups)
        {
            VkCommandBuffer command = VK_NULL_HANDLE;
            if (group.submission == FrameSubmissionGraph::InvalidPass ||
                !frame.request_submission_command_buffer(group.submission, group.queue,
                    command, error))
                return false;
            if (!graph.execute_group(group, command, states, execution,
                    {images_.cmd_pipeline_barrier}, error, frame.compute_queue_flags,
                    frame.semaphore_handoff))
                return false;
        }
    }
    else if (!graph.execute(frame.command_buffer, states, {images_.cmd_pipeline_barrier}, error,
            frame.command_queue_flags, frame.semaphore_handoff, frame.command_queue))
        return false;
    pending_image_index_ = frame.image_index;
    pending_dynamic_safe_ = reject_dynamic_occluders;
    return true;
}

void HiZPyramid::commit_history()
{
    if (pending_image_index_ >= targets_.size())
        return;
    targets_[pending_image_index_].initialized = true;
    last_image_index_ = pending_image_index_;
    pending_image_index_ = UINT32_MAX;
    history_valid_ = true;
    last_dynamic_safe_ = pending_dynamic_safe_;
}

VkImageView HiZPyramid::view(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].view : VK_NULL_HANDLE;
}

VkImage HiZPyramid::image(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].image : VK_NULL_HANDLE;
}

void HiZPyramid::destroy()
{
    if (device_)
    {
        for (Target& target : targets_)
        {
            if (images_.destroy_image_view)
            {
                for (VkImageView mip : target.mips)
                    if (mip) images_.destroy_image_view(device_, mip, nullptr);
                if (target.view) images_.destroy_image_view(device_, target.view, nullptr);
            }
            if (target.image)
            {
                if (image_pool_) image_pool_->release(target.image);
                else if (images_.destroy_image) images_.destroy_image(device_, target.image, nullptr);
            }
            if (target.memory && images_.free_memory) images_.free_memory(device_, target.memory, nullptr);
        }
        if (pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
        if (pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
        if (layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
        if (set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
        if (sampler_ && destroy_sampler_) destroy_sampler_(device_, sampler_, nullptr);
    }
    targets_.clear();
    depth_images_.clear();
    motion_images_.clear();
    reset_history();
    device_ = VK_NULL_HANDLE;
    extent_ = {};
    levels_ = 0;
    sampler_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    image_pool_ = nullptr;
    sharing_ = {};
}
} // namespace xray::render::vulkan
