#include "MotionVectors.h"
#include "MotionVectorShaders.h"
#include "ComputePipeline.h"
#include "FrameImageGraph.h"
#include "ShaderModule.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace xray::render::vulkan
{
namespace
{
template <typename T> T procedure(VkDevice device, PFN_vkGetDeviceProcAddr get, const char* name)
{
    return reinterpret_cast<T>(get(device, name));
}

std::array<float, 4> transform(const std::array<float, 16>& matrix,
    const std::array<float, 4>& point)
{
    std::array<float, 4> result{};
    for (unsigned row = 0; row < 4; ++row)
        for (unsigned column = 0; column < 4; ++column)
            result[row] += matrix[column * 4 + row] * point[column];
    return result;
}

bool camera_cut(const std::array<float, 16>& previous,
    const std::array<float, 16>& current_inverse)
{
    // Compare the previous screen position of several world points visible now.
    // A half-screen jump makes temporal reprojection more harmful than useful.
    constexpr std::array<std::array<float, 2>, 5> samples{{
        {0.f, 0.f}, {-.7f, -.7f}, {.7f, -.7f}, {-.7f, .7f}, {.7f, .7f}}};
    for (const auto& xy : samples)
    {
        auto world = transform(current_inverse, {xy[0], xy[1], .9f, 1.f});
        if (!std::isfinite(world[3]) || std::abs(world[3]) < 1e-6f) return true;
        for (float& component : world) component /= world[3];
        const auto projected = transform(previous, world);
        if (!std::isfinite(projected[3]) || projected[3] <= 1e-6f) return true;
        const float dx = projected[0] / projected[3] - xy[0];
        const float dy = projected[1] / projected[3] - xy[1];
        const float previous_depth = projected[2] / projected[3];
        if (!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(previous_depth) ||
            previous_depth < 0.f || previous_depth > 1.f ||
            std::abs(dx) > 1.f || std::abs(dy) > 1.f)
            return true;
    }
    return false;
}
}

bool MotionVectors::initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    VkExtent2D extent, const std::vector<VkImageView>& depth_views,
    const std::vector<VkImage>& depth_images,
    const VkPhysicalDeviceMemoryProperties& memory, const FrameDispatch& images,
    VkPipelineCache cache, bool storage_extended_formats, std::string& error,
    FrameImagePool* image_pool, bool prefer_reprojected_depth,
    const FrameImageSharing* sharing)
{
    destroy();
    if (!physical || !device || !get || !extent.width || !extent.height || depth_views.empty() ||
        depth_views.size() > 16 || depth_images.size() != depth_views.size() ||
        std::any_of(depth_images.begin(), depth_images.end(), [](VkImage image) { return !image; }) ||
        std::any_of(depth_views.begin(), depth_views.end(), [](VkImageView view) { return !view; }) ||
        !images.get_format_properties || !images.create_image || !images.destroy_image ||
        !images.get_image_memory_requirements || !images.allocate_memory || !images.free_memory ||
        !images.bind_image_memory || !images.create_image_view || !images.destroy_image_view ||
        !images.cmd_pipeline_barrier)
    {
        error = "motion vectors require depth views and Vulkan image procedures";
        return false;
    }
    const VkFormatFeatureFlags required =
        VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (storage_extended_formats && !prefer_reprojected_depth)
    {
        VkFormatProperties rg_support{};
        images.get_format_properties(physical, VK_FORMAT_R16G16_SFLOAT, &rg_support);
        if ((rg_support.optimalTilingFeatures & required) == required)
            format_ = VK_FORMAT_R16G16_SFLOAT;
    }
    if (format_ == VK_FORMAT_UNDEFINED)
    {
        VkFormatProperties rgba_support{};
        images.get_format_properties(physical, VK_FORMAT_R16G16B16A16_SFLOAT, &rgba_support);
        if ((rgba_support.optimalTilingFeatures & required) == required)
            format_ = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    if (format_ == VK_FORMAT_UNDEFINED && storage_extended_formats && prefer_reprojected_depth)
    {
        VkFormatProperties rg_support{};
        images.get_format_properties(physical, VK_FORMAT_R16G16_SFLOAT, &rg_support);
        if ((rg_support.optimalTilingFeatures & required) == required)
            format_ = VK_FORMAT_R16G16_SFLOAT;
    }
    if (format_ == VK_FORMAT_UNDEFINED)
    {
        error = "RG16F/RGBA16F storage and sampling are unavailable for motion vectors";
        return false;
    }
    if (format_ == VK_FORMAT_R16G16B16A16_SFLOAT)
    {
        VkFormatProperties rgba_support{};
        images.get_format_properties(physical, format_, &rgba_support);
        color_attachment_ready_ =
            (rgba_support.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
    }
    device_ = device;
    depth_images_ = depth_images;
    extent_ = extent;
    memory_ = memory;
    images_ = images;
    image_pool_ = image_pool;
    sharing_ = sharing ? *sharing : FrameImageSharing{};
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
        return fail("motion-vector compute procedures unavailable");
    // The pipeline layout uses the Vulkan 1.0 minimum of 128 push-constant bytes.
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (create_sampler(device_, &sampler, nullptr, &sampler_) != VK_SUCCESS)
        return fail("motion-vector sampler creation failed");
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 2;
    set_info.pBindings = bindings;
    if (create_set_layout(device_, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("motion-vector descriptor layout creation failed");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    if (create_layout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("motion-vector pipeline layout creation failed");
    const uint32_t count = static_cast<uint32_t>(depth_views.size());
    const VkDescriptorPoolSize sizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, count}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = count;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = sizes;
    if (create_pool(device_, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("motion-vector descriptor pool creation failed");
    ShaderModule shader;
    const uint32_t* code = format_ == VK_FORMAT_R16G16_SFLOAT
        ? motion_vector_shaders::FillRg : motion_vector_shaders::FillRgba;
    const size_t code_size = format_ == VK_FORMAT_R16G16_SFLOAT
        ? sizeof(motion_vector_shaders::FillRg) : sizeof(motion_vector_shaders::FillRgba);
    if (!shader.initialize(device_, {procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
            procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")},
            code, code_size, error))
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
    targets_.resize(count);
    for (uint32_t i = 0; i < count; ++i)
    {
        Target& target = targets_[i];
        if (!create_target(target, error))
        {
            destroy();
            return false;
        }
        VkDescriptorSetAllocateInfo allocation{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocation.descriptorPool = pool_;
        allocation.descriptorSetCount = 1;
        allocation.pSetLayouts = &set_layout_;
        if (allocate_sets(device_, &allocation, &target.set) != VK_SUCCESS)
            return fail("motion-vector descriptor allocation failed");
        const VkDescriptorImageInfo input{sampler_, depth_views[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorImageInfo output{VK_NULL_HANDLE, target.view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet writes[2]{};
        for (VkWriteDescriptorSet& write : writes)
        {
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = target.set;
            write.descriptorCount = 1;
        }
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &input;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &output;
        update_sets(device_, 2, writes, 0, nullptr);
    }
    reset_history();
    error.clear();
    return true;
}

bool MotionVectors::create_target(Target& target, std::string& error)
{
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = format_;
    image.extent = {extent_.width, extent_.height, 1};
    image.mipLevels = image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        (color_attachment_ready_ ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0);
    sharing_.apply(image);
    if (image_pool_)
    {
        if (!image_pool_->acquire(image, target.image, error))
        {
            error = "motion-vector image: " + error;
            return false;
        }
    }
    else if (images_.create_image(device_, &image, nullptr, &target.image) != VK_SUCCESS)
    {
        error = "motion-vector image creation failed";
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
            error = "motion vectors need device-local image memory";
            return false;
        }
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (images_.allocate_memory(device_, &allocation, nullptr, &target.memory) != VK_SUCCESS ||
            images_.bind_image_memory(device_, target.image, target.memory, 0) != VK_SUCCESS)
        {
            error = "motion-vector image memory allocation failed";
            return false;
        }
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = target.image;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = format_;
    view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view.subresourceRange.levelCount = view.subresourceRange.layerCount = 1;
    if (images_.create_image_view(device_, &view, nullptr, &target.view) != VK_SUCCESS)
    {
        error = "motion-vector image view creation failed";
        return false;
    }
    return true;
}

void MotionVectors::set_view(const float* view_projection, const float* inverse_view_projection)
{
    const auto finite = [](const float* matrix) {
        return matrix && std::all_of(matrix, matrix + 16, [](float value) { return std::isfinite(value); });
    };
    current_view_valid_ = finite(view_projection) && finite(inverse_view_projection);
    if (!current_view_valid_)
    {
        current_inverse_.fill(0.f);
        reset_history();
        return;
    }
    std::copy_n(view_projection, 16, current_vp_.begin());
    std::copy_n(inverse_view_projection, 16, current_inverse_.begin());
    if (history_valid_ && camera_cut(previous_vp_, current_inverse_))
        reset_history();
}

void MotionVectors::commit_view()
{
    if (!current_view_valid_) return;
    previous_vp_ = current_vp_;
    history_valid_ = true;
}

void MotionVectors::reset_history()
{
    previous_vp_.fill(0.f);
    history_valid_ = false;
}

bool MotionVectors::record(const FrameRecordingContext& frame, std::string* error) const
{
    if (error) error->clear();
    const auto fail = [error](const std::string& message) {
        if (error) *error = message;
        return false;
    };
    if (!ready()) return fail("motion-vector pipeline is not initialized");
    if (!frame.command_buffer) return fail("motion-vector command buffer is null");
    if (frame.image_index >= targets_.size()) return fail("motion-vector image index is out of range");
    const Target& target = targets_[frame.image_index];
    // This pass must precede the object-velocity attachment pass in the same
    // graphics prepass. Scheduling it on a generated compute submission would
    // run it after that prepass and overwrite the per-object motion vectors.
    const FrameSubmissionGraph::Queue compute_queue = frame.command_queue;
    std::array<float, 32> matrices{};
    std::copy(current_inverse_.begin(), current_inverse_.end(), matrices.begin());
    if (history_valid_)
        std::copy(previous_vp_.begin(), previous_vp_.end(), matrices.begin() + 16);
    std::string graph_error;
    ImageStateTracker states;
    const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (!states.register_image(depth_images_[frame.image_index], depth_range,
            ImageUse::SampledGraphicsCompute, graph_error))
        return fail(graph_error);
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (!states.register_image(target.image, range, ImageUse::Undefined, graph_error))
        return fail(graph_error);
    FrameImageGraph graph(compute_queue);
    FrameImageGraph::Resource depth;
    if (!graph.import_image(depth_images_[frame.image_index], depth_range, depth, graph_error,
            FrameSubmissionGraph::Queue::Graphics, 0, sharing_.mode,
            sharing_.queue_families, ImageUse::SampledGraphicsCompute))
        return fail(graph_error);
    FrameImageGraph::Resource output;
    if (!graph.import_image(target.image, range, output, graph_error, compute_queue, 0,
            sharing_.mode, sharing_.queue_families, ImageUse::Undefined))
        return fail(graph_error);
    const auto fill = graph.add_command_pass("camera motion vectors", compute_queue,
        [&](VkCommandBuffer command) {
        bind_pipeline_(command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
        bind_sets_(command, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1, &target.set, 0, nullptr);
        push_constants_(command, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
            sizeof(matrices), matrices.data());
        dispatch_(command, (extent_.width + 7) / 8, (extent_.height + 7) / 8, 1);
    });
    if (!graph.read(fill, depth, ImageUse::ComputeSampled, graph_error)) return fail(graph_error);
    if (!graph.write(fill, output, ImageUse::StorageWrite, output, graph_error))
        return fail(graph_error);
    const auto finish = graph.add_pass("motion vectors ready for sampling", compute_queue, [] {});
    if (!graph.read(finish, output, ImageUse::SampledGraphicsCompute, graph_error))
        return fail(graph_error);
    if (!graph.execute(frame.command_buffer, states, {images_.cmd_pipeline_barrier}, graph_error,
            frame.command_queue_flags, frame.semaphore_handoff, frame.command_queue))
        return fail(graph_error);
    return true;
}

VkImage MotionVectors::image(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].image : VK_NULL_HANDLE;
}

VkImageView MotionVectors::view(uint32_t index) const
{
    return index < targets_.size() ? targets_[index].view : VK_NULL_HANDLE;
}

void MotionVectors::destroy()
{
    if (device_)
    {
        for (Target& target : targets_)
        {
            if (target.view && images_.destroy_image_view) images_.destroy_image_view(device_, target.view, nullptr);
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
    device_ = VK_NULL_HANDLE;
    extent_ = {};
    format_ = VK_FORMAT_UNDEFINED;
    color_attachment_ready_ = false;
    pipeline_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    sampler_ = VK_NULL_HANDLE;
    image_pool_ = nullptr;
    sharing_ = {};
    current_view_valid_ = false;
    reset_history();
}
} // namespace xray::render::vulkan
