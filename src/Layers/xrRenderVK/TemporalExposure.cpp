#include "TemporalExposure.h"
#include "TemporalExposureShaders.h"
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
{
    return reinterpret_cast<T>(get(device, name));
}
struct Parameters
{
    float middle_gray;
    float blend;
    float low_luminance;
    uint32_t history_valid;
};
static_assert(sizeof(Parameters) == 16);
constexpr VkDeviceSize HistogramBytes = 2u * 64u * sizeof(uint32_t);
}

bool TemporalExposure::initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    const std::vector<VkImageView>& scene_views, const VkPhysicalDeviceMemoryProperties& memory,
    const FrameDispatch& images, VkPipelineCache cache, std::string& error,
    FrameImagePool* image_pool, FrameImageSharing sharing)
{
    destroy();
    sharing_ = std::move(sharing);
    if (!physical || !device || !get || scene_views.empty() || scene_views.size() > 16 ||
        std::any_of(scene_views.begin(), scene_views.end(), [](VkImageView view) { return !view; }) ||
        !images.get_format_properties || !images.create_image || !images.destroy_image ||
        !images.get_image_memory_requirements || !images.allocate_memory || !images.free_memory ||
        !images.bind_image_memory || !images.create_image_view || !images.destroy_image_view)
    {
        error = "temporal exposure requires scene mip views and image procedures";
        return false;
    }
    VkFormatProperties support{};
    images.get_format_properties(physical, VK_FORMAT_R32_SFLOAT, &support);
    if ((support.optimalTilingFeatures &
            (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) !=
        (VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT))
    {
        error = "R32F storage and sampling unavailable for temporal exposure";
        return false;
    }
    device_ = device;
    images_ = images;
    image_pool_ = image_pool;
    auto create_sampler = procedure<PFN_vkCreateSampler>(device, get, "vkCreateSampler");
    destroy_sampler_ = procedure<PFN_vkDestroySampler>(device, get, "vkDestroySampler");
    auto create_buffer = procedure<PFN_vkCreateBuffer>(device, get, "vkCreateBuffer");
    destroy_buffer_ = procedure<PFN_vkDestroyBuffer>(device, get, "vkDestroyBuffer");
    auto get_buffer_requirements = procedure<PFN_vkGetBufferMemoryRequirements>(device, get,
        "vkGetBufferMemoryRequirements");
    auto bind_buffer_memory = procedure<PFN_vkBindBufferMemory>(device, get, "vkBindBufferMemory");
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
    fill_buffer_ = procedure<PFN_vkCmdFillBuffer>(device, get, "vkCmdFillBuffer");
    barrier_ = procedure<PFN_vkCmdPipelineBarrier>(device, get, "vkCmdPipelineBarrier");
    const auto fail = [&](const char* message) {
        error = message;
        destroy();
        return false;
    };
    if (!create_sampler || !destroy_sampler_ || !create_buffer || !destroy_buffer_ ||
        !get_buffer_requirements || !bind_buffer_memory || !fill_buffer_ || !barrier_ ||
        !create_set_layout || !destroy_set_layout_ ||
        !create_pool || !destroy_pool_ || !allocate_sets || !update_sets || !create_layout ||
        !destroy_layout_ || !create_pipeline || !destroy_pipeline_ || !bind_pipeline_ ||
        !bind_sets_ || !push_constants_ || !dispatch_)
        return fail("temporal exposure compute procedures unavailable");

    VkBufferCreateInfo histogram_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    histogram_info.size = HistogramBytes;
    histogram_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    histogram_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (create_buffer(device_, &histogram_info, nullptr, &histogram_) != VK_SUCCESS)
        return fail("temporal exposure histogram buffer creation failed");
    VkMemoryRequirements histogram_requirements{};
    get_buffer_requirements(device_, histogram_, &histogram_requirements);
    uint32_t histogram_type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        if ((histogram_requirements.memoryTypeBits & (1u << i)) &&
            (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
        {
            histogram_type = i;
            break;
        }
    if (histogram_type == UINT32_MAX)
        return fail("temporal exposure histogram needs device-local buffer memory");
    VkMemoryAllocateInfo histogram_allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    histogram_allocation.allocationSize = histogram_requirements.size;
    histogram_allocation.memoryTypeIndex = histogram_type;
    if (images_.allocate_memory(device_, &histogram_allocation, nullptr, &histogram_memory_) != VK_SUCCESS ||
        bind_buffer_memory(device_, histogram_, histogram_memory_, 0) != VK_SUCCESS)
        return fail("temporal exposure histogram allocation failed");
    VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = VK_FORMAT_R32_SFLOAT;
    image.extent = {1, 1, 1};
    image.mipLevels = image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sharing_.apply(image);
    if (image_pool_)
    {
        if (!image_pool_->acquire(image, image_, error))
        {
            error = "temporal exposure image: " + error;
            destroy();
            return false;
        }
    }
    else
    {
        if (images_.create_image(device_, &image, nullptr, &image_) != VK_SUCCESS)
            return fail("temporal exposure image creation failed");
        VkMemoryRequirements requirements{};
        images_.get_image_memory_requirements(device_, image_, &requirements);
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
            if ((requirements.memoryTypeBits & (1u << i)) &&
                (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))
            {
                type = i;
                break;
            }
        if (type == UINT32_MAX) return fail("temporal exposure needs device-local image memory");
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = type;
        if (images_.allocate_memory(device_, &allocation, nullptr, &memory_) != VK_SUCCESS ||
            images_.bind_image_memory(device_, image_, memory_, 0) != VK_SUCCESS)
            return fail("temporal exposure image allocation failed");
    }
    VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    view.image = image_;
    view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view.format = VK_FORMAT_R32_SFLOAT;
    view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (images_.create_image_view(device_, &view, nullptr, &view_) != VK_SUCCESS)
        return fail("temporal exposure image view creation failed");
    VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
    sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler.maxLod = VK_LOD_CLAMP_NONE;
    if (create_sampler(device_, &sampler, nullptr, &sampler_) != VK_SUCCESS)
        return fail("temporal exposure sampler creation failed");
    const VkDescriptorSetLayoutBinding bindings[]{
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
    VkDescriptorSetLayoutCreateInfo set_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set_info.bindingCount = 3;
    set_info.pBindings = bindings;
    if (create_set_layout(device_, &set_info, nullptr, &set_layout_) != VK_SUCCESS)
        return fail("temporal exposure descriptor layout creation failed");
    const VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Parameters)};
    VkPipelineLayoutCreateInfo layout_info{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &set_layout_;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &range;
    if (create_layout(device_, &layout_info, nullptr, &layout_) != VK_SUCCESS)
        return fail("temporal exposure pipeline layout creation failed");
    const uint32_t count = static_cast<uint32_t>(scene_views.size());
    const VkDescriptorPoolSize sizes[]{
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, count},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count}};
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = count;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = sizes;
    if (create_pool(device_, &pool_info, nullptr, &pool_) != VK_SUCCESS)
        return fail("temporal exposure descriptor pool creation failed");
    const ShaderModuleDispatch shader_dispatch{
        procedure<PFN_vkCreateShaderModule>(device, get, "vkCreateShaderModule"),
        procedure<PFN_vkDestroyShaderModule>(device, get, "vkDestroyShaderModule")};
    ShaderModule histogram_shader, adapt_shader;
    if (!histogram_shader.initialize(device_, shader_dispatch,
            temporal_exposure_shaders::Histogram,
            sizeof(temporal_exposure_shaders::Histogram), error) ||
        !adapt_shader.initialize(device_, shader_dispatch,
            temporal_exposure_shaders::Adapt,
            sizeof(temporal_exposure_shaders::Adapt), error))
    {
        destroy();
        return false;
    }
    if (!create_compute_pipeline(device_, cache, layout_, histogram_shader.handle(), "main",
            create_pipeline, destroy_pipeline_, histogram_pipeline_, error) ||
        !create_compute_pipeline(device_, cache, layout_, adapt_shader.handle(), "main",
            create_pipeline, destroy_pipeline_, pipeline_, error))
    {
        destroy();
        return false;
    }
    sets_.resize(count);
    std::vector<VkDescriptorSetLayout> layouts(count, set_layout_);
    VkDescriptorSetAllocateInfo sets_info{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    sets_info.descriptorPool = pool_;
    sets_info.descriptorSetCount = count;
    sets_info.pSetLayouts = layouts.data();
    if (allocate_sets(device_, &sets_info, sets_.data()) != VK_SUCCESS)
        return fail("temporal exposure descriptor allocation failed");
    for (uint32_t i = 0; i < count; ++i)
    {
        const VkDescriptorImageInfo input{sampler_, scene_views[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        const VkDescriptorImageInfo output{VK_NULL_HANDLE, view_, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorBufferInfo histogram_buffer{histogram_, 0, HistogramBytes};
        VkWriteDescriptorSet writes[3]{};
        for (VkWriteDescriptorSet& write : writes)
        {
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = sets_[i];
            write.descriptorCount = 1;
        }
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[0].pImageInfo = &input;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[1].pImageInfo = &output;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &histogram_buffer;
        update_sets(device_, 3, writes, 0, nullptr);
    }
    error.clear();
    return true;
}

void TemporalExposure::set_parameters(float middle_gray, float low_luminance, float delta_seconds,
                                      float adaptation_speed)
{
    middle_gray_ = std::isfinite(middle_gray) ? std::max(middle_gray, 0.f) : 1.f;
    low_luminance_ = std::isfinite(low_luminance) ? std::clamp(low_luminance, 0.0001f, 1.f) : 0.1f;
    const float delta = std::isfinite(delta_seconds) ? std::clamp(delta_seconds, 0.f, 1.f) : 0.f;
    const float speed = std::isfinite(adaptation_speed) ? std::max(adaptation_speed, 0.f) : 0.f;
    blend_ = 1.f - std::exp(-delta * speed);
}

bool TemporalExposure::record(const FrameRecordingContext& frame) const
{
    if (!ready() || !histogram_pipeline_ || !frame.command_buffer || frame.image_index >= sets_.size())
        return false;
    const VkExtent2D extent = frame.extent.width && frame.extent.height
        ? frame.extent : VkExtent2D{16, 16};
    const uint64_t pixel_count = uint64_t(extent.width) * extent.height;
    if (pixel_count > UINT32_MAX) return false;

    VkBufferMemoryBarrier before_clear{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    before_clear.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    before_clear.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    before_clear.srcQueueFamilyIndex = before_clear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    before_clear.buffer = histogram_;
    before_clear.offset = 0;
    before_clear.size = HistogramBytes;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 1, &before_clear, 0, nullptr);
    fill_buffer_(frame.command_buffer, histogram_, 0, HistogramBytes, 0);
    VkBufferMemoryBarrier clear_to_histogram{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    clear_to_histogram.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    clear_to_histogram.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    clear_to_histogram.srcQueueFamilyIndex = clear_to_histogram.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    clear_to_histogram.buffer = histogram_;
    clear_to_histogram.offset = 0;
    clear_to_histogram.size = HistogramBytes;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &clear_to_histogram, 0, nullptr);

    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogram_pipeline_);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1,
        &sets_[frame.image_index], 0, nullptr);
    dispatch_(frame.command_buffer, (extent.width + 15u) / 16u, (extent.height + 7u) / 8u, 1);

    VkBufferMemoryBarrier histogram_to_adapt{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    histogram_to_adapt.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    histogram_to_adapt.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    histogram_to_adapt.srcQueueFamilyIndex = histogram_to_adapt.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    histogram_to_adapt.buffer = histogram_;
    histogram_to_adapt.offset = 0;
    histogram_to_adapt.size = HistogramBytes;
    barrier_(frame.command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &histogram_to_adapt, 0, nullptr);

    const Parameters params{middle_gray_, blend_, low_luminance_, history_valid_ ? 1u : 0u};
    bind_pipeline_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
    bind_sets_(frame.command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout_, 0, 1,
        &sets_[frame.image_index], 0, nullptr);
    push_constants_(frame.command_buffer, layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(params), &params);
    dispatch_(frame.command_buffer, 1, 1, 1);
    return true;
}

bool TemporalExposure::record_callback(const FrameRecordingContext& frame, void* user, std::string& error)
{
    if (!user || !static_cast<TemporalExposure*>(user)->record(frame))
    {
        error = "Temporal exposure has invalid frame inputs";
        return false;
    }
    error.clear();
    return true;
}

bool TemporalExposure::history_valid_callback(void* user)
{
    return user && static_cast<TemporalExposure*>(user)->history_valid_;
}

void TemporalExposure::destroy()
{
    if (device_)
    {
        if (histogram_pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, histogram_pipeline_, nullptr);
        if (pipeline_ && destroy_pipeline_) destroy_pipeline_(device_, pipeline_, nullptr);
        if (pool_ && destroy_pool_) destroy_pool_(device_, pool_, nullptr);
        if (layout_ && destroy_layout_) destroy_layout_(device_, layout_, nullptr);
        if (set_layout_ && destroy_set_layout_) destroy_set_layout_(device_, set_layout_, nullptr);
        if (sampler_ && destroy_sampler_) destroy_sampler_(device_, sampler_, nullptr);
        if (view_ && images_.destroy_image_view) images_.destroy_image_view(device_, view_, nullptr);
        if (image_)
        {
            if (image_pool_) image_pool_->release(image_);
            else if (images_.destroy_image) images_.destroy_image(device_, image_, nullptr);
        }
        if (memory_ && images_.free_memory) images_.free_memory(device_, memory_, nullptr);
        if (histogram_ && destroy_buffer_) destroy_buffer_(device_, histogram_, nullptr);
        if (histogram_memory_ && images_.free_memory)
            images_.free_memory(device_, histogram_memory_, nullptr);
    }
    sets_.clear();
    device_ = VK_NULL_HANDLE;
    image_ = VK_NULL_HANDLE;
    image_pool_ = nullptr;
    sharing_ = {};
    memory_ = VK_NULL_HANDLE;
    histogram_ = VK_NULL_HANDLE;
    histogram_memory_ = VK_NULL_HANDLE;
    view_ = VK_NULL_HANDLE;
    sampler_ = VK_NULL_HANDLE;
    set_layout_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    layout_ = VK_NULL_HANDLE;
    pipeline_ = VK_NULL_HANDLE;
    histogram_pipeline_ = VK_NULL_HANDLE;
    fill_buffer_ = nullptr;
    barrier_ = nullptr;
    reset_history();
}
} // namespace xray::render::vulkan
