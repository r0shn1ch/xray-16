#pragma once

#include "DetailGroundHeightPass.h"
#include "DetailInteractionPass.h"
#include "FrameContext.h"

#include <array>
#include <memory>
#include <vector>

namespace xray::render::vulkan
{
// Owns the terrain and spring-field images independently of the swapchain.
// Record, submit and consume them on the same graphics+compute queue. The
// caller fences each descriptor/upload slot, commits only submitted work and
// waits for all users before destruction. The pool must outlive this object.
class DetailInteractionField
{
public:
    struct Ground
    {
        const BufferResource* cells{};
        const BufferResource* triangles{};
        uint32_t count{}, width{}, height{};
        int32_t offset_x{}, offset_z{};
    };
    struct Output
    {
        VkImage image{};
        VkImageView view{};
        VkImage previous_image{};
        VkImageView previous_view{};
        VkSampler sampler{};
        float origin[2]{}, texel_size{};
        float previous_origin[2]{};
        uint32_t size{};
        bool previous_valid{};
    };

    DetailInteractionField() = default;
    ~DetailInteractionField() { destroy(); }
    DetailInteractionField(const DetailInteractionField&) = delete;
    DetailInteractionField& operator=(const DetailInteractionField&) = delete;

    bool initialize(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
        const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
        const BufferResourceDispatch& buffers, const FrameDispatch& vk, FrameImagePool& pool,
        uint32_t frame_slots, VkPipelineCache cache, std::string& error,
        uint32_t size = 1024, float texel_size = .0625f, float height_texel_size = .5f)
    {
        destroy();
        if (!physical || !device || !get || !pool.initialized() || !vk.get_format_properties ||
            !vk.create_image_view || !vk.destroy_image_view || !vk.cmd_pipeline_barrier ||
            !size || size > 1024 || !frame_slots || !std::isfinite(texel_size) || texel_size <= 0.f ||
            !std::isfinite(height_texel_size) || height_texel_size <= 0.f)
        { error = "invalid detail interaction field configuration"; return false; }
        const double height_size = std::ceil(double(size) * texel_size / height_texel_size) + 2.;
        if (height_size > limits.maxImageDimension2D || height_size > 16384 || size > limits.maxImageDimension2D)
        { error = "detail interaction image dimensions exceed device limits"; return false; }
        for (VkFormat format : {VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT})
        {
            VkFormatProperties properties{};
            vk.get_format_properties(physical, format, &properties);
            const auto required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
            if ((properties.optimalTilingFeatures & required) != required)
            { error = "detail interaction image format lacks storage or sampling support"; return false; }
        }
        device_ = device; vk_ = vk; pool_ = &pool; size_ = size;
        texel_size_ = texel_size; height_texel_size_ = height_texel_size;
        height_size_ = static_cast<uint32_t>(height_size);
        auto create_sampler = reinterpret_cast<PFN_vkCreateSampler>(get(device, "vkCreateSampler"));
        destroy_sampler_ = reinterpret_cast<PFN_vkDestroySampler>(get(device, "vkDestroySampler"));
        clear_ = reinterpret_cast<PFN_vkCmdClearColorImage>(get(device, "vkCmdClearColorImage"));
        const auto fail = [&] { destroy(); return false; };
        if (!create_sampler || !destroy_sampler_ || !clear_)
        { error = "detail interaction image procedures unavailable"; return fail(); }
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = sampler.minFilter = VK_FILTER_NEAREST;
        sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        sampler.addressModeU = sampler.addressModeV = sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (create_sampler(device_, &sampler, nullptr, &sampler_) != VK_SUCCESS)
        { error = "cannot create detail interaction sampler"; return fail(); }
        for (unsigned i = 0; i < images_.size(); ++i)
        {
            VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
            info.imageType = VK_IMAGE_TYPE_2D;
            info.format = i == 2 ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
            const uint32_t dimension = i == 2 ? height_size_ : size_;
            info.extent = {dimension, dimension, 1};
            info.mipLevels = info.arrayLayers = 1;
            info.samples = VK_SAMPLE_COUNT_1_BIT;
            info.tiling = VK_IMAGE_TILING_OPTIMAL;
            info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
            if (!pool_->acquire(info, images_[i], error)) return fail();
            VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            view.image = images_[i]; view.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view.format = info.format; view.subresourceRange = color_range();
            if (vk_.create_image_view(device_, &view, nullptr, &views_[i]) != VK_SUCCESS ||
                !states_.register_image(images_[i], color_range(), ImageUse::Undefined, error))
            { if (error.empty()) error = "cannot create detail interaction image view"; return fail(); }
        }
        entities_.resize(frame_slots);
        for (auto& entity : entities_)
            if (!entity.initialize(device_, 64 * sizeof(DetailInteractionEntity), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                    memory, buffers, error)) return fail();
        ground_ = std::make_unique<DetailGroundHeightPass>();
        interaction_ = std::make_unique<DetailInteractionPass>();
        if (!ground_->initialize(device_, get, limits, frame_slots, cache, error) ||
            !interaction_->initialize(device_, get, limits, frame_slots, cache, error)) return fail();
        return true;
    }

    bool record(const FrameRecordingContext& frame, const Ground& source,
        const std::vector<DetailInteractionEntity>& entities, const float camera[2],
        float delta, float frequency, float damping, Output& output, std::string& error)
    {
        output = {};
        if (!interaction_ || !frame.command_buffer || frame.frame_index >= entities_.size() || pending_ ||
            (frame.command_queue_flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) !=
                (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT) || entities.size() > 64 ||
            (source.count && (!source.cells || !source.triangles)))
        { error = "invalid detail interaction frame"; return false; }
        DetailInteractionParameters params{};
        const float dummy_origin[2]{};
        if (!detail_interaction_parameters(params, camera, previous_origin_.data(), valid_, delta, frequency,
                damping, static_cast<uint32_t>(entities.size()), dummy_origin, height_texel_size_, size_, texel_size_))
        { error = "invalid detail interaction parameters"; return false; }
        DetailGroundHeightParameters ground{};
        ground.size = height_size_; ground.texel_size = height_texel_size_;
        ground.level_size[0] = source.width; ground.level_size[1] = source.height;
        ground.offset[0] = source.offset_x; ground.offset[1] = source.offset_z; ground.cell_count = source.count;
        for (unsigned axis = 0; axis < 2; ++axis)
        {
            ground.origin[axis] = std::floor(params.origin[axis] / height_texel_size_) * height_texel_size_ - height_texel_size_;
            params.height_origin[axis] = ground.origin[axis];
        }
        if (!entities.empty() && !entities_[frame.frame_index].write(0, entities.data(),
                entities.size() * sizeof(DetailInteractionEntity), error)) return false;
        pending_states_ = states_;
        const uint32_t previous = last_, current = last_ ^ 1u;
        const ImageStateDispatch dispatch{vk_.cmd_pipeline_barrier};
        auto transition = [&](unsigned i, ImageUse use) {
            return pending_states_.transition(frame.command_buffer, images_[i], use, dispatch, error,
                frame.command_queue_flags);
        };
        if (source.count)
        {
            if (!transition(2, ImageUse::StorageWrite) ||
                !ground_->record(frame.command_buffer, frame.frame_index, *source.cells, *source.triangles,
                    views_[2], ground, error)) return false;
        }
        else
        {
            if (!transition(2, ImageUse::TransferDestination)) return false;
            VkClearColorValue empty{}; empty.float32[0] = -1e10f;
            const auto range = color_range();
            clear_(frame.command_buffer, images_[2], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &empty, 1, &range);
        }
        if (!transition(2, ImageUse::ComputeSampled) || !transition(previous, ImageUse::StorageRead) ||
            !transition(current, ImageUse::StorageWrite) ||
            !interaction_->record(frame.command_buffer, frame.frame_index, entities_[frame.frame_index],
                views_[previous], views_[2], sampler_, views_[current], params, error) ||
            !transition(previous, ImageUse::ComputeSampled) ||
            !transition(current, ImageUse::ComputeSampled)) return false;
        pending_ = true; pending_last_ = current;
        pending_origin_ = {params.origin[0], params.origin[1]};
        output.image = images_[current]; output.view = views_[current];
        output.previous_image = images_[previous]; output.previous_view = views_[previous]; output.sampler = sampler_;
        output.origin[0] = params.origin[0]; output.origin[1] = params.origin[1];
        output.previous_origin[0] = params.previous_origin[0]; output.previous_origin[1] = params.previous_origin[1];
        output.texel_size = texel_size_; output.size = size_; output.previous_valid = params.previous_valid != 0;
        return true;
    }

    // Commit after successful queue submission, not merely after recording.
    void commit()
    {
        if (!pending_) return;
        states_ = std::move(pending_states_); last_ = pending_last_; previous_origin_ = pending_origin_;
        valid_ = true; pending_ = false;
    }
    void discard() { pending_ = false; }
    void reset_history() { valid_ = false; pending_ = false; }
    void destroy()
    {
        interaction_.reset(); ground_.reset(); entities_.clear();
        if (device_ && destroy_sampler_ && sampler_) destroy_sampler_(device_, sampler_, nullptr);
        for (unsigned i = 0; i < images_.size(); ++i)
        {
            if (device_ && vk_.destroy_image_view && views_[i]) vk_.destroy_image_view(device_, views_[i], nullptr);
            if (pool_ && images_[i]) pool_->release(images_[i]);
        }
        images_ = {}; views_ = {}; sampler_ = {}; device_ = {}; pool_ = nullptr;
        states_ = {}; pending_states_ = {}; previous_origin_ = {}; valid_ = pending_ = false; last_ = 0;
    }
private:
    static VkImageSubresourceRange color_range() { return {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}; }
    VkDevice device_{};
    FrameDispatch vk_{};
    FrameImagePool* pool_{};
    PFN_vkDestroySampler destroy_sampler_{};
    PFN_vkCmdClearColorImage clear_{};
    std::array<VkImage, 3> images_{};
    std::array<VkImageView, 3> views_{};
    VkSampler sampler_{};
    std::vector<BufferResource> entities_;
    std::unique_ptr<DetailGroundHeightPass> ground_;
    std::unique_ptr<DetailInteractionPass> interaction_;
    ImageStateTracker states_, pending_states_;
    uint32_t size_{}, height_size_{}, last_{}, pending_last_{};
    float texel_size_{}, height_texel_size_{};
    std::array<float, 2> previous_origin_{}, pending_origin_{};
    bool valid_{}, pending_{};
};
}
