#include "xrEngine/stdafx.h"
#include "VisibilityPasses.h"
#include "FrameImageGraph.h"
#include "GpuLevel.h"
#include "LevelModels.h"

#include <algorithm>
#include <cstring>

namespace xray::render::vulkan
{
static_assert(sizeof(LevelVertex) == 64 && offsetof(LevelVertex, normal) == 12 && offsetof(LevelVertex, uv) == 24);

bool VisibilityPasses::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
    uint32_t slots, uint32_t batch_count, VkPipelineCache cache, const BindlessTextureTable& materials, const Source& source,
    std::string& error)
{
    slots_ = 0;
    if (!device || !get || !slots || !batch_count || batch_count > 512 / slots || !materials.layout() ||
        !limits.maxComputeWorkGroupCount[0] || !limits.maxComputeWorkGroupCount[1])
    {
        error = "invalid visibility compute limits";
        return false;
    }
    clear_ = reinterpret_cast<PFN_vkCmdClearColorImage>(get(device, "vkCmdClearColorImage"));
    barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    device_ = device;
    create_view_ = reinterpret_cast<PFN_vkCreateImageView>(get(device, "vkCreateImageView"));
    destroy_view_ = reinterpret_cast<PFN_vkDestroyImageView>(get(device, "vkDestroyImageView"));
    if (!clear_ || !barrier_ || !create_view_ || !destroy_view_)
    {
        error = "visibility image commands unavailable";
        return false;
    }
    const auto load = [&](StorageComputePass& pass, const char* path, uint32_t set_count, uint32_t push_size,
                          const std::vector<StorageComputePass::Binding>& bindings, VkDescriptorSetLayout external) {
        std::vector<uint8_t> bytes;
        if (!source(path, bytes, error))
            return false;
        if (bytes.size() < 20 || bytes.size() % sizeof(uint32_t))
        {
            error = std::string("invalid visibility shader: ") + path;
            return false;
        }
        std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
        std::memcpy(words.data(), bytes.data(), bytes.size());
        return pass.initialize(device, get, set_count, bindings, push_size, limits.maxStorageBufferRange,
            limits.maxUniformBufferRange, cache, words.data(), bytes.size(), error, external);
    };
    constexpr auto buffer = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    constexpr auto sampled = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    constexpr auto image = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    const uint32_t batch_slots = slots * batch_count;
    if (!load(raster_, "vk/visibility_software_raster.cs.spv", batch_slots * 2,
            sizeof(visibility::SoftwareRasterParameters), {{0, image}, {1, buffer}, {2, buffer},
                {3, buffer}, {4, image}, {5, buffer}, {6, VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER}}, VK_NULL_HANDLE) ||
        !load(resolve_, "vk/visibility_material_resolve.cs.spv", batch_slots, sizeof(visibility::MaterialResolveParameters),
            {{0, sampled}, {1, buffer}, {2, buffer}, {3, buffer}, {5, image}, {6, image}}, materials.layout()))
    {
        raster_.destroy();
        resolve_.destroy();
        return false;
    }
    max_dispatch_ = {std::min(limits.maxComputeWorkGroupCount[0], 65535u),
        std::min(limits.maxComputeWorkGroupCount[1], 65535u)};
    slots_ = slots;
    batch_count_ = batch_count;
    material_layout_ = materials.layout();
    return true;
}

bool VisibilityPasses::record(const FrameRecordingContext& frame, const GpuLevel& level, const Targets& targets,
    const visibility::Camera& camera, const BindlessTextureTable& materials,
    const BufferResource& draw_mask, VkBufferView draw_mask_view,
    FrameImagePool& pool, framegraph::ResourceRetirement& retirement,
    std::string& error) const
{
    const auto& batches = level.visibility_draw_ranges();
    const auto entries = level.visibility_draw_entries();
    const auto commands = level.visibility_camera_commands(frame.frame_index);
    if (frame.frame_index >= slots_ || frame.frame_index >= 31 || batches.size() != batch_count_ || !entries.buffer || !commands.buffer ||
        !targets.extent.width || !targets.extent.height || !targets.id_sampler || materials.layout() != material_layout_)
    {
        error = "invalid visibility frame inputs";
        return false;
    }
    std::vector<Geometry> geometry(batches.size());
    for (size_t i = 0; i < batches.size(); ++i)
    {
        auto& buffers = geometry[i];
        buffers.entries = entries;
        if (!level.visibility_geometry(batches[i].geometry_batch, buffers.vertices, buffers.indices))
        {
            error = "visibility geometry batch unavailable";
            return false;
        }
    }

    FrameImageGraph graph;
    ImageStateTracker states;
    using GraphBuffer = FrameImageGraph::BufferResource;
    GraphBuffer draw_entries, draw_commands, mask;
    const auto import_buffer = [&](const VkDescriptorBufferInfo& buffer, BufferUse use, GraphBuffer& resource) {
        return graph.import_buffer(buffer.buffer, buffer.offset, buffer.range, use, resource, error);
    };
    if (!import_buffer(entries, BufferUse::ComputeRead, draw_entries) ||
        !import_buffer(commands, BufferUse::IndirectAndComputeRead, draw_commands) ||
        !import_buffer({draw_mask.handle(), 0, draw_mask.size()}, BufferUse::HostWrite, mask))
        return false;
    std::vector<GraphBuffer> geometry_resources{draw_entries};
    for (const auto& batch : geometry)
        for (const auto& buffer : {batch.vertices, batch.indices})
        {
            GraphBuffer resource;
            if (!import_buffer(buffer, BufferUse::ComputeRead, resource))
                return false;
            if (std::none_of(geometry_resources.begin(), geometry_resources.end(),
                    [resource](GraphBuffer existing) { return existing.index == resource.index; }))
                geometry_resources.push_back(resource);
        }
    const auto read_geometry = [&](FrameImageGraph::Pass pass) {
        for (const auto resource : geometry_resources)
            if (!graph.read_buffer(pass, resource, BufferUse::ComputeRead, error))
                return false;
        return true;
    };
    constexpr VkImageSubresourceRange full_color{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    FrameImageGraph::Resource ids, depth, base, normals;
    VkImageCreateInfo id_info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    id_info.imageType = VK_IMAGE_TYPE_2D;
    id_info.format = VK_FORMAT_R32_UINT;
    id_info.extent = {targets.extent.width, targets.extent.height, 1};
    id_info.mipLevels = id_info.arrayLayers = 1;
    id_info.samples = VK_SAMPLE_COUNT_1_BIT;
    id_info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const auto import = [&](const Target& target, FrameImageGraph::Resource& resource) {
        return target.view && states.register_image(target.image, full_color, ImageUse::Undefined, error) &&
            graph.import_image(target.image, full_color, resource, error);
    };
    if (!graph.declare_image(id_info, VK_IMAGE_ASPECT_COLOR_BIT, ids, error) ||
        !import(targets.depth, depth) || !import(targets.base_color, base) ||
        !import(targets.normals, normals))
        return false;

    const auto clear_pass = graph.add_command_pass("visibility buffer clear", [&](VkCommandBuffer command) {
        const VkClearColorValue zero{};
        VkClearColorValue far_depth{};
        far_depth.uint32[0] = 0x3f800000u;
        clear_(command, graph.image_handle(ids), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &full_color);
        for (const auto target : {targets.base_color, targets.normals})
            clear_(command, target.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &zero, 1, &full_color);
        clear_(command, targets.depth.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &far_depth, 1, &full_color);
    });
    const auto clear_resource = [&](FrameImageGraph::Resource& resource) {
        return graph.write(clear_pass, resource, ImageUse::TransferDestination, resource, error);
    };
    if (!clear_resource(ids) || !clear_resource(depth) || !clear_resource(base) || !clear_resource(normals))
        return false;

    for (const auto phase : {RasterPhase::Depth, RasterPhase::Id})
    {
        const auto raster_pass = graph.add_checked_command_pass(
            phase == RasterPhase::Depth ? "visibility raster depth" : "visibility raster IDs",
            [&, phase](VkCommandBuffer command, std::string& reason) {
                for (uint32_t batch = 0; batch < batches.size(); ++batch)
                {
                    const auto& range = batches[batch];
                    const visibility::SoftwareRasterParameters params{camera, range.first_draw,
                        range.draw_count, 0, range.geometry_batch};
                    if (!rasterize(command, frame.frame_index, batch, phase, geometry[batch], commands, draw_mask_view,
                            params, graph.image_view(ids), targets.depth.view, reason))
                        return false;
                }
                return true;
            });
        if (!read_geometry(raster_pass) ||
            !graph.read_buffer(raster_pass, draw_commands, BufferUse::ComputeRead, error) ||
            !graph.read_buffer(raster_pass, mask, BufferUse::ComputeRead, error) ||
            !graph.read_write(raster_pass, ids, ids, error) || !graph.read_write(raster_pass, depth, depth, error))
            return false;
    }

    const auto resolve_pass = graph.add_checked_command_pass("visibility material resolve",
        [&](VkCommandBuffer command, std::string& reason) {
            const VkDescriptorImageInfo id_texture{targets.id_sampler, graph.image_view(ids),
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            for (uint32_t batch = 0; batch < batches.size(); ++batch)
            {
                const auto& range = batches[batch];
                const visibility::MaterialResolveParameters params{camera, range.first_draw, range.draw_count,
                    static_cast<uint32_t>(level.visibility_textures().size()), range.geometry_batch};
                if (!resolve(command, frame.frame_index, batch, geometry[batch], id_texture, materials,
                        targets.base_color.view, targets.normals.view, targets.extent, params, reason))
                    return false;
            }
            return true;
        });
    if (!read_geometry(resolve_pass) || !graph.read(resolve_pass, ids, ImageUse::ComputeSampled, error) ||
        !graph.read_write(resolve_pass, base, base, error) || !graph.read_write(resolve_pass, normals, normals, error))
        return false;
    const auto export_pass = graph.add_pass("visibility outputs ready", [] {});
    for (const auto resource : {depth, base, normals})
        if (!graph.read(export_pass, resource, ImageUse::SampledGraphicsCompute, error))
            return false;
    if (!graph.materialize_color_2d_images(pool, device_, create_view_, destroy_view_, error))
        return false;
    const bool recorded = graph.execute(frame.command_buffer, states, {barrier_}, error, frame.command_queue_flags,
        frame.semaphore_handoff, FrameSubmissionGraph::Queue::Graphics);
    // A failed pass can still leave commands referencing the ID image.
    graph.retire_allocations(retirement, 1u << frame.frame_index);
    return recorded;
}

bool VisibilityPasses::rasterize(VkCommandBuffer command, uint32_t slot, uint32_t batch, RasterPhase phase,
    const Geometry& geometry,
    const VkDescriptorBufferInfo& draw_commands, VkBufferView draw_mask,
    visibility::SoftwareRasterParameters parameters,
    VkImageView ids, VkImageView depth, std::string& error) const
{
    const auto capacity = std::min<VkDeviceSize>({draw_commands.range / sizeof(VkDrawIndexedIndirectCommand),
        geometry.entries.range / sizeof(visibility::DrawEntry), visibility::EntryCountLimit});
    if (slot >= slots_ || batch >= batch_count_ || parameters.first_draw > capacity ||
        parameters.draw_count > capacity - parameters.first_draw)
    {
        error = "invalid visibility draw range";
        return false;
    }
    if (!parameters.draw_count)
    {
        error.clear();
        return true;
    }
    const uint32_t work_groups = parameters.draw_count;
    const uint32_t groups_x = std::min(work_groups, max_dispatch_.width);
    const uint32_t groups_y = (work_groups - 1) / groups_x + 1;
    if (groups_y > max_dispatch_.height)
    {
        error = "visibility raster dispatch exceeds device limits";
        return false;
    }
    const std::vector<VkDescriptorBufferInfo> buffers{geometry.entries, geometry.indices, geometry.vertices, draw_commands};
    const std::vector<VkDescriptorImageInfo> images{{VK_NULL_HANDLE, ids, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, depth, VK_IMAGE_LAYOUT_GENERAL}};
    parameters.write_ids = phase == RasterPhase::Id ? 1u : 0u;
    const uint32_t descriptor_index = (slot * batch_count_ + batch) * 2 + parameters.write_ids;
    return raster_.record_mixed(command, descriptor_index, buffers, images, &parameters, groups_x, groups_y, error,
        VK_NULL_HANDLE, draw_mask);
}

bool VisibilityPasses::resolve(VkCommandBuffer command, uint32_t slot, uint32_t batch, const Geometry& geometry,
    const VkDescriptorImageInfo& ids, const BindlessTextureTable& materials, VkImageView base_color,
    VkImageView normals, VkExtent2D extent, visibility::MaterialResolveParameters parameters, std::string& error) const
{
    if (!extent.width || !extent.height)
    {
        error = "invalid visibility resolve extent";
        return false;
    }
    const uint32_t groups_x = (extent.width - 1) / 8 + 1;
    const uint32_t groups_y = (extent.height - 1) / 8 + 1;
    if (slot >= slots_ || batch >= batch_count_ || groups_x > max_dispatch_.width || groups_y > max_dispatch_.height)
    {
        error = "visibility resolve dispatch exceeds device limits";
        return false;
    }
    if (materials.layout() != material_layout_ || !materials.set() || parameters.material_count > materials.capacity())
    {
        error = "invalid visibility material table";
        return false;
    }
    return resolve_.record_mixed(command, slot * batch_count_ + batch, {geometry.entries, geometry.indices, geometry.vertices},
        {ids, {VK_NULL_HANDLE, base_color, VK_IMAGE_LAYOUT_GENERAL},
            {VK_NULL_HANDLE, normals, VK_IMAGE_LAYOUT_GENERAL}}, &parameters, groups_x, groups_y, error, materials.set());
}
}
