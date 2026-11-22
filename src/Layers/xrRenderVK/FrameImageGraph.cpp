#include "FrameImageGraph.h"
#include "FrameBufferPool.h"

#include <algorithm>
#include <array>
#include <exception>
#include <set>
#include <utility>
#include <limits>

namespace xray::render::vulkan
{
namespace
{
VkImageUsageFlags required_image_usage(ImageUse use)
{
    switch (use)
    {
    case ImageUse::TransferSource: return VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    case ImageUse::ColorAttachmentWriteTransferSource:
        return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    case ImageUse::TransferDestination:
    case ImageUse::TransferDestinationGeneral: return VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    case ImageUse::Sampled:
    case ImageUse::ComputeSampled:
    case ImageUse::SampledGraphicsCompute: return VK_IMAGE_USAGE_SAMPLED_BIT;
    case ImageUse::StorageRead:
    case ImageUse::StorageWrite:
    case ImageUse::StorageReadWrite: return VK_IMAGE_USAGE_STORAGE_BIT;
    case ImageUse::ColorAttachment: return VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    case ImageUse::DepthStencilAttachment:
    case ImageUse::DepthStencilProduced:
    case ImageUse::DepthStencilReadOnly: return VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    default: return 0;
    }
}

VkBufferUsageFlags required_buffer_usage(BufferUse use)
{
    switch (use)
    {
    case BufferUse::TransferSource: return VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    case BufferUse::TransferDestination: return VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    case BufferUse::IndirectRead: return VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    case BufferUse::IndirectAndComputeRead: return VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    case BufferUse::ComputeWrite:
    case BufferUse::ComputeReadWrite: return VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    default: return 0;
    }
}

template <typename Allocate>
bool try_allocate_transient(Allocate&& allocate, std::string& error)
{
    error.clear();
    try
    {
        return allocate();
    }
    catch (const std::exception& exception)
    {
        error = std::string("Frame image graph: transient allocation threw: ") + exception.what();
        return false;
    }
}

bool valid_sharing_metadata(VkSharingMode mode, const std::vector<uint32_t>& queue_families)
{
    if (mode == VK_SHARING_MODE_EXCLUSIVE)
        return true;
    if (mode != VK_SHARING_MODE_CONCURRENT || queue_families.size() < 2)
        return false;
    std::set<uint32_t> unique(queue_families.begin(), queue_families.end());
    return unique.size() == queue_families.size() &&
        unique.find(VK_QUEUE_FAMILY_IGNORED) == unique.end();
}

bool image_subresources_overlap(const VkImageSubresourceRange& a,
    const VkImageSubresourceRange& b)
{
    return (a.aspectMask & b.aspectMask) &&
        uint64_t(a.baseMipLevel) < uint64_t(b.baseMipLevel) + b.levelCount &&
        uint64_t(b.baseMipLevel) < uint64_t(a.baseMipLevel) + a.levelCount &&
        uint64_t(a.baseArrayLayer) < uint64_t(b.baseArrayLayer) + b.layerCount &&
        uint64_t(b.baseArrayLayer) < uint64_t(a.baseArrayLayer) + a.layerCount;
}

bool same_image_subresources(const VkImageSubresourceRange& a,
    const VkImageSubresourceRange& b)
{
    return a.aspectMask == b.aspectMask && a.baseMipLevel == b.baseMipLevel &&
        a.levelCount == b.levelCount && a.baseArrayLayer == b.baseArrayLayer &&
        a.layerCount == b.layerCount;
}

struct BufferUseInfo { VkAccessFlags access; VkPipelineStageFlags stage; bool writes; };
BufferUseInfo describe(BufferUse use)
{
    switch (use)
    {
    case BufferUse::Undefined: return {0, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, false};
    case BufferUse::HostWrite: return {VK_ACCESS_HOST_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, true};
    case BufferUse::HostRead: return {VK_ACCESS_HOST_READ_BIT, VK_PIPELINE_STAGE_HOST_BIT, false};
    case BufferUse::TransferSource: return {VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, false};
    case BufferUse::TransferDestination: return {VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, true};
    case BufferUse::ComputeRead: return {VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, false};
    case BufferUse::ComputeWrite: return {VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true};
    case BufferUse::ComputeReadWrite: return {VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, true};
    case BufferUse::VertexRead: return {VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, false};
    case BufferUse::FragmentRead: return {VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, false};
    case BufferUse::IndirectRead: return {VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, false};
    case BufferUse::IndirectAndComputeRead: return {VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, false};
    case BufferUse::AllCommandsReadWrite: return {VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, true};
    }
    return {0, 0, false};
}

bool supported_on_queue(BufferUse use, VkQueueFlags flags)
{
    switch (use)
    {
    case BufferUse::VertexRead:
    case BufferUse::FragmentRead:
    case BufferUse::IndirectRead:
        return (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
    case BufferUse::IndirectAndComputeRead:
        return (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) ==
            (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    case BufferUse::ComputeRead:
    case BufferUse::ComputeWrite:
    case BufferUse::ComputeReadWrite:
        return (flags & VK_QUEUE_COMPUTE_BIT) != 0;
    case BufferUse::TransferSource:
    case BufferUse::TransferDestination:
        return (flags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT)) != 0;
    default:
        return true;
    }
}

}

bool FrameImageGraph::import_buffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
    BufferUse initial_use, BufferResource& resource, std::string& error,
    FrameSubmissionGraph::Queue initial_queue, uint64_t initial_submission,
    VkSharingMode sharing_mode, std::vector<uint32_t> sharing_queue_families,
    FrameSubmissionGraph::Pass initial_producer_pass)
{
    if (initial_submission && initial_producer_pass != FrameSubmissionGraph::InvalidPass)
    {
        error = "Frame image graph: buffer import has both a submission token and a producer pass";
        return false;
    }
    if (initial_producer_pass != FrameSubmissionGraph::InvalidPass &&
        (initial_use == BufferUse::Undefined || initial_use == BufferUse::HostWrite ||
            initial_use == BufferUse::HostRead))
    {
        error = "Frame image graph: buffer producer pass has no queue-owned terminal use";
        return false;
    }
    if (buffer == VK_NULL_HANDLE || !size || size == VK_WHOLE_SIZE ||
        offset > std::numeric_limits<VkDeviceSize>::max() - size ||
        describe(initial_use).stage == 0 ||
        !valid_sharing_metadata(sharing_mode, sharing_queue_families))
    {
        error = "Frame image graph: invalid imported buffer range or state";
        return false;
    }
    for (const Buffer& existing : m_buffers)
    {
        if (existing.buffer != buffer)
            continue;
        if (existing.sharing_mode != sharing_mode ||
            existing.sharing_queue_families != sharing_queue_families)
        {
            error = "Frame image graph: imports of one buffer have different sharing metadata";
            return false;
        }
        const bool overlaps = offset < existing.offset + existing.size &&
            existing.offset < offset + size;
        if (overlaps && (existing.initial_use != initial_use ||
                existing.initial_submission != initial_submission || existing.initial_queue != initial_queue ||
                existing.initial_producer_pass != initial_producer_pass))
        {
            error = "Frame image graph: overlapping buffer imports have different initial producer metadata";
            return false;
        }
        if (existing.offset == offset && existing.size == size)
        {
            resource = {uint32_t(&existing - m_buffers.data()), uint32_t(existing.versions.size() - 1), generation_};
            error.clear();
            return true;
        }
    }
    resource = {uint32_t(m_buffers.size()), 0, generation_};
    m_buffers.push_back({buffer, offset, size, initial_use, initial_queue, initial_submission,
        initial_producer_pass, sharing_mode, std::move(sharing_queue_families), {{}}});
    error.clear();
    return true;
}

bool FrameImageGraph::declare_buffer(const BufferDescription& description,
    BufferResource& resource, std::string& error)
{
    if (!description.size || !description.usage)
    {
        error = "Frame image graph: transient buffer requires size and usage";
        return false;
    }
    Buffer buffer;
    buffer.size = description.size;
    buffer.transient_description = description;
    buffer.initial_queue = default_queue_;
    m_buffers.push_back(std::move(buffer));
    resource = {uint32_t(m_buffers.size() - 1), 0, generation_};
    error.clear();
    return true;
}

bool FrameImageGraph::materialize_buffers(const AllocateBuffer& allocate, std::string& error)
{
    ResourceLifetimes lifetimes;
    if (!compile_resource_lifetimes(lifetimes, error))
        return false;
    std::vector<std::pair<size_t, TransientBuffer>> allocations;
    std::set<VkBuffer> handles;
    for (const auto& buffer : m_buffers)
        if (buffer.buffer) handles.insert(buffer.buffer);
    for (size_t index = 0; index < m_buffers.size(); ++index)
    {
        const Buffer& buffer = m_buffers[index];
        if (buffer.buffer || lifetimes.buffers[index].first_use == framegraph::INVALID_INDEX)
            continue;
        TransientBuffer physical;
        if (!allocate || !try_allocate_transient([&] {
                return allocate(buffer.transient_description, physical, error);
            }, error) ||
            !physical.handle || !physical.owner ||
            !valid_sharing_metadata(physical.sharing_mode, physical.queue_families))
        {
            if (error.empty())
                error = "Frame image graph: transient buffer allocation failed";
            return false;
        }
        if (!handles.insert(physical.handle).second)
        {
            error = "Frame image graph: transient buffer allocator returned a handle already in use";
            return false;
        }
        allocations.emplace_back(index, std::move(physical));
    }
    for (auto& pending : allocations)
    {
        Buffer& buffer = m_buffers[pending.first];
        auto& physical = pending.second;
        buffer.buffer = physical.handle;
        buffer.sharing_mode = physical.sharing_mode;
        buffer.sharing_queue_families = std::move(physical.queue_families);
        buffer.allocation = std::move(physical.owner);
        buffer.initial_queue = m_passes[lifetimes.pass_order[lifetimes.buffers[pending.first].first_use]].queue;
    }
    return true;
}

VkBuffer FrameImageGraph::buffer_handle(BufferResource resource) const
{
    if (resource.generation != generation_ || resource.index >= m_buffers.size())
        return VK_NULL_HANDLE;
    const Buffer& buffer = m_buffers[resource.index];
    return resource.version < buffer.versions.size() ? buffer.buffer : VK_NULL_HANDLE;
}

void FrameImageGraph::retire_allocations(framegraph::ResourceRetirement& retirement, uint32_t slots)
{
    const auto retire = [&](auto& resource, auto& handle) {
        if (!resource.allocation)
            return;
        retirement.defer_for_slots(slots, [owner = resource.allocation]() mutable { owner.reset(); });
        resource.allocation.reset();
        handle = VK_NULL_HANDLE;
    };
    for (auto& buffer : m_buffers)
        retire(buffer, buffer.buffer);
    for (auto& image : m_images)
        retire(image, image.image);
}

bool FrameImageGraph::valid_buffer(Pass pass, BufferResource resource, std::string& error) const
{
    if (pass >= m_passes.size() || resource.generation != generation_ || resource.index >= m_buffers.size() ||
        resource.version >= m_buffers[resource.index].versions.size())
    {
        error = "Frame image graph: invalid pass or buffer version";
        return false;
    }
    return true;
}

bool FrameImageGraph::buffer_ranges_overlap(BufferResource first, BufferResource second) const
{
    const Buffer& a = m_buffers[first.index];
    const Buffer& b = m_buffers[second.index];
    return a.buffer && a.buffer == b.buffer && a.offset < b.offset + b.size && b.offset < a.offset + a.size;
}

void FrameImageGraph::add_buffer_alias_hazard_edges(std::vector<std::set<Pass>>& edges,
    std::vector<uint32_t>& indegree) const
{
    struct AccessRange
    {
        Pass pass;
        uint32_t resource_id;
        VkBuffer buffer;
        VkDeviceSize begin, end;
        bool writes;
    };
    std::vector<AccessRange> accesses;
    for (Pass pass = 0; pass < m_passes.size(); ++pass)
    {
        for (const BufferAccess& access : m_passes[pass].buffer_reads)
        {
            const Buffer& buffer = m_buffers[access.resource.index];
            accesses.push_back({pass, access.resource.index, buffer.buffer, buffer.offset,
                buffer.offset + buffer.size, false});
        }
        for (const BufferAccess& access : m_passes[pass].buffer_writes)
        {
            const Buffer& buffer = m_buffers[access.resource.index];
            accesses.push_back({pass, access.resource.index, buffer.buffer, buffer.offset,
                buffer.offset + buffer.size, true});
        }
    }
    const auto has_path = [&edges](Pass from, Pass to) {
        std::vector<Pass> pending{from};
        std::vector<bool> visited(edges.size());
        visited[from] = true;
        while (!pending.empty())
        {
            const Pass current = pending.back();
            pending.pop_back();
            for (Pass next : edges[current])
            {
                if (next == to)
                    return true;
                if (!visited[next])
                {
                    visited[next] = true;
                    pending.push_back(next);
                }
            }
        }
        return false;
    };
    for (size_t i = 0; i < accesses.size(); ++i)
    {
        const AccessRange& a = accesses[i];
        for (size_t j = i + 1; j < accesses.size(); ++j)
        {
            const AccessRange& b = accesses[j];
            if (!a.buffer || a.pass == b.pass || a.resource_id == b.resource_id || a.buffer != b.buffer ||
                (!a.writes && !b.writes) ||
                a.begin >= b.end || b.begin >= a.end)
                continue;
            if (has_path(a.pass, b.pass) || has_path(b.pass, a.pass))
                continue;
            // Distinct imports have independent version chains. Preserve a
            // deterministic order for their otherwise-unordered alias hazard.
            const Pass producer = std::min(a.pass, b.pass);
            const Pass consumer = std::max(a.pass, b.pass);
            if (edges[producer].insert(consumer).second)
                ++indegree[consumer];
        }
    }
}

bool FrameImageGraph::read_buffer(Pass pass, BufferResource resource,
    BufferUse use, std::string& error)
{
    if (!valid_buffer(pass, resource, error)) return false;
    if (use != BufferUse::ComputeRead && use != BufferUse::FragmentRead &&
        use != BufferUse::VertexRead && use != BufferUse::IndirectRead && use != BufferUse::IndirectAndComputeRead &&
        use != BufferUse::TransferSource && use != BufferUse::HostRead)
    {
        error = "Frame image graph: invalid buffer read use";
        return false;
    }
    const Buffer& buffer = m_buffers[resource.index];
    if (!resource.version && buffer.transient_description.usage)
    {
        error = "Frame image graph: transient buffer read requires a written version";
        return false;
    }
    if (buffer.versions[resource.version].producer == pass)
    {
        error = "Frame image graph: a pass cannot read its own buffer output";
        return false;
    }
    for (const BufferAccess& access : m_passes[pass].buffer_reads)
        if (access.resource.index == resource.index || buffer_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping buffer ranges may be accessed only once per pass";
            return false;
        }
    for (const BufferAccess& access : m_passes[pass].buffer_writes)
        if (access.resource.index == resource.index || buffer_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping buffer ranges may be accessed only once per pass";
            return false;
        }
    m_buffers[resource.index].versions[resource.version].readers.push_back(pass);
    m_passes[pass].buffer_reads.push_back({resource, use});
    return true;
}

bool FrameImageGraph::write_buffer(Pass pass, BufferResource resource,
    BufferUse use, BufferResource& next, std::string& error)
{
    if (!valid_buffer(pass, resource, error)) return false;
    if (use != BufferUse::ComputeWrite && use != BufferUse::TransferDestination)
    {
        error = "Frame image graph: invalid buffer write use";
        return false;
    }
    for (const BufferAccess& access : m_passes[pass].buffer_reads)
        if (access.resource.index == resource.index || buffer_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping buffer ranges may be accessed only once per pass";
            return false;
        }
    for (const BufferAccess& access : m_passes[pass].buffer_writes)
        if (access.resource.index == resource.index || buffer_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping buffer ranges may be accessed only once per pass";
            return false;
        }
    Buffer& buffer = m_buffers[resource.index];
    if (resource.version + 1 != buffer.versions.size())
    {
        error = "Frame image graph: buffer write requires latest version";
        return false;
    }
    next = {resource.index, uint32_t(buffer.versions.size()), generation_};
    buffer.versions.push_back({pass, {}});
    m_passes[pass].buffer_writes.push_back({next, use});
    return true;
}

bool FrameImageGraph::read_write_buffer(Pass pass, BufferResource resource,
    BufferUse use, BufferResource& next, std::string& error)
{
    if (!valid_buffer(pass, resource, error)) return false;
    if (!resource.version && m_buffers[resource.index].transient_description.usage)
    {
        error = "Frame image graph: transient buffer read/write requires a written version";
        return false;
    }
    if (use != BufferUse::ComputeReadWrite)
    {
        error = "Frame image graph: invalid buffer read/write use";
        return false;
    }
    for (const BufferAccess& access : m_passes[pass].buffer_reads)
        if (access.resource.index == resource.index || buffer_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping buffer ranges may be accessed only once per pass";
            return false;
        }
    for (const BufferAccess& access : m_passes[pass].buffer_writes)
        if (access.resource.index == resource.index || buffer_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping buffer ranges may be accessed only once per pass";
            return false;
        }
    Buffer& buffer = m_buffers[resource.index];
    if (resource.version + 1 != buffer.versions.size())
    {
        error = "Frame image graph: buffer read/write requires latest version";
        return false;
    }
    buffer.versions[resource.version].readers.push_back(pass);
    next = {resource.index, uint32_t(buffer.versions.size()), generation_};
    buffer.versions.push_back({pass, {}});
    m_passes[pass].buffer_writes.push_back({next, use});
    return true;
}

bool FrameImageGraph::import_image(VkImage image, const VkImageSubresourceRange& range,
    Resource& resource, std::string& error, FrameSubmissionGraph::Queue initial_queue,
    uint64_t initial_submission, VkSharingMode sharing_mode,
    std::vector<uint32_t> sharing_queue_families, ImageUse initial_use,
    FrameSubmissionGraph::Pass initial_producer_pass)
{
    if (initial_submission && initial_producer_pass != FrameSubmissionGraph::InvalidPass)
    {
        error = "Frame image graph: image import has both a submission token and a producer pass";
        return false;
    }
    if (initial_producer_pass != FrameSubmissionGraph::InvalidPass &&
        initial_use == ImageUse::Undefined)
    {
        error = "Frame image graph: image producer pass has no terminal use";
        return false;
    }
    if (image == VK_NULL_HANDLE || !range.aspectMask || !range.levelCount || !range.layerCount ||
        range.levelCount == VK_REMAINING_MIP_LEVELS ||
        range.layerCount == VK_REMAINING_ARRAY_LAYERS ||
        uint64_t(range.baseMipLevel) + range.levelCount > UINT32_MAX ||
        uint64_t(range.baseArrayLayer) + range.layerCount > UINT32_MAX ||
        !valid_sharing_metadata(sharing_mode, sharing_queue_families))
    {
        error = "Frame image graph: invalid imported image range";
        return false;
    }
    for (const Image& existing : m_images)
    {
        if (existing.image != image)
            continue;
        if (existing.sharing_mode != sharing_mode ||
            existing.sharing_queue_families != sharing_queue_families)
        {
            error = "Frame image graph: imports of one image have different sharing metadata";
            return false;
        }
        if (same_image_subresources(existing.range, range))
        {
            if (existing.initial_submission != initial_submission || existing.initial_queue != initial_queue ||
                existing.initial_use != initial_use || existing.initial_producer_pass != initial_producer_pass)
            {
                error = "Frame image graph: repeated image import has different producer metadata";
                return false;
            }
            resource = {uint32_t(&existing - m_images.data()), uint32_t(existing.versions.size() - 1), generation_};
            error.clear();
            return true;
        }
        if (image_subresources_overlap(existing.range, range) &&
            (existing.initial_submission != initial_submission || existing.initial_queue != initial_queue ||
                existing.initial_use != initial_use || existing.initial_producer_pass != initial_producer_pass))
        {
            error = "Frame image graph: overlapping image imports have different producer metadata";
            return false;
        }
    }
    resource = {uint32_t(m_images.size()), 0, generation_};
    m_images.push_back({image, range, initial_queue, initial_use, initial_submission,
        initial_producer_pass, sharing_mode, std::move(sharing_queue_families), {{}}});
    error.clear();
    return true;
}

bool FrameImageGraph::declare_image(const VkImageCreateInfo& info, VkImageAspectFlags aspects,
    Resource& resource, std::string& error)
{
    if (info.sType != VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO || info.pNext ||
        info.initialLayout != VK_IMAGE_LAYOUT_UNDEFINED || !aspects || !info.usage ||
        !info.extent.width || !info.extent.height || !info.extent.depth ||
        !info.mipLevels || !info.arrayLayers ||
        (info.queueFamilyIndexCount && !info.pQueueFamilyIndices))
    {
        error = "Frame image graph: invalid transient image descriptor";
        return false;
    }

    Image image;
    image.range = {aspects, 0, info.mipLevels, 0, info.arrayLayers};
    image.initial_queue = default_queue_;
    image.sharing_mode = info.sharingMode;
    if (info.queueFamilyIndexCount)
        image.sharing_queue_families.assign(info.pQueueFamilyIndices,
            info.pQueueFamilyIndices + info.queueFamilyIndexCount);
    if (!valid_sharing_metadata(image.sharing_mode, image.sharing_queue_families))
    {
        error = "Frame image graph: invalid transient image queue families";
        return false;
    }
    image.transient_info = info;
    image.transient_info.pQueueFamilyIndices = nullptr;
    m_images.push_back(std::move(image));
    resource = {uint32_t(m_images.size() - 1), 0, generation_};
    error.clear();
    return true;
}

bool FrameImageGraph::materialize_images(const AllocateImage& allocate, std::string& error)
{
    ResourceLifetimes lifetimes;
    if (!compile_resource_lifetimes(lifetimes, error))
        return false;
    std::vector<std::pair<size_t, TransientImage>> allocations;
    std::set<VkImage> handles;
    for (const auto& image : m_images)
        if (image.image) handles.insert(image.image);
    for (size_t index = 0; index < m_images.size(); ++index)
    {
        const Image& image = m_images[index];
        if (image.image || lifetimes.images[index].first_use == framegraph::INVALID_INDEX)
            continue;
        VkImageCreateInfo info = image.transient_info;
        info.pQueueFamilyIndices = image.sharing_queue_families.empty()
            ? nullptr : image.sharing_queue_families.data();
        TransientImage physical;
        if (!allocate || !try_allocate_transient([&] { return allocate(info, physical, error); }, error) ||
            !physical.handle || !physical.owner)
        {
            if (error.empty())
                error = "Frame image graph: transient image allocation failed";
            return false;
        }
        if (!handles.insert(physical.handle).second)
        {
            error = "Frame image graph: transient image allocator returned a handle already in use";
            return false;
        }
        allocations.emplace_back(index, std::move(physical));
    }
    for (auto& pending : allocations)
    {
        auto& image = m_images[pending.first];
        image.image = pending.second.handle;
        image.allocation = std::move(pending.second.owner);
        image.initial_queue = m_passes[lifetimes.pass_order[lifetimes.images[pending.first].first_use]].queue;
    }
    return true;
}

VkImage FrameImageGraph::image_handle(Resource resource) const
{
    if (resource.generation != generation_ || resource.index >= m_images.size())
        return VK_NULL_HANDLE;
    const Image& image = m_images[resource.index];
    return resource.version < image.versions.size() ? image.image : VK_NULL_HANDLE;
}

bool FrameImageGraph::materialize_resources(const AllocateBuffer& allocate_buffer,
    const AllocateImage& allocate_image, std::string& error)
{
    std::vector<size_t> unbound_buffers;
    for (size_t index = 0; index < m_buffers.size(); ++index)
        if (!m_buffers[index].buffer) unbound_buffers.push_back(index);
    if (!materialize_buffers(allocate_buffer, error)) return false;
    if (materialize_images(allocate_image, error)) return true;

    for (size_t index : unbound_buffers)
    {
        auto& buffer = m_buffers[index];
        buffer.buffer = VK_NULL_HANDLE;
        buffer.allocation.reset();
        buffer.sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
        buffer.sharing_queue_families.clear();
        buffer.initial_queue = default_queue_;
    }
    return false;
}

FrameImageGraph::Pass FrameImageGraph::add_pass(const char* name, Record record)
{
    return add_pass(name, default_queue_, std::move(record));
}

FrameImageGraph::Pass FrameImageGraph::add_pass(const char* name,
    FrameSubmissionGraph::Queue queue, Record record)
{
    const Pass result = Pass(m_passes.size());
    m_passes.push_back({name ? name : "", queue, std::move(record), {}, {}, {}, {}, {}, {}, {}});
    return result;
}

FrameImageGraph::Pass FrameImageGraph::add_checked_pass(const char* name, CheckedRecord record)
{
    return add_checked_pass(name, default_queue_, std::move(record));
}

FrameImageGraph::Pass FrameImageGraph::add_checked_pass(const char* name,
    FrameSubmissionGraph::Queue queue, CheckedRecord record)
{
    const Pass result = Pass(m_passes.size());
    m_passes.push_back({name ? name : "", queue, {}, std::move(record), {}, {}, {}, {}, {}, {}});
    return result;
}

FrameImageGraph::Pass FrameImageGraph::add_command_pass(const char* name, CommandRecord record)
{
    return add_command_pass(name, default_queue_, std::move(record));
}

FrameImageGraph::Pass FrameImageGraph::add_command_pass(const char* name,
    FrameSubmissionGraph::Queue queue, CommandRecord record)
{
    const Pass result = Pass(m_passes.size());
    m_passes.push_back({name ? name : "", queue, {}, {}, std::move(record), {}, {}, {}, {}, {}});
    return result;
}

FrameImageGraph::Pass FrameImageGraph::add_checked_command_pass(const char* name,
    CheckedCommandRecord record)
{
    return add_checked_command_pass(name, default_queue_, std::move(record));
}

FrameImageGraph::Pass FrameImageGraph::add_checked_command_pass(const char* name,
    FrameSubmissionGraph::Queue queue, CheckedCommandRecord record)
{
    const Pass result = Pass(m_passes.size());
    m_passes.push_back({name ? name : "", queue, {}, {}, {}, std::move(record), {}, {}, {}, {}});
    return result;
}

bool FrameImageGraph::depends_on(Pass after, Pass before, std::string& error)
{
    if (after >= m_passes.size() || before >= m_passes.size() || after == before)
    {
        error = "Frame image graph: explicit dependency references an invalid or identical pass";
        return false;
    }
    const auto duplicate = std::find_if(m_dependencies.begin(), m_dependencies.end(),
        [after, before](const Dependency& dependency) {
            return dependency.after == after && dependency.before == before;
        });
    if (duplicate == m_dependencies.end())
        m_dependencies.push_back({before, after});
    error.clear();
    return true;
}

bool FrameImageGraph::valid(Pass pass, Resource resource, std::string& error) const
{
    if (pass >= m_passes.size() || resource.generation != generation_ || resource.index >= m_images.size() ||
        resource.version >= m_images[resource.index].versions.size())
    {
        error = "Frame image graph: invalid pass or resource version";
        return false;
    }
    return true;
}

bool FrameImageGraph::image_ranges_overlap(Resource first, Resource second) const
{
    const Image& a = m_images[first.index];
    const Image& b = m_images[second.index];
    return a.image && a.image == b.image && image_subresources_overlap(a.range, b.range);
}

void FrameImageGraph::add_image_alias_hazard_edges(std::vector<std::set<Pass>>& edges,
    std::vector<uint32_t>& indegree) const
{
    struct AccessRange
    {
        Pass pass;
        uint32_t resource_id;
        VkImage image;
        VkImageSubresourceRange range;
        bool writes;
    };
    std::vector<AccessRange> accesses;
    for (Pass pass = 0; pass < m_passes.size(); ++pass)
    {
        for (const Access& access : m_passes[pass].reads)
        {
            const Image& image = m_images[access.resource.index];
            accesses.push_back({pass, access.resource.index, image.image, image.range, false});
        }
        for (const Access& access : m_passes[pass].writes)
        {
            const Image& image = m_images[access.resource.index];
            accesses.push_back({pass, access.resource.index, image.image, image.range, true});
        }
    }
    const auto has_path = [&edges](Pass from, Pass to) {
        std::vector<Pass> pending{from};
        std::vector<bool> visited(edges.size());
        visited[from] = true;
        while (!pending.empty())
        {
            const Pass current = pending.back();
            pending.pop_back();
            for (Pass next : edges[current])
            {
                if (next == to)
                    return true;
                if (!visited[next])
                {
                    visited[next] = true;
                    pending.push_back(next);
                }
            }
        }
        return false;
    };
    for (size_t i = 0; i < accesses.size(); ++i)
    {
        const AccessRange& a = accesses[i];
        for (size_t j = i + 1; j < accesses.size(); ++j)
        {
            const AccessRange& b = accesses[j];
            if (!a.image || a.pass == b.pass || a.resource_id == b.resource_id || a.image != b.image ||
                (!a.writes && !b.writes) || !image_subresources_overlap(a.range, b.range))
                continue;
            if (has_path(a.pass, b.pass) || has_path(b.pass, a.pass))
                continue;
            // Distinct imports have independent version chains. Preserve a
            // deterministic order for their otherwise-unordered alias hazard.
            const Pass producer = std::min(a.pass, b.pass);
            const Pass consumer = std::max(a.pass, b.pass);
            if (edges[producer].insert(consumer).second)
                ++indegree[consumer];
        }
    }
}

bool FrameImageGraph::read(Pass pass, Resource resource, ImageUse use, std::string& error)
{
    if (!valid(pass, resource, error))
        return false;
    if (!resource.version && m_images[resource.index].transient_info.sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
    {
        error = "Frame image graph: transient image read requires a written version";
        return false;
    }
    if (use != ImageUse::TransferSource && use != ImageUse::Sampled &&
        use != ImageUse::ComputeSampled && use != ImageUse::SampledGraphicsCompute &&
        use != ImageUse::DepthStencilReadOnly && use != ImageUse::StorageRead &&
        use != ImageUse::Present)
    {
        error = "Frame image graph: invalid read use";
        return false;
    }
    if (m_images[resource.index].versions[resource.version].producer == pass)
    {
        error = "Frame image graph: a pass cannot read its own output version";
        return false;
    }
    for (const Access& access : m_passes[pass].reads)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    for (const Access& access : m_passes[pass].writes)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    m_images[resource.index].versions[resource.version].readers.push_back(pass);
    m_passes[pass].reads.push_back({resource, use});
    return true;
}

bool FrameImageGraph::write(Pass pass, Resource resource, ImageUse use,
    Resource& next, std::string& error)
{
    if (!valid(pass, resource, error))
        return false;
    if (use != ImageUse::TransferDestination && use != ImageUse::TransferDestinationGeneral &&
        use != ImageUse::StorageWrite && use != ImageUse::ColorAttachment &&
        use != ImageUse::DepthStencilAttachment)
    {
        error = "Frame image graph: invalid write use";
        return false;
    }
    for (const Access& access : m_passes[pass].reads)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    for (const Access& access : m_passes[pass].writes)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    Image& image = m_images[resource.index];
    if (resource.version + 1 != image.versions.size())
    {
        error = "Frame image graph: write requires the latest resource version";
        return false;
    }
    next = {resource.index, uint32_t(image.versions.size()), generation_};
    image.versions.push_back({pass, {}});
    m_passes[pass].writes.push_back({next, use});
    return true;
}

bool FrameImageGraph::write_render_pass(Pass pass, Resource resource,
    ImageUse attachment_use, ImageUse final_use, Resource& next, std::string& error,
    ImageUse initial_use)
{
    if (attachment_use != ImageUse::ColorAttachment &&
        attachment_use != ImageUse::DepthStencilAttachment)
    {
        error = "Frame image graph: render-pass output must use a color or depth attachment";
        return false;
    }
    switch (final_use)
    {
    case ImageUse::TransferSource:
    case ImageUse::TransferDestination:
    case ImageUse::TransferDestinationGeneral:
    case ImageUse::Sampled:
    case ImageUse::ComputeSampled:
    case ImageUse::SampledGraphicsCompute:
    case ImageUse::StorageRead:
    case ImageUse::StorageWrite:
    case ImageUse::StorageReadWrite:
    case ImageUse::ColorAttachment:
    case ImageUse::DepthStencilAttachment:
    case ImageUse::DepthStencilReadOnly:
    case ImageUse::Present:
        break;
    default:
        error = "Frame image graph: render-pass final state is invalid or import-only";
        return false;
    }
    if (!valid(pass, resource, error))
        return false;
    for (const Access& access : m_passes[pass].reads)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    for (const Access& access : m_passes[pass].writes)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    Image& image = m_images[resource.index];
    if (resource.version + 1 != image.versions.size())
    {
        error = "Frame image graph: write requires the latest resource version";
        return false;
    }
    next = {resource.index, uint32_t(image.versions.size()), generation_};
    image.versions.push_back({pass, {}});
    m_passes[pass].writes.push_back({next,
        initial_use == ImageUse::Undefined ? attachment_use : initial_use, final_use});
    error.clear();
    return true;
}

bool FrameImageGraph::read_write(Pass pass, Resource resource, Resource& next, std::string& error,
    ImageUse use)
{
    if (!valid(pass, resource, error))
        return false;
    if (!resource.version && m_images[resource.index].transient_info.sType == VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO)
    {
        error = "Frame image graph: transient image read/write requires a written version";
        return false;
    }
    if (use != ImageUse::StorageReadWrite && use != ImageUse::ColorAttachment)
    {
        error = "Frame image graph: invalid read/write use";
        return false;
    }
    for (const Access& access : m_passes[pass].reads)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    for (const Access& access : m_passes[pass].writes)
        if (access.resource.index == resource.index || image_ranges_overlap(access.resource, resource))
        {
            error = "Frame image graph: overlapping image subresources may be accessed only once per pass";
            return false;
        }
    Image& image = m_images[resource.index];
    if (resource.version + 1 != image.versions.size())
    {
        error = "Frame image graph: read_write requires the latest resource version";
        return false;
    }
    image.versions[resource.version].readers.push_back(pass);
    next = {resource.index, uint32_t(image.versions.size()), generation_};
    image.versions.push_back({pass, {}});
    m_passes[pass].writes.push_back({next, use});
    return true;
}

bool FrameImageGraph::build_pass_graph(std::vector<std::set<Pass>>& edges,
    std::vector<Pass>& order, std::string& error) const
{
    edges.assign(m_passes.size(), {});
    std::vector<uint32_t> indegree(m_passes.size());
    const auto edge = [&](Pass from, Pass to) {
        if (from != ~0u && from != to && edges[from].insert(to).second)
            ++indegree[to];
    };
    for (const Image& image : m_images)
        framegraph::add_resource_dependencies(image.versions, edge);
    add_image_alias_hazard_edges(edges, indegree);
    for (const Buffer& buffer : m_buffers)
        framegraph::add_resource_dependencies(buffer.versions, edge);
    add_buffer_alias_hazard_edges(edges, indegree);
    for (const Dependency& dependency : m_dependencies)
    {
        if (dependency.before >= m_passes.size() || dependency.after >= m_passes.size() ||
            dependency.before == dependency.after)
        {
            error = "Frame image graph: explicit dependency became invalid";
            return false;
        }
        edge(dependency.before, dependency.after);
    }

    if (!framegraph::sort_passes(edges, std::move(indegree), order))
    {
        error = "Frame image graph: cyclic pass dependencies";
        return false;
    }
    error.clear();
    return true;
}

bool FrameImageGraph::compile_resource_lifetimes(ResourceLifetimes& lifetimes, std::string& error) const
{
    ResourceLifetimes compiled;
    std::vector<std::set<Pass>> edges;
    if (!build_pass_graph(edges, compiled.pass_order, error))
        return false;

    compiled.images.resize(m_images.size());
    compiled.buffers.resize(m_buffers.size());
    const auto record_image = [&](const Access& access, uint32_t index)
    {
        const auto usage = m_images[access.resource.index].transient_info.usage;
        const auto required = required_image_usage(access.use) | required_image_usage(access.final_use);
        if (usage && ((usage & required) != required || access.use == ImageUse::Present ||
                access.final_use == ImageUse::Present))
        {
            error = "Frame image graph: transient image usage does not support pass " +
                m_passes[compiled.pass_order[index]].name;
            return false;
        }
        compiled.images[access.resource.index].record_use(index);
        return true;
    };
    const auto record_buffer = [&](const BufferAccess& access, uint32_t index)
    {
        const auto usage = m_buffers[access.resource.index].transient_description.usage;
        const auto required = required_buffer_usage(access.use);
        if (usage && (usage & required) != required)
        {
            error = "Frame image graph: transient buffer usage does not support pass " +
                m_passes[compiled.pass_order[index]].name;
            return false;
        }
        compiled.buffers[access.resource.index].record_use(index);
        return true;
    };
    for (uint32_t index = 0; index < compiled.pass_order.size(); ++index)
    {
        const PassInfo& pass = m_passes[compiled.pass_order[index]];
        for (const auto& access : pass.reads)
            if (!record_image(access, index)) return false;
        for (const auto& access : pass.writes)
            if (!record_image(access, index)) return false;
        for (const auto& access : pass.buffer_reads)
            if (!record_buffer(access, index)) return false;
        for (const auto& access : pass.buffer_writes)
            if (!record_buffer(access, index)) return false;
    }
    lifetimes = std::move(compiled);
    return true;
}

bool FrameImageGraph::build_queue_groups(std::vector<QueueGroup>& groups,
    std::string& error) const
{
    std::vector<std::set<Pass>> edges;
    std::vector<Pass> pass_order;
    if (!build_pass_graph(edges, pass_order, error))
        return false;

    std::vector<QueueGroup> candidate;
    std::array<uint32_t, 3> groups_for_queue{};
    for (Pass pass : pass_order)
    {
        const FrameSubmissionGraph::Queue queue = m_passes[pass].queue;
        if (queue != FrameSubmissionGraph::Queue::Graphics &&
            queue != FrameSubmissionGraph::Queue::Compute &&
            queue != FrameSubmissionGraph::Queue::Copy)
        {
            error = "Frame image graph: pass declares an invalid queue class";
            return false;
        }
        if (candidate.empty() || candidate.back().queue != queue)
        {
            const char* queue_name = "graphics";
            const size_t queue_index = static_cast<size_t>(queue);
            if (queue == FrameSubmissionGraph::Queue::Compute)
                queue_name = "compute";
            else if (queue == FrameSubmissionGraph::Queue::Copy)
                queue_name = "copy";
            QueueGroup group;
            group.queue = queue;
            group.name = std::string("frame graph ") + queue_name + " group " +
                std::to_string(groups_for_queue[queue_index]++);
            candidate.push_back(std::move(group));
        }
        candidate.back().passes.push_back(pass);
    }
    groups = std::move(candidate);
    error.clear();
    return true;
}

bool FrameImageGraph::generate_submission_groups(FrameSubmissionGraph& submissions,
    const QueueFamilyForQueue& queue_family_for_queue, std::vector<QueueGroup>& groups,
    std::vector<FrameSubmissionGraph::Pass>& submission_for_pass, std::string& error) const
{
    if (!queue_family_for_queue)
    {
        error = "Frame image graph: queue-family mapping is unavailable";
        return false;
    }

    std::vector<QueueGroup> candidate_groups;
    if (!build_queue_groups(candidate_groups, error))
        return false;

    FrameSubmissionGraph candidate_submissions = submissions;
    std::vector<FrameSubmissionGraph::Pass> candidate_mapping(m_passes.size(),
        FrameSubmissionGraph::InvalidPass);
    std::array<FrameSubmissionGraph::Pass, 3> last_submission_for_queue{
        FrameSubmissionGraph::InvalidPass, FrameSubmissionGraph::InvalidPass,
        FrameSubmissionGraph::InvalidPass};
    for (QueueGroup& group : candidate_groups)
    {
        const uint32_t family = queue_family_for_queue(group.queue);
        if (family == VK_QUEUE_FAMILY_IGNORED ||
            family == FrameSubmissionGraph::InvalidQueueFamily)
        {
            error = "Frame image graph: queue family is unavailable for generated " + group.name;
            return false;
        }
        group.queue_family = family;
        const auto submission = candidate_submissions.add_pass(group.name.c_str(), group.queue, family);
        group.submission = submission;
        const size_t queue_index = static_cast<size_t>(group.queue);
        if (last_submission_for_queue[queue_index] != FrameSubmissionGraph::InvalidPass &&
            !candidate_submissions.depends_on(submission,
                last_submission_for_queue[queue_index], error))
            return false;
        last_submission_for_queue[queue_index] = submission;
        for (Pass pass : group.passes)
            candidate_mapping[pass] = submission;
    }

    const SubmissionForPass map_pass = [&candidate_mapping](Pass pass) {
        return pass < candidate_mapping.size() ? candidate_mapping[pass] :
            FrameSubmissionGraph::InvalidPass;
    };
    if (!connect_submission_dependencies(candidate_submissions, map_pass, error,
            ProducerHandoffMode::DeferToGeneratedGroups) ||
        !add_generated_ownership_transfers(candidate_submissions, candidate_groups,
            candidate_mapping, queue_family_for_queue, error))
        return false;

    std::vector<FrameSubmissionGraph::Pass> execution_order;
    if (!candidate_submissions.execution_order(execution_order, error))
        return false;
    std::vector<size_t> submission_rank(execution_order.size(),
        std::numeric_limits<size_t>::max());
    for (size_t rank = 0; rank < execution_order.size(); ++rank)
        submission_rank[execution_order[rank]] = rank;
    std::stable_sort(candidate_groups.begin(), candidate_groups.end(),
        [&candidate_mapping, &submission_rank](const QueueGroup& first, const QueueGroup& second) {
            const auto first_submission = first.submission != FrameSubmissionGraph::InvalidPass ?
                first.submission : candidate_mapping[first.passes.front()];
            const auto second_submission = second.submission != FrameSubmissionGraph::InvalidPass ?
                second.submission : candidate_mapping[second.passes.front()];
            return submission_rank[first_submission] < submission_rank[second_submission];
        });

    submissions = std::move(candidate_submissions);
    groups = std::move(candidate_groups);
    submission_for_pass = std::move(candidate_mapping);
    error.clear();
    return true;
}

bool FrameImageGraph::add_generated_ownership_transfers(FrameSubmissionGraph& submissions,
    std::vector<QueueGroup>& groups,
    const std::vector<FrameSubmissionGraph::Pass>& submission_for_pass,
    const QueueFamilyForQueue& queue_family_for_queue, std::string& error) const
{
    if (!queue_family_for_queue || submission_for_pass.size() != m_passes.size())
    {
        error = "Frame image graph: generated ownership transfer mapping is incomplete";
        return false;
    }

    const Pass invalid_pass = ~0u;
    std::vector<uint32_t> group_for_pass(m_passes.size(), invalid_pass);
    std::vector<size_t> pass_rank(m_passes.size(), std::numeric_limits<size_t>::max());
    size_t next_rank = 0;
    for (uint32_t group_index = 0; group_index < groups.size(); ++group_index)
    {
        const QueueGroup& group = groups[group_index];
        if (group.queue_family == FrameSubmissionGraph::InvalidQueueFamily)
        {
            error = "Frame image graph: generated queue group has no queue family";
            return false;
        }
        for (Pass pass : group.passes)
        {
            if (pass >= m_passes.size() || group_for_pass[pass] != invalid_pass)
            {
                error = "Frame image graph: generated queue group has an invalid pass map";
                return false;
            }
            group_for_pass[pass] = group_index;
            pass_rank[pass] = next_rank++;
        }
    }
    for (Pass pass = 0; pass < m_passes.size(); ++pass)
        if (group_for_pass[pass] == invalid_pass)
        {
            error = "Frame image graph: a pass was omitted from generated queue groups";
            return false;
        }

    const auto family_for_pass = [&](Pass pass) {
        return pass < group_for_pass.size() ?
            groups[group_for_pass[pass]].queue_family : FrameSubmissionGraph::InvalidQueueFamily;
    };
    const auto validate_concurrent_families = [&](VkSharingMode sharing_mode,
        const std::vector<uint32_t>& sharing_queue_families,
        const std::set<uint32_t>& families, const char* kind, size_t resource_index) {
        if (sharing_mode != VK_SHARING_MODE_CONCURRENT)
            return true;
        const std::set<uint32_t> shared_families(sharing_queue_families.begin(),
            sharing_queue_families.end());
        for (uint32_t family : families)
            if (shared_families.find(family) == shared_families.end())
            {
                error = "Frame image graph: " + std::string(kind) + " " +
                    std::to_string(resource_index) +
                    " uses a queue family outside its concurrent-sharing list";
                return false;
            }
        return true;
    };
    const auto add_serial_dependency = [&](Pass source, Pass destination) {
        const FrameSubmissionGraph::Pass source_submission = submission_for_pass[source];
        const FrameSubmissionGraph::Pass destination_submission = submission_for_pass[destination];
        if (source_submission == FrameSubmissionGraph::InvalidPass ||
            destination_submission == FrameSubmissionGraph::InvalidPass)
        {
            error = "Frame image graph: exclusive ownership chain maps to no submission";
            return false;
        }
        return source_submission == destination_submission ||
            submissions.depends_on(destination_submission, source_submission, error);
    };

    struct InitialReleaseGroup
    {
        FrameSubmissionGraph::Queue queue;
        uint32_t family;
        uint64_t submission;
        FrameSubmissionGraph::Pass producer_pass;
        size_t group_index;
    };
    std::vector<InitialReleaseGroup> initial_release_groups;
    const auto add_initial_release = [&](FrameSubmissionGraph::Queue source_queue,
        uint32_t source_family, uint64_t source_submission,
        FrameSubmissionGraph::Pass source_producer_pass,
        const QueueGroup::OwnershipTransfer& transfer, Pass first_use,
        const std::vector<Pass>& initial_consumers) {
        auto release = std::find_if(initial_release_groups.begin(), initial_release_groups.end(),
            [source_queue, source_family, source_submission, source_producer_pass](
                const InitialReleaseGroup& entry) {
                return entry.queue == source_queue && entry.family == source_family &&
                    entry.submission == source_submission && entry.producer_pass == source_producer_pass;
            });
        if (release == initial_release_groups.end())
        {
            const char* queue_name = source_queue == FrameSubmissionGraph::Queue::Graphics ?
                "graphics" : source_queue == FrameSubmissionGraph::Queue::Compute ?
                    "compute" : "copy";
            QueueGroup group;
            group.queue = source_queue;
            group.name = std::string("frame graph initial ownership release ") + queue_name;
            group.queue_family = source_family;
            group.external_submission = source_submission;
            group.submission = submissions.add_pass(group.name.c_str(), source_queue, source_family);
            if (source_producer_pass != FrameSubmissionGraph::InvalidPass)
            {
                if (submissions.queue(source_producer_pass) != source_queue ||
                    submissions.queue_family(source_producer_pass) != source_family)
                {
                    error = "Frame image graph: initial ownership producer queue metadata is inconsistent";
                    return false;
                }
                if (!submissions.depends_on(group.submission, source_producer_pass, error))
                    return false;
            }
            else if (!source_submission || !submissions.depends_on_submission(group.submission,
                         source_queue, source_submission, error))
            {
                if (error.empty())
                    error = "Frame image graph: external ownership release has no producer";
                return false;
            }
            groups.push_back(std::move(group));
            initial_release_groups.push_back({source_queue, source_family, source_submission,
                source_producer_pass,
                groups.size() - 1});
            release = initial_release_groups.end() - 1;
        }

        QueueGroup& release_group = groups[release->group_index];
        release_group.ownership_releases.push_back(transfer);
        groups[group_for_pass[first_use]].ownership_acquires.push_back(transfer);
        const FrameSubmissionGraph::Pass destination_submission = submission_for_pass[first_use];
        if (destination_submission == FrameSubmissionGraph::InvalidPass ||
            !submissions.depends_on(destination_submission, release_group.submission, error))
            return false;
        for (Pass consumer : initial_consumers)
        {
            if (consumer >= submission_for_pass.size())
            {
                error = "Frame image graph: external ownership consumer pass is invalid";
                return false;
            }
            const FrameSubmissionGraph::Pass consumer_submission = submission_for_pass[consumer];
            if (source_submission && consumer_submission != FrameSubmissionGraph::InvalidPass &&
                !submissions.remove_dependency_on_submission(consumer_submission,
                    source_queue, source_submission, error))
                return false;
        }
        return true;
    };

    struct ImagePassUse { Pass pass; ImageUse use; uint32_t resource_id; };
    struct BufferPassUse { Pass pass; BufferUse use; uint32_t resource_id; };
    std::vector<std::vector<ImagePassUse>> image_accesses(m_images.size());
    std::vector<std::vector<BufferPassUse>> buffer_accesses(m_buffers.size());
    for (Pass pass = 0; pass < m_passes.size(); ++pass)
    {
        const PassInfo& info = m_passes[pass];
        for (const Access& access : info.reads)
            image_accesses[access.resource.index].push_back({pass, access.use, access.resource.index});
        for (const Access& access : info.writes)
            image_accesses[access.resource.index].push_back({pass,
                access.final_use != ImageUse::Undefined ? access.final_use : access.use,
                access.resource.index});
        for (const BufferAccess& access : info.buffer_reads)
            buffer_accesses[access.resource.index].push_back({pass, access.use, access.resource.index});
        for (const BufferAccess& access : info.buffer_writes)
            buffer_accesses[access.resource.index].push_back({pass, access.use, access.resource.index});
    }

    std::vector<bool> image_processed(m_images.size(), false);
    for (uint32_t first_resource = 0; first_resource < m_images.size(); ++first_resource)
    {
        if (image_processed[first_resource])
            continue;
        const VkImage image_handle = m_images[first_resource].image;
        std::vector<uint32_t> resources;
        std::vector<uint32_t> mip_boundaries, layer_boundaries;
        VkImageAspectFlags aspects = 0;
        for (uint32_t resource = first_resource; resource < m_images.size(); ++resource)
        {
            const Image& image = m_images[resource];
            if (image.image != image_handle)
                continue;
            image_processed[resource] = true;
            resources.push_back(resource);
            aspects |= image.range.aspectMask;
            mip_boundaries.push_back(image.range.baseMipLevel);
            mip_boundaries.push_back(image.range.baseMipLevel + image.range.levelCount);
            layer_boundaries.push_back(image.range.baseArrayLayer);
            layer_boundaries.push_back(image.range.baseArrayLayer + image.range.layerCount);
        }
        std::sort(mip_boundaries.begin(), mip_boundaries.end());
        mip_boundaries.erase(std::unique(mip_boundaries.begin(), mip_boundaries.end()),
            mip_boundaries.end());
        std::sort(layer_boundaries.begin(), layer_boundaries.end());
        layer_boundaries.erase(std::unique(layer_boundaries.begin(), layer_boundaries.end()),
            layer_boundaries.end());

        for (VkImageAspectFlags aspect = 1; aspect; aspect <<= 1)
        {
            if (!(aspects & aspect))
                continue;
            for (size_t mip_index = 1; mip_index < mip_boundaries.size(); ++mip_index)
            {
                const uint32_t mip_begin = mip_boundaries[mip_index - 1];
                const uint32_t mip_end = mip_boundaries[mip_index];
                if (mip_begin == mip_end)
                    continue;
                for (size_t layer_index = 1; layer_index < layer_boundaries.size(); ++layer_index)
                {
                    const uint32_t layer_begin = layer_boundaries[layer_index - 1];
                    const uint32_t layer_end = layer_boundaries[layer_index];
                    if (layer_begin == layer_end)
                        continue;
                    const VkImageSubresourceRange cell{aspect, mip_begin, mip_end - mip_begin,
                        layer_begin, layer_end - layer_begin};
                    std::vector<uint32_t> active_resources;
                    std::vector<ImagePassUse> accesses;
                    for (uint32_t resource : resources)
                    {
                        const Image& imported = m_images[resource];
                        const uint32_t imported_mip_end = imported.range.baseMipLevel +
                            imported.range.levelCount;
                        const uint32_t imported_layer_end = imported.range.baseArrayLayer +
                            imported.range.layerCount;
                        if (!(imported.range.aspectMask & aspect) ||
                            imported.range.baseMipLevel > mip_begin || imported_mip_end < mip_end ||
                            imported.range.baseArrayLayer > layer_begin || imported_layer_end < layer_end)
                            continue;
                        active_resources.push_back(resource);
                        accesses.insert(accesses.end(), image_accesses[resource].begin(),
                            image_accesses[resource].end());
                    }
                    if (active_resources.empty() || accesses.empty())
                        continue;
                    std::stable_sort(accesses.begin(), accesses.end(),
                        [&pass_rank](const ImagePassUse& first, const ImagePassUse& second) {
                            return pass_rank[first.pass] < pass_rank[second.pass];
                        });

                    const Image& initial = m_images[active_resources.front()];
                    for (uint32_t resource : active_resources)
                    {
                        const Image& alias = m_images[resource];
                        if (alias.sharing_mode != initial.sharing_mode ||
                            alias.sharing_queue_families != initial.sharing_queue_families ||
                            alias.initial_queue != initial.initial_queue ||
                            alias.initial_submission != initial.initial_submission ||
                            alias.initial_producer_pass != initial.initial_producer_pass ||
                            alias.initial_use != initial.initial_use)
                        {
                            error = "Frame image graph: overlapping image cell has inconsistent import metadata";
                            return false;
                        }
                    }

                    std::set<uint32_t> families;
                    for (const ImagePassUse& access : accesses)
                        families.insert(family_for_pass(access.pass));
                    const bool has_initial_producer = initial.initial_submission ||
                        initial.initial_producer_pass != FrameSubmissionGraph::InvalidPass;
                    if (has_initial_producer || initial.initial_use != ImageUse::Undefined)
                    {
                        const uint32_t initial_family = initial.initial_producer_pass !=
                                FrameSubmissionGraph::InvalidPass ?
                            submissions.queue_family(initial.initial_producer_pass) :
                            queue_family_for_queue(initial.initial_queue);
                        if (initial.initial_producer_pass != FrameSubmissionGraph::InvalidPass &&
                            submissions.queue(initial.initial_producer_pass) != initial.initial_queue)
                        {
                            error = "Frame image graph: initial image producer queue does not "
                                "match its import metadata";
                            return false;
                        }
                        if (initial_family == FrameSubmissionGraph::InvalidQueueFamily ||
                            initial_family == VK_QUEUE_FAMILY_IGNORED)
                        {
                            error = "Frame image graph: initial queue family is unavailable for image " +
                                std::to_string(active_resources.front());
                            return false;
                        }
                        families.insert(initial_family);
                    }
                    if (!validate_concurrent_families(initial.sharing_mode,
                            initial.sharing_queue_families, families, "image",
                            active_resources.front()))
                        return false;
                    if (initial.sharing_mode == VK_SHARING_MODE_CONCURRENT || families.size() < 2)
                        continue;

                    const uint32_t initial_family = initial.initial_producer_pass !=
                            FrameSubmissionGraph::InvalidPass ?
                        submissions.queue_family(initial.initial_producer_pass) :
                        queue_family_for_queue(initial.initial_queue);
                    if (has_initial_producer &&
                        family_for_pass(accesses.front().pass) != initial_family)
                    {
                        if (initial.initial_use == ImageUse::Undefined)
                        {
                            error = "Frame image graph: exclusive image " +
                                std::to_string(active_resources.front()) +
                                " needs its initial use to generate a queue-family release";
                            return false;
                        }
                        const uint32_t source_family = initial_family;
                        const ImagePassUse& first = accesses.front();
                        QueueGroup::OwnershipTransfer transfer;
                        transfer.kind = QueueGroup::OwnershipTransfer::Kind::Image;
                        transfer.resource_id = first.resource_id;
                        transfer.source_pass = invalid_pass;
                        transfer.destination_pass = first.pass;
                        transfer.source_family = source_family;
                        transfer.destination_family = family_for_pass(first.pass);
                        transfer.source_image_use = initial.initial_use;
                        transfer.destination_image_use = first.use;
                        transfer.image_range = cell;
                        std::vector<Pass> initial_consumers;
                        initial_consumers.reserve(accesses.size());
                        for (const ImagePassUse& access : accesses)
                            initial_consumers.push_back(access.pass);
                        std::sort(initial_consumers.begin(), initial_consumers.end());
                        initial_consumers.erase(std::unique(initial_consumers.begin(),
                            initial_consumers.end()), initial_consumers.end());
                        if (!add_initial_release(initial.initial_queue, source_family,
                                initial.initial_submission, initial.initial_producer_pass,
                                transfer, first.pass, initial_consumers))
                            return false;
                    }
                    if (!has_initial_producer && initial.initial_use != ImageUse::Undefined &&
                        family_for_pass(accesses.front().pass) !=
                            queue_family_for_queue(initial.initial_queue))
                    {
                        error = "Frame image graph: exclusive image " +
                            std::to_string(active_resources.front()) +
                            " needs a caller release before its first generated queue-family use";
                        return false;
                    }

                    for (size_t access_index = 1; access_index < accesses.size(); ++access_index)
                        if (!add_serial_dependency(accesses[access_index - 1].pass,
                                accesses[access_index].pass))
                            return false;
                    for (size_t access_index = 1; access_index < accesses.size(); ++access_index)
                    {
                        const ImagePassUse& source = accesses[access_index - 1];
                        const ImagePassUse& destination = accesses[access_index];
                        const uint32_t source_family = family_for_pass(source.pass);
                        const uint32_t destination_family = family_for_pass(destination.pass);
                        if (source_family == destination_family)
                            continue;
                        QueueGroup::OwnershipTransfer transfer;
                        transfer.kind = QueueGroup::OwnershipTransfer::Kind::Image;
                        transfer.resource_id = destination.resource_id;
                        transfer.source_pass = source.pass;
                        transfer.destination_pass = destination.pass;
                        transfer.source_family = source_family;
                        transfer.destination_family = destination_family;
                        transfer.source_image_use = source.use;
                        transfer.destination_image_use = destination.use;
                        transfer.image_range = cell;
                        groups[group_for_pass[source.pass]].ownership_releases.push_back(transfer);
                        groups[group_for_pass[destination.pass]].ownership_acquires.push_back(transfer);
                    }
                }
            }
        }
    }

    std::vector<bool> buffer_processed(m_buffers.size(), false);
    for (uint32_t first_resource = 0; first_resource < m_buffers.size(); ++first_resource)
    {
        if (buffer_processed[first_resource])
            continue;
        const VkBuffer buffer_handle = m_buffers[first_resource].buffer;
        std::vector<uint32_t> resources;
        std::vector<VkDeviceSize> boundaries;
        for (uint32_t resource = first_resource; resource < m_buffers.size(); ++resource)
        {
            const Buffer& buffer = m_buffers[resource];
            if (buffer.buffer != buffer_handle)
                continue;
            buffer_processed[resource] = true;
            resources.push_back(resource);
            boundaries.push_back(buffer.offset);
            boundaries.push_back(buffer.offset + buffer.size);
        }
        std::sort(boundaries.begin(), boundaries.end());
        boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
        for (size_t boundary_index = 1; boundary_index < boundaries.size(); ++boundary_index)
        {
            const VkDeviceSize begin = boundaries[boundary_index - 1];
            const VkDeviceSize end = boundaries[boundary_index];
            if (begin == end)
                continue;
            std::vector<uint32_t> active_resources;
            std::vector<BufferPassUse> accesses;
            for (uint32_t resource : resources)
            {
                const Buffer& imported = m_buffers[resource];
                if (imported.offset > begin || imported.offset + imported.size < end)
                    continue;
                active_resources.push_back(resource);
                accesses.insert(accesses.end(), buffer_accesses[resource].begin(),
                    buffer_accesses[resource].end());
            }
            if (active_resources.empty() || accesses.empty())
                continue;
            std::stable_sort(accesses.begin(), accesses.end(),
                [&pass_rank](const BufferPassUse& first, const BufferPassUse& second) {
                    return pass_rank[first.pass] < pass_rank[second.pass];
                });

            const Buffer& initial = m_buffers[active_resources.front()];
            for (uint32_t resource : active_resources)
            {
                const Buffer& alias = m_buffers[resource];
                if (alias.sharing_mode != initial.sharing_mode ||
                    alias.sharing_queue_families != initial.sharing_queue_families ||
                    alias.initial_queue != initial.initial_queue ||
                    alias.initial_submission != initial.initial_submission ||
                    alias.initial_producer_pass != initial.initial_producer_pass ||
                    alias.initial_use != initial.initial_use)
                {
                    error = "Frame image graph: overlapping buffer interval has inconsistent import metadata";
                    return false;
                }
            }

            const bool host_initial_use = initial.initial_use == BufferUse::HostWrite ||
                initial.initial_use == BufferUse::HostRead ||
                initial.initial_use == BufferUse::Undefined;
            std::set<uint32_t> families;
            for (const BufferPassUse& access : accesses)
                families.insert(family_for_pass(access.pass));
            const bool has_initial_producer = initial.initial_submission ||
                initial.initial_producer_pass != FrameSubmissionGraph::InvalidPass;
            if (has_initial_producer || !host_initial_use)
            {
                const uint32_t initial_family = initial.initial_producer_pass !=
                        FrameSubmissionGraph::InvalidPass ?
                    submissions.queue_family(initial.initial_producer_pass) :
                    queue_family_for_queue(initial.initial_queue);
                if (initial.initial_producer_pass != FrameSubmissionGraph::InvalidPass &&
                    submissions.queue(initial.initial_producer_pass) != initial.initial_queue)
                {
                    error = "Frame image graph: initial buffer producer queue does not match its import metadata";
                    return false;
                }
                if (initial_family == FrameSubmissionGraph::InvalidQueueFamily ||
                    initial_family == VK_QUEUE_FAMILY_IGNORED)
                {
                    error = "Frame image graph: initial queue family is unavailable for buffer " +
                        std::to_string(active_resources.front());
                    return false;
                }
                families.insert(initial_family);
            }
            if (!validate_concurrent_families(initial.sharing_mode,
                    initial.sharing_queue_families, families, "buffer", active_resources.front()))
                return false;
            if (initial.sharing_mode == VK_SHARING_MODE_CONCURRENT || families.size() < 2)
                continue;

            const uint32_t initial_family = initial.initial_producer_pass !=
                    FrameSubmissionGraph::InvalidPass ?
                submissions.queue_family(initial.initial_producer_pass) :
                queue_family_for_queue(initial.initial_queue);
            if (has_initial_producer &&
                family_for_pass(accesses.front().pass) != initial_family)
            {
                if (host_initial_use || initial.initial_use == BufferUse::Undefined)
                {
                    error = "Frame image graph: exclusive buffer " +
                        std::to_string(active_resources.front()) +
                        " has no queue-owned initial use for an ownership release";
                    return false;
                }
                const uint32_t source_family = initial_family;
                const BufferPassUse& first = accesses.front();
                QueueGroup::OwnershipTransfer transfer;
                transfer.kind = QueueGroup::OwnershipTransfer::Kind::Buffer;
                transfer.resource_id = first.resource_id;
                transfer.source_pass = invalid_pass;
                transfer.destination_pass = first.pass;
                transfer.source_family = source_family;
                transfer.destination_family = family_for_pass(first.pass);
                transfer.source_buffer_use = initial.initial_use;
                transfer.destination_buffer_use = first.use;
                transfer.buffer_offset = begin;
                transfer.buffer_size = end - begin;
                std::vector<Pass> initial_consumers;
                initial_consumers.reserve(accesses.size());
                for (const BufferPassUse& access : accesses)
                    initial_consumers.push_back(access.pass);
                std::sort(initial_consumers.begin(), initial_consumers.end());
                initial_consumers.erase(std::unique(initial_consumers.begin(),
                    initial_consumers.end()), initial_consumers.end());
                if (!add_initial_release(initial.initial_queue, source_family,
                        initial.initial_submission, initial.initial_producer_pass,
                        transfer, first.pass, initial_consumers))
                    return false;
            }
            if (!has_initial_producer && !host_initial_use &&
                family_for_pass(accesses.front().pass) != queue_family_for_queue(initial.initial_queue))
            {
                error = "Frame image graph: exclusive buffer " +
                    std::to_string(active_resources.front()) +
                    " needs a caller release before its first generated queue-family use";
                return false;
            }

            for (size_t access_index = 1; access_index < accesses.size(); ++access_index)
                if (!add_serial_dependency(accesses[access_index - 1].pass,
                        accesses[access_index].pass))
                    return false;
            for (size_t access_index = 1; access_index < accesses.size(); ++access_index)
            {
                const BufferPassUse& source = accesses[access_index - 1];
                const BufferPassUse& destination = accesses[access_index];
                const uint32_t source_family = family_for_pass(source.pass);
                const uint32_t destination_family = family_for_pass(destination.pass);
                if (source_family == destination_family)
                    continue;
                QueueGroup::OwnershipTransfer transfer;
                transfer.kind = QueueGroup::OwnershipTransfer::Kind::Buffer;
                transfer.resource_id = destination.resource_id;
                transfer.source_pass = source.pass;
                transfer.destination_pass = destination.pass;
                transfer.source_family = source_family;
                transfer.destination_family = destination_family;
                transfer.source_buffer_use = source.use;
                transfer.destination_buffer_use = destination.use;
                transfer.buffer_offset = begin;
                transfer.buffer_size = end - begin;
                groups[group_for_pass[source.pass]].ownership_releases.push_back(transfer);
                groups[group_for_pass[destination.pass]].ownership_acquires.push_back(transfer);
            }
        }
    }
    error.clear();
    return true;
}

bool FrameImageGraph::connect_submission_dependencies(FrameSubmissionGraph& submissions,
    const SubmissionForPass& submission_for_pass, std::string& error,
    ProducerHandoffMode handoff_mode) const
{
    if (!submission_for_pass)
    {
        error = "Frame image graph: submission mapping is unavailable";
        return false;
    }
    std::vector<std::set<Pass>> edges;
    std::vector<Pass> pass_order;
    if (!build_pass_graph(edges, pass_order, error))
        return false;

    for (Pass producer = 0; producer < edges.size(); ++producer)
    {
        const FrameSubmissionGraph::Pass source = submission_for_pass(producer);
        if (source != FrameSubmissionGraph::InvalidPass &&
            submissions.queue(source) != m_passes[producer].queue)
        {
            error = "Frame image graph: pass queue does not match its mapped submission";
            return false;
        }
        for (Pass consumer : edges[producer])
        {
            const FrameSubmissionGraph::Pass destination = submission_for_pass(consumer);
            if (destination != FrameSubmissionGraph::InvalidPass &&
                submissions.queue(destination) != m_passes[consumer].queue)
            {
                error = "Frame image graph: pass queue does not match its mapped submission";
                return false;
            }
            if (source == FrameSubmissionGraph::InvalidPass ||
                destination == FrameSubmissionGraph::InvalidPass || source == destination)
                continue;
            if (!submissions.depends_on(destination, source, error))
                return false;
        }
    }
    const auto add_external_wait = [&](Pass pass, FrameSubmissionGraph::Queue queue,
        uint64_t submission) {
        const FrameSubmissionGraph::Pass consumer = submission_for_pass(pass);
        if (consumer == FrameSubmissionGraph::InvalidPass)
        {
            error = "Frame image graph: an imported resource with an external producer maps to no submission";
            return false;
        }
        if (submissions.queue(consumer) != m_passes[pass].queue)
        {
            error = "Frame image graph: imported resource consumer queue does not match its submission";
            return false;
        }
        return submissions.depends_on_submission(consumer, queue, submission, error);
    };
    const auto add_in_graph_wait = [&](Pass pass, FrameSubmissionGraph::Pass producer,
        FrameSubmissionGraph::Queue producer_queue, VkSharingMode sharing_mode,
        const std::vector<uint32_t>& sharing_queue_families) {
        const FrameSubmissionGraph::Pass consumer = submission_for_pass(pass);
        if (consumer == FrameSubmissionGraph::InvalidPass)
        {
            error = "Frame image graph: an imported resource with an in-frame producer maps to no submission";
            return false;
        }
        if (producer == FrameSubmissionGraph::InvalidPass)
            return true;
        if (submissions.queue(producer) != producer_queue)
        {
            error = "Frame image graph: imported resource producer queue does not match its submission";
            return false;
        }
        if (consumer == producer)
            return true;
        const uint32_t producer_family = submissions.queue_family(producer);
        const uint32_t consumer_family = submissions.queue_family(consumer);
        if (producer_family != FrameSubmissionGraph::InvalidQueueFamily &&
            consumer_family != FrameSubmissionGraph::InvalidQueueFamily)
        {
            if (sharing_mode == VK_SHARING_MODE_EXCLUSIVE && producer_family != consumer_family)
            {
                if (handoff_mode == ProducerHandoffMode::DeferToGeneratedGroups)
                    return true;
                error = "Frame image graph: exclusive imported resource needs a caller release between "
                    "its in-frame producer and consumer families";
                return false;
            }
            if (sharing_mode == VK_SHARING_MODE_CONCURRENT)
            {
                const auto contains_family = [&sharing_queue_families](uint32_t family) {
                    return std::find(sharing_queue_families.begin(), sharing_queue_families.end(), family) !=
                        sharing_queue_families.end();
                };
                if (!contains_family(producer_family) || !contains_family(consumer_family))
                {
                    error = "Frame image graph: imported resource sharing list omits an "
                        "in-frame producer or consumer family";
                    return false;
                }
            }
        }
        return submissions.depends_on(consumer, producer, error);
    };
    for (const Image& image : m_images)
    {
        if (image.initial_producer_pass != FrameSubmissionGraph::InvalidPass)
        {
            for (Pass reader : image.versions.front().readers)
                if (!add_in_graph_wait(reader, image.initial_producer_pass, image.initial_queue,
                        image.sharing_mode, image.sharing_queue_families))
                    return false;
            if (image.versions.size() > 1 &&
                !add_in_graph_wait(image.versions[1].producer, image.initial_producer_pass,
                    image.initial_queue, image.sharing_mode, image.sharing_queue_families))
                return false;
        }
        if (!image.initial_submission)
            continue;
        for (Pass reader : image.versions.front().readers)
            if (!add_external_wait(reader, image.initial_queue, image.initial_submission))
                return false;
        if (image.versions.size() > 1 &&
            !add_external_wait(image.versions[1].producer, image.initial_queue,
                image.initial_submission))
            return false;
    }
    for (const Buffer& buffer : m_buffers)
    {
        if (buffer.initial_producer_pass != FrameSubmissionGraph::InvalidPass)
        {
            for (Pass reader : buffer.versions.front().readers)
                if (!add_in_graph_wait(reader, buffer.initial_producer_pass, buffer.initial_queue,
                        buffer.sharing_mode, buffer.sharing_queue_families))
                    return false;
            if (buffer.versions.size() > 1 &&
                !add_in_graph_wait(buffer.versions[1].producer, buffer.initial_producer_pass,
                    buffer.initial_queue, buffer.sharing_mode, buffer.sharing_queue_families))
                return false;
        }
        if (!buffer.initial_submission)
            continue;
        for (Pass reader : buffer.versions.front().readers)
            if (!add_external_wait(reader, buffer.initial_queue, buffer.initial_submission))
                return false;
        if (buffer.versions.size() > 1 &&
            !add_external_wait(buffer.versions[1].producer, buffer.initial_queue,
                buffer.initial_submission))
            return false;
    }
    std::vector<FrameSubmissionGraph::Pass> submission_order;
    if (!submissions.execution_order(submission_order, error))
        return false;
    error.clear();
    return true;
}

bool FrameImageGraph::connect_submission_dependencies(FrameSubmissionGraph& submissions,
    const SubmissionForQueue& submission_for_queue,
    const QueueFamilyForQueue& queue_family_for_queue, std::string& error) const
{
    if (!submission_for_queue || !queue_family_for_queue)
    {
        error = "Frame image graph: queue family or submission mapping is unavailable";
        return false;
    }
    const SubmissionForPass submission_for_pass = [&](Pass pass) {
        return pass < m_passes.size() ? submission_for_queue(m_passes[pass].queue) :
            FrameSubmissionGraph::InvalidPass;
    };
    return connect_submission_dependencies(submissions, submission_for_pass,
        queue_family_for_queue, error);
}

bool FrameImageGraph::connect_submission_dependencies(FrameSubmissionGraph& submissions,
    const SubmissionForPass& submission_for_pass,
    const QueueFamilyForQueue& queue_family_for_queue, std::string& error) const
{
    if (!submission_for_pass || !queue_family_for_queue)
    {
        error = "Frame image graph: queue family or submission mapping is unavailable";
        return false;
    }

    const auto queue_family_for_pass = [&](Pass pass) {
        if (pass >= m_passes.size())
            return FrameSubmissionGraph::InvalidQueueFamily;
        const FrameSubmissionGraph::Pass submission = submission_for_pass(pass);
        if (submission != FrameSubmissionGraph::InvalidPass)
        {
            const uint32_t family = submissions.queue_family(submission);
            if (family != FrameSubmissionGraph::InvalidQueueFamily)
                return family;
        }
        return queue_family_for_queue(m_passes[pass].queue);
    };

    const auto validate_resource_families = [&](VkSharingMode sharing_mode,
        const std::vector<uint32_t>& sharing_queue_families,
        FrameSubmissionGraph::Queue initial_queue, uint64_t initial_submission,
        FrameSubmissionGraph::Pass initial_producer_pass,
        const std::vector<Version>& versions, const char* resource_kind, size_t resource_index) {
        std::set<uint32_t> families;
        const auto include_queue = [&](FrameSubmissionGraph::Queue queue) {
            const uint32_t family = queue_family_for_queue(queue);
            if (family == VK_QUEUE_FAMILY_IGNORED)
            {
                error = "Frame image graph: queue family is unavailable for " +
                    std::string(resource_kind) + " " + std::to_string(resource_index);
                return false;
            }
            families.insert(family);
            return true;
        };
        const auto include_pass = [&](Pass pass) {
            const uint32_t family = queue_family_for_pass(pass);
            if (family == VK_QUEUE_FAMILY_IGNORED ||
                family == FrameSubmissionGraph::InvalidQueueFamily)
            {
                error = "Frame image graph: queue family is unavailable for " +
                    std::string(resource_kind) + " " + std::to_string(resource_index);
                return false;
            }
            families.insert(family);
            return true;
        };
        if ((initial_submission || initial_producer_pass != FrameSubmissionGraph::InvalidPass) &&
            !include_queue(initial_queue))
            return false;
        for (const Version& version : versions)
        {
            if (version.producer != ~0u &&
                !include_pass(version.producer))
                return false;
            for (Pass reader : version.readers)
                if (!include_pass(reader))
                    return false;
        }
        if (families.size() > 1 && sharing_mode != VK_SHARING_MODE_CONCURRENT)
        {
            error = "Frame image graph: " + std::string(resource_kind) + " " +
                std::to_string(resource_index) +
                " crosses queue families but was created with exclusive sharing";
            return false;
        }
        if (sharing_mode == VK_SHARING_MODE_CONCURRENT)
        {
            const std::set<uint32_t> shared_families(sharing_queue_families.begin(),
                sharing_queue_families.end());
            for (uint32_t family : families)
                if (shared_families.find(family) == shared_families.end())
                {
                    error = "Frame image graph: " + std::string(resource_kind) + " " +
                        std::to_string(resource_index) +
                        " uses a queue family outside its concurrent-sharing list";
                    return false;
                }
        }
        return true;
    };

    for (size_t i = 0; i < m_images.size(); ++i)
    {
        const Image& image = m_images[i];
        if (!validate_resource_families(image.sharing_mode, image.sharing_queue_families,
                image.initial_queue,
                image.initial_submission, image.initial_producer_pass,
                image.versions, "image", i))
            return false;
    }
    for (size_t i = 0; i < m_images.size(); ++i)
    {
        const Image& first = m_images[i];
        for (size_t j = i + 1; j < m_images.size(); ++j)
        {
            const Image& second = m_images[j];
            if (first.image != second.image || !image_subresources_overlap(first.range, second.range))
                continue;
            std::set<uint32_t> families;
            const auto collect_families = [&](const Image& image) {
                const auto include_queue = [&](FrameSubmissionGraph::Queue queue) {
                    const uint32_t family = queue_family_for_queue(queue);
                    if (family == VK_QUEUE_FAMILY_IGNORED)
                    {
                        error = "Frame image graph: queue family is unavailable for overlapping image subresources";
                        return false;
                    }
                    families.insert(family);
                    return true;
                };
                const auto include_pass = [&](Pass pass) {
                    const uint32_t family = queue_family_for_pass(pass);
                    if (family == VK_QUEUE_FAMILY_IGNORED ||
                        family == FrameSubmissionGraph::InvalidQueueFamily)
                    {
                        error = "Frame image graph: queue family is unavailable for overlapping image subresources";
                        return false;
                    }
                    families.insert(family);
                    return true;
                };
                if ((image.initial_submission || image.initial_producer_pass !=
                        FrameSubmissionGraph::InvalidPass) && !include_queue(image.initial_queue))
                    return false;
                for (const Version& version : image.versions)
                {
                    if (version.producer != ~0u &&
                        !include_pass(version.producer))
                        return false;
                    for (Pass reader : version.readers)
                        if (!include_pass(reader))
                            return false;
                }
                return true;
            };
            if (!collect_families(first) || !collect_families(second))
                return false;
            if (families.size() > 1 && first.sharing_mode != VK_SHARING_MODE_CONCURRENT)
            {
                error = "Frame image graph: overlapping image subresources cross queue families "
                    "but were created with exclusive sharing";
                return false;
            }
        }
    }
    for (size_t i = 0; i < m_buffers.size(); ++i)
    {
        const Buffer& buffer = m_buffers[i];
        if (!validate_resource_families(buffer.sharing_mode, buffer.sharing_queue_families,
                buffer.initial_queue,
                buffer.initial_submission, buffer.initial_producer_pass,
                buffer.versions, "buffer", i))
            return false;
    }
    for (size_t i = 0; i < m_buffers.size(); ++i)
    {
        const Buffer& first = m_buffers[i];
        for (size_t j = i + 1; j < m_buffers.size(); ++j)
        {
            const Buffer& second = m_buffers[j];
            if (first.buffer != second.buffer || first.offset >= second.offset + second.size ||
                second.offset >= first.offset + first.size)
                continue;
            std::set<uint32_t> families;
            const auto collect_families = [&](const Buffer& buffer) {
                const auto include_queue = [&](FrameSubmissionGraph::Queue queue) {
                    const uint32_t family = queue_family_for_queue(queue);
                    if (family == VK_QUEUE_FAMILY_IGNORED)
                    {
                        error = "Frame image graph: queue family is unavailable for overlapping buffer ranges";
                        return false;
                    }
                    families.insert(family);
                    return true;
                };
                const auto include_pass = [&](Pass pass) {
                    const uint32_t family = queue_family_for_pass(pass);
                    if (family == VK_QUEUE_FAMILY_IGNORED ||
                        family == FrameSubmissionGraph::InvalidQueueFamily)
                    {
                        error = "Frame image graph: queue family is unavailable for overlapping buffer ranges";
                        return false;
                    }
                    families.insert(family);
                    return true;
                };
                if ((buffer.initial_submission || buffer.initial_producer_pass !=
                        FrameSubmissionGraph::InvalidPass) && !include_queue(buffer.initial_queue))
                    return false;
                for (const Version& version : buffer.versions)
                {
                    if (version.producer != ~0u &&
                        !include_pass(version.producer))
                        return false;
                    for (Pass reader : version.readers)
                        if (!include_pass(reader))
                            return false;
                }
                return true;
            };
            if (!collect_families(first) || !collect_families(second))
                return false;
            if (families.size() > 1 && first.sharing_mode != VK_SHARING_MODE_CONCURRENT)
            {
                error = "Frame image graph: overlapping buffer ranges cross queue families "
                    "but were created with exclusive sharing";
                return false;
            }
        }
    }
    return connect_submission_dependencies(submissions, submission_for_pass, error);
}

bool FrameImageGraph::begin_execution(ExecutionState& state, std::string& error) const
{
    state = {};
    for (const Image& image : m_images)
        if (!image.image && (image.versions.size() > 1 || !image.versions.front().readers.empty()))
        {
            error = "Frame image graph: used transient image has not been materialized";
            return false;
        }
    for (const Buffer& buffer : m_buffers)
        if (!buffer.buffer && (buffer.versions.size() > 1 || !buffer.versions.front().readers.empty()))
        {
            error = "Frame image graph: used transient buffer has not been materialized";
            return false;
        }
    try
    {
        for (const Image& image : m_images)
            if (image.allocation &&
                !state.transient_images.register_image(image.image, image.range, ImageUse::Undefined, error))
            {
                state = {};
                return false;
            }
        for (size_t i = 0; i < m_buffers.size(); ++i)
        {
            const VkBuffer handle = m_buffers[i].buffer;
            if (!handle || std::any_of(m_buffers.begin(), m_buffers.begin() + i,
                    [handle](const Buffer& existing) { return existing.buffer == handle; }))
                continue;
            std::vector<VkDeviceSize> boundaries;
            for (const Buffer& buffer : m_buffers)
                if (buffer.buffer == handle)
                {
                    boundaries.push_back(buffer.offset);
                    boundaries.push_back(buffer.offset + buffer.size);
                }
            std::sort(boundaries.begin(), boundaries.end());
            boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
            for (size_t boundary = 1; boundary < boundaries.size(); ++boundary)
            {
                const VkDeviceSize begin = boundaries[boundary - 1];
                const VkDeviceSize end = boundaries[boundary];
                const auto owner = std::find_if(m_buffers.begin(), m_buffers.end(),
                    [handle, begin, end](const Buffer& buffer) {
                        return buffer.buffer == handle && buffer.offset <= begin &&
                            buffer.offset + buffer.size >= end;
                    });
                if (owner != m_buffers.end())
                    state.buffers.push_back({handle, begin, end - begin,
                        owner->initial_use, owner->initial_queue});
            }
        }
    }
    catch (...)
    {
        state = {};
        error = "Frame image graph: execution state allocation failed";
        return false;
    }
    state.graph = this;
    error.clear();
    return true;
}

bool FrameImageGraph::execute(VkCommandBuffer command, ImageStateTracker& states,
    const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags,
    bool semaphore_handoff, FrameSubmissionGraph::Queue recording_queue) const
{
    for (const PassInfo& pass : m_passes)
    {
        const bool records_commands = bool(pass.record) || bool(pass.checked_record) ||
            bool(pass.command_record) || bool(pass.checked_command_record);
        if (records_commands && pass.queue != recording_queue)
        {
            error = "Frame image graph: pass '" + pass.name +
                "' records into a command buffer for a different queue class";
            return false;
        }
    }

    std::vector<QueueGroup> groups;
    if (!build_queue_groups(groups, error))
        return false;
    ExecutionState state;
    if (!begin_execution(state, error))
        return false;
    for (const QueueGroup& group : groups)
        if (group.queue == recording_queue &&
            !execute_group(group, command, states, state, vk, error, queue_flags,
                semaphore_handoff))
            return false;
    error.clear();
    return true;
}

bool FrameImageGraph::execute_group(const QueueGroup& group, VkCommandBuffer command,
    ImageStateTracker& states, ExecutionState& state, const ImageStateDispatch& vk,
    std::string& error, VkQueueFlags queue_flags, bool semaphore_handoff) const
{
    const bool external_release_only = group.passes.empty() &&
        !group.ownership_releases.empty() && group.ownership_acquires.empty() &&
        std::all_of(group.ownership_releases.begin(), group.ownership_releases.end(),
            [](const QueueGroup::OwnershipTransfer& transfer) {
                return transfer.source_pass == ~0u;
            });
    if (state.graph != this || (group.passes.empty() && !external_release_only))
    {
        error = "Frame image graph: execution state or command group is invalid";
        return false;
    }
    for (Pass pass : group.passes)
        if (pass >= m_passes.size() || m_passes[pass].queue != group.queue)
        {
            error = "Frame image graph: command group contains an invalid pass";
            return false;
        }
    if (group.queue_family == FrameSubmissionGraph::InvalidQueueFamily &&
        (!group.ownership_releases.empty() || !group.ownership_acquires.empty()))
    {
        error = "Frame image graph: command group has no queue-family ownership metadata";
        return false;
    }
    return execute_internal(command, states, state, vk, error, queue_flags,
        semaphore_handoff, group.queue, &group);
}

bool FrameImageGraph::execute_internal(VkCommandBuffer command, ImageStateTracker& states,
    ExecutionState& state, const ImageStateDispatch& vk, std::string& error,
    VkQueueFlags queue_flags, bool semaphore_handoff,
    FrameSubmissionGraph::Queue recording_queue,
    const QueueGroup* selected_group) const
{
    if (state.graph != this)
    {
        error = "Frame image graph: execution state belongs to a different graph";
        return false;
    }
    const auto image_tracker = [&](const Image& image) -> ImageStateTracker& {
        return image.allocation ? state.transient_images : states;
    };
    std::vector<std::set<Pass>> edges;
    std::vector<Pass> order;
    if (!build_pass_graph(edges, order, error))
        return false;
    if (command == VK_NULL_HANDLE ||
        (!vk.cmd_pipeline_barrier && (!m_images.empty() || !m_buffers.empty())))
    {
        error = "Vulkan image transition procedures are unavailable";
        return false;
    }
    const auto selected = [&](Pass pass) {
        if (!selected_group)
            return m_passes[pass].queue == recording_queue;
        return std::find(selected_group->passes.begin(), selected_group->passes.end(), pass) !=
            selected_group->passes.end();
    };
    if (selected_group && selected_group->queue != recording_queue)
    {
        error = "Frame image graph: command group queue does not match its recording queue";
        return false;
    }
    for (size_t pass_index = 0; pass_index < m_passes.size(); ++pass_index)
    {
        const PassInfo& pass = m_passes[pass_index];
        const bool records_commands = bool(pass.record) || bool(pass.checked_record) ||
            bool(pass.command_record) || bool(pass.checked_command_record);
        const bool records_here = selected_group ? selected(static_cast<Pass>(pass_index)) :
            pass.queue == recording_queue;
        if (!selected_group && records_commands && pass.queue != recording_queue)
        {
            error = "Frame image graph: pass '" + pass.name +
                "' records into a command buffer for a different queue class";
            return false;
        }
        if (records_here && records_commands)
        {
            VkQueueFlags required_flags = 0;
            switch (pass.queue)
            {
            case FrameSubmissionGraph::Queue::Graphics: required_flags = VK_QUEUE_GRAPHICS_BIT; break;
            case FrameSubmissionGraph::Queue::Compute: required_flags = VK_QUEUE_COMPUTE_BIT; break;
            case FrameSubmissionGraph::Queue::Copy:
                required_flags = VK_QUEUE_TRANSFER_BIT | VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
                break;
            }
            if (required_flags && !(queue_flags & required_flags))
            {
                error = "Frame image graph: pass '" + pass.name +
                    "' records commands on an incompatible queue family";
                return false;
            }
        }
        // This command buffer only owns work from recording_queue. Other
        // queue-class accesses are validated when their own command group is
        // recorded; their ordering is carried by FrameSubmissionGraph waits.
        if (!records_here)
            continue;
        const auto validate_image_use = [&](const Access& access) {
            if (image_use_supported_on_queue(access.use, queue_flags) &&
                (access.final_use == ImageUse::Undefined ||
                    image_use_supported_on_queue(access.final_use, queue_flags)))
                return true;
            error = "Frame image graph: image use in pass '" + pass.name +
                "' is unsupported by this queue family";
            return false;
        };
        for (const Access& access : pass.reads)
            if (!validate_image_use(access))
                return false;
        for (const Access& access : pass.writes)
            if (!validate_image_use(access))
                return false;
        const auto validate_queue_use = [&](const BufferAccess& access) {
            if (supported_on_queue(access.use, queue_flags))
                return true;
            const BufferUseInfo use = describe(access.use);
            if (semaphore_handoff && !use.writes)
                return true;
            error = "Frame image graph: buffer use is unsupported by this queue family";
            return false;
        };
        for (const BufferAccess& access : pass.buffer_reads)
            if (!validate_queue_use(access))
                return false;
        for (const BufferAccess& access : pass.buffer_writes)
            if (!validate_queue_use(access))
                return false;
    }
    for (Pass pass : order)
    {
        const PassInfo& info = m_passes[pass];
        if (!selected(pass))
            continue;
        for (const Access& access : info.reads)
        {
            const Image& image = m_images[access.resource.index];
            if (!image_tracker(image).can_transition(image.image, image.range, access.use, error))
                return false;
        }
        for (const Access& access : info.writes)
        {
            const Image& image = m_images[access.resource.index];
            if (!image_tracker(image).can_transition(image.image, image.range, access.use, error))
                return false;
        }
    }
    std::set<const QueueGroup::OwnershipTransfer*> acquired_image_transfers;
    const auto transition_image = [&](Pass pass, const Access& access) {
        const Image& image = m_images[access.resource.index];
        if (selected_group)
        {
            for (const QueueGroup::OwnershipTransfer& transfer : selected_group->ownership_acquires)
            {
                if (transfer.kind != QueueGroup::OwnershipTransfer::Kind::Image ||
                    transfer.destination_pass != pass || transfer.resource_id != access.resource.index ||
                    transfer.destination_image_use != access.use ||
                    acquired_image_transfers.find(&transfer) != acquired_image_transfers.end())
                    continue;
                const VkImageSubresourceRange& range = transfer.image_range.levelCount &&
                    transfer.image_range.layerCount ? transfer.image_range : image.range;
                const bool contained = (range.aspectMask & image.range.aspectMask) == range.aspectMask &&
                    range.baseMipLevel >= image.range.baseMipLevel &&
                    uint64_t(range.baseMipLevel) + range.levelCount <=
                        uint64_t(image.range.baseMipLevel) + image.range.levelCount &&
                    range.baseArrayLayer >= image.range.baseArrayLayer &&
                    uint64_t(range.baseArrayLayer) + range.layerCount <=
                        uint64_t(image.range.baseArrayLayer) + image.range.layerCount;
                if (!contained || transfer.destination_family != selected_group->queue_family ||
                    !image_use_supported_on_queue(access.use, queue_flags))
                {
                    error = "Frame image graph: image ownership acquire does not match its destination pass";
                    return false;
                }
                if (!image_tracker(image).acquire_ownership(command, image.image, range,
                        transfer.source_image_use, transfer.destination_image_use,
                        transfer.source_family, transfer.destination_family, vk, error, queue_flags))
                    return false;
                acquired_image_transfers.insert(&transfer);
            }
        }
        return image_tracker(image).transition(command, image.image, image.range, access.use, vk, error,
            queue_flags, semaphore_handoff);
    };
    std::set<std::pair<const QueueGroup::OwnershipTransfer*, VkDeviceSize>> acquired_buffer_segments;
    const auto transition_buffer = [&](Pass pass, const BufferAccess& access) {
        const Buffer& buffer = m_buffers[access.resource.index];
        const VkDeviceSize end = buffer.offset + buffer.size;
        std::vector<const QueueGroup::OwnershipTransfer*> transfers;
        if (selected_group)
        {
            for (const QueueGroup::OwnershipTransfer& transfer : selected_group->ownership_acquires)
            {
                if (transfer.kind != QueueGroup::OwnershipTransfer::Kind::Buffer ||
                    transfer.destination_pass != pass || transfer.resource_id != access.resource.index ||
                    transfer.destination_buffer_use != access.use)
                    continue;
                if (!transfer.buffer_size || transfer.buffer_offset >
                        std::numeric_limits<VkDeviceSize>::max() - transfer.buffer_size ||
                    transfer.buffer_offset < buffer.offset ||
                    transfer.buffer_offset + transfer.buffer_size > end ||
                    transfer.destination_family != selected_group->queue_family ||
                    !supported_on_queue(access.use, queue_flags))
                {
                    error = "Frame image graph: buffer ownership acquire does not match its destination pass";
                    return false;
                }
                transfers.push_back(&transfer);
            }
        }
        std::vector<VkDeviceSize> acquired_bytes(transfers.size(), 0);
        bool found_segment = false;
        for (BufferStateSegment& segment : state.buffers)
        {
            if (segment.buffer != buffer.buffer || segment.offset >= end ||
                buffer.offset >= segment.offset + segment.size)
                continue;
            found_segment = true;
            const VkDeviceSize segment_end = segment.offset + segment.size;
            size_t transfer_index = transfers.size();
            for (size_t index = 0; index < transfers.size(); ++index)
            {
                const QueueGroup::OwnershipTransfer& transfer = *transfers[index];
                const VkDeviceSize transfer_end = transfer.buffer_offset + transfer.buffer_size;
                if (segment.offset >= transfer_end || transfer.buffer_offset >= segment_end)
                    continue;
                if (transfer_index != transfers.size() || segment.offset < transfer.buffer_offset ||
                    segment_end > transfer_end)
                {
                    error = "Frame image graph: buffer ownership acquire range does not match tracked segments";
                    return false;
                }
                transfer_index = index;
            }

            if (transfer_index != transfers.size())
            {
                const QueueGroup::OwnershipTransfer& transfer = *transfers[transfer_index];
                acquired_bytes[transfer_index] += segment.size;
                if (acquired_buffer_segments.insert({&transfer, segment.offset}).second)
                {
                    if (segment.use != transfer.source_buffer_use ||
                        segment.queue_family != transfer.source_family ||
                        segment.pending_destination_family != transfer.destination_family)
                    {
                        error = "Frame image graph: buffer acquire has no matching release state";
                        return false;
                    }
                    const BufferUseInfo destination = describe(transfer.destination_buffer_use);
                    VkBufferMemoryBarrier barrier{};
                    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                    barrier.srcAccessMask = 0;
                    barrier.dstAccessMask = destination.access;
                    barrier.srcQueueFamilyIndex = transfer.source_family;
                    barrier.dstQueueFamilyIndex = transfer.destination_family;
                    barrier.buffer = segment.buffer;
                    barrier.offset = segment.offset;
                    barrier.size = segment.size;
                    vk.cmd_pipeline_barrier(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        destination.stage, 0, 0, nullptr, 1, &barrier, 0, nullptr);
                    segment.use = transfer.destination_buffer_use;
                    segment.queue = recording_queue;
                    segment.queue_family = transfer.destination_family;
                    segment.pending_destination_family = FrameSubmissionGraph::InvalidQueueFamily;
                }
            }
            else if (segment.pending_destination_family != FrameSubmissionGraph::InvalidQueueFamily)
            {
                error = "Frame image graph: buffer access is missing its ownership acquire";
                return false;
            }

            BufferUse& previous = segment.use;
            const BufferUseInfo before = describe(previous), after = describe(access.use);
            if (semaphore_handoff && !supported_on_queue(previous, queue_flags))
            {
                // The semaphore wait carries the cross-queue dependency. A
                // command buffer must not encode a source stage that its queue
                // family does not support, whether the next use reads or writes.
                previous = access.use;
                segment.queue = recording_queue;
                if (selected_group)
                    segment.queue_family = selected_group->queue_family;
                continue;
            }
            if (!supported_on_queue(access.use, queue_flags) && semaphore_handoff && !after.writes)
            {
                // The following queue waits on the signal semaphore. That wait is
                // the cross-queue memory dependency; do not encode a graphics-only
                // pipeline stage in a compute-only command buffer.
                previous = access.use;
                segment.queue = recording_queue;
                if (selected_group)
                    segment.queue_family = selected_group->queue_family;
                continue;
            }
            if (previous != access.use || after.writes)
            {
                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = before.access;
                barrier.dstAccessMask = after.access;
                barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                barrier.buffer = segment.buffer;
                barrier.offset = segment.offset;
                barrier.size = segment.size;
                vk.cmd_pipeline_barrier(command, before.stage, after.stage, 0, 0, nullptr,
                    1, &barrier, 0, nullptr);
            }
            previous = access.use;
            segment.queue = recording_queue;
            if (selected_group)
                segment.queue_family = selected_group->queue_family;
        }
        if (!transfers.empty() && !found_segment)
        {
            error = "Frame image graph: buffer ownership acquire has no tracked range";
            return false;
        }
        for (size_t index = 0; index < transfers.size(); ++index)
            if (acquired_bytes[index] != transfers[index]->buffer_size)
            {
                error = "Frame image graph: buffer ownership acquire range is not fully tracked";
                return false;
            }
        return true;
    };
    const auto release_group_ownership = [&](Pass pass) {
        if (!selected_group)
            return true;
        for (const QueueGroup::OwnershipTransfer& transfer : selected_group->ownership_releases)
        {
            if (transfer.source_pass != pass)
                continue;
            if (transfer.source_family != selected_group->queue_family)
            {
                error = "Frame image graph: ownership release does not match its source pass";
                return false;
            }
            if (transfer.kind == QueueGroup::OwnershipTransfer::Kind::Image)
            {
                if (!image_use_supported_on_queue(transfer.source_image_use, queue_flags))
                {
                    error = "Frame image graph: image ownership release source use is unsupported by its queue family";
                    return false;
                }
                const Image& image = m_images[transfer.resource_id];
                const VkImageSubresourceRange& range = transfer.image_range.levelCount &&
                    transfer.image_range.layerCount ? transfer.image_range : image.range;
                if (!image_tracker(image).release_ownership(command, image.image, range,
                        transfer.source_image_use, transfer.destination_image_use,
                        transfer.source_family, transfer.destination_family, vk, error, queue_flags))
                    return false;
                continue;
            }

            const Buffer& buffer = m_buffers[transfer.resource_id];
            if (!supported_on_queue(transfer.source_buffer_use, queue_flags))
            {
                error = "Frame image graph: buffer ownership release source use is unsupported by its queue family";
                return false;
            }
            if (!transfer.buffer_size || transfer.buffer_offset >
                    std::numeric_limits<VkDeviceSize>::max() - transfer.buffer_size ||
                transfer.buffer_offset < buffer.offset ||
                transfer.buffer_offset + transfer.buffer_size > buffer.offset + buffer.size)
            {
                error = "Frame image graph: buffer ownership release range is outside its import";
                return false;
            }
            const VkDeviceSize begin = transfer.buffer_offset;
            const VkDeviceSize end = transfer.buffer_offset + transfer.buffer_size;
            bool found_segment = false;
            VkDeviceSize released_bytes = 0;
            for (BufferStateSegment& segment : state.buffers)
            {
                if (segment.buffer != buffer.buffer || segment.offset >= end ||
                    begin >= segment.offset + segment.size)
                    continue;
                found_segment = true;
                const VkDeviceSize segment_end = segment.offset + segment.size;
                if (segment.offset < begin || segment_end > end)
                {
                    error = "Frame image graph: buffer ownership release range does not match tracked segments";
                    return false;
                }
                if (segment.queue_family == FrameSubmissionGraph::InvalidQueueFamily &&
                    segment.queue == selected_group->queue)
                    segment.queue_family = transfer.source_family;
                if (segment.use != transfer.source_buffer_use ||
                    segment.queue_family != transfer.source_family ||
                    segment.pending_destination_family != FrameSubmissionGraph::InvalidQueueFamily)
                {
                    error = "Frame image graph: buffer release source state does not match its tracked state";
                    return false;
                }
                released_bytes += segment.size;
                const BufferUseInfo before = describe(transfer.source_buffer_use);
                VkBufferMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
                barrier.srcAccessMask = before.access;
                barrier.dstAccessMask = 0;
                barrier.srcQueueFamilyIndex = transfer.source_family;
                barrier.dstQueueFamilyIndex = transfer.destination_family;
                barrier.buffer = segment.buffer;
                barrier.offset = segment.offset;
                barrier.size = segment.size;
                vk.cmd_pipeline_barrier(command, before.stage, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                    0, 0, nullptr, 1, &barrier, 0, nullptr);
                segment.pending_destination_family = transfer.destination_family;
            }
            if (!found_segment)
            {
                error = "Frame image graph: buffer ownership release has no tracked range";
                return false;
            }
            if (released_bytes != transfer.buffer_size)
            {
                error = "Frame image graph: buffer ownership release range is not fully tracked";
                return false;
            }
        }
        return true;
    };
    if (selected_group && selected_group->passes.empty() &&
        !release_group_ownership(~0u))
        return false;
    for (Pass pass : order)
    {
        const PassInfo& info = m_passes[pass];
        if (!selected(pass))
            continue;
        for (const Access& access : info.reads)
            if (!transition_image(pass, access))
                return false;
        for (const BufferAccess& access : info.buffer_reads)
            if (!transition_buffer(pass, access))
                return false;
        for (const Access& access : info.writes)
            if (!transition_image(pass, access))
                return false;
        for (const BufferAccess& access : info.buffer_writes)
            if (!transition_buffer(pass, access))
                return false;
        if (info.checked_command_record)
        {
            if (!info.checked_command_record(command, error))
            {
                if (error.empty()) error = "Frame image graph: pass recording failed: " + info.name;
                return false;
            }
        }
        else if (info.command_record) info.command_record(command);
        else if (info.checked_record)
        {
            if (!info.checked_record(error))
            {
                if (error.empty()) error = "Frame image graph: pass recording failed: " + info.name;
                return false;
            }
        }
        else if (info.record) info.record();
        for (const Access& access : info.writes)
        {
            if (access.final_use == ImageUse::Undefined)
                continue;
            const Image& image = m_images[access.resource.index];
            if (!image_tracker(image).set_render_pass_final_state(image.image, image.range,
                    access.final_use, error))
                return false;
        }
        if (!release_group_ownership(pass))
            return false;
    }
    return true;
}
}
