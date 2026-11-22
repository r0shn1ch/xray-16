#pragma once

#include "ImageStateTracker.h"
#include "FrameSubmissionGraph.h"
#include "../xrRenderGraph/ResourceDependencies.h"
#include "../xrRenderGraph/ResourceRetirement.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
class FrameBufferPool;
class FrameImagePool;
enum class BufferUse
{
    Undefined, HostWrite, HostRead, TransferSource, TransferDestination,
    ComputeRead, ComputeWrite, ComputeReadWrite, VertexRead, FragmentRead,
    IndirectRead, IndirectAndComputeRead, AllCommandsReadWrite
};
// A frame-local dependency graph for imported image subresources and buffer ranges.
// Repeated and overlapping imports must agree on producer and sharing metadata
// for their shared ranges. Generated groups record in-frame exclusive ownership
// handoffs, and imported-producer handoffs when a producer pass or submission
// token and source use are declared. Unsubmitted or unspecified state remains
// caller-owned; inconsistent alias metadata is rejected at import time.
class FrameImageGraph
{
public:
    explicit FrameImageGraph(FrameSubmissionGraph::Queue default_queue =
        FrameSubmissionGraph::Queue::Graphics) : default_queue_(default_queue) {}
    FrameImageGraph(const FrameImageGraph&) = delete;
    FrameImageGraph& operator=(const FrameImageGraph&) = delete;

    struct ImageResourceKind;
    using Resource = framegraph::TypedResourceHandle<ImageResourceKind>;
    using Pass = uint32_t;
    using Record = std::function<void()>;
    using CheckedRecord = std::function<bool(std::string&)>;
    using CommandRecord = std::function<void(VkCommandBuffer)>;
    using CheckedCommandRecord = std::function<bool(VkCommandBuffer, std::string&)>;
    using SubmissionForPass = std::function<FrameSubmissionGraph::Pass(Pass)>;
    using SubmissionForQueue = std::function<FrameSubmissionGraph::Pass(FrameSubmissionGraph::Queue)>;
    using QueueFamilyForQueue = std::function<uint32_t(FrameSubmissionGraph::Queue)>;
    enum class ProducerHandoffMode : uint8_t
    {
        RequireCallerRelease,
        DeferToGeneratedGroups,
    };
    struct BufferResourceKind;
    using BufferResource = framegraph::TypedResourceHandle<BufferResourceKind>;
    struct QueueGroup
    {
        struct OwnershipTransfer
        {
            enum class Kind : uint8_t { Image, Buffer };
            Kind kind = Kind::Image;
            uint32_t resource_id = ~0u;
            Pass source_pass = ~0u;
            Pass destination_pass = ~0u;
            uint32_t source_family = FrameSubmissionGraph::InvalidQueueFamily;
            uint32_t destination_family = FrameSubmissionGraph::InvalidQueueFamily;
            ImageUse source_image_use = ImageUse::Undefined;
            ImageUse destination_image_use = ImageUse::Undefined;
            BufferUse source_buffer_use = BufferUse::Undefined;
            BufferUse destination_buffer_use = BufferUse::Undefined;
            VkImageSubresourceRange image_range{};
            VkDeviceSize buffer_offset{};
            VkDeviceSize buffer_size{};
        };

        FrameSubmissionGraph::Queue queue = FrameSubmissionGraph::Queue::Graphics;
        std::string name;
        std::vector<Pass> passes;
        uint32_t queue_family = FrameSubmissionGraph::InvalidQueueFamily;
        FrameSubmissionGraph::Pass submission = FrameSubmissionGraph::InvalidPass;
        uint64_t external_submission = 0;
        std::vector<OwnershipTransfer> ownership_releases;
        std::vector<OwnershipTransfer> ownership_acquires;
    };
    struct BufferStateSegment
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        BufferUse use = BufferUse::Undefined;
        FrameSubmissionGraph::Queue queue = FrameSubmissionGraph::Queue::Graphics;
        uint32_t queue_family = FrameSubmissionGraph::InvalidQueueFamily;
        uint32_t pending_destination_family = FrameSubmissionGraph::InvalidQueueFamily;
    };
    struct ExecutionState
    {
        const FrameImageGraph* graph = nullptr;
        std::vector<BufferStateSegment> buffers;
        ImageStateTracker transient_images;
    };

    bool import_image(VkImage image, const VkImageSubresourceRange& range,
        Resource& resource, std::string& error,
        FrameSubmissionGraph::Queue initial_queue = FrameSubmissionGraph::Queue::Graphics,
        uint64_t initial_submission = 0,
        VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE,
        std::vector<uint32_t> sharing_queue_families = {},
        ImageUse initial_use = ImageUse::Undefined,
        FrameSubmissionGraph::Pass initial_producer_pass = FrameSubmissionGraph::InvalidPass);
    bool import_buffer(VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size,
        BufferUse initial_use, BufferResource& resource, std::string& error,
        FrameSubmissionGraph::Queue initial_queue = FrameSubmissionGraph::Queue::Graphics,
        uint64_t initial_submission = 0,
        VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE,
        std::vector<uint32_t> sharing_queue_families = {},
        FrameSubmissionGraph::Pass initial_producer_pass = FrameSubmissionGraph::InvalidPass);
    struct TransientBuffer
    {
        VkBuffer handle{};
        VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
        std::vector<uint32_t> queue_families;
        std::shared_ptr<void> owner;
    };
    struct BufferDescription
    {
        VkDeviceSize size{};
        VkBufferUsageFlags usage{};
        VkMemoryPropertyFlags memory_properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        bool pooled_memory = true;
        VkQueueFlags shared_queue_flags{};
    };
    using AllocateBuffer = std::function<bool(const BufferDescription&,
        TransientBuffer&, std::string&)>;
    bool declare_buffer(const BufferDescription& description, BufferResource& resource, std::string& error);
    bool declare_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
        BufferResource& resource, std::string& error)
    {
        return declare_buffer({size, usage}, resource, error);
    }
    bool materialize_buffers(const AllocateBuffer& allocate, std::string& error);
    bool materialize_buffers(FrameBufferPool& pool, std::string& error);
    VkBuffer buffer_handle(BufferResource resource) const;
    struct TransientImage
    {
        VkImage handle{};
        std::shared_ptr<void> owner;
    };
    using AllocateImage = std::function<bool(const VkImageCreateInfo&, TransientImage&, std::string&)>;
    bool declare_image(const VkImageCreateInfo& info, VkImageAspectFlags aspects,
        Resource& resource, std::string& error);
    bool materialize_images(const AllocateImage& allocate, std::string& error);
    bool materialize_images(FrameImagePool& pool, std::string& error);
    bool materialize_resources(const AllocateBuffer& allocate_buffer, const AllocateImage& allocate_image,
        std::string& error);
    VkImage image_handle(Resource resource) const;
    // Call after recording the last use. The slots must cover every submitted use.
    void retire_allocations(framegraph::ResourceRetirement& retirement, uint32_t slots);
    Pass add_pass(const char* name, Record record);
    Pass add_checked_pass(const char* name, CheckedRecord record);
    Pass add_pass(const char* name, FrameSubmissionGraph::Queue queue, Record record);
    Pass add_checked_pass(const char* name, FrameSubmissionGraph::Queue queue, CheckedRecord record);
    Pass add_command_pass(const char* name, CommandRecord record);
    Pass add_checked_command_pass(const char* name, CheckedCommandRecord record);
    Pass add_command_pass(const char* name, FrameSubmissionGraph::Queue queue, CommandRecord record);
    Pass add_checked_command_pass(const char* name, FrameSubmissionGraph::Queue queue,
        CheckedCommandRecord record);
    // Orders passes whose dependency is outside the declared resources.
    bool depends_on(Pass after, Pass before, std::string& error);
    bool read(Pass pass, Resource resource, ImageUse use, std::string& error);
    bool write(Pass pass, Resource resource, ImageUse use, Resource& next, std::string& error);
    // Undefined initial_use defaults to attachment_use before render-pass begin.
    bool write_render_pass(Pass pass, Resource resource, ImageUse attachment_use,
        ImageUse final_use, Resource& next, std::string& error,
        ImageUse initial_use = ImageUse::Undefined);
    bool read_write(Pass pass, Resource resource, Resource& next, std::string& error,
        ImageUse use = ImageUse::StorageReadWrite);
    bool read_buffer(Pass pass, BufferResource resource, BufferUse use, std::string& error);
    bool write_buffer(Pass pass, BufferResource resource, BufferUse use,
        BufferResource& next, std::string& error);
    bool read_write_buffer(Pass pass, BufferResource resource, BufferUse use,
        BufferResource& next, std::string& error);
    // Orders command groups by resource hazards.
    bool build_queue_groups(std::vector<QueueGroup>& groups, std::string& error) const;
    struct ResourceLifetimes
    {
        // First and last use are positions in pass_order, across all queues.
        // They describe recording order. GPU reuse additionally requires completion.
        std::vector<Pass> pass_order;
        std::vector<framegraph::ResourceLifetime> images;
        std::vector<framegraph::ResourceLifetime> buffers;
    };
    bool compile_resource_lifetimes(ResourceLifetimes& lifetimes, std::string& error) const;
    // Adds submissions, cross-queue waits and ownership transfers for exclusive resources.
    bool generate_submission_groups(FrameSubmissionGraph& submissions,
        const QueueFamilyForQueue& queue_family_for_queue,
        std::vector<QueueGroup>& groups,
        std::vector<FrameSubmissionGraph::Pass>& submission_for_pass,
        std::string& error) const;
    bool begin_execution(ExecutionState& state, std::string& error) const;
    // Unmapped passes stay local. Generated transfers can defer imported-producer waits.
    bool connect_submission_dependencies(FrameSubmissionGraph& submissions,
        const SubmissionForPass& submission_for_pass, std::string& error,
        ProducerHandoffMode handoff_mode = ProducerHandoffMode::RequireCallerRelease) const;
    // Wait-only mappings require concurrent sharing across queue families.
    bool connect_submission_dependencies(FrameSubmissionGraph& submissions,
        const SubmissionForQueue& submission_for_queue,
        const QueueFamilyForQueue& queue_family_for_queue, std::string& error) const;
    // Uses each mapped submission's family when available, with a per-queue fallback.
    bool connect_submission_dependencies(FrameSubmissionGraph& submissions,
        const SubmissionForPass& submission_for_pass,
        const QueueFamilyForQueue& queue_family_for_queue, std::string& error) const;
    // Records one queue. FrameSubmissionGraph handles cross-queue waits.
    bool execute(VkCommandBuffer command, ImageStateTracker& states,
        const ImageStateDispatch& vk, std::string& error,
        VkQueueFlags queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
        bool semaphore_handoff = false,
        FrameSubmissionGraph::Queue recording_queue = FrameSubmissionGraph::Queue::Graphics) const;
    bool execute_group(const QueueGroup& group, VkCommandBuffer command,
        ImageStateTracker& states, ExecutionState& state,
        const ImageStateDispatch& vk, std::string& error, VkQueueFlags queue_flags,
        bool semaphore_handoff = false) const;

private:
    using Version = framegraph::ResourceVersion;
    struct Image
    {
        VkImage image = VK_NULL_HANDLE;
        VkImageSubresourceRange range{};
        FrameSubmissionGraph::Queue initial_queue = FrameSubmissionGraph::Queue::Graphics;
        ImageUse initial_use = ImageUse::Undefined;
        uint64_t initial_submission{};
        FrameSubmissionGraph::Pass initial_producer_pass = FrameSubmissionGraph::InvalidPass;
        VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
        std::vector<uint32_t> sharing_queue_families;
        std::vector<Version> versions{{}};
        VkImageCreateInfo transient_info{};
        std::shared_ptr<void> allocation;
    };
    struct Access
    {
        Resource resource;
        ImageUse use = ImageUse::Undefined;
        ImageUse final_use = ImageUse::Undefined;
    };
    struct Buffer
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceSize offset = 0, size = 0;
        BufferUse initial_use = BufferUse::Undefined;
        FrameSubmissionGraph::Queue initial_queue = FrameSubmissionGraph::Queue::Graphics;
        uint64_t initial_submission{};
        FrameSubmissionGraph::Pass initial_producer_pass = FrameSubmissionGraph::InvalidPass;
        VkSharingMode sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
        std::vector<uint32_t> sharing_queue_families;
        std::vector<Version> versions{{}};
        BufferDescription transient_description{};
        std::shared_ptr<void> allocation;
    };
    struct BufferAccess
    {
        BufferResource resource;
        BufferUse use = BufferUse::Undefined;
    };
    struct PassInfo
    {
        std::string name;
        FrameSubmissionGraph::Queue queue = FrameSubmissionGraph::Queue::Graphics;
        Record record;
        CheckedRecord checked_record;
        CommandRecord command_record;
        CheckedCommandRecord checked_command_record;
        std::vector<Access> reads;
        std::vector<Access> writes;
        std::vector<BufferAccess> buffer_reads;
        std::vector<BufferAccess> buffer_writes;
    };
    struct Dependency
    {
        Pass before = ~0u;
        Pass after = ~0u;
    };

    bool valid(Pass pass, Resource resource, std::string& error) const;
    bool valid_buffer(Pass pass, BufferResource resource, std::string& error) const;
    bool image_ranges_overlap(Resource first, Resource second) const;
    void add_image_alias_hazard_edges(std::vector<std::set<Pass>>& edges,
        std::vector<uint32_t>& indegree) const;
    bool buffer_ranges_overlap(BufferResource first, BufferResource second) const;
    void add_buffer_alias_hazard_edges(std::vector<std::set<Pass>>& edges,
        std::vector<uint32_t>& indegree) const;
    bool add_generated_ownership_transfers(FrameSubmissionGraph& submissions,
        std::vector<QueueGroup>& groups,
        const std::vector<FrameSubmissionGraph::Pass>& submission_for_pass,
        const QueueFamilyForQueue& queue_family_for_queue,
        std::string& error) const;
    bool build_pass_graph(std::vector<std::set<Pass>>& edges,
        std::vector<Pass>& order, std::string& error) const;
    bool execute_internal(VkCommandBuffer command, ImageStateTracker& states,
        ExecutionState& state, const ImageStateDispatch& vk, std::string& error,
        VkQueueFlags queue_flags, bool semaphore_handoff,
        FrameSubmissionGraph::Queue recording_queue,
        const QueueGroup* selected_group) const;
    std::vector<Image> m_images;
    std::vector<Buffer> m_buffers;
    std::vector<PassInfo> m_passes;
    std::vector<Dependency> m_dependencies;
    const uint64_t generation_ = framegraph::allocate_graph_generation();
    FrameSubmissionGraph::Queue default_queue_;
};
}
