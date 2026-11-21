#pragma once

#include "ImageStateTracker.h"
#include "FrameSubmissionGraph.h"

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
enum class BufferUse
{
    Undefined, HostWrite, HostRead, TransferSource, TransferDestination,
    ComputeRead, ComputeWrite, ComputeReadWrite, VertexRead, FragmentRead,
    IndirectRead, AllCommandsReadWrite
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

    struct Resource
    {
        uint32_t id = ~0u;
        uint32_t version = ~0u;
    };
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
    struct BufferResource
    {
        uint32_t id = ~0u;
        uint32_t version = ~0u;
    };
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
    Pass add_pass(const char* name, Record record);
    Pass add_checked_pass(const char* name, CheckedRecord record);
    Pass add_pass(const char* name, FrameSubmissionGraph::Queue queue, Record record);
    Pass add_checked_pass(const char* name, FrameSubmissionGraph::Queue queue, CheckedRecord record);
    Pass add_command_pass(const char* name, CommandRecord record);
    Pass add_checked_command_pass(const char* name, CheckedCommandRecord record);
    Pass add_command_pass(const char* name, FrameSubmissionGraph::Queue queue, CommandRecord record);
    Pass add_checked_command_pass(const char* name, FrameSubmissionGraph::Queue queue,
        CheckedCommandRecord record);
    // Adds an ordering edge when two passes have a real execution dependency
    // but no image or buffer resource expresses it (for example a raster
    // overlay that augments a compute-produced image).
    bool depends_on(Pass after, Pass before, std::string& error);
    bool read(Pass pass, Resource resource, ImageUse use, std::string& error);
    bool write(Pass pass, Resource resource, ImageUse use, Resource& next, std::string& error);
    // initial_use is the image state expected at render-pass begin. Undefined
    // defaults to attachment_use for render passes whose initialLayout is
    // UNDEFINED or already matches the attachment layout.
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
    // Compiles the pass hazard DAG into one ordered command group per queue
    // class. Each group's pass list is recorded on that queue's command buffer.
    bool build_queue_groups(std::vector<QueueGroup>& groups, std::string& error) const;
    // Creates FrameSubmissionGraph nodes for the compiled queue groups, maps
    // every image-graph pass to its generated node, connects resource waits, and
    // derives paired ownership transfers for in-frame and known external
    // imported exclusive family changes.
    bool generate_submission_groups(FrameSubmissionGraph& submissions,
        const QueueFamilyForQueue& queue_family_for_queue,
        std::vector<QueueGroup>& groups,
        std::vector<FrameSubmissionGraph::Pass>& submission_for_pass,
        std::string& error) const;
    bool begin_execution(ExecutionState& state, std::string& error) const;
    // Converts resource hazards between mapped passes into submission edges.
    // Unmapped passes remain local to their command list.
    // Generated ownership transfers may defer exclusive imported-producer edges.
    bool connect_submission_dependencies(FrameSubmissionGraph& submissions,
        const SubmissionForPass& submission_for_pass, std::string& error,
        ProducerHandoffMode handoff_mode = ProducerHandoffMode::RequireCallerRelease) const;
    // Maps each resource-declared pass by its queue class. Passes assigned to
    // the same submission share that queue's ordering; resource hazards create
    // the cross-queue submission edges automatically.
    bool connect_submission_dependencies(FrameSubmissionGraph& submissions,
        const SubmissionForQueue& submission_for_queue, std::string& error) const;
    // Legacy wait-only mapping: resources crossing distinct queue families must
    // use concurrent sharing. Generated groups provide the ownership-transfer path.
    bool connect_submission_dependencies(FrameSubmissionGraph& submissions,
        const SubmissionForQueue& submission_for_queue,
        const QueueFamilyForQueue& queue_family_for_queue, std::string& error) const;
    // Allows each image-graph pass to map to its own submission group. The
    // group's recorded family is authoritative when available; the callback
    // supplies the family for external submissions and legacy groups. This
    // overload validates sharing only; generated groups record in-frame releases
    // and acquires for exclusive resources.
    bool connect_submission_dependencies(FrameSubmissionGraph& submissions,
        const SubmissionForPass& submission_for_pass,
        const QueueFamilyForQueue& queue_family_for_queue, std::string& error) const;
    // Records only the resource transitions and callbacks belonging to
    // recording_queue. Cross-queue pass dependencies are connected separately
    // through FrameSubmissionGraph and must be synchronized with its waits.
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
    struct Version
    {
        Pass producer = ~0u;
        std::vector<Pass> readers;
    };
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
    FrameSubmissionGraph::Queue default_queue_;
};
}
