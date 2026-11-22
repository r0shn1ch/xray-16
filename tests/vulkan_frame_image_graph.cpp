#include "src/Layers/xrRenderVK/FrameImageGraph.h"

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
std::vector<VkImageMemoryBarrier> barriers;
std::vector<VkBufferMemoryBarrier> buffer_barriers;
std::vector<std::pair<VkPipelineStageFlags, VkPipelineStageFlags>> buffer_stages;
void VKAPI_CALL capture(VkCommandBuffer, VkPipelineStageFlags source, VkPipelineStageFlags target,
    VkDependencyFlags, uint32_t, const VkMemoryBarrier*, uint32_t buffer_count,
    const VkBufferMemoryBarrier* buffers, uint32_t count, const VkImageMemoryBarrier* images)
{
    if (buffer_count)
    {
        buffer_barriers.insert(buffer_barriers.end(), buffers, buffers + buffer_count);
        buffer_stages.emplace_back(source, target);
    }
    if (count) barriers.insert(barriers.end(), images, images + count);
}
}

int main()
{
    using namespace xray::render::vulkan;
    const auto image = reinterpret_cast<VkImage>(uintptr_t(1));
    const auto command = reinterpret_cast<VkCommandBuffer>(uintptr_t(2));
    const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    std::string error;
    {
        FrameImageGraph virtual_graph;
        FrameImageGraph::Resource image;
        FrameImageGraph::BufferResource buffer;
        VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        info.extent = {4, 4, 1};
        info.mipLevels = info.arrayLayers = 1;
        info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        assert(virtual_graph.declare_image(info, VK_IMAGE_ASPECT_COLOR_BIT, image, error));
        assert(virtual_graph.declare_buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, buffer, error));
        const auto write = virtual_graph.add_pass("initialize virtual resources", [] { return true; });
        const auto read = virtual_graph.add_pass("read virtual resources", [] { return true; });
        FrameImageGraph::Resource written_image;
        FrameImageGraph::BufferResource written_buffer;
        assert(!virtual_graph.read(read, image, ImageUse::ComputeSampled, error));
        assert(!virtual_graph.read_write(read, image, written_image, error));
        assert(!virtual_graph.read_buffer(read, buffer, BufferUse::ComputeRead, error));
        assert(!virtual_graph.read_write_buffer(read, buffer, BufferUse::ComputeReadWrite, written_buffer, error));
        assert(virtual_graph.write(write, image, ImageUse::StorageWrite, written_image, error));
        assert(virtual_graph.write_buffer(write, buffer, BufferUse::ComputeWrite, written_buffer, error));
        assert(virtual_graph.read(read, written_image, ImageUse::ComputeSampled, error));
        assert(virtual_graph.read_buffer(read, written_buffer, BufferUse::ComputeRead, error));
        FrameImageGraph::BufferResource second_buffer;
        assert(virtual_graph.declare_buffer(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, second_buffer, error));
        assert(virtual_graph.write_buffer(write, second_buffer, BufferUse::ComputeWrite, second_buffer, error));
        std::weak_ptr<int> staged_buffer;
        unsigned attempts = 0;
        const auto allocate = [&](const FrameImageGraph::BufferDescription& description,
            FrameImageGraph::TransientBuffer& out, std::string&) {
            assert(description.size == 64 && description.usage == VK_BUFFER_USAGE_STORAGE_BUFFER_BIT &&
                description.memory_properties == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT && description.pooled_memory);
            auto owner = std::make_shared<int>(0);
            staged_buffer = owner;
            out.handle = (VkBuffer)(uintptr_t)(++attempts);
            out.owner = owner;
            return true;
        };
        assert(!virtual_graph.materialize_buffers([&](const auto& description, auto& out, auto& reason) {
            if (attempts)
                throw std::runtime_error("allocation interrupted");
            return allocate(description, out, reason);
        }, error));
        assert(staged_buffer.expired() && !virtual_graph.buffer_handle(written_buffer) &&
            !virtual_graph.buffer_handle(second_buffer));
        assert(error.find("allocation interrupted") != std::string::npos);
        assert(virtual_graph.materialize_buffers(allocate, error));
        assert(virtual_graph.buffer_handle(written_buffer) && virtual_graph.buffer_handle(second_buffer));
        const auto copy = virtual_graph.add_pass("copy unflagged image", [] { return true; });
        assert(virtual_graph.read(copy, written_image, ImageUse::TransferSource, error));
        FrameImageGraph::ResourceLifetimes lifetimes;
        assert(!virtual_graph.compile_resource_lifetimes(lifetimes, error));
        assert(error.find("transient image usage does not support pass copy unflagged image") != std::string::npos);
    }
    ImageStateTracker states;
    assert(states.register_image(image, range, ImageUse::Undefined, error));

    FrameImageGraph graph;
    FrameImageGraph::Resource initial;
    assert(graph.import_image(image, range, initial, error));
    {
        FrameImageGraph lifetime_graph;
        FrameImageGraph::Resource used, written, unused;
        const auto unused_image = reinterpret_cast<VkImage>(uintptr_t(3));
        assert(lifetime_graph.import_image(image, range, used, error));
        assert(lifetime_graph.import_image(unused_image, range, unused, error));
        const auto read = lifetime_graph.add_pass("read", [] {});
        const auto write = lifetime_graph.add_pass("write", FrameSubmissionGraph::Queue::Compute, [] {});
        assert(lifetime_graph.write(write, used, ImageUse::StorageWrite, written, error));
        assert(lifetime_graph.read(read, written, ImageUse::Sampled, error));
        FrameImageGraph::ResourceLifetimes lifetimes;
        assert(lifetime_graph.compile_resource_lifetimes(lifetimes, error));
        assert((lifetimes.pass_order == std::vector<FrameImageGraph::Pass>{write, read}));
        assert(lifetimes.images[used.index].first_use == 0 && lifetimes.images[used.index].last_use == 1);
        assert(lifetimes.images[unused.index].first_use == xray::render::framegraph::INVALID_INDEX);
    }
    {
        FrameImageGraph other_graph;
        FrameImageGraph::Resource other_image;
        assert(other_graph.import_image(image, range, other_image, error));
        const auto pass = other_graph.add_pass("read imported image", [] {});
        assert(initial.index == other_image.index && initial.version == other_image.version);
        assert(!other_graph.read(pass, initial, ImageUse::Sampled, error));
        assert(other_graph.read(pass, other_image, ImageUse::Sampled, error));
    }
    FrameImageGraph::Resource rejected_import;
    assert(graph.import_image(image, range, rejected_import, error));
    assert(rejected_import.index == initial.index && rejected_import.version == initial.version);
    const VkImageSubresourceRange two_mips{VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 0, 1};
    assert(graph.import_image(image, two_mips, rejected_import, error));
    assert(rejected_import.index != initial.index);
    const VkImageSubresourceRange next_layer{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 1, 1};
    assert(graph.import_image(image, next_layer, rejected_import, error));
    const VkImageSubresourceRange next_mip{VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1};
    assert(graph.import_image(image, next_mip, rejected_import, error));
    std::vector<int> order;
    const auto final_reader = graph.add_pass("final reader", [&] { order.push_back(4); });
    const auto consumer = graph.add_pass("consumer", [&] { order.push_back(3); });
    const auto reader = graph.add_pass("reader", [&] { order.push_back(2); });
    const auto producer = graph.add_pass("producer", [&] { order.push_back(1); });
    FrameImageGraph::Resource written;
    assert(graph.write(producer, initial, ImageUse::TransferDestination, written, error));
    FrameImageGraph::Resource imported_current;
    assert(graph.import_image(image, range, imported_current, error));
    assert(imported_current.index == written.index && imported_current.version == written.version);
    assert(graph.read(reader, written, ImageUse::TransferSource, error));
    FrameImageGraph::Resource final;
    assert(graph.write(consumer, written, ImageUse::TransferDestination, final, error));
    assert(graph.read(final_reader, final, ImageUse::Sampled, error));
    assert(graph.execute(command, states, {capture}, error));
    assert((order == std::vector<int>{1, 2, 3, 4}));
    assert(barriers.size() == 4);
    assert(barriers[0].newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(barriers[1].newLayout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

    // Explicit ordering edges cover pass pairs whose dependency is an
    // execution contract rather than a tracked image or buffer access.
    FrameImageGraph explicit_dependencies;
    std::vector<int> explicit_order;
    const auto explicit_after = explicit_dependencies.add_checked_command_pass(
        "compute consumer", FrameSubmissionGraph::Queue::Compute,
        [&explicit_order](VkCommandBuffer, std::string& pass_error) {
            explicit_order.push_back(2);
            pass_error.clear();
            return true;
        });
    const auto explicit_before = explicit_dependencies.add_checked_command_pass(
        "graphics producer", FrameSubmissionGraph::Queue::Graphics,
        [&explicit_order](VkCommandBuffer, std::string& pass_error) {
            explicit_order.push_back(1);
            pass_error.clear();
            return true;
        });
    assert(!explicit_dependencies.depends_on(explicit_after, explicit_after, error));
    assert(!explicit_dependencies.depends_on(~FrameImageGraph::Pass(0), explicit_before, error));
    assert(explicit_dependencies.depends_on(explicit_after, explicit_before, error));
    assert(explicit_dependencies.depends_on(explicit_after, explicit_before, error));
    FrameSubmissionGraph explicit_submissions;
    std::vector<FrameImageGraph::QueueGroup> explicit_groups;
    std::vector<FrameSubmissionGraph::Pass> explicit_mapping;
    assert(explicit_dependencies.generate_submission_groups(explicit_submissions,
        [](FrameSubmissionGraph::Queue queue) {
            return queue == FrameSubmissionGraph::Queue::Graphics ? 2u :
                queue == FrameSubmissionGraph::Queue::Compute ? 5u : 7u;
        }, explicit_groups, explicit_mapping, error));
    assert(explicit_groups.size() == 2);
    assert(explicit_groups[0].queue == FrameSubmissionGraph::Queue::Graphics);
    assert(explicit_groups[1].queue == FrameSubmissionGraph::Queue::Compute);
    assert(explicit_submissions.depends_on(explicit_mapping[explicit_after],
        explicit_mapping[explicit_before], error));
    FrameImageGraph::ExecutionState explicit_execution;
    ImageStateTracker explicit_states;
    assert(explicit_dependencies.begin_execution(explicit_execution, error));
    for (const FrameImageGraph::QueueGroup& group : explicit_groups)
        assert(explicit_dependencies.execute_group(group, command, explicit_states,
            explicit_execution, {}, error, group.queue == FrameSubmissionGraph::Queue::Graphics ?
                VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT : VK_QUEUE_COMPUTE_BIT));
    assert((explicit_order == std::vector<int>{1, 2}));
    assert(explicit_dependencies.depends_on(explicit_before, explicit_after, error));
    std::vector<FrameImageGraph::QueueGroup> cyclic_explicit_groups;
    assert(!explicit_dependencies.build_queue_groups(cyclic_explicit_groups, error));
    assert(error.find("cyclic") != std::string::npos);

    // VSM resolve may sample the ray-visibility image produced by RTGI on a
    // compute queue. Its graphics group must wait for that producer even
    // though the image itself is managed by the two feature passes.
    FrameImageGraph vsm_ray_dependency;
    const auto ray_visibility = vsm_ray_dependency.add_checked_command_pass(
        "ray visibility producer", FrameSubmissionGraph::Queue::Compute,
        [](VkCommandBuffer, std::string& pass_error) {
            pass_error.clear();
            return true;
        });
    const auto vsm_resolve = vsm_ray_dependency.add_checked_command_pass(
        "VSM visibility resolve", FrameSubmissionGraph::Queue::Graphics,
        [](VkCommandBuffer, std::string& pass_error) {
            pass_error.clear();
            return true;
        });
    assert(vsm_ray_dependency.depends_on(vsm_resolve, ray_visibility, error));
    FrameSubmissionGraph vsm_ray_submissions;
    std::vector<FrameImageGraph::QueueGroup> vsm_ray_groups;
    std::vector<FrameSubmissionGraph::Pass> vsm_ray_mapping;
    assert(vsm_ray_dependency.generate_submission_groups(vsm_ray_submissions,
        [](FrameSubmissionGraph::Queue queue) {
            return queue == FrameSubmissionGraph::Queue::Graphics ? 2u :
                queue == FrameSubmissionGraph::Queue::Compute ? 5u : 7u;
        }, vsm_ray_groups, vsm_ray_mapping, error));
    assert(vsm_ray_groups.size() == 2);
    assert(vsm_ray_groups[0].queue == FrameSubmissionGraph::Queue::Compute);
    assert(vsm_ray_groups[1].queue == FrameSubmissionGraph::Queue::Graphics);
    assert(vsm_ray_submissions.depends_on(vsm_ray_mapping[vsm_resolve],
        vsm_ray_mapping[ray_visibility], error));
    std::vector<FrameSubmissionGraph::Pass> vsm_ray_order;
    assert(vsm_ray_submissions.execution_order(vsm_ray_order, error));
    assert((vsm_ray_order == std::vector<FrameSubmissionGraph::Pass>{
        vsm_ray_mapping[ray_visibility], vsm_ray_mapping[vsm_resolve]}));
    uint32_t ray_to_vsm_waits = 0;
    for (const auto submission : vsm_ray_order)
        assert(vsm_ray_submissions.submit(submission,
            [submission, &vsm_ray_mapping, ray_visibility](FrameSubmissionGraph::Queue queue) {
                const bool is_ray = submission == vsm_ray_mapping[ray_visibility];
                assert(queue == (is_ray ? FrameSubmissionGraph::Queue::Compute :
                    FrameSubmissionGraph::Queue::Graphics));
                return is_ray ? uint64_t{1301} : uint64_t{1302};
            }, [&ray_to_vsm_waits](FrameSubmissionGraph::Queue destination,
                FrameSubmissionGraph::Queue source, uint64_t token) {
                assert(destination == FrameSubmissionGraph::Queue::Graphics);
                assert(source == FrameSubmissionGraph::Queue::Compute && token == 1301);
                ++ray_to_vsm_waits;
            }, error));
    assert(ray_to_vsm_waits == 1);

    // The VSM graphics pass reads the copied G-buffer depth and must follow
    // both the fixed prepass and a separately submitted motion/depth-copy group.
    const auto vsm_depth_image = reinterpret_cast<VkImage>(uintptr_t(31));
    const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    FrameImageGraph vsm_depth_graph;
    FrameImageGraph::Resource sampled_depth;
    assert(vsm_depth_graph.import_image(vsm_depth_image, depth_range, sampled_depth, error,
        FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
        ImageUse::SampledGraphicsCompute));
    const auto vsm_depth_pass = vsm_depth_graph.add_checked_command_pass("VSM depth consumer",
        FrameSubmissionGraph::Queue::Graphics,
        [](VkCommandBuffer, std::string& pass_error) {
            pass_error.clear();
            return true;
        });
    assert(vsm_depth_graph.read(vsm_depth_pass, sampled_depth,
        ImageUse::SampledGraphicsCompute, error));
    FrameSubmissionGraph vsm_depth_submissions;
    const auto vsm_prepass = vsm_depth_submissions.add_pass("graphics prepass",
        FrameSubmissionGraph::Queue::Graphics, 2);
    const auto motion_depth_copy = vsm_depth_submissions.add_pass("motion and depth copy",
        FrameSubmissionGraph::Queue::Graphics, 2);
    assert(vsm_depth_submissions.depends_on(motion_depth_copy, vsm_prepass, error));
    std::vector<FrameImageGraph::QueueGroup> vsm_depth_groups;
    std::vector<FrameSubmissionGraph::Pass> vsm_depth_mapping;
    assert(vsm_depth_graph.generate_submission_groups(vsm_depth_submissions,
        [](FrameSubmissionGraph::Queue queue) {
            return queue == FrameSubmissionGraph::Queue::Graphics ? 2u :
                FrameSubmissionGraph::InvalidQueueFamily;
        }, vsm_depth_groups, vsm_depth_mapping, error));
    assert(vsm_depth_groups.size() == 1);
    const auto vsm_depth_submission = vsm_depth_mapping[vsm_depth_pass];
    assert(vsm_depth_submissions.depends_on(vsm_depth_submission, vsm_prepass, error));
    assert(vsm_depth_submissions.depends_on(vsm_depth_submission, motion_depth_copy, error));
    std::vector<FrameSubmissionGraph::Pass> vsm_depth_order;
    assert(vsm_depth_submissions.execution_order(vsm_depth_order, error));
    const auto vsm_position = std::find(vsm_depth_order.begin(), vsm_depth_order.end(),
        vsm_depth_submission);
    assert(vsm_position != vsm_depth_order.end());
    assert(std::find(vsm_depth_order.begin(), vsm_position, vsm_prepass) != vsm_position);
    assert(std::find(vsm_depth_order.begin(), vsm_position, motion_depth_copy) != vsm_position);
    ImageStateTracker vsm_depth_states;
    assert(vsm_depth_states.register_image(vsm_depth_image, depth_range,
        ImageUse::SampledGraphicsCompute, error));
    FrameImageGraph::ExecutionState vsm_depth_execution;
    assert(vsm_depth_graph.begin_execution(vsm_depth_execution, error));
    const size_t barriers_before_vsm_depth = barriers.size();
    assert(vsm_depth_graph.execute_group(vsm_depth_groups.front(), command,
        vsm_depth_states, vsm_depth_execution, {capture}, error,
        VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT));
    assert(barriers.size() == barriers_before_vsm_depth);

    FrameImageGraph failed_command_graph;
    failed_command_graph.add_checked_command_pass("failed recording callback",
        [](VkCommandBuffer, std::string& pass_error) {
            pass_error = "synthetic command-recording failure";
            return false;
        });
    ImageStateTracker failed_command_states;
    assert(!failed_command_graph.execute(command, failed_command_states, {}, error,
        VK_QUEUE_GRAPHICS_BIT));
    assert(error == "synthetic command-recording failure");

    FrameImageGraph invalid;
    FrameImageGraph::Resource resource;
    assert(invalid.import_image(image, range, resource, error));
    assert(!invalid.import_image(VK_NULL_HANDLE, range, rejected_import, error));
    const auto pass = invalid.add_pass("write", [] {});
    FrameImageGraph::Resource newer, rejected;
    assert(invalid.write(pass, resource, ImageUse::TransferDestination, newer, error));
    assert(!invalid.write(pass, resource, ImageUse::TransferDestination, rejected, error));
    assert(!invalid.read(pass, newer, ImageUse::Sampled, error));
    const auto invalid_reader = invalid.add_pass("invalid reader", [] {});
    assert(!invalid.read(invalid_reader, newer, ImageUse::TransferDestination, error));

    const auto storage_image = reinterpret_cast<VkImage>(uintptr_t(3));
    assert(states.register_image(storage_image, range, ImageUse::Undefined, error));
    FrameImageGraph storage;
    FrameImageGraph::Resource storage_start;
    assert(storage.import_image(storage_image, range, storage_start, error));
    std::vector<int> storage_order;
    const auto second_update = storage.add_pass("second update", [&] { storage_order.push_back(2); });
    const auto first_update = storage.add_pass("first update", [&] { storage_order.push_back(1); });
    FrameImageGraph::Resource storage_first, storage_second;
    assert(storage.read_write(first_update, storage_start, storage_first, error));
    assert(storage.read_write(second_update, storage_first, storage_second, error));
    const size_t barriers_before_storage = barriers.size();
    assert(storage.execute(command, states, {capture}, error));
    assert((storage_order == std::vector<int>{1, 2}));
    assert(barriers.size() == barriers_before_storage + 2);
    assert(barriers.back().oldLayout == VK_IMAGE_LAYOUT_GENERAL);
    assert(barriers.back().newLayout == VK_IMAGE_LAYOUT_GENERAL);
    assert((barriers.back().srcAccessMask & VK_ACCESS_SHADER_WRITE_BIT) != 0);

    const auto color_image = reinterpret_cast<VkImage>(uintptr_t(4));
    assert(states.register_image(color_image, range, ImageUse::Undefined, error));
    FrameImageGraph color;
    FrameImageGraph::Resource color_start;
    assert(color.import_image(color_image, range, color_start, error));
    const auto first_color = color.add_pass("first color update", [] {});
    const auto second_color = color.add_pass("second color update", [] {});
    FrameImageGraph::Resource color_first, color_second;
    assert(color.read_write(first_color, color_start, color_first, error,
        ImageUse::ColorAttachment));
    assert(color.read_write(second_color, color_first, color_second, error,
        ImageUse::ColorAttachment));
    const size_t barriers_before_color = barriers.size();
    assert(color.execute(command, states, {capture}, error));
    assert(barriers.size() == barriers_before_color + 2);
    assert(barriers.back().oldLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    assert(barriers.back().newLayout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    assert((barriers.back().srcAccessMask & VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT) != 0);

    // A persistent image is written without reading after a history reset,
    // then read and updated once the previous frame's value is valid.
    const auto history_image = reinterpret_cast<VkImage>(uintptr_t(5));
    assert(states.register_image(history_image, range, ImageUse::Undefined, error));
    for (bool valid_history : {false, true, false})
    {
        FrameImageGraph frame_graph;
        FrameImageGraph::Resource history;
        assert(frame_graph.import_image(history_image, range, history, error));
        const auto adapt = frame_graph.add_pass("adapt exposure", [] {});
        const auto composite = frame_graph.add_pass("sample exposure", [] {});
        if (valid_history)
            assert(frame_graph.read_write(adapt, history, history, error));
        else
            assert(frame_graph.write(adapt, history, ImageUse::StorageWrite, history, error));
        assert(frame_graph.read(composite, history, ImageUse::Sampled, error));
        const size_t first_barrier = barriers.size();
        assert(frame_graph.execute(command, states, {capture}, error));
        assert(barriers.size() == first_barrier + 2);
        const auto& compute_barrier = barriers[first_barrier];
        assert(compute_barrier.newLayout == VK_IMAGE_LAYOUT_GENERAL);
        assert((compute_barrier.dstAccessMask & VK_ACCESS_SHADER_WRITE_BIT) != 0);
        assert(((compute_barrier.dstAccessMask & VK_ACCESS_SHADER_READ_BIT) != 0) == valid_history);
        assert(barriers[first_barrier + 1].newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    FrameImageGraph cyclic;
    FrameImageGraph::Resource a, b;
    assert(cyclic.import_image(image, range, a, error));
    assert(cyclic.import_image(storage_image, range, b, error));
    const auto first = cyclic.add_pass("first", [] {});
    const auto second = cyclic.add_pass("second", [] {});
    FrameImageGraph::Resource a1, b1;
    assert(cyclic.write(first, a, ImageUse::TransferDestination, a1, error));
    assert(cyclic.write(second, b, ImageUse::TransferDestination, b1, error));
    assert(cyclic.read(first, b1, ImageUse::TransferSource, error));
    assert(cyclic.read(second, a1, ImageUse::TransferSource, error));
    order.clear();
    assert(!cyclic.execute(command, states, {capture}, error));
    assert(error.find("cyclic") != std::string::npos);

    // A late invalid import must fail before earlier passes record commands
    // or advance the real image state.
    const auto valid_image = reinterpret_cast<VkImage>(uintptr_t(6));
    const auto missing_image = reinterpret_cast<VkImage>(uintptr_t(7));
    assert(states.register_image(valid_image, range, ImageUse::Undefined, error));
    FrameImageGraph preflight;
    FrameImageGraph::Resource valid_resource, missing_resource, valid_written;
    assert(preflight.import_image(valid_image, range, valid_resource, error));
    assert(preflight.import_image(missing_image, range, missing_resource, error));
    int recorded = 0;
    const auto valid_pass = preflight.add_pass("valid", [&] { ++recorded; });
    const auto missing_pass = preflight.add_pass("missing", [&] { ++recorded; });
    assert(preflight.write(valid_pass, valid_resource, ImageUse::TransferDestination,
        valid_written, error));
    assert(preflight.read(missing_pass, missing_resource, ImageUse::Sampled, error));
    const size_t before_failure = barriers.size();
    assert(!preflight.execute(command, states, {capture}, error));
    assert(error.find("not registered") != std::string::npos);
    assert(recorded == 0 && barriers.size() == before_failure);
    assert(states.transition(command, valid_image, range, ImageUse::TransferDestination,
        {capture}, error));
    assert(barriers.back().oldLayout == VK_IMAGE_LAYOUT_UNDEFINED);

    const auto lights = reinterpret_cast<VkBuffer>(uintptr_t(20));
    const auto tiles = reinterpret_cast<VkBuffer>(uintptr_t(21));
    FrameImageGraph buffer_graph;
    FrameImageGraph::BufferResource input, tile_old, tile_new, duplicate;
    assert(buffer_graph.import_buffer(lights, 0, 128, BufferUse::HostWrite, input, error));
    assert(buffer_graph.import_buffer(tiles, 0, 256, BufferUse::FragmentRead, tile_old, error));
    assert(buffer_graph.import_buffer(tiles, 0, 256, BufferUse::FragmentRead, duplicate, error));
    assert(duplicate.index == tile_old.index && duplicate.version == tile_old.version);
    FrameImageGraph overlapping_buffer_imports;
    FrameImageGraph::BufferResource overlapping_full, overlapping_tail;
    assert(overlapping_buffer_imports.import_buffer(tiles, 0, 256,
        BufferUse::FragmentRead, overlapping_full, error));
    assert(overlapping_buffer_imports.import_buffer(tiles, 128, 128,
        BufferUse::FragmentRead, overlapping_tail, error));
    assert(overlapping_tail.index != overlapping_full.index);
    std::vector<int> buffer_order;
    const auto shade_tiles = buffer_graph.add_pass("shade tiles", [&] { buffer_order.push_back(2); });
    const auto cull_tiles = buffer_graph.add_pass("cull tiles", [&] { buffer_order.push_back(1); });
    assert(buffer_graph.read_buffer(cull_tiles, input, BufferUse::ComputeRead, error));
    assert(buffer_graph.write_buffer(cull_tiles, tile_old, BufferUse::ComputeWrite, tile_new, error));
    assert(buffer_graph.read_buffer(shade_tiles, tile_new, BufferUse::FragmentRead, error));
    assert(!buffer_graph.read_buffer(shade_tiles, tile_new, BufferUse::ComputeWrite, error));
    const size_t before_buffers = buffer_barriers.size();
    assert(buffer_graph.execute(command, states, {capture}, error));
    assert((buffer_order == std::vector<int>{1, 2}));
    assert(buffer_barriers.size() == before_buffers + 3);
    assert(buffer_barriers[before_buffers].srcAccessMask == VK_ACCESS_HOST_WRITE_BIT);
    assert(buffer_barriers[before_buffers].dstAccessMask == VK_ACCESS_SHADER_READ_BIT);
    assert(buffer_stages[before_buffers].first == VK_PIPELINE_STAGE_HOST_BIT);
    assert(buffer_barriers[before_buffers + 1].srcAccessMask == VK_ACCESS_SHADER_READ_BIT);
    assert(buffer_barriers[before_buffers + 1].dstAccessMask == VK_ACCESS_SHADER_WRITE_BIT);
    assert(buffer_barriers[before_buffers + 2].srcAccessMask == VK_ACCESS_SHADER_WRITE_BIT);
    assert(buffer_barriers[before_buffers + 2].dstAccessMask == VK_ACCESS_SHADER_READ_BIT);

    // Imports can carry the last queue submission that produced their initial version.
    FrameImageGraph external_graph;
    FrameImageGraph::Resource external_image;
    FrameImageGraph::BufferResource external_buffer, external_buffer_next;
    const auto external_image_handle = reinterpret_cast<VkImage>(uintptr_t(23));
    const auto external_buffer_handle = reinterpret_cast<VkBuffer>(uintptr_t(24));
    assert(external_graph.import_image(external_image_handle, range, external_image, error,
        FrameSubmissionGraph::Queue::Compute, 41));
    assert(external_graph.import_buffer(external_buffer_handle, 0, 64,
        BufferUse::TransferDestination, external_buffer, error,
        FrameSubmissionGraph::Queue::Copy, 73));
    const auto external_read = external_graph.add_pass("external reads", [] {});
    const auto external_write = external_graph.add_pass("external overwrite", [] {});
    assert(external_graph.read(external_read, external_image, ImageUse::Sampled, error));
    assert(external_graph.read_buffer(external_read, external_buffer, BufferUse::FragmentRead, error));
    assert(external_graph.write_buffer(external_write, external_buffer,
        BufferUse::TransferDestination, external_buffer_next, error));

    FrameSubmissionGraph external_submissions;
    const auto graphics_submission = external_submissions.add_pass(
        "graphics consumer", FrameSubmissionGraph::Queue::Graphics);
    const auto map_external_pass = [graphics_submission](FrameImageGraph::Pass) {
        return graphics_submission;
    };
    assert(external_graph.connect_submission_dependencies(
        external_submissions, map_external_pass, error));
    uint32_t compute_waits = 0, copy_waits = 0;
    assert(external_submissions.submit(graphics_submission,
        [](FrameSubmissionGraph::Queue) { return uint64_t{89}; },
        [&compute_waits, &copy_waits](FrameSubmissionGraph::Queue destination,
            FrameSubmissionGraph::Queue source, uint64_t submission) {
            assert(destination == FrameSubmissionGraph::Queue::Graphics);
            if (source == FrameSubmissionGraph::Queue::Compute)
            {
                assert(submission == 41);
                ++compute_waits;
            }
            else if (source == FrameSubmissionGraph::Queue::Copy)
            {
                assert(submission == 73);
                ++copy_waits;
            }
            else
                assert(false && "unexpected external source queue");
        }, error));
    assert(compute_waits == 1 && copy_waits == 1);

    // Generated queue groups must pair exclusive image and buffer ownership
    // barriers with the resource dependency that orders the submissions.
    const auto owned_image = reinterpret_cast<VkImage>(uintptr_t(25));
    const auto owned_buffer = reinterpret_cast<VkBuffer>(uintptr_t(26));
    assert(states.register_image(owned_image, range, ImageUse::Undefined, error));
    FrameImageGraph ownership_graph;
    FrameImageGraph::Resource owned_image_initial, owned_image_written;
    FrameImageGraph::BufferResource owned_buffer_initial, owned_buffer_written;
    assert(ownership_graph.import_image(owned_image, range, owned_image_initial, error));
    assert(ownership_graph.import_buffer(owned_buffer, 0, 128, BufferUse::Undefined,
        owned_buffer_initial, error));
    std::vector<int> ownership_order;
    const auto graphics_producer = ownership_graph.add_pass("graphics ownership producer",
        FrameSubmissionGraph::Queue::Graphics,
        [&] { ownership_order.push_back(1); });
    const auto compute_consumer = ownership_graph.add_pass("compute ownership consumer",
        FrameSubmissionGraph::Queue::Compute,
        [&] { ownership_order.push_back(2); });
    assert(ownership_graph.write(graphics_producer, owned_image_initial,
        ImageUse::TransferDestination, owned_image_written, error));
    assert(ownership_graph.read(compute_consumer, owned_image_written,
        ImageUse::ComputeSampled, error));
    assert(ownership_graph.write_buffer(graphics_producer, owned_buffer_initial,
        BufferUse::TransferDestination, owned_buffer_written, error));
    assert(ownership_graph.read_buffer(compute_consumer, owned_buffer_written,
        BufferUse::ComputeRead, error));

    FrameSubmissionGraph ownership_submissions;
    std::vector<FrameImageGraph::QueueGroup> ownership_groups;
    std::vector<FrameSubmissionGraph::Pass> ownership_mapping;
    const auto family_for_queue = [](FrameSubmissionGraph::Queue queue) {
        switch (queue)
        {
        case FrameSubmissionGraph::Queue::Graphics: return uint32_t{2};
        case FrameSubmissionGraph::Queue::Compute: return uint32_t{5};
        case FrameSubmissionGraph::Queue::Copy: return uint32_t{7};
        }
        return FrameSubmissionGraph::InvalidQueueFamily;
    };
    assert(ownership_graph.generate_submission_groups(ownership_submissions,
        family_for_queue, ownership_groups, ownership_mapping, error));
    assert(ownership_groups.size() == 2);
    assert(ownership_groups[0].queue == FrameSubmissionGraph::Queue::Graphics);
    assert(ownership_groups[1].queue == FrameSubmissionGraph::Queue::Compute);
    assert(ownership_groups[0].ownership_releases.size() == 2);
    assert(ownership_groups[1].ownership_acquires.size() == 2);
    const auto& image_release = ownership_groups[0].ownership_releases[0];
    const auto& image_acquire = ownership_groups[1].ownership_acquires[0];
    assert(image_release.kind == FrameImageGraph::QueueGroup::OwnershipTransfer::Kind::Image);
    assert(image_acquire.kind == FrameImageGraph::QueueGroup::OwnershipTransfer::Kind::Image);
    assert(image_release.resource_id == image_acquire.resource_id);
    assert(image_release.source_pass == graphics_producer);
    assert(image_release.destination_pass == compute_consumer);
    assert(image_release.source_family == 2 && image_release.destination_family == 5);
    assert(image_release.source_image_use == ImageUse::TransferDestination);
    assert(image_release.destination_image_use == ImageUse::ComputeSampled);
    assert(ownership_submissions.depends_on(ownership_mapping[compute_consumer],
        ownership_mapping[graphics_producer], error));
    assert(ownership_submissions.queue_family(ownership_mapping[graphics_producer]) == 2);
    assert(ownership_submissions.queue_family(ownership_mapping[compute_consumer]) == 5);

    FrameImageGraph::ExecutionState ownership_execution;
    assert(ownership_graph.begin_execution(ownership_execution, error));
    const size_t ownership_image_barrier_start = barriers.size();
    const size_t ownership_buffer_barrier_start = buffer_barriers.size();
    const auto queue_flags_for = [](FrameSubmissionGraph::Queue queue) {
        return queue == FrameSubmissionGraph::Queue::Compute ? VK_QUEUE_COMPUTE_BIT :
            VK_QUEUE_GRAPHICS_BIT;
    };
    for (const auto& group : ownership_groups)
        assert(ownership_graph.execute_group(group, command, states, ownership_execution,
            {capture}, error, queue_flags_for(group.queue), true));
    assert((ownership_order == std::vector<int>{1, 2}));
    assert(barriers.size() == ownership_image_barrier_start + 3);
    assert(barriers[ownership_image_barrier_start + 1].srcQueueFamilyIndex == 2);
    assert(barriers[ownership_image_barrier_start + 1].dstQueueFamilyIndex == 5);
    assert(barriers[ownership_image_barrier_start + 1].oldLayout ==
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    assert(barriers[ownership_image_barrier_start + 1].newLayout ==
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    assert(barriers[ownership_image_barrier_start + 2].srcQueueFamilyIndex == 2);
    assert(barriers[ownership_image_barrier_start + 2].dstQueueFamilyIndex == 5);
    assert(buffer_barriers.size() == ownership_buffer_barrier_start + 3);
    assert(buffer_barriers[ownership_buffer_barrier_start + 1].srcQueueFamilyIndex == 2);
    assert(buffer_barriers[ownership_buffer_barrier_start + 1].dstQueueFamilyIndex == 5);
    assert(buffer_barriers[ownership_buffer_barrier_start + 2].srcQueueFamilyIndex == 2);
    assert(buffer_barriers[ownership_buffer_barrier_start + 2].dstQueueFamilyIndex == 5);

    // Graphics and compute can exchange G-buffer images without ownership
    // barriers when the image was created with concurrent queue-family sharing.
    const auto concurrent_image = reinterpret_cast<VkImage>(uintptr_t(28));
    assert(states.register_image(concurrent_image, range, ImageUse::Undefined, error));
    FrameImageGraph concurrent_graph;
    FrameImageGraph::Resource concurrent_initial, concurrent_written, concurrent_output;
    assert(concurrent_graph.import_image(concurrent_image, range, concurrent_initial, error,
        FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_CONCURRENT, {2, 5}));
    const auto concurrent_producer = concurrent_graph.add_pass("concurrent graphics producer",
        FrameSubmissionGraph::Queue::Graphics, [] {});
    const auto concurrent_consumer = concurrent_graph.add_pass("concurrent compute consumer",
        FrameSubmissionGraph::Queue::Compute, [] {});
    assert(concurrent_graph.write(concurrent_producer, concurrent_initial,
        ImageUse::ColorAttachment, concurrent_written, error));
    assert(concurrent_graph.write(concurrent_consumer, concurrent_written,
        ImageUse::StorageWrite, concurrent_output, error));
    const auto concurrent_final = concurrent_graph.add_pass("concurrent graphics consumer",
        FrameSubmissionGraph::Queue::Graphics, [] {});
    assert(concurrent_graph.read(concurrent_final, concurrent_output,
        ImageUse::Sampled, error));
    FrameSubmissionGraph concurrent_submissions;
    const auto concurrent_scene = concurrent_submissions.add_pass("postprocess scene producer",
        FrameSubmissionGraph::Queue::Graphics, 2);
    std::vector<FrameImageGraph::QueueGroup> concurrent_groups;
    std::vector<FrameSubmissionGraph::Pass> concurrent_mapping;
    assert(concurrent_graph.generate_submission_groups(concurrent_submissions,
        family_for_queue, concurrent_groups, concurrent_mapping, error));
    assert(concurrent_groups.size() == 3 &&
        concurrent_groups[0].queue == FrameSubmissionGraph::Queue::Graphics &&
        concurrent_groups[1].queue == FrameSubmissionGraph::Queue::Compute &&
        concurrent_groups[2].queue == FrameSubmissionGraph::Queue::Graphics);
    assert(concurrent_submissions.depends_on(concurrent_groups.front().submission,
        concurrent_scene, error));
    const auto concurrent_presentation = concurrent_submissions.add_pass("postprocess presentation",
        FrameSubmissionGraph::Queue::Graphics, 2);
    for (const auto& group : concurrent_groups)
        assert(concurrent_submissions.depends_on(concurrent_presentation, group.submission, error));
    for (const auto& group : concurrent_groups)
        assert(group.ownership_releases.empty() && group.ownership_acquires.empty());
    assert(concurrent_submissions.depends_on(concurrent_mapping[concurrent_consumer],
        concurrent_mapping[concurrent_producer], error));
    assert(concurrent_submissions.depends_on(concurrent_mapping[concurrent_final],
        concurrent_mapping[concurrent_consumer], error));
    std::vector<FrameSubmissionGraph::Pass> concurrent_order;
    assert(concurrent_submissions.execution_order(concurrent_order, error));
    const auto rank = [&concurrent_order](FrameSubmissionGraph::Pass pass) {
        const auto it = std::find(concurrent_order.begin(), concurrent_order.end(), pass);
        assert(it != concurrent_order.end());
        return std::distance(concurrent_order.begin(), it);
    };
    assert(rank(concurrent_scene) < rank(concurrent_groups[0].submission) &&
        rank(concurrent_groups[0].submission) < rank(concurrent_groups[1].submission) &&
        rank(concurrent_groups[1].submission) < rank(concurrent_groups[2].submission) &&
        rank(concurrent_groups[2].submission) < rank(concurrent_presentation));
    std::vector<std::pair<FrameSubmissionGraph::Queue, FrameSubmissionGraph::Queue>> concurrent_waits;
    uint64_t concurrent_token = 800;
    const auto record_concurrent_wait = [&concurrent_waits](FrameSubmissionGraph::Queue destination,
        FrameSubmissionGraph::Queue source, uint64_t) {
        concurrent_waits.emplace_back(destination, source);
    };
    assert(concurrent_submissions.submit(concurrent_scene,
        [&concurrent_token](FrameSubmissionGraph::Queue) { return ++concurrent_token; },
        record_concurrent_wait, error));
    for (const auto& group : concurrent_groups)
        assert(concurrent_submissions.submit(group.submission,
            [&concurrent_token](FrameSubmissionGraph::Queue) { return ++concurrent_token; },
            record_concurrent_wait, error));
    assert(concurrent_submissions.submit(concurrent_presentation,
        [&concurrent_token](FrameSubmissionGraph::Queue) { return ++concurrent_token; },
        record_concurrent_wait, error));
    assert(std::find(concurrent_waits.begin(), concurrent_waits.end(),
        std::make_pair(FrameSubmissionGraph::Queue::Compute, FrameSubmissionGraph::Queue::Graphics)) !=
        concurrent_waits.end());
    assert(std::find(concurrent_waits.begin(), concurrent_waits.end(),
        std::make_pair(FrameSubmissionGraph::Queue::Graphics, FrameSubmissionGraph::Queue::Compute)) !=
        concurrent_waits.end());
    FrameImageGraph::ExecutionState concurrent_execution;
    assert(concurrent_graph.begin_execution(concurrent_execution, error));
    const size_t concurrent_barrier_start = barriers.size();
    for (const auto& group : concurrent_groups)
        assert(concurrent_graph.execute_group(group, command, states, concurrent_execution,
            {capture}, error, queue_flags_for(group.queue), true));
    assert(barriers.size() == concurrent_barrier_start + 3);
    for (size_t index = concurrent_barrier_start; index < barriers.size(); ++index)
        assert(barriers[index].srcQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED &&
            barriers[index].dstQueueFamilyIndex == VK_QUEUE_FAMILY_IGNORED);

    // Overlapping imports of one physical resource can transfer only the
    // shared subrange while the rest stays owned by the original family.
    const auto partial_alias_image = reinterpret_cast<VkImage>(uintptr_t(31));
    const auto partial_alias_buffer = reinterpret_cast<VkBuffer>(uintptr_t(32));
    const VkImageSubresourceRange full_array_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 0, 2};
    const VkImageSubresourceRange shared_cell{VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 1, 1};
    assert(states.register_image(partial_alias_image, full_array_range, ImageUse::Undefined, error));
    FrameImageGraph partial_alias_graph;
    FrameImageGraph::Resource full_image, shared_image, full_image_written, shared_image_written;
    FrameImageGraph::BufferResource full_buffer, shared_buffer, full_buffer_written,
        shared_buffer_written;
    assert(partial_alias_graph.import_image(partial_alias_image, full_array_range, full_image, error));
    assert(partial_alias_graph.import_image(partial_alias_image, shared_cell, shared_image, error));
    assert(partial_alias_graph.import_buffer(partial_alias_buffer, 0, 256, BufferUse::Undefined,
        full_buffer, error));
    assert(partial_alias_graph.import_buffer(partial_alias_buffer, 64, 64, BufferUse::Undefined,
        shared_buffer, error));
    const auto partial_graphics_first = partial_alias_graph.add_pass(
        "partial alias graphics producer", FrameSubmissionGraph::Queue::Graphics, [] {});
    const auto partial_compute = partial_alias_graph.add_pass(
        "partial alias compute consumer", FrameSubmissionGraph::Queue::Compute, [] {});
    const auto partial_graphics_last = partial_alias_graph.add_pass(
        "partial alias graphics consumer", FrameSubmissionGraph::Queue::Graphics, [] {});
    assert(partial_alias_graph.write(partial_graphics_first, full_image,
        ImageUse::TransferDestination, full_image_written, error));
    assert(partial_alias_graph.read(partial_compute, shared_image,
        ImageUse::ComputeSampled, error));
    assert(partial_alias_graph.write(partial_graphics_last, shared_image,
        ImageUse::TransferDestination, shared_image_written, error));
    assert(partial_alias_graph.write_buffer(partial_graphics_first, full_buffer,
        BufferUse::TransferDestination, full_buffer_written, error));
    assert(partial_alias_graph.read_buffer(partial_compute, shared_buffer,
        BufferUse::ComputeRead, error));
    assert(partial_alias_graph.write_buffer(partial_graphics_last, shared_buffer,
        BufferUse::TransferDestination, shared_buffer_written, error));

    FrameSubmissionGraph partial_alias_submissions;
    std::vector<FrameImageGraph::QueueGroup> partial_alias_groups;
    std::vector<FrameSubmissionGraph::Pass> partial_alias_mapping;
    assert(partial_alias_graph.generate_submission_groups(partial_alias_submissions,
        family_for_queue, partial_alias_groups, partial_alias_mapping, error));
    assert(partial_alias_groups.size() == 3);
    assert(partial_alias_groups[0].queue == FrameSubmissionGraph::Queue::Graphics);
    assert(partial_alias_groups[1].queue == FrameSubmissionGraph::Queue::Compute);
    assert(partial_alias_groups[2].queue == FrameSubmissionGraph::Queue::Graphics);
    assert(partial_alias_groups[0].ownership_releases.size() == 2);
    assert(partial_alias_groups[1].ownership_acquires.size() == 2);
    assert(partial_alias_groups[1].ownership_releases.size() == 2);
    assert(partial_alias_groups[2].ownership_acquires.size() == 2);
    for (const auto& group : partial_alias_groups)
        for (const auto& transfer : group.ownership_releases)
        {
            if (transfer.kind == FrameImageGraph::QueueGroup::OwnershipTransfer::Kind::Image)
                assert(transfer.image_range.aspectMask == shared_cell.aspectMask &&
                    transfer.image_range.baseMipLevel == shared_cell.baseMipLevel &&
                    transfer.image_range.levelCount == shared_cell.levelCount &&
                    transfer.image_range.baseArrayLayer == shared_cell.baseArrayLayer &&
                    transfer.image_range.layerCount == shared_cell.layerCount);
            else
                assert(transfer.buffer_offset == 64 && transfer.buffer_size == 64);
        }
    assert(partial_alias_submissions.depends_on(partial_alias_mapping[partial_compute],
        partial_alias_mapping[partial_graphics_first], error));
    assert(partial_alias_submissions.depends_on(partial_alias_mapping[partial_graphics_last],
        partial_alias_mapping[partial_compute], error));

    FrameImageGraph::ExecutionState partial_alias_execution;
    assert(partial_alias_graph.begin_execution(partial_alias_execution, error));
    const size_t partial_image_barrier_start = barriers.size();
    const size_t partial_buffer_barrier_start = buffer_barriers.size();
    for (const auto& group : partial_alias_groups)
        assert(partial_alias_graph.execute_group(group, command, states, partial_alias_execution,
            {capture}, error, queue_flags_for(group.queue), true));
    std::vector<VkImageMemoryBarrier> partial_image_ownership_barriers;
    for (size_t index = partial_image_barrier_start; index < barriers.size(); ++index)
        if ((barriers[index].srcQueueFamilyIndex == 2 && barriers[index].dstQueueFamilyIndex == 5) ||
            (barriers[index].srcQueueFamilyIndex == 5 && barriers[index].dstQueueFamilyIndex == 2))
            partial_image_ownership_barriers.push_back(barriers[index]);
    assert(partial_image_ownership_barriers.size() == 4);
    for (const VkImageMemoryBarrier& barrier : partial_image_ownership_barriers)
    {
        assert(barrier.subresourceRange.aspectMask == shared_cell.aspectMask);
        assert(barrier.subresourceRange.baseMipLevel == shared_cell.baseMipLevel);
        assert(barrier.subresourceRange.levelCount == shared_cell.levelCount);
        assert(barrier.subresourceRange.baseArrayLayer == shared_cell.baseArrayLayer);
        assert(barrier.subresourceRange.layerCount == shared_cell.layerCount);
    }
    std::vector<VkBufferMemoryBarrier> partial_buffer_ownership_barriers;
    for (size_t index = partial_buffer_barrier_start; index < buffer_barriers.size(); ++index)
        if ((buffer_barriers[index].srcQueueFamilyIndex == 2 &&
                buffer_barriers[index].dstQueueFamilyIndex == 5) ||
            (buffer_barriers[index].srcQueueFamilyIndex == 5 &&
                buffer_barriers[index].dstQueueFamilyIndex == 2))
            partial_buffer_ownership_barriers.push_back(buffer_barriers[index]);
    assert(partial_buffer_ownership_barriers.size() == 4);
    for (const VkBufferMemoryBarrier& barrier : partial_buffer_ownership_barriers)
        assert(barrier.offset == 64 && barrier.size == 64);

    // An external producer token is released only for the aliased cell. The
    // subsequent full-range consumer inherits that wait through the generated
    // submissions instead of waiting on the producer token a second time.
    const auto external_alias_image = reinterpret_cast<VkImage>(uintptr_t(33));
    const auto external_alias_buffer = reinterpret_cast<VkBuffer>(uintptr_t(34));
    assert(states.register_image(external_alias_image, full_array_range,
        ImageUse::ComputeSampled, error));
    FrameImageGraph external_alias_graph;
    FrameImageGraph::Resource external_full_image, external_shared_image;
    FrameImageGraph::BufferResource external_full_buffer, external_shared_buffer;
    assert(external_alias_graph.import_image(external_alias_image, full_array_range,
        external_full_image, error, FrameSubmissionGraph::Queue::Compute, 707,
        VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::ComputeSampled));
    assert(external_alias_graph.import_image(external_alias_image, shared_cell,
        external_shared_image, error, FrameSubmissionGraph::Queue::Compute, 707,
        VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::ComputeSampled));
    assert(external_alias_graph.import_buffer(external_alias_buffer, 0, 256,
        BufferUse::ComputeWrite, external_full_buffer, error,
        FrameSubmissionGraph::Queue::Compute, 707));
    assert(external_alias_graph.import_buffer(external_alias_buffer, 64, 64,
        BufferUse::ComputeWrite, external_shared_buffer, error,
        FrameSubmissionGraph::Queue::Compute, 707));
    const auto external_alias_graphics = external_alias_graph.add_pass(
        "external alias graphics consumer", FrameSubmissionGraph::Queue::Graphics, [] {});
    const auto external_alias_compute = external_alias_graph.add_pass(
        "external alias full compute consumer", FrameSubmissionGraph::Queue::Compute, [] {});
    assert(external_alias_graph.read(external_alias_graphics, external_shared_image,
        ImageUse::Sampled, error));
    assert(external_alias_graph.read(external_alias_compute, external_full_image,
        ImageUse::ComputeSampled, error));
    assert(external_alias_graph.read_buffer(external_alias_graphics, external_shared_buffer,
        BufferUse::VertexRead, error));
    assert(external_alias_graph.read_buffer(external_alias_compute, external_full_buffer,
        BufferUse::ComputeRead, error));

    FrameSubmissionGraph external_alias_submissions;
    std::vector<FrameImageGraph::QueueGroup> external_alias_groups;
    std::vector<FrameSubmissionGraph::Pass> external_alias_mapping;
    assert(external_alias_graph.generate_submission_groups(external_alias_submissions,
        family_for_queue, external_alias_groups, external_alias_mapping, error));
    assert(external_alias_groups.size() == 3);
    const auto external_alias_release = std::find_if(external_alias_groups.begin(),
        external_alias_groups.end(), [](const FrameImageGraph::QueueGroup& group) {
            return group.passes.empty();
        });
    assert(external_alias_release != external_alias_groups.end());
    assert(external_alias_release->queue == FrameSubmissionGraph::Queue::Compute);
    assert(external_alias_release->external_submission == 707);
    assert(external_alias_release->ownership_releases.size() == 2);
    for (const auto& transfer : external_alias_release->ownership_releases)
    {
        if (transfer.kind == FrameImageGraph::QueueGroup::OwnershipTransfer::Kind::Image)
            assert(transfer.image_range.baseMipLevel == shared_cell.baseMipLevel &&
                transfer.image_range.levelCount == shared_cell.levelCount &&
                transfer.image_range.baseArrayLayer == shared_cell.baseArrayLayer &&
                transfer.image_range.layerCount == shared_cell.layerCount);
        else
            assert(transfer.buffer_offset == 64 && transfer.buffer_size == 64);
    }
    assert(external_alias_submissions.depends_on(external_alias_mapping[external_alias_graphics],
        external_alias_release->submission, error));
    assert(external_alias_submissions.depends_on(external_alias_mapping[external_alias_compute],
        external_alias_mapping[external_alias_graphics], error));
    uint32_t external_alias_token_waits = 0, external_alias_release_waits = 0;
    std::vector<FrameSubmissionGraph::Pass> external_alias_order;
    assert(external_alias_submissions.execution_order(external_alias_order, error));
    for (FrameSubmissionGraph::Pass submission : external_alias_order)
        assert(external_alias_submissions.submit(submission,
            [submission, external_release_submission = external_alias_release->submission](
                FrameSubmissionGraph::Queue) {
                return submission == external_release_submission ? uint64_t{808} : uint64_t{809};
            }, [&external_alias_token_waits, &external_alias_release_waits](FrameSubmissionGraph::Queue,
                FrameSubmissionGraph::Queue source, uint64_t token) {
                if (token == 707)
                {
                    assert(source == FrameSubmissionGraph::Queue::Compute);
                    ++external_alias_token_waits;
                }
                if (token == 808)
                {
                    assert(source == FrameSubmissionGraph::Queue::Compute);
                    ++external_alias_release_waits;
                }
            }, error));
    assert(external_alias_token_waits == 0 && external_alias_release_waits == 1);

    FrameImageGraph::ExecutionState external_alias_execution;
    assert(external_alias_graph.begin_execution(external_alias_execution, error));
    for (const auto& group : external_alias_groups)
        assert(external_alias_graph.execute_group(group, command, states,
            external_alias_execution, {capture}, error, queue_flags_for(group.queue), true));
    assert(barriers.back().newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    // An imported producer on another queue family gets one release-only
    // submission. Its producer token is waited once, then the consumer waits
    // for that release instead of waiting on the same external token again.
    const auto external_owned_image = reinterpret_cast<VkImage>(uintptr_t(27));
    const auto external_owned_buffer = reinterpret_cast<VkBuffer>(uintptr_t(28));
    assert(states.register_image(external_owned_image, range,
        ImageUse::ComputeSampled, error));
    FrameImageGraph external_ownership_graph;
    FrameImageGraph::Resource external_owned_image_resource;
    FrameImageGraph::BufferResource external_owned_buffer_resource;
    assert(external_ownership_graph.import_image(external_owned_image, range,
        external_owned_image_resource, error, FrameSubmissionGraph::Queue::Compute, 404,
        VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::ComputeSampled));
    assert(external_ownership_graph.import_buffer(external_owned_buffer, 0, 96,
        BufferUse::ComputeWrite, external_owned_buffer_resource, error,
        FrameSubmissionGraph::Queue::Compute, 404));
    std::vector<int> external_ownership_order;
    const auto external_graphics_consumer = external_ownership_graph.add_pass(
        "external ownership graphics consumer", FrameSubmissionGraph::Queue::Graphics,
        [&] { external_ownership_order.push_back(1); });
    assert(external_ownership_graph.read(external_graphics_consumer,
        external_owned_image_resource, ImageUse::Sampled, error));
    assert(external_ownership_graph.read_buffer(external_graphics_consumer,
        external_owned_buffer_resource, BufferUse::VertexRead, error));

    FrameSubmissionGraph external_ownership_submissions;
    std::vector<FrameImageGraph::QueueGroup> external_ownership_groups;
    std::vector<FrameSubmissionGraph::Pass> external_ownership_mapping;
    assert(external_ownership_graph.generate_submission_groups(external_ownership_submissions,
        family_for_queue, external_ownership_groups, external_ownership_mapping, error));
    assert(external_ownership_groups.size() == 2);
    const auto& external_release_group = external_ownership_groups[0];
    const auto& external_consumer_group = external_ownership_groups[1];
    assert(external_release_group.passes.empty());
    assert(external_release_group.queue == FrameSubmissionGraph::Queue::Compute);
    assert(external_release_group.queue_family == 5);
    assert(external_release_group.external_submission == 404);
    assert(external_release_group.ownership_releases.size() == 2);
    assert(external_consumer_group.passes.size() == 1);
    assert(external_consumer_group.ownership_acquires.size() == 2);
    assert(external_ownership_submissions.depends_on(
        external_ownership_mapping[external_graphics_consumer],
        external_release_group.submission, error));

    std::vector<std::pair<FrameSubmissionGraph::Queue, uint64_t>> external_waits;
    for (const auto& group : external_ownership_groups)
    {
        const uint64_t token = group.queue == FrameSubmissionGraph::Queue::Compute ? 505 : 606;
        assert(external_ownership_submissions.submit(group.submission,
            [token](FrameSubmissionGraph::Queue) { return token; },
            [&external_waits](FrameSubmissionGraph::Queue destination,
                FrameSubmissionGraph::Queue source, uint64_t submission) {
                assert(destination == FrameSubmissionGraph::Queue::Graphics);
                assert(source == FrameSubmissionGraph::Queue::Compute);
                external_waits.emplace_back(source, submission);
            }, error));
    }
    assert(external_waits.size() == 1);
    assert(external_waits[0].first == FrameSubmissionGraph::Queue::Compute);
    assert(external_waits[0].second == 505);

    FrameImageGraph::ExecutionState external_ownership_execution;
    assert(external_ownership_graph.begin_execution(external_ownership_execution, error));
    const size_t external_image_barrier_start = barriers.size();
    const size_t external_buffer_barrier_start = buffer_barriers.size();
    for (const auto& group : external_ownership_groups)
        assert(external_ownership_graph.execute_group(group, command, states,
            external_ownership_execution, {capture}, error, queue_flags_for(group.queue), true));
    assert((external_ownership_order == std::vector<int>{1}));
    assert(barriers.size() == external_image_barrier_start + 2);
    assert(barriers[external_image_barrier_start].srcQueueFamilyIndex == 5);
    assert(barriers[external_image_barrier_start].dstQueueFamilyIndex == 2);
    assert(barriers[external_image_barrier_start + 1].srcQueueFamilyIndex == 5);
    assert(barriers[external_image_barrier_start + 1].dstQueueFamilyIndex == 2);
    assert(buffer_barriers.size() == external_buffer_barrier_start + 2);
    assert(buffer_barriers[external_buffer_barrier_start].srcQueueFamilyIndex == 5);
    assert(buffer_barriers[external_buffer_barrier_start].dstQueueFamilyIndex == 2);
    assert(buffer_barriers[external_buffer_barrier_start + 1].srcQueueFamilyIndex == 5);
    assert(buffer_barriers[external_buffer_barrier_start + 1].dstQueueFamilyIndex == 2);

    // Carry a resource through two frame graphs. The first frame transfers it
    // from graphics to compute; the second imports the compute submission that
    // actually produced its state and performs an external release/acquire back
    // to graphics.
    const auto multi_frame_image = reinterpret_cast<VkImage>(uintptr_t(30));
    assert(states.register_image(multi_frame_image, range, ImageUse::ColorAttachment, error));
    FrameImageGraph first_frame_graph;
    FrameImageGraph::Resource first_frame_input, first_frame_written;
    assert(first_frame_graph.import_image(multi_frame_image, range, first_frame_input, error,
        FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {},
        ImageUse::ColorAttachment));
    const auto first_frame_graphics = first_frame_graph.add_pass(
        "frame one graphics producer", FrameSubmissionGraph::Queue::Graphics, [] {});
    const auto first_frame_compute = first_frame_graph.add_pass(
        "frame one compute consumer", FrameSubmissionGraph::Queue::Compute, [] {});
    assert(first_frame_graph.write(first_frame_graphics, first_frame_input,
        ImageUse::TransferDestination, first_frame_written, error));
    assert(first_frame_graph.read(first_frame_compute, first_frame_written,
        ImageUse::ComputeSampled, error));

    FrameSubmissionGraph first_frame_submissions;
    std::vector<FrameImageGraph::QueueGroup> first_frame_groups;
    std::vector<FrameSubmissionGraph::Pass> first_frame_mapping;
    assert(first_frame_graph.generate_submission_groups(first_frame_submissions,
        family_for_queue, first_frame_groups, first_frame_mapping, error));
    std::vector<FrameSubmissionGraph::Pass> first_frame_order;
    assert(first_frame_submissions.execution_order(first_frame_order, error));
    for (const auto submission : first_frame_order)
        assert(first_frame_submissions.submit(submission,
            [submission, &first_frame_mapping, first_frame_compute](FrameSubmissionGraph::Queue queue) {
                assert(queue == (submission == first_frame_mapping[first_frame_compute] ?
                    FrameSubmissionGraph::Queue::Compute : FrameSubmissionGraph::Queue::Graphics));
                return queue == FrameSubmissionGraph::Queue::Compute ? uint64_t{505} : uint64_t{404};
            }, [](FrameSubmissionGraph::Queue destination, FrameSubmissionGraph::Queue source,
                uint64_t token) {
                assert(destination == FrameSubmissionGraph::Queue::Compute);
                assert(source == FrameSubmissionGraph::Queue::Graphics && token == 404);
            }, error));
    const uint64_t frame_one_compute_token =
        first_frame_submissions.submission(first_frame_mapping[first_frame_compute]);
    assert(frame_one_compute_token == 505);
    FrameImageGraph::ExecutionState first_frame_execution;
    assert(first_frame_graph.begin_execution(first_frame_execution, error));
    for (const auto submission : first_frame_order)
    {
        const auto group = std::find_if(first_frame_groups.begin(), first_frame_groups.end(),
            [submission](const FrameImageGraph::QueueGroup& candidate) {
                return candidate.submission == submission;
            });
        assert(group != first_frame_groups.end());
        assert(first_frame_graph.execute_group(*group, command, states, first_frame_execution,
            {capture}, error, queue_flags_for(group->queue), true));
    }

    FrameImageGraph second_frame_graph;
    FrameImageGraph::Resource second_frame_input;
    assert(second_frame_graph.import_image(multi_frame_image, range, second_frame_input, error,
        FrameSubmissionGraph::Queue::Compute, frame_one_compute_token,
        VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::ComputeSampled));
    const auto second_frame_graphics = second_frame_graph.add_pass(
        "frame two graphics consumer", FrameSubmissionGraph::Queue::Graphics, [] {});
    assert(second_frame_graph.read(second_frame_graphics, second_frame_input,
        ImageUse::Sampled, error));
    FrameSubmissionGraph second_frame_submissions;
    std::vector<FrameImageGraph::QueueGroup> second_frame_groups;
    std::vector<FrameSubmissionGraph::Pass> second_frame_mapping;
    assert(second_frame_graph.generate_submission_groups(second_frame_submissions,
        family_for_queue, second_frame_groups, second_frame_mapping, error));
    const auto second_frame_release = std::find_if(second_frame_groups.begin(), second_frame_groups.end(),
        [](const FrameImageGraph::QueueGroup& group) { return group.passes.empty(); });
    assert(second_frame_release != second_frame_groups.end());
    assert(second_frame_release->external_submission == frame_one_compute_token);
    std::vector<std::pair<FrameSubmissionGraph::Queue, uint64_t>> multi_frame_waits;
    std::vector<FrameSubmissionGraph::Pass> second_frame_order;
    assert(second_frame_submissions.execution_order(second_frame_order, error));
    for (const auto submission : second_frame_order)
        assert(second_frame_submissions.submit(submission,
            [submission, second_frame_release](FrameSubmissionGraph::Queue queue) {
                assert(queue == (submission == second_frame_release->submission ?
                    FrameSubmissionGraph::Queue::Compute : FrameSubmissionGraph::Queue::Graphics));
                return submission == second_frame_release->submission ? uint64_t{606} : uint64_t{707};
            }, [&multi_frame_waits](FrameSubmissionGraph::Queue destination,
                FrameSubmissionGraph::Queue source, uint64_t token) {
                assert(destination == FrameSubmissionGraph::Queue::Graphics);
                assert(source == FrameSubmissionGraph::Queue::Compute && token == 606);
                multi_frame_waits.emplace_back(source, token);
            }, error));
    assert(multi_frame_waits.size() == 1);
    assert(multi_frame_waits[0] == std::make_pair(FrameSubmissionGraph::Queue::Compute, uint64_t{606}));
    FrameImageGraph::ExecutionState second_frame_execution;
    assert(second_frame_graph.begin_execution(second_frame_execution, error));
    for (const auto submission : second_frame_order)
    {
        const auto group = std::find_if(second_frame_groups.begin(), second_frame_groups.end(),
            [submission](const FrameImageGraph::QueueGroup& candidate) {
                return candidate.submission == submission;
            });
        assert(group != second_frame_groups.end());
        assert(second_frame_graph.execute_group(*group, command, states, second_frame_execution,
            {capture}, error, queue_flags_for(group->queue), true));
    }
    const auto& final_transfer = barriers.back();
    assert(final_transfer.srcQueueFamilyIndex == 5);
    assert(final_transfer.dstQueueFamilyIndex == 2);

    // Persistent buffers carry the producer submission and exact imported
    // range into the next frame's external release/acquire handoff.
    const auto multi_frame_buffer = reinterpret_cast<VkBuffer>(uintptr_t(31));
    FrameImageGraph buffer_frame_one;
    FrameImageGraph::BufferResource buffer_frame_one_input, buffer_frame_one_output;
    assert(buffer_frame_one.import_buffer(multi_frame_buffer, 16, 96, BufferUse::HostWrite,
        buffer_frame_one_input, error));
    const auto buffer_frame_one_graphics = buffer_frame_one.add_pass(
        "buffer frame one graphics producer", FrameSubmissionGraph::Queue::Graphics, [] {});
    const auto buffer_frame_one_compute = buffer_frame_one.add_pass(
        "buffer frame one compute reader", FrameSubmissionGraph::Queue::Compute, [] {});
    assert(buffer_frame_one.write_buffer(buffer_frame_one_graphics, buffer_frame_one_input,
        BufferUse::TransferDestination, buffer_frame_one_output, error));
    assert(buffer_frame_one.read_buffer(buffer_frame_one_compute, buffer_frame_one_output,
        BufferUse::ComputeRead, error));
    FrameSubmissionGraph buffer_frame_one_submissions;
    std::vector<FrameImageGraph::QueueGroup> buffer_frame_one_groups;
    std::vector<FrameSubmissionGraph::Pass> buffer_frame_one_mapping;
    assert(buffer_frame_one.generate_submission_groups(buffer_frame_one_submissions,
        family_for_queue, buffer_frame_one_groups, buffer_frame_one_mapping, error));
    assert(buffer_frame_one_groups.size() == 2);
    assert(buffer_frame_one_groups[0].ownership_releases.size() == 1);
    assert(buffer_frame_one_groups[1].ownership_acquires.size() == 1);
    std::vector<FrameSubmissionGraph::Pass> buffer_frame_one_order;
    assert(buffer_frame_one_submissions.execution_order(buffer_frame_one_order, error));
    for (const auto submission : buffer_frame_one_order)
        assert(buffer_frame_one_submissions.submit(submission,
            [submission, &buffer_frame_one_mapping, buffer_frame_one_compute](FrameSubmissionGraph::Queue queue) {
                const bool is_compute = submission == buffer_frame_one_mapping[buffer_frame_one_compute];
                assert(queue == (is_compute ? FrameSubmissionGraph::Queue::Compute :
                    FrameSubmissionGraph::Queue::Graphics));
                return is_compute ? uint64_t{808} : uint64_t{707};
            }, [](FrameSubmissionGraph::Queue destination, FrameSubmissionGraph::Queue source,
                uint64_t token) {
                assert(destination == FrameSubmissionGraph::Queue::Compute);
                assert(source == FrameSubmissionGraph::Queue::Graphics && token == 707);
            }, error));
    const uint64_t buffer_frame_one_compute_token =
        buffer_frame_one_submissions.submission(buffer_frame_one_mapping[buffer_frame_one_compute]);
    assert(buffer_frame_one_compute_token == 808);
    FrameImageGraph::ExecutionState buffer_frame_one_execution;
    assert(buffer_frame_one.begin_execution(buffer_frame_one_execution, error));
    for (const auto submission : buffer_frame_one_order)
    {
        const auto group = std::find_if(buffer_frame_one_groups.begin(), buffer_frame_one_groups.end(),
            [submission](const FrameImageGraph::QueueGroup& candidate) {
                return candidate.submission == submission;
            });
        assert(group != buffer_frame_one_groups.end());
        assert(buffer_frame_one.execute_group(*group, command, states, buffer_frame_one_execution,
            {capture}, error, queue_flags_for(group->queue), true));
    }

    FrameImageGraph buffer_frame_two;
    FrameImageGraph::BufferResource buffer_frame_two_input;
    assert(buffer_frame_two.import_buffer(multi_frame_buffer, 16, 96, BufferUse::ComputeRead,
        buffer_frame_two_input, error, FrameSubmissionGraph::Queue::Compute,
        buffer_frame_one_compute_token, VK_SHARING_MODE_EXCLUSIVE, {}));
    const auto buffer_frame_two_graphics = buffer_frame_two.add_pass(
        "buffer frame two graphics consumer", FrameSubmissionGraph::Queue::Graphics, [] {});
    assert(buffer_frame_two.read_buffer(buffer_frame_two_graphics, buffer_frame_two_input,
        BufferUse::VertexRead, error));
    FrameSubmissionGraph buffer_frame_two_submissions;
    std::vector<FrameImageGraph::QueueGroup> buffer_frame_two_groups;
    std::vector<FrameSubmissionGraph::Pass> buffer_frame_two_mapping;
    assert(buffer_frame_two.generate_submission_groups(buffer_frame_two_submissions,
        family_for_queue, buffer_frame_two_groups, buffer_frame_two_mapping, error));
    const auto buffer_frame_two_release = std::find_if(buffer_frame_two_groups.begin(),
        buffer_frame_two_groups.end(), [](const FrameImageGraph::QueueGroup& group) {
            return group.passes.empty();
        });
    assert(buffer_frame_two_release != buffer_frame_two_groups.end());
    assert(buffer_frame_two_release->queue == FrameSubmissionGraph::Queue::Compute);
    assert(buffer_frame_two_release->external_submission == buffer_frame_one_compute_token);
    assert(buffer_frame_two_release->ownership_releases.size() == 1);
    assert(buffer_frame_two_release->ownership_releases[0].buffer_offset == 16);
    assert(buffer_frame_two_release->ownership_releases[0].buffer_size == 96);
    std::vector<std::pair<FrameSubmissionGraph::Queue, uint64_t>> buffer_frame_two_waits;
    std::vector<FrameSubmissionGraph::Pass> buffer_frame_two_order;
    assert(buffer_frame_two_submissions.execution_order(buffer_frame_two_order, error));
    for (const auto submission : buffer_frame_two_order)
        assert(buffer_frame_two_submissions.submit(submission,
            [submission, buffer_frame_two_release](FrameSubmissionGraph::Queue queue) {
                const bool is_release = submission == buffer_frame_two_release->submission;
                assert(queue == (is_release ? FrameSubmissionGraph::Queue::Compute :
                    FrameSubmissionGraph::Queue::Graphics));
                return is_release ? uint64_t{909} : uint64_t{1001};
            }, [&buffer_frame_two_waits](FrameSubmissionGraph::Queue destination,
                FrameSubmissionGraph::Queue source, uint64_t token) {
                assert(destination == FrameSubmissionGraph::Queue::Graphics);
                assert(source == FrameSubmissionGraph::Queue::Compute && token == 909);
                buffer_frame_two_waits.emplace_back(source, token);
            }, error));
    assert(buffer_frame_two_waits.size() == 1);
    assert(buffer_frame_two_waits[0] ==
        std::make_pair(FrameSubmissionGraph::Queue::Compute, uint64_t{909}));
    FrameImageGraph::ExecutionState buffer_frame_two_execution;
    assert(buffer_frame_two.begin_execution(buffer_frame_two_execution, error));
    const size_t buffer_frame_two_barrier_start = buffer_barriers.size();
    for (const auto submission : buffer_frame_two_order)
    {
        const auto group = std::find_if(buffer_frame_two_groups.begin(), buffer_frame_two_groups.end(),
            [submission](const FrameImageGraph::QueueGroup& candidate) {
                return candidate.submission == submission;
            });
        assert(group != buffer_frame_two_groups.end());
        assert(buffer_frame_two.execute_group(*group, command, states, buffer_frame_two_execution,
            {capture}, error, queue_flags_for(group->queue), true));
    }
    assert(buffer_barriers.size() == buffer_frame_two_barrier_start + 2);
    for (size_t i = buffer_frame_two_barrier_start; i < buffer_barriers.size(); ++i)
    {
        assert(buffer_barriers[i].srcQueueFamilyIndex == 5);
        assert(buffer_barriers[i].dstQueueFamilyIndex == 2);
        assert(buffer_barriers[i].offset == 16);
        assert(buffer_barriers[i].size == 96);
    }

    // GPU-particle input staging uses a generated Copy group when a transfer
    // queue is available, then publishes the uploaded ranges to compute.
    FrameImageGraph copy_upload_graph;
    FrameImageGraph::BufferResource staging_resource, copy_target_initial, copy_target_written;
    assert(copy_upload_graph.import_buffer(reinterpret_cast<VkBuffer>(uintptr_t(30)), 0, 64,
        BufferUse::HostWrite, staging_resource, error, FrameSubmissionGraph::Queue::Copy));
    assert(copy_upload_graph.import_buffer(reinterpret_cast<VkBuffer>(uintptr_t(31)), 0, 64,
        BufferUse::Undefined, copy_target_initial, error, FrameSubmissionGraph::Queue::Copy));
    std::vector<int> copy_upload_order;
    const auto copy_upload = copy_upload_graph.add_checked_command_pass("copy staged particle inputs",
        FrameSubmissionGraph::Queue::Copy,
        [&copy_upload_order](VkCommandBuffer, std::string&) {
            copy_upload_order.push_back(1);
            return true;
        });
    const auto copy_consumer = copy_upload_graph.add_checked_command_pass("consume uploaded particle inputs",
        FrameSubmissionGraph::Queue::Compute,
        [&copy_upload_order](VkCommandBuffer, std::string&) {
            copy_upload_order.push_back(2);
            return true;
        });
    assert(copy_upload_graph.read_buffer(copy_upload, staging_resource,
        BufferUse::TransferSource, error));
    assert(copy_upload_graph.write_buffer(copy_upload, copy_target_initial,
        BufferUse::TransferDestination, copy_target_written, error));
    assert(copy_upload_graph.read_buffer(copy_consumer, copy_target_written,
        BufferUse::ComputeRead, error));
    FrameSubmissionGraph copy_upload_submissions;
    std::vector<FrameImageGraph::QueueGroup> copy_upload_groups;
    std::vector<FrameSubmissionGraph::Pass> copy_upload_mapping;
    assert(copy_upload_graph.generate_submission_groups(copy_upload_submissions,
        family_for_queue, copy_upload_groups, copy_upload_mapping, error));
    assert(copy_upload_groups.size() == 2);
    assert(copy_upload_groups[0].queue == FrameSubmissionGraph::Queue::Copy);
    assert(copy_upload_groups[1].queue == FrameSubmissionGraph::Queue::Compute);
    assert(copy_upload_submissions.depends_on(copy_upload_mapping[copy_consumer],
        copy_upload_mapping[copy_upload], error));
    std::vector<FrameSubmissionGraph::Pass> copy_upload_order_submissions;
    assert(copy_upload_submissions.execution_order(copy_upload_order_submissions, error));
    assert((copy_upload_order_submissions == std::vector<FrameSubmissionGraph::Pass>{
        copy_upload_mapping[copy_upload], copy_upload_mapping[copy_consumer]}));
    uint32_t copy_to_compute_waits = 0;
    for (const auto submission : copy_upload_order_submissions)
        assert(copy_upload_submissions.submit(submission,
            [submission, &copy_upload_mapping, copy_upload](FrameSubmissionGraph::Queue queue) {
                const bool is_copy = submission == copy_upload_mapping[copy_upload];
                assert(queue == (is_copy ? FrameSubmissionGraph::Queue::Copy :
                    FrameSubmissionGraph::Queue::Compute));
                return is_copy ? uint64_t{1201} : uint64_t{1202};
            }, [&copy_to_compute_waits](FrameSubmissionGraph::Queue destination,
                FrameSubmissionGraph::Queue source, uint64_t token) {
                assert(destination == FrameSubmissionGraph::Queue::Compute);
                assert(source == FrameSubmissionGraph::Queue::Copy && token == 1201);
                ++copy_to_compute_waits;
            }, error));
    assert(copy_to_compute_waits == 1);
    FrameImageGraph::ExecutionState copy_upload_execution;
    assert(copy_upload_graph.begin_execution(copy_upload_execution, error));
    const size_t copy_upload_barrier_start = buffer_barriers.size();
    for (const auto submission : copy_upload_order_submissions)
    {
        const auto group = std::find_if(copy_upload_groups.begin(), copy_upload_groups.end(),
            [submission](const FrameImageGraph::QueueGroup& candidate) {
                return candidate.submission == submission;
            });
        assert(group != copy_upload_groups.end());
        const VkQueueFlags queue_flags = group->queue == FrameSubmissionGraph::Queue::Copy ?
            VK_QUEUE_TRANSFER_BIT : VK_QUEUE_COMPUTE_BIT;
        assert(copy_upload_graph.execute_group(*group, command, states, copy_upload_execution,
            {capture}, error, queue_flags, true));
    }
    assert((copy_upload_order == std::vector<int>{1, 2}));
    assert(buffer_barriers.size() >= copy_upload_barrier_start + 3);
    const auto copy_release = std::find_if(buffer_barriers.begin() + copy_upload_barrier_start,
        buffer_barriers.end(), [](const VkBufferMemoryBarrier& barrier) {
            return barrier.srcQueueFamilyIndex == 7 && barrier.dstQueueFamilyIndex == 5;
        });
    assert(copy_release != buffer_barriers.end());
    assert(copy_release->buffer == reinterpret_cast<VkBuffer>(uintptr_t(31)));
    assert(copy_release->offset == 0 && copy_release->size == 64);
    const auto compute_acquire = std::find_if(copy_release + 1, buffer_barriers.end(),
        [](const VkBufferMemoryBarrier& barrier) {
            return barrier.srcQueueFamilyIndex == 7 && barrier.dstQueueFamilyIndex == 5;
        });
    assert(compute_acquire != buffer_barriers.end());
    assert(compute_acquire->buffer == copy_release->buffer &&
        compute_acquire->offset == copy_release->offset && compute_acquire->size == copy_release->size);

    FrameImageGraph unknown_external_image_graph;
    FrameImageGraph::Resource unknown_external_image;
    assert(unknown_external_image_graph.import_image(
        reinterpret_cast<VkImage>(uintptr_t(29)), range, unknown_external_image, error,
        FrameSubmissionGraph::Queue::Compute, 407));
    const auto unknown_external_consumer = unknown_external_image_graph.add_pass(
        "unknown external image state", FrameSubmissionGraph::Queue::Graphics, [] {});
    assert(unknown_external_image_graph.read(unknown_external_consumer,
        unknown_external_image, ImageUse::Sampled, error));
    FrameSubmissionGraph unknown_external_submissions;
    std::vector<FrameImageGraph::QueueGroup> unknown_external_groups;
    std::vector<FrameSubmissionGraph::Pass> unknown_external_mapping;
    assert(!unknown_external_image_graph.generate_submission_groups(
        unknown_external_submissions, family_for_queue,
        unknown_external_groups, unknown_external_mapping, error));
    assert(error.find("initial use") != std::string::npos);
    assert(unknown_external_groups.empty() && unknown_external_mapping.empty());

    // Exclusive imports with a queue-owned initial state cannot be used on a
    // different family unless the caller supplies a producer submission token
    // or performs the release itself.
    FrameImageGraph missing_image_release_graph;
    FrameImageGraph::Resource missing_image_release;
    assert(missing_image_release_graph.import_image(
        reinterpret_cast<VkImage>(uintptr_t(32)), range, missing_image_release, error,
        FrameSubmissionGraph::Queue::Graphics, 0, VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::Sampled));
    const auto missing_image_consumer = missing_image_release_graph.add_pass(
        "missing image producer token", FrameSubmissionGraph::Queue::Compute, [] {});
    assert(missing_image_release_graph.read(missing_image_consumer,
        missing_image_release, ImageUse::ComputeSampled, error));
    FrameSubmissionGraph missing_image_submissions;
    std::vector<FrameImageGraph::QueueGroup> missing_image_groups;
    std::vector<FrameSubmissionGraph::Pass> missing_image_mapping;
    assert(!missing_image_release_graph.generate_submission_groups(
        missing_image_submissions, family_for_queue,
        missing_image_groups, missing_image_mapping, error));
    assert(error.find("needs a caller release") != std::string::npos);
    assert(missing_image_groups.empty() && missing_image_mapping.empty());

    FrameImageGraph missing_buffer_release_graph;
    FrameImageGraph::BufferResource missing_buffer_release;
    assert(missing_buffer_release_graph.import_buffer(
        reinterpret_cast<VkBuffer>(uintptr_t(33)), 0, 64, BufferUse::AllCommandsReadWrite,
        missing_buffer_release, error, FrameSubmissionGraph::Queue::Graphics, 0));
    const auto missing_buffer_consumer = missing_buffer_release_graph.add_pass(
        "missing buffer producer token", FrameSubmissionGraph::Queue::Compute, [] {});
    assert(missing_buffer_release_graph.read_buffer(missing_buffer_consumer,
        missing_buffer_release, BufferUse::ComputeRead, error));
    FrameSubmissionGraph missing_buffer_submissions;
    std::vector<FrameImageGraph::QueueGroup> missing_buffer_groups;
    std::vector<FrameSubmissionGraph::Pass> missing_buffer_mapping;
    assert(!missing_buffer_release_graph.generate_submission_groups(
        missing_buffer_submissions, family_for_queue,
        missing_buffer_groups, missing_buffer_mapping, error));
    assert(error.find("needs a caller release") != std::string::npos);
    assert(missing_buffer_groups.empty() && missing_buffer_mapping.empty());

    // Buffer-to-image dependencies share one pass order.
    const auto mixed_image = reinterpret_cast<VkImage>(uintptr_t(8));
    const auto mixed_buffer = reinterpret_cast<VkBuffer>(uintptr_t(22));
    assert(states.register_image(mixed_image, range, ImageUse::Undefined, error));
    FrameImageGraph mixed;
    FrameImageGraph::Resource image_initial, image_final;
    FrameImageGraph::BufferResource buffer_initial, buffer_final;
    assert(mixed.import_image(mixed_image, range, image_initial, error));
    assert(mixed.import_buffer(mixed_buffer, 0, 64, BufferUse::Undefined, buffer_initial, error));
    std::vector<int> mixed_order;
    const auto output = mixed.add_pass("image output", [&] { mixed_order.push_back(3); });
    const auto convert = mixed.add_pass("buffer to image", [&] { mixed_order.push_back(2); });
    const auto fill = mixed.add_pass("buffer fill", [&] { mixed_order.push_back(1); });
    assert(mixed.write_buffer(fill, buffer_initial, BufferUse::ComputeWrite, buffer_final, error));
    assert(mixed.read_buffer(convert, buffer_final, BufferUse::ComputeRead, error));
    assert(mixed.write(convert, image_initial, ImageUse::TransferDestination, image_final, error));
    assert(mixed.read(output, image_final, ImageUse::Sampled, error));
    assert(mixed.execute(command, states, {capture}, error));
    assert((mixed_order == std::vector<int>{1, 2, 3}));
}
