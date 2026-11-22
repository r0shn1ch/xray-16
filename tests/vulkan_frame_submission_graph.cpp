#include "src/Layers/xrRenderVK/FrameContext.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
template <typename T> T handle(uintptr_t value)
{
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}
}

int main()
{
    FrameSubmissionGraph graph;
    const auto first = graph.add_pass("first", FrameSubmissionGraph::Queue::Compute);
    const auto second = graph.add_pass("second", FrameSubmissionGraph::Queue::Graphics);
    const auto third = graph.add_pass("third", FrameSubmissionGraph::Queue::Graphics);
    std::string error;
    assert(FrameSubmissionGraph::is_fixed_submission_pass(first, first, second));
    assert(FrameSubmissionGraph::is_fixed_submission_pass(second, first, second));
    assert(!FrameSubmissionGraph::is_fixed_submission_pass(third, first, second));
    assert(graph.depends_on(second, first, error));
    assert(graph.depends_on(third, second, error));
    assert(graph.depends_on(third, first, error));
    assert(graph.depends_on(third, first, error)); // duplicate edge is idempotent
    assert(!graph.depends_on(first, third, error)); // reject a cycle before submission

    std::vector<FrameSubmissionGraph::Pass> order;
    assert(graph.execution_order(order, error));
    assert((order == std::vector<FrameSubmissionGraph::Pass>{first, second, third}));

    const auto external = graph.add_pass("external wait", FrameSubmissionGraph::Queue::Graphics);
    assert(graph.depends_on_submission(external, FrameSubmissionGraph::Queue::Compute, 17, error));
    assert(graph.depends_on_submission(external, FrameSubmissionGraph::Queue::Compute, 17, error));
    uint32_t wait_count = 0;
    assert(graph.submit(external,
        [](FrameSubmissionGraph::Queue) { return uint64_t{29}; },
        [&wait_count](FrameSubmissionGraph::Queue destination,
            FrameSubmissionGraph::Queue source, uint64_t submission) {
            assert(destination == FrameSubmissionGraph::Queue::Graphics);
            assert(source == FrameSubmissionGraph::Queue::Compute);
            assert(submission == 17);
            ++wait_count;
        }, error));
    assert(wait_count == 1);
    assert(graph.submission(external) == 29);
    assert(graph.submit(first, [](FrameSubmissionGraph::Queue) { return uint64_t{17}; }, {}, error));
    assert(graph.depends_on_submission(second, FrameSubmissionGraph::Queue::Compute, 17, error));
    uint32_t shared_wait_count = 0;
    const auto wait = [&](FrameSubmissionGraph::Queue destination,
        FrameSubmissionGraph::Queue source, uint64_t submission) {
        assert(destination == FrameSubmissionGraph::Queue::Graphics);
        assert(source == FrameSubmissionGraph::Queue::Compute && submission == 17);
        ++shared_wait_count;
    };
    assert(graph.prepare(second, wait, error));
    assert(graph.prepare(second, wait, error));
    assert(shared_wait_count == 1);
    const auto late = graph.add_pass("late producer", FrameSubmissionGraph::Queue::Compute);
    const auto consumer = graph.add_pass("pending consumer", FrameSubmissionGraph::Queue::Graphics);
    assert(graph.depends_on_submission(consumer, FrameSubmissionGraph::Queue::Compute, 17, error));
    assert(graph.depends_on(consumer, late, error));
    assert(!graph.prepare(consumer, wait, error));
    assert(shared_wait_count == 1);
    assert(graph.submit(late, [](FrameSubmissionGraph::Queue) { return uint64_t{17}; }, {}, error));
    assert(graph.prepare(consumer, wait, error));
    assert(shared_wait_count == 2);
    assert(graph.remove_dependency_on_submission(consumer, FrameSubmissionGraph::Queue::Compute, 17, error));
    assert(graph.prepare(consumer, wait, error));
    assert(shared_wait_count == 2);

    FrameRecordingContext source{};
    source.command_buffer = handle<VkCommandBuffer>(101);
    source.compute_command_buffer = handle<VkCommandBuffer>(102);
    source.render_pass = handle<VkRenderPass>(103);
    source.framebuffer = handle<VkFramebuffer>(104);
    source.extent = {800, 600};
    source.image_index = 2;
    source.frame_index = 1;
    source.mip_level = 3;
    source.viewport_offset = {7, 9};
    source.asynchronous_copy_available = true;
    source.semaphore_handoff = true;
    source.graphics_queue_flags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
    source.compute_queue_flags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    source.copy_queue_flags = VK_QUEUE_TRANSFER_BIT;
    source.graphics_queue_family = 4;
    source.compute_queue_family = 5;
    source.copy_queue_family = 6;
    source.previous_graphics_submission = 77;
    source.submission_graph = &graph;
    source.compute_submission_pass = first;
    source.graphics_prepass_pass = second;
    source.graphics_continuation_pass = third;
    source.request_submission_command_buffer = [](auto, auto, auto&, auto&) { return true; };
    source.geometry_motion_submission_pass = external;
    auto ray_visibility = external;
    source.ray_visibility_submission_pass = &ray_visibility;

    const VkExtent2D generated_extent{320, 180};
    const VkOffset2D generated_offset{11, 13};
    const auto generated = make_frame_recording_context(source,
        handle<VkCommandBuffer>(201), handle<VkRenderPass>(202), handle<VkFramebuffer>(203),
        generated_extent, 4, generated_offset, FrameSubmissionGraph::Queue::Compute);
    assert(generated.command_buffer == handle<VkCommandBuffer>(201));
    assert(generated.compute_command_buffer == VK_NULL_HANDLE);
    assert(generated.render_pass == handle<VkRenderPass>(202));
    assert(generated.framebuffer == handle<VkFramebuffer>(203));
    assert(generated.extent.width == generated_extent.width && generated.extent.height == generated_extent.height);
    assert(generated.image_index == source.image_index && generated.frame_index == source.frame_index);
    assert(generated.mip_level == 4 && generated.viewport_offset.x == 11 && generated.viewport_offset.y == 13);
    assert(generated.command_queue == FrameSubmissionGraph::Queue::Compute);
    assert(generated.command_queue_flags == source.compute_queue_flags);
    assert(generated.graphics_queue_flags == source.graphics_queue_flags);
    assert(generated.compute_queue_family == source.compute_queue_family);
    assert(generated.copy_queue_family == source.copy_queue_family);
    assert(generated.asynchronous_copy_available && generated.semaphore_handoff);
    assert(generated.previous_graphics_submission == 77);
    assert(!generated.submission_graph && !generated.request_submission_command_buffer);
    assert(generated.compute_submission_pass == FrameSubmissionGraph::InvalidPass);
    assert(generated.graphics_prepass_pass == FrameSubmissionGraph::InvalidPass);
    assert(generated.graphics_continuation_pass == FrameSubmissionGraph::InvalidPass);
    assert(generated.geometry_motion_submission_pass == FrameSubmissionGraph::InvalidPass);
    assert(!generated.ray_visibility_submission_pass);

    const auto copy_recording = make_frame_recording_context(source,
        handle<VkCommandBuffer>(301), VK_NULL_HANDLE, VK_NULL_HANDLE, generated_extent,
        0, {}, FrameSubmissionGraph::Queue::Copy);
    assert(copy_recording.command_queue_flags == source.copy_queue_flags);
}
