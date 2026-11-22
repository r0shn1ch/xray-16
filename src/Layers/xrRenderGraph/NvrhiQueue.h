#pragma once

#include "FrameSubmissionGraph.h"
#include <nvrhi/nvrhi.h>
#include <stdexcept>

namespace xray::render::framegraph
{
inline nvrhi::CommandQueue to_nvrhi_queue(FrameSubmissionGraph::Queue queue)
{
    switch (queue)
    {
    case FrameSubmissionGraph::Queue::Graphics: return nvrhi::CommandQueue::Graphics;
    case FrameSubmissionGraph::Queue::Compute: return nvrhi::CommandQueue::Compute;
    case FrameSubmissionGraph::Queue::Copy: return nvrhi::CommandQueue::Copy;
    }
    throw std::invalid_argument("Invalid render graph queue");
}

inline FrameSubmissionGraph::QueueWait nvrhi_queue_wait(nvrhi::IDevice* device)
{
    return [device](FrameSubmissionGraph::Queue destination, FrameSubmissionGraph::Queue source,
               uint64_t submission) {
        if (!device) throw std::runtime_error("NVRHI device is unavailable");
        device->queueWaitForCommandList(to_nvrhi_queue(destination), to_nvrhi_queue(source), submission);
    };
}

inline nvrhi::CommandListHandle create_nvrhi_command_list(nvrhi::IDevice& device,
    FrameSubmissionGraph::Queue queue)
{
    nvrhi::CommandListParameters parameters;
    parameters.enableImmediateExecution = false;
    parameters.queueType = to_nvrhi_queue(queue);
    return device.createCommandList(parameters);
}

inline uint64_t submit_nvrhi_command_list(nvrhi::IDevice& device, nvrhi::ICommandList* list,
    FrameSubmissionGraph::Queue queue)
{
    return device.executeCommandList(list, to_nvrhi_queue(queue));
}
}
