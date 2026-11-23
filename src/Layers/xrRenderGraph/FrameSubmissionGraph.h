#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <exception>
#include <functional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace xray::render::framegraph
{
// Orders command-list submissions and emits the cross-queue waits declared
// between them. Submission tokens are increasing timeline values per queue.
class FrameSubmissionGraph
{
public:
    enum class Queue : uint8_t { Graphics, Compute, Copy };
    using QueueFamily = uint32_t;
    using Pass = uint32_t;
    static constexpr Pass InvalidPass = UINT32_MAX;
    static constexpr QueueFamily InvalidQueueFamily = UINT32_MAX;
    using QueueWait = std::function<void(Queue destination, Queue source, uint64_t submission)>;
    using Execute = std::function<uint64_t(Queue queue)>;

    static bool is_fixed_submission_pass(Pass candidate, Pass compute_submission,
        Pass graphics_prepass)
    {
        return candidate == compute_submission || candidate == graphics_prepass;
    }

    Pass add_pass(const char* name, Queue queue,
        QueueFamily queue_family = InvalidQueueFamily)
    {
        const Pass pass = static_cast<Pass>(passes_.size());
        passes_.push_back({name ? name : "", queue, queue_family, {}, 0});
        return pass;
    }

    bool depends_on(Pass consumer, Pass producer, std::string& error)
    {
        if (!valid(consumer) || !valid(producer) || consumer == producer)
        {
            error = "Frame submission graph: invalid pass dependency";
            return false;
        }
        auto& dependencies = passes_[consumer].dependencies;
        for (const Dependency& dependency : dependencies)
            if (!dependency.external && dependency.producer == producer)
            {
                error.clear();
                return true;
            }
        if (passes_[consumer].submission)
        {
            error = "Frame submission graph: cannot add a dependency to a submitted pass";
            return false;
        }
        if (has_path(consumer, producer))
        {
            error = "Frame submission graph: dependency would create a cycle";
            return false;
        }
        dependencies.push_back({producer});
        error.clear();
        return true;
    }

    bool depends_on_submission(Pass consumer, Queue source_queue, uint64_t submission,
        std::string& error)
    {
        if (!valid(consumer))
        {
            error = "Frame submission graph: invalid consumer pass";
            return false;
        }
        if (submission)
        {
            auto& dependencies = passes_[consumer].dependencies;
            for (const Dependency& dependency : dependencies)
                if (dependency.external && dependency.source_queue == source_queue &&
                    dependency.submission == submission)
                {
                    error.clear();
                    return true;
                }
            if (passes_[consumer].submission)
            {
                error = "Frame submission graph: cannot add a dependency to a submitted pass";
                return false;
            }
            dependencies.push_back({InvalidPass, source_queue, submission, true});
        }
        error.clear();
        return true;
    }

    // Redirects a wait on an external producer through an in-graph handoff
    // submission. This prevents a binary producer semaphore from being waited
    // by both the release and consumer submissions.
    bool remove_dependency_on_submission(Pass consumer, Queue source_queue, uint64_t submission,
        std::string& error)
    {
        if (!valid(consumer) || !submission)
        {
            error = "Frame submission graph: invalid external dependency removal";
            return false;
        }
        if (passes_[consumer].submission)
        {
            error = "Frame submission graph: cannot remove a dependency from a submitted pass";
            return false;
        }
        auto& dependencies = passes_[consumer].dependencies;
        const auto index = static_cast<size_t>(source_queue);
        if (index >= passes_[consumer].queued_waits.size())
        {
            error = "Frame submission graph: invalid dependency queue";
            return false;
        }
        const auto matching = std::find_if(dependencies.begin(), dependencies.end(),
            [source_queue, submission](const Dependency& dependency) {
                return dependency.external && dependency.source_queue == source_queue &&
                    dependency.submission == submission;
            });
        if (matching != dependencies.end() && passes_[consumer].queued_waits[index] >= submission)
        {
            error = "Frame submission graph: cannot remove a dependency after its queue wait was prepared";
            return false;
        }
        dependencies.erase(std::remove_if(dependencies.begin(), dependencies.end(),
            [source_queue, submission](const Dependency& dependency) {
                return dependency.external && dependency.source_queue == source_queue &&
                    dependency.submission == submission;
            }), dependencies.end());
        error.clear();
        return true;
    }

    // Returns a stable topological order without submitting any command lists.
    bool execution_order(std::vector<Pass>& order, std::string& error) const
    {
        std::vector<std::vector<Pass>> outgoing(passes_.size());
        std::vector<uint32_t> indegree(passes_.size());
        for (Pass consumer = 0; consumer < passes_.size(); ++consumer)
        {
            for (const Dependency& dependency : passes_[consumer].dependencies)
            {
                if (dependency.external)
                    continue;
                if (!valid(dependency.producer) || dependency.producer == consumer)
                {
                    error = "Frame submission graph: invalid pass dependency";
                    return false;
                }
                outgoing[dependency.producer].push_back(consumer);
                ++indegree[consumer];
            }
        }

        std::set<Pass> ready;
        for (Pass pass = 0; pass < passes_.size(); ++pass)
            if (!indegree[pass])
                ready.insert(pass);
        std::vector<Pass> result;
        result.reserve(passes_.size());
        while (!ready.empty())
        {
            const Pass pass = *ready.begin();
            ready.erase(ready.begin());
            result.push_back(pass);
            for (Pass consumer : outgoing[pass])
                if (!--indegree[consumer])
                    ready.insert(consumer);
        }
        if (result.size() != passes_.size())
        {
            error = "Frame submission graph: cyclic pass dependencies";
            return false;
        }
        order = std::move(result);
        error.clear();
        return true;
    }

    bool prepare(Pass consumer, const QueueWait& queue_wait, std::string& error)
    {
        if (!valid(consumer))
        {
            error = "Frame submission graph: invalid consumer pass";
            return false;
        }
        PassInfo& consumer_info = passes_[consumer];
        // Validate the whole pass before changing the destination queue.
        for (const Dependency& dependency : consumer_info.dependencies)
        {
            if (!dependency.external &&
                (!valid(dependency.producer) || !passes_[dependency.producer].submission))
            {
                error = "Frame submission graph: dependency submitted after its consumer: " +
                    consumer_info.name;
                return false;
            }
        }
        std::array<uint64_t, 3> required_waits{};
        for (const Dependency& dependency : consumer_info.dependencies)
        {
            Queue source_queue = dependency.source_queue;
            uint64_t submission = dependency.submission;
            if (!dependency.external)
            {
                source_queue = passes_[dependency.producer].queue;
                submission = passes_[dependency.producer].submission;
            }
            if (source_queue != consumer_info.queue)
            {
                const auto index = static_cast<size_t>(source_queue);
                if (index >= required_waits.size())
                {
                    error = "Frame submission graph: invalid dependency queue for " + consumer_info.name;
                    return false;
                }
                auto& required = required_waits[index];
                required = std::max(required, submission);
            }
        }
        for (size_t index = 0; index < required_waits.size(); ++index)
        {
            const uint64_t submission = required_waits[index];
            auto& queued = consumer_info.queued_waits[index];
            if (submission <= queued)
                continue;
            if (!queue_wait)
            {
                error = "Frame submission graph: cross-queue dependency has no wait operation";
                return false;
            }
            try
            {
                queue_wait(consumer_info.queue, static_cast<Queue>(index), submission);
                queued = submission;
            }
            catch (const std::exception& exception)
            {
                error = "Frame submission graph: queue wait failed for " + consumer_info.name +
                    ": " + exception.what();
                return false;
            }
        }
        error.clear();
        return true;
    }

    bool submit(Pass pass, const Execute& execute, const QueueWait& queue_wait,
        std::string& error)
    {
        if (!valid(pass) || !execute)
        {
            error = "Frame submission graph: invalid pass submission";
            return false;
        }
        PassInfo& info = passes_[pass];
        if (info.submission)
        {
            error = "Frame submission graph: pass submitted more than once: " + info.name;
            return false;
        }
        if (!prepare(pass, queue_wait, error))
            return false;
        try
        {
            info.submission = execute(info.queue);
        }
        catch (const std::exception& exception)
        {
            error = "Frame submission graph: command-list submission failed for " + info.name +
                ": " + exception.what();
            return false;
        }
        if (!info.submission)
        {
            error = "Frame submission graph: command-list submission returned no token: " + info.name;
            return false;
        }
        error.clear();
        return true;
    }

    uint64_t submission(Pass pass) const
    {
        return valid(pass) ? passes_[pass].submission : 0;
    }
    Queue queue(Pass pass) const
    {
        return valid(pass) ? passes_[pass].queue : Queue::Graphics;
    }
    QueueFamily queue_family(Pass pass) const
    {
        return valid(pass) ? passes_[pass].queue_family : InvalidQueueFamily;
    }

private:
    bool has_path(Pass from, Pass to) const
    {
        std::vector<Pass> pending{from};
        std::vector<bool> visited(passes_.size());
        visited[from] = true;
        while (!pending.empty())
        {
            const Pass current = pending.back();
            pending.pop_back();
            for (Pass consumer = 0; consumer < passes_.size(); ++consumer)
            {
                for (const Dependency& dependency : passes_[consumer].dependencies)
                {
                    if (dependency.external || dependency.producer != current)
                        continue;
                    if (consumer == to)
                        return true;
                    if (!visited[consumer])
                    {
                        visited[consumer] = true;
                        pending.push_back(consumer);
                    }
                }
            }
        }
        return false;
    }

    struct Dependency
    {
        Pass producer = InvalidPass;
        Queue source_queue = Queue::Graphics;
        uint64_t submission{};
        bool external{};
    };
    struct PassInfo
    {
        std::string name;
        Queue queue = Queue::Graphics;
        QueueFamily queue_family = InvalidQueueFamily;
        std::vector<Dependency> dependencies;
        uint64_t submission{};
        std::array<uint64_t, 3> queued_waits{};
    };

    bool valid(Pass pass) const { return pass < passes_.size(); }
    std::vector<PassInfo> passes_;
};
}
