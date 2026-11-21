#include "xrEngine/stdafx.h"
#include "GpuParticleCommands.h"
#include <algorithm>

namespace xray::render::vulkan
{
bool GpuParticleCommands::begin_generation(uint32_t root, std::string& error)
{
    if (root >= roots_.size() || root > UINT32_MAX / 2)
    { error = "invalid GPU particle root"; return false; }
    auto& r = roots_[root];
    if (!r.pending.empty() || r.generation == UINT32_MAX)
    { error = "GPU particle root still has pending commands or exhausted generations"; return false; }
    ++r.generation; r.serial = r.submitted = r.acknowledged = 0;
    error.clear(); return true;
}
bool GpuParticleCommands::enqueue(uint32_t root, fg::GpuPapiCommand command, std::string& error)
{
    if (root >= roots_.size() || root > UINT32_MAX / 2 || !roots_[root].generation ||
        roots_[root].serial == UINT32_MAX || command.type > 7)
    { error = "invalid GPU particle command or root"; return false; }
    auto& r = roots_[root];
    command.emitter = root * 2; command.generation = r.generation; command.serial = r.serial + 1;
    r.pending.push_back(command); ++r.serial;
    error.clear(); return true;
}
bool GpuParticleCommands::snapshot(Batch& batch, std::string& error)
{
    Batch result;
    for (const auto& r : roots_)
    {
        if (r.pending.empty()) continue;
        if (r.pending.size() > UINT32_MAX - result.commands.size())
        { error = "GPU particle command batch overflow"; return false; }
        result.ranges.push_back({static_cast<uint32_t>(result.commands.size()), static_cast<uint32_t>(r.pending.size())});
        result.commands.insert(result.commands.end(), r.pending.begin(), r.pending.end());
    }
    for (auto& r : roots_) if (!r.pending.empty()) r.submitted = r.pending.back().serial;
    batch = std::move(result); error.clear(); return true;
}
bool GpuParticleCommands::acknowledge(uint32_t root, uint32_t generation, uint32_t serial, std::string& error)
{
    if (root >= roots_.size()) { error = "invalid GPU particle root"; return false; }
    auto& r = roots_[root];
    if (generation != r.generation) { error.clear(); return true; }
    if (serial > r.submitted)
    { error = "GPU particle status acknowledges an unsubmitted command"; return false; }
    if (serial > r.acknowledged)
    {
        const auto end = std::upper_bound(r.pending.begin(), r.pending.end(), serial,
            [](uint32_t value, const fg::GpuPapiCommand& item) { return value < item.serial; });
        r.pending.erase(r.pending.begin(), end); r.acknowledged = serial;
    }
    error.clear(); return true;
}
bool GpuParticleCommands::acknowledge(const std::vector<fg::GpuPapiStatus>& statuses, std::string& error)
{
    if (statuses.size() != roots_.size()) { error = "GPU particle status root count mismatch"; return false; }
    for (size_t i = 0; i < statuses.size(); ++i)
        if (statuses[i].generation == roots_[i].generation && statuses[i].serial > roots_[i].submitted)
        { error = "GPU particle status acknowledges an unsubmitted command"; return false; }
    for (size_t i = 0; i < statuses.size(); ++i)
        if (!acknowledge(static_cast<uint32_t>(i), statuses[i].generation, statuses[i].serial, error)) return false;
    error.clear(); return true;
}
bool GpuParticleCommands::state(uint32_t root, uint32_t& generation, uint32_t& serial) const
{
    if (root >= roots_.size()) return false;
    generation = roots_[root].generation; serial = roots_[root].serial; return true;
}
}
