#include "VsmPageRender.h"
#include "BufferResource.h"
#include <algorithm>
#include <cmath>

namespace xray::render::vulkan
{
bool read_vsm_static_jobs(const BufferResource& work, uint32_t axis, uint32_t levels,
    uint32_t budget, uint32_t atlas_slots, std::vector<VsmPageJob>& jobs, std::string& error)
{
    jobs.clear();
    const uint64_t page_count = uint64_t(axis) * axis * levels;
    if (!axis || axis > 32768 || !levels || levels > 6 || page_count > UINT32_MAX ||
        budget > page_count || work.size() < 16 + uint64_t(budget) * 16)
    { error = "invalid VSM static job buffer dimensions"; return false; }
    uint32_t header[4]{};
    if (!work.read(0, header, sizeof(header), error)) return false;
    if (header[1]) { error = "VSM static caster binning overflowed"; return false; }
    const uint32_t count = std::min(header[0], budget);
    std::vector<std::array<uint32_t, 4>> entries(count);
    if (count && !work.read(16, entries.data(), entries.size() * sizeof(entries[0]), error)) return false;
    std::vector<uint8_t> seen(atlas_slots, 0);
    const uint32_t per_level = axis * axis;
    for (const auto& entry : entries)
    {
        if (entry[0] >= atlas_slots || entry[1] >= page_count || seen[entry[0]] ||
            (entry[2] != 1 && entry[2] != 2))
        { jobs.clear(); error = "invalid VSM static page job"; return false; }
        seen[entry[0]] = 1;
        const uint32_t within = entry[1] % per_level;
        jobs.push_back({entry[0], entry[1] / per_level, within % axis, within / axis});
    }
    error.clear(); return true;
}

bool read_vsm_dynamic_jobs(const BufferResource& counters, const BufferResource& pages,
    uint32_t axis, uint32_t levels, uint32_t atlas_slots, std::vector<VsmPageJob>& jobs, std::string& error)
{
    jobs.clear();
    const uint64_t count = uint64_t(axis) * axis * levels;
    if (!axis || axis > 65535 || !levels || levels > 65535 || count > UINT32_MAX ||
        counters.size() < (uint64_t(levels) + 1) * 4 || pages.size() < uint64_t(atlas_slots) * 16)
    { error = "invalid VSM dynamic job buffer dimensions"; return false; }
    std::vector<uint32_t> counts(size_t(levels) + 1);
    if (!counters.read(0, counts.data(), counts.size() * sizeof(counts[0]), error)) return false;
    const uint32_t accepted = std::min(counts[0], atlas_slots);
    uint64_t counted = 0;
    for (uint32_t level = 0; level < levels; ++level)
        counted += counts[level + 1];
    if (counts[0] > count || counted != accepted)
    { error = "inconsistent VSM dynamic page counters"; return false; }
    std::vector<std::array<uint32_t, 4>> entries(accepted);
    if (accepted && !pages.read(0, entries.data(), entries.size() * sizeof(entries[0]), error)) return false;
    std::vector<uint32_t> virtual_pages;
    virtual_pages.reserve(accepted);
    for (uint32_t slot = 0; slot < accepted; ++slot)
    {
        const auto& entry = entries[slot];
        if (entry[0] >= levels || entry[1] >= axis || entry[2] >= axis || !counts[entry[0] + 1])
        { error = "invalid VSM dynamic page job"; return false; }
        --counts[entry[0] + 1];
        virtual_pages.push_back((entry[0] * axis + entry[2]) * axis + entry[1]);
    }
    std::sort(virtual_pages.begin(), virtual_pages.end());
    if (std::adjacent_find(virtual_pages.begin(), virtual_pages.end()) != virtual_pages.end())
    { error = "duplicate VSM dynamic page job"; return false; }
    jobs.reserve(accepted);
    for (uint32_t slot = 0; slot < accepted; ++slot)
        jobs.push_back({slot, entries[slot][0], entries[slot][1], entries[slot][2]});
    error.clear(); return true;
}

bool record_vsm_pages(const VsmAtlas& atlas, const FrameRecordingContext& frame,
    const VsmPageProjection& projection, const std::vector<VsmPageJob>& jobs,
    VsmPageDraw draw, void* user, std::string& error, VsmPagePrepare prepare)
{
    if (!frame.command_buffer || !atlas.render_pass() || !draw || !projection.axis || projection.axis > 32768 ||
        !projection.levels || projection.levels > 6 || !std::isfinite(projection.near_depth) ||
        !std::isfinite(projection.far_depth) || projection.far_depth <= projection.near_depth ||
        !std::all_of(projection.light_view, projection.light_view + 16, [](float v) { return std::isfinite(v); }))
    { error = "invalid VSM page render request"; return false; }
    std::vector<std::array<float, 16>> matrices;
    matrices.reserve(jobs.size());
    for (const auto& job : jobs)
    {
        if (job.slot >= atlas.page_count() || job.level >= projection.levels ||
            job.x >= projection.axis || job.y >= projection.axis)
        { error = "VSM page render job is outside the atlas or clipmap"; return false; }
        const auto& level = projection.level[job.level];
        if (!std::isfinite(level[0]) || !std::isfinite(level[1]) || !std::isfinite(level[2]) || level[2] <= 0.f)
        { error = "invalid VSM page render clipmap"; return false; }
        const float width = level[2] / float(projection.axis);
        const float x = level[0] + float(job.x) * width;
        const float y = level[1] + float(job.y) * width;
        const float sx = 2.f / width, sy = 2.f / width, sz = 1.f / (projection.far_depth - projection.near_depth);
        const float tx = -1.f - sx * x, ty = -1.f - sy * y, tz = -projection.near_depth * sz;
        std::array<float, 16> matrix{};
        for (unsigned c = 0; c < 4; ++c)
        {
            const float* col = projection.light_view + c * 4;
            matrix[c * 4] = sx * col[0] + tx * col[3];
            matrix[c * 4 + 1] = sy * col[1] + ty * col[3];
            matrix[c * 4 + 2] = sz * col[2] + tz * col[3];
            matrix[c * 4 + 3] = col[3];
        }
        if (!std::all_of(matrix.begin(), matrix.end(), [](float v) { return std::isfinite(v); }))
        { error = "VSM page projection is non-finite"; return false; }
        matrices.push_back(matrix);
    }
    for (size_t i = 0; i < jobs.size(); ++i)
    {
        // Prepare visibility before begin_page enters the atlas render pass;
        // compute dispatches cannot be recorded inside the shadow pass.
        FrameRecordingContext prepare_frame = frame;
        prepare_frame.extent = {atlas.page_size(), atlas.page_size()};
        if (prepare && !prepare(prepare_frame, matrices[i].data(), jobs[i], user, error))
        { if (error.empty()) error = "VSM page visibility preparation failed"; return false; }
        FrameRecordingContext page_frame{};
        if (!atlas.begin_page(frame, jobs[i].slot, page_frame, error)) return false;
        const bool recorded = draw(page_frame, matrices[i].data(), jobs[i], user, error);
        atlas.end_page(frame.command_buffer);
        if (!recorded) { if (error.empty()) error = "VSM page draw failed"; return false; }
    }
    error.clear(); return true;
}
}
