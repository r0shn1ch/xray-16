#pragma once
#include "VsmAtlas.h"
#include <array>
#include <vector>

namespace xray::render::vulkan
{
class GpuLevel;
class DeferredPass;
class BufferResource;
struct VsmPageJob { uint32_t slot{}, level{}, x{}, y{}; };
struct VsmPageProjection
{
    float light_view[16]{}; // column-major world-to-light transform
    float level[6][4]{};   // light-space origin xy, full width, reserved
    float near_depth{}, far_depth{};
    uint32_t axis{}, levels{};
};
using VsmPageDraw = bool (*)(const FrameRecordingContext& frame, const float* light_view_projection,
    const VsmPageJob& job, void* user, std::string& error);
using VsmPagePrepare = bool (*)(const FrameRecordingContext& frame, const float* light_view_projection,
    const VsmPageJob& job, void* user, std::string& error);

// Caller supplies completed/validated jobs and holds the atlas write lease.
// Pages must only be published if this function and the submission succeed.
bool record_vsm_pages(const VsmAtlas& atlas, const FrameRecordingContext& frame,
    const VsmPageProjection& projection, const std::vector<VsmPageJob>& jobs,
    VsmPageDraw draw, void* user, std::string& error, VsmPagePrepare prepare = nullptr);

// Read after the producing submission's fence has completed. The work buffers
// must remain unchanged until the corresponding pages have been drawn and, for
// static pages, published with the same cache parameters.
bool read_vsm_static_jobs(const BufferResource& work, uint32_t axis, uint32_t levels,
    uint32_t budget, uint32_t atlas_slots, std::vector<VsmPageJob>& jobs, std::string& error);
bool read_vsm_dynamic_jobs(const BufferResource& counters, const BufferResource& pages,
    uint32_t axis, uint32_t levels, uint32_t atlas_slots, std::vector<VsmPageJob>& jobs,
    std::string& error);

struct VsmLevelPageContext
{
    const GpuLevel* level{};
    const DeferredPass* pass{};
    bool include_static{true}, include_trees{true};
    bool use_cluster_visibility{};
};
bool prepare_vsm_level_page(const FrameRecordingContext& frame, const float* light_view_projection,
    const VsmPageJob& job, void* user, std::string& error);
bool record_vsm_level_page(const FrameRecordingContext& frame, const float* light_view_projection,
    const VsmPageJob& job, void* user, std::string& error);
}
