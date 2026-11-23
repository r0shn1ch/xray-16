#include "xrEngine/stdafx.h"
#include "VsmPageRender.h"
#include "GpuLevel.h"
#include "DeferredPass.h"
#include <algorithm>

namespace xray::render::vulkan
{
bool prepare_vsm_level_page(const FrameRecordingContext& frame, const float* light_view_projection,
    const VsmPageJob&, void* user, std::string& error)
{
    auto* context = static_cast<VsmLevelPageContext*>(user);
    if (!context || !context->level || !light_view_projection)
    { error = "VSM page visibility needs level geometry"; return false; }
    context->use_cluster_visibility = context->include_static && context->level->has_cluster_visibility();
    if (!context->use_cluster_visibility)
    { error.clear(); return true; }
    float matrix[16];
    std::copy_n(light_view_projection, 16, matrix);
    return context->level->record_cluster_shadow_visibility(frame, matrix, error);
}

bool record_vsm_level_page(const FrameRecordingContext& frame, const float* light_view_projection,
    const VsmPageJob&, void* user, std::string& error)
{
    const auto* context = static_cast<const VsmLevelPageContext*>(user);
    if (!context || !context->level || !context->pass || !light_view_projection)
    { error = "VSM level page needs level geometry and a shadow pipeline"; return false; }
    float matrix[16];
    std::copy_n(light_view_projection, 16, matrix);
    if (!context->level->record_sun_shadow(frame, *context->pass, matrix,
            context->include_static, context->include_trees, context->use_cluster_visibility))
    { error = "VSM level shadow geometry recording failed"; return false; }
    error.clear(); return true;
}
}
