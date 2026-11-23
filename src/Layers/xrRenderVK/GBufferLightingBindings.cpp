#include "GBufferTargets.h"
#include "SunShadowTargets.h"

namespace xray::render::vulkan
{
bool GBufferTargets::bind_lighting(DeferredPass& deferred, std::string& error)
{
    if (!device_ || !sampler_ || targets_.empty())
    {
        error = "G-buffer targets are not initialized";
        return false;
    }
    for (auto& target : targets_)
    {
        if (!target.lighting && !deferred.gbuffer(target.albedo.view, target.normal.view,
                target.sampled_depth.view, target.surface.view, sampler_, target.lighting, error))
            return false;
        if (samples_ != VK_SAMPLE_COUNT_1_BIT && !target.multisampled_lighting &&
            !deferred.gbuffer_multisampled(target.multisampled[0].view, target.multisampled[1].view,
                target.multisampled[MultisampledDepthAttachment].view, target.multisampled[2].view,
                target.sampled_depth.view,
                target.normal.view, sampler_, target.multisampled_lighting, error))
            return false;
        if (target.multisampled_lighting)
            deferred.pair_gbuffer_sets(target.lighting, target.multisampled_lighting);
    }
    error.clear();
    return true;
}

void GBufferTargets::bind_sun_shadow(DeferredPass& deferred, const SunShadowTargets& shadow)
{
    for (uint32_t i = 0; i < targets_.size(); ++i)
    {
        deferred.bind_sun_shadow(targets_[i].lighting, shadow.view(i), shadow.sampler(), shadow.uniform(i));
    }
}

void GBufferTargets::bind_rain_shadow(DeferredPass& deferred, const SunShadowTargets& shadow)
{
    for (uint32_t i = 0; i < targets_.size(); ++i)
        deferred.bind_rain_shadow(targets_[i].lighting, shadow.view(i), shadow.sampler());
}

void GBufferTargets::release_lighting(DeferredPass& deferred)
{
    for (auto& target : targets_)
    {
        deferred.release_gbuffer(target.lighting);
        target.lighting = target.multisampled_lighting = VK_NULL_HANDLE;
    }
}
}
