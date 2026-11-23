#include "xrEngine/stdafx.h"
#include "VulkanLevelRender.h"
#include "VulkanGameDevice.h"
#include "VulkanModelVisual.h"
#include "ParticleVisual.h"
#include <algorithm>
#include <functional>
namespace xray::render::vulkan
{
namespace
{
bool retirement_idle(VkDevice device, PFN_vkDeviceWaitIdle wait)
{
    const VkResult result = wait(device);
    return result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST;
}
} // namespace
IRenderVisual *VulkanLevelRender::model_Duplicate(IRenderVisual *visual)
{
    auto found = models_.find(visual);
    const VulkanModelVisual *source = found == models_.end() ? nullptr : found->second.get();
    if (!source)
        for (const auto &entry : models_)
            if ((source = entry.second->find(visual)))
                break;
    R_ASSERT2(source, "Vulkan duplication requires a live model instance");
    auto copy = std::make_unique<VulkanModelVisual>(*source);
    IRenderVisual *result = copy.get();
    models_.emplace(result, std::move(copy));
    return result;
}

void VulkanLevelRender::model_Delete(IRenderVisual *&visual, bool discard)
{
    if (!visual)
        return;
    if (auto particle = particle_effects_.find(visual); particle != particle_effects_.end())
    {
        if (game_device_)
            game_device_->discard_model_draws(visual);
        if (game_device_)
            game_device_->retire(std::move(particle->second));
        else if (device_ && wait_idle_)
            R_ASSERT2(retirement_idle(device_, wait_idle_), "particle retirement requires a live device");
        particle_effects_.erase(particle);
        visual = nullptr;
        return;
    }
    if (auto particle = particle_groups_.find(visual); particle != particle_groups_.end())
    {
        if (game_device_)
            game_device_->discard_model_draws(visual);
        if (game_device_)
            game_device_->retire(std::move(particle->second));
        else if (device_ && wait_idle_)
            R_ASSERT2(retirement_idle(device_, wait_idle_), "particle retirement requires a live device");
        particle_groups_.erase(particle);
        visual = nullptr;
        return;
    }
    auto found = models_.find(visual);
    R_ASSERT2(found != models_.end(), "Vulkan model deletion received an unknown visual");
    model_visual_index_.clear();
    if (game_device_)
        game_device_->discard_model_draws(visual);
    const bool pooled = !discard && !found->second->cache_name().empty();
    std::vector<IKinematics *> doomed_skeletons;
    std::function<void(const VulkanModelVisual &)> collect_skeletons = [&](const VulkanModelVisual &node) {
        if (node.skeleton())
            doomed_skeletons.push_back(node.skeleton());
        for (const auto &child : node.children())
            collect_skeletons(*child);
    };
    collect_skeletons(*found->second);
    const auto attached_mark = [&](const Wallmark &mark) {
        return std::find(doomed_skeletons.begin(), doomed_skeletons.end(), mark.skeleton) != doomed_skeletons.end();
    };
    // Returning an instance to its pool does not destroy GPU resources.
    // Retain its per-frame pose buffers and descriptors; reuse waits on the
    // normal frame-slot fence before writing them. Attached marks still need
    // retirement because they are actually erased here.
    for (auto it = wallmarks_.begin(); it != wallmarks_.end();)
        if (attached_mark(*it))
        {
            if (game_device_)
                game_device_->retire(std::move(it->model));
            it = wallmarks_.erase(it);
        }
        else
            ++it;
    if (pooled)
        model_pool_.emplace(found->second->cache_name(), std::move(found->second));
    else if (game_device_)
        game_device_->retire(std::move(found->second));
    else if (device_ && wait_idle_)
        R_ASSERT2(retirement_idle(device_, wait_idle_), "model retirement requires idle frames");
    models_.erase(found);
    visual = nullptr;
}

void VulkanLevelRender::models_Clear(bool)
{
    // The engine calls models_Clear(false) while live objects still hold
    // visuals. As in CModelPool::ClearPool, only unused instances are evicted.
    if (model_pool_.empty())
        return;
    if (game_device_)
        for (auto &entry : model_pool_)
            game_device_->retire(std::move(entry.second));
    else if (device_ && wait_idle_)
        R_ASSERT2(retirement_idle(device_, wait_idle_), "Vulkan model clear requires idle GPU frames");
    model_pool_.clear();
    for (auto it = model_gpu_cache_.begin(); it != model_gpu_cache_.end();)
        if (it->second.expired())
            it = model_gpu_cache_.erase(it);
        else
            ++it;
}

void VulkanLevelRender::destroy_all_models()
{
    model_visual_index_.clear();
    if (models_.empty() && model_pool_.empty() && model_bases_.empty() && particle_effects_.empty() &&
        particle_groups_.empty())
    {
        particle_catalog_.reset();
        return;
    }
    if (game_device_)
        game_device_->discard_scene_draws();
    if (game_device_)
        R_ASSERT2(game_device_->wait_idle(), "Vulkan model teardown requires completed frames");
    else if (device_ && wait_idle_)
        R_ASSERT2(retirement_idle(device_, wait_idle_), "Vulkan model teardown requires idle GPU frames");
    models_.clear();
    particle_effects_.clear();
    particle_groups_.clear();
    particle_catalog_.reset();
    model_pool_.clear();
    model_bases_.clear();
    model_gpu_cache_.clear();
}

} // namespace xray::render::vulkan
