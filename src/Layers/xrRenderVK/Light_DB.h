#pragma once
#include "LevelLights.h"
#include "VulkanGameLighting.h"

class IReader;
namespace xray::render::vulkan
{
class CLight_DB
{
public:
    void Load(IReader& level);
    void Unload();
    void Update(const std::vector<VulkanLight*>& dynamic);
    const auto& DynamicLights() const { return dynamic_; }
    const auto& SceneLights() const { return scene_; }
    const auto& HemiLights() const { return hemi_; }
private:
    std::vector<VulkanLightSnapshot> static_, dynamic_, scene_;
    VulkanHemiLightGrid hemi_;
};
}
