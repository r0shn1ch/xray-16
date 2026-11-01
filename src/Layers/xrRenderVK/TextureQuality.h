#pragma once
#include <string>
#include <algorithm>
#include <cctype>
#include "RenderPolicy.h"
namespace xray::render::vulkan
{
inline unsigned world_texture_limit(std::string name, unsigned preset)
{
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    // Interface atlases, fonts, maps, loading screens and baked-light atlases
    // keep their original texel grids. Cubes are preserved by the DDS decoder.
    if (name.find("lmap") != std::string::npos)
        return 0;
    for (const char *prefix : {"ui", "intro", "map", "sky", "lmap", "wpn\\wpn_crosshair"})
        if (name.rfind(prefix, 0) == 0)
            return 0;
    return render_quality(preset).texture_limit;
}
inline unsigned world_texture_limit(std::string name, const std::string &parameters)
{
    unsigned preset = 4;
    const char *names[]{"Minimum", "Low", "Default", "High", "Extreme"};
    for (unsigned i = 0; i < 5; ++i)
        if (parameters.find(std::string("-android-mobile-preset ") + names[i]) != std::string::npos)
            preset = i;
    return world_texture_limit(std::move(name), preset);
}
} // namespace xray::render::vulkan
