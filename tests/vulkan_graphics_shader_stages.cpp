#include "src/Layers/xrRenderVK/GraphicsShaderStages.h"

#include <cassert>
#include <cstdint>
#include <string>
#include <type_traits>

namespace
{
template <class T> T handle(uintptr_t value)
{
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}
}

int main()
{
    using namespace xray::render::vulkan;
    const auto module = [](uintptr_t value) { return handle<VkShaderModule>(value); };
    GraphicsShaderStages stages;
    std::string error;
    assert(build_graphics_shader_stages({module(1), {}, {}, {}, module(5)}, stages, error));
    assert(stages.count == 2 && !stages.tessellation);
    assert(stages.stages[0].stage == VK_SHADER_STAGE_VERTEX_BIT);
    assert(stages.stages[1].stage == VK_SHADER_STAGE_FRAGMENT_BIT);
    assert(!build_graphics_shader_stages({module(1), module(2), {}, {}, module(5)}, stages, error));
    assert(stages.count == 0 && !error.empty());
    assert(build_graphics_shader_stages({module(1), module(2), module(3), module(4), module(5)},
        stages, error));
    assert(stages.count == 5 && stages.tessellation && error.empty());
    for (uint32_t i = 0; i < stages.count; ++i)
    {
        assert(stages.stages[i].module == module(i + 1));
        assert(std::string(stages.stages[i].pName) == "main");
    }
    assert(stages.stages[1].stage == VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT);
    assert(stages.stages[2].stage == VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT);
    assert(stages.stages[3].stage == VK_SHADER_STAGE_GEOMETRY_BIT);
    assert(build_graphics_shader_stages({module(1), {}, {}, module(4), module(5)}, stages, error));
    assert(stages.count == 3 && !stages.tessellation);
    assert(stages.stages[1].stage == VK_SHADER_STAGE_GEOMETRY_BIT);
}
