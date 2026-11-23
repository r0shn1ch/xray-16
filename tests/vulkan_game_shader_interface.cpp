#include "src/Layers/xrRenderVK/GameShaderInterface.h"

#include <cassert>
#include <string>

int main()
{
    using namespace xray::render::vulkan;
    ShaderReflection shader;
    std::string error;
    shader.descriptors = {
        {0, 19, ShaderResourceKind::CombinedImageSampler},
        {1, 0, ShaderResourceKind::UniformBuffer},
        {2, 5, ShaderResourceKind::CombinedImageSampler}};
    assert(validate_game_shader_interface(shader, false, error));
    assert(!validate_game_shader_interface(shader, true, error));
    shader.descriptors[1] = {1, 0, ShaderResourceKind::StorageBuffer};
    shader.descriptors[2] = {2, 0, ShaderResourceKind::UniformBuffer};
    shader.descriptors.push_back({3, 5, ShaderResourceKind::CombinedImageSampler});
    assert(validate_game_shader_interface(shader, true, error));
    shader.descriptors.push_back({4, 0, ShaderResourceKind::CombinedImageSampler});
    assert(!validate_game_shader_interface(shader, true, error));
    assert(error.find("set 4 binding 0") != std::string::npos);
    shader.descriptors.pop_back();
    shader.descriptors[0] = {0, SurfaceMaterialTextureCount, ShaderResourceKind::CombinedImageSampler};
    assert(!validate_game_shader_interface(shader, true, error));
    shader.descriptors[0] = {0, 0, ShaderResourceKind::StorageImage};
    assert(!validate_game_shader_interface(shader, true, error));
    shader.descriptors[0] = {0, 0, ShaderResourceKind::CombinedImageSampler, 2, {301}};
    shader.constants.push_back({301, ~0u, 0, 32, 0, 2, {}});
    assert(validate_game_shader_interface(shader, true, error));
    shader.descriptors[0] = {0, SurfaceMaterialTextureCount - 1,
        ShaderResourceKind::CombinedImageSampler, 2, {302}};
    shader.constants.push_back({302, ~0u, 0, 32, 0, 2, {}});
    assert(!validate_game_shader_interface(shader, true, error));
    shader.descriptors[0] = {0, 0, ShaderResourceKind::CombinedImageSampler};
    shader.constants.clear();
    shader.push_constants = shader.push_constant_extent_known = true;
    shader.push_constant_bytes = 260;
    assert(!validate_game_shader_interface(shader, true, error));
    shader.push_constant_extent_known = false;
    shader.push_constant_bytes = 0;
    assert(!validate_game_shader_interface(shader, true, error));
    assert(error.find("extent could not be reflected") != std::string::npos);

    ShaderReflection specialized_array;
    specialized_array.descriptors.push_back({0, 0, ShaderResourceKind::CombinedImageSampler,
        2, {101}, false, {}});
    specialized_array.constants.push_back({101, 7, 0, 32, 0, 2, {}});
    assert(validate_game_shader_interface(specialized_array, false, {}, error));
    assert(validate_game_shader_interface(specialized_array, false, {{7, 4, 3}}, error));
    assert(!validate_game_shader_interface(specialized_array, false,
        {{7, 4, SurfaceMaterialTextureCount + 1u}}, error));
    assert(error.find("count=") != std::string::npos);

    ShaderReflection specialized_weather_array;
    specialized_weather_array.descriptors.push_back({2, 1,
        ShaderResourceKind::CombinedImageSampler, 2, {201}, false, {}});
    specialized_weather_array.constants.push_back({201, 9, 0, 32, 0, 2, {}});
    assert(!validate_game_shader_interface(specialized_weather_array, false, {}, error));
    assert(validate_game_shader_interface(specialized_weather_array, false, {}, error, true));
}
