#include "xrEngine/stdafx.h"
#include "DeferredShaderFactory.h"
#include "DeferredShaders.h"
#include "DeferredAlphaTestShaders.h"
#include "WeatherShaders.h"
#include "ShadowShaders.h"
#include "LocalLightShaders.h"
#include "WaterShaders.h"
#include "DistortionShaders.h"
#include "VulkanGameShaders.h"
#include <algorithm>

namespace xray::render::vulkan
{
namespace
{
bool read_shader_file(pcstr path, std::vector<uint8_t>& bytes)
{
    IReader* reader = FS.r_open(path);
    if (!reader)
        return false;
    const auto* begin = static_cast<const uint8_t*>(reader->pointer());
    bytes.clear();
    if (reader->length())
        bytes.assign(begin, begin + reader->length());
    FS.r_close(reader);
    return true;
}

} // namespace

bool read_engine_shader(const std::string& name, std::vector<uint8_t>& bytes, std::string& error)
{
    std::string relative = name;
    std::replace(relative.begin(), relative.end(), '/', '\\');
    string_path path;
    if (!FS.exist(path, "$game_shaders$", relative.c_str()))
    {
        error = "missing game shader resource: " + name;
        return false;
    }
    if (!read_shader_file(path, bytes))
    {
        error = "could not open game shader resource: " + name;
        return false;
    }
    error.clear();
    return true;
}

void configure_engine_shader_resources(VkDevice device, const ShaderModuleDispatch &dispatch,
    GameShaderResources &resources, bool geometry_supported, bool tessellation_supported)
{
    GameShaderResources::Source bundled_source;
#if defined(XR_PLATFORM_ANDROID)
    const FS_Path *shader_path = FS.get_path("$game_shaders$");
    if (shader_path && shader_path->m_Overlay)
    {
        const std::string overlay_root = shader_path->m_Overlay;
        bundled_source = [overlay_root](const std::string &name,
            std::vector<uint8_t> &bytes, std::string &reason) {
            std::string relative = name;
            std::replace(relative.begin(), relative.end(), '/', '\\');
            if (relative.empty() || relative.front() == '\\' || relative.find(':') != std::string::npos ||
                relative.find("..") != std::string::npos || relative.find('\0') != std::string::npos)
            {
                reason = "invalid bundled shader path: " + name;
                return false;
            }
            string_path path;
            strconcat(sizeof(path), path, overlay_root.c_str(), relative.c_str());
            if (!read_shader_file(path, bytes))
            {
                reason = "could not open bundled game shader resource: " + name;
                return false;
            }
            reason.clear();
            return true;
        };
    }
#endif
    resources.configure(device, dispatch, read_engine_shader, geometry_supported,
        tessellation_supported, std::move(bundled_source));
}

bool DeferredShaderFactory::create(VkDevice device, const ShaderModuleDispatch &shader_dispatch, const ScenePassDispatch &pass_dispatch,
                                   VkRenderPass geometry_pass, VkRenderPass light_pass, DeferredPass &pass, GameShaderResources &resources, std::string &error,
                                   VkRenderPass shadow_pass, VkRenderPass local_shadow_pass)
{
    return build(device, shader_dispatch, &pass_dispatch, pass_dispatch.gbuffer_samples, geometry_pass, light_pass,
        pass, resources, error, shadow_pass, local_shadow_pass);
}

bool DeferredShaderFactory::rebuild_scene(VkDevice device, const ShaderModuleDispatch& shader_dispatch,
    VkSampleCountFlagBits samples, VkRenderPass geometry_pass, VkRenderPass light_pass,
    DeferredPass& pass, GameShaderResources& resources, std::string& error, VkRenderPass local_shadow_pass)
{
    return build(device, shader_dispatch, nullptr, samples, geometry_pass, light_pass,
        pass, resources, error, VK_NULL_HANDLE, local_shadow_pass);
}

bool DeferredShaderFactory::build(VkDevice device, const ShaderModuleDispatch& shader_dispatch,
    const ScenePassDispatch* initial_dispatch, VkSampleCountFlagBits samples,
    VkRenderPass geometry_pass, VkRenderPass light_pass, DeferredPass& pass, GameShaderResources& resources,
    std::string& error, VkRenderPass shadow_pass, VkRenderPass local_shadow_pass)
{
    ShaderModule vertex, fragment, alpha_test_fragment, transparent_fragment, light_vertex, light_fragment, weather_fragment;
    ShaderModule multisampled_light_fragment, multisampled_weather_fragment;
    ShaderModule shadow_vertex, shadow_opaque, shadow_cutout, shadow_skin;
    ShaderModule local_light;
    ShaderModule multisampled_local_light;
    ShaderModule water;
    if (!vertex.initialize(device, shader_dispatch, deferred_shaders::GBufferVertex, sizeof(deferred_shaders::GBufferVertex), error) ||
        !fragment.initialize(device, shader_dispatch, deferred_shaders::GBufferFragment, sizeof(deferred_shaders::GBufferFragment), error) ||
        !alpha_test_fragment.initialize(device, shader_dispatch, deferred_shaders::GBufferAlphaTestFragment,
            sizeof(deferred_shaders::GBufferAlphaTestFragment), error) ||
        !transparent_fragment.initialize(device, shader_dispatch, game_shaders::TransparentFragment, sizeof(game_shaders::TransparentFragment), error) ||
        !light_vertex.initialize(device, shader_dispatch, deferred_shaders::LightVertex, sizeof(deferred_shaders::LightVertex), error) ||
        !light_fragment.initialize(device, shader_dispatch, deferred_shaders::LightFragment, sizeof(deferred_shaders::LightFragment), error) ||
        !multisampled_light_fragment.initialize(device, shader_dispatch, deferred_shaders::LightFragmentMsaa,
            sizeof(deferred_shaders::LightFragmentMsaa), error) ||
        !weather_fragment.initialize(device, shader_dispatch, weather_shaders::Fragment, sizeof(weather_shaders::Fragment), error) ||
        !multisampled_weather_fragment.initialize(device, shader_dispatch, weather_shaders::FragmentMsaa,
            sizeof(weather_shaders::FragmentMsaa), error) ||
        (shadow_pass && (!shadow_vertex.initialize(device, shader_dispatch, shadow_shaders::Vertex, sizeof(shadow_shaders::Vertex), error) ||
            !shadow_opaque.initialize(device, shader_dispatch, shadow_shaders::Opaque, sizeof(shadow_shaders::Opaque), error) ||
            !shadow_cutout.initialize(device, shader_dispatch, shadow_shaders::Cutout, sizeof(shadow_shaders::Cutout), error))) ||
        (shadow_pass && !shadow_skin.initialize(device, shader_dispatch, shadow_shaders::SkinVertex,
            sizeof(shadow_shaders::SkinVertex), error)) ||
        (local_shadow_pass && !local_light.initialize(device, shader_dispatch,
            local_light_shaders::Fragment, sizeof(local_light_shaders::Fragment), error)) ||
        (local_shadow_pass && !multisampled_local_light.initialize(device, shader_dispatch,
            local_light_shaders::FragmentMsaa, sizeof(local_light_shaders::FragmentMsaa), error)) ||
        !water.initialize(device, shader_dispatch, water_shaders::Fragment,
            sizeof(water_shaders::Fragment), error))
        return false;
    if (initial_dispatch)
    {
        if (!pass.initialize(device, geometry_pass, light_pass, vertex.handle(), fragment.handle(), alpha_test_fragment.handle(), transparent_fragment.handle(),
                         light_vertex.handle(), light_fragment.handle(), weather_fragment.handle(), *initial_dispatch, error,
                         shadow_pass, shadow_vertex.handle(), shadow_opaque.handle(), shadow_cutout.handle(),
                         local_shadow_pass, local_light.handle(), water.handle(), shadow_skin.handle(),
                         multisampled_light_fragment.handle(), multisampled_weather_fragment.handle(),
                         multisampled_local_light.handle()))
            return false;
    }
    else if (!pass.rebuild_scene_pipelines(geometry_pass, light_pass, samples, vertex.handle(), fragment.handle(),
        alpha_test_fragment.handle(), transparent_fragment.handle(), light_vertex.handle(), light_fragment.handle(),
        weather_fragment.handle(), local_light.handle(), water.handle(), multisampled_light_fragment.handle(),
        multisampled_weather_fragment.handle(), multisampled_local_light.handle(), error))
        return false;
    ShaderModule distortion_vertex, distortion_fragment, distortion_composite;
    if (!distortion_vertex.initialize(device, shader_dispatch, distortion_shaders::Vertex,
            sizeof(distortion_shaders::Vertex), error) ||
        !distortion_fragment.initialize(device, shader_dispatch, distortion_shaders::Fragment,
            sizeof(distortion_shaders::Fragment), error) ||
        !distortion_composite.initialize(device, shader_dispatch, distortion_shaders::Composite,
            sizeof(distortion_shaders::Composite), error) ||
        !pass.initialize_distortion(distortion_vertex.handle(), distortion_fragment.handle(), light_vertex.handle(),
            distortion_composite.handle(), error)) return false;
    return reload_game_pipelines(device, shader_dispatch, pass, resources, error);
}

bool DeferredShaderFactory::reload_game_pipelines(VkDevice device, const ShaderModuleDispatch &shader_dispatch, DeferredPass &pass,
                                                  GameShaderResources &resources, std::string &error)
{
    pass.begin_game_pipeline_reload();

    // Startup and on-demand legacy SVS/SPS pairs use the same device-owned
    // resource cache and VFS resolver as IRender::shader_compile.
    (void)device;
    (void)shader_dispatch;
    struct Pair
    {
        const char *vertex;
        const char *fragment;
        SurfaceMode mode;
    };
    constexpr Pair pairs[]{{"level_opaque", "level_opaque", SurfaceMode::Opaque},          {"object_opaque", "object_opaque", SurfaceMode::Opaque},
                           {"level_lightmap", "level_lightmap", SurfaceMode::Opaque},
                           {"level_lightmap_cutout", "level_lightmap_cutout", SurfaceMode::AlphaTest},
                           {"level_cutout", "level_cutout", SurfaceMode::AlphaTest},       {"object_cutout", "object_cutout", SurfaceMode::AlphaTest},
                           {"object_blended", "object_blended", SurfaceMode::Transparent}, {"object_double_sided", "object_double_sided", SurfaceMode::Opaque},
                           {"tree_opaque", "level_opaque", SurfaceMode::Opaque},           {"progressive_opaque", "level_opaque", SurfaceMode::Opaque}};
    for (const auto& pair : pairs)
    {
        const std::string vertex_name = std::string("vk\\") + pair.vertex + ".vs";
        const std::string fragment_name = std::string("vk\\") + pair.fragment + ".ps";
        if (!resources.pipeline(pass, vertex_name, fragment_name, pair.mode, false, false, error))
        {
            pass.abort_game_pipeline_reload();
            return false;
        }
    }
    for (const char* mode : {"set", "blended", "additive", "alpha_add", "multiply", "multiply_2x"})
        for (bool hud : {false, true})
        {
            const std::string vertex_name = "vk\\level_opaque.vs";
            const std::string fragment_name = std::string("vk\\particle_") +
                (hud ? "hud_" : "world_") + mode + ".ps";
            if (!resources.pipeline(pass, vertex_name, fragment_name,
                    hud ? SurfaceMode::Opaque : SurfaceMode::Transparent,
                    hud, false, error))
            {
                pass.abort_game_pipeline_reload();
                return false;
            }
        }
    struct SkinnedFragment { const char* name; SurfaceMode mode; };
    constexpr SkinnedFragment skinned_fragments[]{
        {"object_opaque", SurfaceMode::Opaque},
        {"object_cutout", SurfaceMode::AlphaTest},
        {"skinned_blended", SurfaceMode::Transparent}
    };
    for (unsigned weights = 1; weights <= 4; ++weights)
    {
        const std::string vertex_name = "vk\\skinned_" + std::to_string(weights) + ".vs";
        for (const auto &variant : skinned_fragments)
        {
            const std::string fragment_name = std::string("vk\\") + variant.name + ".ps";
            if (!resources.pipeline(pass, vertex_name, fragment_name, variant.mode, false, true, error))
            {
                pass.abort_game_pipeline_reload();
                return false;
            }
        }
    }
    for (unsigned weights = 1; weights <= 4; ++weights)
    {
        const std::string vertex_name = "vk\\hud_skinned_" + std::to_string(weights) + ".vs";
        if (!resources.pipeline(pass, vertex_name, "vk\\hud_blended.ps",
                SurfaceMode::Opaque, true, true, error))
        {
            pass.abort_game_pipeline_reload();
            return false;
        }
    }
    pass.commit_game_pipeline_reload();
    error.clear();
    return true;
}
} // namespace xray::render::vulkan
