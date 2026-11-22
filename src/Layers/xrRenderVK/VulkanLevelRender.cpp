#include "xrEngine/stdafx.h"
#include "VulkanLevelRender.h"
#include "VulkanSettings.h"
#include "ShadowStability.h"
#include "VulkanGameDevice.h"
#include "DeferredShaderFactory.h"
#include "VulkanModelVisual.h"
#include "SpecialVisuals.h"
#include "ParticleVisual.h"
#include "EngineParticleSource.h"
#include "VulkanVisual.h"
#include "VulkanUIShader.h"
#include "VulkanDrawUtils.h"
#include "VulkanDebugRender.h"
#include "VulkanWallMarkArray.h"
#include "WallmarkGeometry.h"
#include "xrEngine/IGame_Level.h"
#include "xrEngine/IGame_Persistent.h"
#include "xrEngine/Rain.h"
#include "xrEngine/xrImage_Resampler.h"
#include "xrEngine/thunderbolt.h"
#include "xrEngine/device.h"
#include "xrEngine/Engine.h"
#include "xrEngine/XR_IOConsole.h"
#include "xrEngine/defines.h"
#include "xrEngine/xr_object.h"
#include "xrEngine/IRenderable.h"
#include "xrEngine/CustomHUD.h"
#include "xrEngine/IGameFont.hpp"
#include "xrEngine/vis_common.h"
#include "xrCDB/ISpatial.h"
#include "xrCDB/Frustum.h"
#include "xrCore/FMesh.hpp"
#include "xrCore/Media/Image.hpp"
#include "xrCore/PostProcess/PPInfo.hpp"
#include "xrMaterialSystem/GameMtlLib.h"

#include <SDL.h>
#include <SDL_vulkan.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <type_traits>
#include <vector>

extern ENGINE_API float psHUD_FOV;

namespace xray::render::vulkan
{
namespace
{
Fmatrix hud_view_projection()
{
    Fmatrix projection, combined;
    const float far_plane = g_pGamePersistent ? g_pGamePersistent->Environment().CurrentEnv.far_plane : 500.f;
    projection.build_projection(deg2rad(psHUD_FOV * Device.fFOV), Device.fASPECT, HUD_VIEWPORT_NEAR, far_plane);
    combined.mul(projection, Device.mView);
    return combined;
}

bool retirement_idle(VkDevice device, PFN_vkDeviceWaitIdle wait)
{
    const VkResult result = wait(device);
    // Vulkan permits destroying a lost device's resources without a successful
    // wait. Keep all other wait failures visible during normal operation.
    return result == VK_SUCCESS || result == VK_ERROR_DEVICE_LOST;
}

float screen_detail_scale(VkExtent2D extent, float fov_degrees)
{
    if (!extent.width || !extent.height || !std::isfinite(fov_degrees) || fov_degrees <= 0.f ||
        !std::isfinite(geometry_lod_scale))
        return 0.f;
    const float fov_scale = 90.f / fov_degrees;
    const float scale = float(extent.width) * float(extent.height) * fov_scale * fov_scale *
        (0.0001f + std::max(geometry_lod_scale, 0.f));
    return std::isfinite(scale) && scale > 0.f ? scale : 0.f;
}

float screen_space_area(float radius, float distance_squared)
{
    if (!std::isfinite(radius) || radius <= 0.f || !std::isfinite(distance_squared) || distance_squared < 0.f)
        return 0.f;
    return radius / std::max(distance_squared, 1e-6f);
}

float screen_threshold(float pixel_radius, float divisor, float detail_scale)
{
    if (!std::isfinite(pixel_radius) || pixel_radius <= 0.f || !std::isfinite(divisor) || divisor <= 0.f ||
        !std::isfinite(detail_scale) || detail_scale <= 0.f)
        return 0.f;
    const float normalized = pixel_radius / divisor;
    return normalized * normalized / detail_scale;
}

std::vector<std::string> engine_shader_defines(const VulkanWindowDevice& window,
    s32 skinning, s32 msaa_sample, uint32_t msaa_samples)
{
    std::vector<std::string> defines;
    const auto add = [&defines](bool enabled, const char* name, const std::string& value = "1") {
        if (enabled)
            defines.emplace_back(std::string(name) + "=" + value);
    };
    const PhysicalDevice& physical = window.physical();
    add(true, "XRAY_VULKAN");
    add(true, "XRAY_VK_VENDOR_ID", std::to_string(physical.properties.vendorID));
    add(true, "XRAY_VK_DEVICE_ID", std::to_string(physical.properties.deviceID));
    add(true, "XRAY_VK_API_VERSION", std::to_string(physical.properties.apiVersion));
    add(true, "XRAY_VK_DRIVER_VERSION", std::to_string(physical.properties.driverVersion));
    add(physical.features.samplerAnisotropy, "XRAY_VK_FEATURE_SAMPLER_ANISOTROPY");
    add(physical.features.multiDrawIndirect, "XRAY_VK_FEATURE_MULTI_DRAW_INDIRECT");
    add(physical.features.geometryShader, "XRAY_VK_FEATURE_GEOMETRY_SHADER");
    add(physical.features.tessellationShader, "XRAY_VK_FEATURE_TESSELLATION_SHADER");
    add(physical.features.shaderStorageImageExtendedFormats,
        "XRAY_VK_FEATURE_STORAGE_IMAGE_EXTENDED_FORMATS");
    add(physical.timeline_semaphore, "XRAY_VK_FEATURE_TIMELINE_SEMAPHORE");
    add(window.ray_tracing_enabled() && physical.ray_tracing.ray_query,
        "XRAY_VK_FEATURE_RAY_QUERY");
    add(window.ray_tracing_enabled() && physical.ray_tracing.ray_tracing_pipeline,
        "XRAY_VK_FEATURE_RAY_TRACING_PIPELINE");
    add(window.bindless_textures_enabled(), "XRAY_VK_FEATURE_BINDLESS_TEXTURES");
#if defined(XR_PLATFORM_ANDROID)
    add(true, "XRAY_ANDROID");
#elif defined(XR_PLATFORM_WINDOWS)
    add(true, "XRAY_WINDOWS");
#endif
    if (skinning < 0)
        add(true, "SKIN_NONE");
    else if (skinning <= 4)
        add(true, (std::string("SKIN_") + std::to_string(skinning)).c_str());
    add(motion_blur_strength > 0.f, "USE_MBLUR");
    add(dof_strength > 0.f, "USE_DOF");
    const std::string quality = std::to_string(quality_preset + 1);
    add(sun_shafts_strength > 0.f, "SUN_SHAFTS_QUALITY", quality);
    add(ao_strength > 0.f, "SSAO_QUALITY", quality);
    add(steep_parallax != 0, "ALLOW_STEEPPARALLAX");
    if (msaa_samples > 1)
    {
        add(true, "USE_MSAA");
        add(true, "MSAA_SAMPLES", std::to_string(msaa_samples));
        add(true, "ISAMPLE", std::to_string(msaa_sample < 0 ? 0 : msaa_sample));
    }
    return defines;
}
} // namespace
static_assert(!std::is_abstract_v<VulkanLevelRender>, "Vulkan IRender implementation must be complete");
VulkanLevelRender::VulkanLevelRender()
{
    ViewBase._clear();
}
VulkanLevelRender::VulkanLevelRender(VulkanGameDevice &device) : VulkanLevelRender()
{
    external_game_device_ = &device;
}
void VulkanLevelRender::create()
{
    m_skinning = -1;
    m_MSAASample = -1;
}
void VulkanLevelRender::destroy()
{
    if (frame_phase_.active())
        frame_phase_.reset();
    destroy_all_models();
    level_Unload();
}

void VulkanLevelRender::DumpStatistics(IGameFont &font, IPerformanceAlert *)
{
    font.OutNext("Vulkan: %u draws, %u triangles", frame_draw_calls_, frame_triangles_);
    font.OutNext("Textures: %zu, %llu bytes", textures_ ? textures_->resident_count() : 0,
                 static_cast<unsigned long long>(textures_ ? textures_->resident_bytes() : 0));
    font.OutNext("Models: %zu live, %zu pooled", models_.size(), model_pool_.size());
}

xrImTextureData VulkanLevelRender::GetImGuiTextureId(pcstr name)
{
    xrImTextureData result{};
    if (!game_device_ || !name || !*name)
        return result;
    VkDescriptorSet descriptor{};
    VkExtent2D extent{};
    auto it = imgui_textures_.find(name);
    if (it == imgui_textures_.end())
    {
        std::string error;
        if (!textures_->ui(name, game_device_->ui_pass(), descriptor, error, &extent))
        {
            Msg("! [renderer-vulkan] ImGui texture %s: %s", name, error.c_str());
            return result;
        }
        imgui_textures_.emplace(name, ImGuiTexture{descriptor, extent});
    }
    else
    {
        descriptor = it->second.descriptor;
        extent = it->second.extent;
    }
    static_assert(sizeof(descriptor) <= sizeof(result.texture));
    std::memcpy(&result.texture, &descriptor, sizeof(descriptor));
    result.size.set(float(extent.width), float(extent.height));
    return result;
}

void VulkanLevelRender::models_Prefetch()
{
    // Uploads are queued at material creation; force completion at the engine's prefetch barrier.
    ResourcesDeferredUpload();
}
bool VulkanLevelRender::occ_visible(vis_data &visual)
{
    return occ_visible(visual.box);
}
bool VulkanLevelRender::occ_visible(Fbox &bounds)
{
    u32 mask = ViewBase.getMask();
    return ViewBase.testAABB(bounds.data(), mask) != fcvNone;
}
bool VulkanLevelRender::occ_visible(sPoly &polygon)
{
    return polygon.size() >= 3 && ViewBase.testPolyInside(polygon);
}
void VulkanLevelRender::BeforeWorldRender()
{
    if (gpu_particle_emitters_ && game_device_ && !game_device_->gpu_particle_emitters())
        set_gpu_particle_emitters(nullptr);
}
void VulkanLevelRender::AfterWorldRender()
{
    if (!game_device_ || !g_pGameLevel || !g_pGameLevel->pHUD ||
        !g_pGameLevel->pHUD->RenderActiveItemUIQuery())
        return;

    // The foreground records this world-space batch after HUD models, against
    // their depth. Keep the detector screen on the same HUD projection.
    const Fmatrix projection = hud_view_projection();
    auto& ui = game_device_->ui();
    ui.set_world_view_projection(&projection);
    g_pGameLevel->pHUD->RenderActiveItemUI();
    ui.set_world_view_projection(nullptr);
}
void VulkanLevelRender::ObtainRequiredWindowFlags(u32 &flags)
{
    flags |= SDL_WINDOW_VULKAN;
}
void VulkanLevelRender::overdrawBegin()
{
}
void VulkanLevelRender::overdrawEnd()
{
}
void VulkanLevelRender::DeferredLoad(bool enabled)
{
    deferred_load_ = enabled;
}
void VulkanLevelRender::ResourcesDeferredUpload()
{
    if (textures_)
        R_ASSERT2(textures_->finish_uploads(), "Vulkan deferred texture upload failed");
}
void VulkanLevelRender::ResourcesDeferredUnload()
{
    if (textures_)
        textures_->retire_unused();
    models_Clear(false);
}
void VulkanLevelRender::ResourcesGetMemoryUsage(u32 &m_base, u32 &c_base, u32 &m_lmaps, u32 &c_lmaps)
{
    const uint64_t bytes = textures_ ? textures_->resident_bytes() : 0;
    m_base = static_cast<u32>(std::min<uint64_t>(bytes, UINT32_MAX));
    c_base = textures_ ? static_cast<u32>(textures_->resident_count()) : 0;
    m_lmaps = c_lmaps = 0;
}
void VulkanLevelRender::ResourcesDestroyNecessaryTextures()
{
    ResourcesDeferredUnload();
}
void VulkanLevelRender::ResourcesStoreNecessaryTextures()
{
    ResourcesDeferredUpload();
}
void VulkanLevelRender::ResourcesDumpMemoryUsage()
{
    Msg("* [renderer-vulkan] textures: %zu, approximate bytes: %llu", textures_ ? textures_->resident_count() : 0,
        static_cast<unsigned long long>(textures_ ? textures_->resident_bytes() : 0));
}
void VulkanLevelRender::OnAssetsChanged()
{
    if (frame_phase_.active())
    {
        assets_dirty_ = true;
        return;
    }
    assets_dirty_ = false;
    if (device_ && wait_idle_)
        R_ASSERT2(wait_idle_(device_) == VK_SUCCESS, "Vulkan shader reload needs idle GPU frames");
    if (game_device_)
    {
        refresh_engine_shader_defines();
        std::string shader_error;
        if (!game_device_->reload_game_shaders(shader_error))
            Msg("! [renderer-vulkan] keeping previous game pipelines after reload failure: %s", shader_error.c_str());
    }
    const auto release_ray_views = [&] {
        if (game_device_) game_device_->release_trail_texture();
        if (game_device_) game_device_->release_level_glows();
        for (VkImageView view : ray_material_views_)
            textures_->release_environment(view);
        ray_material_views_.clear();
        for (VkImageView view : ray_diffuse_material_views_)
            textures_->release_environment(view);
        ray_diffuse_material_views_.clear();
    };
    const auto rebind_ray_views = [&] {
        if (!ray_materials_.set()) return;
        std::string error;
        const auto& names = ray_material_names_;
        for (size_t slot = 0; slot < names.size(); ++slot)
        {
            VkImageView view = VK_NULL_HANDLE;
            if (!textures_->projection_texture(names[slot], view, error))
                break;
            ray_material_views_.push_back(view);
            if (!ray_materials_.write(static_cast<uint32_t>(slot), view, textures_->sampler(), error))
                break;
        }
        if (ray_material_views_.size() != names.size() || !error.empty())
        {
            Msg("! [renderer-vulkan] ray materials disabled after texture reload: %s", error.c_str());
            release_ray_materials();
            return;
        }
        if (!ray_diffuse_materials_.set()) return;
        const auto& diffuse_names = ray_diffuse_material_names_;
        for (size_t slot = 0; slot < diffuse_names.size(); ++slot)
        {
            VkImageView view = VK_NULL_HANDLE;
            if (!textures_->projection_texture(diffuse_names[slot], view, error,
                    ray_diffuse_material_linear_[slot] != 0)) break;
            ray_diffuse_material_views_.push_back(view);
            if (!ray_diffuse_materials_.write(static_cast<uint32_t>(slot), view, textures_->sampler(), error)) break;
        }
        if (ray_diffuse_material_views_.size() != diffuse_names.size() || !error.empty())
        {
            Msg("! [renderer-vulkan] ray diffuse materials disabled after texture reload: %s", error.c_str());
            release_ray_diffuse_materials();
        }
    };
    if (g_pGamePersistent && game_device_)
    {
        auto &environment = g_pGamePersistent->Environment();
        const auto visit = [](CEnvironment::EnvsMap &presets, bool create) {
            for (auto &[name, descriptors] : presets)
                for (CEnvDescriptor *descriptor : descriptors)
                {
                    if (!descriptor || !descriptor->m_pDescriptor)
                        continue;
                    if (create)
                        descriptor->on_device_create();
                    else
                        descriptor->on_device_destroy();
                }
        };
        if (textures_)
        {
            std::string error;
            const auto before = [&] {
                release_ray_views();
                if (game_device_)
                    game_device_->detach_color_maps_for_asset_reload();
                // Old weather descriptors must be retired before old image views.
                environment.m_pRender->Clear();
                visit(environment.WeatherCycles, false);
                visit(environment.WeatherFXs, false);
            };
            const auto after = [&] {
                rebind_ray_views();
                std::string color_map_error;
                if (!game_device_->refresh_color_maps_after_asset_reload(color_map_error))
                    Msg("! [renderer-vulkan] color maps unavailable after asset reload: %s", color_map_error.c_str());
                visit(environment.WeatherCycles, true);
                visit(environment.WeatherFXs, true);
            };
            if (!textures_->reload_assets(error, before, after))
                Msg("! [renderer-vulkan] keeping previous textures after reload failure: %s", error.c_str());
        }
    }
    else if (textures_)
    {
        std::string error;
        const auto before = [&] {
            release_ray_views();
            if (game_device_)
                game_device_->detach_color_maps_for_asset_reload();
        };
        const auto after = [&] {
            rebind_ray_views();
            if (game_device_)
            {
                std::string color_map_error;
                if (!game_device_->refresh_color_maps_after_asset_reload(color_map_error))
                    Msg("! [renderer-vulkan] color maps unavailable after asset reload: %s", color_map_error.c_str());
            }
        };
        if (!textures_->reload_assets(error, before, after))
            Msg("! [renderer-vulkan] keeping previous textures after reload failure: %s", error.c_str());
    }
    models_Clear(false);
    particle_catalog_.reset();
    flush_resource_failures();
}
void VulkanLevelRender::Screenshot(ScreenshotMode mode, pcstr name)
{
    if (!game_device_)
        return;
    if (mode != SM_NORMAL && (!name || !*name))
    {
        Msg("! [renderer-vulkan] screenshot requires a path or postfix");
        return;
    }
    screenshot_ = std::make_unique<ScreenshotRequest>(ScreenshotRequest{mode, name ? name : ""});
    game_device_->request_screenshot();
}

void VulkanLevelRender::save_screenshot()
{
    if (!screenshot_)
        return;
    std::vector<u8> pixels;
    VkExtent2D extent{};
    VkFormat format{};
    if (!game_device_->take_screenshot(pixels, extent, format))
    {
        Msg("! [renderer-vulkan] screenshot readback unavailable on this surface");
        screenshot_.reset();
        return;
    }
    if (format == VK_FORMAT_B8G8R8A8_UNORM)
        for (size_t i = 0; i < pixels.size(); i += 4)
            std::swap(pixels[i], pixels[i + 2]);
    const auto request = std::move(screenshot_);
    string_path filename{};
    IWriter *writer = nullptr;
    if (request->mode == SM_NORMAL)
    {
        string64 time;
        xr_sprintf(filename, "ss_%s_%s_(%s).jpg", Core.UserName, timestamp(time),
                   g_pGameLevel ? g_pGameLevel->name().c_str() : "mainmenu");
        writer = FS.w_open("$screenshots$", filename);
    }
    else if (request->mode == SM_FOR_GAMESAVE)
        writer = FS.w_open(request->name.c_str());
    else
    {
        strconcat(sizeof(filename), filename, request->name.c_str(), ".tga");
        writer = FS.w_open("$screenshots$", filename);
    }
    if (!writer)
    {
        Msg("! [renderer-vulkan] screenshot output could not be opened");
        return;
    }

    if (request->mode == SM_NORMAL)
    {
        std::vector<u8> rgb(size_t(extent.width) * extent.height * 3);
        for (size_t i = 0, j = 0; i < pixels.size(); i += 4, j += 3)
            std::copy_n(pixels.data() + i, 3, rgb.data() + j);
        XRay::Media::Image img(extent.width, extent.height, rgb.data(), XRay::Media::ImageDataFormat::RGB8);
        if (!img.SaveJPEG(*writer, 100))
            Msg("! [renderer-vulkan] JPEG screenshot encoding failed");
    }
    else if (request->mode == SM_FOR_GAMESAVE)
    {
        // 128x128 BC1 DDS, matching the game save preview's on-disk format.
        constexpr u32 side = 128;
        u32 header[32]{};
        header[0] = 0x20534444;
        header[1] = 124;
        header[2] = 0x81007;
        header[3] = side;
        header[4] = side;
        header[5] = side * side / 2;
        header[19] = 32;
        header[20] = 4;
        header[21] = 0x31545844;
        header[27] = 0x1000;
        writer->w(header, sizeof(header));
        std::array<u8, side * side * 3> scaled{};
        for (u32 y = 0; y < side; ++y)
            for (u32 x = 0; x < side; ++x)
            {
                const size_t src =
                    (size_t(y) * extent.height / side * extent.width + size_t(x) * extent.width / side) * 4;
                const size_t dst = (size_t(y) * side + x) * 3;
                std::copy_n(pixels.data() + src, 3, scaled.data() + dst);
            }
        for (u32 by = 0; by < side; by += 4)
            for (u32 bx = 0; bx < side; bx += 4)
            {
                u8 low[3]{255, 255, 255}, high[3]{};
                for (u32 y = 0; y < 4; ++y)
                    for (u32 x = 0; x < 4; ++x)
                        for (u32 c = 0; c < 3; ++c)
                        {
                            u8 v = scaled[((by + y) * side + bx + x) * 3 + c];
                            low[c] = std::min(low[c], v);
                            high[c] = std::max(high[c], v);
                        }
                const auto pack = [](const u8 *v) -> u16 {
                    return u16((v[0] >> 3) << 11 | (v[1] >> 2) << 5 | (v[2] >> 3));
                };
                u16 endpoints[2]{pack(high), pack(low)};
                if (endpoints[0] <= endpoints[1])
                    endpoints[0] = endpoints[1] < 65535 ? endpoints[1] + 1 : 65535;
                u8 palette[4][3]{};
                for (u32 c = 0; c < 3; ++c)
                {
                    palette[0][c] = high[c];
                    palette[1][c] = low[c];
                    palette[2][c] = (2 * high[c] + low[c]) / 3;
                    palette[3][c] = (high[c] + 2 * low[c]) / 3;
                }
                u32 selectors = 0;
                for (u32 y = 0; y < 4; ++y)
                    for (u32 x = 0; x < 4; ++x)
                    {
                        const u8 *p = &scaled[((by + y) * side + bx + x) * 3];
                        u32 best = 0, distance = UINT32_MAX;
                        for (u32 i = 0; i < 4; ++i)
                        {
                            u32 d = 0;
                            for (u32 c = 0; c < 3; ++c)
                            {
                                int delta = int(p[c]) - palette[i][c];
                                d += delta * delta;
                            }
                            if (d < distance)
                            {
                                distance = d;
                                best = i;
                            }
                        }
                        selectors |= best << (2 * (y * 4 + x));
                    }
                writer->w(endpoints, sizeof(endpoints));
                writer->w(&selectors, sizeof(selectors));
            }
    }
    else if (request->mode == SM_FOR_LEVELMAP || request->mode == SM_FOR_CUBEMAP)
    {
        // Match the other renderers' square TGA capture used by level-map
        // fragments and cubemap faces.
        const u32 side = extent.height;
        std::vector<u32> square(size_t(side) * side);
        imf_Process(square.data(), side, side, reinterpret_cast<u32*>(pixels.data()),
            extent.width, extent.height, imf_lanczos3);
        XRay::Media::Image img(side, side, square.data(), XRay::Media::ImageDataFormat::RGBA8);
        img.SaveTGA(*writer, true);
    }
    else
    {
        const u32 side = extent.height;
        std::vector<u8> square(size_t(side) * side * 4);
        for (u32 y = 0; y < side; ++y)
            for (u32 x = 0; x < side; ++x)
            {
                const u32 source_x = u32(uint64_t(x) * extent.width / side);
                std::copy_n(pixels.data() + (size_t(y) * extent.width + source_x) * 4, 4,
                            square.data() + (size_t(y) * side + x) * 4);
            }
        XRay::Media::Image img(side, side, square.data(), XRay::Media::ImageDataFormat::RGBA8);
        img.SaveTGA(*writer, true);
    }
    FS.w_close(writer);
}
void VulkanLevelRender::SetPostProcessParams(const SPPInfo &ppi)
{
    gray_ = std::clamp(ppi.gray, 0.f, 1.f);
    postprocess_.effect[0] = std::max(ppi.blur, 0.f);
    postprocess_.effect[1] = ppi.duality.h;
    postprocess_.effect[2] = ppi.duality.v;
    postprocess_.effect[3] = gray_;
    postprocess_.noise[0] = std::clamp(ppi.noise.intensity, 0.f, 1.f);
    postprocess_.noise[1] = std::max(ppi.noise.grain, 1.f);
    postprocess_.noise[2] = std::max(ppi.noise.fps, .01f);
    postprocess_.tint[0] = ppi.color_base.r * 2.f;
    postprocess_.tint[1] = ppi.color_base.g * 2.f;
    postprocess_.tint[2] = ppi.color_base.b * 2.f;
    postprocess_.add[0] = ppi.color_add.r * 2.f;
    postprocess_.add[1] = ppi.color_add.g * 2.f;
    postprocess_.add[2] = ppi.color_add.b * 2.f;
    postprocess_.gray_weights[0] = ppi.color_gray.r;
    postprocess_.gray_weights[1] = ppi.color_gray.g;
    postprocess_.gray_weights[2] = ppi.color_gray.b;
    postprocess_.color_map[0] = std::clamp(ppi.cm_influence, 0.f, 1.f);
    postprocess_.color_map[1] = std::clamp(ppi.cm_interpolate, 0.f, 1.f);
    color_map_a_ = postprocess_.color_map[0] > 0.f && ppi.cm_tex1.size() ? ppi.cm_tex1.c_str() : "";
    color_map_b_ = !color_map_a_.empty() && ppi.cm_tex2.size() ? ppi.cm_tex2.c_str() : color_map_a_;
    if (color_map_a_.empty())
        postprocess_.color_map[0] = 0.f;
}
void VulkanLevelRender::setGamma(float value)
{
    gamma_ = std::clamp(value, .01f, 4.f);
}
void VulkanLevelRender::setBrightness(float value)
{
    brightness_ = std::clamp(value, 0.f, 4.f);
}
void VulkanLevelRender::setContrast(float value)
{
    contrast_ = std::clamp(value, 0.f, 4.f);
}
void VulkanLevelRender::updateGamma()
{
    // Console and post-precache calls apply the requested grade immediately;
    // End copies these constants into the submitted frame as well.
    postprocess_.grade[0] = gamma_;
    postprocess_.grade[1] = brightness_;
    postprocess_.grade[2] = contrast_;
}
namespace
{
bool wallmark_geometry(const Fvector (&triangle)[3], const Fvector &point, float size, const std::string &texture,
                       ModelGeometry &result)
{
    if (!(size > EPS_L) || texture.empty() || !_valid(point))
        return false;
    Fvector normal;
    normal.mknormal(triangle[0], triangle[1], triangle[2]);
    if (!_valid(normal) || normal.square_magnitude() < EPS_L)
        return false;
    Fvector axis;
    axis.set(std::abs(normal.y) > .99f ? 1.f : 0.f, std::abs(normal.y) > .99f ? 0.f : 1.f, 0.f);
    Fvector right, up;
    right.crossproduct(axis, normal).normalize_safe();
    up.crossproduct(normal, right).normalize_safe();
    std::array<WallmarkVertex, 3> input;
    for (size_t i = 0; i < 3; ++i)
    {
        Fvector delta;
        delta.sub(triangle[i], point);
        input[i].position = {triangle[i].x, triangle[i].y, triangle[i].z};
        input[i].u = .5f + delta.dotproduct(right) / size;
        input[i].v = .5f - delta.dotproduct(up) / size;
    }
    const auto clipped = clip_wallmark_triangle(input);
    if (clipped.empty())
        return false;
    result = {};
    result.texture = texture;
    // Runtime wallmarks do not come from an OGF and therefore have no OGF
    // shader name. Resolve their actual blender from shaders.xr explicitly.
    result.shader = "effects\\wallmark";
    result.mode = SurfaceMode::Transparent;
    for (const auto &vertex : clipped)
    {
        ModelVertex model;
        Fvector offset = normal;
        Fvector to_camera;
        to_camera.sub(Device.vCameraPosition, point);
        if (offset.dotproduct(to_camera) < 0.f)
            offset.invert();
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const float component = axis == 0 ? offset.x : axis == 1 ? offset.y : offset.z;
            model.position[axis] = vertex.position[axis] + component * .003f;
        }
        model.normal[0] = normal.x;
        model.normal[1] = normal.y;
        model.normal[2] = normal.z;
        model.uv[0] = vertex.u;
        model.uv[1] = vertex.v;
        result.indices.push_back(static_cast<uint32_t>(result.vertices.size()));
        result.vertices.push_back(model);
    }
    return true;
}

std::string model_cache_name(pcstr name)
{
    std::string result = name ? name : "";
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c == '\\' ? '/' : c)); });
    if (result.size() >= 4 && result.compare(result.size() - 4, 4, ".ogf") == 0)
        result.resize(result.size() - 4);
    return result;
}

float distance_squared(const Fvector &left, const Fvector &right)
{
    const float x = left.x - right.x;
    const float y = left.y - right.y;
    const float z = left.z - right.z;
    const float result = x * x + y * y + z * z;
    return std::isfinite(result) ? result : 0.f;
}

Fvector visual_center(const LevelVisual *visual)
{
    Fvector center;
    if (visual)
        center.set(visual->bounds[6], visual->bounds[7], visual->bounds[8]);
    else
        center.set(0.f, 0.f, 0.f);
    return center;
}

DeferredEnvironment current_environment()
{
    DeferredEnvironment environment;
    if (g_pGamePersistent)
    {
        const CEnvDescriptorMixer &current = g_pGamePersistent->Environment().CurrentEnv;
        environment.sun_direction[0] = current.sun_dir.x;
        environment.sun_direction[1] = current.sun_dir.y;
        environment.sun_direction[2] = current.sun_dir.z;
        environment.sun_color[0] = current.sun_color.x;
        environment.sun_color[1] = current.sun_color.y;
        environment.sun_color[2] = current.sun_color.z;
        environment.ambient_color[0] = current.ambient.x;
        environment.ambient_color[1] = current.ambient.y;
        environment.ambient_color[2] = current.ambient.z;
        environment.hemi_color[0] = current.hemi_color.x;
        environment.hemi_color[1] = current.hemi_color.y;
        environment.hemi_color[2] = current.hemi_color.z;
        environment.fog_color[0] = current.fog_color.x;
        environment.fog_color[1] = current.fog_color.y;
        environment.fog_color[2] = current.fog_color.z;
        environment.fog_near = current.fog_near;
        environment.fog_far = current.fog_far;
    }
    return environment;
}

} // namespace

VulkanLevelRender::~VulkanLevelRender()
{
    Destroy();
}

void VulkanLevelRender::Create(SDL_Window *window, u32 &width, u32 &height, float &half_width, float &half_height)
{
    if (!window)
    {
        xrDebug::Fatal(DEBUG_INFO, "Vulkan renderer requires the engine SDL window");
        return;
    }
    const u32 window_flags = SDL_GetWindowFlags(window);
    if (!(window_flags & SDL_WINDOW_VULKAN) || (window_flags & SDL_WINDOW_OPENGL))
    {
        xrDebug::Fatal(DEBUG_INFO, "Vulkan renderer requires an SDL Vulkan window without OpenGL");
        return;
    }

    Destroy();
    Msg("[renderer-vulkan] device.create begin requested=%ux%u flags=0x%x", width, height, window_flags);
    auto device = external_game_device_ ? nullptr : std::make_unique<VulkanGameDevice>();
    auto *resources = external_game_device_ ? external_game_device_ : device.get();
    std::string error;
    if (!resources->initialize(window, {width, height}, error))
    {
        xrDebug::Fatal(DEBUG_INFO, "Vulkan device creation failed: %s", error.c_str());
        return;
    }
    const auto &gpu = resources->window().physical().properties;
    Msg("[renderer-vulkan] gpu.ready name='%s' vendor=0x%04x device=0x%04x driver=0x%x api=%u.%u", gpu.deviceName,
        gpu.vendorID, gpu.deviceID, gpu.driverVersion, VK_VERSION_MAJOR(gpu.apiVersion),
        VK_VERSION_MINOR(gpu.apiVersion));

    // Engine UI coordinates and font selection follow the display. Scene
    // resources use FrameContext::extent() independently of this logical size.
    const VkExtent2D extent = resources->window().frame().presentation_extent();
    if (!extent.width || !extent.height)
    {
        xrDebug::Fatal(DEBUG_INFO, "Vulkan swapchain has an empty extent");
        return;
    }
    width = extent.width;
    height = extent.height;
    half_width = static_cast<float>(width) * 0.5f;
    half_height = static_cast<float>(height) * 0.5f;

    owned_game_device_ = std::move(device);
    window_ = window;
    int drawable_width = 0, drawable_height = 0;
    SDL_Vulkan_GetDrawableSize(window, &drawable_width, &drawable_height);
    surface_state_.reset();
    surface_state_.finish_reset(drawable_width > 0 ? drawable_width : extent.width,
                                drawable_height > 0 ? drawable_height : extent.height);
    bind_level_device(*resources);
    Msg("[renderer-vulkan] gameplay device initialized: logical=%ux%u swapchain=%ux%u", width, height, extent.width,
        extent.height);
    context_state_.device_created();
    R_ASSERT2(device_resource_state_.device_created(),
              "Vulkan renderer device resource state was not empty at creation");
    reset_in_progress_ = false;
}

bool VulkanLevelRender::start_benchmark(uint32_t frames)
{
    if (!game_device_ || !benchmark_.arm(frames, FrameContext::FramesInFlight))
        return false;
    const auto& window = game_device_->window();
    const auto scene = window.frame().extent();
    const auto presentation = window.frame().presentation_extent();
    const auto& gpu = window.physical().properties;
    std::ostringstream context;
    context << "{\"source_sha\": " << std::quoted(xrCore::GetBuildCommit())
            << ", \"frame_clock\": \"steady wall time\", \"cpu_scope\": \"renderer recording/submission/present\""
            << ", \"gpu\": " << std::quoted(gpu.deviceName) << ", \"driver_version\": " << gpu.driverVersion
            << ", \"scene_extent\": [" << scene.width << ", " << scene.height
            << "], \"presentation_extent\": [" << presentation.width << ", " << presentation.height
            << "], \"quality_preset\": " << quality_preset << ", \"texture_lod\": " << psTextureLOD
            << ", \"anisotropy\": " << texture_anisotropy << ", \"antialiasing_mode\": " << antialiasing
            << ", \"msaa_requested\": " << gbuffer_msaa_samples
            << ", \"scene_samples\": " << static_cast<uint32_t>(window.frame().scene_samples())
            << ", \"upscaler\": " << spatial_upscaler
            << ", \"render_scale_percent\": " << render_scale_percent
            << ", \"ao_strength\": " << ao_strength << ", \"ao_directions_override\": " << ao_directions_override
            << ", \"sun_shafts_strength\": " << sun_shafts_strength
            << ", \"sun_shaft_samples_override\": " << sun_shaft_samples_override
            << ", \"volumetric_lights\": " << volumetric_lights_enabled
            << ", \"motion_blur_strength\": " << motion_blur_strength << ", \"dof_strength\": " << dof_strength
            << ", \"detail_radius\": " << detail_radius << ", \"detail_spacing\": " << detail_spacing
            << ", \"point_shadows\": " << point_shadow_budget << ", \"spot_shadows\": " << spot_shadow_budget << '}';
    benchmark_context_ = context.str();
    Device.seqAppDeactivate.Add(this);
    Msg("[renderer-vulkan] benchmark.armed frames=%u; close console and return to unpaused gameplay", frames);
    return true;
}

void VulkanLevelRender::request_sampler_settings_reset()
{
    if (textures_ && game_device_)
        game_device_->request_reset();
}

void VulkanLevelRender::interrupt_benchmark(const char* reason)
{
    if (benchmark_.interrupt(reason))
        save_benchmark();
}

void VulkanLevelRender::save_benchmark()
{
    Device.seqAppDeactivate.Remove(this);
    char* directory = SDL_GetPrefPath("OpenXRay", "vulkan-metrics");
    bool saved = false;
    if (directory)
    {
        const std::string path = std::string(directory) + "frame-benchmark.json";
        std::ofstream output(path, std::ios::trunc);
        output << "{\n  \"context\": " << benchmark_context_ << ",\n  \"capture\": ";
        saved = benchmark_.write_json(output);
        output << "}\n";
        output.flush();
        saved = saved && output.good();
        Msg("%s [renderer-vulkan] benchmark.report path=%s", saved ? "*" : "!", path.c_str());
        SDL_free(directory);
    }
    const auto summary = benchmark_.metrics().summary();
    Msg("[renderer-vulkan] benchmark.finished complete=%d saved=%d frames=%llu requested=%u "
        "frame-p50=%.2fms frame-p95=%.2fms frame-p99=%.2fms",
        benchmark_.complete() ? 1 : 0, saved ? 1 : 0, static_cast<unsigned long long>(summary.frames),
        benchmark_.requested_frames(), summary.frame_p50_ms, summary.frame_p95_ms, summary.frame_p99_ms);
}

void VulkanLevelRender::Destroy()
{
    interrupt_benchmark("device destroyed");
    if (game_device_)
        Msg("[renderer-vulkan] device.destroy begin level-models=%zu frame-count=%llu", models_.size(),
            static_cast<unsigned long long>(metrics_.summary().frames));
    if (metrics_.summary().frames)
    {
        char *directory = SDL_GetPrefPath("OpenXRay", "vulkan-metrics");
        if (directory)
        {
            const std::string path = std::string(directory) + "frame-metrics.json";
            if (!metrics_.write_json(path))
                Msg("! [renderer-vulkan] could not save frame metrics: %s", path.c_str());
            else
                Msg("* [renderer-vulkan] frame metrics: %s", path.c_str());
            SDL_free(directory);
        }
    }
    metrics_ = {};
    last_metric_frame_ = {};
    device_loss_reported_ = false;
    // Shutdown can interrupt a calculated frame that will never be submitted.
    frame_phase_.reset();
    // Game objects own the reference-counted light resources. Disable them
    // while the renderer is shutting down, but never delete through this
    // non-owning registry.
    for (VulkanLight *light : lights_)
        if (light)
            light->set_active(false);
    for (VulkanGlow *glow : glows_)
        if (glow)
            glow->set_active(false);
    if (device_resource_state_.can_create_factory_objects())
        OnDeviceDestroy(false);
    else
    {
        destroy_all_models();
        level_Unload();
    }
    if (game_device_)
        game_device_->use_scene_visibility(false);
    set_gpu_particle_emitters(nullptr);
    game_device_ = nullptr;
    device_ = VK_NULL_HANDLE;
    queue_ = VK_NULL_HANDLE;
    pool_ = VK_NULL_HANDLE;
    memory_ = {};
    upload_ = {};
    textures_ = nullptr;
    pass_ = nullptr;
    wait_idle_ = nullptr;
    window_ = nullptr;
    surface_state_.reset();
    reset_in_progress_ = false;
    frame_phase_.reset();
    camera_state_.reset();
    ViewBase._clear();
    context_state_.device_destroyed();
    R_ASSERT2(device_resource_state_.device_destroyed(),
              "Vulkan renderer device resources must be destroyed before the device");
    clear_target_pending_ = false;
    frame_clear_target_ = false;
    if (owned_game_device_)
    {
        owned_game_device_->destroy();
        owned_game_device_.reset();
    }
    else if (external_game_device_)
        external_game_device_->destroy();
}

void VulkanLevelRender::Reset(SDL_Window *window, u32 &width, u32 &height, float &half_width, float &half_height)
{
    R_ASSERT2(window && window == window_ && (SDL_GetWindowFlags(window) & SDL_WINDOW_VULKAN),
              "Vulkan renderer reset requires its original SDL Vulkan window");
    if (!game_device_)
    {
        xrDebug::Fatal(DEBUG_INFO, "Vulkan renderer reset requires an initialized device");
        return;
    }

    if (!reset_in_progress_)
        reset_begin();

    int drawable_width = 0, drawable_height = 0;
    SDL_Vulkan_GetDrawableSize(window, &drawable_width, &drawable_height);
    Msg("[renderer-vulkan] swapchain.reset begin drawable=%dx%d replace-surface=%d", drawable_width, drawable_height,
        surface_state_.replace_surface() ? 1 : 0);
    if (drawable_width <= 0 || drawable_height <= 0)
    {
        surface_state_.request_reset();
        reset_end();
        return;
    }

    std::string error;
    if (textures_)
    {
        const auto release_ray_views = [&] {
            if (game_device_)
            {
                game_device_->release_trail_texture();
                game_device_->release_level_glows();
            }
            for (VkImageView view : ray_material_views_)
                textures_->release_environment(view);
            ray_material_views_.clear();
            for (VkImageView view : ray_diffuse_material_views_)
                textures_->release_environment(view);
            ray_diffuse_material_views_.clear();
        };
        const auto rebind_ray_views = [&] {
            std::string bind_error;
            if (ray_materials_.set())
            {
                for (size_t slot = 0; slot < ray_material_names_.size(); ++slot)
                {
                    VkImageView view = VK_NULL_HANDLE;
                    if (!textures_->projection_texture(ray_material_names_[slot], view, bind_error))
                        break;
                    ray_material_views_.push_back(view);
                    if (!ray_materials_.write(static_cast<uint32_t>(slot), view, textures_->sampler(), bind_error))
                        break;
                }
                if (ray_material_views_.size() != ray_material_names_.size() || !bind_error.empty())
                {
                    Msg("! [renderer-vulkan] ray materials disabled after sampler update: %s", bind_error.c_str());
                    release_ray_materials();
                    return;
                }
            }
            if (ray_diffuse_materials_.set())
            {
                for (size_t slot = 0; slot < ray_diffuse_material_names_.size(); ++slot)
                {
                    VkImageView view = VK_NULL_HANDLE;
                    if (!textures_->projection_texture(ray_diffuse_material_names_[slot], view, bind_error,
                            ray_diffuse_material_linear_[slot] != 0))
                        break;
                    ray_diffuse_material_views_.push_back(view);
                    if (!ray_diffuse_materials_.write(static_cast<uint32_t>(slot), view, textures_->sampler(),
                            bind_error))
                        break;
                }
                if (ray_diffuse_material_views_.size() != ray_diffuse_material_names_.size() || !bind_error.empty())
                {
                    Msg("! [renderer-vulkan] ray diffuse materials disabled after sampler update: %s",
                        bind_error.c_str());
                    release_ray_diffuse_materials();
                }
            }
        };
        const auto retire_weather_descriptors = [&] {
            if (g_pGamePersistent && game_device_)
            {
                auto &environment = g_pGamePersistent->Environment();
                environment.m_pRender->Clear();
                const auto visit = [](CEnvironment::EnvsMap &presets) {
                    for (auto &[name, descriptors] : presets)
                        for (CEnvDescriptor *descriptor : descriptors)
                            if (descriptor && descriptor->m_pDescriptor)
                                descriptor->on_device_destroy();
                };
                visit(environment.WeatherCycles);
                visit(environment.WeatherFXs);
            }
        };
        const auto restore_weather_descriptors = [&] {
            if (g_pGamePersistent && game_device_)
            {
                auto &environment = g_pGamePersistent->Environment();
                const auto visit = [](CEnvironment::EnvsMap &presets) {
                    for (auto &[name, descriptors] : presets)
                        for (CEnvDescriptor *descriptor : descriptors)
                            if (descriptor && descriptor->m_pDescriptor)
                                descriptor->on_device_create();
                };
                visit(environment.WeatherCycles);
                visit(environment.WeatherFXs);
            }
        };
        const auto before_sampler_rebind = [&] {
            release_ray_views();
            retire_weather_descriptors();
        };
        const auto after_sampler_rebind = [&] {
            rebind_ray_views();
            restore_weather_descriptors();
        };
        if (textures_->texture_quality_changed())
        {
            const auto before_texture_reload = [&] {
                before_sampler_rebind();
                if (game_device_)
                    game_device_->detach_color_maps_for_asset_reload();
            };
            const auto after_texture_reload = [&] {
                rebind_ray_views();
                if (game_device_)
                {
                    std::string color_map_error;
                    if (!game_device_->refresh_color_maps_after_asset_reload(color_map_error))
                        Msg("! [renderer-vulkan] color maps unavailable after texture quality change: %s",
                            color_map_error.c_str());
                }
                restore_weather_descriptors();
            };
            if (!textures_->reload_assets(error, before_texture_reload, after_texture_reload, true))
            {
                reset_end();
                xrDebug::Fatal(DEBUG_INFO, "Vulkan texture quality update failed: %s", error.c_str());
                return;
            }
        }
        if (!textures_->update_sampler_settings(static_cast<float>(texture_anisotropy), texture_mip_bias, error,
                before_sampler_rebind, after_sampler_rebind))
        {
            reset_end();
            xrDebug::Fatal(DEBUG_INFO, "Vulkan sampler update failed: %s", error.c_str());
            return;
        }
    }
    if (!game_device_->recreate_swapchain(
            {static_cast<uint32_t>(drawable_width), static_cast<uint32_t>(drawable_height)}, error,
            surface_state_.replace_surface()))
    {
        reset_end();
        xrDebug::Fatal(DEBUG_INFO, "Vulkan swapchain recreation failed: %s", error.c_str());
        return;
    }
    refresh_engine_shader_defines();
    const VkExtent2D extent = game_device_->window().frame().presentation_extent();
    width = extent.width;
    height = extent.height;
    half_width = static_cast<float>(width) * 0.5f;
    half_height = static_cast<float>(height) * 0.5f;
    game_device_->clear_reset_required();
    surface_state_.finish_reset(drawable_width, drawable_height);
    Msg("[renderer-vulkan] swapchain.reset ready logical=%ux%u extent=%ux%u", width, height, extent.width,
        extent.height);
    reset_end();
}

void VulkanLevelRender::reset_begin()
{
    interrupt_benchmark("surface reset");
    R_ASSERT2(game_device_ && !reset_in_progress_ && !frame_phase_.active(),
              "Vulkan renderer reset_begin requires an idle initialized renderer");
    std::string error;
    if (!game_device_->prepare_for_reset(error))
    {
        xrDebug::Fatal(DEBUG_INFO, "Vulkan renderer reset preparation failed: %s", error.c_str());
        return;
    }
    reset_in_progress_ = true;
}

void VulkanLevelRender::reset_end()
{
    R_ASSERT2(reset_in_progress_, "Vulkan renderer reset_end without reset_begin");
    reset_in_progress_ = false;
}

DeviceState VulkanLevelRender::GetDeviceState()
{
    if (!game_device_ || !window_ || !game_device_->window().device())
        return DeviceState::Lost;
    if (game_device_->device_lost())
    {
        if (!device_loss_reported_)
        {
            device_loss_reported_ = true;
            Msg("! [renderer-vulkan] VK_ERROR_DEVICE_LOST: stopping gameplay; restart the app to create a new Vulkan "
                "device");
            Engine.Event.Defer("KERNEL:quit");
        }
        return DeviceState::Lost;
    }
    if (textures_)
    {
        if (textures_->texture_quality_changed())
            game_device_->request_reset();
    }
    int drawable_width = 0, drawable_height = 0;
    SDL_Vulkan_GetDrawableSize(window_, &drawable_width, &drawable_height);
    switch (surface_state_.state(drawable_width, drawable_height, game_device_->device_lost(),
                                 game_device_->reset_required() || game_device_->surface_lost()))
    {
    case SurfaceState::Lost:
        return DeviceState::Lost;
    case SurfaceState::NeedReset:
        return DeviceState::NeedReset;
    case SurfaceState::Normal:
        return DeviceState::Normal;
    }
    return DeviceState::Lost;
}

void VulkanLevelRender::OnAppLifecycleChanged(bool active)
{
    if (!game_device_ || !surface_state_.change_activity(active))
        return;
    Msg("[renderer-vulkan] android.lifecycle active=%d", active ? 1 : 0);
    if (!active)
    {
        if (!game_device_->wait_idle())
            Msg("! Vulkan: device idle wait failed while suspending the Android surface");
        return;
    }

    // The lifecycle state requests a new surface once a drawable returns.
}

void VulkanLevelRender::SetupStates()
{
    R_ASSERT2(game_device_ && game_device_->window().device(),
              "Vulkan renderer state setup requires a created Vulkan device");
    R_ASSERT2(device_resource_state_.setup_states(), "Vulkan renderer SetupStates was called before device creation");
    game_device_->ui().setup_states();
}

void VulkanLevelRender::OnCameraUpdated()
{
    camera_state_.on_camera_updated(Device.mFullTransform.m, Device.vCameraPosition.x, Device.vCameraPosition.y,
                                    Device.vCameraPosition.z);
    update_view_frustum();
}

void VulkanLevelRender::SetCacheXform(Fmatrix &view, Fmatrix &project)
{
    camera_state_.set_cache_xform(view.m, project.m);
    update_view_frustum();
}

void VulkanLevelRender::update_view_frustum()
{
    // Tracers and shared visibility queries read IRender::ViewBase before
    // Calculate. Keep it in sync with the camera, as the other renderers do.
    Fmatrix transform = current_view_projection();
    ViewBase.CreateFromMatrix(transform, FRUSTUM_P_LRTB | FRUSTUM_P_FAR);
}

IRender::RenderContext VulkanLevelRender::GetCurrentContext() const
{
    static_assert(VulkanRenderContextState::NoContext == IRender::NoContext);
    static_assert(VulkanRenderContextState::PrimaryContext == IRender::PrimaryContext);
    static_assert(VulkanRenderContextState::HelperContext == IRender::HelperContext);
    return static_cast<IRender::RenderContext>(context_state_.current());
}

void VulkanLevelRender::MakeContextCurrent(IRender::RenderContext context)
{
    R_ASSERT2(context_state_.make_current(static_cast<int>(context)),
              "Vulkan renderer received an unknown render context");
}

Fmatrix VulkanLevelRender::current_view_projection() const
{
    Fmatrix result = Device.mFullTransform;
    if (camera_state_.has_view_projection())
        camera_state_.copy_view_projection(result.m);
    return result;
}

void VulkanLevelRender::OnDeviceCreate(pcstr shader_archive)
{
    R_ASSERT2(game_device_ && game_device_->window().device(),
              "Vulkan renderer resources require a created Vulkan device");
    R_ASSERT2(device_resource_state_.setup_states(), "Vulkan renderer resources require SetupStates first");
    if (device_resource_state_.can_create_factory_objects())
        return;

    IReader *shader_file = shader_archive ? FS.r_open(shader_archive) : nullptr;
    std::string shader_error;
    if (!shader_file || !textures_->load_blender_library(*shader_file, shader_error))
    {
        if (shader_file)
            FS.r_close(shader_file);
        xrDebug::Fatal(DEBUG_INFO, "Vulkan shaders.xr material library failed (%s): %s",
                       shader_archive ? shader_archive : "<null>",
                       shader_error.empty() ? "missing archive" : shader_error.c_str());
        return;
    }
    // The VFS may have opened this virtual path from a .db/.xdb archive.
    // An explicit diagnostic switch exports the exact selected file (including
    // loose mod overrides) without teaching the renderer any archive format.
    if (std::strstr(Core.Params, "-vk_export_shaders_xr"))
    {
        string_path exported;
        FS.update_path(exported, "$app_data_root$", "vulkan-shaders.xr");
        IWriter *output = FS.w_open(exported);
        if (output)
        {
            // load_blender_library uses open_chunk(), which changes the
            // reader cursor. pointer() points at that cursor, not at byte 0.
            shader_file->seek(0);
            output->w(shader_file->pointer(), shader_file->length());
            FS.w_close(output);
            Msg("[renderer-vulkan] shaders.xr exported file='%s' bytes=%zu", exported, shader_file->length());
        }
        else
            Msg("! [renderer-vulkan] could not export shaders.xr to '%s'", exported);
    }
    FS.r_close(shader_file);
    Msg("[renderer-vulkan] materials.ready shaders.xr blender-count=%zu", textures_->blender_count());

    game_device_->ui().CreateUIGeom();
    if (auto *utils = dynamic_cast<VulkanDrawUtils *>(GEnv.DU))
        utils->OnDeviceCreate();
    R_ASSERT2(device_resource_state_.device_resources_created(),
              "Vulkan renderer device resources were created out of order");
}

void VulkanLevelRender::OnDeviceDestroy(bool)
{
    Msg("[renderer-vulkan] teardown level renderer begin models=%zu", models_.size());
    if (device_ && wait_idle_)
        R_ASSERT2(retirement_idle(device_, wait_idle_), "Vulkan UI teardown requires idle GPU frames");
    Msg("[renderer-vulkan] teardown level renderer idle");
    if (auto *utils = dynamic_cast<VulkanDrawUtils *>(GEnv.DU))
        utils->OnDeviceDestroy();
#ifdef DEBUG
    if (auto *debug = dynamic_cast<VulkanDebugRender *>(GEnv.DRender))
        debug->OnDeviceDestroy();
#endif
    if (game_device_ && textures_)
        for (const auto &[name, texture] : imgui_textures_)
            textures_->release_ui(texture.descriptor, game_device_->ui_pass());
    imgui_textures_.clear();
    screenshot_.reset();
    destroy_all_models();
    Msg("[renderer-vulkan] teardown dynamic models cleared");
    if (!game_device_)
    {
        level_Unload();
        return;
    }

    // level_Unload waits for submitted work before dropping level buffers.
    Msg("[renderer-vulkan] teardown level unload begin");
    level_Unload();
    Msg("[renderer-vulkan] teardown level unload end");
    game_device_->ui().DestroyUIGeom();
    R_ASSERT2(device_resource_state_.device_resources_destroyed(),
              "Vulkan renderer device resources were not active during destruction");
    Msg("[renderer-vulkan] teardown level renderer end");
}

std::unique_ptr<VulkanUIShader> VulkanLevelRender::create_ui_shader()
{
    R_ASSERT2(game_device_ && device_resource_state_.can_create_factory_objects(),
              "Vulkan UI shaders require OnDeviceCreate before level loading");
    return game_device_->create_ui_shader();
}

std::unique_ptr<VulkanFontRender> VulkanLevelRender::create_font_render()
{
    R_ASSERT2(game_device_ && device_resource_state_.can_create_factory_objects(),
              "Vulkan fonts require OnDeviceCreate before level loading");
    return game_device_->create_font_render();
}

void VulkanLevelRender::Begin()
{
    R_ASSERT2(game_device_ && !frame_phase_.active(), "Vulkan renderer Begin requires an initialized idle frame");
    if (assets_dirty_)
        OnAssetsChanged();
    game_device_->begin_frame();
    if (textures_)
        textures_->retire_completed_uploads();
    level_.retire_completed_uploads();
    retire_model_uploads();
    for (auto& mark : wallmarks_)
        if (mark.model)
            mark.model->retire_completed_uploads();
    frame_draw_calls_ = frame_triangles_ = 0;
    if (pass_)
        pass_->reset_draw_statistics();
    R_ASSERT2(frame_phase_.begin(), "Vulkan renderer frame phase could not begin");
    frame_clear_target_ = clear_target_pending_;
    clear_target_pending_ = false;
}

void VulkanLevelRender::Calculate()
{
    R_ASSERT2(game_device_ && frame_phase_.active() && !frame_phase_.world_calculated() &&
                  !frame_phase_.world_rendered(),
              "Vulkan renderer Calculate requires one active frame");

    // The game queries ROS from simulation, AI, rain and HUD code. Refresh its
    // environment and local-light values before the frame's render callbacks.
    const DeferredEnvironment environment = current_environment();
    Lights.Update(lights_);
    game_device_->queue_lights(Lights.SceneLights());
    VulkanHemiLightGrid scene_lights;
    if (!object_specifics_.empty())
        scene_lights.reset(Lights.SceneLights());
    unsigned ray_budget = unsigned(std::max(ros_ray_budget, 0));
    for (size_t i = 0; i < object_specifics_.size(); ++i)
    {
        auto *object = object_specifics_[(i + Device.dwFrame) % object_specifics_.size()];
        if (object)
            object->update(environment.ambient_color, environment.hemi_color, environment.sun_color,
                           scene_lights, &Lights.HemiLights(), &ray_budget, game_device_);
    }

    std::vector<uint32_t> visible_roots;
    IGameObject *view_entity = g_pGameLevel ? g_pGameLevel->CurrentViewEntity() : nullptr;
    size_t camera_sector = view_entity ? view_entity->Sector() : IRender_Sector::INVALID_SECTOR_ID;
    const Fmatrix view_projection_matrix = current_view_projection();
    Fvector camera_position;
    if (camera_state_.has_camera_position())
        camera_position.set(camera_state_.camera_position()[0], camera_state_.camera_position()[1],
                            camera_state_.camera_position()[2]);
    else
        camera_position.set(Device.vCameraPosition);
    if (g_pGameLevel)
    {
        for (float y : {-1.f, 1.f})
        {
            collide::rq_result hit;
            Fvector ray;
            ray.set(0, y, 0);
            const bool found =
                g_pGameLevel->ObjectSpace.RayPick(camera_position, ray, 500.f, collide::rqtStatic, hit, nullptr);
            const size_t static_sector = found ? g_pGameLevel->ObjectSpace.GetStaticTris()[hit.element].sector
                                               : IRender_Sector::INVALID_SECTOR_ID;
            const size_t selected =
                level_.camera_sector(camera_position, ray, found ? hit.range : 500.f, static_sector);
            if (selected != IRender_Sector::INVALID_SECTOR_ID)
            {
                camera_sector = selected;
                break;
            }
        }
    }
    const auto selected_camera_sector = static_cast<IRender_Sector::sector_id_t>(camera_sector);
    if (selected_camera_sector != IRender_Sector::INVALID_SECTOR_ID &&
        selected_camera_sector != last_camera_sector_)
    {
        if (g_pGamePersistent)
            g_pGamePersistent->OnSectorChanged(selected_camera_sector);
        last_camera_sector_ = selected_camera_sector;
    }
    if (!level_.visible_sector_roots(camera_sector, view_projection_matrix, camera_position, visible_roots))
        level_.all_level_roots(visible_roots);
    level_.prepare_details(camera_position, psDeviceFlags.test(rsDrawDetails) ? float(detail_radius) : 0.f);

    float view_projection[16];
    static_assert(sizeof(Fmatrix) == sizeof(view_projection));
    std::memcpy(view_projection, &view_projection_matrix, sizeof(view_projection));
    for (uint32_t root : visible_roots)
    {
        const LevelVisual *node = level_.visual_node(root);
        const Fvector center = visual_center(node);
        const float distance = distance_squared(center, camera_position);
        game_device_->queue_level_visual(root, view_projection, false, distance, nullptr,
                                         lod_for_distance(node ? node->bounds[9] : 0.f, distance));
    }

    // Game objects submit their models only when renderable_Render is called.
    // The static level roots above do not include actors, weapons, physics
    // objects or particles. Traverse the engine's spatial database just as the
    // other gameplay renderers do, then let add_Visual enqueue each submission.
    if (g_pGamePersistent && psDeviceFlags.test(rsDrawDynamic))
    {
        xr_vector<ISpatial *> visible;
        g_pGamePersistent->SpatialSpace.q_frustum(visible, 0, STYPE_RENDERABLE, ViewBase);
        for (ISpatial *spatial : visible)
        {
            if (!spatial)
                continue;
            IRenderable *renderable = spatial->dcast_Renderable();
            if (!renderable || !renderable->GetRenderData().visual)
                continue;
            renderable->renderable_Render(0, renderable);
        }
    }
    // Casters are selected independently of the camera visibility pass. In
    // first person the actor body and NPCs behind the camera never submit a
    // color draw, but must still cast shadows into the visible scene.
    if (g_pGamePersistent && psDeviceFlags.test(rsDrawDynamic))
    {
        xr_vector<ISpatial *> casters;
        g_pGamePersistent->SpatialSpace.q_sphere(casters, 0, STYPE_RENDERABLE, camera_position,
                                                 sun_shadow_span(quality_preset));
        const auto submit_caster = [&](IRenderable *object) {
            if (!object || !object->renderable_ShadowGenerate())
                return;
            auto &data = object->GetRenderData();
            if (data.visual)
                queue_visual(object, data.visual, data.xform, true);
        };
        for (ISpatial *spatial : casters)
            submit_caster(spatial ? spatial->dcast_Renderable() : nullptr);
        // CActor::setVisible(!HUDview()) also removes STYPE_RENDERABLE, so a
        // spatial query alone cannot find the first-person body.
        if (g_pGameLevel)
        {
            submit_caster(g_pGameLevel->CurrentEntity());
            submit_caster(g_pGameLevel->CurrentViewEntity());
        }
    }
    if (g_pGameLevel && g_pGameLevel->pHUD)
        g_pGameLevel->pHUD->Render_Last(IRender::PrimaryContext);

    R_ASSERT2(frame_phase_.calculate(), "Vulkan renderer world calculation is out of order");
}

bool VulkanLevelRender::UpdateProjectileTrail(u64 id, const Fvector& start, const Fvector& end, float width, u32 frame, float dt, u32 color)
{
    return game_device_ && game_device_->update_projectile_trail(id, start, end, width, frame, dt, color);
}

void VulkanLevelRender::UpdateSmokeTrail(const Fvector& position, const Fvector& direction, float dt, bool hud)
{
    if (!game_device_) return;
    Fvector corrected_position = position, corrected_direction = direction;
    if (hud && psHUD_FOV > 0.01f)
    {
        Fmatrix inverse_view, scale, scaled_view, warp;
        inverse_view.invert(Device.mView);
        scale.identity(); scale._11 = scale._22 = 1.f / psHUD_FOV;
        scaled_view.mul(scale, Device.mView); warp.mul(inverse_view, scaled_view);
        warp.transform_tiny(corrected_position); warp.transform_dir(corrected_direction);
        corrected_direction.normalize_safe();
    }
    game_device_->update_smoke_trail(corrected_position, corrected_direction, dt, Device.dwFrame);
}

void VulkanLevelRender::Render()
{
    R_ASSERT2(game_device_ && frame_phase_.active() && frame_phase_.world_calculated() &&
                  !frame_phase_.world_rendered(),
              "Vulkan renderer Render requires a calculated world in the active frame");
    // Geometry is recorded into the deferred G-buffer at End(), after all
    // engine callbacks have queued their transforms and before the UI pass.
    if (g_pGamePersistent && g_pGameLevel)
    {
        auto &environment = g_pGamePersistent->Environment();
        environment.RenderSky();
        environment.RenderClouds();
        if (environment.eff_Rain)
            environment.eff_Rain->Render();
        environment.RenderFlares();
        if (environment.eff_Thunderbolt)
            environment.eff_Thunderbolt->Render();
    }
    for (const VulkanGlow *glow : glows_)
    {
        if (!glow || !glow->active() || glow->radius() <= EPS_L)
            continue;
        const auto &location = glow->position();
        const float facing =
            glow_facing(location, glow->direction(),
                        {Device.vCameraPosition.x, Device.vCameraPosition.y, Device.vCameraPosition.z}, glow->radius());
        if (facing <= EPS_L)
            continue;
        Fvector center, right, up;
        center.set(location[0], location[1], location[2]);
        const auto direction = Fvector().sub(center, Device.vCameraPosition);
        const float distance = direction.magnitude();
        if (std::isfinite(distance) && distance > glow->radius())
        {
            auto &cached = glow_visibility_[glow];
            if (cached.first == 0 || Device.dwFrame - cached.first >= 8)
            {
                collide::rq_result hit;
                Fvector ray;
                ray.div(direction, distance);
                cached.second = !g_pGameLevel->ObjectSpace.RayPick(Device.vCameraPosition, ray,
                                                                   distance - std::min(glow->radius() * 0.25f, 0.25f),
                                                                   collide::rqtStatic, hit, nullptr);
                cached.first = Device.dwFrame;
            }
            if (!cached.second)
                continue;
        }
        const VkDescriptorSet texture = game_device_->glow_texture(glow->texture());
        if (!texture)
            continue;
        right.mul(Device.vCameraRight, glow->radius());
        up.mul(Device.vCameraTop, glow->radius());
        const auto &tint = glow->color();
        const u32 packed = color_rgba_f(tint[0] * facing, tint[1] * facing, tint[2] * facing, tint[3] * facing);
        auto &ui = game_device_->ui();
        ui.SetWorldGlow(texture);
        ui.CacheSetXformWorld(Fidentity);
        ui.CacheSetCullMode(IUIRender::cmNONE);
        ui.StartPrimitive(4, IUIRender::ptTriStrip, IUIRender::pttLIT);
        const float uv[4][2]{{0, 0}, {0, 1}, {1, 0}, {1, 1}};
        const Fvector corners[4]{Fvector().add(center, right).sub(up), Fvector().add(center, right).add(up),
                                 Fvector().sub(center, right).sub(up), Fvector().sub(center, right).add(up)};
        for (size_t i = 0; i < 4; ++i)
            ui.PushPoint(corners[i].x, corners[i].y, corners[i].z, packed, uv[i][0], uv[i][1]);
        ui.FlushPrimitive();
    }
    R_ASSERT2(frame_phase_.render(), "Vulkan renderer world rendering is out of order");
}

void VulkanLevelRender::RenderMenu()
{
    R_ASSERT2(game_device_ && frame_phase_.active(), "Vulkan renderer RenderMenu requires an active frame");
    R_ASSERT2(frame_phase_.render_menu(), "Vulkan renderer menu callbacks are out of order");
    if (!g_pGamePersistent)
        return;

    // PP UI contains a distortion field, not displayable RGB. Keep its
    // batches separate so they can sample the completed menu image.
    g_pGamePersistent->OnRenderPPUI_main();
    game_device_->ui().set_distortion(true);
    g_pGamePersistent->OnRenderPPUI_PP();
    game_device_->ui().set_distortion(false);
}

void VulkanLevelRender::Clear()
{
    R_ASSERT2(game_device_, "Vulkan renderer Clear requires an initialized device");
    // GBufferTargets begins each frame with clear load operations for albedo,
    // normal and depth, so this request is already part of the deferred pass.
    // Match D3D's optional backbuffer clear using the engine's existing flag.
    if (psDeviceFlags.test(rsClearBB))
        ClearTarget();
}

void VulkanLevelRender::ClearTarget()
{
    R_ASSERT2(game_device_, "Vulkan renderer ClearTarget requires an initialized device");
    if (frame_phase_.active())
        frame_clear_target_ = true;
    else
        clear_target_pending_ = true;
}

void VulkanLevelRender::End()
{
    R_ASSERT2(game_device_ && frame_phase_.active(), "Vulkan renderer End requires a begun frame");
    R_ASSERT2(frame_phase_.can_end(), "Vulkan renderer End was reached before the calculated world was rendered");
    // Loading overlays need the whole frame; a narrow gap at their bottom
    // otherwise exposes the already rendered level before user input.
    const bool loading = load_screen_renderer.IsActive();
    const bool render_world = frame_phase_.world_rendered() && !loading;
    R_ASSERT2(frame_phase_.end(), "Vulkan renderer frame end is out of order");

    float mvp[16];
    static_assert(sizeof(Fmatrix) == sizeof(mvp));
    const Fmatrix view_projection = current_view_projection();
    std::memcpy(mvp, &view_projection, sizeof(mvp));

    if (render_world)
        for (auto it = wallmarks_.begin(); it != wallmarks_.end();)
        {
            Wallmark &mark = *it;
            if (mark.remaining_ttl <= 0.f)
            {
                game_device_->retire(std::move(mark.model));
                it = wallmarks_.erase(it);
                continue;
            }
            Fvector object_center = mark.center;
            if (mark.skeleton && mark.bone < mark.skeleton->LL_BoneCount())
            {
                mark.skeleton->CalculateBones();
                mark.skeleton->LL_GetBoneInstance(mark.bone).mRenderTransform.transform_tiny(object_center,
                                                                                                mark.center);
            }
            Fvector world_center = object_center;
            if (mark.transform)
                mark.transform->transform_tiny(world_center, object_center);
            const bool visible = ViewBase.testSphere_dirty(world_center, mark.radius);
            const float opacity = mark.lifetime > 0.f ?
                std::clamp(1.f - mark.remaining_ttl / mark.lifetime, 0.f, 1.f) : 1.f;
            Fmatrix transform = view_projection;
            if (mark.transform)
                transform.mul(view_projection, *mark.transform);
            float matrix[16];
            std::memcpy(matrix, &transform, sizeof(matrix));
            game_device_->queue_model(*mark.model, mark.skeleton, matrix, false,
                                      distance_squared(world_center, Device.vCameraPosition), mark.model.get(), 1.f,
                                      mark.transform, nullptr, false, nullptr, opacity);
            mark.remaining_ttl -= std::max(Device.fTimeDelta, 0.f) * (visible ? .1f : 1.f);
            if (mark.remaining_ttl <= 0.f)
            {
                game_device_->retire(std::move(mark.model));
                it = wallmarks_.erase(it);
            }
            else
                ++it;
        }

    const DeferredEnvironment environment = current_environment();
    DeferredLight light = make_environment_deferred_light(environment);
    for (unsigned i = 0; i < 3; ++i)
        light.color[i] *= sun_enabled ? sun_scale : 0.f;
    postprocess_.grade[0] = gamma_;
    postprocess_.grade[1] = brightness_;
    postprocess_.grade[2] = contrast_;
    postprocess_.noise[3] = Device.fTimeGlobal;
    game_device_->set_postprocess(postprocess_, color_map_a_, color_map_b_);
    FrameStatus status = FrameStatus::Presented;
    std::string error;
    const bool clear_target = frame_clear_target_ || loading;
    frame_clear_target_ = false;
    const auto cpu_begin = std::chrono::steady_clock::now();
    if (!game_device_->render(level_, mvp, light, status, error, render_world, clear_target, &environment))
    {
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_begin).count();
        repeated_render_errors_ = error == last_render_error_ ? repeated_render_errors_ + 1 : 1;
        last_render_error_ = error;
        interrupt_benchmark("frame submission failed");
        if (repeated_render_errors_ <= 4 || (repeated_render_errors_ & (repeated_render_errors_ - 1)) == 0)
            Msg("! Vulkan frame recording/submission failed occurrence=%u frame=%u precache=%u world=%d cpu=%.2fms: %s",
                repeated_render_errors_, Device.dwFrame, Device.dwPrecacheFrame, render_world ? 1 : 0, elapsed,
                error.c_str());
        return;
    }
    if (repeated_render_errors_)
        Msg("[renderer-vulkan] frame recording recovered after %u failures reason=%s", repeated_render_errors_,
            last_render_error_.c_str());
    repeated_render_errors_ = 0;
    last_render_error_.clear();
    const auto now = std::chrono::steady_clock::now();
    const double cpu_ms = std::chrono::duration<double, std::milli>(now - cpu_begin).count();
    if (Device.dwPrecacheFrame &&
        (Device.dwPrecacheFrame == Device.dwPrecacheTotal || Device.dwPrecacheFrame % 10 == 0))
        Msg("[renderer-vulkan] frame.end frame=%u precache=%u status=%d cpu=%.2fms ui-draws=%u", Device.dwFrame,
            Device.dwPrecacheFrame, static_cast<int>(status), cpu_ms, game_device_->last_ui_draw_calls());
    const double frame_ms = last_metric_frame_.time_since_epoch().count()
                                ? std::chrono::duration<double, std::milli>(now - last_metric_frame_).count()
                                : cpu_ms;
    last_metric_frame_ = now;
    const auto geometry_metric_begin = std::chrono::steady_clock::now();
    uint64_t geometry_bytes = level_.geometry_bytes();
    xr_set<const GpuModel *> counted;
    const std::function<void(const VulkanModelVisual &)> count_model = [&](const VulkanModelVisual &visual) {
        if (visual.has_gpu() && counted.insert(&visual.gpu()).second)
            geometry_bytes += visual.gpu().geometry_bytes();
        for (const auto &lod : visual.lod_gpu_handles())
            if (lod && counted.insert(lod.get()).second)
                geometry_bytes += lod->geometry_bytes();
        for (const auto &child : visual.children())
            count_model(*child);
        if (const auto *lod = visual.skeleton_lod())
            count_model(*lod);
    };
    for (const auto &[visual, instance] : models_)
        count_model(*instance);
    for (const auto &[name, base] : model_bases_)
        count_model(*base);
    for (const auto &[name, instance] : model_pool_)
        count_model(*instance);
    const double geometry_metric_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - geometry_metric_begin).count();
    if (status == FrameStatus::Presented)
        save_screenshot();
    if (pass_)
    {
        frame_draw_calls_ = pass_->draw_calls() + game_device_->last_ui_draw_calls();
        frame_triangles_ = pass_->triangles() + game_device_->last_ui_triangles();
    }
    const VulkanFrameSample sample{frame_ms, cpu_ms, game_device_->window().frame().take_gpu_frame_ms(),
                                   textures_ ? textures_->resident_bytes() : 0, geometry_bytes, frame_draw_calls_};
    metrics_.record(sample);
    log_metrics_.record(sample);
    const bool was_recording = benchmark_.recording();
    const bool eligible = render_world && Device.dwPrecacheFrame == 0 && Device.b_is_Active &&
                          !Device.Paused() && (!Console || !Console->bVisible) &&
                          (!g_pGamePersistent || !g_pGamePersistent->IsMainMenuActive());
    if (benchmark_.frame(std::chrono::duration<double, std::milli>(now.time_since_epoch()).count(), sample,
                         eligible, status == FrameStatus::Presented))
        save_benchmark();
    else if (!was_recording && benchmark_.recording())
        Msg("[renderer-vulkan] benchmark.started frames=%u", benchmark_.requested_frames());
    if (metrics_.frame_count() && metrics_.frame_count() % 300 == 0)
    {
        const auto sample = log_metrics_.summary();
        Msg("[renderer-vulkan] frame.progress frames=%llu window=%llu fps=%.1f frame-p95=%.2fms cpu-p95=%.2fms gpu-p95=%.2fms "
            "gpu-missing=%llu textures=%llu geometry=%llu geometry-metric-ms=%.2f draws=%u",
            static_cast<unsigned long long>(metrics_.frame_count()),
            static_cast<unsigned long long>(sample.frames), sample.fps, sample.frame_p95_ms, sample.cpu_p95_ms,
            sample.gpu_p95_ms.value_or(-1.), static_cast<unsigned long long>(sample.gpu_unavailable_frames),
            static_cast<unsigned long long>(sample.texture_bytes),
            static_cast<unsigned long long>(sample.geometry_bytes), geometry_metric_ms,
            frame_draw_calls_);
        log_metrics_ = {};
    }
    if (status == FrameStatus::RecreateRequired)
        surface_state_.request_reset();
}

void VulkanLevelRender::bind_level_device(VulkanGameDevice &resources)
{
    auto &window = resources.window();
    bind_level_device(window.device(), window.queue(), window.frame().command_pool(), window.physical().memory,
                      resources.buffer_upload(), resources.textures(), resources.deferred(),
                      resources.device_wait_idle_proc());
    game_device_ = &resources;
    refresh_engine_shader_defines();
    set_gpu_particle_emitters(resources.gpu_particle_emitters());
    resources.use_scene_visibility(true);
}

void VulkanLevelRender::refresh_engine_shader_defines()
{
    if (!game_device_)
        return;
    const VulkanWindowDevice& window = game_device_->window();
    // These defaults are shared by runtime-created pipelines. The legacy
    // shader_compile hook adds its per-shader skinning and sample options
    // separately; do not persist those transient values as pipeline defaults.
    game_device_->shader_resources().set_engine_defines(engine_shader_defines(window,
        -1, -1, static_cast<uint32_t>(window.frame().scene_samples())));
}

void VulkanLevelRender::bind_level_device(VkDevice device, VkQueue queue, VkCommandPool pool,
                                          const VkPhysicalDeviceMemoryProperties &memory,
                                          const BufferUploadDispatch &upload, GameTextureFactory &textures,
                                          DeferredPass &pass, PFN_vkDeviceWaitIdle wait_idle)
{
    level_Unload();
    if (game_device_)
        game_device_->use_scene_visibility(false);
    set_gpu_particle_emitters(nullptr);
    game_device_ = nullptr;
    device_ = device;
    queue_ = queue;
    pool_ = pool;
    memory_ = memory;
    upload_ = upload;
    textures_ = &textures;
    pass_ = &pass;
    wait_idle_ = wait_idle;
}

void VulkanLevelRender::flush_render_failures()
{
    if (repeated_render_errors_)
        Msg("! [renderer-vulkan] frame recording generation ended failures=%u reason=%s", repeated_render_errors_,
            last_render_error_.c_str());
    repeated_render_errors_ = 0;
    last_render_error_.clear();
}

void VulkanLevelRender::flush_resource_failures()
{
    resource_failures_.report_repeats([](const std::string &context, uint64_t count) {
        Msg("! [renderer-vulkan] repeated resource failure %s occurrences=%llu", context.c_str(),
            static_cast<unsigned long long>(count));
    });
    resource_failures_.clear();
}

void VulkanLevelRender::retire_model_uploads()
{
    auto completed = std::remove_if(pending_model_uploads_.begin(), pending_model_uploads_.end(),
        [](const std::weak_ptr<GpuModel>& weak) {
            auto model = weak.lock();
            return !model || model->retire_completed_uploads();
        });
    pending_model_uploads_.erase(completed, pending_model_uploads_.end());
}

void VulkanLevelRender::level_Load(IReader *reader)
{
    last_camera_sector_ = IRender_Sector::INVALID_SECTOR_ID;
    flush_render_failures();
    flush_resource_failures();
    Msg("[renderer-vulkan] level.load begin reader=%d device=%d queue=%d pool=%d textures=%d pass=%d idle=%d "
        "resources=%d frame=%d",
        reader != nullptr, device_ != VK_NULL_HANDLE, queue_ != VK_NULL_HANDLE, pool_ != VK_NULL_HANDLE,
        textures_ != nullptr, pass_ != nullptr, wait_idle_ != nullptr, device_resource_state_.can_load_level(),
        frame_phase_.active());
    R_ASSERT2(!frame_phase_.active(), "Vulkan level replacement requires an idle frame");
    R_ASSERT2(reader && device_ && queue_ && pool_ && textures_ && pass_ && wait_idle_ &&
                  device_resource_state_.can_load_level(),
              "Vulkan level load requires OnDeviceCreate and an initialized gameplay device");
    // Replacing a level must wait for every submitted draw before its buffers
    // and visual identities are released. Uploads use the same graphics queue.
    const VkResult idle_result = wait_idle_(device_);
    if (idle_result != VK_SUCCESS)
        Msg("! [renderer-vulkan] level.load device wait failed result=%d", int(idle_result));
    R_ASSERT2(idle_result == VK_SUCCESS, "Vulkan queue did not become idle before level load");
    if (game_device_)
        game_device_->discard_scene_draws();
    wallmarks_.clear();
    pending_model_uploads_.clear();
    models_Clear(true);
    model_bases_.clear();
    model_gpu_cache_.clear();
    release_visibility_materials();
    release_ray_materials();
    Msg("[renderer-vulkan] level.load file=$level$/level reading geometry");
    std::string error;
    PFN_vkGetBufferDeviceAddress ray_address = nullptr;
    if (game_device_ && game_device_->window().ray_tracing_enabled())
    {
        const auto& window = game_device_->window();
        const char* name = window.physical().properties.apiVersion >= VK_API_VERSION_1_2 ?
            "vkGetBufferDeviceAddress" : "vkGetBufferDeviceAddressKHR";
        ray_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(
            window.device_proc()(device_, name));
        R_ASSERT2(ray_address, "RT-enabled Vulkan device lacks buffer device address procedure");
    }
    const VulkanWindowDevice* device_owner = game_device_ ? &game_device_->window() : nullptr;
    if (!level_.load(*reader, device_, queue_, pool_, memory_, upload_, *textures_, *pass_, error,
            ray_address, device_owner ? device_owner->device_proc() : nullptr,
            device_owner ? device_owner->physical().properties.limits.maxStorageBufferRange : 0))
    {
        Msg("! [renderer-vulkan] level.load file=$level$/level failed: %s", error.c_str());
        xrDebug::Fatal(DEBUG_INFO, "Vulkan level load file=$level$/level failed: %s", error.c_str());
    }
    if (device_owner && (device_owner->physical().graphics_queue_flags & VK_QUEUE_COMPUTE_BIT) &&
        !level_.enable_gpu_details(device_owner->device_proc(), device_owner->physical().properties.limits, error))
        Msg("! [renderer-vulkan] GPU detail preparation unavailable: %s", error.c_str());
    if (ray_address)
    {
        const auto& window = game_device_->window();
        auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            window.instance_proc()(window.instance(), "vkGetPhysicalDeviceProperties2"));
        if (get_properties)
        {
            VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
            VkPhysicalDeviceDescriptorIndexingProperties indexing{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &acceleration;
            acceleration.pNext = &indexing;
            get_properties(window.physical().handle, &properties);
            auto dispatch = load_acceleration_structure_dispatch(device_, window.device_proc(),
                window.physical().properties.apiVersion >= VK_API_VERSION_1_2);
            if (!level_.build_ray_scene(queue_, dispatch, acceleration, error))
                Msg("! [renderer-vulkan] level ray scene unavailable: %s", error.c_str());
            else if (level_.ray_scene_address())
                Msg("[renderer-vulkan] level ray scene ready BLAS batches=%zu",
                    level_.ray_geometry_batches().size());
            const auto& ray_names = level_.ray_texture_names();
            ray_material_names_ = ray_names;
            ray_diffuse_material_names_ = level_.ray_diffuse_texture_names();
            ray_diffuse_material_linear_ = level_.ray_diffuse_texture_linear();
            if (level_.ray_scene_address() && window.bindless_textures_enabled())
            {
                const uint32_t required_slots = std::max(1u,
                    static_cast<uint32_t>(ray_names.size()));
                const uint32_t limit = ray_query_texture_capacity(window.physical().properties,
                    indexing, acceleration);
                const uint64_t diffuse_base = ray_diffuse_material_names_.size();
                const uint64_t total_limit = indexing.maxUpdateAfterBindDescriptorsInAllPools;
                const uint64_t base_total = uint64_t(required_slots) + diffuse_base;
                const uint64_t spare = base_total <= total_limit ? total_limit - base_total : 0;
                const uint32_t alpha_reserve = static_cast<uint32_t>(std::min<uint64_t>(
                    std::min<uint64_t>(64u, spare / 2), limit >= required_slots ? limit - required_slots : 0));
                const uint32_t diffuse_reserve = static_cast<uint32_t>(std::min<uint64_t>(
                    std::min<uint64_t>(128u, spare - alpha_reserve), limit >= diffuse_base ? limit - diffuse_base : 0));
                const auto& scene = level_.ray_scene();
                const auto max_buffer_range = window.physical().properties.limits.maxStorageBufferRange;
                const bool buffers_fit = scene.hit_batch_buffer().size() <= max_buffer_range &&
                    scene.hit_geometry_buffer().size() <= max_buffer_range &&
                    scene.texture_transform_buffer(0).size() <= max_buffer_range;
                if (required_slots > limit)
                    Msg("! [renderer-vulkan] level ray materials need %u bindless slots; GPU limit=%u",
                        required_slots, limit);
                else if (!buffers_fit)
                    Msg("! [renderer-vulkan] level ray material buffers exceed GPU storage buffer range=%u",
                        max_buffer_range);
                else
                {
                    bool ready = ray_materials_.initialize(device_, window.device_proc(),
                        required_slots + alpha_reserve, error);
                    for (size_t slot = 0; ready && slot < ray_names.size(); ++slot)
                    {
                        VkImageView view = VK_NULL_HANDLE;
                        ready = textures_->projection_texture(ray_names[slot], view, error);
                        if (ready)
                        {
                            ray_material_views_.push_back(view);
                            ready = ray_materials_.write(static_cast<uint32_t>(slot), view,
                                textures_->sampler(), error);
                        }
                    }
                    if (ready && !textures_->finish_uploads())
                    {
                        error = "ray material texture uploads did not complete";
                        ready = false;
                    }
                    if (ready && window.physical().ray_tracing.ray_query &&
                        !ray_queries_.initialize(device_, window.device_proc(), FrameContext::FramesInFlight,
                            VK_NULL_HANDLE, ray_materials_, error))
                        ready = false;
                    if (ready && ray_queries_.ready() && !game_device_->prepare_ray_frames(error))
                        ready = false;
                    if (ready && ray_queries_.ready()) game_device_->set_ray_queries(&ray_queries_, &ray_materials_);
                    if (ready && ray_queries_.ready() && !level_.ray_diffuse_texture_names().empty())
                    {
                        error.clear();
                        const auto& diffuse_names = level_.ray_diffuse_texture_names();
                        const uint64_t diffuse_slots = diffuse_names.size();
                        const uint64_t pool_slots = uint64_t(required_slots) + alpha_reserve + diffuse_slots + diffuse_reserve;
                        bool diffuse_ready = diffuse_slots <= limit &&
                            pool_slots <= indexing.maxUpdateAfterBindDescriptorsInAllPools;
                        if (diffuse_ready)
                            diffuse_ready = ray_diffuse_materials_.initialize(device_, window.device_proc(),
                                static_cast<uint32_t>(diffuse_slots + diffuse_reserve), error);
                        for (size_t slot = 0; diffuse_ready && slot < diffuse_names.size(); ++slot)
                        {
                            VkImageView view = VK_NULL_HANDLE;
                            diffuse_ready = textures_->projection_texture(diffuse_names[slot], view, error,
                                level_.ray_diffuse_texture_linear()[slot] != 0);
                            if (diffuse_ready)
                            {
                                ray_diffuse_material_views_.push_back(view);
                                diffuse_ready = ray_diffuse_materials_.write(static_cast<uint32_t>(slot), view,
                                    textures_->sampler(), error);
                            }
                        }
                        if (diffuse_ready && !textures_->finish_uploads())
                        {
                            error = "ray diffuse texture uploads did not complete";
                            diffuse_ready = false;
                        }
                        if (diffuse_ready)
                            diffuse_ready = ray_surfaces_.initialize(device_, window.device_proc(),
                                FrameContext::FramesInFlight, window.physical().properties.limits, indexing,
                                ray_diffuse_materials_, VK_NULL_HANDLE, error);
                        if (diffuse_ready)
                            diffuse_ready = game_device_->prepare_indirect_frames(ray_surfaces_, ray_materials_, error);
                        if (!diffuse_ready)
                        {
                            Msg("! [renderer-vulkan] ray diffuse materials unavailable: %s",
                                error.empty() ? "GPU descriptor limits exceeded" : error.c_str());
                            release_ray_diffuse_materials();
                        }
                        else
                        {
                            game_device_->set_model_ray_preparer([this](GpuModel& model, IKinematics* skeleton,
                                uint32_t slot, VkCommandBuffer command, const VkTransformMatrixKHR& transform,
                                RayGeometryBatch& batch, std::vector<TextureMatrixSet>& matrices,
                                bool& updated, std::string& reason) {
                                return prepare_model_ray_batch(model, skeleton, slot, command, transform, batch, matrices, updated, reason);
                            });
                            Msg("[renderer-vulkan] ray diffuse material table ready textures=%zu", diffuse_names.size());
                        }
                    }
                    if (ready)
                        Msg("[renderer-vulkan] level ray material table ready textures=%zu",
                            ray_material_views_.size());
                    else
                    {
                        Msg("! [renderer-vulkan] level ray material table unavailable: %s", error.c_str());
                        release_ray_materials();
                    }
                }
            }
        }
        else
            Msg("! [renderer-vulkan] level ray scene properties procedure unavailable");
    }
    if (device_owner && !prepare_visibility_materials(*device_owner, error))
        Msg("! [renderer-vulkan] visibility buffer materials unavailable: %s", error.c_str());
    Lights.Load(*reader);
    Msg("[renderer-vulkan] level.load ready visuals=%zu models=%zu textures=%zu", level_.visual_count(),
        level_.model_count(), textures_->resident_count());
}

void VulkanLevelRender::level_Unload()
{
    last_camera_sector_ = IRender_Sector::INVALID_SECTOR_ID;
    interrupt_benchmark("level unloaded");
    Lights.Unload();
    flush_render_failures();
    flush_resource_failures();
    if (device_ && (level_.visual_count() || !models_.empty()))
        Msg("[renderer-vulkan] level.unload begin visuals=%zu models=%zu", level_.visual_count(), models_.size());
    R_ASSERT2(!frame_phase_.active(), "Vulkan level unload requires an idle frame");
    if (device_ && wait_idle_)
        R_ASSERT2(retirement_idle(device_, wait_idle_), "Vulkan queue did not become idle before level unload");
    if (game_device_)
    {
        game_device_->discard_scene_draws();
        game_device_->release_level_glows();
    }
    wallmarks_.clear();
    glow_visibility_.clear();
    pending_model_uploads_.clear();
    models_Clear(true);
    model_bases_.clear();
    model_gpu_cache_.clear();
    release_visibility_materials();
    release_ray_materials();
    level_.destroy();
}

bool VulkanLevelRender::prepare_visibility_materials(const VulkanWindowDevice& window, std::string& error)
{
    const auto& names = level_.visibility_textures();
    error.clear();
    const char* enabled = SDL_getenv("XRAY_VK_VISIBILITY_BUFFER");
    if (!enabled || std::strcmp(enabled, "1") != 0 || names.empty() || !window.bindless_textures_enabled() ||
        !(window.physical().graphics_queue_flags & VK_QUEUE_COMPUTE_BIT))
        return true;

    auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        window.instance_proc()(window.instance(), "vkGetPhysicalDeviceProperties2"));
    if (!get_properties)
    {
        error = "descriptor indexing properties unavailable";
        return false;
    }
    VkPhysicalDeviceDescriptorIndexingProperties indexing{};
    indexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &indexing;
    get_properties(window.physical().handle, &properties);
    const uint64_t total = uint64_t(ray_materials_.capacity()) + ray_diffuse_materials_.capacity() + names.size();
    const auto per_stage = std::min({indexing.maxPerStageDescriptorUpdateAfterBindSamplers,
        indexing.maxPerStageDescriptorUpdateAfterBindSampledImages,
        indexing.maxDescriptorSetUpdateAfterBindSamplers, indexing.maxDescriptorSetUpdateAfterBindSampledImages});
    if (names.size() >= per_stage || total > indexing.maxUpdateAfterBindDescriptorsInAllPools)
    {
        error = "visibility materials exceed descriptor indexing limits";
        return false;
    }
    if (!visibility_materials_.initialize(device_, window.device_proc(), static_cast<uint32_t>(names.size()), error))
        return false;

    bool ready = true;
    visibility_material_views_.reserve(names.size());
    for (uint32_t slot = 0; slot < names.size(); ++slot)
    {
        VkImageView view{};
        const auto& texture = names[slot];
        if (!textures_->projection_texture(texture.name, view, error))
        {
            ready = false;
            break;
        }
        visibility_material_views_.push_back(view);
        const auto sampler = texture.lightmap ? textures_->lightmap_sampler() : textures_->sampler();
        if (!visibility_materials_.write(slot, view, sampler, error, VK_IMAGE_LAYOUT_GENERAL))
        {
            ready = false;
            break;
        }
    }
    if (ready && !textures_->finish_uploads())
    {
        error = "visibility texture uploads did not complete";
        ready = false;
    }
    if (ready)
    {
        auto passes = std::make_unique<VisibilityPasses>();
        if (passes->initialize(device_, window.device_proc(), properties.properties.limits,
                FrameContext::FramesInFlight, static_cast<uint32_t>(level_.visibility_draw_ranges().size()),
                pass_->pipeline_cache(), visibility_materials_, read_engine_shader, error))
        {
            visibility_passes_ = std::move(passes);
            game_device_->set_level_visibility(visibility_passes_.get(), &visibility_materials_);
            return true;
        }
    }
    release_visibility_materials();
    return false;
}

void VulkanLevelRender::release_visibility_materials()
{
    if (visibility_passes_ && game_device_)
        game_device_->set_level_visibility(nullptr, nullptr);
    visibility_passes_.reset();
    visibility_materials_.destroy();
    if (textures_)
        for (VkImageView view : visibility_material_views_)
            textures_->release_environment(view);
    visibility_material_views_.clear();
}

void VulkanLevelRender::release_ray_materials()
{
    if (game_device_) game_device_->set_model_ray_preparer({});
    release_ray_diffuse_materials();
    if (game_device_) game_device_->release_ray_frames();
    ray_queries_.destroy();
    ray_materials_.destroy();
    if (textures_)
        for (VkImageView view : ray_material_views_)
            textures_->release_environment(view);
    ray_material_views_.clear();
    ray_material_names_.clear();
}

bool VulkanLevelRender::prepare_model_ray_batch(GpuModel& model, IKinematics* skeleton, uint32_t slot,
    VkCommandBuffer command, const VkTransformMatrixKHR& transform, RayGeometryBatch& batch,
    std::vector<TextureMatrixSet>& matrices, bool& recorded_update, std::string& error)
{
    recorded_update = false;
    if (!model.ray_geometry_enabled()) { error.clear(); return true; }
    if (!game_device_) { error = "model ray device unavailable"; return false; }
    const auto& window = game_device_->window();
    const auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
        window.instance_proc()(window.instance(), "vkGetPhysicalDeviceProperties2"));
    if (!get_properties) { error = "model ray properties procedure unavailable"; return false; }
    VkPhysicalDeviceAccelerationStructurePropertiesKHR acceleration{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties.pNext = &acceleration;
    get_properties(window.physical().handle, &properties);
    const auto dispatch = load_acceleration_structure_dispatch(device_, window.device_proc(),
        window.physical().properties.apiVersion >= VK_API_VERSION_1_2);
    VkDeviceAddress address = 0;
    const std::vector<RayModelGeometry>* geometry = nullptr;
    if (skeleton)
    {
        if (!model.prepare_skinned_ray_blas(slot, command, *skeleton, dispatch, acceleration,
                address, geometry, recorded_update, error)) return false;
    }
    else
    {
        if (!model.prepare_static_ray_blas(dispatch, acceleration, error)) return false;
        address = model.static_ray_blas_address();
        geometry = &model.static_ray_geometry();
    }
    if (!geometry || geometry->empty()) { error.clear(); return true; }
    std::vector<RayModelTextureSlots> slots;
    if (!acquire_model_ray_materials(*geometry, slots, error) ||
        !make_model_ray_batch(address, transform, *geometry, slots, batch, error))
        return false;
    matrices.clear();
    for (const auto& mesh : *geometry) matrices.push_back(mesh.texture_matrices);
    return true;
}

void VulkanLevelRender::release_ray_diffuse_materials()
{
    if (game_device_) game_device_->release_indirect_frames();
    ray_surfaces_.destroy();
    ray_diffuse_materials_.destroy();
    if (textures_)
        for (VkImageView view : ray_diffuse_material_views_)
            textures_->release_environment(view);
    ray_diffuse_material_views_.clear();
    ray_diffuse_material_names_.clear();
    ray_diffuse_material_linear_.clear();
}

bool VulkanLevelRender::acquire_model_ray_materials(const std::vector<RayModelGeometry>& geometry,
    std::vector<RayModelTextureSlots>& slots, std::string& error)
{
    if (!textures_ || !ray_materials_.set() || !ray_diffuse_materials_.set())
    { error = "model ray material tables are unavailable"; return false; }
    auto acquire = [&](const std::string& name, bool linear, bool alpha, uint32_t& result) -> bool {
        result = UINT32_MAX;
        if (name.empty()) return true;
        auto& names = alpha ? ray_material_names_ : ray_diffuse_material_names_;
        auto& views = alpha ? ray_material_views_ : ray_diffuse_material_views_;
        auto& table = alpha ? ray_materials_ : ray_diffuse_materials_;
        for (size_t i = 0; i < names.size(); ++i)
            if (names[i] == name && (alpha || (ray_diffuse_material_linear_[i] != 0) == linear))
            { result = static_cast<uint32_t>(i); return true; }
        if (names.size() >= table.capacity())
        { error = "model ray material table capacity exhausted"; return false; }
        VkImageView view = VK_NULL_HANDLE;
        if (!textures_->projection_texture(name, view, error, !alpha && linear)) return false;
        if (!textures_->finish_uploads())
        {
            textures_->release_environment(view);
            error = "model ray material upload did not complete";
            return false;
        }
        const uint32_t index = static_cast<uint32_t>(names.size());
        if (!table.write(index, view, textures_->sampler(), error))
        { textures_->release_environment(view); return false; }
        names.push_back(name);
        views.push_back(view);
        if (!alpha) ray_diffuse_material_linear_.push_back(linear ? 1 : 0);
        result = index;
        return true;
    };
    std::vector<RayModelTextureSlots> collected;
    collected.reserve(geometry.size());
    for (const auto& material : geometry)
    {
        RayModelTextureSlots item{};
        if ((material.alpha_cutoff > 0.f && !acquire(material.diffuse_texture, false, true, item.alpha)) ||
            !acquire(material.diffuse_texture, false, false, item.diffuse) ||
            !acquire(material.pbr_texture, true, false, item.pbr) ||
            !acquire(material.metallic_texture, true, false, item.metallic) ||
            !acquire(material.roughness_texture, true, false, item.roughness) ||
            !acquire(material.ao_texture, true, false, item.ao) ||
            !acquire(material.normal_texture, true, false, item.normal) ||
            !acquire(material.normal_aux_texture, true, false, item.normal_aux))
            return false;
        collected.push_back(item);
    }
    slots = std::move(collected);
    error.clear();
    return true;
}

IRenderVisual *VulkanLevelRender::getVisual(int index)
{
    return index >= 0 ? level_.get_visual(static_cast<size_t>(index)) : nullptr;
}

HRESULT VulkanLevelRender::shader_compile(pcstr name, IReader *source, pcstr entry, pcstr target, u32 flags,
                                          void *&result)
{
    result = nullptr;
    // Prefer host-compiled SPIR-V when available; 64-bit Android builds also
    // compile missing source variants through Slang.
    GameShaderStage stage{};
    if (!device_ || !game_device_ || !source || !source->length() || !name || !*name || !entry || !*entry ||
        !GameShaderResources::stage_from_target(target, stage))
    {
        Msg("! [renderer-vulkan] shader_compile: invalid device, entry or stage for '%s'", name ? name : "<unnamed>");
        return E_FAIL;
    }
    refresh_engine_shader_defines();
    std::string error;
    GameShaderResource *resource = nullptr;
    const std::string relative = std::string(getShaderPath()) + name;
    const auto *source_bytes = static_cast<const uint8_t *>(source->pointer());
    const size_t source_size = source->length();
    const std::vector<std::string> defines = GameShaderResources::has_manifest_variant(relative, stage, entry)
        ? std::vector<std::string>{}
        : engine_shader_defines(game_device_->window(), m_skinning, m_MSAASample,
              static_cast<uint32_t>(game_device_->window().frame().scene_samples()));
    const bool compiled = game_device_->shader_resources().shader(relative, stage, entry, flags,
        resource, error, source_bytes, source_size, false, defines);
    if (!compiled)
    {
        Msg("! [renderer-vulkan] shader_compile: '%s': %s", name, error.c_str());
        return E_FAIL;
    }
    result = resource;
    return S_OK;
}

IRenderVisual *VulkanLevelRender::model_Create(pcstr name, IReader *data)
{
    R_ASSERT2(device_ && queue_ && pool_ && textures_ && pass_ && wait_idle_ &&
                  device_resource_state_.can_create_factory_objects(),
              "Vulkan model creation requires an initialized gameplay device");
    retire_model_uploads();
    const std::string cache_name = model_cache_name(name);
    if (!cache_name.empty())
    {
        auto reused = model_pool_.find(cache_name);
        if (reused != model_pool_.end())
        {
            auto instance = std::move(reused->second);
            model_pool_.erase(reused);
            instance->reset_instance_state();
            IRenderVisual *visual = instance.get();
            models_.emplace(visual, std::move(instance));
            return visual;
        }
    }
    std::string error;
    std::unique_ptr<VulkanModelVisual> model;
    if (cache_name.empty())
        model = load_model_base(name, data, "", error);
    else if (VulkanModelVisual *base = get_model_base(name, data, cache_name, error))
        model = std::make_unique<VulkanModelVisual>(*base);
    if (!model)
    {
        const std::string context =
            "file=" + std::string(name ? name : "<reader>.ogf") + " stage=model_Create reason=" + error;
        if (resource_failures_.record(context))
            Msg("! [renderer-vulkan] %s", context.c_str());
        return nullptr;
    }
    IRenderVisual *visual = model.get();
    models_.emplace(visual, std::move(model));
    return visual;
}

void VulkanLevelRender::set_gpu_particle_emitters(GpuParticleEmitters* emitters)
{
    if (gpu_particle_emitters_ == emitters) return;
    for (auto& item : particle_effects_) item.second->detach_gpu_emitter();
    for (auto& item : particle_groups_) item.second->detach_gpu_emitters();
    gpu_particle_emitters_ = emitters;
}

IRenderVisual *VulkanLevelRender::model_CreateParticles(pcstr name)
{
    R_ASSERT2(device_ && textures_ && pass_ && wait_idle_ && device_resource_state_.can_create_factory_objects(),
              "Vulkan particles require an initialized gameplay device");
    if (!name || !*name)
        return nullptr;
    const auto unavailable_particle = [&](const char *reason) -> IRenderVisual * {
        // Gameplay calls IParticleCustom immediately after creation. A mod's
        // unsupported particle definition must not turn into a null visual.
        const std::string context = "file=$game_data$/particles.xr particle='" + std::string(name) +
                                    "' stage=model_CreateParticles reason=" + reason + "; using inert visual";
        if (resource_failures_.record(context))
            Msg("! [renderer-vulkan] %s", context.c_str());
        auto catalog = std::make_shared<ParticleCatalog>();
        auto &definition = catalog->groups.emplace_back();
        definition.name = name;
        definition.time_limit = 1.f;
        auto visual = std::make_unique<VulkanParticleGroup>(catalog, definition);
        IRenderVisual *result = visual.get();
        particle_groups_.emplace(result, std::move(visual));
        return result;
    };
    if (!particle_catalog_)
    {
        string_path path;
        FS.update_path(path, _game_data_, "particles.xr");
        IReader *source = FS.r_open(path);
        if (!source)
        {
            Msg("! [renderer-vulkan] particle library not found: %s", path);
            particle_catalog_ = std::make_shared<ParticleCatalog>();
            return unavailable_particle("library not found");
        }
        auto catalog = std::make_shared<ParticleCatalog>();
        std::string error;
        const auto normalized = particle_library_bytes(*source);
        const bool parsed = parse_particle_catalog({normalized.data(), normalized.size()}, *catalog, error);
        FS.r_close(source);
        if (!parsed)
        {
            Msg("! [renderer-vulkan] particle library: %s", error.c_str());
            particle_catalog_ = std::make_shared<ParticleCatalog>();
            return unavailable_particle(error.c_str());
        }
        if (!error.empty())
            Msg("! [renderer-vulkan] particle library: %s", error.c_str());
        particle_catalog_ = std::move(catalog);
    }
    std::string error;
    if (game_device_ && gpu_particle_emitters_)
        game_device_->set_gpu_particle_catalog(particle_catalog_);
    if (const auto *def = particle_catalog_->effect(name))
    {
        auto visual = std::make_unique<VulkanParticleEffect>(particle_catalog_, *def);
        if (!visual->initialize(device_, memory_, upload_.buffer, *textures_, *pass_, error))
        {
            return unavailable_particle(error.c_str());
        }
        if (gpu_particle_emitters_ && !visual->attach_gpu_emitter(*gpu_particle_emitters_, error))
            Msg("! [renderer-vulkan] GPU particle mirror '%s' unavailable: %s", def->name.c_str(), error.c_str());
        IRenderVisual *result = visual.get();
        particle_effects_.emplace(result, std::move(visual));
        return result;
    }
    if (const auto *def = particle_catalog_->group(name))
    {
        auto visual = std::make_unique<VulkanParticleGroup>(particle_catalog_, *def);
        if (!visual->initialize(device_, memory_, upload_.buffer, *textures_, *pass_, error))
        {
            return unavailable_particle(error.c_str());
        }
        if (gpu_particle_emitters_ && !visual->attach_gpu_emitters(*gpu_particle_emitters_, error))
            Msg("! [renderer-vulkan] GPU particle group mirror '%s' unavailable: %s", def->name.c_str(), error.c_str());
        IRenderVisual *result = visual.get();
        particle_groups_.emplace(result, std::move(visual));
        return result;
    }
    return unavailable_particle("effect or group not found");
}

std::unique_ptr<VulkanModelVisual> VulkanLevelRender::load_model_base(pcstr name, IReader *data,
                                                                      const std::string &cache_name, std::string &error,
                                                                      bool skeletal_child)
{
    VisualRecord record;
    const bool parsed =
        data ? load_engine_model_reader(*data, name, record, error) : load_engine_model_visual(name, record, error);
    if (!parsed)
        return nullptr;
    auto result = create_model_tree(record, "", cache_name, error, skeletal_child);
    if (!result)
        error = std::string(name && *name ? name : "<reader>.ogf") + " / type=" + std::to_string(record.type) +
                " shader_id=" + std::to_string(record.shader_id) + " create: " + error;
    return result;
}

VulkanModelVisual *VulkanLevelRender::get_model_base(pcstr name, IReader *data, const std::string &cache_name,
                                                     std::string &error, bool skeletal_child)
{
    auto found = model_bases_.find(cache_name);
    if (found != model_bases_.end())
        return found->second.get();
    auto base = load_model_base(name, data, cache_name, error, skeletal_child);
    if (!base)
        return nullptr;
    VulkanModelVisual *visual = base.get();
    model_bases_.emplace(cache_name, std::move(base));
    return visual;
}

std::unique_ptr<VulkanModelVisual> VulkanLevelRender::create_model_tree(const VisualRecord &record,
                                                                        const std::string &inherited_texture,
                                                                        const std::string &cache_name,
                                                                        std::string &error, bool skeletal_child,
                                                                        unsigned skeleton_lod_depth)
{
    const bool skeleton = record.type == 3 || record.type == 10;
    const bool skinned = record.type == 4 || record.type == 5;
    const bool tree = record.type == 7 || record.type == 11;
    if (record.type != 0 && record.type != 1 && record.type != 2 && record.type != 6 && !tree && !skeleton &&
        !(skeletal_child && skinned))
    {
        error = "unsupported OGF model type " + std::to_string(record.type);
        return nullptr;
    }
    if ((record.type == 0 || record.type == 2 || tree) &&
        (!record.embedded_children.empty() || !record.linked_children.empty()))
    {
        error = "OGF mesh unexpectedly contains child references";
        return nullptr;
    }
    if (skinned && (!record.embedded_children.empty() || !record.linked_children.empty()))
    {
        error = "OGF skinned child unexpectedly contains nested visuals";
        return nullptr;
    }
    if ((record.type == 1 || record.type == 6) && (record.embedded_children.empty() == record.linked_children.empty()))
    {
        error = "OGF hierarchy needs exactly one nonempty child table";
        return nullptr;
    }
    std::shared_ptr<GpuModel> gpu;
    if (record.type == 0 || record.type == 2 || tree || skinned)
    {
        if (!cache_name.empty())
            gpu = model_gpu_cache_[cache_name].lock();
        if (!gpu)
        {
            gpu = std::make_shared<GpuModel>();
            PFN_vkGetBufferDeviceAddress ray_address = nullptr;
            if (game_device_ && game_device_->window().ray_tracing_enabled())
            {
                const auto& window = game_device_->window();
                const char* name = window.physical().properties.apiVersion >= VK_API_VERSION_1_2 ?
                    "vkGetBufferDeviceAddress" : "vkGetBufferDeviceAddressKHR";
                ray_address = reinterpret_cast<PFN_vkGetBufferDeviceAddress>(window.device_proc()(device_, name));
            }
            if (!gpu->load_record(record, inherited_texture, device_, queue_, pool_, memory_, upload_, *textures_,
                                  *pass_, error, ray_address))
            {
                if (!ray_address) return nullptr;
                Msg("! [renderer-vulkan] model ray buffers unavailable, retrying raster: %s", error.c_str());
                if (!gpu->load_record(record, inherited_texture, device_, queue_, pool_, memory_, upload_, *textures_,
                                      *pass_, error)) return nullptr;
            }
            if (!gpu->retire_completed_uploads())
                pending_model_uploads_.push_back(gpu);
            if (!cache_name.empty())
                model_gpu_cache_[cache_name] = gpu;
        }
    }
    else if (!record.linked_children.empty())
    {
        for (uint32_t child : record.linked_children)
            if (!level_.get_visual(child))
            {
                error = "OGF hierarchy refers to a missing level visual";
                return nullptr;
            }
    }
    auto model = std::make_unique<VulkanModelVisual>(record, cache_name, std::move(gpu),
                                                     record.linked_children.empty() ? nullptr : &level_);
    if (skeleton && !skeletal_child && !record.skeleton_lod.empty() && skeleton_lod_depth == 0)
    {
        VisualRecord lod_record;
        if (!load_engine_model_visual(record.skeleton_lod.c_str(), lod_record, error))
        {
            Msg("! [renderer-vulkan] ignoring optional skeleton LOD '%s': %s",
                record.skeleton_lod.c_str(), error.c_str());
            error.clear();
        }
        else if (!is_static_skeleton_lod_type(lod_record.type))
        {
            Msg("! [renderer-vulkan] ignoring optional skeleton LOD '%s' with non-static OGF type=%u",
                record.skeleton_lod.c_str(), unsigned(lod_record.type));
        }
        else
        {
            const std::string lod_cache_name = cache_name.empty()
                ? std::string("skeleton-lod:") + record.skeleton_lod
                : cache_name + ":lod:" + record.skeleton_lod;
            auto lod_visual = create_model_tree(lod_record, "", lod_cache_name, error, false,
                                                skeleton_lod_depth + 1);
            if (lod_visual)
                model->set_skeleton_lod(std::move(lod_visual));
            else
            {
                Msg("! [renderer-vulkan] ignoring unusable optional skeleton LOD '%s': %s",
                    record.skeleton_lod.c_str(), error.c_str());
                error.clear();
            }
        }
    }
    if (record.type == 6)
    {
        IReader source(const_cast<uint8_t *>(record.source.data()), record.source.size());
        IReader *raw = source.open_chunk(OGF_LODDEF2);
        LodFacets facets;
        if (!raw || !decode_lod_facets({static_cast<const uint8_t *>(raw->pointer()), raw->length()}, 0, facets, error))
        {
            if (raw)
                raw->close();
            if (error.empty())
                error = "OGF LOD has no valid facets";
            return nullptr;
        }
        raw->close();
        std::array<std::shared_ptr<GpuModel>, 8> lod_gpu;
        const auto &texture = !record.texture.empty()             ? record.texture
                              : !record.embedded_children.empty() ? record.embedded_children.front().texture
                                                                  : inherited_texture;
        for (size_t face = 0; face < lod_gpu.size(); ++face)
        {
            ModelGeometry geometry;
            geometry.type = 0;
            geometry.texture = texture;
            geometry.shader = !record.shader.empty()              ? record.shader
                              : !record.embedded_children.empty() ? record.embedded_children.front().shader
                                                                  : std::string{};
            geometry.mode = SurfaceMode::Unresolved;
            geometry.indices = std::move(facets.meshes[face].indices);
            for (const LevelVertex &vertex : facets.meshes[face].vertices)
            {
                ModelVertex converted;
                std::copy_n(vertex.position, 3, converted.position);
                std::copy_n(vertex.normal, 3, converted.normal);
                std::copy_n(vertex.uv, 2, converted.uv);
                geometry.vertices.push_back(converted);
            }
            lod_gpu[face] = std::make_shared<GpuModel>();
            if (!lod_gpu[face]->load_decoded(std::move(geometry), device_, queue_, pool_, memory_, upload_, *textures_,
                                             *pass_, error))
                return nullptr;
            if (!lod_gpu[face]->retire_completed_uploads())
                pending_model_uploads_.push_back(lod_gpu[face]);
        }
        model->set_lod(std::move(lod_gpu), facets.normals);
    }
    if (skeleton)
    {
        if (record.embedded_children.empty() || !record.linked_children.empty())
        {
            error = "Vulkan skeleton needs embedded child meshes";
            return nullptr;
        }
        SkeletonBones bones;
        if (!parse_skeleton_bones({record.source.data(), record.source.size()}, bones, error))
            return nullptr;
        auto instance = VulkanKinematics::create(bones, model.get(), error);
        if (!instance)
            return nullptr;
        if (record.type == 3)
        {
            std::vector<std::string> bone_names;
            bone_names.reserve(bones.bones.size());
            for (const auto &bone : bones.bones)
                bone_names.push_back(bone.name);
            std::vector<MotionSlot> motions;
            if (!parse_skeleton_motions({record.source.data(), record.source.size()}, bone_names, cache_name,
                                        load_engine_motion_files, motions, error))
                return nullptr;
            instance->set_motions(std::move(motions));
        }
        // The regular model pool assigns runtime material indices after OGF
        // loading. Physics uses these indices for bone contacts; leaving the
        // OGF's material names unresolved makes every contact use u16(-1).
        R_ASSERT2(GMLib.GetMaterial("default_object"), "game material library has no default_object");
        const u16 fallback_material = GMLib.GetMaterialIdx("default_object");
        for (u16 bone = 0; bone < instance->LL_BoneCount(); ++bone)
        {
            CBoneData &data = instance->LL_GetData(bone);
            const char *material_name = data.game_mtl_name.c_str();
            if (material_name && GMLib.GetMaterial(material_name))
                data.game_mtl_idx = GMLib.GetMaterialIdx(material_name);
            else
            {
                data.game_mtl_idx = fallback_material;
                if (material_name && *material_name)
                    Msg("! [renderer-vulkan] OGF '%s' bone='%s' material='%s' missing; using default_object",
                        cache_name.c_str(), data.name.c_str(), material_name);
            }
        }
        model->set_skeleton(std::move(instance));
    }
    const std::string texture = record.texture.empty() ? inherited_texture : record.texture;
    for (const auto &child : record.embedded_children)
    {
        auto instance = create_model_tree(child, texture, "", error, skeletal_child || skeleton,
                                          skeleton_lod_depth);
        if (!instance)
            return nullptr;
        model->add_child(std::move(instance));
    }
    if (skeleton)
    {
        ModelGeometry geometry;
        if (!decode_model_geometry(record, geometry, error) || !model->skeleton()->attach_geometry(geometry, error))
            return nullptr;
    }
    return model;
}

IRenderVisual *VulkanLevelRender::model_CreateChild(pcstr name, IReader *data)
{
    R_ASSERT2(device_ && queue_ && pool_ && textures_ && pass_ && wait_idle_ &&
                  device_resource_state_.can_create_factory_objects(),
              "Vulkan child model creation requires an initialized gameplay device");
    retire_model_uploads();
    const std::string filename = model_cache_name(name);
    const std::string cache_name = filename.empty() ? "" : "child:" + filename;
    std::string error;
    std::unique_ptr<VulkanModelVisual> child;
    if (cache_name.empty())
        child = load_model_base(name, data, "", error, true);
    else if (VulkanModelVisual *base = get_model_base(name, data, cache_name, error, true))
    {
        child = std::make_unique<VulkanModelVisual>(*base);
        // Children belong to their parent; deleting one must not put it in
        // the pool of standalone model instances.
        child->set_cache_name("");
    }
    if (!child)
    {
        Msg("! [renderer-vulkan] OGF child '%s': %s", name ? name : "<reader>", error.c_str());
        return nullptr;
    }
    IRenderVisual *visual = child.get();
    models_.emplace(visual, std::move(child));
    return visual;
}

void VulkanLevelRender::add_Visual(u32, IRenderable *root, IRenderVisual *visual, Fmatrix &world)
{
    queue_visual(root, visual, world, false);
}

void VulkanLevelRender::queue_visual(IRenderable *root, IRenderVisual *visual, Fmatrix &world, bool shadow_only)
{
    R_ASSERT2(game_device_, "Vulkan level visuals require a bound gameplay device");
    if (shadow_only && (particle_effects_.count(visual) || particle_groups_.count(visual)))
        return;
    if (particle_effects_.count(visual) || particle_groups_.count(visual))
    {
        const bool hud = !shadow_only && root && root->renderable_HUD();
        const Fmatrix view_projection = hud ? hud_view_projection() : current_view_projection();
        Fvector camera_position;
        if (camera_state_.has_camera_position())
            camera_position.set(camera_state_.camera_position()[0], camera_state_.camera_position()[1],
                                camera_state_.camera_position()[2]);
        else
            camera_position.set(Device.vCameraPosition);
        const Fvector center = visual->getVisData().sphere.P;
        float transform[16];
        std::memcpy(transform, &view_projection, sizeof(transform));
        game_device_->queue_particle(visual, transform, Device.vCameraRight, Device.vCameraTop, hud,
                                     distance_squared(center, camera_position));
        return;
    }
    const auto direct_model = models_.find(visual);
    const int visual_index = direct_model == models_.end() ? level_.find_visual_index(visual) : -1;
    const VulkanModelVisual *model = nullptr;
    const void *model_owner = nullptr;
    if (direct_model != models_.end())
    {
        model = direct_model->second.get();
        model_owner = direct_model->first;
    }
    else if (visual_index < 0)
    {
        const auto cached = model_visual_index_.find(visual);
        if (cached != model_visual_index_.end())
        {
            model = cached->second.first;
            model_owner = cached->second.second;
        }
        else
            for (const auto &entry : models_)
                if ((model = entry.second->find(visual)))
                {
                    model_owner = entry.first;
                    model_visual_index_.emplace(visual, std::make_pair(model, model_owner));
                    break;
                }
    }
    R_ASSERT2(visual_index >= 0 || model, "Vulkan scene submission received a visual outside this renderer");
    const bool hud = !shadow_only && root && root->renderable_HUD();
    const Fmatrix view_projection = hud ? hud_view_projection() : current_view_projection();
    Fmatrix mvp;
    mvp.mul(view_projection, world);
    Fvector center =
        model ? model->visibility().sphere.P : visual_center(level_.visual_node(static_cast<size_t>(visual_index)));
    Fvector world_center;
    world.transform_tiny(world_center, center);
    center = world_center;
    static_assert(sizeof(Fmatrix) == 16 * sizeof(float));
    float transform[16];
    std::memcpy(transform, &mvp, sizeof(transform));
    Fvector camera_position;
    if (camera_state_.has_camera_position())
        camera_position.set(camera_state_.camera_position()[0], camera_state_.camera_position()[1],
                            camera_state_.camera_position()[2]);
    else
        camera_position.set(Device.vCameraPosition);
    const float sort_distance = distance_squared(center, camera_position);
    const float scale = std::max({world.i.magnitude(), world.j.magnitude(), world.k.magnitude()});
    const float radius =
        (model ? model->visibility().sphere.R : level_.visual_node(static_cast<size_t>(visual_index))->bounds[9]) *
        scale;
    const float lod = lod_for_distance(radius, sort_distance);
    const VkExtent2D scene_extent = game_device_->window().frame().extent();
    const float detail_scale = screen_detail_scale(scene_extent, Device.fFOV);
    const float ssa = screen_space_area(radius, sort_distance);
    if (root && !hud && ssa <= screen_threshold(ssa_discard, 1.f, detail_scale))
        return;
    if (model)
    {
        const float *hemi_faces =
            root && root->renderable_ROS() ? root->renderable_ROS()->get_luminocity_hemi_cube() : nullptr;
        std::function<void(const VulkanModelVisual &, IKinematics *)> submit = [&](const VulkanModelVisual &node,
                                                                                   IKinematics *skeleton) {
            if (node.skeleton())
                skeleton = node.skeleton();
            if (!shadow_only && !hud && skeleton && node.skeleton_lod() &&
                ssa * .5f <= screen_threshold(skeleton_lod_threshold, 3.f, detail_scale))
            {
                submit(*node.skeleton_lod(), nullptr);
                return;
            }
            if (!shadow_only && node.has_lod() && lod < .33f)
            {
                std::array<std::array<float, 3>, 8> normals = node.lod_normals();
                for (auto &normal : normals)
                {
                    Fvector direction, transformed;
                    direction.set(normal[0], normal[1], normal[2]);
                    world.transform_dir(transformed, direction);
                    normal = {transformed.x, transformed.y, transformed.z};
                }
                Fvector center;
                world.transform_tiny(center, node.visibility().sphere.P);
                const uint8_t facet =
                    select_lod_facet(normals, {center.x - camera_position.x, center.y - camera_position.y,
                                               center.z - camera_position.z});
                game_device_->queue_model(node.lod_gpu(facet), skeleton, transform, hud, sort_distance, model_owner,
                                          lod, &world, hemi_faces, shadow_only, &node.visibility().sphere);
                return;
            }
            if (node.has_gpu())
                game_device_->queue_model(node.gpu(), skeleton, transform, hud, sort_distance, model_owner, lod, &world,
                                          hemi_faces, shadow_only, &node.visibility().sphere);
            for (const auto &child : node.children())
                submit(*child, skeleton);
            if (!shadow_only && !node.linked().empty())
            {
                R_ASSERT2(node.linked_valid(), "Vulkan model has level references from an unloaded level");
                for (uint32_t index : node.linked())
                    game_device_->queue_level_visual(index, transform, hud, sort_distance, model_owner, lod, &world);
            }
        };
        submit(*model, nullptr);
    }
    else if (!shadow_only)
        game_device_->queue_level_visual(static_cast<uint32_t>(visual_index), transform, hud, sort_distance, nullptr,
                                         lod);
}

IRender_ObjectSpecific *VulkanLevelRender::ros_create(IRenderable *parent)
{
    R_ASSERT2(parent, "Vulkan object lighting requires a renderable parent");
    auto *object = xr_new<VulkanObjectSpecific>(parent);
    object_specifics_.push_back(object);
    return object;
}

void VulkanLevelRender::create_wallmark(ModelGeometry &&geometry, const Fvector &center, IKinematics *skeleton,
                                        const Fmatrix *transform)
{
    if (!device_ || !game_device_)
        return;
    const float lifetime = std::clamp(wallmark_ttl, 1.f, 600.f);
    Fvector bounds_min{}, bounds_max{};
    bounds_min.set(FLT_MAX, FLT_MAX, FLT_MAX);
    bounds_max.set(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (const ModelVertex &vertex : geometry.vertices)
    {
        bounds_min.x = std::min(bounds_min.x, vertex.position[0]);
        bounds_min.y = std::min(bounds_min.y, vertex.position[1]);
        bounds_min.z = std::min(bounds_min.z, vertex.position[2]);
        bounds_max.x = std::max(bounds_max.x, vertex.position[0]);
        bounds_max.y = std::max(bounds_max.y, vertex.position[1]);
        bounds_max.z = std::max(bounds_max.z, vertex.position[2]);
    }
    const float dx = bounds_max.x - bounds_min.x;
    const float dy = bounds_max.y - bounds_min.y;
    const float dz = bounds_max.z - bounds_min.z;
    const float radius = .5f * std::sqrt(dx * dx + dy * dy + dz * dz);
    Fvector stored_center = center;
    u16 bone = UINT16_MAX;
    if (transform)
    {
        Fmatrix inverse;
        inverse.invert(*transform);
        inverse.transform_tiny(stored_center, center);
        if (skeleton && !geometry.vertices.empty())
        {
            bone = geometry.vertices.front().bones[0];
            if (bone < skeleton->LL_BoneCount())
            {
                Fmatrix inverse_bone;
                inverse_bone.invert(skeleton->LL_GetBoneInstance(bone).mRenderTransform);
                Fvector bind_center;
                inverse_bone.transform_tiny(bind_center, stored_center);
                stored_center = bind_center;
            }
        }
    }
    const std::string texture = geometry.texture;
    auto model = std::make_unique<GpuModel>();
    std::string error;
    if (!model->load_decoded(std::move(geometry), device_, queue_, pool_, memory_, upload_, *textures_, *pass_, error))
    {
        Msg("! [renderer-vulkan] wallmark: %s", error.c_str());
        return;
    }
    Wallmark mark{std::move(model), skeleton, transform, stored_center, bone, lifetime, lifetime, radius, texture};
    if (!skeleton)
    {
        for (Wallmark &existing : wallmarks_)
        {
            if (existing.skeleton || existing.texture != mark.texture || !existing.center.similar(center, .02f))
                continue;
            game_device_->retire(std::move(existing.model));
            existing = std::move(mark);
            return;
        }
    }
    wallmarks_.push_back(std::move(mark));
}

void VulkanLevelRender::add_StaticWallmark(const wm_shader &shader, const Fvector &point, float size,
                                           CDB::TRI *triangle, Fvector *vertices)
{
    if (!triangle || !vertices || triangle->suppress_wm)
        return;
    const auto *material = dynamic_cast<const VulkanUIShader *>(&*shader);
    R_ASSERT2(material, "Vulkan wallmark requires a Vulkan material");
    Fvector corners[3] = {vertices[triangle->verts[0]], vertices[triangle->verts[1]], vertices[triangle->verts[2]]};
    ModelGeometry geometry;
    if (wallmark_geometry(corners, point, size, material->texture_name(), geometry))
        create_wallmark(std::move(geometry), point);
}

void VulkanLevelRender::add_StaticWallmark(IWallMarkArray *array, const Fvector &point, float size, CDB::TRI *triangle,
                                           Fvector *vertices)
{
    auto *materials = dynamic_cast<VulkanWallMarkArray *>(array);
    R_ASSERT2(materials, "Vulkan wallmark requires a Vulkan material array");
    if (!triangle || !vertices || triangle->suppress_wm)
        return;
    const std::string *texture = materials->select_texture();
    if (!texture)
        return;
    Fvector corners[3] = {vertices[triangle->verts[0]], vertices[triangle->verts[1]], vertices[triangle->verts[2]]};
    ModelGeometry geometry;
    if (wallmark_geometry(corners, point, size, *texture, geometry))
        create_wallmark(std::move(geometry), point);
}

void VulkanLevelRender::add_SkeletonWallmark(const Fmatrix *transform, IKinematics *skeleton, IWallMarkArray *array,
                                             const Fvector &start, const Fvector &direction, float size)
{
    auto *materials = dynamic_cast<VulkanWallMarkArray *>(array);
    R_ASSERT2(materials && transform && skeleton,
              "Vulkan skeletal wallmark requires a live skeleton and material array");
    const std::string *texture = materials->select_texture();
    if (!texture || !(size > EPS_L))
        return;
    skeleton->CalculateBones();
    IKinematics::pick_result picked{};
    u16 bone = BI_NONE;
    float nearest = size * 3.f;
    for (u16 id = 0; id < skeleton->LL_BoneCount(); ++id)
    {
        IKinematics::pick_result hit{};
        if (skeleton->PickBone(*transform, hit, nearest, start, direction, id))
        {
            nearest = hit.dist;
            picked = hit;
            bone = id;
        }
    }
    if (bone == BI_NONE)
        return;
    Fvector contact;
    contact.mad(start, direction, nearest);
    ModelGeometry geometry;
    if (!wallmark_geometry(picked.tri, contact, size, *texture, geometry))
        return;
    // Store the decal in bind space; the model's ordinary skinning path then
    // carries it with the hit bone as its pose changes.
    Fmatrix inverse_world, inverse_bone;
    inverse_world.invert(*transform);
    inverse_bone.invert(skeleton->LL_GetBoneInstance(bone).mRenderTransform);
    for (auto &vertex : geometry.vertices)
    {
        Fvector world, local, bind;
        world.set(vertex.position[0], vertex.position[1], vertex.position[2]);
        inverse_world.transform_tiny(local, world);
        inverse_bone.transform_tiny(bind, local);
        vertex.position[0] = bind.x;
        vertex.position[1] = bind.y;
        vertex.position[2] = bind.z;
        vertex.bones[0] = bone;
    }
    geometry.type = 4;
    create_wallmark(std::move(geometry), contact, skeleton, transform);
}

void VulkanLevelRender::clear_static_wallmarks()
{
    if (game_device_)
        R_ASSERT2(game_device_->wait_idle(), "Vulkan wallmark clear requires idle GPU frames");
    wallmarks_.erase(
        std::remove_if(wallmarks_.begin(), wallmarks_.end(), [](const Wallmark &mark) { return !mark.skeleton; }),
        wallmarks_.end());
}

void VulkanLevelRender::ros_destroy(IRender_ObjectSpecific *&object)
{
    if (!object)
        return;
    auto *vulkan_object = dynamic_cast<VulkanObjectSpecific *>(object);
    R_ASSERT2(vulkan_object, "Vulkan renderer received foreign object-specific data");
    if (!vulkan_object)
        return;
    object_specifics_.erase(std::remove(object_specifics_.begin(), object_specifics_.end(), vulkan_object),
                            object_specifics_.end());
    xr_delete(vulkan_object);
    object = nullptr;
}

IRender_Light *VulkanLevelRender::light_create()
{
    auto *light = xr_new<VulkanLight>();
    lights_.push_back(light);
    return light;
}

void VulkanLevelRender::light_destroy(IRender_Light *light)
{
    // Called by IRender_Light's base destructor after the derived subobject is
    // gone: compare identities only and do not dereference the pointer.
    lights_.erase(std::remove_if(lights_.begin(), lights_.end(),
                                 [light](const VulkanLight *candidate) {
                                     return static_cast<const IRender_Light *>(candidate) == light;
                                 }),
                  lights_.end());
}

IRender_Glow *VulkanLevelRender::glow_create()
{
    auto *glow = xr_new<VulkanGlow>();
    glows_.push_back(glow);
    return glow;
}

void VulkanLevelRender::glow_destroy(IRender_Glow *glow)
{
    // Like lights, glows are xr_resources and call back from their base
    // destructor. Avoid accessing the object during that callback.
    glow_visibility_.erase(static_cast<VulkanGlow *>(glow));
    glows_.erase(std::remove_if(glows_.begin(), glows_.end(),
                                [glow](const VulkanGlow *candidate) {
                                    return static_cast<const IRender_Glow *>(candidate) == glow;
                                }),
                 glows_.end());
}
} // namespace xray::render::vulkan
