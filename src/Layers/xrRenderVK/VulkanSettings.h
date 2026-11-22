#pragma once

#include <cstdint>
#include "RenderPolicy.h"
#include <string>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace xray::render::vulkan
{
// Values exposed through the game's renderer console. The Vulkan renderer
// reads these every frame, so the UI and saved user config share one state.
#if defined(XR_PLATFORM_ANDROID)
inline constexpr unsigned DefaultVulkanQuality = 1;
#else
inline constexpr unsigned DefaultVulkanQuality = 2;
#endif
inline constexpr int PresetAnisotropy[]{2, 4, 8, 16, 16};
inline constexpr uint32_t PresetSunShaftSamples[]{8, 12, 16, 24, 32};
inline constexpr uint32_t PresetAODirections[]{4, 8, 8, 12, 16};
inline constexpr float PresetDetailSpacing[]{1.f, .75f, .5f, .375f, .25f};
inline uint32_t quality_preset = DefaultVulkanQuality;
inline int detail_radius = RenderQualities[DefaultVulkanQuality].detail_distance;
inline float detail_spacing = PresetDetailSpacing[DefaultVulkanQuality];
inline int tessellation_enabled = 0;
inline int legacy_level_lights = 0;
inline int detail_update_cells = RenderQualities[DefaultVulkanQuality].detail_updates;
inline int texture_anisotropy = PresetAnisotropy[DefaultVulkanQuality];
inline float texture_mip_bias = 0.f;
inline float detail_prepare_ms = 2.f;
inline float ssa_discard = 3.5f;
inline float geometry_lod_scale = .75f;
inline float skeleton_lod_threshold = 64.f;
inline int use_light_grid = 1;
inline int steep_parallax = 0;
inline int dynamic_wet_surfaces = 1;
inline float dynamic_wet_surfaces_near = 5.f;
inline float dynamic_wet_surfaces_far = 20.f;
inline int dynamic_wet_surfaces_resolution = 256;
inline int point_shadow_budget = RenderQualities[DefaultVulkanQuality].point_shadows;
inline int spot_shadow_budget = RenderQualities[DefaultVulkanQuality].spot_shadows;
inline int ros_ray_budget = 128;
inline float hemi_sky_scale = .08f;
inline float hemi_light_scale = .2f, hemi_light_flow = .1f, hemi_smooth = 1.f;
inline float ambient_scale = 1.f;
inline float hemisphere_scale = 1.f;
inline float sun_scale = 1.f;
inline int sun_enabled = 1;
inline int tonemap_enabled = 1;
inline float tonemap_middlegray = 1.f;
inline float tonemap_amount = .7f;
inline float tonemap_adaptation_speed = 1.5f;
inline float tonemap_low_luminance = .0001f;
inline float wallmark_ttl = 50.f;
inline float bloom_intensity = .14f;
inline float hdr_paper_white_nits = 203.f;
inline float hdr_peak_nits = 1000.f;
inline float hdr_black_nits = 0.f;
inline int hdr_calibration_pattern = 0;
inline float bloom_threshold = .72f;
inline float bloom_mip = 4.f;
inline float bloom_knee = .12f;
inline int bloom_fast = 0;
inline constexpr float default_legacy_bloom_threshold = .00001f;
inline float legacy_bloom_threshold = default_legacy_bloom_threshold;
inline float legacy_bloom_kernel_g = 3.f;
inline float legacy_bloom_kernel_b = .7f;
inline float legacy_bloom_kernel_scale = .7f;
inline float legacy_bloom_speed = 100.f;
inline float ao_strength = 0.75f;
inline float dof_strength = 0.f;
inline float dof_focus = 2.5f;
inline float motion_blur_strength = 0.f;
inline float sun_shafts_strength = 1.f;
inline int volumetric_lights_enabled = 1;
inline float rtgi_strength = 1.f;
inline int rtgi_available = 0;
// Requested G-buffer MSAA sample count (0 disables MSAA). This setting is
// latched when the Vulkan game device is created.
inline uint32_t gbuffer_msaa_samples = 1;
// Independent of desktop MSAA: off, FXAA, or temporal color resolve.
inline uint32_t antialiasing = 1;
// Scene reconstruction at presentation resolution: bilinear, cubic or temporal.
inline uint32_t spatial_upscaler = 0;
inline uint32_t render_scale_percent = 0;
// Optional FSR 3 frame interpolation (requires the FSR upscaler and dedicated Vulkan queues).
inline uint32_t frame_generation = 0;
// Read-only menu capability: the build and selected Vulkan device expose the required queue roles.
inline int frame_generation_available = 0;

inline void set_vulkan_quality(unsigned preset)
{
    preset = std::min<unsigned>(preset, std::size(PresetAnisotropy) - 1);
    quality_preset = preset;
    const auto& quality = render_quality(quality_preset);
    detail_radius = quality.detail_distance;
    detail_spacing = PresetDetailSpacing[quality_preset];
    detail_update_cells = quality.detail_updates;
    point_shadow_budget = quality.point_shadows;
    spot_shadow_budget = quality.spot_shadows;
    texture_anisotropy = PresetAnisotropy[quality_preset];
}

// Interpret explicit command-line overrides as defaults. The saved user
// config and subsequent changes in the game own these settings.
inline void apply_vulkan_launch_settings(const char *parameters)
{
    if (!parameters)
        return;
    std::istringstream input(parameters);
    std::string token, value;
    while (input >> token)
    {
        if (token == "-android-mobile-preset" && input >> value)
        {
            const char *names[]{"Minimum", "Low", "Default", "High", "Extreme"};
            for (uint32_t i = 0; i < 5; ++i)
                if (value == names[i])
                {
                    set_vulkan_quality(i);
                    break;
                }
        }
        else if ((token == "-vk-detail-distance" || token == "-vk-anisotropy") && input >> value)
        {
            char *end = nullptr;
            const long distance = std::strtol(value.c_str(), &end, 10);
            if (end != value.c_str() && !*end)
            {
                if (token == "-vk-detail-distance")
                    detail_radius = int(std::clamp(distance, 0L, 80L));
                else
                    texture_anisotropy = int(std::clamp(distance, 1L, 16L));
            }
        }
        else if (token == "-vk-msaa" && input >> value)
        {
            char *end = nullptr;
            const long samples = std::strtol(value.c_str(), &end, 10);
            if (end != value.c_str() && !*end && (samples == 0 || samples == 1 || samples == 2 || samples == 4))
                gbuffer_msaa_samples = samples == 0 ? 1u : static_cast<uint32_t>(samples);
        }
        else if (token == "-vk-ambient-scale" || token == "-vk-hemi-scale" || token == "-vk-sun-scale")
        {
            if (!(input >> value))
                break;
            char *end = nullptr;
            const float scale = std::strtof(value.c_str(), &end);
            if (end != value.c_str() && !*end && std::isfinite(scale))
                (token == "-vk-ambient-scale" ? ambient_scale
                 : token == "-vk-hemi-scale"  ? hemisphere_scale
                                              : sun_scale) = std::clamp(scale, 0.f, 3.f);
        }
        else if (token == "-vk-no-tonemap")
            tonemap_enabled = 0;
    }
}
} // namespace xray::render::vulkan
