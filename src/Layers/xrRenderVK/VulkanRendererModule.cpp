#include "xrEngine/stdafx.h"
#include "Include/xrRender/xrRender.h"
#include "VulkanProbe.h"
#include "VulkanLevelRender.h"
#include "VulkanGameDevice.h"
#include "VulkanRenderFactory.h"
#include "VulkanDrawUtils.h"
#include "VulkanDebugRender.h"
#include "VulkanSettings.h"
#include "EngineParticleSource.h"
#include "ParticleCatalog.h"
#include "xrEngine/XR_IOConsole.h"
#include "xrEngine/xr_ioc_cmd.h"
#include "xrEngine/editor_base.h"
#include <SDL.h>
#include <cstring>
#include <algorithm>
#include <cstdio>
#include <cmath>
#include <charconv>

namespace xray::render::vulkan
{
namespace
{
const xr_token quality_tokens[]{{"Minimum", 0}, {"Low", 1}, {"Default", 2}, {"High", 3}, {"Extreme", 4}, {nullptr, 0}};
const xr_token aa_tokens[]{{"off", 0}, {"FXAA", 1}, {"TAA", 2}, {nullptr, 0}};
const xr_token msaa_tokens[]{{"off", 0}, {"2x", 2}, {"4x", 4}, {nullptr, 0}};
const xr_token upscale_tokens[]{{"Bilinear", 0}, {"Cubic", 1}, {"Temporal", 2}, {"FSR 3.1", 3}, {nullptr, 0}};
const xr_token render_scale_tokens[]{{"Auto", 0}, {"50%", 50}, {"67%", 67}, {"75%", 75},
    {"83%", 83}, {"100%", 100}, {nullptr, 0}};
const xr_token frame_generation_tokens[]{{"off", 0}, {"on", 1}, {nullptr, 0}};

class VulkanPreset final : public CCC_Token
{
  public:
    explicit VulkanPreset(VulkanLevelRender& renderer)
        : CCC_Token("_preset", &quality_preset, quality_tokens), renderer_(renderer)
    {
    }
    void Execute(pcstr args) override
    {
        const unsigned previous_preset = quality_preset;
        const int previous_anisotropy = texture_anisotropy;
        CCC_Token::Execute(args);
        if (get_token_id(quality_tokens, args) >= 0)
        {
            set_vulkan_quality(quality_preset);
            if (quality_preset != previous_preset || texture_anisotropy != previous_anisotropy)
                renderer_.request_sampler_settings_reset();
            const auto& quality = render_quality(quality_preset);
            Msg("[renderer-vulkan] quality preset=%s texture-limit=%u anisotropy=%d detail-distance=%d "
                "detail-spacing=%.3f detail-updates=%d point-shadows=%d spot-shadows=%d sun-span=%.0f "
                "ao-directions=%u sun-shaft-samples=%u",
                get_token_name(quality_tokens, quality_preset), quality.texture_limit, texture_anisotropy,
                detail_radius, detail_spacing, detail_update_cells, point_shadow_budget, spot_shadow_budget,
                quality.sun_span, PresetAODirections[quality_preset], PresetSunShaftSamples[quality_preset]);
        }
    }
  private:
    VulkanLevelRender& renderer_;
};

class VulkanSamplerAnisotropy final : public CCC_Integer
{
  public:
    explicit VulkanSamplerAnisotropy(VulkanLevelRender& renderer)
        : CCC_Integer("r__tf_aniso", &texture_anisotropy, 1, 16), renderer_(renderer)
    {
    }
    void Execute(pcstr args) override
    {
        const int previous = texture_anisotropy;
        CCC_Integer::Execute(args);
        if (texture_anisotropy != previous)
            renderer_.request_sampler_settings_reset();
    }
  private:
    VulkanLevelRender& renderer_;
};

class VulkanSamplerMipBias final : public CCC_Float
{
  public:
    VulkanSamplerMipBias(pcstr name, VulkanLevelRender& renderer)
        : CCC_Float(name, &texture_mip_bias, -3.f, 3.f), renderer_(renderer)
    {
    }
    void Execute(pcstr args) override
    {
        const float previous = texture_mip_bias;
        CCC_Float::Execute(args);
        if (texture_mip_bias != previous)
        {
            renderer_.request_sampler_settings_reset();
        }
    }
  private:
    VulkanLevelRender& renderer_;
};

class VulkanLegacyBloomKernel final : public CCC_Float
{
public:
    enum class Parameter { KernelG, KernelB, KernelScale };

    VulkanLegacyBloomKernel(pcstr name, float* setting, float minimum, float maximum, Parameter parameter)
        : CCC_Float(name, setting, minimum, maximum), parameter_(parameter)
    {
    }

    void Execute(pcstr args) override
    {
        CCC_Float::Execute(args);
        switch (parameter_)
        {
        case Parameter::KernelG:
        case Parameter::KernelB:
            bloom_mip = std::clamp(4.f * (legacy_bloom_kernel_g / 3.f) *
                (legacy_bloom_kernel_b / .7f), 1.f, 12.f);
            break;
        case Parameter::KernelScale:
            bloom_intensity = std::clamp(.2f * legacy_bloom_kernel_scale, 0.f, 2.f);
            break;
        }
    }

private:
    Parameter parameter_;
};

class VulkanSwitch final : public CCC_Integer
{
public:
    VulkanSwitch(pcstr name, int* state) : CCC_Integer(name, state, 0, 1) {}

    void Execute(pcstr args) override
    {
        if (!xr_stricmp(args, "on") || !xr_strcmp(args, "1"))
            *value = 1;
        else if (!xr_stricmp(args, "off") || !xr_strcmp(args, "0"))
            *value = 0;
        else
            InvalidSyntax();
    }
};

class VulkanReadonlyState final : public CCC_Integer
{
public:
    VulkanReadonlyState(pcstr name, int* state) : CCC_Integer(name, state, 0, 1) {}
    void Execute(pcstr) override { InvalidSyntax(); }
    void Save(IWriter*) override {}
};

class VulkanBenchmarkCommand final : public IConsole_Command
{
public:
    explicit VulkanBenchmarkCommand(VulkanLevelRender& renderer)
        : IConsole_Command("r_vk_benchmark"), renderer_(renderer) {}
    void Execute(pcstr args) override
    {
        if (!xr_strcmp(args, "cancel"))
        {
            renderer_.interrupt_benchmark("cancelled by user");
            return;
        }
        uint32_t frames = 0;
        if (!VulkanFrameBenchmark::parse_frame_count(args, frames))
        {
            InvalidSyntax();
            return;
        }
        if (!renderer_.start_benchmark(frames))
            Msg("! [renderer-vulkan] benchmark not armed: capture already active or device unavailable");
    }
    void Info(TInfo& info) override { xr_strcpy(info, "<1..36000 game frames> | cancel"); }
private:
    VulkanLevelRender& renderer_;
};

bool load_particle_document(std::vector<uint8_t>& library, std::string& error)
{
    string_path path;
    FS.update_path(path, _game_data_, "particles.xr");
    IReader* source = FS.r_open(path);
    if (!source)
    {
        error = "could not open particles.xr";
        return false;
    }
    const bool loaded = editable_particle_library_bytes(*source, library, error);
    FS.r_close(source);
    return loaded;
}

bool export_particle_document(const std::vector<uint8_t>& library, std::string& message)
{
    string_path path;
    FS.update_path(path, "$app_data_root$", "particles_edited.xr");
    convert_path_separators(path);
    FILE* file = std::fopen(path, "wbx");
    if (!file)
    {
        message = std::string("could not create export (existing files are preserved): ") + path;
        return false;
    }
    const bool wrote = std::fwrite(library.data(), 1, library.size(), file) == library.size();
    const bool closed = std::fclose(file) == 0;
    if (!wrote || !closed)
    {
        std::remove(path);
        message = std::string("incomplete particle export: ") + path;
        return false;
    }
    message = std::string("Exported particle library to ") + path;
    return true;
}

class VulkanParticleEditor final : public xray::editor::ide_tool
{
public:
    pcstr tool_name() const override { return "Particle Editor"; }

    void on_tool_frame() override
    {
        if (ImGui::Begin(tool_name(), &get_open_state(), get_default_window_flags()))
        {
            ImGui::BeginDisabled(dirty_);
            if (ImGui::Button("Open particles.xr"))
                open();
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(!dirty_);
            if (ImGui::Button("Export copy") && export_particle_document(library_, message_))
                dirty_ = false;
            ImGui::EndDisabled();
            if (!message_.empty())
                ImGui::TextWrapped("%s", message_.c_str());

            if (!catalog_.effects.empty())
            {
                filter_.Draw("Filter effects");
                if (ImGui::BeginListBox("##effects", ImVec2(-1.f, 180.f)))
                {
                    for (size_t i = 0; i < catalog_.effects.size(); ++i)
                    {
                        const auto& effect = catalog_.effects[i];
                        if (!filter_.PassFilter(effect.name.c_str()))
                            continue;
                        ImGui::PushID(static_cast<int>(i));
                        if (ImGui::Selectable(effect.name.c_str(), selected_ == i))
                        {
                            selected_ = i;
                            limit_ = static_cast<int>(effect.max_particles);
                        }
                        ImGui::PopID();
                    }
                    ImGui::EndListBox();
                }
                if (selected_ < catalog_.effects.size())
                {
                    auto& effect = catalog_.effects[selected_];
                    ImGui::Text("Shader: %s", effect.shader.c_str());
                    ImGui::Text("Texture: %s", effect.texture.c_str());
                    ImGui::InputInt("Particle limit", &limit_);
                    ImGui::BeginDisabled(limit_ < 1 || limit_ > 65535 || uint32_t(limit_) == effect.max_particles);
                    if (ImGui::Button("Apply to draft") &&
                        edit_particle_effect_limit(library_, effect.name, static_cast<uint32_t>(limit_), message_))
                    {
                        effect.max_particles = static_cast<uint32_t>(limit_);
                        dirty_ = true;
                    }
                    ImGui::EndDisabled();
                }
            }
        }
        ImGui::End();
    }

private:
    void open()
    {
        std::vector<uint8_t> bytes;
        ParticleCatalog catalog;
        if (!load_particle_document(bytes, message_) ||
            !parse_particle_catalog({bytes.data(), bytes.size()}, catalog, message_))
            return;
        library_ = std::move(bytes);
        catalog_ = std::move(catalog);
        selected_ = SIZE_MAX;
    }

    std::vector<uint8_t> library_;
    ParticleCatalog catalog_;
    std::string message_;
    ImGuiTextFilter filter_;
    size_t selected_{SIZE_MAX};
    int limit_{};
    bool dirty_{};
};

class VulkanParticleLimitCommand final : public IConsole_Command
{
public:
    VulkanParticleLimitCommand() : IConsole_Command("r_vk_particle_export_limit") {}

    void Execute(pcstr args) override
    {
        const std::string input(args);
        const auto end = input.find_last_not_of(" \t");
        const auto separator = end == std::string::npos ? end : input.find_last_of(" \t", end);
        if (separator == std::string::npos)
        {
            InvalidSyntax();
            return;
        }
        const auto name_end = separator ? input.find_last_not_of(" \t", separator - 1) : std::string::npos;
        uint32_t limit{};
        const auto parsed = std::from_chars(input.data() + separator + 1, input.data() + end + 1, limit);
        if (name_end == std::string::npos || parsed.ec != std::errc{} || parsed.ptr != input.data() + end + 1)
        {
            InvalidSyntax();
            return;
        }

        std::vector<uint8_t> library;
        std::string error;
        if (!load_particle_document(library, error) ||
            !edit_particle_effect_limit(library, input.substr(0, name_end + 1), limit, error) ||
            !export_particle_document(library, error))
            Msg("! [renderer-vulkan] particle export failed: %s", error.c_str());
        else
            Msg("[renderer-vulkan] %s", error.c_str());
    }

    void Info(TInfo& info) override { xr_strcpy(info, "<effect name> <1..65535> (exports particles_edited.xr)"); }
};

void register_vulkan_options(VulkanLevelRender& renderer)
{
    static VulkanBenchmarkCommand benchmark(renderer);
    static VulkanParticleLimitCommand particle_limit;
    Console->AddCommand(&benchmark);
    Console->AddCommand(&particle_limit);
    static VulkanPreset preset(renderer);
    static CCC_Token aa("r_vk_aa", &antialiasing, aa_tokens);
    static CCC_Token msaa("r_vk_msaa", &gbuffer_msaa_samples, msaa_tokens);
    static CCC_Token upscaler("r_vk_upscaler", &spatial_upscaler, upscale_tokens);
    static CCC_Token render_scale("r_vk_render_scale", &render_scale_percent, render_scale_tokens);
    static CCC_Token frame_generation_option("r_vk_frame_generation", &frame_generation,
        frame_generation_tokens);
    Console->AddCommand(&aa);
    Console->AddCommand(&msaa);
    Console->AddCommand(&upscaler);
    Console->AddCommand(&render_scale);
    Console->AddCommand(&frame_generation_option);
    static CCC_Float detail_time("r_vk_detail_prepare_ms", &detail_prepare_ms, 0.1f, 20.f);
    Console->AddCommand(&detail_time);
    static CCC_Integer light_grid("r_vk_light_grid", &use_light_grid, 0, 1);
    Console->AddCommand(&light_grid);
    static VulkanSwitch steep_parallax_switch("r_vk_steep_parallax", &steep_parallax);
    Console->AddCommand(&steep_parallax_switch);
    static VulkanSwitch wet_surfaces_switch("r3_dynamic_wet_surfaces", &dynamic_wet_surfaces);
    static CCC_Float wet_surfaces_near("r3_dynamic_wet_surfaces_near", &dynamic_wet_surfaces_near, 5.f, 70.f);
    static CCC_Float wet_surfaces_far("r3_dynamic_wet_surfaces_far", &dynamic_wet_surfaces_far, 20.f, 100.f);
    static CCC_Integer wet_surfaces_resolution("r3_dynamic_wet_surfaces_sm_res",
        &dynamic_wet_surfaces_resolution, 64, 2048);
    Console->AddCommand(&wet_surfaces_switch);
    Console->AddCommand(&wet_surfaces_near);
    Console->AddCommand(&wet_surfaces_far);
    Console->AddCommand(&wet_surfaces_resolution);
    static CCC_Integer detail_updates("r_vk_detail_update_cells", &detail_update_cells, 1, 256);
    static CCC_Integer point_shadows("r_vk_point_shadows", &point_shadow_budget, 0, 2);
    static CCC_Integer spot_shadows("r_vk_spot_shadows", &spot_shadow_budget, 0, 4);
    static CCC_Integer ros_rays("r_vk_ros_ray_budget", &ros_ray_budget, 0, 2048);
    static CCC_Float sky_scale("r2_dhemi_sky_scale", &hemi_sky_scale, 0.f, 100.f);
    static CCC_Float light_scale("r2_dhemi_light_scale", &hemi_light_scale, 0.f, 100.f);
    static CCC_Float light_flow("r2_dhemi_light_flow", &hemi_light_flow, 0.f, 1.f);
    static CCC_Float smooth("r2_dhemi_smooth", &hemi_smooth, 0.f, 10.f);
    static CCC_Float detail_density("r__detail_density", &detail_spacing, .25f, 1.f);
    static VulkanSwitch tessellation("r4_enable_tessellation", &tessellation_enabled);
    static VulkanSwitch legacy_lights("r2_allow_r1_lights", &legacy_level_lights);
    static VulkanSamplerAnisotropy anisotropy(renderer);
    static VulkanSamplerMipBias mip_bias("r_vk_texture_mip_bias", renderer);
    static VulkanSamplerMipBias r1_mip_bias("r1_tf_mipbias", renderer);
    static VulkanSamplerMipBias r2_mip_bias("r2_tf_mipbias", renderer);
    static CCC_Integer details("r__detail_radius", &detail_radius, 0, 80);
    static CCC_Float discard_small_objects("r__ssaDISCARD", &ssa_discard, 0.f, 20.f);
    static CCC_Float geometry_lod("r__geometry_lod", &geometry_lod_scale, .1f, 2.f);
    static CCC_Float skeleton_lod("r2_ssa_lod_a", &skeleton_lod_threshold, 16.f, 96.f);
    static CCC_Float ambient("r2_sun_lumscale_amb", &ambient_scale, 0.f, 3.f);
    static CCC_Float hemisphere("r2_sun_lumscale_hemi", &hemisphere_scale, 0.f, 3.f);
    static CCC_Float sun("r2_sun_lumscale", &sun_scale, 0.f, 3.f);
    static VulkanSwitch sun_switch("r2_sun", &sun_enabled);
    static VulkanSwitch volumetric_switch("r2_volumetric_lights", &volumetric_lights_enabled);
    static VulkanSwitch tone_switch("r2_tonemap", &tonemap_enabled);
    static CCC_Float tone_gray("r2_tonemap_middlegray", &tonemap_middlegray, 0.f, 2.f);
    static CCC_Float tone_amount("r2_tonemap_amount", &tonemap_amount, 0.f, 1.f);
    static CCC_Float tone_adaptation("r_vk_tonemap_adaptation", &tonemap_adaptation_speed, 0.f, 20.f);
    static CCC_Float legacy_tone_adaptation("r2_tonemap_adaptation", &tonemap_adaptation_speed, .01f, 10.f);
    static CCC_Float legacy_tone_lowlum("r2_tonemap_lowlum", &tonemap_low_luminance, .0001f, 1.f);
    static CCC_Float wallmark_lifetime("r__wallmark_ttl", &wallmark_ttl, 1.f, 600.f);
    static CCC_Float bloom_strength("r_vk_bloom_intensity", &bloom_intensity, 0.f, 2.f);
    static CCC_Float hdr_paper_white("r_vk_hdr_paper_white", &hdr_paper_white_nits, 80.f, 1000.f);
    static CCC_Float hdr_peak("r_vk_hdr_peak", &hdr_peak_nits, 80.f, 10000.f);
    static CCC_Float hdr_black("r_vk_hdr_black", &hdr_black_nits, 0.f, 10.f);
    static CCC_Integer hdr_pattern("r_vk_hdr_calibration_pattern", &hdr_calibration_pattern, 0, 1);
    static CCC_Float bloom_cutoff("r_vk_bloom_threshold", &bloom_threshold, 0.f, 4.f);
    static CCC_Float bloom_radius("r_vk_bloom_mip", &bloom_mip, 1.f, 12.f);
    static CCC_Float bloom_softness("r_vk_bloom_knee", &bloom_knee, 0.f, 2.f);
    static CCC_Float legacy_bloom_cutoff("r2_ls_bloom_threshold", &legacy_bloom_threshold, 0.f, 1.f);
    static VulkanLegacyBloomKernel legacy_bloom_scale("r2_ls_bloom_kernel_scale", &legacy_bloom_kernel_scale,
        .5f, 2.f, VulkanLegacyBloomKernel::Parameter::KernelScale);
    static VulkanLegacyBloomKernel legacy_bloom_g("r2_ls_bloom_kernel_g", &legacy_bloom_kernel_g,
        1.f, 7.f, VulkanLegacyBloomKernel::Parameter::KernelG);
    static VulkanLegacyBloomKernel legacy_bloom_b("r2_ls_bloom_kernel_b", &legacy_bloom_kernel_b,
        .01f, 1.f, VulkanLegacyBloomKernel::Parameter::KernelB);
    static CCC_Float legacy_bloom_speed_command("r2_ls_bloom_speed", &legacy_bloom_speed, 0.f, 100.f);
    static VulkanSwitch legacy_bloom_fast("r2_ls_bloom_fast", &bloom_fast);
    static CCC_Float ao("r_vk_ao_strength", &ao_strength, 0.f, 2.f);
    static CCC_Float dof("r_vk_dof_strength", &dof_strength, 0.f, 1.f);
    static CCC_Float focus("r_vk_dof_focus", &dof_focus, .2f, 100.f);
    static CCC_Float motion_blur("r_vk_motion_blur", &motion_blur_strength, 0.f, 1.f);
    static CCC_Float sun_shafts("r_vk_sun_shafts_strength", &sun_shafts_strength, 0.f, 1.f);
    static CCC_Float rtgi("r_vk_rtgi_strength", &rtgi_strength, 0.f, 2.f);
    static VulkanReadonlyState rtgi_support("r_vk_rtgi_available", &rtgi_available);
    static VulkanReadonlyState frame_generation_support("r_vk_frame_generation_available",
        &frame_generation_available);
    apply_vulkan_launch_settings(Core.Params);
    IConsole_Command *options[]{&point_shadows, &spot_shadows, &ros_rays, &sky_scale,
                                &light_scale,   &light_flow,   &smooth};
    for (auto *option : options)
        Console->AddCommand(option);
    Console->AddCommand(&preset);
    Console->AddCommand(&ao);
    Console->AddCommand(&dof);
    Console->AddCommand(&motion_blur);
    Console->AddCommand(&sun_shafts);
    Console->AddCommand(&rtgi);
    Console->AddCommand(&rtgi_support);
    Console->AddCommand(&frame_generation_support);
    Console->AddCommand(&focus);
    Console->AddCommand(&details);
    Console->AddCommand(&discard_small_objects);
    Console->AddCommand(&geometry_lod);
    Console->AddCommand(&skeleton_lod);
    Console->AddCommand(&detail_density);
    Console->AddCommand(&tessellation);
    Console->AddCommand(&legacy_lights);
    Console->AddCommand(&anisotropy);
    Console->AddCommand(&mip_bias);
    Console->AddCommand(&r1_mip_bias);
    Console->AddCommand(&r2_mip_bias);
    Console->AddCommand(&detail_updates);
    Console->AddCommand(&ambient);
    Console->AddCommand(&hemisphere);
    Console->AddCommand(&sun);
    Console->AddCommand(&sun_switch);
    Console->AddCommand(&volumetric_switch);
    Console->AddCommand(&tone_switch);
    Console->AddCommand(&tone_gray);
    Console->AddCommand(&tone_amount);
    Console->AddCommand(&tone_adaptation);
    Console->AddCommand(&legacy_tone_adaptation);
    Console->AddCommand(&legacy_tone_lowlum);
    Console->AddCommand(&wallmark_lifetime);
    Console->AddCommand(&bloom_strength);
    Console->AddCommand(&hdr_paper_white);
    Console->AddCommand(&hdr_peak);
    Console->AddCommand(&hdr_black);
    Console->AddCommand(&hdr_pattern);
    Console->AddCommand(&bloom_cutoff);
    Console->AddCommand(&bloom_radius);
    Console->AddCommand(&bloom_softness);
    Console->AddCommand(&legacy_bloom_cutoff);
    Console->AddCommand(&legacy_bloom_scale);
    Console->AddCommand(&legacy_bloom_g);
    Console->AddCommand(&legacy_bloom_b);
    Console->AddCommand(&legacy_bloom_speed_command);
    Console->AddCommand(&legacy_bloom_fast);
}

class VulkanRendererModule final : public RendererModule
{
    xr_vector<std::pair<pcstr, int>> modes;
    bool probe_attempted{};
    bool loader_available{};
    std::string probe_error;
    VulkanGameDevice device;
    VulkanLevelRender renderer{device};
    VulkanRenderFactory factory{device};
    VulkanDrawUtils draw_utils{device};
    std::unique_ptr<VulkanParticleEditor> particle_editor;
#ifdef DEBUG
    VulkanDebugRender debug_render{device};
#endif

  public:
    const xr_vector<std::pair<pcstr, int>> &ObtainSupportedModes() override
    {
        if (!probe_attempted)
        {
            if (std::strstr(Core.Params, "-vk_validation"))
                SDL_setenv("XRAY_VK_VALIDATION", "1", 1);
            probe_attempted = true;
            Msg("[renderer-vulkan] probe.begin validation=%s", SDL_getenv("XRAY_VK_VALIDATION") ? "requested" : "off");
            loader_available = probe_vulkan_loader(probe_error);
            if (loader_available)
            {
                modes.emplace_back("renderer_vulkan", 7);
                Msg("[renderer-vulkan] probe.ready mode=renderer_vulkan");
            }
            else
            {
                Msg("! [renderer-vulkan] unavailable: %s", probe_error.c_str());
            }
        }
        return modes;
    }

    bool CheckGameRequirements() override
    {
        if (!probe_attempted)
            ObtainSupportedModes();
        if (!loader_available)
            return false;
        Msg("[renderer-vulkan] requirements.begin shader-root=$game_shaders$");
        // The Vulkan game path owns its model/material resources and does not
        // instantiate the GLES CResourceManager. Check the VFS assets before
        // selecting a Vulkan SDL window; an incomplete install may use GLES
        // in auto mode, while an explicit request will report the failure.
        constexpr pcstr required[]{"vk\\level_opaque.vs",        "vk\\level_opaque.ps",        "vk\\object_opaque.vs",
                                   "vk\\object_opaque.ps",       "vk\\level_cutout.vs",        "vk\\level_cutout.ps",
                                   "vk\\object_cutout.vs",       "vk\\object_cutout.ps",       "vk\\object_blended.vs",
                                   "vk\\object_blended.ps",      "vk\\skinned_blended.ps",     "vk\\hud_blended.ps",
                                   "vk\\object_double_sided.vs", "vk\\object_double_sided.ps", "vk\\tree_opaque.vs",
                                   "vk\\progressive_opaque.vs",  "vk\\skinned_1.vs",           "vk\\skinned_2.vs",
                                   "vk\\skinned_3.vs",           "vk\\skinned_4.vs",           "vk\\hud_skinned_1.vs",
                                   "vk\\hud_skinned_2.vs",       "vk\\hud_skinned_3.vs",       "vk\\hud_skinned_4.vs"};
        for (pcstr name : required)
        {
            string_path path;
            if (!FS.exist(path, "$game_shaders$", name, ".spv"))
            {
                Msg("! [renderer-vulkan] missing gameplay shader: %s.spv", name);
                return false;
            }
            IReader *reader = FS.r_open(path);
            const size_t length = xr_strlen(name);
            const uint32_t stage = length >= 3 && name[length - 2] == 'v' ? 0u : 4u;
            const bool valid = reader && has_spirv_entry(reader->pointer(), reader->length(), stage, "main");
            if (reader)
                FS.r_close(reader);
            if (!valid)
            {
                Msg("! [renderer-vulkan] invalid gameplay shader stage/entry: %s.spv", name);
                return false;
            }
        }
        Msg("[renderer-vulkan] requirements.ready shader-count=%zu", sizeof(required) / sizeof(*required));
        return true;
    }

    void SetupEnv(pcstr mode) override
    {
        R_ASSERT2(mode && xr_strcmp(mode, "renderer_vulkan") == 0, "Vulkan module received another renderer mode");
        R_ASSERT2(!GEnv.Render || GEnv.Render == &renderer, "Another renderer is still bound to GEnv");
        GEnv.Render = &renderer;
        register_vulkan_options(renderer);
        particle_editor = std::make_unique<VulkanParticleEditor>();
        Msg("[renderer-vulkan] module.selected mode=%s", mode);
        GEnv.RenderFactory = &factory;
        GEnv.UIRender = &device.ui();
        GEnv.DU = &draw_utils;
#ifdef DEBUG
        GEnv.DRender = &debug_render;
#endif
    }

    void ClearEnv() override
    {
        if (GEnv.Render != &renderer)
            return;
        Msg("[renderer-vulkan] module.clear begin");
        particle_editor.reset();
        renderer.Destroy();
        GEnv.DRender = nullptr;
        GEnv.DU = nullptr;
        GEnv.UIRender = nullptr;
        GEnv.RenderFactory = nullptr;
        GEnv.Render = nullptr;
    }
};
} // namespace

RendererModule *GetRendererModule()
{
    // xrCore's logging lock can be destroyed before global renderer objects.
    // The engine explicitly calls ClearEnv during renderer shutdown; keep the
    // module alive until process exit instead of running its destructor after
    // xrCore's static teardown and logging through an invalid lock.
    static auto *module = new VulkanRendererModule;
    return module;
}
} // namespace xray::render::vulkan
