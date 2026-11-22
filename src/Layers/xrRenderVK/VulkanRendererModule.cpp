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
#include "ParticleVisual.h"
#include "ParticleSprite.h"
#include "xrParticles/psystem.h"
#include "xrEngine/XR_IOConsole.h"
#include "xrEngine/xr_ioc_cmd.h"
#include "xrEngine/editor_base.h"
#include <SDL.h>
#include <cstring>
#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cmath>
#include <charconv>
#include <string_view>
#include <unordered_set>

namespace xray::render::vulkan
{
namespace
{
const xr_token quality_tokens[]{{"Minimum", 0}, {"Low", 1}, {"Default", 2}, {"High", 3}, {"Extreme", 4}, {nullptr, 0}};
const xr_token aa_tokens[]{{"off", 0}, {"FXAA", 1}, {"TAA", 2}, {nullptr, 0}};
const xr_token msaa_tokens[]{{"1x", 1}, {"2x", 2}, {"4x", 4}, {nullptr, 0}};
const xr_token ao_quality_tokens[]{{"Auto", 0}, {"Low", 4}, {"Medium", 8}, {"High", 12},
    {"Ultra", 16}, {nullptr, 0}};
const xr_token sun_shafts_quality_tokens[]{{"Auto", 0}, {"Low", 8}, {"Medium", 16},
    {"High", 24}, {"Ultra", 32}, {nullptr, 0}};

class VulkanMsaaOption final : public CCC_Token
{
public:
    VulkanMsaaOption() : CCC_Token("r_vk_msaa", &gbuffer_msaa_samples, msaa_tokens) {}

    void Execute(pcstr args) override
    {
        // Older user.ltx files saved the single-sample choice as "off".
        CCC_Token::Execute(xr_stricmp(args, "off") == 0 ? "1x" : args);
    }
};
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

constexpr pcstr particle_export_name = "particles_edited";

bool particle_copy_filename(pcstr stem, std::string& filename, std::string& error)
{
    const std::string_view name(stem);
    if (name.empty() || name.size() > 63 || !std::all_of(name.begin(), name.end(), [](char c)
        { return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-'; }))
    {
        error = "Use up to 63 lowercase letters, digits, underscores or hyphens for the copy name.";
        return false;
    }
    filename = std::string(name) + ".xr";
    return true;
}

bool load_particle_document(std::vector<uint8_t>& library, std::string& error, pcstr copy_stem = nullptr)
{
    std::string filename = "particles.xr";
    if (copy_stem && !particle_copy_filename(copy_stem, filename, error))
        return false;
    string_path path;
    FS.update_path(path, copy_stem ? "$app_data_root$" : _game_data_, filename.c_str());
    IReader* source = FS.r_open(path);
    if (!source)
    {
        error = std::string("Could not open particle library: ") + path;
        return false;
    }
    const bool loaded = editable_particle_library_bytes(*source, library, error);
    FS.r_close(source);
    return loaded;
}

bool export_particle_document(const std::vector<uint8_t>& library, std::string& message,
    pcstr copy_stem = particle_export_name)
{
    std::string filename;
    if (!particle_copy_filename(copy_stem, filename, message))
        return false;
    string_path path;
    FS.update_path(path, "$app_data_root$", filename.c_str());
    convert_path_separators(path);
    FILE* file = std::fopen(path, "wbx");
    if (!file)
    {
        const int error = errno;
        message = std::string(error == EEXIST ? "Copy already exists: " : "Could not create copy: ") + path;
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
    explicit VulkanParticleEditor(VulkanLevelRender& renderer) : renderer_(renderer) {}
    pcstr tool_name() const override { return "Particle Editor"; }

    void on_tool_frame() override
    {
        if (ImGui::Begin(tool_name(), &get_open_state(), get_default_window_flags()))
        {
            ImGui::InputText("Copy name", copy_name_, sizeof(copy_name_));
            ImGui::BeginDisabled(dirty_);
            if (ImGui::Button("Open particles.xr"))
                open();
            ImGui::SameLine();
            if (ImGui::Button("Open copy"))
                open(copy_name_);
            ImGui::EndDisabled();
            ImGui::SameLine();
            ImGui::BeginDisabled(library_.empty());
            if (ImGui::Button("Export copy") && export_particle_document(library_, message_, copy_name_))
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
                            if (selected_ != i)
                                preview_.reset();
                            selected_group_ = SIZE_MAX;
                            selected_ = i;
                            limit_ = static_cast<int>(effect.max_particles);
                            duration_enabled_ = effect.time_limit >= 0.f;
                            duration_ = duration_enabled_ ? effect.time_limit : 1.f;
                            xr_strcpy(shader_, effect.shader.c_str());
                            xr_strcpy(texture_, effect.texture.c_str());
                        }
                        ImGui::PopID();
                    }
                    for (size_t i = 0; i < catalog_.groups.size(); ++i)
                    {
                        const auto& group = catalog_.groups[i];
                        if (!filter_.PassFilter(group.name.c_str()))
                            continue;
                        ImGui::PushID(&group);
                        const auto label = group.name + " (group)";
                        if (ImGui::Selectable(label.c_str(), selected_group_ == i))
                        {
                            if (selected_group_ != i)
                            {
                                preview_.reset();
                                group_item_ = SIZE_MAX;
                                adding_group_item_ = false;
                                group_duration_ = std::max(0.f, group.authored_time_limit);
                            }
                            selected_ = SIZE_MAX;
                            selected_group_ = i;
                        }
                        ImGui::PopID();
                    }
                    ImGui::EndListBox();
                }
                if (selected_ < catalog_.effects.size())
                {
                    auto& effect = catalog_.effects[selected_];
                    const bool sprite = (effect.flags & 1u) != 0;
                    if (sprite)
                    {
                        ImGui::InputText("Sprite shader", shader_, sizeof(shader_));
                        ImGui::InputText("Sprite texture", texture_, sizeof(texture_));
                    }
                    const bool material_changed = sprite && (effect.shader != shader_ || effect.texture != texture_);
                    ImGui::InputInt("Particle limit", &limit_);
                    ImGui::Checkbox("Limit emission duration", &duration_enabled_);
                    ImGui::BeginDisabled(!duration_enabled_);
                    ImGui::InputFloat("Emission duration (s)", &duration_, 0.1f, 1.f, "%.3f");
                    ImGui::EndDisabled();
                    const float duration = duration_enabled_ ? duration_ : -1.f;
                    ImGui::BeginDisabled(limit_ < 1 || limit_ > 65535 ||
                        (sprite && !texture_[0]) ||
                        (duration_enabled_ && (!std::isfinite(duration_) || duration_ < 0.f)) ||
                        (!material_changed && uint32_t(limit_) == effect.max_particles && duration == effect.time_limit));
                    if (ImGui::Button("Apply to draft"))
                    {
                        ParticleSpriteMaterial material{shader_, texture_};
                        if (edit_particle_effect_limit(library_, effect.name, static_cast<uint32_t>(limit_), message_,
                            duration == effect.time_limit ? std::nullopt : std::optional<float>{duration},
                            material_changed ? &material : nullptr))
                        {
                            effect.max_particles = static_cast<uint32_t>(limit_);
                            effect.time_limit = duration;
                            if (material_changed)
                            {
                                effect.shader = std::move(material.shader);
                                effect.texture = std::move(material.texture);
                            }
                            preview_.reset();
                            dirty_ = true;
                        }
                    }
                    ImGui::EndDisabled();
                    draw_preview();
                }
                else if (selected_group_ < catalog_.groups.size())
                {
                    draw_group_timing();
                    draw_preview();
                }
            }
        }
        ImGui::End();
        if (!get_open_state())
            preview_.reset();
    }

private:
    void choose_effect(pcstr label, std::string& selected, bool allow_empty = false)
    {
        if (!ImGui::BeginCombo(label, selected.empty() ? "None" : selected.c_str()))
            return;
        if (allow_empty && ImGui::Selectable("None", selected.empty()))
            selected.clear();
        ImGuiListClipper clipper;
        clipper.Begin(static_cast<int>(catalog_.effects.size()));
        while (clipper.Step())
        {
            for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
            {
                const auto& name = catalog_.effects[i].name;
                ImGui::PushID(i);
                if (ImGui::Selectable(name.c_str(), selected == name))
                    selected = name;
                ImGui::PopID();
            }
        }
        ImGui::EndCombo();
    }

    void draw_group_timing()
    {
        auto& group = catalog_.groups[selected_group_];
        ImGui::Text("%zu effects", group.effects.size());
        ImGui::InputFloat("Group duration (s, 0 = automatic)", &group_duration_, .1f, 1.f, "%.3f");
        ImGui::BeginDisabled(!std::isfinite(group_duration_) || group_duration_ < 0.f ||
            group_duration_ == std::max(0.f, group.authored_time_limit));
        if (ImGui::Button("Apply group duration") &&
            edit_particle_group_duration(library_, group.name, group_duration_, message_))
        {
            group.authored_time_limit = group_duration_;
            preview_.reset();
            dirty_ = true;
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(catalog_.effects.empty());
        if (ImGui::Button("New entry"))
        {
            adding_group_item_ = true;
            group_item_ = SIZE_MAX;
            group_effect_ = catalog_.effects.front().name;
            group_begin_ = 0.f;
            group_end_ = 1.f;
        }
        ImGui::EndDisabled();
        const char* selection = group_item_ < group.effects.size()
            ? group.effects[group_item_].effect.c_str() : "Select an effect";
        if (ImGui::BeginCombo("Group effect", selection))
        {
            for (size_t i = 0; i < group.effects.size(); ++i)
            {
                const auto& item = group.effects[i];
                ImGui::PushID(&item);
                const auto label = std::to_string(i + 1) + ". " + item.effect;
                if (ImGui::Selectable(label.c_str(), group_item_ == i))
                {
                    group_item_ = i;
                    adding_group_item_ = false;
                    group_begin_ = item.begin;
                    group_end_ = item.end;
                    group_effect_ = item.effect;
                    const std::string* children[]{&item.on_play, &item.on_birth, &item.on_dead};
                    for (size_t child = 0; child < group_children_.size(); ++child)
                        group_children_[child] = (item.flags & particle_group_child_flags[child])
                            ? *children[child] : std::string{};
                }
                ImGui::PopID();
            }
            ImGui::EndCombo();
        }
        if (!adding_group_item_ && group_item_ >= group.effects.size())
            return;
        choose_effect("Main effect", group_effect_);
        ImGui::InputFloat("Start (s)", &group_begin_, .1f, 1.f, "%.3f");
        ImGui::InputFloat("End (s)", &group_end_, .1f, 1.f, "%.3f");
        if (adding_group_item_)
        {
            ImGui::BeginDisabled(!std::isfinite(group_begin_) || !std::isfinite(group_end_) ||
                group_begin_ < 0.f || group_end_ <= group_begin_);
            if (ImGui::Button("Add entry to draft") &&
                add_particle_group_item(library_, group.name, group_effect_, group_begin_, group_end_, message_))
            {
                group.effects.push_back({group_effect_, {}, {}, {}, group_begin_, group_end_, 1u << 2});
                group_item_ = group.effects.size() - 1;
                adding_group_item_ = false;
                preview_.reset();
                dirty_ = true;
            }
            ImGui::EndDisabled();
            return;
        }
        const auto& item = group.effects[group_item_];
        constexpr pcstr labels[]{"Attached child", "On particle birth", "On particle death"};
        const std::string* current[]{&item.on_play, &item.on_birth, &item.on_dead};
        ParticleGroupChildEdits children;
        bool children_changed = false;
        for (size_t child = 0; child < children.size(); ++child)
        {
            choose_effect(labels[child], group_children_[child], true);
            const bool enabled = (item.flags & particle_group_child_flags[child]) != 0;
            if ((enabled && *current[child] != group_children_[child]) ||
                (!enabled && !group_children_[child].empty()))
            {
                children[child] = group_children_[child];
                children_changed = true;
            }
        }
        ImGui::BeginDisabled(!std::isfinite(group_begin_) || !std::isfinite(group_end_) ||
            group_begin_ < 0.f || group_end_ <= group_begin_ ||
            (!children_changed && group_begin_ == item.begin && group_end_ == item.end && group_effect_ == item.effect));
        if (ImGui::Button("Apply group edit to draft") &&
            edit_particle_group_item(library_, group.name, group_item_, group_begin_, group_end_,
                message_, &group_effect_, children))
        {
            auto& edited = catalog_.groups[selected_group_].effects[group_item_];
            edited.begin = group_begin_;
            edited.end = group_end_;
            edited.effect = group_effect_;
            std::string* fields[]{&edited.on_play, &edited.on_birth, &edited.on_dead};
            for (size_t child = 0; child < children.size(); ++child)
            {
                if (!children[child])
                    continue;
                *fields[child] = *children[child];
                const auto flag = particle_group_child_flags[child];
                if (fields[child]->empty())
                    edited.flags &= ~flag;
                else
                    edited.flags |= flag;
            }
            preview_.reset();
            dirty_ = true;
        }
        ImGui::EndDisabled();
        ImGui::SameLine();
        const bool duplicate = ImGui::Button("Duplicate entry");
        ImGui::SameLine();
        const bool remove = ImGui::Button("Remove entry");
        if ((duplicate || remove) && change_particle_group_items(library_, group.name, group_item_,
            duplicate ? ParticleGroupItemOperation::Duplicate : ParticleGroupItemOperation::Remove, message_))
        {
            if (duplicate)
            {
                const ParticleGroupItem copy = item;
                group.effects.insert(group.effects.begin() + group_item_ + 1, copy);
            }
            else
                group.effects.erase(group.effects.begin() + group_item_);
            group_item_ = SIZE_MAX;
            preview_.reset();
            dirty_ = true;
        }
    }

    void draw_preview()
    {
        const bool group = selected_group_ < catalog_.groups.size();
        if (ImGui::Button("Play preview"))
        {
            auto catalog = std::make_shared<ParticleCatalog>();
            if (group)
            {
                // Read the serialized draft so group children see the same
                // flags and values as they will after opening the exported copy.
                if (parse_particle_catalog({library_.data(), library_.size()}, *catalog, message_))
                {
                    const auto* definition = catalog->group(catalog_.groups[selected_group_].name);
                    if (definition)
                        preview_ = std::make_unique<VulkanParticleGroup>(catalog, *definition);
                }
            }
            else
            {
                catalog->effects.push_back(catalog_.effects[selected_]);
                preview_ = std::make_unique<VulkanParticleEffect>(catalog, catalog->effects.front());
            }
            if (preview_)
                preview_->Play();
            preview_texture_failures_.clear();
        }
        ImGui::SameLine();
        ImGui::BeginDisabled(!preview_);
        if (ImGui::Button("Stop preview"))
            preview_.reset();
        ImGui::EndDisabled();
        if (!preview_)
            return;

        const auto& io = ImGui::GetIO();
        preview_->OnFrame(static_cast<u32>(std::clamp(io.DeltaTime * 1000.f, 0.f, 100.f)));
        const size_t total = preview_->ParticlesCount();
        if (group)
            ImGui::Text("%zu particles", total);
        else
            ImGui::Text("%zu / %u particles", total, catalog_.effects[selected_].max_particles);
        if (!preview_texture_failures_.empty())
            ImGui::TextDisabled("Preview texture unavailable");
        ImGui::SliderFloat("Preview zoom", &preview_zoom_, 4.f, 128.f, "%.0f px/m", ImGuiSliderFlags_Logarithmic);
        const ImVec2 size(std::max(1.f, ImGui::GetContentRegionAvail().x), 180.f);
        ImGui::InvisibleButton("##particle_preview", size);
        const ImVec2 min = ImGui::GetItemRectMin();
        const ImVec2 max = ImGui::GetItemRectMax();
        const ImVec2 center((min.x + max.x) * .5f, (min.y + max.y) * .5f);
        auto* draw = ImGui::GetWindowDrawList();
        draw->AddRectFilled(min, max, ImGui::GetColorU32(ImGuiCol_FrameBg));
        draw->AddLine({min.x, center.y}, {max.x, center.y}, ImGui::GetColorU32(ImGuiCol_Border));
        draw->AddLine({center.x, min.y}, {center.x, max.y}, ImGui::GetColorU32(ImGuiCol_Border));
        draw->PushClipRect(min, max, true);
        const ImU32 color = ImGui::GetColorU32(ImGuiCol_Text);
        constexpr size_t max_preview_points = 4096;
        const size_t stride = std::max(size_t{1}, (total + max_preview_points - 1) / max_preview_points);
        size_t sample_offset = 0;
        const auto draw_effect = [&](const VulkanParticleEffect& visual)
        {
            PAPI::Particle* particles = nullptr;
            u32 count = 0;
            visual.particles(particles, count);
            if (!particles || !count)
                return;
            const auto& effect = visual.definition();
            xrImTextureData texture{};
            if ((effect.flags & 1u) && preview_texture_failures_.find(effect.texture) == preview_texture_failures_.end())
            {
                texture = renderer_.GetImGuiTextureId(effect.texture.c_str());
                if (texture.texture == ImTextureID_Invalid)
                    preview_texture_failures_.insert(effect.texture);
            }
            draw_preview_particles(draw, min, max, center, color, effect, visual, texture.texture,
                particles, count, sample_offset, stride);
            sample_offset = (sample_offset + stride - count % stride) % stride;
        };
        if (group)
            static_cast<VulkanParticleGroup&>(*preview_).visit_effects(draw_effect);
        else
            draw_effect(static_cast<VulkanParticleEffect&>(*preview_));
        draw->PopClipRect();
    }

    void draw_preview_particles(ImDrawList* draw, ImVec2 min, ImVec2 max, ImVec2 center, ImU32 color,
        const ParticleEffectDef& effect, const VulkanParticleEffect& visual, ImTextureID texture,
        const PAPI::Particle* particles, u32 count, size_t sample_offset, size_t stride) const
    {
        for (size_t i = sample_offset; i < count; i += stride)
        {
            const auto& particle = particles[i];
            const auto pos = visual.particle_world_position(particle);
            const ImVec2 point(center.x + (pos.x - pos.z) * preview_zoom_ * .7071f,
                center.y - (pos.y - (pos.x + pos.z) * .35f) * preview_zoom_);
            if (!std::isfinite(point.x) || !std::isfinite(point.y))
                continue;
            if (texture != ImTextureID_Invalid)
            {
                const float half_width = std::abs(particle.size.x) * preview_zoom_ * .5f;
                const float half_height = std::abs(particle.size.y) * preview_zoom_ * .5f;
                if (!std::isfinite(half_width) || !std::isfinite(half_height) || !std::isfinite(particle.rot.x))
                    continue;
                const auto [u, v] = particle_sprite_uv_origin(particle.frame, effect.frame_count,
                    effect.frame_columns, effect.frame_size);
                ImVec2 positions[4], uvs[4];
                constexpr unsigned order[]{3, 2, 1, 0};
                for (unsigned corner = 0; corner < 4; ++corner)
                {
                    const auto sprite = particle_sprite_corner(order[corner], half_width, half_height, particle.rot.x);
                    positions[corner] = {point.x + sprite.right, point.y - sprite.up};
                    uvs[corner] = {u + sprite.u * effect.frame_size[0], v + sprite.v * effect.frame_size[1]};
                }
                const u32 argb = particle.color;
                const ImU32 tint = IM_COL32((argb >> 16) & 255u, (argb >> 8) & 255u, argb & 255u, argb >> 24);
                draw->AddImageQuad(texture, positions[0], positions[1], positions[2], positions[3],
                    uvs[0], uvs[1], uvs[2], uvs[3], tint);
            }
            else if (point.x >= min.x && point.x < max.x && point.y >= min.y && point.y < max.y)
                draw->AddRectFilled({point.x - 1.f, point.y - 1.f}, {point.x + 1.f, point.y + 1.f}, color);
        }
    }

    void open(pcstr copy_stem = nullptr)
    {
        std::vector<uint8_t> bytes;
        ParticleCatalog catalog;
        if (!load_particle_document(bytes, message_, copy_stem) ||
            !parse_particle_catalog({bytes.data(), bytes.size()}, catalog, message_))
            return;
        library_ = std::move(bytes);
        catalog_ = std::move(catalog);
        selected_ = SIZE_MAX;
        selected_group_ = SIZE_MAX;
        group_item_ = SIZE_MAX;
        adding_group_item_ = false;
        preview_.reset();
    }

    VulkanLevelRender& renderer_;
    std::vector<uint8_t> library_;
    ParticleCatalog catalog_;
    std::string message_;
    char copy_name_[64]{"particles_edited"};
    char shader_[256]{}, texture_[256]{};
    ImGuiTextFilter filter_;
    std::unique_ptr<IParticleCustom> preview_;
    std::unordered_set<std::string> preview_texture_failures_;
    float preview_zoom_{32.f};
    size_t selected_{SIZE_MAX};
    size_t selected_group_{SIZE_MAX};
    size_t group_item_{SIZE_MAX};
    bool adding_group_item_{};
    std::string group_effect_;
    std::array<std::string, 3> group_children_;
    float group_begin_{}, group_end_{};
    float group_duration_{};
    int limit_{};
    float duration_{1.f};
    bool duration_enabled_{};
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
    static VulkanMsaaOption msaa;
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
    static CCC_Token ao_quality("r_vk_ao_quality", &ao_directions_override, ao_quality_tokens);
    static CCC_Float dof("r_vk_dof_strength", &dof_strength, 0.f, 1.f);
    static CCC_Float focus("r_vk_dof_focus", &dof_focus, .2f, 100.f);
    static CCC_Float motion_blur("r_vk_motion_blur", &motion_blur_strength, 0.f, 1.f);
    static CCC_Float sun_shafts("r_vk_sun_shafts_strength", &sun_shafts_strength, 0.f, 1.f);
    static CCC_Token sun_shafts_quality("r_vk_sun_shafts_quality", &sun_shaft_samples_override,
        sun_shafts_quality_tokens);
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
    Console->AddCommand(&ao_quality);
    Console->AddCommand(&dof);
    Console->AddCommand(&motion_blur);
    Console->AddCommand(&sun_shafts);
    Console->AddCommand(&sun_shafts_quality);
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
        particle_editor = std::make_unique<VulkanParticleEditor>(renderer);
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
