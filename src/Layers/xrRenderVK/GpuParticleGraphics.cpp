#include "xrEngine/stdafx.h"
#include "GpuParticleGraphics.h"
#include "ShaderModule.h"
#include <cstring>

namespace xray::render::vulkan
{
bool GpuParticleGraphics::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const VkPhysicalDeviceLimits& limits, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& dispatch, VkDescriptorSetLayout material_layout, VkPipelineCache cache,
    uint32_t frames, uint32_t draws, GpuParticleCompute::Source source, std::string& error,
    VkSampleCountFlagBits gbuffer_samples, VkRenderPass velocity_render_pass)
{
    destroy();
    const uint32_t sample_count = static_cast<uint32_t>(gbuffer_samples);
    if (!device || !get || !material_layout || !velocity_render_pass || !source || !frames || !draws ||
        !sample_count || (sample_count & (sample_count - 1)) || sample_count > 64 ||
        uint64_t(frames) * draws > 4096 || draws > UINT32_MAX / 2)
    { error = "invalid GPU particle graphics configuration"; return false; }
    if (!bindings_.initialize(device, get, limits, frames * draws, error)) return false;
    device_ = device; get_ = get; limits_ = limits; material_layout_ = material_layout;
    cache_ = cache; draws_per_frame_ = draws; gbuffer_samples_ = gbuffer_samples; source_ = std::move(source);
    for (uint32_t i = 0; i < frames; ++i)
    {
        auto slot = std::make_unique<Slot>();
        if (!slot->uniforms.initialize(device, limits, memory, dispatch, draws * 2, error))
        { destroy(); return false; }
        slots_.push_back(std::move(slot));
    }
    ShaderModule velocity_vertex, velocity_fragment;
    ShaderModuleDispatch shader_dispatch{
        reinterpret_cast<PFN_vkCreateShaderModule>(get_(device_, "vkCreateShaderModule")),
        reinterpret_cast<PFN_vkDestroyShaderModule>(get_(device_, "vkDestroyShaderModule"))};
    std::vector<uint8_t> bytes;
    if (!source_("vk/gpu_particle_draw.vs.spv", bytes, error) ||
        !has_spirv_entry(bytes.data(), bytes.size(), 0, "main") ||
        !velocity_vertex.initialize_bytes(device_, shader_dispatch, bytes.data(), bytes.size(), error))
    { if (error.empty()) error = "GPU particle motion vertex shader unavailable"; destroy(); return false; }
    bytes.clear();
    if (!source_("vk/gpu_particle_velocity.ps.spv", bytes, error) ||
        !has_spirv_entry(bytes.data(), bytes.size(), 4, "main") ||
        !velocity_fragment.initialize_bytes(device_, shader_dispatch, bytes.data(), bytes.size(), error))
    { if (error.empty()) error = "GPU particle motion fragment shader unavailable"; destroy(); return false; }
    if (!velocity_opaque_.initialize(device_, get_, limits_, velocity_render_pass, material_layout_,
            bindings_.layout(), velocity_vertex.handle(), velocity_fragment.handle(), cache_, 0, false, false,
            error, VK_NULL_HANDLE, VK_SAMPLE_COUNT_1_BIT, true, false) ||
        !velocity_transparent_.initialize(device_, get_, limits_, velocity_render_pass, material_layout_,
            bindings_.layout(), velocity_vertex.handle(), velocity_fragment.handle(), cache_, 0, false, false,
            error, VK_NULL_HANDLE, VK_SAMPLE_COUNT_1_BIT, true, true))
    { destroy(); return false; }
    error.clear(); return true;
}
bool GpuParticleGraphics::begin_frame(uint32_t frame, std::string& error)
{
    if (frame >= slots_.size()) { error = "invalid GPU particle graphics frame"; return false; }
    slots_[frame]->uniforms.reset(); slots_[frame]->draws = 0; error.clear(); return true;
}
bool GpuParticleGraphics::record(const FrameRecordingContext& frame, const GpuParticleBuffers& buffers,
    VkDescriptorSet material, const std::string& fragment, uint32_t blend, bool hud, bool gbuffer,
    const GpuParticleDrawParams& camera, const float* constants, const float* mvp, std::string& error,
    VkDescriptorSetLayout scene_layout, VkDescriptorSet scene)
{
    if (!device_ || frame.frame_index >= slots_.size() || !frame.render_pass || !material ||
        fragment.empty() || !constants || !mvp || (!!scene_layout != !!scene) ||
        slots_[frame.frame_index]->draws >= draws_per_frame_)
    { error = "GPU particle graphics frame unavailable or draw capacity exhausted"; return false; }
    uint64_t render_pass_key{};
    static_assert(sizeof(frame.render_pass) <= sizeof(render_pass_key));
    std::memcpy(&render_pass_key, &frame.render_pass, sizeof(frame.render_pass));
    uint64_t scene_layout_key{};
    static_assert(sizeof(scene_layout) <= sizeof(scene_layout_key));
    std::memcpy(&scene_layout_key, &scene_layout, sizeof(scene_layout));
    const std::string key = std::to_string(render_pass_key) + ":" +
        std::to_string(scene_layout_key) + ":" + std::to_string(blend) + ":" + std::to_string(hud) + ":" +
        std::to_string(gbuffer) + ":" + std::to_string(gbuffer_samples_) + ":" + fragment;
    auto found = pipelines_.find(key);
    if (found == pipelines_.end())
    {
        ShaderModuleDispatch shader_dispatch{
            reinterpret_cast<PFN_vkCreateShaderModule>(get_(device_, "vkCreateShaderModule")),
            reinterpret_cast<PFN_vkDestroyShaderModule>(get_(device_, "vkDestroyShaderModule"))};
        ShaderModule modules[2]; std::vector<uint8_t> bytes;
        const std::string names[]{"vk/gpu_particle_draw.vs.spv", fragment + ".spv"};
        for (uint32_t i = 0; i < 2; ++i)
        {
            bytes.clear();
            if (!source_(names[i], bytes, error) || !has_spirv_entry(bytes.data(), bytes.size(), i ? 4 : 0, "main") ||
                !modules[i].initialize_bytes(device_, shader_dispatch, bytes.data(), bytes.size(), error))
            { if (error.empty()) error = "invalid GPU particle graphics shader: " + names[i]; return false; }
        }
        auto pipeline = std::make_unique<GpuParticleDrawPass>();
        if (!pipeline->initialize(device_, get_, limits_, frame.render_pass, material_layout_, bindings_.layout(),
                modules[0].handle(), modules[1].handle(), cache_, blend, hud, gbuffer, error, scene_layout,
                gbuffer_samples_)) return false;
        found = pipelines_.emplace(key, std::move(pipeline)).first;
    }
    auto& slot = *slots_[frame.frame_index];
    VkDescriptorBufferInfo camera_info{}, material_info{};
    if (!slot.uniforms.append(&camera, sizeof(camera), camera_info, error) ||
        !slot.uniforms.append(constants, 256, material_info, error)) return false;
    const uint32_t descriptor = frame.frame_index * draws_per_frame_ + slot.draws++;
    float matrices[32];
    std::memcpy(matrices, mvp, sizeof(float) * 16);
    std::memcpy(matrices + 16, mvp, sizeof(float) * 16);
    return found->second->record(frame, bindings_, buffers, material, camera_info, material_info,
        descriptor, camera.bucket, matrices, error, scene);
}
bool GpuParticleGraphics::record_velocity(const FrameRecordingContext& frame, const GpuParticleBuffers& buffers,
    VkDescriptorSet material, const GpuParticleDrawParams& camera, const float* constants,
    const float* current_mvp, const float* previous_mvp, bool transparent, std::string& error)
{
    if (!device_ || frame.frame_index >= slots_.size() || !material || !constants || !current_mvp ||
        !previous_mvp || slots_[frame.frame_index]->draws >= draws_per_frame_)
    { error = "GPU particle velocity draw capacity unavailable"; return false; }
    auto& slot = *slots_[frame.frame_index];
    VkDescriptorBufferInfo camera_info{}, material_info{};
    if (!slot.uniforms.append(&camera, sizeof(camera), camera_info, error) ||
        !slot.uniforms.append(constants, 256, material_info, error)) return false;
    const uint32_t descriptor = frame.frame_index * draws_per_frame_ + slot.draws++;
    float matrices[32];
    std::memcpy(matrices, current_mvp, sizeof(float) * 16);
    std::memcpy(matrices + 16, previous_mvp, sizeof(float) * 16);
    GpuParticleDrawPass& pipeline = transparent ? velocity_transparent_ : velocity_opaque_;
    return pipeline.record(frame, bindings_, buffers, material, camera_info, material_info,
        descriptor, camera.bucket, matrices, error);
}
void GpuParticleGraphics::clear_pipelines() { pipelines_.clear(); }
void GpuParticleGraphics::rebind_gbuffer_samples(VkSampleCountFlagBits samples)
{
    if (gbuffer_samples_ == samples)
        return;
    clear_pipelines();
    gbuffer_samples_ = samples;
}
void GpuParticleGraphics::destroy()
{
    velocity_transparent_.destroy(); velocity_opaque_.destroy();
    clear_pipelines(); slots_.clear(); bindings_.destroy(); source_ = {}; device_ = VK_NULL_HANDLE;
    get_ = nullptr; material_layout_ = VK_NULL_HANDLE; cache_ = VK_NULL_HANDLE; draws_per_frame_ = 0;
    gbuffer_samples_ = VK_SAMPLE_COUNT_1_BIT;
}
}
