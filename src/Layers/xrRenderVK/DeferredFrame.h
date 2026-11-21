#pragma once

#include "GBufferTargets.h"
#include "GpuLevel.h"
#include "SunShadowTargets.h"
#include "LocalShadowTargets.h"
#include "WaterTargets.h"
#include "HiZPyramid.h"
#include "MotionVectors.h"

namespace xray::render::vulkan
{
using ShadowRecorder = void (*)(const FrameRecordingContext &, const float (&)[16], void *);
// Connects the offscreen G-buffer, level draws, fullscreen lighting, and an
// optional UI recorder to a single FrameContext submission and present.
class DeferredFrame
{
  public:
    void set_distortion_recorder(FrameRecorder recorder, void *user)
    {
        distortion_ = recorder;
        distortion_data_ = user;
    }
    void set_shadow_recorder(ShadowRecorder recorder, void *user)
    {
        shadow_models_ = recorder;
        shadow_models_data_ = user;
    }
    void set_light_grid_recorder(FrameRecorder recorder, void *user)
    {
        light_grid_ = recorder;
        light_grid_data_ = user;
    }
    void set_hiz_pyramid(HiZPyramid* pyramid) { hiz_ = pyramid; }
    void set_motion_vectors(MotionVectors* vectors) { motion_ = vectors; }
    void set_hiz_dynamic_mask_enabled(bool enabled) { hiz_dynamic_mask_enabled_ = enabled; }
    using ComputeRecorder = bool (*)(const FrameRecordingContext&, void*, std::string&);
    void set_compute_recorder(ComputeRecorder recorder, void* user) { compute_ = recorder; compute_data_ = user; }
    void set_object_velocity_recorder(FrameCheckedRecorder recorder, void* user)
    { object_velocity_ = recorder; object_velocity_data_ = user; }
    using RayRecorder = bool (*)(const FrameRecordingContext&, const GpuLevel&, const SunShadowUniform&,
        const DeferredLight&, void*);
    void set_ray_recorder(RayRecorder recorder, void* user) { ray_ = recorder; ray_data_ = user; }
    using VsmRecorder = bool (*)(const FrameRecordingContext&, void*);
    void set_vsm_recorder(VsmRecorder recorder, void* user) { vsm_ = recorder; vsm_data_ = user; }
    bool render(FrameContext &frame, GBufferTargets &targets, const GpuLevel &level, const DeferredPass &pass,
                const float (&mvp)[16], const DeferredLight &light, FrameStatus &status, std::string &error,
                FrameRecorder ui = nullptr, void *ui_data = nullptr, FrameRecorder hud = nullptr,
                void *hud_data = nullptr, FrameRecorder models = nullptr, void *models_data = nullptr,
                FrameRecorder transparent = nullptr, void *transparent_data = nullptr,
                FrameRecorder level_visuals = nullptr, void *level_data = nullptr, bool render_world = true,
                bool clear_target = false, VkDescriptorSet weather_set = VK_NULL_HANDLE,
                const WeatherLighting *weather = nullptr, FrameReadbackRecorder readback = nullptr,
                void *readback_data = nullptr, FrameRecorder compositor = nullptr, void *compositor_data = nullptr,
                SunShadowTargets *shadows = nullptr, const SunShadowUniform *shadow_uniform = nullptr,
                LocalShadowTargets *local_shadows = nullptr,
                const std::vector<LocalLightUniform> *local_lights = nullptr, FrameRecorder local_lighting = nullptr,
                void *local_data = nullptr, WaterTargets *water = nullptr,
                SunShadowTargets *rain_shadows = nullptr,
                const WaterSceneUniform *water_uniform = nullptr, FrameInterpassRecorder final_capture = nullptr,
                void *final_capture_data = nullptr, FrameRecorder final_overlay = nullptr,
                void *final_overlay_data = nullptr, FrameInterpassRecorder display_capture = nullptr,
                void *display_capture_data = nullptr, FrameRecorder display_grade = nullptr,
                void *display_grade_data = nullptr,
                FrameRecordValidation validate_recording = nullptr, void *validation_data = nullptr);

  private:
    static bool validate_recording(void *user_data, std::string& error);
    FrameRecordValidation external_validation_{};
    void* external_validation_data_{};
    FrameRecorder light_grid_{};
    ComputeRecorder compute_{};
    void* compute_data_{};
    RayRecorder ray_{};
    VsmRecorder vsm_{};
    void* vsm_data_{};
    void* ray_data_{};
    HiZPyramid* hiz_{};
    MotionVectors* motion_{};
    bool hiz_dynamic_mask_enabled_{};
    FrameCheckedRecorder object_velocity_{};
    void* object_velocity_data_{};
    void *light_grid_data_{};
    ShadowRecorder shadow_models_{};
    void *shadow_models_data_{};
    static void geometry(const FrameRecordingContext &frame, void *user_data);
    static void lighting(const FrameRecordingContext &frame, void *user_data);
    static bool lighting_inputs(uint32_t image_index, const FrameSampledImageSink& sink,
        void* user_data, std::string& error);
    static void overlay(const FrameRecordingContext &frame, void *user_data);
    static void foreground(const FrameRecordingContext &frame, void *user_data);
    static void distortion_overlay(const FrameRecordingContext &frame, void *user_data);
    static bool capture_distortion(VkCommandBuffer command, VkImage image, uint32_t index, void *user_data,
        std::string& error);
    static bool capture(VkCommandBuffer command, VkImage image, uint32_t index, void *user_data,
        std::string& error);
    GBufferTargets *targets_{};
    SunShadowTargets *shadows_{};
    SunShadowTargets *rain_shadows_{};
    SunShadowUniform shadow_uniform_{};
    FrameSubmissionGraph::Pass gbuffer_input_producer_pass_ = FrameSubmissionGraph::InvalidPass;
    FrameSubmissionGraph::Pass depth_input_producer_pass_ = FrameSubmissionGraph::InvalidPass;
    FrameSubmissionGraph::Pass shadow_input_producer_pass_ = FrameSubmissionGraph::InvalidPass;
    LocalShadowTargets *local_shadows_{};
    const std::vector<LocalLightUniform> *local_lights_{};
    FrameRecorder local_lighting_{};
    void *local_data_{};
    WaterTargets *water_{};
    WaterSceneUniform water_uniform_{};
    FrameContext *frame_{};
    const GpuLevel *level_{};
    const DeferredPass *pass_{};
    const float *mvp_{};
    DeferredLight light_{};
    VkDescriptorSet weather_set_{};
    WeatherLighting weather_{};
    FrameRecorder ui_{};
    void *ui_data_{};
    FrameRecorder hud_{};
    void *hud_data_{};
    FrameRecorder models_{};
    void *models_data_{};
    FrameRecorder transparent_{};
    FrameRecorder distortion_{};
    void *distortion_data_{};
    void *transparent_data_{};
    FrameRecorder level_visuals_{};
    void *level_data_{};
    bool render_world_{true};
    bool recorded_{};
    FrameRecorder compositor_{};
    void *compositor_data_{};
};
} // namespace xray::render::vulkan
