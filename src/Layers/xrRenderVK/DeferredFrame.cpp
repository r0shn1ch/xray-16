#include "xrEngine/stdafx.h"
#include "DeferredFrame.h"
#include "FrameImageGraph.h"

#include <array>
#include <cstring>
#include <chrono>
#include "xrEngine/device.h"

namespace xray::render::vulkan
{
void DeferredFrame::geometry(const FrameRecordingContext &frame, void *user_data)
{
    auto &context = *static_cast<DeferredFrame *>(user_data);
    context.gbuffer_input_producer_pass_ = frame.graphics_prepass_pass;
    context.depth_input_producer_pass_ = frame.graphics_prepass_pass;
    context.shadow_input_producer_pass_ = frame.graphics_prepass_pass;
    context.pass_->notify_frame_ready(frame.frame_index);
    FrameRecordingContext compute_frame = frame;
    if (frame.compute_command_buffer)
    {
        compute_frame.command_buffer = frame.compute_command_buffer;
        compute_frame.compute_command_buffer = VK_NULL_HANDLE;
        compute_frame.command_queue_flags = frame.compute_queue_flags;
        compute_frame.semaphore_handoff = frame.semaphore_handoff;
    }
    if (context.render_world_ && context.compute_)
    {
        std::string error;
        if (!context.compute_(compute_frame, context.compute_data_, error))
        {
            Msg("! [renderer-vulkan] pre-geometry compute failed: %s", error.c_str());
            context.recorded_ = false;
            return;
        }
    }
    if (context.render_world_ && context.light_grid_)
        context.light_grid_(compute_frame, context.light_grid_data_);
    if (context.render_world_ && context.level_->has_cluster_visibility())
    {
        float camera_mvp[16];
        std::memcpy(camera_mvp, context.mvp_, sizeof(camera_mvp));
        std::string error;
        // The sun-shadow draw below consumes the generated indirect commands
        // in this graphics prepass. Record culling on the same queue so its
        // compute-to-indirect barrier orders the producer before that draw;
        // the independent light-grid and particle work remains asynchronous.
        const bool hiz_history_valid = context.hiz_dynamic_mask_enabled_ && context.hiz_ &&
            context.hiz_->previous_view_is_dynamic_safe() && context.motion_ && context.motion_->history_valid();
        if (!context.level_->record_cluster_visibility(frame, camera_mvp,
                context.shadow_uniform_.sun_view_projection, SunShadowTargets::Size, error,
                hiz_history_valid ? context.motion_->previous_view_projection() : nullptr,
                hiz_history_valid ? context.hiz_->previous_image() : VK_NULL_HANDLE,
                hiz_history_valid ? context.hiz_->previous_view() : VK_NULL_HANDLE,
                hiz_history_valid ? context.hiz_->sampler() : VK_NULL_HANDLE,
                hiz_history_valid ? &context.hiz_->sharing() : nullptr,
                hiz_history_valid ? context.hiz_->levels() : 0, hiz_history_valid))
        {
            Msg("! [renderer-vulkan] Cluster DAG visibility compute failed: %s", error.c_str());
            context.recorded_ = false;
            return;
        }
    }
    const auto has_local_shadow_inputs = [&context] {
        return context.local_shadows_ && context.render_world_ && context.local_lights_ &&
            !context.local_lights_->empty();
    };
    const auto prepare_local_shadow_lights = [&context, &has_local_shadow_inputs](uint32_t image) {
        if (!has_local_shadow_inputs())
            return true;
        std::string error;
        if (!context.local_shadows_->reserve_lights(image, context.local_lights_->size(), error))
        {
            Msg("! [renderer-vulkan] local light storage: %s", error.c_str());
            return false;
        }
        for (uint32_t i = 0; i < context.local_lights_->size(); ++i)
        {
            if (!context.local_shadows_->write_light(image, i, (*context.local_lights_)[i], error))
            {
                Msg("! [renderer-vulkan] local light uniform write failed: %s", error.c_str());
                return false;
            }
        }
        return true;
    };
    const auto record_shadow_maps = [&context](const FrameRecordingContext& shadow_commands) {
        const auto shadow_begin = std::chrono::steady_clock::now();
        if (context.shadows_)
        {
            FrameRecordingContext shadow_frame;
            std::string error;
            if (!context.shadows_->begin(shadow_commands, context.shadow_uniform_, shadow_frame, error))
            {
                context.recorded_ = false;
                return false;
            }
            // Keep the depth image cleared and shader-readable, but do not
            // traverse the whole scene for a black sun (rainy/cloudy weather).
            if (context.render_world_ && (!Core.Params || !strstr(Core.Params, "-vk-no-sun-shadows")) &&
                (context.light_.color[0] > 1e-4f || context.light_.color[1] > 1e-4f || context.light_.color[2] > 1e-4f))
                context.recorded_ &= context.level_->record_sun_shadow(shadow_frame, *context.pass_,
                    context.shadow_uniform_.sun_view_projection);
            if (context.render_world_ && context.shadow_models_ &&
                (!Core.Params || !strstr(Core.Params, "-vk-no-sun-shadows")) &&
                (context.light_.color[0] > 1e-4f || context.light_.color[1] > 1e-4f || context.light_.color[2] > 1e-4f))
                context.shadow_models_(shadow_frame, context.shadow_uniform_.sun_view_projection,
                    context.shadow_models_data_);
            context.shadows_->end(shadow_commands.command_buffer);
        }
        if (context.rain_shadows_)
        {
            FrameRecordingContext rain_frame;
            std::string error;
            if (!context.rain_shadows_->begin(shadow_commands, context.shadow_uniform_, rain_frame, error))
            {
                Msg("! [renderer-vulkan] rain shadow target begin failed: %s", error.c_str());
                context.recorded_ = false;
                return false;
            }
            if (context.render_world_ && context.shadow_uniform_.rain_params[0] > 1e-4f)
            {
                context.recorded_ &= context.level_->record_sun_shadow(rain_frame, *context.pass_,
                    context.shadow_uniform_.rain_view_projection, true, true, false);
                if (context.shadow_models_)
                    context.shadow_models_(rain_frame, context.shadow_uniform_.rain_view_projection,
                        context.shadow_models_data_);
            }
            context.rain_shadows_->end(shadow_commands.command_buffer);
        }
        if (Device.dwFrame % 300 == 0)
            Msg("[renderer-vulkan] pass.cpu frame=%u pass=shadow-total ms=%.3f", Device.dwFrame,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - shadow_begin).count());
        return context.recorded_;
    };
    const auto record_local_shadow_maps = [&context, &has_local_shadow_inputs](const FrameRecordingContext& commands) {
        if (!has_local_shadow_inputs())
            return true;
        if (!context.local_shadows_->initialize_layers(commands))
            return false;
        for (uint32_t i = 0; i < context.local_lights_->size(); ++i)
        {
            const LocalLightUniform& light = (*context.local_lights_)[i];
            if (light.shadow_params[2] < .5f)
                continue;
            const uint32_t slot = uint32_t(light.shadow_params[0]) / LocalShadowFaces;
            const uint32_t faces = light.color_type[3] >= .5f ? 1 : LocalShadowFaces;
            for (uint32_t face = 0; face < faces; ++face)
            {
                FrameRecordingContext shadow_frame;
                const bool cache = context.local_shadows_->caching();
                std::string error;
                if (!cache || !context.local_shadows_->cached_face(slot, face, light.shadow_matrices[face]))
                {
                    const bool gpu_cluster_visibility = context.level_->has_cluster_visibility();
                    FrameRecordingContext visibility_frame = commands;
                    visibility_frame.extent = {LocalShadowSize, LocalShadowSize};
                    if (gpu_cluster_visibility && !context.level_->record_cluster_shadow_visibility(
                            visibility_frame, light.shadow_matrices[face], error))
                    {
                        Msg("! [renderer-vulkan] Local shadow Cluster DAG visibility failed: %s", error.c_str());
                        return false;
                    }
                    if (!context.local_shadows_->begin_face(commands, slot, face, shadow_frame, cache))
                        return false;
                    const bool recorded = context.level_->record_sun_shadow(shadow_frame, *context.pass_,
                        light.shadow_matrices[face], true, true, gpu_cluster_visibility);
                    if (recorded && !cache && context.shadow_models_)
                        context.shadow_models_(shadow_frame, light.shadow_matrices[face], context.shadow_models_data_);
                    context.local_shadows_->end(commands.command_buffer);
                    if (!recorded)
                        return false;
                    if (cache)
                        context.local_shadows_->remember_face(slot, face, light.shadow_matrices[face]);
                }
                if (cache)
                {
                    if (!context.local_shadows_->copy_cached_face(commands, slot, face) ||
                        !context.local_shadows_->begin_face(commands, slot, face, shadow_frame, false, true))
                        return false;
                    if (context.shadow_models_)
                        context.shadow_models_(shadow_frame, light.shadow_matrices[face], context.shadow_models_data_);
                    context.local_shadows_->end(commands.command_buffer);
                }
            }
        }
        return true;
    };
    if (context.recorded_ && !prepare_local_shadow_lights(frame.image_index))
        context.recorded_ = false;
    // Shadow images are imported and their render-pass final layout is recorded
    // below. The generated group then provides the submission edge from the
    // geometry prepass to the lighting continuation.
    const bool generated_shadow_submission = frame.submission_graph && frame.request_submission_command_buffer &&
        frame.graphics_prepass_pass != FrameSubmissionGraph::InvalidPass &&
        frame.graphics_queue_family != VK_QUEUE_FAMILY_IGNORED;
    if (context.recorded_ && generated_shadow_submission)
    {
        FrameImageGraph graph(FrameSubmissionGraph::Queue::Graphics);
        const VkImageSubresourceRange shadow_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        FrameImageGraph::Resource sun_shadow{};
        FrameImageGraph::Resource rain_shadow{};
        FrameImageGraph::Resource local_current_array{};
        FrameImageGraph::Resource local_cache_array{};
        std::vector<FrameImageGraph::Resource> local_current_faces;
        std::vector<FrameImageGraph::Resource> local_cache_faces;
        const bool local_shadow_graph = has_local_shadow_inputs();
        const bool local_shadow_cache = local_shadow_graph && context.local_shadows_->caching();
        const VkImageSubresourceRange local_shadow_range{
            local_shadow_graph ? context.local_shadows_->image_aspects() : VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT),
            0, 1, 0, LocalShadowSlots * LocalShadowFaces};
        const ImageUse local_current_initial = local_shadow_graph &&
            context.local_shadows_->layers_initialized(frame.image_index) ?
                ImageUse::DepthStencilReadOnly : ImageUse::Undefined;
        const ImageUse local_cache_initial = local_shadow_cache && context.local_shadows_->cache_layers_initialized() ?
            ImageUse::DepthStencilReadOnly : ImageUse::Undefined;
        std::string error;
        bool graph_ready = true;
        if (context.shadows_)
            graph_ready = graph.import_image(context.shadows_->image(frame.image_index), shadow_range,
                sun_shadow, error, FrameSubmissionGraph::Queue::Graphics, 0,
                VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::Undefined);
        if (graph_ready && context.rain_shadows_)
            graph_ready = graph.import_image(context.rain_shadows_->image(frame.image_index), shadow_range,
                rain_shadow, error, FrameSubmissionGraph::Queue::Graphics, 0,
                VK_SHARING_MODE_EXCLUSIVE, {}, ImageUse::Undefined);
        if (graph_ready && local_shadow_graph)
            graph_ready = graph.import_image(context.local_shadows_->image(frame.image_index), local_shadow_range,
                local_current_array, error, FrameSubmissionGraph::Queue::Graphics, 0,
                VK_SHARING_MODE_EXCLUSIVE, {}, local_current_initial);
        if (graph_ready && local_shadow_graph)
        {
            local_current_faces.resize(LocalShadowSlots * LocalShadowFaces);
            for (uint32_t layer = 0; graph_ready && layer < local_current_faces.size(); ++layer)
            {
                const VkImageSubresourceRange layer_range{local_shadow_range.aspectMask, 0, 1, layer, 1};
                graph_ready = graph.import_image(context.local_shadows_->image(frame.image_index), layer_range,
                    local_current_faces[layer], error, FrameSubmissionGraph::Queue::Graphics, 0,
                    VK_SHARING_MODE_EXCLUSIVE, {}, local_current_initial);
            }
        }
        if (graph_ready && local_shadow_cache)
            graph_ready = graph.import_image(context.local_shadows_->cache_image(), local_shadow_range,
                local_cache_array, error, FrameSubmissionGraph::Queue::Graphics, 0,
                VK_SHARING_MODE_EXCLUSIVE, {}, local_cache_initial);
        if (graph_ready && local_shadow_cache)
        {
            local_cache_faces.resize(LocalShadowSlots * LocalShadowFaces);
            for (uint32_t layer = 0; graph_ready && layer < local_cache_faces.size(); ++layer)
            {
                const VkImageSubresourceRange layer_range{local_shadow_range.aspectMask, 0, 1, layer, 1};
                graph_ready = graph.import_image(context.local_shadows_->cache_image(), layer_range,
                    local_cache_faces[layer], error, FrameSubmissionGraph::Queue::Graphics, 0,
                    VK_SHARING_MODE_EXCLUSIVE, {}, local_cache_initial);
            }
        }
        const auto shadow_pass = graph.add_checked_command_pass("scene shadow maps",
            FrameSubmissionGraph::Queue::Graphics,
            [&frame, &record_shadow_maps](VkCommandBuffer command, std::string& error) {
                FrameRecordingContext shadow_commands = frame;
                shadow_commands.command_buffer = command;
                shadow_commands.compute_command_buffer = VK_NULL_HANDLE;
                shadow_commands.command_queue = FrameSubmissionGraph::Queue::Graphics;
                shadow_commands.command_queue_flags = frame.graphics_queue_flags;
                if (record_shadow_maps(shadow_commands))
                    return true;
                error = "sun or rain shadow recording failed";
                return false;
            });
        FrameImageGraph::Resource sun_shadow_rendered{};
        FrameImageGraph::Resource rain_shadow_rendered{};
        if (graph_ready && context.shadows_)
            graph_ready = graph.write_render_pass(shadow_pass, sun_shadow,
                ImageUse::DepthStencilAttachment, ImageUse::DepthStencilReadOnly,
                sun_shadow_rendered, error);
        if (graph_ready && context.rain_shadows_)
            graph_ready = graph.write_render_pass(shadow_pass, rain_shadow,
                ImageUse::DepthStencilAttachment, ImageUse::DepthStencilReadOnly,
                rain_shadow_rendered, error);
        const auto graph_commands = [&frame](VkCommandBuffer command) {
            FrameRecordingContext commands = frame;
            commands.command_buffer = command;
            commands.compute_command_buffer = VK_NULL_HANDLE;
            commands.command_queue = FrameSubmissionGraph::Queue::Graphics;
            commands.command_queue_flags = frame.graphics_queue_flags;
            return commands;
        };
        const bool initialize_local_current = local_shadow_graph &&
            !context.local_shadows_->layers_initialized(frame.image_index);
        const bool initialize_local_cache = local_shadow_cache &&
            !context.local_shadows_->cache_layers_initialized();
        if (graph_ready && (initialize_local_current || initialize_local_cache))
        {
            const auto initialize_pass = graph.add_checked_command_pass("initialize local shadow arrays",
                FrameSubmissionGraph::Queue::Graphics,
                [&frame, &context](VkCommandBuffer command, std::string& pass_error) {
                    FrameRecordingContext commands = frame;
                    commands.command_buffer = command;
                    commands.compute_command_buffer = VK_NULL_HANDLE;
                    commands.command_queue = FrameSubmissionGraph::Queue::Graphics;
                    commands.command_queue_flags = frame.graphics_queue_flags;
                    if (context.local_shadows_->initialize_layers(commands, true))
                        return true;
                    pass_error = "local shadow array initialization failed";
                    return false;
                });
            FrameImageGraph::Resource initialized{};
            if (initialize_local_current)
                graph_ready = graph.write_render_pass(initialize_pass, local_current_array,
                    ImageUse::DepthStencilAttachment, ImageUse::DepthStencilReadOnly, initialized, error);
            if (graph_ready && initialize_local_cache)
                graph_ready = graph.write_render_pass(initialize_pass, local_cache_array,
                    ImageUse::DepthStencilAttachment, ImageUse::DepthStencilReadOnly, initialized, error);
        }
        const auto record_local_static_face = [&context, &graph_commands](
            VkCommandBuffer command, uint32_t slot, uint32_t face, const std::array<float, 16>& matrix,
            bool cached, bool include_dynamic, std::string& pass_error) {
            FrameRecordingContext commands = graph_commands(command);
            float shadow_matrix[16];
            std::memcpy(shadow_matrix, matrix.data(), sizeof(shadow_matrix));
            const bool gpu_cluster_visibility = context.level_->has_cluster_visibility();
            FrameRecordingContext visibility = commands;
            visibility.extent = {LocalShadowSize, LocalShadowSize};
            if (gpu_cluster_visibility && !context.level_->record_cluster_shadow_visibility(
                    visibility, shadow_matrix, pass_error))
            {
                Msg("! [renderer-vulkan] Local shadow Cluster DAG visibility failed: %s", pass_error.c_str());
                return false;
            }
            FrameRecordingContext shadow_frame;
            if (!context.local_shadows_->begin_face(commands, slot, face, shadow_frame, cached, false, true))
            {
                pass_error = "local shadow face render pass could not begin";
                return false;
            }
            const bool recorded = context.level_->record_sun_shadow(shadow_frame, *context.pass_,
                shadow_matrix, true, true, gpu_cluster_visibility);
            if (recorded && include_dynamic && context.shadow_models_)
                context.shadow_models_(shadow_frame, shadow_matrix, context.shadow_models_data_);
            context.local_shadows_->end(command);
            if (!recorded)
            {
                pass_error = "local shadow geometry recording failed";
                return false;
            }
            if (cached)
                context.local_shadows_->remember_face(slot, face, matrix.data());
            return true;
        };
        if (graph_ready && local_shadow_graph)
        {
            for (uint32_t light_index = 0; graph_ready && light_index < context.local_lights_->size(); ++light_index)
            {
                const LocalLightUniform& light = (*context.local_lights_)[light_index];
                if (light.shadow_params[2] < .5f)
                    continue;
                const uint32_t slot = uint32_t(light.shadow_params[0]) / LocalShadowFaces;
                const uint32_t face_count = light.color_type[3] >= .5f ? 1 : LocalShadowFaces;
                for (uint32_t face = 0; graph_ready && face < face_count; ++face)
                {
                    const uint32_t layer = slot * LocalShadowFaces + face;
                    if (slot >= LocalShadowSlots || layer >= local_current_faces.size())
                    {
                        error = "local shadow light selected an invalid array layer";
                        graph_ready = false;
                        break;
                    }
                    std::array<float, 16> matrix{};
                    std::memcpy(matrix.data(), light.shadow_matrices[face], sizeof(matrix));
                    const bool cache_face = local_shadow_cache;
                    const bool render_static = !cache_face || !context.local_shadows_->cached_face(
                        slot, face, matrix.data());
                    FrameImageGraph::Resource cached_face = cache_face ? local_cache_faces[layer] :
                        FrameImageGraph::Resource{};
                    if (render_static)
                    {
                        const auto static_pass = graph.add_checked_command_pass(
                            cache_face ? "render cached local shadow face" : "render local shadow face",
                            FrameSubmissionGraph::Queue::Graphics,
                            [&record_local_static_face, matrix, slot, face, cache_face](
                                VkCommandBuffer command, std::string& pass_error) {
                                return record_local_static_face(command, slot, face, matrix,
                                    cache_face, !cache_face, pass_error);
                            });
                        FrameImageGraph::Resource rendered_cache{};
                        const FrameImageGraph::Resource destination = cache_face ? cached_face :
                            local_current_faces[layer];
                        if (!graph.write_render_pass(static_pass, destination,
                                ImageUse::DepthStencilAttachment, ImageUse::DepthStencilReadOnly,
                                rendered_cache, error))
                        {
                            graph_ready = false;
                            break;
                        }
                        if (cache_face)
                            cached_face = rendered_cache;
                    }
                    if (!cache_face)
                        continue;
                    const auto copy_pass = graph.add_checked_command_pass("copy cached local shadow face",
                        FrameSubmissionGraph::Queue::Graphics,
                        [&context, image = frame.image_index, slot, face](VkCommandBuffer command,
                            std::string& pass_error) {
                            if (context.local_shadows_->record_cached_face_copy(command, image, slot, face))
                                return true;
                            pass_error = "cached local shadow face copy failed";
                            return false;
                        });
                    FrameImageGraph::Resource copied_face{};
                    if (!graph.read(copy_pass, cached_face, ImageUse::TransferSource, error) ||
                        !graph.write(copy_pass, local_current_faces[layer], ImageUse::TransferDestination,
                            copied_face, error))
                    {
                        graph_ready = false;
                        break;
                    }
                    const auto restore_pass = graph.add_checked_command_pass("restore cached local shadow face",
                        FrameSubmissionGraph::Queue::Graphics,
                        [](VkCommandBuffer, std::string&) { return true; });
                    if (!graph.read(restore_pass, cached_face, ImageUse::DepthStencilReadOnly, error) ||
                        !graph.depends_on(restore_pass, copy_pass, error))
                    {
                        graph_ready = false;
                        break;
                    }
                    const auto dynamic_pass = graph.add_checked_command_pass("draw dynamic local shadow models",
                        FrameSubmissionGraph::Queue::Graphics,
                        [&context, matrix, slot, face, &graph_commands](
                            VkCommandBuffer command, std::string& pass_error) {
                            FrameRecordingContext commands = graph_commands(command);
                            float shadow_matrix[16];
                            std::memcpy(shadow_matrix, matrix.data(), sizeof(shadow_matrix));
                            FrameRecordingContext shadow_frame;
                            if (!context.local_shadows_->begin_face(commands, slot, face, shadow_frame,
                                    false, true, true))
                            {
                                pass_error = "dynamic local shadow render pass could not begin";
                                return false;
                            }
                            if (context.shadow_models_)
                                context.shadow_models_(shadow_frame, shadow_matrix, context.shadow_models_data_);
                            context.local_shadows_->end(command);
                            return true;
                        });
                    FrameImageGraph::Resource dynamic_face{};
                    if (!graph.write_render_pass(dynamic_pass, copied_face,
                            ImageUse::DepthStencilAttachment, ImageUse::DepthStencilReadOnly,
                            dynamic_face, error))
                    {
                        graph_ready = false;
                        break;
                    }
                }
            }
        }
        std::vector<FrameImageGraph::QueueGroup> groups;
        std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
        const auto queue_family = [&frame](FrameSubmissionGraph::Queue queue) {
            return queue == FrameSubmissionGraph::Queue::Graphics ? frame.graphics_queue_family :
                FrameSubmissionGraph::InvalidQueueFamily;
        };
        ImageStateTracker execution_states;
        FrameImageGraph::ExecutionState execution;
        if (graph_ready && context.shadows_)
            graph_ready = execution_states.register_image(context.shadows_->image(frame.image_index),
                shadow_range, ImageUse::Undefined, error);
        if (graph_ready && context.rain_shadows_)
            graph_ready = execution_states.register_image(context.rain_shadows_->image(frame.image_index),
                shadow_range, ImageUse::Undefined, error);
        if (graph_ready && local_shadow_graph)
            graph_ready = execution_states.register_image(context.local_shadows_->image(frame.image_index),
                local_shadow_range, local_current_initial, error);
        if (graph_ready && local_shadow_cache)
            graph_ready = execution_states.register_image(context.local_shadows_->cache_image(),
                local_shadow_range, local_cache_initial, error);
        if (graph_ready)
            graph_ready = graph.begin_execution(execution, error);
        if (graph_ready)
            graph_ready = graph.generate_submission_groups(*frame.submission_graph, queue_family,
                groups, submission_for_pass, error);
        if (!graph_ready)
        {
            Msg("! [renderer-vulkan] shadow graph scheduling failed: %s", error.c_str());
            context.recorded_ = false;
        }
        for (const FrameImageGraph::QueueGroup& group : groups)
        {
            if (!context.recorded_)
                break;
            if (group.queue != FrameSubmissionGraph::Queue::Graphics ||
                !frame.submission_graph->depends_on(group.submission, frame.graphics_prepass_pass, error))
            {
                Msg("! [renderer-vulkan] shadow group has an invalid graphics dependency: %s", error.c_str());
                context.recorded_ = false;
                break;
            }
            VkCommandBuffer command = VK_NULL_HANDLE;
            if (!frame.request_submission_command_buffer(group.submission, group.queue, command, error))
            {
                Msg("! [renderer-vulkan] shadow group command buffer unavailable: %s", error.c_str());
                context.recorded_ = false;
                break;
            }
            if (!graph.execute_group(group, command, execution_states, execution,
                    {frame.cmd_pipeline_barrier}, error,
                    frame.graphics_queue_flags, frame.semaphore_handoff))
            {
                Msg("! [renderer-vulkan] shadow group recording failed: %s", error.c_str());
                context.recorded_ = false;
                break;
            }
            context.shadow_input_producer_pass_ = group.submission;
        }
    }
    else if (context.recorded_)
    {
        context.recorded_ &= record_shadow_maps(frame);
        if (context.recorded_)
            context.recorded_ &= record_local_shadow_maps(frame);
    }
    float transform[16];
    std::memcpy(transform, context.mvp_, sizeof(transform));
    FrameImageGraph geometry_graph(FrameSubmissionGraph::Queue::Graphics);
    ImageStateTracker geometry_states;
    std::array<FrameImageGraph::Resource, 8> geometry_attachments{};
    const uint32_t attachment_count = context.targets_->graph_attachment_count();
    const FrameImageSharing& geometry_sharing = context.targets_->image_sharing();
    std::string geometry_error;
    bool geometry_graph_ready = attachment_count > 0 && attachment_count <= geometry_attachments.size();
    if (!geometry_graph_ready)
        geometry_error = "G-buffer attachment count is invalid";
    for (uint32_t attachment = 0; geometry_graph_ready && attachment < attachment_count; ++attachment)
    {
        const VkImage image = context.targets_->graph_attachment_image(frame.image_index, attachment);
        const VkImageSubresourceRange range{
            context.targets_->graph_attachment_aspect(attachment), 0, 1, 0, 1};
        if (image == VK_NULL_HANDLE || !range.aspectMask)
        {
            geometry_error = "G-buffer render-pass attachment is unavailable";
            geometry_graph_ready = false;
            break;
        }
        geometry_graph_ready = geometry_graph.import_image(image, range,
                geometry_attachments[attachment], geometry_error,
                FrameSubmissionGraph::Queue::Graphics, 0, geometry_sharing.mode,
                geometry_sharing.queue_families, ImageUse::Undefined) &&
            geometry_states.register_image(image, range, ImageUse::Undefined, geometry_error);
    }
    const auto geometry_pass = geometry_graph.add_checked_command_pass("G-buffer geometry",
        FrameSubmissionGraph::Queue::Graphics,
        [&context, &frame, &transform, targets = context.targets_](
            VkCommandBuffer command, std::string& pass_error) {
            FrameRecordingContext recording_frame = frame;
            recording_frame.command_buffer = command;
            recording_frame.command_queue = FrameSubmissionGraph::Queue::Graphics;
            recording_frame.command_queue_flags = frame.graphics_queue_flags;
            FrameRecordingContext geometry_frame;
            if (!targets->begin(recording_frame, geometry_frame))
            {
                pass_error = "G-buffer geometry render pass could not begin";
                return false;
            }
            bool recorded = true;
            if (context.render_world_)
            {
                if (context.level_visuals_)
                    context.level_visuals_(geometry_frame, context.level_data_);
                else
                    recorded = context.level_->record(geometry_frame, *context.pass_, transform,
                        GeometryPhase::OpaqueAndAlphaTest);
                if (context.models_)
                    context.models_(geometry_frame, context.models_data_);
            }
            targets->end(command);
            if (!recorded)
            {
                pass_error = "G-buffer level geometry recording failed";
                return false;
            }
            return true;
        });
    for (uint32_t attachment = 0; geometry_graph_ready && attachment < attachment_count; ++attachment)
    {
        const VkImageAspectFlags aspect = context.targets_->graph_attachment_aspect(attachment);
        const bool depth_attachment = (aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0;
        const ImageUse attachment_use = depth_attachment ? ImageUse::DepthStencilAttachment :
            ImageUse::ColorAttachment;
        const ImageUse final_use = depth_attachment ? ImageUse::DepthStencilReadOnly :
            ImageUse::SampledGraphicsCompute;
        FrameImageGraph::Resource written;
        geometry_graph_ready = geometry_graph.write_render_pass(geometry_pass,
            geometry_attachments[attachment], attachment_use, final_use, written, geometry_error);
    }
    if (geometry_graph_ready)
        geometry_graph_ready = geometry_graph.execute(frame.command_buffer, geometry_states,
            {frame.cmd_pipeline_barrier}, geometry_error, frame.graphics_queue_flags,
            frame.semaphore_handoff, FrameSubmissionGraph::Queue::Graphics);
    if (!geometry_graph_ready)
    {
        Msg("! [renderer-vulkan] G-buffer graph recording failed: %s", geometry_error.c_str());
        context.recorded_ = false;
        return;
    }
    if (!context.targets_->copy_scene_depth(frame))
    {
        Msg("! [renderer-vulkan] G-buffer scene depth copy failed");
        context.recorded_ = false;
        return;
    }
    FrameRecordingContext geometry_consumers = frame;
    FrameSubmissionGraph::Pass ray_visibility_submission_pass = frame.graphics_prepass_pass;
    geometry_consumers.ray_visibility_submission_pass = &ray_visibility_submission_pass;
    if (context.recorded_ && context.motion_ && context.motion_->ready())
    {
        FrameImageGraph graph(FrameSubmissionGraph::Queue::Graphics);
        std::string error;
        const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        const FrameImageSharing& sharing = context.targets_->image_sharing();
        FrameImageGraph::Resource depth, sampled_depth, copied_sampled_depth;
        if (!graph.import_image(context.targets_->depth_image(frame.image_index), depth_range,
                depth, error, FrameSubmissionGraph::Queue::Graphics, 0,
                sharing.mode, sharing.queue_families, ImageUse::DepthStencilProduced) ||
            !graph.import_image(context.targets_->sampled_depth_image(frame.image_index), depth_range,
                sampled_depth, error, FrameSubmissionGraph::Queue::Graphics, 0,
                sharing.mode, sharing.queue_families, ImageUse::Undefined))
        {
            Msg("! [renderer-vulkan] depth-copy graph import failed: %s", error.c_str());
            context.recorded_ = false;
        }
        const auto depth_copy_pass = graph.add_checked_command_pass("G-buffer depth copy",
            FrameSubmissionGraph::Queue::Graphics,
            [targets = context.targets_, image_index = frame.image_index](
                VkCommandBuffer command, std::string& pass_error) {
                if (targets->record_depth_copy(command, image_index))
                    return true;
                pass_error = "G-buffer depth copy recording failed";
                return false;
            });
        if (context.recorded_ &&
            (!graph.read(depth_copy_pass, depth, ImageUse::TransferSource, error) ||
                !graph.write(depth_copy_pass, sampled_depth, ImageUse::TransferDestination,
                    copied_sampled_depth, error)))
        {
            Msg("! [renderer-vulkan] depth-copy graph setup failed: %s", error.c_str());
            context.recorded_ = false;
        }
        const auto depth_ready_pass = graph.add_pass("G-buffer depth ready",
            FrameSubmissionGraph::Queue::Graphics, [] {});
        const ImageUse sampled_depth_use = (frame.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT)
            ? ImageUse::SampledGraphicsCompute : ImageUse::Sampled;
        if (context.recorded_ &&
            (!graph.read(depth_ready_pass, depth, ImageUse::DepthStencilReadOnly, error) ||
                !graph.read(depth_ready_pass, copied_sampled_depth, sampled_depth_use, error)))
        {
            Msg("! [renderer-vulkan] depth-ready graph setup failed: %s", error.c_str());
            context.recorded_ = false;
        }
        const auto motion_pass = graph.add_checked_command_pass("camera motion vectors",
            FrameSubmissionGraph::Queue::Graphics,
            [&context, &frame](VkCommandBuffer command, std::string& pass_error) {
                FrameRecordingContext recorded_frame = frame;
                recorded_frame.command_buffer = command;
                recorded_frame.compute_command_buffer = VK_NULL_HANDLE;
                recorded_frame.command_queue = FrameSubmissionGraph::Queue::Graphics;
                recorded_frame.command_queue_flags = frame.graphics_queue_flags;
                if (!context.motion_->record(recorded_frame, &pass_error))
                {
                    if (pass_error.empty()) pass_error = "camera motion-vector pass failed";
                    return false;
                }
                return true;
            });
        if (context.recorded_ &&
            (!graph.read(motion_pass, copied_sampled_depth, sampled_depth_use, error) ||
                !graph.depends_on(motion_pass, depth_ready_pass, error)))
        {
            Msg("! [renderer-vulkan] motion graph depth dependency failed: %s", error.c_str());
            context.recorded_ = false;
        }
        FrameImageGraph::Pass object_velocity_pass = ~0u;
        if (context.render_world_ && context.object_velocity_)
        {
            object_velocity_pass = graph.add_checked_command_pass("object velocity overlay",
                FrameSubmissionGraph::Queue::Graphics,
                [&context, &frame](VkCommandBuffer command, std::string& pass_error) {
                    FrameRecordingContext recorded_frame = frame;
                    recorded_frame.command_buffer = command;
                    recorded_frame.compute_command_buffer = VK_NULL_HANDLE;
                    recorded_frame.command_queue = FrameSubmissionGraph::Queue::Graphics;
                    recorded_frame.command_queue_flags = frame.graphics_queue_flags;
                    if (!context.object_velocity_(recorded_frame, context.object_velocity_data_, pass_error))
                        return false;
                    return true;
                });
            if (!graph.depends_on(object_velocity_pass, motion_pass, error))
            {
                Msg("! [renderer-vulkan] motion graph dependency failed: %s", error.c_str());
                context.recorded_ = false;
            }
        }

        const bool generated = frame.submission_graph && frame.request_submission_command_buffer &&
            frame.graphics_prepass_pass != FrameSubmissionGraph::InvalidPass &&
            frame.graphics_queue_family != VK_QUEUE_FAMILY_IGNORED;
        if (context.recorded_ && generated)
        {
            std::vector<FrameImageGraph::QueueGroup> groups;
            std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
            const auto queue_family = [&frame](FrameSubmissionGraph::Queue queue) {
                return queue == FrameSubmissionGraph::Queue::Graphics ? frame.graphics_queue_family :
                    FrameSubmissionGraph::InvalidQueueFamily;
            };
            ImageStateTracker execution_states;
            FrameImageGraph::ExecutionState execution;
            if (!execution_states.register_image(context.targets_->depth_image(frame.image_index),
                    depth_range, ImageUse::DepthStencilProduced, error) ||
                !execution_states.register_image(context.targets_->sampled_depth_image(frame.image_index),
                    depth_range, ImageUse::Undefined, error))
            {
                Msg("! [renderer-vulkan] depth-copy state tracking failed: %s", error.c_str());
                context.recorded_ = false;
            }
            if (!graph.generate_submission_groups(*frame.submission_graph, queue_family,
                    groups, submission_for_pass, error) || !graph.begin_execution(execution, error))
            {
                Msg("! [renderer-vulkan] motion graph scheduling failed: %s", error.c_str());
                context.recorded_ = false;
            }
            for (const FrameImageGraph::QueueGroup& group : groups)
            {
                if (!context.recorded_)
                    break;
                if (group.queue != FrameSubmissionGraph::Queue::Graphics ||
                    !frame.submission_graph->depends_on(group.submission,
                        frame.graphics_prepass_pass, error))
                {
                    Msg("! [renderer-vulkan] motion group has an invalid graphics dependency: %s",
                        error.c_str());
                    context.recorded_ = false;
                    break;
                }
                VkCommandBuffer command = VK_NULL_HANDLE;
                if (!frame.request_submission_command_buffer(group.submission, group.queue, command, error))
                {
                    Msg("! [renderer-vulkan] motion group command buffer unavailable: %s", error.c_str());
                    context.recorded_ = false;
                    break;
                }
                if (!graph.execute_group(group, command, execution_states, execution,
                        {frame.cmd_pipeline_barrier}, error,
                        frame.graphics_queue_flags, frame.semaphore_handoff))
                {
                    Msg("! [renderer-vulkan] motion group recording failed: %s", error.c_str());
                    context.recorded_ = false;
                    break;
                }
                context.depth_input_producer_pass_ = group.submission;
                geometry_consumers.geometry_motion_submission_pass = group.submission;
            }
        }
        else if (context.recorded_)
        {
            ImageStateTracker execution_states;
            if (!execution_states.register_image(context.targets_->depth_image(frame.image_index),
                    depth_range, ImageUse::DepthStencilProduced, error) ||
                !execution_states.register_image(context.targets_->sampled_depth_image(frame.image_index),
                    depth_range, ImageUse::Undefined, error))
            {
                Msg("! [renderer-vulkan] depth-copy state tracking failed: %s", error.c_str());
                context.recorded_ = false;
            }
            if (!graph.execute(frame.command_buffer, execution_states,
                    {frame.cmd_pipeline_barrier}, error,
                    frame.command_queue_flags, frame.semaphore_handoff, frame.command_queue))
            {
                Msg("! [renderer-vulkan] motion graph recording failed: %s", error.c_str());
                context.recorded_ = false;
            }
        }
    }
    else if (context.recorded_)
        context.recorded_ &= context.targets_->copy_depth(frame);
    if (context.recorded_ && context.hiz_ && context.hiz_->ready())
        context.recorded_ &= context.hiz_->record(geometry_consumers, context.hiz_dynamic_mask_enabled_);
    if (context.recorded_ && context.render_world_ && context.ray_)
        context.recorded_ &= context.ray_(geometry_consumers, *context.level_, context.shadow_uniform_, context.light_,
            context.ray_data_);
    if (context.recorded_ && context.render_world_ && context.vsm_)
    {
        const bool generated = frame.submission_graph && frame.request_submission_command_buffer &&
            frame.graphics_prepass_pass != FrameSubmissionGraph::InvalidPass &&
            frame.graphics_queue_family != VK_QUEUE_FAMILY_IGNORED;
        if (generated)
        {
            FrameImageGraph graph(FrameSubmissionGraph::Queue::Graphics);
            const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
            const FrameImageSharing& sharing = context.targets_->image_sharing();
            const ImageUse sampled_depth_initial_use =
                (geometry_consumers.graphics_queue_flags & VK_QUEUE_COMPUTE_BIT) ?
                    ImageUse::SampledGraphicsCompute : ImageUse::Sampled;
            FrameImageGraph::Resource sampled_depth;
            std::string error;
            if (!graph.import_image(context.targets_->sampled_depth_image(frame.image_index), depth_range,
                    sampled_depth, error, FrameSubmissionGraph::Queue::Graphics, 0,
                    sharing.mode, sharing.queue_families, sampled_depth_initial_use,
                    context.depth_input_producer_pass_))
            {
                Msg("! [renderer-vulkan] VSM depth import failed: %s", error.c_str());
                context.recorded_ = false;
            }
            const auto vsm_pass = graph.add_checked_command_pass("virtual shadow maps",
                FrameSubmissionGraph::Queue::Graphics,
                [&context, &geometry_consumers](VkCommandBuffer command, std::string& pass_error) {
                    FrameRecordingContext recorded_frame = geometry_consumers;
                    recorded_frame.command_buffer = command;
                    recorded_frame.compute_command_buffer = VK_NULL_HANDLE;
                    recorded_frame.command_queue = FrameSubmissionGraph::Queue::Graphics;
                    recorded_frame.command_queue_flags = geometry_consumers.graphics_queue_flags;
                    if (!context.vsm_(recorded_frame, context.vsm_data_))
                    {
                        pass_error = "VSM request and page-render pass failed";
                        return false;
                    }
                    return true;
                });
            if (context.recorded_ && !graph.read(vsm_pass, sampled_depth,
                    ImageUse::SampledGraphicsCompute, error))
            {
                Msg("! [renderer-vulkan] VSM depth dependency setup failed: %s", error.c_str());
                context.recorded_ = false;
            }
            std::vector<FrameImageGraph::QueueGroup> groups;
            std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
            const auto queue_family = [&frame](FrameSubmissionGraph::Queue queue) {
                return queue == FrameSubmissionGraph::Queue::Graphics ? frame.graphics_queue_family :
                    FrameSubmissionGraph::InvalidQueueFamily;
            };
            ImageStateTracker execution_states;
            FrameImageGraph::ExecutionState execution;
            if (context.recorded_ &&
                (!execution_states.register_image(context.targets_->sampled_depth_image(frame.image_index),
                        depth_range, sampled_depth_initial_use, error) ||
                    !graph.generate_submission_groups(*frame.submission_graph, queue_family,
                        groups, submission_for_pass, error) || !graph.begin_execution(execution, error)))
            {
                Msg("! [renderer-vulkan] VSM graph scheduling failed: %s", error.c_str());
                context.recorded_ = false;
            }
            for (const FrameImageGraph::QueueGroup& group : groups)
            {
                if (!context.recorded_)
                    break;
                if (group.queue != FrameSubmissionGraph::Queue::Graphics)
                {
                    Msg("! [renderer-vulkan] VSM group has an invalid queue class");
                    context.recorded_ = false;
                    break;
                }
                VkCommandBuffer command = VK_NULL_HANDLE;
                if (!frame.request_submission_command_buffer(group.submission, group.queue, command, error))
                {
                    Msg("! [renderer-vulkan] VSM group command buffer unavailable: %s", error.c_str());
                    context.recorded_ = false;
                    break;
                }
                if (!graph.execute_group(group, command, execution_states, execution,
                        {frame.cmd_pipeline_barrier}, error,
                        frame.graphics_queue_flags, frame.semaphore_handoff))
                {
                    Msg("! [renderer-vulkan] VSM group recording failed: %s", error.c_str());
                    context.recorded_ = false;
                    break;
                }
            }
        }
        else
            context.recorded_ &= context.vsm_(geometry_consumers, context.vsm_data_);
    }
}

void DeferredFrame::lighting(const FrameRecordingContext &frame, void *user_data)
{
    auto &context = *static_cast<DeferredFrame *>(user_data);
    // A failed prepass can leave RT descriptors or attachments unready.
    // FrameContext still invokes this callback while finishing the submission.
    if (!context.recorded_)
        return;
    context.recorded_ &=
        context.pass_->record_lighting(frame, context.targets_->multisampled_lighting_set(frame.image_index), context.light_,
                                       context.weather_set_, context.weather_set_ ? &context.weather_ : nullptr);
    // The base panel receives local lights in both full-screen and four-pane
    // views; the other three panels display G-buffer channels only.
    if (context.render_world_ && context.local_lighting_ && context.shadow_uniform_.options[1] < 2.5f)
        context.local_lighting_(frame, context.local_data_);
    if (!context.water_)
        overlay(frame, user_data);
}

bool DeferredFrame::lighting_inputs(uint32_t image_index, const FrameSampledImageSink& sink,
    void* user_data, std::string& error)
{
    const auto& context = *static_cast<DeferredFrame*>(user_data);
    if (!context.targets_)
    {
        error = "deferred lighting has no G-buffer targets";
        return false;
    }
    const FrameImageSharing& sharing = context.targets_->image_sharing();
    for (uint32_t attachment = 0; attachment < context.targets_->graph_attachment_count(); ++attachment)
    {
        if (!context.targets_->lighting_samples_attachment(attachment))
            continue;
        const VkImageAspectFlags aspect = context.targets_->graph_attachment_aspect(attachment);
        const VkImage image = context.targets_->graph_attachment_image(image_index, attachment);
        if (!image)
        {
            error = "deferred lighting G-buffer attachment is unavailable";
            return false;
        }
        const ImageUse producer_use = aspect == VK_IMAGE_ASPECT_DEPTH_BIT ?
            ImageUse::DepthStencilProduced : ImageUse::SampledGraphicsCompute;
        if (!sink({image, {aspect, 0, 1, 0, 1},
                aspect == VK_IMAGE_ASPECT_DEPTH_BIT ? ImageUse::DepthStencilReadOnly :
                    ImageUse::SampledGraphicsCompute, sharing,
                context.gbuffer_input_producer_pass_, producer_use}, error))
            return false;
    }
    const VkImage depth = context.targets_->sampled_depth_image(image_index);
    if (!depth)
    {
        error = "deferred lighting sampled depth is unavailable";
        return false;
    }
    if (!sink({depth, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1},
            ImageUse::SampledGraphicsCompute, sharing,
            context.depth_input_producer_pass_, ImageUse::SampledGraphicsCompute}, error))
        return false;
    const VkImageSubresourceRange shadow_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    if (context.shadows_ && !sink({context.shadows_->image(image_index), shadow_range,
            ImageUse::DepthStencilReadOnly, {}, context.shadow_input_producer_pass_,
            ImageUse::DepthStencilReadOnly}, error))
        return false;
    if (context.rain_shadows_ && !sink({context.rain_shadows_->image(image_index), shadow_range,
            ImageUse::DepthStencilReadOnly, {}, context.shadow_input_producer_pass_,
            ImageUse::DepthStencilReadOnly}, error))
        return false;
    if (context.render_world_ && context.local_shadows_ && context.local_lights_ &&
        !context.local_lights_->empty())
    {
        const VkImageSubresourceRange local_range{context.local_shadows_->image_aspects(),
            0, 1, 0, LocalShadowSlots * LocalShadowFaces};
        if (!sink({context.local_shadows_->image(image_index), local_range,
                ImageUse::DepthStencilReadOnly, {}, context.shadow_input_producer_pass_,
                ImageUse::DepthStencilReadOnly}, error))
            return false;
    }
    error.clear();
    return true;
}

bool DeferredFrame::capture(VkCommandBuffer command, VkImage image, uint32_t index, void *user_data,
    std::string& error)
{
    auto &context = *static_cast<DeferredFrame *>(user_data);
    if (!context.water_->write_uniform(index, context.water_uniform_, error))
    {
        context.recorded_ = false;
        return false;
    }
    if (!context.water_->capture(command, image, index))
    {
        context.recorded_ = false;
        error = "water scene image capture failed";
        return false;
    }
    error.clear();
    return true;
}

void DeferredFrame::overlay(const FrameRecordingContext &frame, void *user_data)
{
    auto &context = *static_cast<DeferredFrame *>(user_data);
    const float debug_mode = context.shadow_uniform_.options[1];
    // Only the full-screen base view shows forward particles and the HUD.
    // Their screen-space coordinates do not map into the repeated G-buffer
    // panels and would contaminate the shadow, lightmap and albedo views.
    const bool isolated_gbuffer = debug_mode > 0.5f && (debug_mode < 1.5f || debug_mode > 2.5f);
    if (context.render_world_ && !isolated_gbuffer)
    {
        if (context.transparent_)
            context.transparent_(frame, context.transparent_data_);
        else if (!context.level_visuals_)
        {
            float transform[16];
            std::memcpy(transform, context.mvp_, sizeof(transform));
            context.recorded_ &= context.level_->record(frame, *context.pass_, transform, GeometryPhase::Transparent);
        }
    }
    if (!(context.distortion_ && context.water_ && context.render_world_))
        foreground(frame, user_data);
}

bool DeferredFrame::capture_distortion(VkCommandBuffer command, VkImage image, uint32_t index,
    void *user_data, std::string& error)
{
    auto &context = *static_cast<DeferredFrame *>(user_data);
    const float mode = context.shadow_uniform_.options[1];
    if (mode > .5f && (mode < 1.5f || mode > 2.5f))
    {
        error.clear();
        return true;
    }
    if (!context.water_->capture_refraction(command, image, index))
    {
        context.recorded_ = false;
        error = "water refraction image capture failed";
        return false;
    }
    context.frame_->record_distortion_mask(command, index, context.distortion_, context.distortion_data_);
    if (!context.water_->capture_distortion(command, image, index))
    {
        context.recorded_ = false;
        error = "water distortion image capture failed";
        return false;
    }
    error.clear();
    return true;
}

bool DeferredFrame::validate_recording(void *user_data, std::string& error)
{
    auto& context = *static_cast<DeferredFrame*>(user_data);
    if (!context.recorded_)
    {
        error = "Vulkan deferred level command recording failed";
        return false;
    }
    return !context.external_validation_ ||
        context.external_validation_(context.external_validation_data_, error);
}

void DeferredFrame::distortion_overlay(const FrameRecordingContext &frame, void *user_data)
{
    auto &context = *static_cast<DeferredFrame *>(user_data);
    const float debug_mode = context.shadow_uniform_.options[1];
    const bool isolated_gbuffer = debug_mode > .5f && (debug_mode < 1.5f || debug_mode > 2.5f);
    if (!isolated_gbuffer)
        context.recorded_ &= context.pass_->composite_distortion(frame);
    foreground(frame, user_data);
}

void DeferredFrame::foreground(const FrameRecordingContext &frame, void *user_data)
{
    auto &context = *static_cast<DeferredFrame *>(user_data);
    const float debug_mode = context.shadow_uniform_.options[1];
    const bool isolated_gbuffer = debug_mode > .5f && (debug_mode < 1.5f || debug_mode > 2.5f);
    // The HUD uses its own depth hierarchy and is never hidden by world geometry.
    context.frame_->clear_depth(frame.command_buffer);
    if (context.hud_ && !isolated_gbuffer)
        context.hud_(frame, context.hud_data_);
    if (context.ui_)
        context.ui_(frame, context.ui_data_);
}

bool DeferredFrame::render(FrameContext &frame, GBufferTargets &targets, const GpuLevel &level,
                           const DeferredPass &pass, const float (&mvp)[16], const DeferredLight &light,
                           FrameStatus &status, std::string &error, FrameRecorder ui, void *ui_data, FrameRecorder hud,
                           void *hud_data, FrameRecorder models, void *models_data, FrameRecorder transparent,
                           void *transparent_data, FrameRecorder level_visuals, void *level_data, bool render_world,
                           bool clear_target, VkDescriptorSet weather_set, const WeatherLighting *weather,
                           FrameReadbackRecorder readback, void *readback_data, FrameRecorder compositor,
                           void *compositor_data, SunShadowTargets *shadows, const SunShadowUniform *shadow_uniform,
                           LocalShadowTargets *local_shadows, const std::vector<LocalLightUniform> *local_lights,
                           FrameRecorder local_lighting, void *local_data, WaterTargets *water,
                           SunShadowTargets *rain_shadows, const WaterSceneUniform *water_uniform,
                           FrameInterpassRecorder final_capture,
                           void *final_capture_data, FrameRecorder final_overlay, void *final_overlay_data,
                           FrameInterpassRecorder display_capture, void *display_capture_data,
                           FrameRecorder display_grade, void *display_grade_data,
                           FrameRecordValidation validate_recording, void *validation_data)
{
    targets_ = &targets;
    shadows_ = shadows;
    if (shadow_uniform)
        shadow_uniform_ = *shadow_uniform;
    local_shadows_ = local_shadows;
    local_lights_ = local_lights;
    local_lighting_ = local_lighting;
    local_data_ = local_data;
    water_ = water;
    rain_shadows_ = rain_shadows;
    if (water_uniform)
        water_uniform_ = *water_uniform;
    frame_ = &frame;
    level_ = &level;
    pass_ = &pass;
    mvp_ = mvp;
    light_ = light;
    weather_set_ = weather_set;
    if (weather)
        weather_ = *weather;
    // Native 2D UI follows scene postprocessing; the 3D HUD stays in foreground.
    const bool native_ui = frame.presentation_render_pass() != VK_NULL_HANDLE;
    ui_ = native_ui ? nullptr : ui;
    ui_data_ = ui_data;
    hud_ = hud;
    hud_data_ = hud_data;
    models_ = models;
    models_data_ = models_data;
    transparent_ = transparent;
    transparent_data_ = transparent_data;
    level_visuals_ = level_visuals;
    level_data_ = level_data;
    render_world_ = render_world;
    recorded_ = true;
    external_validation_ = validate_recording;
    external_validation_data_ = validation_data;
    compositor_ = compositor;
    compositor_data_ = compositor_data;
    const VkClearColorValue clear{{0, 0, 0, 1}};
    const bool refract_world = distortion_ && water_ && render_world_;
    if (!frame.render_frame(clear, status, error, lighting, this, readback, readback_data, geometry, this, clear_target,
                            compositor_, compositor_data_, water_ ? capture : nullptr, this, water_ ? overlay : nullptr,
                            this, native_ui ? nullptr : final_capture, final_capture_data,
                            native_ui ? nullptr : final_overlay, final_overlay_data,
                            refract_world ? capture_distortion : nullptr, this,
                            refract_world ? distortion_overlay : nullptr, this,
                            native_ui ? ui : nullptr, ui_data,
                            native_ui ? final_capture : nullptr, final_capture_data,
                            native_ui ? final_overlay : nullptr, final_overlay_data,
                            display_capture, display_capture_data, display_grade, display_grade_data,
                            render_world && (compute_ || light_grid_),
                            DeferredFrame::validate_recording, this, DeferredFrame::lighting_inputs, this))
        return false;
    if (!recorded_)
    {
        error = "Vulkan deferred level command recording failed";
        return false;
    }
    error.clear();
    return true;
}
} // namespace xray::render::vulkan
