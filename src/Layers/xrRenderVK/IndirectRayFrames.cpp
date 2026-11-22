#include "IndirectRayFrames.h"
#include "IndirectRadianceShaders.h"
#include "IndirectLocalShaders.h"
#include "RestirGIShaders.h"
#include "WorldRadianceCacheShaders.h"
#include "ReferencePathTracePass.h"
#include "LightingLimits.h"
#include "FrameImageGraph.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace xray::render::vulkan
{
namespace
{
constexpr uint32_t WorldRadianceCacheCapacity = 32768;
struct WorldRadianceCacheCellLayout
{
    uint32_t state{};
    int32_t cell_x{}, cell_y{}, cell_z{};
    uint32_t normal_class{}, sample_count{}, last_frame{}, cell_level{};
    float radiance_r{}, radiance_g{}, radiance_b{}, radiance_padding{};
    float normal_x{}, normal_y{}, normal_z{}, normal_padding{};
    float direction_r_x{}, direction_r_y{}, direction_r_z{}, direction_r_padding{};
    float direction_g_x{}, direction_g_y{}, direction_g_z{}, direction_g_padding{};
    float direction_b_x{}, direction_b_y{}, direction_b_z{}, direction_b_padding{};
    float second_order_r[5]{};
    float second_order_g[5]{};
    float second_order_b[5]{};
};
static_assert(sizeof(WorldRadianceCacheCellLayout) == 172);

uint32_t world_cache_signature(const LevelRayScene& scene, const float* sun_direction_distance,
    const float* sky_color, const float* sun_color, const std::vector<IndirectLocalLight>& local_lights)
{
    uint32_t hash = 2166136261u;
    const auto mix = [](uint32_t& target, uint32_t value) {
        target = (target ^ value) * 16777619u;
    };
    const auto quantize = [](float value, float scale) {
        if (!std::isfinite(value)) return 0x7fc00000u;
        const double scaled = std::clamp(double(value) * scale,
            double(std::numeric_limits<int32_t>::min()), double(std::numeric_limits<int32_t>::max()));
        return static_cast<uint32_t>(static_cast<int32_t>(std::llround(scaled)));
    };

    const VkDeviceAddress scene_address = scene.address();
    mix(hash, static_cast<uint32_t>(scene_address));
    mix(hash, static_cast<uint32_t>(scene_address >> 32));
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        mix(hash, quantize(sun_direction_distance[axis], 512.f));
        mix(hash, quantize(sky_color[axis], 16.f));
        mix(hash, quantize(sun_color[axis], 16.f));
    }

    // Light order can change as the renderer builds its visible-light list, so
    // fold each light independently and combine the hashes commutatively.
    uint32_t light_xor = 0;
    uint32_t light_sum = 0;
    for (const IndirectLocalLight& light : local_lights)
    {
        uint32_t light_hash = 2166136261u;
        for (uint32_t axis = 0; axis < 3; ++axis)
        {
            mix(light_hash, quantize(light.position_range[axis], 0.25f)); // 4 m spatial bins
            mix(light_hash, quantize(light.direction_cone[axis], 16.f));
            mix(light_hash, quantize(light.color_type[axis], 16.f));
            mix(light_hash, quantize(light.spot_right[axis], 16.f));
        }
        mix(light_hash, quantize(light.position_range[3], 0.25f));
        mix(light_hash, quantize(light.direction_cone[3], 32.f));
        mix(light_hash, quantize(light.color_type[3], 1.f));
        mix(light_hash, quantize(light.spot_right[3], 16.f));
        mix(light_hash, light.projection_slot);
        light_xor ^= light_hash;
        light_sum += light_hash * 0x9e3779b9u;
    }
    mix(hash, static_cast<uint32_t>(local_lights.size()));
    mix(hash, light_xor);
    mix(hash, light_sum);
    return hash ? hash : 1u;
}
}

bool IndirectRayFrames::initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
    const std::vector<VkImageView>& depth, const std::vector<VkImageView>& normal, VkSampler sampler,
    const VkPhysicalDeviceProperties& properties, const VkPhysicalDeviceMemoryProperties& memory,
    const BufferResourceDispatch& buffers, const BindlessTextureTable& alpha_textures,
    VkPipelineCache cache, std::string& error,
    const std::vector<VkImage>& depth_images, const std::vector<VkImage>& normal_images,
    const std::vector<VkImageView>& albedo_views, const std::vector<VkImageView>& surface_views,
    const std::vector<VkImage>& albedo_images, const std::vector<VkImage>& surface_images,
    const FrameImageSharing* sharing)
{
    destroy();
    const uint64_t pixels = uint64_t(extent.width) * extent.height;
    const uint64_t groups = (pixels + 63) / 64;
    const uint64_t groups_x = groups < 65535 ? groups : 65535;
    const uint64_t groups_y = groups_x ? (groups + groups_x - 1) / groups_x : 0;
    if (!device || !get || !sampler || !pixels || pixels > UINT32_MAX || depth.empty() ||
        depth.size() > 16 || depth.size() != normal.size() ||
        pixels > properties.limits.maxStorageBufferRange / sizeof(RayQueryInput) ||
        pixels > properties.limits.maxStorageBufferRange / sizeof(RayQueryHit) ||
        pixels > properties.limits.maxStorageBufferRange / sizeof(RaySurface) ||
        pixels > properties.limits.maxStorageBufferRange / (sizeof(float) * 4) ||
        pixels > properties.limits.maxStorageBufferRange / sizeof(RestirGIReservoir) ||
        properties.limits.maxPerStageDescriptorStorageBuffers < 7 ||
        properties.limits.maxDescriptorSetStorageBuffers < 7 ||
        properties.limits.maxPerStageDescriptorSampledImages < 4 ||
        properties.limits.maxDescriptorSetSampledImages < 4 ||
        properties.limits.maxPerStageDescriptorSamplers < 4 ||
        properties.limits.maxDescriptorSetSamplers < 4 ||
        groups_x > properties.limits.maxComputeWorkGroupCount[0] ||
        groups_y > properties.limits.maxComputeWorkGroupCount[1] ||
        (uint64_t(extent.width) + 7) / 8 > properties.limits.maxComputeWorkGroupCount[0] ||
        (uint64_t(extent.height) + 7) / 8 > properties.limits.maxComputeWorkGroupCount[1] ||
        depth_images.size() != normal_images.size() ||
        (!depth_images.empty() && (depth_images.size() != depth.size() ||
            std::any_of(depth_images.begin(), depth_images.end(), [](VkImage image) { return image == VK_NULL_HANDLE; }) ||
            std::any_of(normal_images.begin(), normal_images.end(), [](VkImage image) { return image == VK_NULL_HANDLE; }))) ||
        albedo_views.size() != surface_views.size() || albedo_images.size() != surface_images.size() ||
        (!albedo_views.empty() && (albedo_views.size() != depth.size() ||
            std::any_of(albedo_views.begin(), albedo_views.end(), [](VkImageView view) { return view == VK_NULL_HANDLE; }) ||
            std::any_of(surface_views.begin(), surface_views.end(), [](VkImageView view) { return view == VK_NULL_HANDLE; }))) ||
        (!albedo_images.empty() && (albedo_images.size() != depth.size() ||
            std::any_of(albedo_images.begin(), albedo_images.end(), [](VkImage image) { return image == VK_NULL_HANDLE; }) ||
            std::any_of(surface_images.begin(), surface_images.end(), [](VkImage image) { return image == VK_NULL_HANDLE; }))) ||
        (albedo_views.empty() != albedo_images.empty()) ||
        (!albedo_images.empty() && depth_images.empty()))
    {
        error = "indirect ray extent exceeds device limits";
        return false;
    }
    cmd_pipeline_barrier_ = reinterpret_cast<PFN_vkCmdPipelineBarrier>(get(device, "vkCmdPipelineBarrier"));
    cmd_fill_buffer_ = reinterpret_cast<PFN_vkCmdFillBuffer>(get(device, "vkCmdFillBuffer"));
    if (!cmd_pipeline_barrier_ || !cmd_fill_buffer_)
    { error = "indirect frame graph requires vkCmdPipelineBarrier"; destroy(); return false; }
    extent_ = extent;
    alpha_texture_capacity_ = alpha_textures.capacity();
    depth_images_.assign(depth_images.begin(), depth_images.end());
    normal_images_.assign(normal_images.begin(), normal_images.end());
    normal_views_.assign(normal.begin(), normal.end());
    albedo_views_.assign(albedo_views.begin(), albedo_views.end());
    surface_views_.assign(surface_views.begin(), surface_views.end());
    albedo_images_.assign(albedo_images.begin(), albedo_images.end());
    surface_images_.assign(surface_images.begin(), surface_images.end());
    sharing_ = sharing ? *sharing : FrameImageSharing{};
    local_shadows_ = properties.limits.maxPerStageDescriptorStorageBuffers >= 8 &&
        properties.limits.maxDescriptorSetStorageBuffers >= 8;
    rays_.resize(depth.size());
    hits_.resize(depth.size());
    surfaces_.resize(depth.size());
    primary_surfaces_.resize(depth.size());
    sun_rays_.resize(depth.size());
    sun_hits_.resize(depth.size());
    if (local_shadows_) { local_rays_.resize(depth.size()); local_hits_.resize(depth.size()); }
    radiance_.resize(depth.size());
    restir_radiance_.resize(depth.size());
    reservoir_history_.resize(2);
    local_lights_.resize(FrameContext::FramesInFlight);
    for (auto& buffer : local_lights_)
        if (!buffer.initialize(device, XRAY_VK_MAX_LOCAL_LIGHTS * sizeof(IndirectLocalLight),
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                memory, buffers, error))
        { destroy(); return false; }
    if (uint64_t(WorldRadianceCacheCapacity) * sizeof(WorldRadianceCacheCellLayout) > properties.limits.maxStorageBufferRange ||
        !world_radiance_cache_.initialize(device,
            VkDeviceSize(WorldRadianceCacheCapacity) * sizeof(WorldRadianceCacheCellLayout),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, buffers, error) ||
        !world_cache_update_pass_.initialize(device, get, FrameContext::FramesInFlight, 4, 16,
            properties.limits.maxStorageBufferRange, cache, world_radiance_cache_shaders::Update,
            sizeof(world_radiance_cache_shaders::Update), error))
    { if (error.empty()) error = "world radiance cache exceeds device limits"; destroy(); return false; }
    world_cache_clear_pending_ = true;
    for (auto& buffer : reservoir_history_)
        if (!buffer.initialize(device, pixels * sizeof(RestirGIReservoir), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, memory, buffers, error))
        { destroy(); return false; }
    const uint32_t projection_capacity = std::min(8u, alpha_textures.capacity());
    const uint64_t path_trace_sampled_descriptors =
        uint64_t(alpha_textures.capacity()) + projection_capacity + 1u;
    path_trace_enabled_ = properties.limits.maxPerStageDescriptorStorageBuffers >= 10 &&
        properties.limits.maxDescriptorSetStorageBuffers >= 10 &&
        properties.limits.maxBoundDescriptorSets >= 3 &&
        path_trace_sampled_descriptors <= properties.limits.maxPerStageDescriptorSampledImages &&
        path_trace_sampled_descriptors <= properties.limits.maxDescriptorSetSampledImages &&
        path_trace_sampled_descriptors <= properties.limits.maxPerStageDescriptorSamplers &&
        path_trace_sampled_descriptors <= properties.limits.maxDescriptorSetSamplers &&
        pixels <= properties.limits.maxStorageBufferRange / sizeof(ReferencePathTraceHistoryCell);
    if (path_trace_enabled_)
    {
        path_trace_history_.resize(2);
        bool allocated = true;
        for (auto& history : path_trace_history_)
            if (!history.initialize(device, pixels * sizeof(ReferencePathTraceHistoryCell),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    memory, buffers, error))
            {
                allocated = false;
                break;
            }
        if (!allocated)
        {
            for (auto& history : path_trace_history_) history.destroy();
            path_trace_history_.clear();
            path_trace_enabled_ = false;
            error.clear();
        }
    }
    std::vector<BufferResource*> outputs;
    for (size_t i = 0; i < depth.size(); ++i)
    {
        const auto storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        const auto device_local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (!rays_[i].initialize(device, pixels * sizeof(RayQueryInput), storage, device_local, memory, buffers, error) ||
            !hits_[i].initialize(device, pixels * sizeof(RayQueryHit), storage, device_local, memory, buffers, error) ||
            !sun_rays_[i].initialize(device, pixels * sizeof(RayQueryInput),
                storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, device_local, memory, buffers, error) ||
            !sun_hits_[i].initialize(device, pixels * sizeof(RayQueryHit), storage, device_local, memory, buffers, error) ||
            (local_shadows_ && !local_rays_[i].initialize(device, pixels * sizeof(RayQueryInput), storage, device_local, memory, buffers, error)) ||
            (local_shadows_ && !local_hits_[i].initialize(device, pixels * sizeof(RayQueryHit), storage, device_local, memory, buffers, error)) ||
            !radiance_[i].initialize(device, pixels * sizeof(float) * 4,
                storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, device_local, memory, buffers, error) ||
            !restir_radiance_[i].initialize(device, pixels * sizeof(float) * 4,
                storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, device_local, memory, buffers, error) ||
            !primary_surfaces_[i].initialize(device, pixels * sizeof(uint32_t) * 2,
                storage, device_local, memory, buffers, error) ||
            !surfaces_[i].initialize(device, pixels * sizeof(RaySurface),
                storage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, device_local, memory, buffers, error))
        { destroy(); return false; }
        outputs.push_back(&rays_[i]);
    }
    std::vector<BufferResource*> primary_surface_outputs;
    for (auto& buffer : primary_surfaces_) primary_surface_outputs.push_back(&buffer);
    if (!generator_.initialize(device, get, extent, depth, normal, sampler, outputs, cache, error,
            RayGeneratePass::Mode::Indirect, primary_surface_outputs) ||
        !queries_.initialize(device, get, FrameContext::FramesInFlight, cache, alpha_textures, error) ||
        !sun_generator_.initialize(device, get, FrameContext::FramesInFlight,
            properties.limits.maxStorageBufferRange, cache, error) ||
        !sun_queries_.initialize(device, get, FrameContext::FramesInFlight, cache, alpha_textures, error) ||
        (local_shadows_ && !local_generator_.initialize(device, get, FrameContext::FramesInFlight, 3, 16,
            properties.limits.maxStorageBufferRange, cache, indirect_local_shaders::Generate,
            sizeof(indirect_local_shaders::Generate), error)) ||
        (local_shadows_ && !local_queries_.initialize(device, get, FrameContext::FramesInFlight, cache, alpha_textures, error)))
    { destroy(); return false; }
    const std::vector<StorageComputePass::Binding> restir_bindings{
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, {5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {6, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, {7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER},
        {8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, {9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}};
    if (!restir_spatial_pass_.initialize(device, get, FrameContext::FramesInFlight, restir_bindings, 20,
            properties.limits.maxStorageBufferRange, 0, cache, restir_gi_shaders::Spatial,
            sizeof(restir_gi_shaders::Spatial), error))
    { destroy(); return false; }
    reset_reservoir_history();
    const uint32_t capacity = projection_capacity;
    for (uint32_t slot = 0; slot < FrameContext::FramesInFlight && capacity; ++slot)
    {
        auto table = std::make_unique<BindlessTextureTable>();
        if (!table->initialize(device, get, capacity, error)) { projection_tables_.clear(); break; }
        projection_tables_.push_back(std::move(table));
    }
    if (path_trace_enabled_ && !reference_path_trace_pass_.initialize(device, get,
            FrameContext::FramesInFlight, properties.limits.maxStorageBufferRange, cache,
            alpha_textures, projection_tables_.empty() ? VK_NULL_HANDLE : projection_tables_[0]->layout(), error))
    {
        reference_path_trace_pass_.destroy();
        for (auto& history : path_trace_history_) history.destroy();
        path_trace_history_.clear();
        path_trace_enabled_ = false;
        error.clear();
    }
    std::vector<StorageComputePass::Binding> layout;
    for (uint32_t binding = 0; binding < (local_shadows_ ? 7u : 6u); ++binding)
        layout.push_back({binding, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
    layout.push_back({7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER});
    if (!projection_tables_.empty())
    {
        const uint32_t* code = local_shadows_ ? indirect_radiance_shaders::EvaluateProjection :
            indirect_radiance_shaders::EvaluateProjectionUnshadowed;
        const size_t size = local_shadows_ ? sizeof(indirect_radiance_shaders::EvaluateProjection) :
            sizeof(indirect_radiance_shaders::EvaluateProjectionUnshadowed);
    if (!radiance_pass_.initialize(device, get, FrameContext::FramesInFlight, layout, 48,
                properties.limits.maxStorageBufferRange, 0, cache, code, size, error, projection_tables_[0]->layout()))
            projection_tables_.clear();
    }
    if (projection_tables_.empty() && !radiance_pass_.initialize(device, get, FrameContext::FramesInFlight,
            layout, 48, properties.limits.maxStorageBufferRange, 0, cache,
            local_shadows_ ? indirect_radiance_shaders::Evaluate : indirect_radiance_shaders::EvaluateUnshadowed,
            local_shadows_ ? sizeof(indirect_radiance_shaders::Evaluate) : sizeof(indirect_radiance_shaders::EvaluateUnshadowed), error))
    { destroy(); return false; }
    error.clear();
    return true;
}

bool IndirectRayFrames::attach_output(VkDevice device, PFN_vkGetDeviceProcAddr get,
    const std::vector<VkImage>& images, const std::vector<VkImageView>& views,
    VkPipelineCache cache, std::string& error)
{
    release_output();
    if (images.size() != radiance_.size() || views.size() != radiance_.size() || images.empty())
    { error = "indirect output image count does not match frame resources"; return false; }
    std::vector<BufferResource*> buffers;
    for (auto& buffer : restir_radiance_) buffers.push_back(&buffer);
    if (!image_resolve_.initialize(device, get, extent_, images, views, buffers, cache, error,
            RayVisibilityPass::Mode::IndirectRadiance))
        return false;
    output_layouts_.assign(images.size(), VK_IMAGE_LAYOUT_UNDEFINED);
    output_views_.assign(views.begin(), views.end());
    return true;
}

bool IndirectRayFrames::prepare_output(VkPhysicalDevice physical, VkDevice device, PFN_vkGetDeviceProcAddr get,
    const FrameDispatch& dispatch, FrameImagePool& pool, VkPipelineCache cache, std::string& error)
{
    if (!physical || !device || !get || !dispatch.get_format_properties || !dispatch.create_image_view ||
        !dispatch.destroy_image_view || restir_radiance_.empty())
    { error = "indirect image creation requires initialized frame resources"; return false; }
    VkFormatProperties properties{};
    dispatch.get_format_properties(physical, VK_FORMAT_R16G16B16A16_SFLOAT, &properties);
    const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((properties.optimalTilingFeatures & required) != required)
    { error = "indirect output requires sampled and storage RGBA16F images"; return false; }
    std::vector<VkImage> images(restir_radiance_.size());
    std::vector<VkImageView> views(restir_radiance_.size());
    const auto discard = [&] {
        for (VkImageView view : views) if (view) dispatch.destroy_image_view(device, view, nullptr);
        for (VkImage image : images) if (image) pool.release(image);
    };
    VkImageCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    info.imageType = VK_IMAGE_TYPE_2D; info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    info.extent = {extent_.width, extent_.height, 1}; info.mipLevels = info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT; info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    sharing_.apply(info);
    for (size_t i = 0; i < images.size(); ++i)
    {
        if (!pool.acquire(info, images[i], error)) { discard(); return false; }
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = images[i]; view.viewType = VK_IMAGE_VIEW_TYPE_2D; view.format = info.format;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (dispatch.create_image_view(device, &view, nullptr, &views[i]) != VK_SUCCESS)
        { error = "indirect output image view creation failed"; discard(); return false; }
    }
    if (!attach_output(device, get, images, views, cache, error)) { discard(); return false; }
    output_device_ = device;
    destroy_output_view_ = dispatch.destroy_image_view;
    output_pool_ = &pool;
    owned_images_.assign(images.begin(), images.end());
    error.clear();
    return true;
}

bool IndirectRayFrames::record(const FrameRecordingContext& frame, const LevelRayScene& scene,
    RaySurfacePass& surface_pass, const float* inverse_view_projection, float max_distance,
    uint32_t seed, const float* sun_direction_distance, const float* sky_color, const float* sun_color,
    const std::vector<IndirectLocalLight>& local_lights, std::string& error,
    const std::vector<VkImageView>& projections, VkSampler projection_sampler,
    VkImageView motion_view, bool motion_history_valid, bool temporal_history_valid,
    VkImage motion_image, bool motion_has_previous_depth)
{
    path_trace_history_reused_ = false;
    if (!frame.command_buffer || frame.image_index >= rays_.size() || !inverse_view_projection ||
        !(max_distance > 0.f) || !scene.address() || !sun_direction_distance || !sky_color || !sun_color ||
        frame.frame_index >= local_lights_.size() || local_lights.size() > XRAY_VK_MAX_LOCAL_LIGHTS ||
        !projection_sampler || !motion_view)
    { error = "invalid indirect ray recording request"; return false; }
    const uint32_t current_world_cache_signature = world_cache_signature(scene,
        sun_direction_distance, sky_color, sun_color, local_lights);
    if (world_cache_signature_ != current_world_cache_signature)
    {
        world_cache_signature_ = current_world_cache_signature;
        world_cache_clear_pending_ = true;
    }
    FrameRecordingContext reduced_frame = frame;
    reduced_frame.extent = extent_;
    const FrameImageSharing frame_sharing = frame_image_sharing(frame.graphics_queue_family,
        frame.compute_queue_family);
    const bool sharing_matches_frame = sharing_.mode == frame_sharing.mode &&
        sharing_.queue_families == frame_sharing.queue_families;
    // The RTGI graph reads global bindless alpha textures and the ray scene's
    // acceleration-structure backing buffers, which are not imported into this
    // graph for ownership transfers. Keep those accesses on the graphics
    // family until those external resources are created for concurrent sharing.
    const bool compute_inputs_are_queue_shareable =
        frame.graphics_queue_family != VK_QUEUE_FAMILY_IGNORED &&
        frame.compute_queue_family != VK_QUEUE_FAMILY_IGNORED &&
        frame.compute_queue_family == frame.graphics_queue_family &&
        (frame.compute_queue_flags & VK_QUEUE_COMPUTE_BIT) != 0;
    const bool use_generated_compute = frame.submission_graph &&
        frame.compute_submission_pass != FrameSubmissionGraph::InvalidPass &&
        frame.graphics_prepass_pass != FrameSubmissionGraph::InvalidPass &&
        frame.graphics_continuation_pass != FrameSubmissionGraph::InvalidPass &&
        frame.request_submission_command_buffer && sharing_matches_frame &&
        compute_inputs_are_queue_shareable;
    const FrameSubmissionGraph::Queue recording_queue = use_generated_compute ?
        FrameSubmissionGraph::Queue::Compute : frame.command_queue;
    FrameRecordingContext recording_frame = frame;
    float constants[4]{0.f, float(extent_.width), float(extent_.height), max_distance};
    static_assert(sizeof(seed) == sizeof(constants[0]));
    std::memcpy(&constants[0], &seed, sizeof(seed));
    const uint32_t count = extent_.width * extent_.height;
    auto& rays = rays_[frame.image_index];
    auto& hits = hits_[frame.image_index];
    auto& light_buffer = local_lights_[frame.frame_index];
    VkDescriptorSet projection_set = VK_NULL_HANDLE;
    if (!projection_tables_.empty())
    {
        auto& table = *projection_tables_[frame.frame_index];
        if (projections.size() > table.capacity() || (!projections.empty() && !projection_sampler))
        { error = "invalid indirect projection textures"; return false; }
        for (size_t index = 0; index < projections.size(); ++index)
            if (!table.write(static_cast<uint32_t>(index), projections[index], projection_sampler, error)) return false;
        for (const auto& light : local_lights)
            if (light.projection_slot != UINT32_MAX && light.projection_slot >= projections.size())
            { error = "indirect light projection slot is out of bounds"; return false; }
        projection_set = table.set();
    }
    if (!local_lights.empty() && !light_buffer.write(0, local_lights.data(),
            local_lights.size() * sizeof(IndirectLocalLight), error)) return false;
    reservoir_output_index_ = reservoir_history_valid_ ? (reservoir_history_index_ ^ 1u) : 0u;
    const uint32_t reservoir_input_index = reservoir_output_index_ ^ 1u;
    struct Constants { float sky[4], sun[4]; uint32_t count, groups_x, light_count, seed; } parameters{};
    std::memcpy(parameters.sky, sky_color, sizeof(float) * 3);
    std::memcpy(parameters.sun, sun_color, sizeof(float) * 3);
    const uint32_t groups = static_cast<uint32_t>((uint64_t(count) + 63) / 64);
    parameters.count = count;
    parameters.light_count = static_cast<uint32_t>(local_lights.size());
    parameters.seed = seed;
    parameters.groups_x = groups < 65535u ? groups : 65535u;
    const auto info = [](const BufferResource& buffer) {
        return VkDescriptorBufferInfo{buffer.handle(), 0, buffer.size()};
    };
    std::vector<VkDescriptorBufferInfo> buffers{info(rays), info(surfaces_[frame.image_index]),
        info(sun_rays_[frame.image_index]), info(sun_hits_[frame.image_index]),
        info(radiance_[frame.image_index]), info(light_buffer)};
    if (local_shadows_) buffers.push_back(info(local_hits_[frame.image_index]));
    buffers.push_back(info(world_radiance_cache_));

    FrameImageGraph graph(recording_queue);
    ImageStateTracker image_states;
    using GraphBuffer = FrameImageGraph::BufferResource;
    const auto import_device_buffer = [&](const BufferResource& buffer, GraphBuffer& resource) {
        return graph.import_buffer(buffer.handle(), 0, buffer.size(), BufferUse::AllCommandsReadWrite,
            resource, error, FrameSubmissionGraph::Queue::Graphics, frame.previous_graphics_submission,
            buffer.sharing_mode(), buffer.sharing_queue_families());
    };
    GraphBuffer rays_resource, hits_resource, surfaces_resource, primary_surfaces_resource;
    GraphBuffer sun_rays_resource, sun_hits_resource, radiance_resource;
    GraphBuffer previous_reservoir_resource, output_reservoir_resource, resolved_radiance_resource;
    if (!import_device_buffer(rays, rays_resource) ||
        !import_device_buffer(hits, hits_resource) ||
        !import_device_buffer(surfaces_[frame.image_index], surfaces_resource) ||
        !import_device_buffer(primary_surfaces_[frame.image_index], primary_surfaces_resource) ||
        !import_device_buffer(sun_rays_[frame.image_index], sun_rays_resource) ||
        !import_device_buffer(sun_hits_[frame.image_index], sun_hits_resource) ||
        !import_device_buffer(radiance_[frame.image_index], radiance_resource) ||
        !import_device_buffer(reservoir_history_[reservoir_input_index], previous_reservoir_resource) ||
        !import_device_buffer(reservoir_history_[reservoir_output_index_], output_reservoir_resource) ||
        !import_device_buffer(restir_radiance_[frame.image_index], resolved_radiance_resource))
        return false;
    const bool path_trace_history_valid = path_trace_enabled_ && motion_history_valid &&
        reservoir_history_valid_ && motion_image != VK_NULL_HANDLE;
    path_trace_history_reused_ = path_trace_history_valid;
    GraphBuffer previous_path_trace_history_resource{}, next_path_trace_history_resource{};
    if (path_trace_enabled_)
    {
        if ((path_trace_history_valid && !graph.import_buffer(
                path_trace_history_[reservoir_input_index].handle(), 0,
                path_trace_history_[reservoir_input_index].size(), BufferUse::AllCommandsReadWrite,
                previous_path_trace_history_resource, error, FrameSubmissionGraph::Queue::Graphics,
                frame.previous_graphics_submission,
                path_trace_history_[reservoir_input_index].sharing_mode(),
                path_trace_history_[reservoir_input_index].sharing_queue_families())) ||
            !graph.import_buffer(path_trace_history_[reservoir_output_index_].handle(), 0,
                path_trace_history_[reservoir_output_index_].size(),
                path_trace_history_valid ? BufferUse::AllCommandsReadWrite : BufferUse::Undefined,
                next_path_trace_history_resource, error, FrameSubmissionGraph::Queue::Graphics,
                frame.previous_graphics_submission,
                path_trace_history_[reservoir_output_index_].sharing_mode(),
                path_trace_history_[reservoir_output_index_].sharing_queue_families()))
            return false;
    }

    GraphBuffer world_cache_resource{};
    if (!graph.import_buffer(world_radiance_cache_.handle(), 0, world_radiance_cache_.size(),
            world_cache_clear_pending_ ? BufferUse::Undefined : BufferUse::AllCommandsReadWrite,
            world_cache_resource, error, FrameSubmissionGraph::Queue::Graphics,
            frame.previous_graphics_submission, world_radiance_cache_.sharing_mode(),
            world_radiance_cache_.sharing_queue_families()))
        return false;
    if (world_cache_clear_pending_)
    {
        const auto clear_cache = graph.add_checked_pass("clear world radiance cache", [&](std::string& pass_error) {
            pass_error.clear();
            cmd_fill_buffer_(recording_frame.command_buffer, world_radiance_cache_.handle(), 0,
                world_radiance_cache_.size(), 0);
            return true;
        });
        GraphBuffer cleared_cache;
        if (!graph.write_buffer(clear_cache, world_cache_resource, BufferUse::TransferDestination,
                cleared_cache, error))
            return false;
        world_cache_resource = cleared_cache;
    }

    GraphBuffer light_resource{};
    if (!local_lights.empty() &&
        !graph.import_buffer(light_buffer.handle(), 0, light_buffer.size(), BufferUse::HostWrite,
            light_resource, error))
        return false;

    GraphBuffer local_rays_resource{}, local_hits_resource{};
    const bool use_local_shadow_rays = local_shadows_ &&
        std::any_of(local_lights.begin(), local_lights.end(), [](const IndirectLocalLight& light) {
            return light.color_type[3] >= IndirectLocalLightShadowFlag;
        });
    if (use_local_shadow_rays &&
        (!import_device_buffer(local_rays_[frame.image_index], local_rays_resource) ||
         !import_device_buffer(local_hits_[frame.image_index], local_hits_resource)))
        return false;

    FrameImageGraph::Resource depth_resource{}, normal_resource{}, motion_resource{};
    const bool have_primary_images = !depth_images_.empty() && !normal_images_.empty();
    if (have_primary_images)
    {
        const VkImageSubresourceRange depth_range{VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        const VkImageSubresourceRange normal_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!image_states.register_image(depth_images_[frame.image_index], depth_range,
                ImageUse::SampledGraphicsCompute, error) ||
            !image_states.register_image(normal_images_[frame.image_index], normal_range,
                ImageUse::SampledGraphicsCompute, error) ||
            !graph.import_image(depth_images_[frame.image_index], depth_range, depth_resource, error,
                FrameSubmissionGraph::Queue::Graphics, frame.previous_graphics_submission,
                sharing_.mode, sharing_.queue_families, ImageUse::SampledGraphicsCompute,
                frame.graphics_prepass_pass) ||
            !graph.import_image(normal_images_[frame.image_index], normal_range, normal_resource, error,
                FrameSubmissionGraph::Queue::Graphics, frame.previous_graphics_submission,
                sharing_.mode, sharing_.queue_families, ImageUse::SampledGraphicsCompute,
                frame.graphics_prepass_pass))
            return false;
    }
    if (motion_image != VK_NULL_HANDLE)
    {
        const VkImageSubresourceRange motion_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!image_states.register_image(motion_image, motion_range,
                ImageUse::SampledGraphicsCompute, error) ||
            !graph.import_image(motion_image, motion_range, motion_resource, error,
                FrameSubmissionGraph::Queue::Graphics, frame.previous_graphics_submission,
                sharing_.mode, sharing_.queue_families, ImageUse::SampledGraphicsCompute,
                frame.geometry_motion_submission_pass))
            return false;
    }
    const bool material_inputs_valid = !albedo_views_.empty() && !surface_views_.empty() &&
        !albedo_images_.empty() && !surface_images_.empty();
    FrameImageGraph::Resource albedo_resource{}, gbuffer_surface_resource{};
    if (material_inputs_valid)
    {
        const VkImageSubresourceRange color_range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        if (!image_states.register_image(albedo_images_[frame.image_index], color_range,
                ImageUse::SampledGraphicsCompute, error) ||
            !image_states.register_image(surface_images_[frame.image_index], color_range,
                ImageUse::SampledGraphicsCompute, error) ||
            !graph.import_image(albedo_images_[frame.image_index], color_range, albedo_resource, error,
                FrameSubmissionGraph::Queue::Graphics, frame.previous_graphics_submission,
                sharing_.mode, sharing_.queue_families, ImageUse::SampledGraphicsCompute,
                frame.graphics_prepass_pass) ||
            !graph.import_image(surface_images_[frame.image_index], color_range,
                gbuffer_surface_resource, error, FrameSubmissionGraph::Queue::Graphics,
                frame.previous_graphics_submission, sharing_.mode, sharing_.queue_families,
                ImageUse::SampledGraphicsCompute, frame.graphics_prepass_pass))
            return false;
    }

    const auto primary = graph.add_checked_pass("indirect primary ray generation", [&](std::string& pass_error) {
        if (!generator_.record(reduced_frame, inverse_view_projection, constants))
        { pass_error = "indirect primary ray generation failed"; return false; }
        return true;
    });
    if (have_primary_images &&
        (!graph.read(primary, depth_resource, ImageUse::ComputeSampled, error) ||
         !graph.read(primary, normal_resource, ImageUse::ComputeSampled, error)))
        return false;
    GraphBuffer generated_rays, generated_primary_surfaces;
    if (!graph.write_buffer(primary, rays_resource, BufferUse::ComputeWrite, generated_rays, error) ||
        !graph.write_buffer(primary, primary_surfaces_resource, BufferUse::ComputeWrite,
            generated_primary_surfaces, error))
        return false;

    const auto trace_primary = graph.add_checked_pass("indirect primary ray query", [&](std::string& pass_error) {
        return queries_.record(recording_frame.command_buffer, frame.frame_index, scene, rays,
            hits, count, pass_error);
    });
    if (!graph.read_buffer(trace_primary, generated_rays, BufferUse::ComputeRead, error))
        return false;
    GraphBuffer primary_hits;
    if (!graph.write_buffer(trace_primary, hits_resource, BufferUse::ComputeWrite, primary_hits, error))
        return false;

    const auto reconstruct = graph.add_checked_pass("indirect hit surface reconstruction", [&](std::string& pass_error) {
        return surface_pass.record(recording_frame.command_buffer, frame.frame_index, scene, rays, hits,
            surfaces_[frame.image_index], count, pass_error);
    });
    if (!graph.read_buffer(reconstruct, generated_rays, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(reconstruct, primary_hits, BufferUse::ComputeRead, error))
        return false;
    GraphBuffer reconstructed_surfaces;
    if (!graph.write_buffer(reconstruct, surfaces_resource, BufferUse::ComputeWrite,
            reconstructed_surfaces, error))
        return false;

    const auto generate_sun = graph.add_checked_pass("indirect sun ray generation", [&](std::string& pass_error) {
        return sun_generator_.record(recording_frame.command_buffer, frame.frame_index,
            surfaces_[frame.image_index], sun_rays_[frame.image_index], count,
            sun_direction_distance, pass_error);
    });
    if (!graph.read_buffer(generate_sun, reconstructed_surfaces, BufferUse::ComputeRead, error))
        return false;
    GraphBuffer generated_sun_rays;
    if (!graph.write_buffer(generate_sun, sun_rays_resource, BufferUse::ComputeWrite,
            generated_sun_rays, error))
        return false;

    const auto trace_sun = graph.add_checked_pass("indirect sun ray query", [&](std::string& pass_error) {
        return sun_queries_.record(recording_frame.command_buffer, frame.frame_index, scene,
            sun_rays_[frame.image_index], sun_hits_[frame.image_index], count, pass_error);
    });
    if (!graph.read_buffer(trace_sun, generated_sun_rays, BufferUse::ComputeRead, error))
        return false;
    GraphBuffer traced_sun_hits;
    if (!graph.write_buffer(trace_sun, sun_hits_resource, BufferUse::ComputeWrite,
            traced_sun_hits, error))
        return false;

    GraphBuffer traced_local_hits{};
    if (use_local_shadow_rays)
    {
        const uint32_t local_parameters[]{count, parameters.groups_x, parameters.light_count, seed};
        const std::vector<VkDescriptorBufferInfo> local_buffers{info(surfaces_[frame.image_index]),
            info(light_buffer), info(local_rays_[frame.image_index])};
        const auto generate_local = graph.add_checked_pass("indirect local ray generation",
            [&, local_buffers, local_parameters](std::string& pass_error) {
            return local_generator_.record(recording_frame.command_buffer, frame.frame_index, local_buffers,
                local_parameters, parameters.groups_x,
                (groups + parameters.groups_x - 1) / parameters.groups_x, pass_error);
        });
        if (!graph.read_buffer(generate_local, reconstructed_surfaces, BufferUse::ComputeRead, error) ||
            !graph.read_buffer(generate_local, light_resource, BufferUse::ComputeRead, error))
            return false;
        GraphBuffer generated_local_rays;
        if (!graph.write_buffer(generate_local, local_rays_resource, BufferUse::ComputeWrite,
                generated_local_rays, error))
            return false;

        const auto trace_local = graph.add_checked_pass("indirect local ray query", [&](std::string& pass_error) {
            return local_queries_.record(recording_frame.command_buffer, frame.frame_index, scene,
                local_rays_[frame.image_index], local_hits_[frame.image_index], count, pass_error);
        });
        if (!graph.read_buffer(trace_local, generated_local_rays, BufferUse::ComputeRead, error) ||
            !graph.write_buffer(trace_local, local_hits_resource, BufferUse::ComputeWrite,
                traced_local_hits, error))
            return false;
    }

    const auto evaluate_radiance = graph.add_checked_pass("indirect radiance evaluation", [&](std::string& pass_error) {
        if (!radiance_pass_.record(recording_frame.command_buffer, frame.frame_index, buffers, &parameters,
                parameters.groups_x, (groups + parameters.groups_x - 1) / parameters.groups_x,
                pass_error, projection_set))
        { if (pass_error.empty()) pass_error = "indirect radiance evaluation failed"; return false; }
        return true;
    });
    if (!graph.read_buffer(evaluate_radiance, generated_rays, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(evaluate_radiance, reconstructed_surfaces, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(evaluate_radiance, generated_sun_rays, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(evaluate_radiance, traced_sun_hits, BufferUse::ComputeRead, error) ||
        (!local_lights.empty() && !graph.read_buffer(evaluate_radiance, light_resource,
            BufferUse::ComputeRead, error)) ||
        (use_local_shadow_rays && !graph.read_buffer(evaluate_radiance, traced_local_hits,
            BufferUse::ComputeRead, error)) ||
        !graph.read_buffer(evaluate_radiance, world_cache_resource, BufferUse::ComputeRead, error))
        return false;
    GraphBuffer evaluated_radiance;
    if (!graph.write_buffer(evaluate_radiance, radiance_resource, BufferUse::ComputeWrite,
            evaluated_radiance, error))
        return false;

    ReferencePathTraceParameters path_trace_parameters{};
    std::memcpy(path_trace_parameters.sky, sky_color, sizeof(float) * 3);
    path_trace_parameters.sky[3] = static_cast<float>(extent_.width);
    std::memcpy(path_trace_parameters.sun_direction_distance, sun_direction_distance, sizeof(float) * 4);
    std::memcpy(path_trace_parameters.sun_color, sun_color, sizeof(float) * 3);
    path_trace_parameters.dimensions[0] = count;
    path_trace_parameters.dimensions[1] = parameters.groups_x;
    path_trace_parameters.dimensions[2] = alpha_texture_capacity_;
    path_trace_parameters.dimensions[3] = seed;
    path_trace_parameters.control[0] = 2;
    path_trace_parameters.control[1] = path_trace_history_valid ? 1u : 0u;
    path_trace_parameters.control[2] = 64;
    path_trace_parameters.control[3] = static_cast<uint32_t>(local_lights.size());
    path_trace_parameters.limits[0] = max_distance;
    path_trace_parameters.limits[1] = 0.01f;
    path_trace_parameters.limits[2] = 0.1f;
    path_trace_parameters.limits[3] = static_cast<float>(projection_tables_.empty() ? 0u : projections.size());
    FrameImageGraph::BufferResource path_traced_radiance = evaluated_radiance;
    if (path_trace_enabled_)
    {
        const auto path_trace = graph.add_checked_pass("diffuse reference path trace", [&](std::string& pass_error) {
            return reference_path_trace_pass_.record(recording_frame.command_buffer, frame.frame_index, scene,
                rays, surfaces_[frame.image_index], path_trace_history_[reservoir_input_index],
                path_trace_history_[reservoir_output_index_], light_buffer, projection_set,
                motion_view, projection_sampler, radiance_[frame.image_index], path_trace_parameters, pass_error);
        });
        FrameImageGraph::BufferResource written_path_trace_history;
        if (!graph.read_buffer(path_trace, generated_rays, BufferUse::ComputeRead, error) ||
            !graph.read_buffer(path_trace, reconstructed_surfaces, BufferUse::ComputeRead, error) ||
            !(path_trace_history_valid ?
                graph.read_buffer(path_trace, previous_path_trace_history_resource,
                    BufferUse::ComputeRead, error) : true) ||
            !graph.write_buffer(path_trace, next_path_trace_history_resource,
                BufferUse::ComputeWrite, written_path_trace_history, error) ||
            (path_trace_history_valid && !graph.read(path_trace, motion_resource,
                ImageUse::SampledGraphicsCompute, error)) ||
            (!local_lights.empty() && !graph.read_buffer(path_trace, light_resource,
                BufferUse::ComputeRead, error)) ||
            !graph.read_write_buffer(path_trace, evaluated_radiance, BufferUse::ComputeReadWrite,
                path_traced_radiance, error))
            return false;
    }

    const uint32_t cache_constants[]{count, parameters.groups_x, seed, 0u};
    const std::vector<VkDescriptorBufferInfo> cache_buffers{info(surfaces_[frame.image_index]),
        info(radiance_[frame.image_index]), info(rays), info(world_radiance_cache_)};
    const auto update_world_cache = graph.add_checked_pass("update world radiance cache", [&](std::string& pass_error) {
        return world_cache_update_pass_.record(recording_frame.command_buffer, frame.frame_index, cache_buffers,
            cache_constants, parameters.groups_x, (groups + parameters.groups_x - 1) / parameters.groups_x,
            pass_error);
    });
    FrameImageGraph::BufferResource next_world_cache;
    if (!graph.read_buffer(update_world_cache, reconstructed_surfaces, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(update_world_cache, generated_rays, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(update_world_cache, path_traced_radiance, BufferUse::ComputeRead, error) ||
        !graph.read_write_buffer(update_world_cache, world_cache_resource, BufferUse::ComputeReadWrite,
            next_world_cache, error))
        return false;

    const uint32_t spatial_constants[]{extent_.width, extent_.height, seed,
        temporal_history_valid && reservoir_history_valid_ ? (motion_has_previous_depth ? 3u : 1u) : 0u,
        material_inputs_valid ? 1u : 0u};
    const std::vector<VkDescriptorBufferInfo> spatial_buffers{info(radiance_[frame.image_index]),
        info(primary_surfaces_[frame.image_index]), info(rays), info(reservoir_history_[reservoir_input_index]),
        info(reservoir_history_[reservoir_output_index_]), info(restir_radiance_[frame.image_index])};
    const VkImageView fallback_view = normal_views_[frame.image_index];
    const std::vector<VkDescriptorImageInfo> spatial_images{
        {projection_sampler, material_inputs_valid ? albedo_views_[frame.image_index] : fallback_view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {projection_sampler, fallback_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {projection_sampler, material_inputs_valid ? surface_views_[frame.image_index] : fallback_view,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {projection_sampler, motion_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    const auto resample = graph.add_checked_pass("spatiotemporal ReSTIR GI", [&](std::string& pass_error) {
        return restir_spatial_pass_.record_mixed(recording_frame.command_buffer, frame.frame_index, spatial_buffers,
            spatial_images, spatial_constants, (extent_.width + 7) / 8,
            (extent_.height + 7) / 8, pass_error);
    });
    if (!graph.read_buffer(resample, path_traced_radiance, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(resample, generated_primary_surfaces, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(resample, generated_rays, BufferUse::ComputeRead, error) ||
        !graph.read_buffer(resample, previous_reservoir_resource, BufferUse::ComputeRead, error))
        return false;
    FrameImageGraph::BufferResource next_reservoir, next_radiance;
    if (!graph.write_buffer(resample, output_reservoir_resource, BufferUse::ComputeWrite,
            next_reservoir, error) ||
        !graph.write_buffer(resample, resolved_radiance_resource, BufferUse::ComputeWrite,
            next_radiance, error))
        return false;
    if (motion_image != VK_NULL_HANDLE &&
        !graph.read(resample, motion_resource, ImageUse::ComputeSampled, error))
        return false;
    if (material_inputs_valid &&
        (!graph.read(resample, albedo_resource, ImageUse::ComputeSampled, error) ||
         !graph.read(resample, normal_resource, ImageUse::ComputeSampled, error) ||
         !graph.read(resample, gbuffer_surface_resource, ImageUse::ComputeSampled, error)))
        return false;
    if (!output_layouts_.empty())
    {
        const auto resolve = graph.add_checked_pass("resolve indirect radiance", [&](std::string& pass_error) {
            if (!image_resolve_.record(reduced_frame, output_layouts_[frame.image_index]))
            {
                pass_error = "indirect radiance image resolve failed";
                return false;
            }
            return true;
        });
        if (!graph.read_buffer(resolve, next_radiance, BufferUse::ComputeRead, error))
            return false;
    }
    if (use_generated_compute)
    {
        std::vector<FrameImageGraph::QueueGroup> groups;
        std::vector<FrameSubmissionGraph::Pass> submission_for_pass;
        const auto queue_family = [&](FrameSubmissionGraph::Queue queue) {
            if (queue == FrameSubmissionGraph::Queue::Compute) return frame.compute_queue_family;
            if (queue == FrameSubmissionGraph::Queue::Copy) return frame.copy_queue_family;
            return frame.graphics_queue_family;
        };
        if (!graph.generate_submission_groups(*frame.submission_graph, queue_family,
                groups, submission_for_pass, error))
            return false;
        FrameImageGraph::ExecutionState execution;
        if (!graph.begin_execution(execution, error))
            return false;
        const ImageStateDispatch state_dispatch{cmd_pipeline_barrier_};
        for (const FrameImageGraph::QueueGroup& group : groups)
        {
            if (group.queue != FrameSubmissionGraph::Queue::Compute || group.passes.empty() ||
                group.submission == FrameSubmissionGraph::InvalidPass ||
                !frame.submission_graph->depends_on(group.submission, frame.graphics_prepass_pass, error) ||
                !frame.submission_graph->depends_on(frame.graphics_continuation_pass, group.submission, error))
                return false;
            VkCommandBuffer group_command = VK_NULL_HANDLE;
            if (!frame.request_submission_command_buffer(group.submission, group.queue, group_command, error))
                return false;
            recording_frame.command_buffer = group_command;
            recording_frame.command_queue = group.queue;
            recording_frame.command_queue_flags = frame.compute_queue_flags;
            reduced_frame.command_buffer = group_command;
            reduced_frame.command_queue = group.queue;
            reduced_frame.command_queue_flags = frame.compute_queue_flags;
            if (!graph.execute_group(group, group_command, image_states, execution, state_dispatch, error,
                    frame.compute_queue_flags, frame.semaphore_handoff))
                return false;
        }
    }
    else if (!graph.execute(recording_frame.command_buffer, image_states, {cmd_pipeline_barrier_}, error,
                 frame.command_queue_flags, frame.semaphore_handoff, frame.command_queue))
        return false;
    error.clear();
    return true;
}

void IndirectRayFrames::destroy()
{
    generator_.destroy();
    queries_.destroy();
    sun_generator_.destroy();
    sun_queries_.destroy();
    local_generator_.destroy();
    local_queries_.destroy();
    radiance_pass_.destroy();
    reference_path_trace_pass_.destroy();
    world_cache_update_pass_.destroy();
    restir_spatial_pass_.destroy();
    projection_tables_.clear();
    release_output();
    rays_.clear();
    hits_.clear();
    surfaces_.clear();
    primary_surfaces_.clear();
    sun_rays_.clear();
    sun_hits_.clear();
    local_rays_.clear();
    local_hits_.clear();
    radiance_.clear();
    restir_radiance_.clear();
    reservoir_history_.clear();
    for (auto& history : path_trace_history_) history.destroy();
    path_trace_history_.clear();
    world_radiance_cache_.destroy();
    local_lights_.clear();
    depth_images_.clear();
    normal_images_.clear();
    albedo_images_.clear();
    surface_images_.clear();
    normal_views_.clear();
    albedo_views_.clear();
    surface_views_.clear();
    cmd_pipeline_barrier_ = nullptr;
    cmd_fill_buffer_ = nullptr;
    extent_ = {};
    local_shadows_ = false;
    path_trace_enabled_ = false;
    path_trace_history_reused_ = false;
    reservoir_history_index_ = reservoir_output_index_ = 0;
    alpha_texture_capacity_ = 0;
    reservoir_history_valid_ = false;
    sharing_ = {};
    world_cache_clear_pending_ = true;
    world_cache_signature_ = 0;
}

void IndirectRayFrames::release_output()
{
    image_resolve_.destroy();
    if (output_device_ && destroy_output_view_ && output_pool_)
        for (VkImageView view : output_views_) if (view) destroy_output_view_(output_device_, view, nullptr);
    if (output_pool_) for (VkImage image : owned_images_) if (image) output_pool_->release(image);
    output_layouts_.clear(); output_views_.clear(); owned_images_.clear();
    output_device_ = VK_NULL_HANDLE; destroy_output_view_ = nullptr; output_pool_ = nullptr;
}
}
