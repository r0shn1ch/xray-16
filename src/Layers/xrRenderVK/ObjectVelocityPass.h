#pragma once

#include "FrameContext.h"
#include <array>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct TreeWindParameters;
struct DetailWindParameters;
// Depth-equal raster overlay for opaque and cutout meshes. LevelVertex draws
// bind current and previous vertices; ModelVertex draws bind two bone poses.
class ObjectVelocityPass
{
public:
    ~ObjectVelocityPass() { destroy(); }
    ObjectVelocityPass() = default;
    ObjectVelocityPass(const ObjectVelocityPass&) = delete;
    ObjectVelocityPass& operator=(const ObjectVelocityPass&) = delete;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, VkExtent2D extent,
        VkFormat depth_format, const std::vector<VkImageView>& velocity_views,
        const std::vector<VkImageView>& depth_views, VkPipelineCache cache, std::string& error,
        VkDescriptorSetLayout material_layout = VK_NULL_HANDLE,
        VkDescriptorSetLayout pose_layout = VK_NULL_HANDLE, uint32_t max_draw_indirect_count = 1);
    bool begin(const FrameRecordingContext& frame) const;
    void draw(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer previous_vertices, VkBuffer indices,
        uint32_t index_count, uint32_t first_index, int32_t vertex_offset, const float* current_mvp,
        const float* previous_mvp, VkDescriptorSet material = VK_NULL_HANDLE,
        float alpha_ref = 0.5f, const float* base_rows = nullptr,
        const TreeWindParameters* current_wind = nullptr, const TreeWindParameters* previous_wind = nullptr,
        VkBuffer indirect_commands = VK_NULL_HANDLE, VkDeviceSize indirect_offset = 0,
        uint32_t indirect_draw_count = 0) const;
    void draw_skinned(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer indices,
        uint32_t index_count, uint32_t first_index, const float* current_mvp,
        const float* previous_mvp, VkDescriptorSet current_pose, VkDescriptorSet previous_pose,
        VkDescriptorSet material = VK_NULL_HANDLE, float alpha_ref = 0.5f,
        const float* base_rows = nullptr) const;
    bool draw_details(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer indices, uint32_t index_count,
        VkBuffer instances, uint32_t instance_count, const float* current_mvp, const float* previous_mvp,
        const DetailWindParameters& current_wind, const DetailWindParameters& previous_wind,
        VkDescriptorSet material = VK_NULL_HANDLE, float alpha_ref = .5f, const float* base_rows = nullptr,
        VkBuffer indirect_commands = VK_NULL_HANDLE, bool transparent = false,
        VkDeviceSize instance_offset = 0) const;
    void draw_lod(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer indices,
        uint32_t index_count, uint32_t first_index, const float* mvp,
        VkDescriptorSet material, float alpha_ref, float blend) const;
    void draw_particle_velocity(const FrameRecordingContext& frame, VkBuffer vertices, VkBuffer previous_vertices,
        VkBuffer indices, uint32_t index_count, const float* current_mvp, const float* previous_mvp,
        VkDescriptorSet material, float alpha_ref, const float* base_rows, bool transparent = false) const;
    void end(const FrameRecordingContext& frame) const;
    bool ready() const { return pipeline_ != VK_NULL_HANDLE; }
    bool lod_ready() const { return lod_pipeline_ != VK_NULL_HANDLE; }
    VkRenderPass render_pass() const { return pass_; }
    void destroy();

private:
    VkDevice device_{};
    VkExtent2D extent_{};
    VkRenderPass pass_{};
    VkPipelineLayout layout_{};
    VkPipeline pipeline_{};
    VkPipeline cutout_pipeline_{};
    VkPipeline detail_pipeline_{}, detail_cutout_pipeline_{};
    VkPipeline detail_transparent_pipeline_{};
    VkPipeline lod_pipeline_{};
    VkPipeline particle_velocity_pipeline_{};
    VkPipeline particle_transparent_velocity_pipeline_{};
    VkPipeline skinned_pipeline_{};
    VkPipeline skinned_cutout_pipeline_{};
    std::vector<VkFramebuffer> framebuffers_;
    PFN_vkDestroyRenderPass destroy_pass_{};
    PFN_vkDestroyFramebuffer destroy_framebuffer_{};
    PFN_vkDestroyPipelineLayout destroy_layout_{};
    PFN_vkDestroyPipeline destroy_pipeline_{};
    PFN_vkCmdBeginRenderPass begin_pass_{};
    PFN_vkCmdEndRenderPass end_pass_{};
    PFN_vkCmdBindPipeline bind_pipeline_{};
    PFN_vkCmdBindDescriptorSets bind_sets_{};
    PFN_vkCmdSetViewport set_viewport_{};
    PFN_vkCmdSetScissor set_scissor_{};
    PFN_vkCmdBindVertexBuffers bind_vertices_{};
    PFN_vkCmdBindIndexBuffer bind_indices_{};
    PFN_vkCmdPushConstants push_constants_{};
    PFN_vkCmdDrawIndexed draw_indexed_{};
    PFN_vkCmdDrawIndexedIndirect draw_indexed_indirect_{};
    uint32_t max_draw_indirect_count_{1};
};
}
