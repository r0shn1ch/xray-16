#pragma once
#include "StorageComputePass.h"
#include <array>
#include <functional>

namespace xray::render::vulkan
{
// Owns the three upstream PAPI compute pipelines. The caller owns all buffers
// and supplies one descriptor-set slot per in-flight dispatch of each pass.
class GpuParticleCompute
{
public:
    struct Shader { const uint32_t* code{}; size_t bytes{}; };
    enum class Pass { Initialize, Simulate, Compact, CaptureMotion };
    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, const std::array<Shader, 4>& shaders, std::string& error);
    using Source = std::function<bool(const std::string&, std::vector<uint8_t>&, std::string&)>;
    bool initialize_from_source(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceLimits& limits,
        uint32_t slots, VkPipelineCache cache, const Source& source, std::string& error);
    bool record(Pass pass, VkCommandBuffer command, uint32_t slot,
        const std::vector<VkDescriptorBufferInfo>& buffers, uint32_t groups, std::string& error) const;
    void destroy();
private:
    StorageComputePass init_, simulate_, compact_, capture_motion_;
    uint32_t max_groups_{};
};
}
