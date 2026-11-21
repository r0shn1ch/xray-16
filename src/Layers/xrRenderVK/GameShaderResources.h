#pragma once

#include "ShaderModule.h"
#include "ShaderReflection.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace xray::render::vulkan
{
class DeferredPass;
class RuntimeSlangCompiler;
enum class SurfaceMode : uint8_t;
enum class GameShaderStage : uint32_t
{
    Vertex = 0,
    Hull = 1,
    Domain = 2,
    Geometry = 3,
    Pixel = 4,
    Compute = 5
};

// Vulkan counterparts of legacy shader resources. A returned resource is
// borrowed until clear/reload/device teardown.
struct GameShaderResource
{
    std::string name;
    std::string entry_point;
    GameShaderStage stage{};
    uint32_t compile_flags{};
    std::string precompiled_rejection;
    std::vector<uint8_t> runtime_source;
    std::vector<std::string> runtime_defines;
    ShaderModule module;
    ShaderReflection reflection;
    uint64_t content_fingerprint{};
};

class GameShaderResources
{
  public:
    using Source = std::function<bool(const std::string &, std::vector<uint8_t> &, std::string &)>;
    using RuntimeCompile = std::function<bool(const std::string &, GameShaderStage, const char *, uint32_t,
        const Source &, const uint8_t *, size_t, const std::vector<std::string> &,
        std::vector<uint8_t> &, std::string &)>;
    GameShaderResources();
    ~GameShaderResources();
    void configure(VkDevice device, const ShaderModuleDispatch &dispatch, Source source,
        bool geometry_supported = false, bool tessellation_supported = false, Source bundled_source = {},
        RuntimeCompile runtime_compile = {});
    void set_engine_defines(const std::vector<std::string>& defines)
    {
        engine_defines_ = defines;
    }
    static bool merge_defines(const std::vector<std::string>& base,
        const std::vector<std::string>& overrides, std::vector<std::string>& result,
        std::string& error);
    bool pipeline_defines(const std::string& name, GameShaderStage stage, const char* entry,
        const std::vector<std::string>& overrides, std::vector<std::string>& result,
        std::string& error) const;
    bool shader(const std::string &name, GameShaderStage stage, const char *entry, uint32_t flags,
        GameShaderResource *&result, std::string &error, const uint8_t *source_override = nullptr,
        size_t source_override_size = 0, bool force_runtime_compile = false,
        const std::vector<std::string> &defines = {});
    bool compute_pipeline(const std::string &name, VkPipelineCache cache, VkPipelineLayout layout,
        PFN_vkCreateComputePipelines create, PFN_vkDestroyPipeline destroy,
        VkPipeline &result, std::string &error,
        const VkSpecializationInfo *specialization = nullptr);
    bool pipeline(DeferredPass &pass, const std::string &svs, const std::string &sps, SurfaceMode mode,
        bool hud, bool skinned, std::string &error, const std::string &shs = {},
        const std::string &sds = {}, const std::string &sgs = {},
        const VkSpecializationInfo *vertex_specialization = nullptr,
        const VkSpecializationInfo *hull_specialization = nullptr,
        const VkSpecializationInfo *domain_specialization = nullptr,
        const VkSpecializationInfo *geometry_specialization = nullptr,
        const VkSpecializationInfo *pixel_specialization = nullptr,
        const std::vector<std::string> &defines = {}, uint64_t *variant_hash = nullptr);
    void clear()
    {
        resources_.clear();
    }
    void destroy();
    size_t size() const
    {
        return resources_.size();
    }
    static bool canonical_name(const std::string &name, GameShaderStage stage, std::string &result, std::string &error);
    static bool has_manifest_variant(const std::string &name, GameShaderStage stage, const char *entry);
    static bool stage_from_target(const char *target, GameShaderStage &stage);

  private:
    VkDevice device_{};
    ShaderModuleDispatch dispatch_{};
    Source source_;
    Source bundled_source_;
    RuntimeCompile runtime_compile_;
    std::unique_ptr<RuntimeSlangCompiler> runtime_compiler_;
    bool geometry_supported_{};
    bool tessellation_supported_{};
    std::vector<std::string> engine_defines_;
    std::unordered_map<std::string, std::unique_ptr<GameShaderResource>> resources_;
};
} // namespace xray::render::vulkan
