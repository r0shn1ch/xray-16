#pragma once

#include "GameShaderResources.h"
#include <memory>

namespace xray::render::vulkan
{
// Runtime HLSL-to-SPIR-V compilation for game shaders that do not have a
// matching precompiled module. Includes are resolved through the same game
// shader VFS used by the renderer.
class RuntimeSlangCompiler
{
  public:
    RuntimeSlangCompiler();
    ~RuntimeSlangCompiler();
    RuntimeSlangCompiler(const RuntimeSlangCompiler &) = delete;
    RuntimeSlangCompiler &operator=(const RuntimeSlangCompiler &) = delete;

    bool compile(const std::string &path, GameShaderStage stage, const char *entry, uint32_t flags,
        const GameShaderResources::Source &source, const uint8_t *source_override, size_t source_override_size,
        const std::vector<std::string> &defines, std::vector<uint8_t> &spirv, std::string &error);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace xray::render::vulkan
