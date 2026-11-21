#pragma once
#include "GpuParticlePrograms.h"
#include "ShaderMaterialLibrary.h"
#include <vulkan/vulkan.h>

namespace xray::render::vulkan
{
class GameTextureFactory;
class DeferredPass;
struct ParticleCatalog;
// Owns texture leases independently of individual CPU particle visuals.
// Clear only after GPU idle and before destroying the texture factory/pass.
class GpuParticleMaterials
{
public:
    struct Entry
    {
        VkDescriptorSet set{};
        TextureMatrixSet matrices{};
        std::string name;
        int blend{-1}, alpha_ref{128};
        uint8_t priority{1};
        bool screen{}, distortion{}, resolved{};
    };
    bool sync(GpuParticlePrograms& programs, const ParticleCatalog& catalog,
        GameTextureFactory& textures, DeferredPass& pass, std::string& error);
    const Entry* get(uint32_t program) const { return program < entries_.size() ? &entries_[program] : nullptr; }
    bool fragment(uint32_t program, bool hud, std::string& name, bool& gbuffer, std::string& error) const;
    bool constants(uint32_t program, float time, const float* view, const DeferredPass& pass,
        std::array<float, 64>& output, std::string& error) const;
    void clear(GameTextureFactory& textures, DeferredPass& pass);
private:
    std::vector<Entry> entries_;
};
}
