#pragma once
#include "gpu_particles/gpu_particle_types.h"
#include <array>
#include <limits>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
// Retains commands until GPU readback acknowledges their generation and
// serial. A snapshot can therefore be submitted again after a delayed frame.
class GpuParticleCommands
{
public:
    struct Batch { std::vector<fg::GpuPapiCommand> commands; std::vector<std::array<uint32_t, 2>> ranges; };
    explicit GpuParticleCommands(uint32_t roots) : roots_(roots) {}
    bool begin_generation(uint32_t root, std::string& error);
    bool enqueue(uint32_t root, fg::GpuPapiCommand command, std::string& error);
    bool snapshot(Batch& batch, std::string& error);
    bool acknowledge(uint32_t root, uint32_t generation, uint32_t serial, std::string& error);
    bool acknowledge(const std::vector<fg::GpuPapiStatus>& statuses, std::string& error);
    bool state(uint32_t root, uint32_t& generation, uint32_t& serial) const;
private:
    struct Root
    {
        uint32_t generation{}, serial{}, submitted{}, acknowledged{};
        std::vector<fg::GpuPapiCommand> pending;
    };
    std::vector<Root> roots_;
};
}
