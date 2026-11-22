#pragma once

namespace xray::render::vulkan
{
enum class ParticleGroupEntryPhase { Waiting, Emitting, Finished };

inline ParticleGroupEntryPhase advance_particle_group_entry(ParticleGroupEntryPhase phase,
    float next_time, float begin, float end)
{
    // Starting takes precedence when a single frame crosses both boundaries.
    // Stop on the following update so the effect can simulate in the start frame.
    if (phase == ParticleGroupEntryPhase::Waiting && next_time >= begin)
        return ParticleGroupEntryPhase::Emitting;
    if (phase == ParticleGroupEntryPhase::Emitting && next_time >= end)
        return ParticleGroupEntryPhase::Finished;
    return phase;
}
}
