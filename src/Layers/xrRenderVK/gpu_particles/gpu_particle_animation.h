#ifndef GPU_PARTICLE_ANIMATION_H
#define GPU_PARTICLE_ANIMATION_H
#include "gpu_particle_types.h"
void GpuAnimateParticle(inout GpuPapiParticle particle,GpuPapiProgram program) {
    if (program.frameCount == 0) return;
    float frame = float(particle.frame) / 255.0 + ((particle.flags & 1u) != 0 ? -1.0 : 1.0) * program.frameSpeed * 0.033;
    frame = fmod(frame,float(program.frameCount));
    if (frame < 0) frame += float(program.frameCount);
    particle.frame = uint(int(floor(frame * 255.0))) & 65535u;
}
#endif
