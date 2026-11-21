#pragma once
#include "GpuParticleStaticCollision.h"

namespace xray::render::vulkan
{
// Caches immutable CDB geometry for the current game level. The returned data
// stays owned by this object and is rebuilt only when the CDB model changes.
class GpuParticleLevelCollision
{
public:
    bool update(bool& changed, std::string& error);
    bool update_dynamic(bool enabled, bool& changed, std::string& error);
    const GpuParticleCollisionInput& data() const { return data_; }
    void clear();
private:
    const void *level_{}, *model_{}, *triangles_{}, *vertices_{};
    uint32_t triangle_count_{}, vertex_count_{};
    GpuParticleCollisionInput data_;
};
}
