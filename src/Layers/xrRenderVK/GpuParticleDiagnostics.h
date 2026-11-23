#pragma once

#include <cstdint>

namespace xray::render::vulkan
{
// Throttle repeated capacity diagnostics while immediately reporting new bits.
class GpuParticleDiagnostics
{
public:
    bool should_report(uint32_t flags, uint32_t now_ms)
    {
        if (!flags) return false;
        if (!reported_ || (flags & ~last_flags_) || uint32_t(now_ms - last_ms_) >= 5000u)
        {
            reported_ = true;
            last_flags_ = flags;
            last_ms_ = now_ms;
            return true;
        }
        return false;
    }
    void reset() { reported_ = false; last_flags_ = last_ms_ = 0; }
private:
    bool reported_{};
    uint32_t last_flags_{}, last_ms_{};
};
}
