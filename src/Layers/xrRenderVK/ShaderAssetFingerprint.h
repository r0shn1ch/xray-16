#pragma once

#include <cstdint>
#include <string>

namespace xray::render::vulkan
{

// Hashes the effective precompiled game-shader set, including an Android
// engine-data overlay while preserving loose game/mod override precedence.
bool shader_asset_fingerprint(uint64_t &result, std::string &error);

}
