#pragma once

#if defined(XR_PLATFORM_ANDROID)

#include <cstdint>

// Tagged SDL_MOUSEMOTION used to carry a normalized absolute tap position.
// The high bit is outside SDL's mouse-button state mask.
constexpr std::uint32_t AndroidTouchMousePositionState = 1u << 31;

// Read by the native input thread. Java only updates this atomic bit mask;
// action binding lookup and receiver callbacks stay on the engine thread.
std::uint32_t AndroidTouchControlMask() noexcept;
// Consume presses that may have been released between two engine frames.
std::uint32_t AndroidTouchPressMask() noexcept;

#endif
