#pragma once
#include "Engine.h"
#include "GrassInteractionTypes.h"

namespace xray::grass
{
ENGINE_API void begin_frame(uint32_t frame, uint32_t time_ms, const float* focus);
ENGINE_API void submit(const Sample& sample);
ENGINE_API void publish_frame();
ENGINE_API void forget(const void* instance);
ENGINE_API void reset();
ENGINE_API Frame snapshot();
}
