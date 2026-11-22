#include "stdafx.h"
#include "GrassInteractionCollector.h"
#include "GrassInteractionState.h"

namespace xray::grass
{
namespace { State& state() { static State value; return value; } }
void begin_frame(uint32_t frame, uint32_t time_ms, const float* focus) { state().begin(frame,time_ms,focus); }
void submit(const Sample& sample) { state().submit(sample); }
void publish_frame() { state().publish(); }
void forget(const void* instance) { state().forget(instance); }
void reset() { state().reset(); }
Frame snapshot() { return state().snapshot(); }
}
