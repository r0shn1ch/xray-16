#include <algorithm>
#include <cassert>
#include <cmath>
#include <glm/glm.hpp>

using glm::clamp;
using glm::dot;
using glm::max;
using glm::vec3;

// Compile and exercise the same bright-pass function included by postprocess.frag.
#include "src/Layers/xrRenderVK/smoke/bloom.glsl"

int main()
{
    const float threshold = 0.8f;
    const float knee = 0.15f;
    const vec3 below = bloom_highlights(vec3(0.4f), threshold, knee);
    const vec3 at_knee = bloom_highlights(vec3(0.75f), threshold, knee);
    const vec3 above = bloom_highlights(vec3(0.95f), threshold, knee);
    assert(glm::length(below) < 1e-6f);
    assert(std::isfinite(at_knee.x) && at_knee.x > 0.f);
    assert(above.x > at_knee.x);

    // The highlight extraction scales RGB together, preserving the source hue.
    const vec3 colored = bloom_highlights(vec3(0.95f, 0.85f, 0.7f), threshold, knee);
    assert(colored.x > colored.y && colored.y > colored.z);
    assert(std::isfinite(colored.x) && std::isfinite(colored.y) && std::isfinite(colored.z));
}
