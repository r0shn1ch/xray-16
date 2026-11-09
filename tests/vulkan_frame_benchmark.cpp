#include "src/Layers/xrRenderVK/VulkanFrameBenchmark.h"
#include <cassert>
#include <sstream>

using namespace xray::render::vulkan;

int main()
{
    uint32_t frames = 777;
    for (const char* text : {"", "0", "36001", "-1", "-4294967295", "4294967297", "+2", "2x", " 2", "2 "})
    {
        assert(!VulkanFrameBenchmark::parse_frame_count(text, frames));
        assert(frames == 777); // Invalid input never changes the requested count.
    }
    assert(VulkanFrameBenchmark::parse_frame_count("1", frames) && frames == 1);
    assert(VulkanFrameBenchmark::parse_frame_count("000120", frames) && frames == 120);
    assert(VulkanFrameBenchmark::parse_frame_count("36000", frames) && frames == 36000);
    VulkanFrameBenchmark benchmark;
    VulkanFrameSample sample{99999., 5., 9000., 100, 200, 3};
    assert(!benchmark.arm(0, 2) && !benchmark.arm(36001, 2));
    assert(benchmark.arm(4, 2) && !benchmark.arm(10, 2));
    // Startup/loading/menu time, including an unpresented frame, is not a game
    // interval. The first eligible completed frame primes the wall clock.
    assert(!benchmark.frame(10., sample, false, true));
    assert(!benchmark.frame(2000., sample, true, false));
    assert(!benchmark.recording() && benchmark.metrics().frame_count() == 0);
    assert(!benchmark.frame(10000., sample, true, true) && benchmark.recording());
    assert(!benchmark.frame(10010., sample, true, true));
    assert(!benchmark.frame(10030., sample, true, true));
    sample.gpu_ms = 3.;
    assert(!benchmark.frame(10060., sample, true, true));
    // Real long frames must remain visible in p99, rather than being dropped
    // as apparent loading/background gaps based on an arbitrary duration cap.
    assert(benchmark.frame(12060., sample, true, true));
    assert(benchmark.complete() && !benchmark.active());
    const auto stats = benchmark.metrics().summary();
    assert(stats.frames == 4 && stats.frame_p50_ms == 20. && stats.frame_p99_ms == 2000.);
    assert(stats.cpu_p50_ms == 5. && stats.gpu_p99_ms.value() == 3.);
    assert(stats.gpu_unavailable_frames == 2); // Reject pre-window GPU queries.
    assert(!benchmark.frame(99999., sample, true, true) && benchmark.metrics().frame_count() == 4);
    std::ostringstream json;
    assert(benchmark.write_json(json));
    assert(json.str().find("\"complete\": true") != std::string::npos);
    assert(json.str().find("\"requested_frames\": 4") != std::string::npos);

    // Menu/console/pause, a failed present, or explicit reset/deactivation must
    // preserve partial evidence with complete=false, never silently resume it.
    for (int interruption = 0; interruption < 3; ++interruption)
    {
        assert(benchmark.arm(10, 2));
        assert(!benchmark.frame(0., sample, true, true));
        assert(!benchmark.frame(16., sample, true, true));
        if (interruption == 0) assert(benchmark.frame(32., sample, false, true));
        if (interruption == 1) assert(benchmark.frame(32., sample, true, false));
        if (interruption == 2) assert(benchmark.interrupt("surface reset"));
        assert(!benchmark.active() && !benchmark.complete());
        assert(benchmark.metrics().frame_count() == 1);
        assert(!benchmark.frame(9000., sample, true, true));
        json.str(""); json.clear();
        assert(benchmark.write_json(json));
        assert(json.str().find("\"complete\": false") != std::string::npos);
        assert(json.str().find("\"interruption\": \"\"") == std::string::npos);
    }
    assert(benchmark.arm(10, 2));
    assert(benchmark.interrupt("cancelled while armed"));
    assert(benchmark.metrics().frame_count() == 0);
    assert(benchmark.arm(1, 2));
    assert(!benchmark.frame(10., sample, true, true));
    assert(benchmark.frame(9., sample, true, true));
    assert(!benchmark.complete() && benchmark.metrics().frame_count() == 0);
}
