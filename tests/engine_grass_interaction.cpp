#include "src/xrEngine/GrassInteractionState.h"
#include <atomic>
#include <cassert>
#include <limits>
#include <thread>

using namespace xray::grass;
static Sample sample(uint16_t id, float x, float z=0.f)
{
    Sample result;
    result.instance = reinterpret_cast<const void*>(uintptr_t(id)+1);
    result.object_id = id; result.position[0] = x; result.position[2] = z;
    result.radius = 1.f; result.mass = 100.f;
    return result;
}
int main()
{
    const float focus[3]{};
    State state;
    assert(state.begin(1,100,focus));
    assert(state.submit(sample(1,0.f)));
    assert(state.snapshot().entities.empty()); // Staging never leaks a partial frame.
    state.publish();
    auto frame = state.snapshot();
    assert(frame.frame == 1 && frame.entities.size() == 1 && frame.entities[0].velocity[0] == 0.f);
    const auto revision = frame.revision;
    assert(state.begin(2,120,focus));
    assert(state.submit(sample(1,.1f)) && state.submit(sample(1,.2f)));
    assert(state.snapshot().revision == revision && state.snapshot().entities[0].position[0] == 0.f);
    state.publish();
    frame = state.snapshot();
    assert(frame.entities.size() == 1 && std::abs(frame.entities[0].velocity[0]-10.f) < 1e-5f);
    state.forget(sample(1,0.f).instance);
    assert(state.snapshot().entities.empty());
    state.begin(3,140,focus);
    state.submit(sample(1,7.f)); // A pooled instance with the same network ID is new history.
    state.publish();
    assert(state.snapshot().entities[0].velocity[0] == 0.f);
    state.begin(4,160,focus);
    auto reused = sample(1,8.f); reused.object_id = 2;
    state.submit(reused); state.publish();
    assert(state.snapshot().entities[0].velocity[0] == 0.f);
    state.begin(5,180,focus); reused.position[0] = 100.f;
    state.submit(reused); state.publish();
    assert(state.snapshot().entities[0].velocity[0] == 0.f); // Teleport.
    state.begin(8,240,focus); reused.position[0] = 100.1f;
    state.submit(reused); state.publish();
    assert(state.snapshot().entities[0].velocity[0] == 0.f); // Missing updates.
    state.begin(9,240,focus); reused.position[0] = 100.2f;
    state.submit(reused); state.publish();
    assert(state.snapshot().entities[0].velocity[0] == 0.f); // Paused clock.
    state.begin(10,600,focus); reused.position[0] = 100.3f;
    state.submit(reused); state.publish();
    assert(state.snapshot().entities[0].velocity[0] == 0.f); // Long clock gap.
    state.begin(11,620,focus); state.submit(reused);
    state.forget(reused.instance); state.publish();
    assert(state.snapshot().entities.empty()); // Destroyed during update.

    state.reset(); state.begin(1,100,focus);
    auto small = sample(1,0); small.radius = .01f; small.mass = 1.f;
    auto large = sample(2,1); large.radius = 20.f; large.mass = 1000.f;
    auto unknown_mass = sample(3,2); unknown_mass.mass = 0.f;
    assert(state.submit(small) && state.submit(large) && state.submit(unknown_mass));
    auto invalid = sample(4,3); invalid.position[0] = std::numeric_limits<float>::quiet_NaN();
    assert(!state.submit(invalid));
    invalid = sample(4,3); invalid.radius = -1.f;
    assert(!state.submit(invalid));
    state.publish(); frame = state.snapshot();
    assert(frame.entities.size() == 3 && frame.entities[0].radius == .2f && frame.entities[0].weight == .3f);
    assert(frame.entities[1].radius == 3.f && frame.entities[1].weight == 2.f && frame.entities[2].weight == 1.f);
    state.begin(2,120,focus); state.submit(sample(1,4));
    state.begin(3,140,focus); // An abandoned update is not committed.
    state.submit(sample(1,5)); state.publish();
    assert(state.snapshot().entities[0].velocity[0] == 0.f);

    state.reset(); state.begin(1,100,focus);
    for (uint16_t id=600; id>0; --id) state.submit(sample(id,float(id)));
    state.publish(); frame = state.snapshot();
    assert(frame.entities.size() == State::Capacity && frame.entities.front().position[0] == 1.f &&
        frame.entities.back().position[0] == float(State::Capacity));
    // A reader can retain its snapshot after the level is destroyed.
    state.reset(); assert(state.snapshot().entities.empty() && frame.entities.size() == State::Capacity);

    State concurrent;
    std::atomic<bool> done{false};
    std::atomic<unsigned> reads{0};
    std::thread reader([&] {
        while (!done.load(std::memory_order_acquire))
        {
            const auto snapshot = concurrent.snapshot();
            if (snapshot.frame)
            {
                assert(snapshot.entities.size() == 32);
                for (const auto& entity : snapshot.entities)
                    assert(entity.position[0] == float(snapshot.frame));
            }
            reads.fetch_add(1,std::memory_order_release);
        }
    });
    while (!reads.load(std::memory_order_acquire)) std::this_thread::yield();
    for (uint32_t f=1; f<=100; ++f)
    {
        concurrent.begin(f,f*16,focus);
        for (uint16_t id=0; id<32; ++id) concurrent.submit(sample(id,float(f),float(id)*.1f));
        concurrent.publish();
    }
    done.store(true,std::memory_order_release); reader.join();
    assert(reads.load() > 0);
}
