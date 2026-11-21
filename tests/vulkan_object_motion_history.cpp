#include "src/Layers/xrRenderVK/ObjectMotionHistory.h"
#include <cassert>
#include <limits>

using xray::render::vulkan::ObjectMotionHistory;

int main()
{
    ObjectMotionHistory history;
    using Kind = ObjectMotionHistory::Kind;
    using Matrix = ObjectMotionHistory::Matrix;
    const auto* instance = reinterpret_cast<const void*>(uintptr_t{0x100});
    const auto* other_instance = reinterpret_cast<const void*>(uintptr_t{0x200});
    const auto* model = reinterpret_cast<const void*>(uintptr_t{0x300});
    const ObjectMotionHistory::Key first{Kind::Model, instance, model, 0};
    const ObjectMotionHistory::Key second{Kind::Model, other_instance, model, 0};
    const ObjectMotionHistory::Key level{Kind::LevelVisual, instance, nullptr, 7};
    Matrix a{}, b{}, c{}, previous{};
    a[0] = 1.f; b[0] = 2.f; c[0] = 3.f;

    bool moved = true;
    history.begin_frame();
    assert(!history.stage(first, a, previous, &a, &moved) && !moved);
    assert(!history.stage(second, b, previous));
    history.commit_frame();

    history.begin_frame();
    assert(history.stage(first, b, previous, &a, &moved) && previous == a && !moved);
    assert(history.stage(first, c, previous, &b, &moved) && previous == a && moved);
    assert(history.stage(second, c, previous) && previous == b);
    assert(!history.stage(level, a, previous));
    history.discard_frame();
    // A rejected submission must not become temporal history.
    history.begin_frame();
    assert(history.stage(first, b, previous, &b, &moved) && previous == a && moved);
    history.commit_frame();

    history.begin_frame();
    assert(history.stage(first, c, previous, &b, &moved) && previous == b && !moved);
    assert(!history.stage(second, c, previous));
    assert(!history.stage({Kind::Model, nullptr, model, 0}, c, previous));
    Matrix invalid = c;
    invalid[4] = std::numeric_limits<float>::quiet_NaN();
    assert(!history.stage(level, invalid, previous));
    history.forget_instance(instance);
    assert(!history.stage(first, c, previous));
    history.commit_frame();

    history.reset();
    history.begin_frame();
    assert(!history.stage(first, a, previous));
}
