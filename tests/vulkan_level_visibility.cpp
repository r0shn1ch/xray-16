#include "src/Layers/xrRenderVK/LevelVisibility.h"
#include "src/Layers/xrRenderVK/LevelHierarchyBounds.h"

#include <cassert>

using namespace xray::render::vulkan;

namespace
{
LevelPortal portal(uint16_t front, uint16_t back, float x)
{
    LevelPortal result;
    result.sector_front = front;
    result.sector_back = back;
    result.vertices = {{{x, -0.25f, 0.5f}, {x, 0.25f, 0.5f}, {x, 0.25f, 0.75f}, {x, -0.25f, 0.75f}}};
    result.center = {x, 0.f, 0.625f};
    result.radius = 0.36f;
    return result;
}
} // namespace

int main()
{
    LevelPortal floor_portal;
    floor_portal.sector_front = 3;
    floor_portal.sector_back = 7;
    floor_portal.vertices = {{{-1, 0, -1}, {-1, 0, 1}, {1, 0, 1}, {1, 0, -1}}};
    Fvector above, below, down, up;
    above.set(0, 1, 0);
    below.set(0, -1, 0);
    down.set(0, -1, 0);
    up.set(0, 1, 0);
    assert(camera_sector_from_portals({floor_portal}, above, down, 5, 9) == 3);
    assert(camera_sector_from_portals({floor_portal}, below, up, 5, 9) == 7);
    assert(camera_sector_from_portals({floor_portal}, above, down, .5f, 9) == 9);
    above.x = 3;
    assert(camera_sector_from_portals({floor_portal}, above, down, 5, 9) == 9);

    // The exported parent sphere is deliberately wrong; the rebuilt bounds
    // must still contain the distant child, including shared subtrees.
    std::vector<LevelVisual> nodes(4);
    for (auto &node : nodes)
        node.bounds = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    nodes[0].children = {1, 2};
    nodes[1].children = {3};
    nodes[2].bounds[6] = -10;
    nodes[3].bounds[6] = 20;
    auto bounds = level_hierarchy_bounds(nodes);
    assert(bounds[0][0] - bounds[0][3] <= -11 && bounds[0][0] + bounds[0][3] >= 21);
    assert(bounds[1][0] + bounds[1][3] >= 21);
    nodes[3].children = {0};
    bounds = level_hierarchy_bounds(nodes);
    assert(bounds[0][3] < 0 && bounds[1][3] < 0); // cycle fails open
    nodes[3].children.clear();
    nodes[3].bounds[9] = 0;
    bounds = level_hierarchy_bounds(nodes);
    assert(bounds[0][3] < 0 && bounds[1][3] < 0); // unknown bounds fail open
    Fmatrix view_projection;
    view_projection.identity();
    Fvector camera;
    camera.set(0.f, 0.f, 0.f);

    // The camera sector and a neighboring sector behind an in-frustum portal
    // are visited; the chain stops at a portal outside the view frustum.
    const std::vector<LevelSector> sectors{{0, {0}}, {1, {0, 1}}, {2, {1}}};
    const std::vector<LevelPortal> portals{portal(0, 1, 0.5f), portal(1, 2, 5.f)};
    std::vector<uint32_t> roots;
    assert(select_visible_sector_roots(sectors, portals, 3, 0, view_projection, camera, roots));
    assert((roots == std::vector<uint32_t>{0, 1}));
    append_unsectored_level_roots({0, 1, 2, 3, 3}, sectors, 4, roots);
    assert((roots == std::vector<uint32_t>{0, 1, 3}));

    // A malformed portal reached after some valid sectors must not expose a
    // partial visibility set as a successful result.
    auto broken = portals;
    broken[1].sector_back = 99;
    assert(!select_visible_sector_roots(sectors, broken, 3, 1, view_projection, camera, roots));
    assert(roots.empty());

    // Any invalid visibility input asks the caller to fail open, and leaves
    // the output clear so stale roots cannot be reused.
    roots = {99};
    assert(!select_visible_sector_roots(sectors, portals, 3, sectors.size(), view_projection, camera, roots));
    assert(roots.empty());
}
