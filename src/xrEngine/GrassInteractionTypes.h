#pragma once
#include <cstdint>
#include <vector>

namespace xray::grass
{
// Renderer-independent contact packet. This is also the GPU structured-buffer
// layout; renderer modules do not need to dereference a gameplay object.
struct Entity
{
    float position[3]{}, radius{};
    float velocity[3]{}, weight{};
};
static_assert(sizeof(Entity) == 32);

struct Sample
{
    const void* instance{};
    uint16_t object_id{};
    float position[3]{};
    float radius{}, mass{};
};

struct Frame
{
    uint32_t frame{}, time_ms{};
    uint64_t revision{};
    std::vector<Entity> entities;
};
}
