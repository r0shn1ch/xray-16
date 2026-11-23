#include "src/Layers/xrRenderVK/DetailAssets.h"
#include "src/Layers/xrRenderVK/DetailWind.h"
#include "src/Layers/xrRenderVK/DetailInteraction.h"

#include <cassert>
#include <cstring>
#include <vector>

using namespace xray::render::vulkan;
using Bytes = std::vector<uint8_t>;

static void u32(Bytes& bytes, uint32_t value)
{
    for (int i = 0; i < 4; ++i) bytes.push_back(uint8_t(value >> (i * 8)));
}
static void f32(Bytes& bytes, float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    u32(bytes, bits);
}
static LevelBytes span(const Bytes& bytes) { return {bytes.data(), bytes.size()}; }

int main()
{
    {
        DetailInteractionParameters params;
        const float camera[]{0.f, 0.f}, origin[]{-32.f, -32.f}, height_origin[]{-100.f, -100.f};
        assert(detail_interaction_parameters(params, camera, origin, true, .04f, 1.f, .5f, 1, height_origin, .5f));
        assert(params.origin[0] == -32.f && params.origin[1] == -32.f && params.previous_valid == 1 &&
            params.shift[0] == 0 && params.shift[1] == 0 && params.entity_count == 1);
        assert(std::abs(params.contact_rate - (1.f - std::exp(-1.f))) < 1e-6f);
        const auto step = params;
        assert(detail_interaction_parameters(params, camera, origin, true, .08f, 1.f, .5f, 1, height_origin, .5f));
        const float x = .2f, v = .4f;
        const float x1 = step.spring[0]*x + step.spring[1]*v, v1 = step.spring[2]*x + step.spring[3]*v;
        assert(std::abs(step.spring[0]*x1 + step.spring[1]*v1 - (params.spring[0]*x + params.spring[1]*v)) < 1e-5f);
        assert(std::abs(step.spring[2]*x1 + step.spring[3]*v1 - (params.spring[2]*x + params.spring[3]*v)) < 1e-5f);
        const float moved[]{.0625f, -.0625f};
        assert(detail_interaction_parameters(params, moved, origin, true, 0.f, 1.f, .5f, 0, height_origin, .5f));
        assert(params.shift[0] == 1 && params.shift[1] == -1 && params.previous_valid == 1);
        assert(params.spring[0] == 1.f && params.spring[1] == 0.f && params.spring[2] == 0.f && params.spring[3] == 1.f);
        const float teleported[]{1000.f, 0.f};
        assert(detail_interaction_parameters(params, teleported, origin, true, 1.f, 1.f, .5f, 0, height_origin, .5f));
        assert(params.previous_valid == 0 && params.delta == .1f);
        assert(!detail_interaction_parameters(params, camera, origin, true, .04f, 1.f, .5f, 65, height_origin, .5f));
        assert(!detail_interaction_parameters(params, camera, origin, true, .04f, 1.f, .5f, 0, height_origin, 0.f));
    }
    Bytes header, model, slots;
    for (uint32_t value : {3u, 1u, 0u, 0u, 2u, 1u}) u32(header, value);
    const char shader[] = "details\\set";
    const char texture[] = "plants\\grass";
    model.insert(model.end(), shader, shader + sizeof(shader));
    model.insert(model.end(), texture, texture + sizeof(texture));
    u32(model, 0); f32(model, .5f); f32(model, 1.5f);
    u32(model, 3); u32(model, 3);
    for (int vertex = 0; vertex < 3; ++vertex)
    {
        f32(model, float(vertex)); f32(model, float(vertex)); f32(model, 0);
        f32(model, float(vertex) / 2); f32(model, 1);
    }
    for (uint16_t index : {uint16_t(0), uint16_t(1), uint16_t(2)})
    { slots.reserve(32); model.push_back(uint8_t(index)); model.push_back(0); }
    const uint64_t occupied = 100u | (10u << 12) | (0u << 20) |
        (uint64_t(63) << 26) | (uint64_t(63) << 32) | (uint64_t(63) << 38);
    u32(slots, uint32_t(occupied)); u32(slots, uint32_t(occupied >> 32));
    slots.insert(slots.end(), {0xff, 0xff, 0, 0, 0, 0, 0, 0});
    slots.insert(slots.end(), 16, 0xff);
    DetailAssets decoded;
    std::string error;
    assert(decode_detail_assets(span(header), {span(model)}, span(slots), decoded, error));
    assert(decoded.width == 2 && decoded.prototypes[0].vertices[2].position[0] == 2.f);
    assert(decoded.cells[0].ids[0] == 0 && decoded.cells[0].ground == -180.f);
    assert(decoded.cells[0].palette[0][3] == 15);
    assert(decoded.prototypes[0].flags == 0 && decoded.prototypes[0].vertices[0].lightmap_uv[0] == 0.f &&
        decoded.prototypes[0].vertices[1].lightmap_uv[0] == .5f &&
        decoded.prototypes[0].vertices[2].lightmap_uv[0] == 1.f);
    const size_t flags_offset = sizeof(shader) + sizeof(texture);
    model[flags_offset] = 1; model[flags_offset + 3] = 0xa5;
    assert(decode_detail_assets(span(header), {span(model)}, span(slots), decoded, error));
    assert(decoded.prototypes[0].flags == 0xa5000001u);
    for (const auto& vertex : decoded.prototypes[0].vertices) assert(vertex.lightmap_uv[0] == 0.f);
    model[flags_offset] = 0;
    assert(decode_detail_assets(span(header), {span(model)}, span(slots), decoded, error));
    assert(decoded.prototypes[0].flags == 0xa5000000u && decoded.prototypes[0].vertices[2].lightmap_uv[0] == 1.f);
    assert(detail_wind_weight(1.f, 0.f, 0) == 0.f && detail_wind_weight(1.f, 2.f, 1) == 0.f);
    assert(detail_wind_weight(-10.f, 1.f, 0) == -2.f);
    assert(detail_wind_weight(10.f, 1.f, 0) == 32767.f / 16384.f);
    assert(detail_wind_weight(.123456f, 1.f, 0) == std::floor(.123456f * 16384.f) / 16384.f);
    assert(detail_second_wave(0.f, -0.f) == detail_second_wave(-0.f, 0.f));
    DetailWindAnimation animation;
    animation.normal = {{4.f, 8.f}, {.1f, .2f}, 2.f};
    animation.fast = {{2.f, 4.f}, {.5f, .6f}, 4.f};
    animation.configured = true;
    const auto wind = animation.advance(.25f, .5f);
    assert(std::abs(std::hypot(wind.directions[0], wind.directions[1]) - .3f) < 1e-6f);
    assert(std::abs(wind.amplitude() - .4f) < 1e-6f);
    const auto paused = animation.advance(.25f, .5f);
    assert(wind.phase == paused.phase && wind.directions[0] == paused.directions[0]);
    const auto wrapped = animation.advance(1000.f, 1.f);
    assert(std::abs(wrapped.phase - (wind.phase + .03f * 4.f / 6.283185307179586f)) < 1e-6f);
    const float root[]{2.f, 3.f, 4.f}, tip[]{2.f, 5.f, 4.f};
    assert((detail_wind_displacement(root, root, 1.f, wind) == std::array<float, 3>{}));
    assert((detail_wind_displacement(tip, root, 0.f, wind) == std::array<float, 3>{}));
    const auto displacement = detail_wind_displacement(tip, root, 1.f, wind);
    assert(std::hypot(displacement[0], displacement[2]) <= 2.f * wind.amplitude() * 1.001f);
    assert(detail_wind_radius(1.f, 2.f, wind.amplitude()) >= 1.f + std::hypot(displacement[0], displacement[2]));
    const auto previous = decoded.cells.size();
    model.back() = 3; // out-of-range index leaves the old level intact
    assert(!decode_detail_assets(span(header), {span(model)}, span(slots), decoded, error));
    assert(decoded.cells.size() == previous);
    model.back() = 0;
    header[0] = 4; // unsupported format version
    assert(!decode_detail_assets(span(header), {span(model)}, span(slots), decoded, error));
}
