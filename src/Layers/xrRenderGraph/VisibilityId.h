#pragma once

#include <cstddef>
#include <cstdint>

namespace xray::render::visibility
{
constexpr uint32_t TriangleBits = 7;
constexpr uint32_t TrianglesPerEntry = 1u << TriangleBits;
constexpr uint32_t EntryCountLimit = UINT32_MAX >> TriangleBits;
constexpr uint32_t Background = 0;

constexpr uint32_t pack_id(uint32_t entry, uint32_t triangle)
{
    if (entry >= EntryCountLimit || triangle >= TrianglesPerEntry)
        return Background;
    return ((entry + 1u) << TriangleBits) | triangle;
}

constexpr bool valid_id(uint32_t id) { return id >= TrianglesPerEntry; }
constexpr uint32_t entry_index(uint32_t id) { return (id >> TriangleBits) - 1u; }
constexpr uint32_t triangle_index(uint32_t id) { return id & (TrianglesPerEntry - 1u); }

struct DrawEntry
{
    uint32_t first_index;
    uint32_t index_count;
    int32_t vertex_offset;
    uint32_t material_slot;
    uint32_t geometry_batch;
    uint32_t lightmap_slot{UINT32_MAX};
    float material_class{1.f / 3.f};
    uint32_t pbr_slot{UINT32_MAX};
    uint32_t metallic_slot{UINT32_MAX};
    uint32_t roughness_slot{UINT32_MAX};
    uint32_t ao_slot{UINT32_MAX};
    uint32_t bump_slot{UINT32_MAX};
    uint32_t bump_aux_slot{UINT32_MAX};
    uint32_t normal_slot{UINT32_MAX};
    float bump_height{};
    uint32_t parallax_slot{UINT32_MAX};
    float alpha_cutoff{-1.f};
    uint32_t detail_slot{UINT32_MAX};
    uint32_t detail_bump_slot{UINT32_MAX};
    uint32_t detail_bump_aux_slot{UINT32_MAX};
    float detail_scale{1.f};
    uint32_t detail_mode{};
    float emission{};
    float base_u_x{1.f}, base_u_y{}, base_u_offset{};
    float base_v_x{}, base_v_y{1.f}, base_v_offset{};
    float detail_u_x{1.f}, detail_u_y{}, detail_u_offset{};
    float detail_v_x{}, detail_v_y{1.f}, detail_v_offset{};
    uint32_t environment_cube{UINT32_MAX};
    float environment_xx{1.f}, environment_xy{}, environment_xz{};
    float environment_yx{}, environment_yy{1.f}, environment_yz{};
    float environment_zx{}, environment_zy{}, environment_zz{1.f};
};

struct Viewport
{
    float origin[2];
    float extent[2];
};

struct Camera
{
    float view_projection[16];
    Viewport viewport;
};

struct SoftwareRasterParameters
{
    Camera camera;
    uint32_t first_draw;
    uint32_t draw_count;
    uint32_t write_ids;
    uint32_t geometry_batch;
    uint32_t material_count{};
    uint32_t padding[3]{};
    float eye_and_steep[4]{};
};

struct MaterialResolveParameters
{
    Camera camera;
    uint32_t first_draw;
    uint32_t draw_count;
    uint32_t material_count;
    uint32_t geometry_batch;
    float eye_and_steep[4]{};
};

static_assert(sizeof(DrawEntry) == 180);
static_assert(sizeof(Viewport) == 16);
static_assert(offsetof(Camera, viewport) == 64);
static_assert(sizeof(Camera) == 80);
static_assert(sizeof(SoftwareRasterParameters) == 128);
static_assert(sizeof(MaterialResolveParameters) == 112);

static_assert(pack_id(0, 0) == 128);
static_assert(pack_id(EntryCountLimit - 1, TrianglesPerEntry - 1) == UINT32_MAX);
static_assert(pack_id(EntryCountLimit, 0) == Background);
static_assert(pack_id(0, TrianglesPerEntry) == Background);
static_assert(entry_index(UINT32_MAX) == EntryCountLimit - 1);
static_assert(triangle_index(UINT32_MAX) == TrianglesPerEntry - 1);
}
