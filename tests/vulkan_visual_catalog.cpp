#include "src/Layers/xrRenderVK/VisualCatalog.h"

#include <cassert>
#include <cstring>

using namespace xray::render::vulkan;
using Bytes = std::vector<uint8_t>;

static void u32(Bytes& data, uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i) data.push_back(uint8_t(value >> (8 * i)));
}

static void chunk(Bytes& data, uint32_t id, const Bytes& body)
{
    u32(data, id);
    u32(data, static_cast<uint32_t>(body.size()));
    data.insert(data.end(), body.begin(), body.end());
}

static Bytes visual(uint8_t type)
{
    Bytes header(44);
    header[0] = 4;
    header[1] = type;
    const float radius = 7.f;
    std::memcpy(header.data() + 4 + 9 * sizeof(float), &radius, sizeof(radius));
    Bytes data;
    chunk(data, 1, header);
    return data;
}

int main()
{
    for (const uint8_t type : { uint8_t(0), uint8_t(1), uint8_t(2), uint8_t(6), uint8_t(7), uint8_t(11) })
        assert(is_static_skeleton_lod_type(type));
    for (const uint8_t type : { uint8_t(3), uint8_t(4), uint8_t(5), uint8_t(8), uint8_t(9), uint8_t(10), uint8_t(12) })
        assert(!is_static_skeleton_lod_type(type));

    Bytes table;
    for (uint8_t type = 0; type <= 12; ++type)
    {
        Bytes record = visual(type);
        if (type == 1)
        {
            Bytes links;
            u32(links, 1);
            u32(links, 0);
            chunk(record, 10, links);
        }
        if (type == 3)
        {
            Bytes embedded;
            chunk(embedded, 0, visual(5));
            chunk(record, 9, embedded);
        }
        chunk(table, type, record);
    }
    auto bytes = [](const Bytes& data) { return LevelBytes{data.data(), data.size()}; };
    std::vector<VisualRecord> records;
    std::string error;
    assert(parse_level_visuals(bytes(table), records, error));
    assert(error.empty() && records.size() == 13);
    assert(records[1].linked_children.size() == 1 && records[1].linked_children[0] == 0);
    assert(records[1].bounds[9] == 7.f);
    assert(records[3].embedded_children.size() == 1 && records[3].embedded_children[0].type == 5);

    // OGF_S_LODS is written with IWriter::w_string and read with
    // IReader::r_string, so its path is CR/LF terminated rather than a Z-string.
    const std::string lodName = "dynamics\\devices\\dev_detector_1\\dev_detector_1_lod.ogf";
    Bytes skinned = visual(10);
    Bytes lodChunk(lodName.begin(), lodName.end());
    lodChunk.push_back('\r');
    lodChunk.push_back('\n');
    lodChunk.push_back(0xA5); // Optional trailing bytes remain opaque.
    lodChunk.push_back(0x5A);
    chunk(skinned, 23, lodChunk);
    VisualRecord skinnedRecord;
    assert(parse_ogf_visual(bytes(skinned), skinnedRecord, error));
    assert(skinnedRecord.type == 10 && skinnedRecord.skeleton_lod == lodName);

    // Retail OGF files can put CR/LF and NUL padding before the optional
    // skeleton LOD path. Keep it from becoming a bogus visual name at load.
    Bytes padded_skinned = visual(10);
    Bytes padded_lod{ '\r', '\n', 0, 0, '\r', '\n' };
    padded_lod.insert(padded_lod.end(), lodName.begin(), lodName.end());
    padded_lod.push_back('\r');
    padded_lod.push_back('\n');
    chunk(padded_skinned, 23, padded_lod);
    assert(parse_ogf_visual(bytes(padded_skinned), skinnedRecord, error));
    assert(skinnedRecord.type == 10 && skinnedRecord.skeleton_lod == lodName);

    Bytes blank_lod_skinned = visual(10);
    chunk(blank_lod_skinned, 23, Bytes{ '\r', '\n', 0, 0, '\r', '\n' });
    assert(parse_ogf_visual(bytes(blank_lod_skinned), skinnedRecord, error));
    assert(skinnedRecord.type == 10 && skinnedRecord.skeleton_lod.empty());

    Bytes cyclic = visual(1), links;
    u32(links, 1); u32(links, 0);
    chunk(cyclic, 10, links);
    Bytes bad_table;
    chunk(bad_table, 0, cyclic);
    assert(!parse_level_visuals(bytes(bad_table), records, error));
    assert(records.size() == 13);
    Bytes invalid = visual(13);
    VisualRecord record;
    assert(!parse_ogf_visual(bytes(invalid), record, error));
}
