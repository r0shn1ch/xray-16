#include "VisualCatalog.h"

#include <cstring>
#include <cmath>
#include <functional>

namespace xray::render::vulkan
{
namespace
{
constexpr uint32_t header_id = 1, texture_id = 2, children_id = 9, linked_id = 10;
constexpr uint32_t skeleton_lods_id = 23;
constexpr unsigned max_depth = 32;
constexpr size_t max_visuals = 1'000'000;

uint32_t read_u32(const uint8_t* p)
{
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

bool fail(std::string* error, std::string message)
{
    if (error && error->empty()) *error = std::move(message);
    return false;
}

bool each_chunk(LevelBytes bytes, const std::function<bool(uint32_t, LevelBytes)>& callback,
    std::string* error = nullptr)
{
    if (bytes.size && !bytes.data) return fail(error, "chunk table has a null payload");
    size_t offset = 0;
    while (offset < bytes.size)
    {
        if (bytes.size - offset < 8) return fail(error, "truncated chunk header");
        const uint32_t id = read_u32(bytes.data + offset);
        const uint32_t length = read_u32(bytes.data + offset + 4);
        offset += 8;
        // IReader::open_chunk handles engine compression. This parser only
        // accepts the uncompressed payload it receives from that reader.
        if (id & 0x80000000u)
            return fail(error, "compressed chunk was not normalized id=" + std::to_string(id));
        if (length > bytes.size - offset)
            return fail(error, "chunk payload exceeds table id=" + std::to_string(id));
        if (!callback(id, {bytes.data + offset, length}))
        {
            if (error && error->empty())
                *error = "invalid chunk payload id=" + std::to_string(id);
            return false;
        }
        offset += length;
    }
    return true;
}

bool read_string(LevelBytes bytes, size_t& offset, std::string& output)
{
    const size_t begin = offset;
    while (offset < bytes.size && bytes.data[offset]) ++offset;
    if (offset == bytes.size) return false;
    output.assign(reinterpret_cast<const char*>(bytes.data + begin), offset - begin);
    ++offset;
    return true;
}

void read_ogf_line_string(LevelBytes bytes, size_t& offset, std::string& output)
{
    // OGF writers can leave empty lines or NUL padding before the optional
    // skeleton LOD name. Ignore that prefix, then consume only its first line.
    while (offset < bytes.size &&
        (bytes.data[offset] == '\r' || bytes.data[offset] == '\n' || bytes.data[offset] == 0))
        ++offset;

    output.clear();
    if (offset == bytes.size)
        return;

    const size_t begin = offset;
    while (offset < bytes.size && bytes.data[offset] != '\r' && bytes.data[offset] != '\n' && bytes.data[offset] != 0)
        ++offset;

    output.assign(reinterpret_cast<const char*>(bytes.data + begin), offset - begin);
}

bool valid_optional_skeleton_lod(const std::string& name)
{
    if (name.empty() || name.size() > 1024 || name.front() == '/' || name.front() == '\\' ||
        name.find(':') != std::string::npos)
        return false;

    size_t component_begin = 0;
    for (size_t i = 0; i <= name.size(); ++i)
    {
        const bool separator = i == name.size() || name[i] == '/' || name[i] == '\\';
        if (i < name.size() && static_cast<unsigned char>(name[i]) < 0x20)
            return false;
        if (!separator)
            continue;

        const std::string component = name.substr(component_begin, i - component_begin);
        if (component.empty() || component == "." || component == "..")
            return false;
        component_begin = i + 1;
    }
    return true;
}

bool parse_visual(LevelBytes bytes, VisualRecord& result, std::string& error, unsigned depth)
{
    if (depth > max_depth) return fail(&error, "OGF visual nesting limit exceeded");
    if (bytes.size > 64 * 1024 * 1024) return fail(&error, "OGF visual exceeds 64 MiB limit");
    if (!bytes.data) return fail(&error, "OGF visual has a null payload");
    VisualRecord visual;
    bool header = false, children = false, links = false, texture = false, skeleton_lod = false;
    if (!each_chunk(bytes, [&](uint32_t id, LevelBytes body)
        {
            if (id == header_id)
            {
                if (header) return fail(&error, "duplicate OGF header chunk");
                if (body.size != 44) return fail(&error, "OGF header size is " + std::to_string(body.size) + ", expected 44");
                if (body.data[0] != 4) return fail(&error, "unsupported OGF format version=" + std::to_string(body.data[0]));
                if (body.data[1] > 12) return fail(&error, "unsupported OGF visual type=" + std::to_string(body.data[1]));
                header = true;
                visual.type = body.data[1];
                visual.shader_id = uint16_t(body.data[2]) | (uint16_t(body.data[3]) << 8);
                std::memcpy(visual.bounds.data(), body.data + 4, sizeof(float) * 10);
                for (size_t i = 0; i < visual.bounds.size(); ++i)
                    if (!std::isfinite(visual.bounds[i]))
                        return fail(&error, "OGF header has a non-finite bound value index=" + std::to_string(i));
                if (visual.bounds[9] < 0)
                    return fail(&error, "OGF header has a negative bounding-sphere radius");
            }
            else if (id == texture_id)
            {
                if (texture) return fail(&error, "duplicate OGF texture chunk");
                texture = true;
                size_t position = 0;
                if (!read_string(body, position, visual.texture) ||
                    !read_string(body, position, visual.shader))
                    return fail(&error, "malformed OGF texture/shader strings");
                // FBasicVisual::Load reads the two Z-strings and leaves any
                // format-specific trailing bytes opaque.
            }
            else if (id == skeleton_lods_id) // OGF_S_LODS
            {
                if (skeleton_lod) return fail(&error, "duplicate OGF skeleton LOD chunk");
                skeleton_lod = true;
                size_t position = 0;
                read_ogf_line_string(body, position, visual.skeleton_lod);
                // CKinematics::Load reads the LOD name with IReader::r_string;
                // OGF writers terminate that string with CR/LF. An empty line
                // means the optional LOD is absent; trailing format data stays opaque.
                // A malformed optional reference must not make the base model
                // unusable; VulkanLevelRender can render the skeleton without its LOD.
                if (!valid_optional_skeleton_lod(visual.skeleton_lod))
                    visual.skeleton_lod.clear();
            }
            else if (id == linked_id)
            {
                if (links) return fail(&error, "duplicate OGF linked-children chunk");
                if (body.size < 4) return fail(&error, "truncated OGF linked-children chunk");
                links = true;
                const uint32_t count = read_u32(body.data);
                if (count > max_visuals || size_t(count) * 4 != body.size - 4)
                    return fail(&error, "invalid OGF linked-children count=" + std::to_string(count));
                visual.linked_children.reserve(count);
                for (uint32_t i = 0; i < count; ++i)
                    visual.linked_children.push_back(read_u32(body.data + 4 + size_t(i) * 4));
            }
            else if (id == children_id)
            {
                if (children) return fail(&error, "duplicate OGF embedded-children chunk");
                children = true;
                uint32_t expected = 0;
                std::string child_error;
                const bool loaded = each_chunk(body, [&](uint32_t child_id, LevelBytes child)
                    {
                        if (child_id != expected)
                            return fail(&child_error, "non-contiguous OGF child index=" + std::to_string(child_id) +
                                " expected=" + std::to_string(expected));
                        ++expected;
                        if (expected > max_visuals)
                            return fail(&child_error, "OGF embedded-child count exceeds limit");
                        VisualRecord record;
                        if (!parse_visual(child, record, child_error, depth + 1)) return false;
                        visual.embedded_children.push_back(std::move(record));
                        return true;
                    }, &child_error);
                if (!loaded)
                    return fail(&error, "OGF embedded child " + std::to_string(expected) + " / " + child_error);
            }
            return true; // Keep all model-specific chunks verbatim in source.
        }, &error)) return false;
    if (!header) return fail(&error, "OGF visual has no header chunk");
    if (links && children) return fail(&error, "OGF visual has both linked and embedded children");
    if (skeleton_lod && visual.type != 3 && visual.type != 10)
        return fail(&error, "OGF skeleton LOD chunk used by visual type=" + std::to_string(visual.type));
    visual.source.assign(bytes.data, bytes.data + bytes.size);
    result = std::move(visual);
    error.clear();
    return true;
}
}

bool parse_ogf_visual(LevelBytes bytes, VisualRecord& result, std::string& error)
{
    error.clear();
    if (!parse_visual(bytes, result, error, 0))
    {
        uint32_t type = UINT32_MAX, shader = UINT32_MAX;
        each_chunk(bytes, [&](uint32_t id, LevelBytes body) {
            if (id == header_id && body.size >= 4)
            {
                type = body.data[1];
                shader = uint16_t(body.data[2]) | (uint16_t(body.data[3]) << 8);
            }
            return true;
        });
        error = "invalid OGF visual record type=" + std::to_string(type) + " shader_id=" +
            std::to_string(shader) + " / " + error;
        return false;
    }
    error.clear();
    return true;
}

bool parse_level_visuals(LevelBytes bytes, std::vector<VisualRecord>& result, std::string& error)
{
    std::vector<VisualRecord> records;
    uint32_t expected = 0;
    std::string parse_error;
    if (!each_chunk(bytes, [&](uint32_t id, LevelBytes body) {
            if (id != expected)
                return fail(&parse_error, "non-contiguous level visual index=" + std::to_string(id) +
                    " expected=" + std::to_string(expected));
            if (expected >= max_visuals)
                return fail(&parse_error, "level visual count exceeds limit");
            VisualRecord record;
            std::string visual_error;
            if (!parse_visual(body, record, visual_error, 0))
                return fail(&parse_error, "level visual index=" + std::to_string(expected) + " / " + visual_error);
            records.push_back(std::move(record));
            ++expected;
            return true;
        }, &parse_error))
    {
        error = "invalid level visual table / " + parse_error;
        return false;
    }
    std::vector<uint8_t> visiting(records.size());
    std::function<bool(size_t)> validate = [&](size_t index)
    {
        if (visiting[index] == 1) return false;
        if (visiting[index] == 2) return true;
        visiting[index] = 1;
        for (uint32_t child : records[index].linked_children)
            if (child >= records.size() || !validate(child)) return false;
        visiting[index] = 2;
        return true;
    };
    for (size_t i = 0; i < records.size(); ++i)
        if (!validate(i))
        {
            error = "invalid or cyclic OGF child references";
            return false;
        }
    result = std::move(records);
    error.clear();
    return true;
}
}
