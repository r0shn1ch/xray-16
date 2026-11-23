#include "xrEngine/stdafx.h"
#include "EngineParticleSource.h"

namespace xray::render::vulkan
{
namespace
{
void particle_chunk(std::vector<uint8_t>& out, uint32_t id, const void* data, size_t size)
{
    for (unsigned shift = 0; shift < 32; shift += 8) out.push_back(uint8_t(id >> shift));
    for (unsigned shift = 0; shift < 32; shift += 8) out.push_back(uint8_t(size >> shift));
    if (size)
    {
        const auto* bytes = static_cast<const uint8_t*>(data);
        out.insert(out.end(), bytes, bytes + size);
    }
}


bool copy_editor_chunks(IReader& source, unsigned depth, std::vector<uint8_t>& result)
{
    if (source.length() > editor_library_limit)
        return false;
    source.rewind();
    while (source.elapsed())
    {
        if (source.elapsed() < 8)
            return false;
        const uint32_t id = source.r_u32();
        const uint32_t size = source.r_u32();
        if (size > static_cast<size_t>(source.elapsed()))
            return false;
        if (id & CFS_CompressMark)
        {
            if (size < 4)
                return false;
            const auto* header = static_cast<const uint8_t*>(source.pointer());
            const uint32_t decoded = uint32_t(header[0]) | uint32_t(header[1]) << 8 |
                uint32_t(header[2]) << 16 | uint32_t(header[3]) << 24;
            if (!decoded || decoded > editor_library_limit)
                return false;
        }
        source.advance(size);
    }

    uint32_t tagged_id{};
    for (IReader* child = source.open_chunk_iterator(tagged_id); child;
         child = source.open_chunk_iterator(tagged_id, child))
    {
        const uint32_t id = tagged_id & ~CFS_CompressMark;
        std::vector<uint8_t> descendants;
        const bool container = (depth == 0 && (id == 3 || id == 4)) || depth == 1;
        if (container && !copy_editor_chunks(*child, depth + 1, descendants))
        {
            child->close();
            return false;
        }
        const size_t size = container ? descendants.size() : child->length();
        if (result.size() > editor_library_limit - 8 || size > editor_library_limit - 8 - result.size())
        {
            child->close();
            return false;
        }
        particle_chunk(result, id, container ? descendants.data() : child->pointer(), size);
    }
    return true;
}
}

bool editable_particle_library_bytes(IReader& source, std::vector<uint8_t>& bytes, std::string& error)
{
    std::vector<uint8_t> expanded;
    if (!copy_editor_chunks(source, 0, expanded))
    {
        error = "invalid or oversized particle library";
        return false;
    }
    bytes = std::move(expanded);
    error.clear();
    return true;
}

// IReader::open_chunk transparently decompresses CFS_CompressMark chunks.
// Reassemble only the particle library fields used by the portable decoder.
std::vector<uint8_t> particle_library_bytes(IReader& source)
{
    std::vector<uint8_t> library;
    if (IReader* version = source.open_chunk(1))
    {
        particle_chunk(library, 1, version->pointer(), version->length());
        version->close();
    }
    for (uint32_t kind : {3u, 4u})
    {
        IReader* container = source.open_chunk(kind);
        if (!container) continue;
        std::vector<uint8_t> definitions;
        // Particle effect records store optional fields above the small
        // sequential range too: collision, velocity scale and path alignment
        // use their PED_CHUNK_* ids (0x21, 0x22 and 0x25). Preserve those
        // fields while rebuilding the portable chunk table.
        for (uint32_t index = 0; index < 100000; ++index)
        {
            IReader* definition = container->open_chunk(index);
            if (!definition) break;
            std::vector<uint8_t> fields;
            const auto append_field = [&](uint32_t field) {
                if (IReader* value = definition->open_chunk(field))
                {
                    particle_chunk(fields, field, value->pointer(), value->length());
                    value->close();
                }
            };
            if (kind == 3)
                for (uint32_t field : particle_effect_chunk_ids) append_field(field);
            else
                for (uint32_t field : particle_group_chunk_ids) append_field(field);
            particle_chunk(definitions, index, fields.data(), fields.size());
            definition->close();
        }
        particle_chunk(library, kind, definitions.data(), definitions.size());
        container->close();
    }
    return library;
}

}
