#pragma once
#include <array>
#include <cstdint>
#include <string>

namespace xray::render::vulkan
{
// Borrowed unordered_map values survive rehashes. Invalidate before replacing
// or destroying the registry. Exact names and variant guard hash collisions.
template <class Value> class ShaderPairCache
{
    struct Entry { std::string vertex, fragment; uint64_t variant{}; const Value* value{}; };
    std::array<Entry,128> entries_{};
    static size_t slot(const char* vertex, const char* fragment, uint64_t variant)
    {
        uint64_t hash=14695981039346656037ull;
        for (const char* name : {vertex,fragment})
        {
            for (const unsigned char* p=reinterpret_cast<const unsigned char*>(name);*p;++p)
                hash=(hash^*p)*1099511628211ull;
            hash=(hash^0xffu)*1099511628211ull;
        }
        for (unsigned shift = 0; shift < 64; shift += 8)
            hash=(hash^static_cast<uint8_t>(variant >> shift))*1099511628211ull;
        return size_t(hash)%128;
    }
public:
    const Value* find(const char* vertex, const char* fragment, uint64_t variant = 0) const
    {
        const auto& item=entries_[slot(vertex,fragment,variant)];
        return item.value && item.vertex==vertex && item.fragment==fragment && item.variant==variant ? item.value : nullptr;
    }
    void remember(const char* vertex, const char* fragment, const Value* value, uint64_t variant = 0)
    {
        if (!value) return;
        auto& item=entries_[slot(vertex,fragment,variant)];
        item.vertex=vertex;item.fragment=fragment;item.variant=variant;item.value=value;
    }
    void clear() { for(auto& item:entries_)item.value=nullptr; }
};
}
