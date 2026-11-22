#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace xray::render::framegraph
{
// Quad-tree allocator shared by static and dynamic local shadow atlas views.
constexpr uint32_t kLocalShadowAtlas = 4096;
constexpr uint32_t kLocalAtlasLevels = 6;
constexpr uint32_t kLocalAtlasNodes = 1365;
struct LocalAtlasAllocator {
    uint16_t nodeX[kLocalAtlasNodes] = {};
    uint16_t nodeY[kLocalAtlasNodes] = {};
    uint8_t nodeLevel[kLocalAtlasNodes] = {};
    uint8_t nodeState[kLocalAtlasNodes] = {};
    uint16_t nodeSlot[kLocalAtlasNodes] = {};
    std::vector<uint32_t> freeList[kLocalAtlasLevels];
    uint32_t usedTexels = 0;
    bool built = false;

    void Reset(uint32_t width = kLocalShadowAtlas, uint32_t height = kLocalShadowAtlas);
    uint32_t Alloc(uint32_t level);
    bool AllocateFaces(uint32_t level, uint32_t count, std::array<uint32_t, 6>& faces);
    void Free(uint32_t node);
    void Rect(uint32_t node, uint32_t& x, uint32_t& y, uint32_t& size) const;
    uint32_t UsedTexels() const { return usedTexels; }

    static uint32_t SizeOf(uint32_t level) { return kLocalShadowAtlas >> level; }
    static uint32_t LevelOf(uint32_t size);

private:
    void Build();
    void AddFreeRegion(uint32_t node, uint32_t width, uint32_t height);
    void ListPush(uint32_t node);
    void ListRemove(uint32_t node);
    uint32_t Take(uint32_t level);
};

inline uint32_t LocalAtlasAllocator::LevelOf(uint32_t size)
{
    uint32_t level = 0;
    uint32_t s = kLocalShadowAtlas;
    while (s > size && level + 1 < kLocalAtlasLevels) {
        s >>= 1;
        ++level;
    }
    return level;
}

inline void LocalAtlasAllocator::Build()
{
    nodeX[0] = 0;
    nodeY[0] = 0;
    nodeLevel[0] = 0;
    uint32_t base = 0;
    uint32_t count = 1;
    for (uint32_t level = 1; level < kLocalAtlasLevels; ++level) {
        const uint32_t childBase = base + count;
        const uint32_t size = SizeOf(level);
        for (uint32_t p = 0; p < count; ++p) {
            const uint32_t parent = base + p;
            for (uint32_t k = 0; k < 4; ++k) {
                const uint32_t child = childBase + p * 4 + k;
                nodeX[child] = uint16_t(nodeX[parent] + (k & 1u) * size);
                nodeY[child] = uint16_t(nodeY[parent] + (k >> 1) * size);
                nodeLevel[child] = uint8_t(level);
            }
        }
        base = childBase;
        count *= 4;
    }
    built = true;
}

inline void LocalAtlasAllocator::ListPush(uint32_t node)
{
    std::vector<uint32_t>& list = freeList[nodeLevel[node]];
    nodeSlot[node] = uint16_t(list.size());
    list.push_back(node);
}

inline void LocalAtlasAllocator::ListRemove(uint32_t node)
{
    std::vector<uint32_t>& list = freeList[nodeLevel[node]];
    const uint32_t pos = nodeSlot[node];
    const uint32_t last = list.back();
    list[pos] = last;
    nodeSlot[last] = uint16_t(pos);
    list.pop_back();
}

inline void LocalAtlasAllocator::AddFreeRegion(uint32_t node, uint32_t width, uint32_t height)
{
    const uint32_t x = nodeX[node], y = nodeY[node];
    if (x >= width || y >= height) return;
    const uint32_t level = nodeLevel[node], size = SizeOf(level);
    if (size <= width - x && size <= height - y)
    {
        nodeState[node] = 0;
        ListPush(node);
        return;
    }
    if (level + 1 == kLocalAtlasLevels) return;
    nodeState[node] = 1;
    for (uint32_t child = node * 4 + 4; child > node * 4; --child)
        AddFreeRegion(child, width, height);
}

inline void LocalAtlasAllocator::Reset(uint32_t width, uint32_t height)
{
    if (!built)
        Build();
    for (uint32_t i = 0; i < kLocalAtlasNodes; ++i) {
        nodeState[i] = 3;
        nodeSlot[i] = 0;
    }
    for (uint32_t l = 0; l < kLocalAtlasLevels; ++l)
        freeList[l].clear();
    usedTexels = 0;
    AddFreeRegion(0, width, height);
}

inline uint32_t LocalAtlasAllocator::Take(uint32_t level)
{
    if (!freeList[level].empty()) {
        const uint32_t node = freeList[level].back();
        ListRemove(node);
        nodeState[node] = 3;
        return node;
    }
    if (level == 0)
        return uint32_t(-1);
    const uint32_t parent = Take(level - 1);
    if (parent == uint32_t(-1))
        return uint32_t(-1);
    nodeState[parent] = 1;
    const uint32_t parentBase = ((1u << (2u * (level - 1))) - 1u) / 3u;
    const uint32_t base = ((1u << (2u * level)) - 1u) / 3u + (parent - parentBase) * 4u;
    for (uint32_t k = 3; k > 0; --k) {
        nodeState[base + k] = 0;
        ListPush(base + k);
    }
    nodeState[base] = 3;
    return base;
}

inline uint32_t LocalAtlasAllocator::Alloc(uint32_t level)
{
    if (!built)
        Reset();
    if (level >= kLocalAtlasLevels)
        return uint32_t(-1);
    const uint32_t node = Take(level);
    if (node == uint32_t(-1))
        return uint32_t(-1);
    nodeState[node] = 2;
    const uint32_t size = SizeOf(level);
    usedTexels += size * size;
    return node;
}

inline void LocalAtlasAllocator::Free(uint32_t node)
{
    if (node >= kLocalAtlasNodes || nodeState[node] != 2)
        return;
    uint32_t level = nodeLevel[node];
    const uint32_t size = SizeOf(level);
    usedTexels -= size * size;
    nodeState[node] = 0;
    ListPush(node);
    while (level > 0) {
        const uint32_t levelBase = ((1u << (2u * level)) - 1u) / 3u;
        const uint32_t sibBase = levelBase + (((node - levelBase) >> 2) << 2);
        bool all = true;
        for (uint32_t k = 0; k < 4; ++k)
            if (nodeState[sibBase + k] != 0)
                all = false;
        if (!all)
            break;
        for (uint32_t k = 0; k < 4; ++k) {
            ListRemove(sibBase + k);
            nodeState[sibBase + k] = 3;
        }
        const uint32_t parentBase = ((1u << (2u * (level - 1))) - 1u) / 3u;
        node = parentBase + ((node - levelBase) >> 2);
        --level;
        nodeState[node] = 0;
        ListPush(node);
    }
}

inline void LocalAtlasAllocator::Rect(uint32_t node, uint32_t& x, uint32_t& y, uint32_t& size) const
{
    x = nodeX[node];
    y = nodeY[node];
    size = SizeOf(nodeLevel[node]);
}

inline bool LocalAtlasAllocator::AllocateFaces(uint32_t level, uint32_t count,
    std::array<uint32_t, 6>& faces)
{
    if (!count || count > faces.size())
        return false;
    std::array<uint32_t, 6> allocated;
    allocated.fill(UINT32_MAX);
    for (uint32_t face = 0; face < count; ++face)
    {
        allocated[face] = Alloc(level);
        if (allocated[face] == UINT32_MAX)
        {
            while (face)
                Free(allocated[--face]);
            return false;
        }
    }
    faces = allocated;
    return true;
}
}
