#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
struct DdsTexture
{
    VkFormat format = VK_FORMAT_UNDEFINED;
    VkExtent3D extent{};
    uint32_t mip_levels{};
    // Vulkan array layers; cube maps use six layers as well.
    uint32_t layers = 1;
    bool cube{};
    // Volume images use extent.depth and always have one array layer.
    bool volume{};
    std::vector<std::uint8_t> pixels;
    std::vector<VkBufferImageCopy> copies;
};

// Accepts legacy 2D/cube/volume DDS and DX10 2D/cube arrays and volumes from
// the game's virtual filesystem. 1D targets are unsupported.
bool decode_dds(const void* data, size_t size, bool bc_supported, DdsTexture& result, std::string& error, uint32_t maximum_dimension = 0, uint32_t mip_skip = 0);
}
