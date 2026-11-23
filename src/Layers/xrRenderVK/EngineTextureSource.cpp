#include "EngineTextureSource.h"

namespace xray::render::vulkan
{
bool upload_engine_texture(VkDevice device, VkQueue queue, VkCommandPool pool,
    const VkPhysicalDeviceMemoryProperties& memory_types, const TextureUploadDispatch& vk,
    const void* bytes, size_t size, bool bc_supported, UploadedTexture& result,
    std::vector<PendingTextureUpload>& pending, ImageStateTracker& states, std::string& error,
    VkExtent3D* decoded_extent, bool* decoded_cube,
    uint32_t maximum_dimension, uint32_t mip_skip, bool force_linear,
    uint32_t* decoded_layers, bool* decoded_volume)
{
    DdsTexture decoded;
    if (!decode_dds(bytes, size, bc_supported, decoded, error, maximum_dimension, mip_skip))
        return false;
    if (force_linear)
    {
        switch (decoded.format)
        {
        case VK_FORMAT_R8G8B8A8_SRGB: decoded.format = VK_FORMAT_R8G8B8A8_UNORM; break;
        case VK_FORMAT_B8G8R8A8_SRGB: decoded.format = VK_FORMAT_B8G8R8A8_UNORM; break;
        case VK_FORMAT_BC1_RGB_SRGB_BLOCK: decoded.format = VK_FORMAT_BC1_RGB_UNORM_BLOCK; break;
        case VK_FORMAT_BC1_RGBA_SRGB_BLOCK: decoded.format = VK_FORMAT_BC1_RGBA_UNORM_BLOCK; break;
        case VK_FORMAT_BC2_SRGB_BLOCK: decoded.format = VK_FORMAT_BC2_UNORM_BLOCK; break;
        case VK_FORMAT_BC3_SRGB_BLOCK: decoded.format = VK_FORMAT_BC3_UNORM_BLOCK; break;
        case VK_FORMAT_BC7_SRGB_BLOCK: decoded.format = VK_FORMAT_BC7_UNORM_BLOCK; break;
        default: break;
        }
    }
    if (!upload_texture(device, queue, pool, memory_types, vk, decoded, result, pending, states, error))
        return false;
    if (decoded_extent) *decoded_extent = decoded.extent;
    if (decoded_cube) *decoded_cube = decoded.cube;
    if (decoded_layers) *decoded_layers = decoded.layers;
    if (decoded_volume) *decoded_volume = decoded.volume;
    return true;
}
}
