#include "xrEngine/stdafx.h"
#include "src/Layers/xrRenderVK/EngineLevelModels.h"
#include "src/Layers/xrRenderVK/GpuModel.h"
#include "src/Layers/xrRenderVK/ObjectVelocityPass.h"
#include "src/Layers/xrRenderVK/VsmPageRender.h"
#include "src/Layers/xrRenderVK/VulkanModelVisual.h"
#include "src/Layers/xrRenderVK/ShaderAssetFingerprint.h"
#include "src/Layers/xrRenderVK/VulkanVisual.h"
#include "src/Layers/xrRenderVK/VulkanUIShader.h"
#include "src/Layers/xrRenderVK/VulkanUIRender.h"
#include "xrEngine/device.h"
#include "src/xrEngine/defines.h"
#ifdef NDEBUG
#undef NDEBUG // Production headers use Release ABI; mock assertions retain side effects.
#endif
#include <cassert>
#include "vulkan_mock_device.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <gli/save_dds.hpp>
#include <gli/texture.hpp>
#include <limits>
using namespace xray::render::vulkan;
namespace
{
template <typename Handle, typename Info>
VkResult VKAPI_PTR create_velocity_object(VkDevice, const Info*, const VkAllocationCallbacks*, Handle* result)
{
    *result = vk_mock::handle<Handle>(vk_mock::next_handle++);
    return VK_SUCCESS;
}
template <typename Handle>
void VKAPI_PTR destroy_velocity_object(VkDevice, Handle, const VkAllocationCallbacks*) {}
void VKAPI_PTR begin_velocity_pass(VkCommandBuffer, const VkRenderPassBeginInfo*, VkSubpassContents) {}
void VKAPI_PTR end_velocity_pass(VkCommandBuffer) {}
PFN_vkVoidFunction VKAPI_PTR velocity_proc(VkDevice, const char* name)
{
#define VELOCITY_PROC(api, function) if (std::strcmp(name, #api) == 0) return reinterpret_cast<PFN_vkVoidFunction>(function)
    VELOCITY_PROC(vkCreateRenderPass, (create_velocity_object<VkRenderPass, VkRenderPassCreateInfo>));
    VELOCITY_PROC(vkDestroyRenderPass, destroy_velocity_object<VkRenderPass>);
    VELOCITY_PROC(vkCreateFramebuffer, (create_velocity_object<VkFramebuffer, VkFramebufferCreateInfo>));
    VELOCITY_PROC(vkDestroyFramebuffer, destroy_velocity_object<VkFramebuffer>);
    VELOCITY_PROC(vkCreateShaderModule, (create_velocity_object<VkShaderModule, VkShaderModuleCreateInfo>));
    VELOCITY_PROC(vkDestroyShaderModule, destroy_velocity_object<VkShaderModule>);
    VELOCITY_PROC(vkCreatePipelineLayout, vk_mock::layout);
    VELOCITY_PROC(vkDestroyPipelineLayout, vk_mock::destroy_layout);
    VELOCITY_PROC(vkCreateGraphicsPipelines, vk_mock::pipeline);
    VELOCITY_PROC(vkDestroyPipeline, vk_mock::destroy_pipeline);
    VELOCITY_PROC(vkCmdBeginRenderPass, begin_velocity_pass);
    VELOCITY_PROC(vkCmdEndRenderPass, end_velocity_pass);
    VELOCITY_PROC(vkCmdBindPipeline, vk_mock::bind_pipeline);
    VELOCITY_PROC(vkCmdBindDescriptorSets, vk_mock::bind_descriptor);
    VELOCITY_PROC(vkCmdSetViewport, vk_mock::viewport);
    VELOCITY_PROC(vkCmdSetScissor, vk_mock::scissor);
    VELOCITY_PROC(vkCmdBindVertexBuffers, vk_mock::bind_vertices);
    VELOCITY_PROC(vkCmdBindIndexBuffer, vk_mock::bind_indices);
    VELOCITY_PROC(vkCmdPushConstants, vk_mock::push);
    VELOCITY_PROC(vkCmdDrawIndexed, vk_mock::draw);
    VELOCITY_PROC(vkCmdDrawIndexedIndirect, vk_mock::draw_indexed_indirect);
#undef VELOCITY_PROC
    return nullptr;
}
unsigned model_ray_builds{}, model_ray_updates{}, model_ray_destroyed{};
VkResult VKAPI_PTR create_model_as(VkDevice, const VkAccelerationStructureCreateInfoKHR*,
    const VkAllocationCallbacks*, VkAccelerationStructureKHR* out)
{ *out = vk_mock::handle<VkAccelerationStructureKHR>(vk_mock::next_handle++); return VK_SUCCESS; }
void VKAPI_PTR destroy_model_as(VkDevice, VkAccelerationStructureKHR, const VkAllocationCallbacks*)
{ ++model_ray_destroyed; }
void VKAPI_PTR model_as_sizes(VkDevice, VkAccelerationStructureBuildTypeKHR,
    const VkAccelerationStructureBuildGeometryInfoKHR*, const uint32_t*, VkAccelerationStructureBuildSizesInfoKHR* sizes)
{ sizes->accelerationStructureSize = 1024; sizes->buildScratchSize = 512; sizes->updateScratchSize = 512; }
void VKAPI_PTR model_as_build(VkCommandBuffer, uint32_t count,
    const VkAccelerationStructureBuildGeometryInfoKHR* info, const VkAccelerationStructureBuildRangeInfoKHR* const*)
{
    assert(count == 1);
    if (info->mode == VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) ++model_ray_updates;
    else ++model_ray_builds;
}
VkDeviceAddress VKAPI_PTR model_as_address(VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR* info)
{ return static_cast<VkDeviceAddress>((uintptr_t)info->accelerationStructure) * 4096u; }
VkDeviceAddress VKAPI_PTR model_buffer_address(VkDevice, const VkBufferDeviceAddressInfo* info)
{ return static_cast<VkDeviceAddress>((uintptr_t)info->buffer) * 4096u; }
VkDeviceAddress VKAPI_PTR model_ray_address(VkDevice, const VkBufferDeviceAddressInfo* info)
{
    const auto required = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    assert((vk_mock::buffer_usages.at(info->buffer) & required) == required);
    return static_cast<VkDeviceAddress>((uintptr_t)info->buffer) * 4096u;
}
void put32(std::vector<uint8_t> &bytes, uint32_t n)
{
    for (unsigned i = 0; i < 4; ++i)
        bytes.push_back(static_cast<uint8_t>(n >> (i * 8)));
}
void put16(std::vector<uint8_t> &bytes, uint16_t n)
{
    bytes.push_back(static_cast<uint8_t>(n));
    bytes.push_back(static_cast<uint8_t>(n >> 8));
}
void putfloat(std::vector<uint8_t> &bytes, float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, 4);
    put32(bytes, bits);
}
void chunk(std::vector<uint8_t> &dst, uint32_t id, const std::vector<uint8_t> &body)
{
    put32(dst, id);
    put32(dst, static_cast<uint32_t>(body.size()));
    dst.insert(dst.end(), body.begin(), body.end());
}
VisualRecord record(unsigned links)
{
    VisualRecord child;
    child.type = 5;
    std::vector<uint8_t> vertices, indices;
    constexpr uint32_t formats[]{0, 0x12071980, 0x240e3300, 0x481c6600, 0x5a238f80};
    put32(vertices, formats[links]);
    put32(vertices, 3);
    for (unsigned v = 0; v < 3; ++v)
    {
        if (links > 1)
            for (unsigned i = 0; i < links; ++i)
                put16(vertices, i);
        for (unsigned i = 0; i < 12; ++i)
            putfloat(vertices, i == 0 ? float(v) : 0);
        if (links > 1)
            for (unsigned i = 0; i + 1 < links; ++i)
                putfloat(vertices, links == 2 ? 0.25f : 1.0f / links);
        putfloat(vertices, 0.5f);
        putfloat(vertices, 0.75f);
        if (links == 1)
            put32(vertices, 3);
    }
    put32(indices, 3);
    put16(indices, 0);
    put16(indices, 1);
    put16(indices, 2);
    chunk(child.source, 3, vertices);
    chunk(child.source, 4, indices);
    VisualRecord root;
    root.type = 3;
    root.embedded_children.push_back(std::move(child));
    return root;
}
} // namespace
namespace
{
using Bytes = std::vector<uint8_t>;
void save(const std::filesystem::path &name, const Bytes &bytes)
{
    std::ofstream file(name, std::ios::binary);
    file.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    assert(file.good());
}
Bytes compressed(uint32_t id, Bytes body)
{
    CMemoryWriter writer;
    writer.w_chunk(id | CFS_CompressMark, body.data(), body.size());
    const auto *bytes = static_cast<const uint8_t *>(writer.pointer());
    return {bytes, bytes + writer.size()};
}
Bytes encoded(const VisualRecord &visual)
{
    Bytes header(44, 0), result;
    header[0] = 4;
    header[1] = visual.type;
    chunk(result, 1, header);
    Bytes texture{'f', 'i', 'x', 't', 'u', 'r', 'e', 0, 'd', 'e', 'f', 'a', 'u', 'l', 't', 0};
    chunk(result, 2, texture);
    result.insert(result.end(), visual.source.begin(), visual.source.end());
    if (!visual.embedded_children.empty())
    {
        Bytes children;
        unsigned id = 0;
        for (const auto &child : visual.embedded_children)
        {
            auto bytes = compressed(id++, encoded(child));
            children.insert(children.end(), bytes.begin(), bytes.end());
        }
        auto bytes = compressed(9, std::move(children));
        result.insert(result.end(), bytes.begin(), bytes.end());
    }
    return result;
}
VisualRecord plain(unsigned type)
{
    VisualRecord result;
    result.type = type;
    Bytes vb, ib, swi;
    put32(vb, 0x112);
    put32(vb, 3);
    for (unsigned v = 0; v < 3; ++v)
        for (unsigned f = 0; f < 8; ++f)
            putfloat(vb, f == 0 ? float(v) : f == 5 ? 1.f : 0.f);
    put32(ib, 3);
    put16(ib, 0);
    put16(ib, 1);
    put16(ib, 2);
    chunk(result.source, 3, vb);
    chunk(result.source, 4, ib);
    if (type == 2 || type == 4)
    {
        swi.resize(16);
        put32(swi, 1);
        put32(swi, 0);
        put16(swi, 1);
        put16(swi, 3);
        chunk(result.source, 6, swi);
    }
    return result;
}
} // namespace
int main()
{
    Core.Initialize("vulkan_asset_flow_test", nullptr, false);
    const auto directory = std::filesystem::temp_directory_path() / "openxray_vulkan_asset_flow";
    std::filesystem::create_directories(directory);
    const auto base_textures = directory / "base_textures";
    std::filesystem::create_directories(base_textures);
    gli::texture image(gli::TARGET_2D, gli::FORMAT_RGBA8_UNORM_PACK8, {4, 4, 1}, 1, 1, 1);
    std::memset(image.data(), 255, image.size());
    std::vector<char> dds;
    assert(gli::save_dds(image, dds));
    save(directory / "fixture.dds", Bytes(dds.begin(), dds.end()));
    save(directory / "hemi_lmap.dds", Bytes(dds.begin(), dds.end()));
    // Deliberately no color_lmap.dds: the CoP hemi path must not load slot 1.
    std::filesystem::remove(directory / "color_lmap.dds");
    FS.append_path("$level$", directory.string().c_str(), nullptr, false);
    FS.append_path("$game_textures$", base_textures.string().c_str(), nullptr, false);
    FS.append_path("$game_meshes$", directory.string().c_str(), nullptr, false);
    {
        const auto game_shaders = directory / "game_shaders";
        const auto engine_shaders = directory / "engine_shaders";
        std::filesystem::remove_all(game_shaders);
        std::filesystem::remove_all(engine_shaders);
        std::filesystem::create_directories(game_shaders / "vk");
        std::filesystem::create_directories(engine_shaders / "vk");
        save(engine_shaders / "vk" / "overlay.spv", Bytes{'o', 'v', 'e', 'r', 'l', 'a', 'y'});
        auto *shader_path = FS.append_path("$game_shaders$", game_shaders.string().c_str(), nullptr, true);
        shader_path->_set_overlay(engine_shaders.string().c_str());
        FS.rescan_path(shader_path->m_Overlay, true);

        uint64_t overlay_fingerprint = 0;
        std::string fingerprint_error;
        assert(shader_asset_fingerprint(overlay_fingerprint, fingerprint_error));
        assert(fingerprint_error.empty());

        save(engine_shaders / "vk" / "overlay.spv", Bytes{'c', 'h', 'a', 'n', 'g', 'e', 'd'});
        uint64_t changed_fingerprint = 0;
        assert(shader_asset_fingerprint(changed_fingerprint, fingerprint_error));
        assert(changed_fingerprint != overlay_fingerprint);

        // A loose game/mod shader with the same relative name must be the
        // effective input because r_open gives it priority over the overlay.
        save(game_shaders / "vk" / "overlay.spv", Bytes{'o', 'v', 'e', 'r', 'r', 'i', 'd', 'e'});
        FS.rescan_path(shader_path->m_Path, true);
        uint64_t override_fingerprint = 0;
        assert(shader_asset_fingerprint(override_fingerprint, fingerprint_error));
        assert(override_fingerprint != changed_fingerprint);
    }
    using namespace vk_mock;
    const auto device = handle<VkDevice>(1);
    const auto queue = handle<VkQueue>(2);
    const auto pool_handle = handle<VkCommandPool>(3);
    const auto rp = handle<VkRenderPass>(4);
    VkPhysicalDeviceMemoryProperties memory{};
    memory.memoryTypeCount = 1;
    memory.memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    DeferredPass pass;
    std::string error;
    assert(pass.initialize(device, rp, rp, handle<VkShaderModule>(5), handle<VkShaderModule>(6), handle<VkShaderModule>(7), handle<VkShaderModule>(8),
                           handle<VkShaderModule>(9), handle<VkShaderModule>(10), handle<VkShaderModule>(11), scene_dispatch(), error,
                           rp,handle<VkShaderModule>(12),handle<VkShaderModule>(13),handle<VkShaderModule>(14)));
    assert(pass.create_game_pipeline("vk\\object_opaque.vs", "vk\\object_opaque.ps", handle<VkShaderModule>(5), handle<VkShaderModule>(6), SurfaceMode::Opaque,
                                     false, error));
    for (unsigned weights = 1; weights <= 4; ++weights)
    {
        assert(pass.create_game_pipeline("vk\\skinned_" + std::to_string(weights) + ".vs", "vk\\skinned_opaque.ps", handle<VkShaderModule>(5),
                                         handle<VkShaderModule>(6), SurfaceMode::Opaque, false, error, true));
        assert(pass.create_game_pipeline("vk\\hud_skinned_" + std::to_string(weights) + ".vs", "vk\\hud_blended.ps", handle<VkShaderModule>(5),
                                         handle<VkShaderModule>(6), SurfaceMode::Opaque, true, error, true));
    }
    GameTextureFactory textures;
    {
        const auto thm = [](const char* detail, float scale, bool malformed = false)
        {
            Bytes data, parameters(malformed ? 7 : 8, 0), extension;
            if (!malformed) parameters[6] = 0x80; // diffuse detail
            extension.insert(extension.end(), detail, detail + std::strlen(detail) + 1);
            putfloat(extension, scale);
            chunk(data, 0x0812, parameters);
            chunk(data, 0x0815, extension);
            return data;
        };
        // A loose $level$ replacement takes priority over a game texture
        // metadata file with the same arbitrary name. Removing it restores
        // the lower-priority result; a malformed override must fail visibly.
        save(base_textures / "arb_42.thm", thm("base_detail", 1.25f));
        save(directory / "arb_42.thm", thm("mod_detail", 3.5f));
        FS.rescan_path(FS.get_path("$game_textures$")->m_Path, false);
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        GameTextureFactory::DetailMetadata detail;
        std::string metadata_error;
        assert(textures.texture_detail_metadata("arb_42", detail, metadata_error) &&
            detail.diffuse && detail.texture == "mod_detail" && detail.scale == 3.5f);
        save(directory / "arb_42.thm", thm("mod_detail", 3.5f, true));
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        assert(!textures.texture_detail_metadata("arb_42", detail, metadata_error) &&
            metadata_error.find("arb_42.thm") != std::string::npos &&
            metadata_error.find("TEXTUREPARAM") != std::string::npos);
        std::filesystem::remove(directory / "arb_42.thm");
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        assert(textures.texture_detail_metadata("arb_42", detail, metadata_error) &&
            detail.diffuse && detail.texture == "base_detail" && detail.scale == 1.25f);
        Bytes unnamed(8, 0), unnamed_thm;
        unnamed[6] = 0x80; // diffuse detail bit without DETAIL_EXT
        unnamed[7] = 0x04; // bump detail bit without DETAIL_EXT
        chunk(unnamed_thm, 0x0812, unnamed);
        save(directory / "unnamed_detail.thm", unnamed_thm);
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        assert(textures.texture_detail_metadata("unnamed_detail", detail, metadata_error) &&
            !detail.diffuse && !detail.normal && detail.texture.empty() && metadata_error.empty());
        std::filesystem::remove(directory / "alloy_d.thm");
        save(directory / "alloy_pbr.dds", Bytes{0});
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        assert(textures.texture_detail_metadata("alloy_d", detail, metadata_error) &&
            detail.pbr_texture == "alloy_pbr");
        Bytes pbr_thm, pbr_parameters(8, 0), pbr_name{'m','o','d','\\','p','a','c','k','e','d',0};
        chunk(pbr_thm, 0x0812, pbr_parameters);
        chunk(pbr_thm, 0x081e, pbr_name);
        save(directory / "alloy_d.thm", pbr_thm);
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        assert(textures.texture_detail_metadata("alloy_d", detail, metadata_error) &&
            detail.pbr_texture == "mod\\packed");
    }
    {
        Bytes params(8, 0), extension{'m','o','d','\\','d','e','t','a','i','l',0};
        params[6] = 0x80; // flDiffuseDetail (bit 23)
        putfloat(extension, 2.5f);
        GameTextureFactory::DetailMetadata metadata;
        assert(GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            extension.data(), extension.size(), metadata, error));
        assert(metadata.diffuse && !metadata.normal && metadata.texture == "mod\\detail" &&
            metadata.scale == 2.5f);
        params[7] = 0x04; // flBumpDetail (bit 26)
        assert(GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            extension.data(), extension.size(), metadata, error) && metadata.normal);
        // A CoP texture can retain detail bits without a named detail map.
        // The classic TextureDescrManager simply leaves its detail association
        // unset; both absent and empty DETAIL_EXT must behave the same here.
        assert(GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            nullptr, 0, metadata, error) && !metadata.diffuse && !metadata.normal &&
            metadata.texture.empty() && metadata.scale == 1.f && error.empty());
        Bytes empty_detail{0};
        putfloat(empty_detail, 3.f);
        assert(GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            empty_detail.data(), empty_detail.size(), metadata, error) &&
            !metadata.diffuse && !metadata.normal && metadata.texture.empty() &&
            metadata.scale == 3.f && error.empty());
        assert(!GameTextureFactory::decode_detail_metadata(params.data(), 7,
            extension.data(), extension.size(), metadata, error) &&
            error.find("TEXTUREPARAM") != std::string::npos);
        assert(!GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            extension.data(), 10, metadata, error) &&
            error.find("DETAIL_EXT") != std::string::npos);
        Bytes bump;
        putfloat(bump, .08f);
        bump.insert(bump.end(), {2, 0, 0, 0, 'm', 'o', 'd', '\\', 'n', 'o', 'r', 'm', 'a', 'l', 0});
        Bytes material{2, 0, 0, 0};
        putfloat(material, .4f);
        Bytes normal{'m', 'o', 'd', '\\', 'e', 'x', 't', 0};
        GameTextureFactory::MetadataExtras extras{bump.data(), bump.size(),
            material.data(), material.size(), normal.data(), normal.size()};
        Bytes packed{'p','b','r',0}, metallic{'m',0}, roughness{'r',0}, ao{'a',0}, parallax{'h',0};
        extras.pbr = packed.data(); extras.pbr_size = packed.size();
        extras.metallic = metallic.data(); extras.metallic_size = metallic.size();
        extras.roughness = roughness.data(); extras.roughness_size = roughness.size();
        extras.ao = ao.data(); extras.ao_size = ao.size();
        extras.parallax = parallax.data(); extras.parallax_size = parallax.size();
        assert(GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            extension.data(), extension.size(), metadata, error, extras));
        assert(metadata.bump_mode == 2 && metadata.bump_texture == "mod\\normal" &&
            metadata.external_normal_texture == "mod\\ext" && metadata.material == 2 &&
            metadata.material_weight > .39f && metadata.material_weight < .41f);
        assert(metadata.pbr_texture == "pbr" && metadata.metallic_texture == "m" &&
            metadata.roughness_texture == "r" && metadata.ao_texture == "a" && metadata.parallax_texture == "h");
        extras.pbr_size = packed.size() - 1;
        assert(!GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            extension.data(), extension.size(), metadata, error, extras) && error.find("PBR_PACKED") != std::string::npos);
        extras.pbr_size = packed.size();
        extras.bump_size = 8;
        assert(!GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            extension.data(), extension.size(), metadata, error, extras) &&
            error.find("BUMP") != std::string::npos);
        extras.bump_size = bump.size();
        extras.external_normal_size = normal.size() - 1;
        assert(!GameTextureFactory::decode_detail_metadata(params.data(), params.size(),
            extension.data(), extension.size(), metadata, error, extras) &&
            error.find("NORMALMAP") != std::string::npos);
    }
    VkPhysicalDeviceFeatures features{};
    VkPhysicalDeviceProperties properties{};
    assert(textures.initialize(device, queue, pool_handle, memory, false,
        features, properties, 1.0f, texture_dispatch(), sampler, destroy_sampler, idle, error));
    VkDescriptorSet wrap_set{}, clamp_set{};
    assert(textures.material("fixture", pass, wrap_set, error) &&
        textures.material("fixture", pass, clamp_set, error, true));
    assert(wrap_set && clamp_set && wrap_set != clamp_set);
    textures.release_material(wrap_set, pass);
    textures.release_material(clamp_set, pass);
    ScenePass ui_pass;
    assert(ui_pass.initialize(device, rp, handle<VkShaderModule>(5), handle<VkShaderModule>(6),
        handle<VkShaderModule>(7), handle<VkShaderModule>(8), scene_dispatch(), error, true));
    VkDescriptorSet ui_clamp{}, ui_wrap{};
    assert(textures.ui("fixture", ui_pass, ui_clamp, error) &&
        textures.ui("fixture", ui_pass, ui_wrap, error, nullptr, false));
    assert(ui_clamp && ui_wrap && ui_clamp != ui_wrap);
    assert(sampler_address.at(sampled_by.at(ui_clamp)) == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    assert(sampler_address.at(sampled_by.at(ui_wrap)) == VK_SAMPLER_ADDRESS_MODE_REPEAT);
    assert(textures.reload_assets(error));
    assert(sampler_address.at(sampled_by.at(ui_clamp)) == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    assert(sampler_address.at(sampled_by.at(ui_wrap)) == VK_SAMPLER_ADDRESS_MODE_REPEAT);
    assert(textures.retain_ui(ui_wrap, ui_pass));
    textures.release_ui(ui_clamp, ui_pass);
    textures.release_ui(ui_wrap, ui_pass);
    textures.release_ui(ui_wrap, ui_pass);
    std::array<std::string, SurfaceMaterialTextureCount> surface_names;
    surface_names.fill("fixture");
    SurfaceParameters surface_parameters{};
    VkDescriptorSet surface_wrap{}, surface_clamp{};
    assert(textures.surface_material(surface_names, surface_parameters, pass,
        surface_wrap, error) &&
        textures.surface_material(surface_names, surface_parameters, pass,
            surface_clamp, error, false, true));
    assert(surface_wrap && surface_clamp && surface_wrap != surface_clamp);
    assert(sampler_address.at(sampled_by.at(surface_wrap)) == VK_SAMPLER_ADDRESS_MODE_REPEAT);
    assert(sampler_address.at(sampled_by.at(surface_clamp)) == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    assert(textures.reload_assets(error));
    assert(sampler_address.at(sampled_by.at(surface_wrap)) == VK_SAMPLER_ADDRESS_MODE_REPEAT);
    assert(sampler_address.at(sampled_by.at(surface_clamp)) == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
    textures.release_terrain_material(surface_wrap, pass);
    textures.release_terrain_material(surface_clamp, pass);
    // All surface slots retain their own DDS across a reload. The key must
    // also distinguish a changed slot and per-material surface factors.
    std::array<std::string, SurfaceMaterialTextureCount> distinct_names{};
    for (size_t role = 0; role < distinct_names.size(); ++role)
    {
        distinct_names[role] = "surface_role_" + std::to_string(role);
        save(directory / (distinct_names[role] + ".dds"), Bytes(dds.begin(), dds.end()));
    }
    FS.rescan_path(FS.get_path("$level$")->m_Path, false);
    {
        gli::texture packed(gli::TARGET_2D, gli::FORMAT_RGBA8_SRGB_PACK8, {4, 4, 1}, 1, 1, 1);
        std::memset(packed.data(), 127, packed.size());
        std::vector<char> packed_dds;
        assert(gli::save_dds(packed, packed_dds));
        save(directory / "pbr_color_space.dds", Bytes(packed_dds.begin(), packed_dds.end()));
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        std::array<std::string, SurfaceMaterialTextureCount> names;
        names.fill("fixture");
        names[19] = "pbr_color_space";
        SurfaceParameters parameters;
        parameters.external_normal = 2.f;
        const size_t start = created_image_formats.size();
        VkDescriptorSet packed_set{}, color_set{};
        assert(textures.surface_material(names, parameters, pass, packed_set, error));
        assert(created_image_formats.size() > start && created_image_formats.back() == VK_FORMAT_R8G8B8A8_UNORM);
        assert(textures.material("pbr_color_space", pass, color_set, error));
        assert(created_image_formats.back() == VK_FORMAT_R8G8B8A8_SRGB);
        textures.release_terrain_material(packed_set, pass);
        textures.release_material(color_set, pass);
    }
    {
        VkDescriptorSet atlas{};
        assert(textures.material("fixture", pass, atlas, error, false, true));
        assert(atlas != wrap_set);
        assert(textures.reload_assets(error));
        textures.release_material(atlas, pass);
    }
    const size_t before_surface = textures.resident_count();
    VkDescriptorSet roles_a{}, roles_again{}, changed_parameter{}, changed_slot{};
    assert(textures.surface_material(distinct_names, surface_parameters, pass, roles_a, error));
    assert(textures.surface_material(distinct_names, surface_parameters, pass, roles_again, error));
    assert(roles_a && roles_a == roles_again &&
        textures.resident_count() == before_surface + distinct_names.size());
    surface_parameters.emissive = .3f;
    assert(textures.surface_material(distinct_names, surface_parameters, pass,
        changed_parameter, error) && changed_parameter != roles_a);
    surface_parameters.emissive = 0;
    auto swapped_names = distinct_names;
    std::swap(swapped_names[11], swapped_names[12]);
    assert(textures.surface_material(swapped_names, surface_parameters, pass,
        changed_slot, error) && changed_slot != roles_a && changed_slot != changed_parameter);
    assert(textures.reload_assets(error) &&
        textures.resident_count() == before_surface + distinct_names.size());
    VkDescriptorSet after_reload{};
    assert(textures.surface_material(distinct_names, surface_parameters, pass,
        after_reload, error) && after_reload == roles_a);
    textures.release_terrain_material(roles_again, pass);
    textures.release_terrain_material(after_reload, pass);
    textures.release_terrain_material(changed_parameter, pass);
    textures.release_terrain_material(changed_slot, pass);
    assert(textures.resident_count() == before_surface + distinct_names.size());
    textures.release_terrain_material(roles_a, pass);
    textures.retire_unused();
    assert(textures.resident_count() == before_surface);
    // The mock uses an actual serialized blender definition. Runtime loading
    // must never infer alpha/depth behavior from a texture or shader name.
    {
        Bytes blender(176, 0), records, library;
        const char cls[] = "LM      ";
        for (size_t i = 0; i < 8; ++i) blender[i] = cls[7 - i];
        std::memcpy(blender.data() + 8, "default", 8);
        blender[172] = 1;
        chunk(records, 0, blender);
        Bytes tracer(176,0);
        const char screen_cls[]="S_SET   ";
        for(size_t i=0;i<8;++i)tracer[i]=screen_cls[7-i];
        std::memcpy(tracer.data()+8,"mod_world_effect",17);tracer[172]=4;
        put32(tracer,0);const char base_scope[]="Base Texture";
        tracer.insert(tracer.end(),base_scope,base_scope+sizeof(base_scope));
        put32(tracer,7);const char blend_name[]="Blending";
        tracer.insert(tracer.end(),blend_name,blend_name+sizeof(blend_name));
        put32(tracer,5);put32(tracer,0);
        put32(tracer,6);const char ztest_name[]="Z-test";
        tracer.insert(tracer.end(),ztest_name,ztest_name+sizeof(ztest_name));put32(tracer,1);
        chunk(records,1,tracer);
        chunk(library, 2, records);
        IReader reader(library.data(), library.size());
        assert(textures.load_blender_library(reader, error));
    }
    FrameRecordingContext frame{handle<VkCommandBuffer>(12), rp, handle<VkFramebuffer>(13), {960, 540}, 0, 0};
    float mvp[16]{};
    {
        VulkanUIShader tracer(textures,ui_pass);tracer.create("mod_world_effect","fixture");
        assert(tracer.inited() && tracer.world_depth() && tracer.blend_mode()==5);
        VulkanUIShader copied(textures,ui_pass);copied.Copy(tracer);assert(copied.world_depth());
        VulkanUIRender world_ui;
        world_ui.configure(device,memory,upload_dispatch().buffer,ui_pass,
            {[&](VkDescriptorSet set){return textures.retain_ui(set,ui_pass);},
             [&](VkDescriptorSet set){textures.release_ui(set,ui_pass);}});
        world_ui.CreateUIGeom();Device.mFullTransform.identity();
        world_ui.StartPrimitive(3,IUIRender::ptTriList,IUIRender::pttLIT);
        world_ui.PushPoint(-1,1,.5f,0xffffffffu,0,0);
        world_ui.PushPoint(3,1,.5f,0xffffffffu,1,0);
        world_ui.PushPoint(-1,-3,.5f,0xffffffffu,0,1);
        // Match CBulletManager: shader is selected only after vertex submission.
        world_ui.SetShader(tracer);world_ui.FlushPrimitive();
        const auto before=draws;
        assert(world_ui.record(frame,error) && draws==before); // overlay excludes world effects
        assert(!world_ui.record(frame,error,false,VK_NULL_HANDLE,VK_NULL_HANDLE,true));
        assert(world_ui.record(frame,error,false,VK_NULL_HANDLE,tracer.descriptor(),true) && draws==before+1);
        world_ui.DestroyUIGeom();copied.destroy();tracer.destroy();
    }
    mvp[0] = mvp[5] = mvp[10] = mvp[15] = 1;
    assert(pass.create_game_pipeline("vk\\level_opaque.vs", "vk\\level_opaque.ps", handle<VkShaderModule>(5), handle<VkShaderModule>(6), SurfaceMode::Opaque,
                                     false, error));
    assert(pass.create_game_pipeline("vk\\level_lod.vs", "vk\\level_lod.ps", handle<VkShaderModule>(5), handle<VkShaderModule>(6), SurfaceMode::AlphaTest,
                                     false, error));
    for (auto profile : {LevelGameProfile::SoC, LevelGameProfile::CS, LevelGameProfile::CoP})
    {
        ShadowOfChernobylMode = profile == LevelGameProfile::SoC;
        ClearSkyMode = profile == LevelGameProfile::CS;
        CallOfPripyatMode = profile == LevelGameProfile::CoP;
        Bytes vb, ib, geom, level, shaders, visual, container, header(44, 0), table;
        put32(vb, 1);
        auto decl = [&](uint16_t offset, uint8_t type, uint8_t usage, uint8_t index=0) {
            vb.insert(vb.end(), {0, 0, uint8_t(offset), uint8_t(offset >> 8), type, 0, usage, index});
        };
        decl(0, 2, 0);
        decl(12, 2, 3);
        decl(24, 1, 5);
        if(profile==LevelGameProfile::CoP) decl(32,1,5,1);
        vb.insert(vb.end(), {0xff, 0, 0, 0, 17, 0, 0, 0});
        put32(vb, 3);
        for (unsigned v = 0; v < 3; ++v)
            for (unsigned f = 0; f < (profile==LevelGameProfile::CoP?10u:8u); ++f)
                putfloat(vb, f == 0 ? float(v) : f == 5 ? 1.f : 0.f);
        put32(ib, 1);
        put32(ib, 3);
        put16(ib, 0);
        put16(ib, 1);
        put16(ib, 2);
        chunk(geom, 9, vb);
        chunk(geom, 10, ib);
        save(directory / "level.geom", geom);
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        put32(shaders, 1);
        const std::string material = profile==LevelGameProfile::CoP?
            "default/fixture, color_lmap, hemi_lmap":"default/fixture";
        shaders.insert(shaders.end(),material.c_str(),material.c_str()+material.size()+1);
        header[0] = 4;
        chunk(visual, 1, header);
        for (unsigned n : {0u, 0u, 3u, 0u, 0u, 3u})
            put32(container, n);
        chunk(visual, 21, container);
        if(profile==LevelGameProfile::CoP) {
            Bytes fast_vertices,fast_geometry,fast_chunks;
            put32(fast_vertices,1);fast_vertices.insert(fast_vertices.end(),{0,0,0,0,2,0,0,0,0xff,0,0,0,17,0,0,0});
            put32(fast_vertices,3);for(unsigned v=0;v<3;++v) {
                putfloat(fast_vertices,.25f*(v+1));putfloat(fast_vertices,0);putfloat(fast_vertices,0);
            }
            chunk(fast_geometry,9,fast_vertices);chunk(fast_geometry,10,ib);
            save(directory/"level.geomX",fast_geometry);
            chunk(fast_chunks,21,container);chunk(visual,22,fast_chunks);
            FS.rescan_path(FS.get_path("$level$")->m_Path,false);
        }
        table = compressed(0, visual);
        chunk(level, 1, Bytes{14, 0, 1, 0});
        chunk(level, 2, shaders);
        auto packed = compressed(3, table);
        level.insert(level.end(), packed.begin(), packed.end());
        IReader reader(level.data(), level.size());
        LevelModelData data;
        assert(load_engine_level_models(reader, data, error) && data.models.size() == 1);
        std::vector<VisualRecord> catalog;
        assert(load_engine_visual_catalog(reader, catalog, error) && catalog.size() == 1);
        if(profile==LevelGameProfile::CoP)
            assert(pass.create_game_pipeline("vk\\level_lightmap.vs","vk\\level_lightmap.ps",
                handle<VkShaderModule>(5),handle<VkShaderModule>(6),SurfaceMode::Opaque,false,error));
        GpuLevel gpu;
        const bool level_loaded = gpu.load(reader, device, queue, pool_handle, memory, upload_dispatch(), textures, pass, error);
        if (!level_loaded)
            std::fprintf(stderr, "vulkan_asset_flow: GpuLevel::load failed: %s\n", error.c_str());
        assert(level_loaded);
        assert(gpu.visual_count() == 1 && gpu.model_count() == 1 && textures.finish_uploads());
        GpuLevel::DetailGroundSource empty_ground;
        assert(gpu.detail_ground_source(0,empty_ground,error) && !empty_ground.cell_count && !empty_ground.cells);
        assert(gpu.retire_completed_uploads());
        if (profile == LevelGameProfile::SoC)
        {
            const auto resident = textures.resident_count();
            const auto live_images = images.size();
            VkImageView weather{};
            assert(textures.environment("fixture", weather, error));
            const VkImageView original_weather = weather;
            unsigned callbacks = 0;
            const auto before = [&] { ++callbacks; textures.release_environment(weather); weather = VK_NULL_HANDLE; };
            const auto after = [&] { ++callbacks; assert(textures.environment("fixture", weather, error)); };
            save(directory / "fixture.dds", Bytes{1, 2, 3, 4});
            assert(!textures.reload_assets(error, before, after));
            assert(callbacks == 0 && weather == original_weather);
            assert(textures.resident_count() == resident && images.size() == live_images);
            save(directory / "fixture.dds", Bytes(dds.begin(), dds.end()));
            assert(textures.reload_assets(error, before, after) && error.empty());
            assert(callbacks == 2 && weather && weather != original_weather);
            assert(textures.resident_count() == resident && images.size() == live_images);
            textures.release_environment(weather);
        }
        const auto before = draws;
        assert(gpu.record(frame, pass, mvp) && draws == before + 1);
        if(profile==LevelGameProfile::CoP) {
            // The shaded low LOD must keep regular UV/lightmap vertices;
            // the shadow pass uses the position-only proxy in the same batch.
            assert(gpu.record_visual(0,frame,pass,mvp,GeometryPhase::OpaqueAndAlphaTest,0.f,false));
            assert(last_vertex_base==0);
            assert(gpu.record_sun_shadow(frame,pass,mvp));
            assert(last_vertex_base==3);
            auto vsm_frame = frame;
            vsm_frame.render_pass = handle<VkRenderPass>(777);
            vsm_frame.viewport_offset = {8, 0};
            VsmLevelPageContext vsm_context{&gpu, &pass};
            const auto shadow_draws = draws;
            assert(!record_vsm_level_page(vsm_frame, mvp, {}, &vsm_context, error));
            pass.set_vsm_shadow_pass(vsm_frame.render_pass);
            assert(record_vsm_level_page(vsm_frame, mvp, {}, &vsm_context, error));
            assert(draws > shadow_draws);
            const auto before_tree_filter = draws;
            vsm_context.include_static = false;
            assert(record_vsm_level_page(vsm_frame, mvp, {}, &vsm_context, error));
            assert(draws == before_tree_filter);
            vsm_context.include_static = true;
            vsm_context.include_trees = false;
            assert(record_vsm_level_page(vsm_frame, mvp, {}, &vsm_context, error));
            assert(draws > before_tree_filter);
            TextureMatrixSet shadow_matrix{};
            shadow_matrix.enabled[0] = true;
            shadow_matrix.values[0].mode = 2;
            const Fmatrix previous_view = Device.mView;
            Device.mView.identity();
            const VkBuffer fixture_buffer = buffers.begin()->first;
            assert(pass.record_sun_shadow(vsm_frame, fixture_buffer, fixture_buffer, 3, mvp,
                bound_textures.back(), true, 0, .5f, 0, VK_NULL_HANDLE, false, &shadow_matrix));
            assert(last_texture_rows[0] == .5f && last_texture_rows[3] == .5f &&
                last_texture_rows[5] == -.5f && last_texture_rows[7] == .5f);
            Device.mView = previous_view;
            const auto static_pass = vsm_frame.render_pass;
            vsm_frame.render_pass = handle<VkRenderPass>(778);
            pass.set_vsm_shadow_pass(static_pass, vsm_frame.render_pass);
            assert(record_vsm_level_page(vsm_frame, mvp, {}, &vsm_context, error));
            pass.set_vsm_shadow_pass(VK_NULL_HANDLE);
            assert(!record_vsm_level_page(vsm_frame, mvp, {}, &vsm_context, error));
            assert(gpu.record_hud_visual(0,frame,pass,mvp,0.f));
            assert(last_vertex_base==0); // HUD must not sample a depth proxy either.
        }
        if(profile==LevelGameProfile::CoP) {
            const auto set=bound_textures.back();
            assert(sampler_address.at(sampled_bindings.at(set).at(0))==VK_SAMPLER_ADDRESS_MODE_REPEAT);
            const auto hemi_sampler=sampled_bindings.at(set).at(1);
            assert(sampler_address.at(hemi_sampler)==VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
            assert(sampler_max_lod.at(hemi_sampler)==0.f);
        }
        // A mod may supply an incomplete optional geomX. Keep the valid
        // primary geometry for each profile and report the fallback at VFS load.
        Bytes partial_fast;
        chunk(partial_fast, 9, vb);
        save(directory / "level.geomX", partial_fast);
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        LevelModelData fallback;
        assert(load_engine_level_models(reader, fallback, error) &&
            fallback.models.size() == 1 && fallback.models[0].indices.size() == 3);
        std::filesystem::remove(directory / "level.geomX");
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        // A missing VB chunk must leave the uploaded scene available.
        Bytes missing;
        chunk(missing, 10, ib);
        save(directory / "level.geom", missing);
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        assert(!gpu.load(reader, device, queue, pool_handle, memory, upload_dispatch(), textures, pass, error));
        assert(error.find(level_profile_name(profile)) != std::string::npos && error.find("level.geom") != std::string::npos);
        assert(gpu.visual_count() == 1 && gpu.record(frame, pass, mvp));
        gpu.destroy();
        assert(buffers.empty() && allocations.empty() && images.empty());
    }
    {
        // Exercise the real GpuLevel upload/packing/selection path. The CPU
        // cluster index vectors have already been released when velocity runs.
        ShadowOfChernobylMode = true;
        ClearSkyMode = CallOfPripyatMode = false;
        constexpr unsigned cells = 24, side = cells + 1;
        constexpr unsigned vertex_count = side * side, index_count = cells * cells * 6;
        Bytes vb, ib, geom, shaders, container;
        put32(vb, 1);
        for (auto declaration : {std::array<uint8_t, 8>{0, 0, 0, 0, 2, 0, 0, 0},
                std::array<uint8_t, 8>{0, 0, 12, 0, 2, 0, 3, 0},
                std::array<uint8_t, 8>{0, 0, 24, 0, 1, 0, 5, 0},
                std::array<uint8_t, 8>{0xff, 0, 0, 0, 17, 0, 0, 0}})
            vb.insert(vb.end(), declaration.begin(), declaration.end());
        put32(vb, vertex_count);
        for (unsigned y = 0; y < side; ++y)
            for (unsigned x = 0; x < side; ++x)
            {
                const float u = float(x) / cells, v = float(y) / cells;
                for (float value : {-0.8f + 1.6f * u, -0.8f + 1.6f * v,
                        0.5f + 0.15f * std::sin(u * 6.2831853f) * std::cos(v * 6.2831853f),
                        0.f, 0.f, 1.f, u, v})
                    putfloat(vb, value);
            }
        put32(ib, 1); put32(ib, index_count);
        for (unsigned y = 0; y < cells; ++y)
            for (unsigned x = 0; x < cells; ++x)
            {
                const unsigned a = y * side + x, b = a + 1, c = a + side, d = c + 1;
                for (unsigned index : {a, b, c, b, d, c}) put16(ib, static_cast<uint16_t>(index));
            }
        chunk(geom, 9, vb); chunk(geom, 10, ib);
        save(directory / "level.geom", geom);
        FS.rescan_path(FS.get_path("$level$")->m_Path, false);
        put32(shaders, 1);
        const char material[] = "default/fixture";
        shaders.insert(shaders.end(), material, material + sizeof(material));
        for (unsigned value : {0u, 0u, vertex_count, 0u, 0u, index_count}) put32(container, value);
        ObjectVelocityPass velocity;
        assert(velocity.initialize(device, velocity_proc, frame.extent, VK_FORMAT_D32_SFLOAT,
            {handle<VkImageView>(41)}, {handle<VkImageView>(42)}, VK_NULL_HANDLE, error));
        const auto old_camera = Device.vCameraPosition;
        const float old_fov = Device.fFOV;
        Device.vCameraPosition.set(0.f, 0.f, -2.f);
        Device.fFOV = 60.f;
        for (uint8_t type : {uint8_t(0), uint8_t(2)})
        {
            Bytes header(44, 0), visual, visuals, level;
            header[0] = 4; header[1] = type;
            chunk(visual, 1, header); chunk(visual, 21, container);
            if (type == 2)
            {
                Bytes windows(16, 0);
                put32(windows, 2);
                put32(windows, 0); put16(windows, index_count / 3); put16(windows, vertex_count);
                put32(windows, 0); put16(windows, 0); put16(windows, 2);
                chunk(visual, 6, windows);
            }
            chunk(visuals, 0, visual);
            chunk(level, 1, Bytes{14, 0, 1, 0});
            chunk(level, 2, shaders); chunk(level, 3, visuals);
            IReader reader(level.data(), level.size());
            GpuLevel gpu;
            char cluster_flag[] = "-vk-cluster-dag";
            char* old_params = Core.Params;
            Core.Params = cluster_flag;
            const bool loaded = gpu.load(reader, device, queue, pool_handle, memory, upload_dispatch(), textures, pass, error);
            Core.Params = old_params;
            assert(loaded && gpu.retire_completed_uploads() && !gpu.has_cluster_visibility());
            indexed_draws.clear();
            assert(gpu.record_visual(0, frame, pass, mvp, GeometryPhase::OpaqueAndAlphaTest, 1.f, false));
            const auto shaded_draws = indexed_draws;
            assert(shaded_draws.size() > 1); // Confirm actual cluster selection.
            indexed_draws.clear();
            assert(velocity.begin(frame));
            assert(gpu.record_object_velocity_visual(0, frame, velocity, mvp, mvp));
            velocity.end(frame);
            assert(indexed_draws.size() == shaded_draws.size());
            for (size_t i = 0; i < shaded_draws.size(); ++i)
            {
                assert(indexed_draws[i].indices == shaded_draws[i].indices);
                assert(indexed_draws[i].count == shaded_draws[i].count);
                assert(indexed_draws[i].first_index == shaded_draws[i].first_index);
                assert(indexed_draws[i].vertex_base == shaded_draws[i].vertex_base);
            }
            if (type == 2)
            {
                indexed_draws.clear();
                assert(gpu.record_visual(0, frame, pass, mvp, GeometryPhase::OpaqueAndAlphaTest, 0.f, false));
                assert(gpu.record_hud_visual(0, frame, pass, mvp, 0.f));
                assert(velocity.begin(frame));
                assert(gpu.record_object_velocity_visual(0, frame, velocity, mvp, mvp, 0.f));
                velocity.end(frame);
                assert(indexed_draws.empty());
            }
            gpu.destroy();
            assert(buffers.empty() && allocations.empty() && images.empty());
        }
        Device.vCameraPosition = old_camera;
        Device.fFOV = old_fov;
        CallOfPripyatMode = true;
        ShadowOfChernobylMode = false;
    }
    for (unsigned type = 0; type <= 5; ++type)
        for (unsigned weights = 1; weights <= (type >= 3 ? 4u : 1u); ++weights)
        {
            VisualRecord source = type >= 3 ? record(weights) : plain(type == 1 ? 0 : type);
            if (type == 1)
            {
                VisualRecord hierarchy;
                hierarchy.type = 1;
                hierarchy.embedded_children.push_back(source);
                source = std::move(hierarchy);
            }
            if (type == 4 || type == 5)
            {
                auto leaf = std::move(source.embedded_children[0]);
                source = std::move(leaf);
                source.type = type;
                if (type == 4)
                {
                    Bytes swi(16);
                    put32(swi, 1);
                    put32(swi, 0);
                    put16(swi, 1);
                    put16(swi, 3);
                    chunk(source.source, 6, swi);
                }
            }
            auto bytes = encoded(source);
            save(directory / "matrix.ogf", bytes);
            FS.rescan_path(FS.get_path("$level$")->m_Path, false);
            VisualRecord parsed;
            if (!load_engine_model_visual("matrix.ogf", parsed, error))
            {
                std::fprintf(stderr, "type=%u: %s\n", type, error.c_str());
                assert(false);
            }
            assert(parsed.type == type);
            auto gpu = std::make_shared<GpuModel>();
            assert(gpu->load_record(parsed, "fixture", device, queue, pool_handle, memory, upload_dispatch(), textures, pass, error,
                &model_ray_address));
            std::unique_ptr<GpuModel> raster_gpu;
            if (type == 4)
            {
                raster_gpu = std::make_unique<GpuModel>();
                assert(raster_gpu->load_record(parsed, "fixture", device, queue, pool_handle, memory,
                    upload_dispatch(), textures, pass, error));
            }
            std::vector<RayModelGeometry> ray_geometry;
            assert(gpu->collect_static_ray_geometry(ray_geometry, error));
            if (type <= 2)
            {
                assert(ray_geometry.size() == 1 && ray_geometry[0].primitive_count == 1);
                assert(ray_geometry[0].diffuse_texture == "fixture" && ray_geometry[0].alpha_cutoff == 0.f);
                assert(ray_geometry[0].geometry.geometry.triangles.vertexStride == sizeof(LevelVertex));
                RayModelTextureSlots slots{};
                slots.material_id = 17;
                slots.alpha = 1; slots.diffuse = 2; slots.pbr = 3; slots.normal = 4; slots.normal_aux = 5;
                RayGeometryBatch batch;
                const VkTransformMatrixKHR transform{{{1.f, 0.f, 0.f, 2.f}, {0.f, 1.f, 0.f, 0.f}, {0.f, 0.f, 1.f, 0.f}}};
                assert(make_model_ray_batch(4096, transform, ray_geometry, {slots}, batch, error));
                assert(batch.external_blas == 4096 && batch.transform.matrix[0][3] == 2.f &&
                    batch.material_ids[0] == 17 && batch.diffuse_texture_slots[0] == 2 &&
                    batch.pbr_texture_slots[0] == 3 && batch.normal_texture_slots[0] == 4 &&
                    batch.normal_aux_texture_slots[0] == 5);
                slots.diffuse = UINT32_MAX;
                assert(!make_model_ray_batch(4096, transform, ray_geometry, {slots}, batch, error));
                assert(batch.external_blas == 4096 && batch.diffuse_texture_slots[0] == 2);
            }
            else assert(ray_geometry.empty());
            assert(textures.finish_uploads());
            assert(gpu->retire_completed_uploads());
            if (raster_gpu)
                assert(raster_gpu->retire_completed_uploads());
            auto instance_record = parsed;
            if (type >= 3)
                instance_record.type = 3; // skinned leaves are owned by a skeleton
            auto visual = std::make_unique<VulkanModelVisual>(instance_record, "matrix.ogf", gpu);
            if (type >= 3)
            {
                Bytes names;
                put32(names, 4);
                for (unsigned bone = 0; bone < 4; ++bone)
                {
                    const auto name = std::string("bone") + std::to_string(bone);
                    const std::string parent = bone ? "bone0" : "";
                    names.insert(names.end(), name.begin(), name.end());
                    names.push_back(0);
                    names.insert(names.end(), parent.begin(), parent.end());
                    names.push_back(0);
                    names.resize(names.size() + 60);
                }
                Bytes ogf;
                chunk(ogf, 13, names);
                SkeletonBones bones;
                assert(parse_skeleton_bones({ogf.data(), ogf.size()}, bones, error));
                auto skeleton = VulkanKinematics::create(bones, visual.get(), error);
                assert(skeleton);
                ModelGeometry geometry;
                assert(decode_model_geometry(parsed, geometry, error));
                if (type != 3)
                {
                    ModelGeometry root;
                    root.type = 3;
                    root.children.push_back(std::move(geometry));
                    geometry = std::move(root);
                }
                assert(skeleton->attach_geometry(geometry, error));
                visual->set_skeleton(std::move(skeleton));
                if (raster_gpu)
                {
                    const auto before_raster = draws;
                    assert(raster_gpu->record_animated(frame, pass, mvp, *visual->skeleton(), error));
                    assert(draws == before_raster + 1);
                }
            }
            if (type >= 3)
            {
                AccelerationStructureDispatch dispatch{};
                dispatch.buffers = buffer_dispatch();
                dispatch.get_buffer_address = model_buffer_address;
                dispatch.create = create_model_as;
                dispatch.destroy = destroy_model_as;
                dispatch.get_build_sizes = model_as_sizes;
                dispatch.cmd_build = model_as_build;
                dispatch.get_address = model_as_address;
                VkPhysicalDeviceAccelerationStructurePropertiesKHR properties{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
                properties.maxGeometryCount = properties.maxPrimitiveCount = 1024;
                properties.minAccelerationStructureScratchOffsetAlignment = 16;
                const auto builds_before = model_ray_builds, updates_before = model_ray_updates;
                VkDeviceAddress addresses[2]{};
                const std::vector<RayModelGeometry>* skin_geometry = nullptr;
                bool updated = false;
                for (uint32_t slot = 0; slot < 2; ++slot)
                {
                    assert(gpu->prepare_skinned_ray_blas(slot, frame.command_buffer, *visual->skeleton(),
                        dispatch, properties, addresses[slot], skin_geometry, updated, error));
                    assert(addresses[slot] && skin_geometry && skin_geometry->size() == 1 && !updated);
                }
                assert(addresses[0] != addresses[1] && model_ray_builds == builds_before + 2);
                VkDeviceAddress updated_address{};
                assert(gpu->prepare_skinned_ray_blas(0, frame.command_buffer, *visual->skeleton(),
                    dispatch, properties, updated_address, skin_geometry, updated, error));
                assert(updated && updated_address == addresses[0] && model_ray_updates == updates_before + 1);
            }
            const auto before = draws;
            for (unsigned slot = 0; slot < FrameContext::FramesInFlight; ++slot)
            {
                frame.frame_index = slot;
                if (type >= 3)
                    assert(gpu->record_animated(frame, pass, mvp, *visual->skeleton(), error));
                else
                    assert(gpu->record(frame, pass, mvp, nullptr, 0, error));
            }
            assert(draws == before + FrameContext::FramesInFlight);
            auto copy = std::make_unique<VulkanModelVisual>(*visual);
            if (type >= 3)
            {
                assert(copy->skeleton() != visual->skeleton());
                assert(gpu->record_animated(frame, pass, mvp, *copy->skeleton(), error));
                const auto before_hud = draws;
                assert(gpu->record_animated(frame, pass, mvp, *visual->skeleton(), error, GeometryPhase::Hud));
                assert(gpu->record_animated(frame, pass, mvp, *copy->skeleton(), error, GeometryPhase::Hud));
                assert(draws == before_hud + 2); // All 1–4 weight GPU HUD variants.
                const auto pool_buffers=buffers.size();
                const auto pool_frees=descriptor_frees;
                const auto pool_destroys=buffers_destroyed;
                visual->reset_instance_state(); // Return/reuse keeps GPU pose ownership.
                for(unsigned slot=0;slot<FrameContext::FramesInFlight;++slot) {
                    frame.frame_index=slot;
                    assert(gpu->record_animated(frame,pass,mvp,*visual->skeleton(),error));
                }
                assert(buffers.size()==pool_buffers && descriptor_frees==pool_frees && buffers_destroyed==pool_destroys);
                const auto ray_destroyed_before = model_ray_destroyed;
                visual->release_pose_buffers();
                assert(model_ray_destroyed == ray_destroyed_before + 2);
                assert(gpu->record_animated(frame, pass, mvp, *copy->skeleton(), error));
                visual->reset_instance_state();
                assert(!visual->skeleton()->GetUpdateTracksCalback());
                assert(gpu->record_animated(frame, pass, mvp, *visual->skeleton(), error, GeometryPhase::Hud));
                for (int spawn = 0; spawn < 3; ++spawn)
                {
                    auto replacement = std::make_unique<VulkanModelVisual>(*visual);
                    assert(replacement->skeleton() != visual->skeleton());
                    assert(gpu->record_animated(frame, pass, mvp, *replacement->skeleton(), error,
                        GeometryPhase::Hud));
                    replacement->release_pose_buffers();
                    replacement.reset();
                }
            }
            if (raster_gpu)
            {
                raster_gpu->release_instance(visual->skeleton());
                raster_gpu.reset();
            }
            gpu.reset();
            visual.reset();
            assert(!buffers.empty());
            copy.reset();
            assert(buffers.empty() && allocations.empty() && images.empty() && textures.resident_count() == 0);
            // Corrupt data must include filename and type/id and preserve output.
            if (type == 0 || type == 2 || type == 4 || type == 5)
            {
                bytes.pop_back();
                IReader reader(bytes.data(), bytes.size());
                ModelGeometry previous;
                previous.type = 99;
                assert(!load_engine_model_geometry("broken.ogf", &reader, previous, error));
                assert(error.find("broken.ogf") != std::string::npos && previous.type == 99);
            }
        }
    // Nested compression is normalized through IReader, including empty chunks.
    auto nested = encoded(record(1));
    IReader compressed_reader(nested.data(), nested.size());
    VisualRecord normalized;
    assert(load_engine_model_reader(compressed_reader, "compressed.ogf", normalized, error));
    assert(normalized.embedded_children.size() == 1);
    auto oversized = compressed(1, Bytes(44, 0));
    oversized[8] = 0xff;
    oversized[9] = 0xff;
    oversized[10] = 0xff;
    oversized[11] = 0x7f;
    IReader invalid_compressed(oversized.data(), oversized.size());
    assert(!load_engine_model_reader(invalid_compressed, "bad-compression.ogf", normalized, error));
    assert(error.find("bad-compression.ogf") != std::string::npos && error.find("id=1") != std::string::npos);
    textures.destroy();
    pass.destroy();
    assert(buffers.empty() && allocations.empty() && images.empty() && submits > 0 && descriptor_frees > 0);
    std::filesystem::remove_all(directory);
    Core._destroy();
}
