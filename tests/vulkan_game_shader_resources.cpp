#include "src/Layers/xrRenderVK/GameShaderResources.h"

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace xray::render::vulkan;

namespace
{
template <typename T> T handle(uintptr_t value)
{
    if constexpr (std::is_pointer_v<T>) return reinterpret_cast<T>(value);
    else return static_cast<T>(value);
}

unsigned created{}, destroyed{}, source_reads{};
bool shader_changed{};
unsigned compute_pipelines{};
unsigned failed_pipeline_cleanups{};
VkResult VKAPI_PTR create_module(VkDevice, const VkShaderModuleCreateInfo* info,
    const VkAllocationCallbacks*, VkShaderModule* module)
{
    assert(info->codeSize >= 10 * sizeof(uint32_t));
    *module = handle<VkShaderModule>(100 + ++created);
    return VK_SUCCESS;
}
void VKAPI_PTR destroy_module(VkDevice, VkShaderModule, const VkAllocationCallbacks*)
{
    ++destroyed;
}
VkResult VKAPI_PTR create_compute(VkDevice, VkPipelineCache, uint32_t count,
    const VkComputePipelineCreateInfo* info, const VkAllocationCallbacks*, VkPipeline* pipeline)
{
    assert(count == 1 && info->layout == handle<VkPipelineLayout>(7));
    assert(info->stage.stage == VK_SHADER_STAGE_COMPUTE_BIT && info->stage.module);
    assert(std::strcmp(info->stage.pName, "main") == 0);
    *pipeline = handle<VkPipeline>(200 + ++compute_pipelines);
    return VK_SUCCESS;
}
VkResult VKAPI_PTR fail_compute(VkDevice, VkPipelineCache, uint32_t,
    const VkComputePipelineCreateInfo*, const VkAllocationCallbacks*, VkPipeline* pipeline)
{
    *pipeline = handle<VkPipeline>(999);
    return VK_ERROR_INITIALIZATION_FAILED;
}
void VKAPI_PTR destroy_pipeline(VkDevice, VkPipeline pipeline, const VkAllocationCallbacks*)
{
    assert(pipeline == handle<VkPipeline>(999));
    ++failed_pipeline_cleanups;
}
}

int main()
{
    std::vector<std::string> effective_defines;
    std::string define_error;
    const std::vector<std::string> engine_defines{
        "XRAY_VULKAN", "XRAY_VK_DEVICE_ID=42", "SKIN_NONE"};
    const std::vector<std::string> mod_overrides{
        "XRAY_VK_DEVICE_ID=77", "MOD_FEATURE=1"};
    assert(GameShaderResources::merge_defines(engine_defines, mod_overrides,
        effective_defines, define_error));
    assert((effective_defines == std::vector<std::string>{
        "XRAY_VULKAN", "XRAY_VK_DEVICE_ID=77", "SKIN_NONE", "MOD_FEATURE=1"}));
    assert(!GameShaderResources::merge_defines({"XRAY_VULKAN", "XRAY_VULKAN"}, {},
        effective_defines, define_error));
    assert(define_error.find("duplicate base preprocessor define") != std::string::npos);
    GameShaderResources variant_defines;
    variant_defines.set_engine_defines(engine_defines);
    assert(GameShaderResources::has_manifest_variant("vk\\level_opaque.vs",
        GameShaderStage::Vertex, "main"));
    assert(variant_defines.pipeline_defines("vk\\level_opaque.vs", GameShaderStage::Vertex,
        "main", mod_overrides, effective_defines, define_error));
    assert(effective_defines == mod_overrides);
    assert(variant_defines.pipeline_defines("vk\\mods\\weapon.vs", GameShaderStage::Vertex,
        "main", mod_overrides, effective_defines, define_error));
    assert((effective_defines == std::vector<std::string>{
        "XRAY_VULKAN", "XRAY_VK_DEVICE_ID=77", "SKIN_NONE", "MOD_FEATURE=1"}));

    struct StageCase { GameShaderStage stage; const char* suffix; const char* target; };
    const std::array<StageCase, 6> stages{{
        {GameShaderStage::Vertex, ".vs", "vs_5_0"},
        {GameShaderStage::Hull, ".hs", "hs_5_0"},
        {GameShaderStage::Domain, ".ds", "ds_5_0"},
        {GameShaderStage::Geometry, ".gs", "gs_5_0"},
        {GameShaderStage::Pixel, ".ps", "ps_5_0"},
        {GameShaderStage::Compute, ".cs", "cs_6_0"},
    }};
    GameShaderStage parsed{};
    GameShaderResources resources;
    resources.configure(handle<VkDevice>(1), {create_module, destroy_module},
        [&](const std::string& name, std::vector<uint8_t>& bytes, std::string& error) {
            if (name.size() >= 9 && name.compare(name.size() - 9, 9, ".spv.meta") == 0)
            {
                error = "no metadata for test shader";
                return false;
            }
            ++source_reads;
            uint32_t model = UINT32_MAX;
            for (const auto& item : stages)
                if (name.find(std::string(item.suffix) + ".spv") != std::string::npos)
                    model = static_cast<uint32_t>(item.stage);
            if (name.find("broken") != std::string::npos) model = 0;
            if (model == UINT32_MAX) { error = "missing test shader"; return false; }
            const uint32_t code[]{0x07230203, 0x00010000, shader_changed ? 1u : 0u, 2, 0,
                0x0005000f, model, 1, 0x6e69616d, 0};
            bytes.resize(sizeof(code));
            std::memcpy(bytes.data(), code, sizeof(code));
            return true;
        });
    std::string error;
    for (const auto& item : stages)
    {
        assert(GameShaderResources::stage_from_target(item.target, parsed) && parsed == item.stage);
        std::string canonical;
        assert(GameShaderResources::canonical_name("mods/effect", item.stage, canonical, error));
        assert(canonical == std::string("mods\\effect") + item.suffix);
        GameShaderResource* shader = nullptr;
        assert(resources.shader("mods/effect", item.stage, "main", 0, shader, error));
        assert(shader && shader->stage == item.stage);
        assert(shader->reflection.stage == static_cast<uint32_t>(item.stage) &&
            shader->reflection.descriptors.empty() && !shader->reflection.push_constants);
        GameShaderResource* reused = nullptr;
        assert(resources.shader(canonical + ".spv", item.stage, "main", 0, reused, error));
        assert(reused == shader);
    }
    assert(created == stages.size() && source_reads == stages.size() * 2);
    VkPipeline pipeline = VK_NULL_HANDLE;
    assert(resources.compute_pipeline("mods/effect", VK_NULL_HANDLE,
        handle<VkPipelineLayout>(7), create_compute, destroy_pipeline, pipeline, error));
    assert(pipeline == handle<VkPipeline>(201) && compute_pipelines == 1);
    assert(created == stages.size() && source_reads == stages.size() * 2 + 1);
    shader_changed = true;
    GameShaderResource* stale = nullptr;
    assert(!resources.shader("mods/effect.vs", GameShaderStage::Vertex, "main", 0, stale, error));
    assert(!stale && error.find("changed while cached") != std::string::npos);
    assert(created == stages.size() && destroyed == 0);
    shader_changed = false;
    assert(!resources.compute_pipeline("mods/effect", VK_NULL_HANDLE,
        VK_NULL_HANDLE, create_compute, destroy_pipeline, pipeline, error));
    assert(pipeline == VK_NULL_HANDLE && compute_pipelines == 1);
    assert(!resources.compute_pipeline("mods/effect", VK_NULL_HANDLE,
        handle<VkPipelineLayout>(7), fail_compute, destroy_pipeline, pipeline, error));
    assert(pipeline == VK_NULL_HANDLE && failed_pipeline_cleanups == 1);
    GameShaderResource* shader = nullptr;
    assert(!resources.shader("mods/broken.cs", GameShaderStage::Compute, "main", 0, shader, error));
    assert(!shader && error.find("stage/entry mismatch") != std::string::npos);
    assert(!resources.shader("mods/effect.cs", GameShaderStage::Pixel, "main", 0, shader, error));
    assert(!resources.shader("mods/effect", GameShaderStage::Compute, "other", 0, shader, error));
    assert(!resources.shader("mods/effect", GameShaderStage::Compute, "main", 1, shader, error));
    assert(!GameShaderResources::stage_from_target("", parsed));
    resources.clear();
    assert(destroyed == stages.size());
    shader_changed = true;
    assert(resources.shader("mods/effect.vs", GameShaderStage::Vertex, "main", 0, shader, error));
    assert(shader && created == stages.size() + 1);
    resources.clear();
    assert(destroyed == stages.size() + 1);

    // A valid mod metadata file selects a source/include pair and compile
    // defines. When its precompiled binary is stale, the same defines must
    // reach the runtime compiler before the resulting module is accepted.
    const std::string mod_source = "float4 main() : SV_Position { return 1; }\n";
    const std::string mod_include = "#define MOD_HELPER 1\n";
    const uint32_t stale_spirv_words[]{0x07230203, 0x00010000, 1, 2, 0,
        0x0005000f, 0, 1, 0x6e69616d, 0};
    const uint32_t compiled_spirv_words[]{0x07230203, 0x00010000, 2, 2, 0,
        0x0005000f, 0, 1, 0x6e69616d, 0};
    std::vector<uint8_t> stale_spirv(sizeof(stale_spirv_words));
    std::vector<uint8_t> compiled_spirv(sizeof(compiled_spirv_words));
    std::memcpy(stale_spirv.data(), stale_spirv_words, sizeof(stale_spirv_words));
    std::memcpy(compiled_spirv.data(), compiled_spirv_words, sizeof(compiled_spirv_words));
    const auto fingerprint_text = [](uint64_t value) {
        std::ostringstream output;
        output << std::hex << std::setw(16) << std::setfill('0') << value;
        return output.str();
    };
    const std::string mod_metadata =
        "XRAY_VULKAN_SHADER_METADATA_V1\n"
        "source=vk/mods/custom.vs\n"
        "stage=vs\n"
        "entry=main\n"
        "defines=XRAY_TEST_VARIANT=1|XRAY_MATERIAL_MODE=2\n";
    const std::string complete_mod_metadata = mod_metadata +
        "source-fingerprint=" + fingerprint_text(shader_bytes_fingerprint(
            reinterpret_cast<const uint8_t*>(mod_source.data()), mod_source.size())) + "\n" +
        "spirv-fingerprint=" + fingerprint_text(shader_bytes_fingerprint(
            compiled_spirv.data(), compiled_spirv.size())) + "\n" +
        "dependency=vk/mods/common.h=" + fingerprint_text(shader_bytes_fingerprint(
            reinterpret_cast<const uint8_t*>(mod_include.data()), mod_include.size())) + "\n";
    const std::vector<uint8_t> metadata_bytes(complete_mod_metadata.begin(), complete_mod_metadata.end());
    std::vector<std::string> received_defines;
    std::string received_compile_path;
    unsigned runtime_compile_calls = 0;
    GameShaderResources mod_resources;
    mod_resources.configure(handle<VkDevice>(2), {create_module, destroy_module},
        [&](const std::string& path, std::vector<uint8_t>& bytes, std::string& source_error) {
            if (path == "mods\\custom.vs.spv.meta") bytes = metadata_bytes;
            else if (path == "mods\\custom.vs.spv") bytes = stale_spirv;
            else if (path == "vk/mods/custom.vs") bytes.assign(mod_source.begin(), mod_source.end());
            else if (path == "vk/mods/common.h") bytes.assign(mod_include.begin(), mod_include.end());
            else { source_error = "unexpected test shader path"; return false; }
            return true;
        }, false, false, {},
        [&](const std::string& path, GameShaderStage stage, const char* entry, uint32_t flags,
            const GameShaderResources::Source& source, const uint8_t* override_bytes,
            size_t override_size, const std::vector<std::string>& defines,
            std::vector<uint8_t>& output, std::string&) {
            ++runtime_compile_calls;
            received_compile_path = path;
            received_defines = defines;
            assert(stage == GameShaderStage::Vertex && std::strcmp(entry, "main") == 0 && flags == 0);
            assert(override_bytes && override_size == mod_source.size() &&
                std::memcmp(override_bytes, mod_source.data(), mod_source.size()) == 0);
            std::vector<uint8_t> include_bytes;
            std::string include_error;
            assert(source("vk/mods/common.h", include_bytes, include_error));
            assert(std::string(include_bytes.begin(), include_bytes.end()) == mod_include);
            output = compiled_spirv;
            return true;
        });
    GameShaderResource* mod_shader = nullptr;
    assert(mod_resources.shader("mods/custom", GameShaderStage::Vertex, "main", 0,
        mod_shader, error, reinterpret_cast<const uint8_t*>(mod_source.data()), mod_source.size()));
    assert(mod_shader && runtime_compile_calls == 1);
    assert(received_compile_path == "vk/mods/custom.vs");
    assert((received_defines == std::vector<std::string>{
        "XRAY_TEST_VARIANT=1", "XRAY_MATERIAL_MODE=2"}));
    assert(mod_shader->precompiled_rejection.find("fingerprint") != std::string::npos);
    GameShaderResource* mod_shader_option = nullptr;
    const std::vector<std::string> option_defines{"XRAY_RENDER_PATH=1"};
    assert(mod_resources.shader("mods/custom", GameShaderStage::Vertex, "main", 0,
        mod_shader_option, error, reinterpret_cast<const uint8_t*>(mod_source.data()),
        mod_source.size(), false, option_defines));
    assert(mod_shader_option && mod_shader_option != mod_shader && mod_resources.size() == 2);
    assert(runtime_compile_calls == 2);
    assert((received_defines == std::vector<std::string>{
        "XRAY_TEST_VARIANT=1", "XRAY_MATERIAL_MODE=2", "XRAY_RENDER_PATH=1"}));
    assert(mod_shader_option->precompiled_rejection.find("requested shader defines") != std::string::npos);
    GameShaderResource* mod_shader_option_reused = nullptr;
    assert(mod_resources.shader("mods/custom", GameShaderStage::Vertex, "main", 0,
        mod_shader_option_reused, error, reinterpret_cast<const uint8_t*>(mod_source.data()),
        mod_source.size(), false, option_defines));
    assert(mod_shader_option_reused == mod_shader_option && runtime_compile_calls == 2);
    GameShaderResource* mod_shader_override = nullptr;
    const std::vector<std::string> override_defines{"XRAY_TEST_VARIANT=9"};
    assert(mod_resources.shader("mods/custom", GameShaderStage::Vertex, "main", 0,
        mod_shader_override, error, reinterpret_cast<const uint8_t*>(mod_source.data()),
        mod_source.size(), false, override_defines));
    assert(mod_shader_override && mod_shader_override != mod_shader_option && runtime_compile_calls == 3);
    assert((received_defines == std::vector<std::string>{
        "XRAY_TEST_VARIANT=9", "XRAY_MATERIAL_MODE=2"}));
    GameShaderResource* invalid_variant = nullptr;
    const std::vector<std::string> invalid_defines{"9XRAY_INVALID=1"};
    assert(!mod_resources.shader("mods/custom", GameShaderStage::Vertex, "main", 0,
        invalid_variant, error, reinterpret_cast<const uint8_t*>(mod_source.data()),
        mod_source.size(), false, invalid_defines));
    assert(!invalid_variant && error.find("invalid preprocessor define") != std::string::npos);
    assert(runtime_compile_calls == 3);
    mod_resources.clear();
    assert(created == stages.size() + 4 && destroyed == stages.size() + 4);

    std::vector<std::string> compute_defines;
    GameShaderResources compute_variant_resources;
    compute_variant_resources.configure(handle<VkDevice>(1), {create_module, destroy_module},
        [](const std::string& name, std::vector<uint8_t>&, std::string& source_error) {
            source_error = name.find(".spv.meta") != std::string::npos
                ? "no compute metadata fixture" : "no precompiled compute fixture";
            return false;
        }, false, false, {},
        [&compute_defines](const std::string&, GameShaderStage stage, const char* entry, uint32_t,
            const GameShaderResources::Source&, const uint8_t*, size_t,
            const std::vector<std::string>& defines, std::vector<uint8_t>& output,
            std::string& compile_error) {
            assert(stage == GameShaderStage::Compute && std::strcmp(entry, "main") == 0);
            compute_defines = defines;
            const uint32_t code[]{0x07230203, 0x00010000, 1, 2, 0,
                0x0005000f, 5, 1, 0x6e69616d, 0};
            output.resize(sizeof(code));
            std::memcpy(output.data(), code, sizeof(code));
            compile_error.clear();
            return true;
        });
    const std::vector<std::string> compute_engine_defines{
        "XRAY_VULKAN", "XRAY_VK_DEVICE_ID=42", "SKIN_NONE"};
    compute_variant_resources.set_engine_defines(compute_engine_defines);
    const unsigned modules_before_compute_variant = created;
    const unsigned pipelines_before_compute_variant = compute_pipelines;
    VkPipeline compute_variant_pipeline = VK_NULL_HANDLE;
    assert(compute_variant_resources.compute_pipeline("mods/runtime_compute", VK_NULL_HANDLE,
        handle<VkPipelineLayout>(7), create_compute, destroy_pipeline,
        compute_variant_pipeline, error));
    assert(compute_variant_pipeline && compute_pipelines == pipelines_before_compute_variant + 1);
    assert(created == modules_before_compute_variant + 1);
    assert(compute_defines == compute_engine_defines);
    compute_variant_resources.destroy();
}
