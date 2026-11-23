#include "GameShaderResources.h"
#include "RuntimeSlangCompiler.h"
#include "RuntimeGameShaderVariants.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <unordered_set>
#include <utility>

namespace xray::render::vulkan
{
namespace
{
const char* shader_stage_name(GameShaderStage stage)
{
    switch (stage)
    {
    case GameShaderStage::Vertex: return "vs";
    case GameShaderStage::Hull: return "hs";
    case GameShaderStage::Domain: return "ds";
    case GameShaderStage::Geometry: return "gs";
    case GameShaderStage::Pixel: return "ps";
    case GameShaderStage::Compute: return "cs";
    }
    return "";
}

const generated::RuntimeGameShaderVariant* runtime_shader_variant(
    const std::string& name, GameShaderStage stage, const char* entry)
{
    std::string portable_name = name;
    std::replace(portable_name.begin(), portable_name.end(), '\\', '/');
    for (const auto& variant : generated::RuntimeGameShaderVariants)
        if (portable_name == variant.name && std::strcmp(shader_stage_name(stage), variant.stage) == 0 &&
            entry && std::strcmp(entry, variant.entry) == 0)
            return &variant;
    return nullptr;
}

std::vector<std::string> runtime_shader_defines(const generated::RuntimeGameShaderVariant* variant)
{
    std::vector<std::string> defines;
    if (!variant || !variant->defines || !variant->defines[0])
        return defines;

    const std::string packed(variant->defines);
    size_t begin = 0;
    while (begin < packed.size())
    {
        const size_t end = packed.find('|', begin);
        defines.emplace_back(packed.substr(begin, end == std::string::npos ? end : end - begin));
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return defines;
}

bool shader_define_name(const std::string& define, std::string& name)
{
    const size_t separator = define.find('=');
    name = define.substr(0, separator);
    if (name.empty() || !((name[0] >= 'a' && name[0] <= 'z') ||
            (name[0] >= 'A' && name[0] <= 'Z') || name[0] == '_'))
        return false;
    return std::none_of(name.begin() + 1, name.end(), [](unsigned char ch) {
        return !((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
            (ch >= '0' && ch <= '9') || ch == '_');
    });
}

bool valid_shader_define(const std::string& define, std::string& name)
{
    if (define.find_first_of("\r\n") != std::string::npos || define.find('\0') != std::string::npos)
        return false;
    return shader_define_name(define, name);
}

bool merge_shader_defines(std::vector<std::string>& defines,
    const std::vector<std::string>& additional, std::string& error)
{
    std::unordered_set<std::string> base_names;
    for (const std::string& define : defines)
    {
        std::string name;
        if (!valid_shader_define(define, name))
        {
            error = "runtime shader variant contains an invalid base preprocessor define";
            return false;
        }
        if (!base_names.insert(name).second)
        {
            error = "runtime shader variant contains a duplicate base preprocessor define: " + name;
            return false;
        }
    }

    for (const std::string& define : additional)
    {
        std::string name;
        if (!valid_shader_define(define, name))
        {
            error = "runtime shader variant contains an invalid preprocessor define";
            return false;
        }
        const auto existing = std::find_if(defines.begin(), defines.end(), [&name](const std::string& item) {
            return item.substr(0, item.find('=')) == name;
        });
        if (existing == defines.end())
            defines.push_back(define);
        else
            *existing = define;
    }
    error.clear();
    return true;
}

struct RuntimeShaderMetadata
{
    std::string source;
    std::string stage;
    std::string entry;
    std::vector<std::string> defines;
    uint64_t source_fingerprint{};
    uint64_t spirv_fingerprint{};
    std::vector<std::pair<std::string, uint64_t>> dependencies;
};

bool parse_shader_fingerprint(const std::string& text, uint64_t& fingerprint)
{
    if (text.size() != 16)
        return false;
    fingerprint = 0;
    for (const unsigned char ch : text)
    {
        uint8_t digit = 0;
        if (ch >= '0' && ch <= '9') digit = static_cast<uint8_t>(ch - '0');
        else if (ch >= 'a' && ch <= 'f') digit = static_cast<uint8_t>(ch - 'a' + 10);
        else if (ch >= 'A' && ch <= 'F') digit = static_cast<uint8_t>(ch - 'A' + 10);
        else return false;
        fingerprint = (fingerprint << 4) | digit;
    }
    return true;
}

bool safe_shader_metadata_path(const std::string& path)
{
    if (path.compare(0, 3, "vk/") != 0 || path.size() <= 3 ||
        path.find('\\') != std::string::npos || path.find(':') != std::string::npos)
        return false;
    size_t begin = 0;
    while (begin <= path.size())
    {
        const size_t end = path.find('/', begin);
        const std::string component = path.substr(begin,
            end == std::string::npos ? std::string::npos : end - begin);
        if (component.empty() || component == "." || component == "..")
            return false;
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return true;
}

bool parse_runtime_shader_metadata(const std::vector<uint8_t>& bytes,
    RuntimeShaderMetadata& metadata, std::string& error)
{
    constexpr char magic[] = "XRAY_VULKAN_SHADER_METADATA_V1";
    if (bytes.empty() || bytes.size() > 1024 * 1024 ||
        std::find(bytes.begin(), bytes.end(), uint8_t{0}) != bytes.end())
    {
        error = "shader metadata is empty, oversized or contains NUL bytes";
        return false;
    }

    std::istringstream stream{std::string(bytes.begin(), bytes.end())};
    std::string line;
    if (!std::getline(stream, line))
    {
        error = "shader metadata has no version header";
        return false;
    }
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
    if (line != magic)
    {
        error = "shader metadata version is unsupported";
        return false;
    }

    bool have_source = false, have_stage = false, have_entry = false;
    bool have_defines = false, have_source_fingerprint = false, have_spirv_fingerprint = false;
    std::unordered_set<std::string> dependency_paths;
    while (std::getline(stream, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        const size_t separator = line.find('=');
        if (separator == std::string::npos)
        {
            error = "shader metadata contains a malformed field";
            return false;
        }
        const std::string key = line.substr(0, separator);
        const std::string value = line.substr(separator + 1);
        if (key == "source" && !have_source && safe_shader_metadata_path(value))
        {
            metadata.source = value;
            have_source = true;
        }
        else if (key == "stage" && !have_stage && !value.empty())
        {
            metadata.stage = value;
            have_stage = true;
        }
        else if (key == "entry" && !have_entry && !value.empty())
        {
            metadata.entry = value;
            have_entry = true;
        }
        else if (key == "defines" && !have_defines)
        {
            have_defines = true;
            if (!value.empty())
            {
                size_t begin = 0;
                while (begin <= value.size())
                {
                    const size_t end = value.find('|', begin);
                    const std::string define = value.substr(begin,
                        end == std::string::npos ? std::string::npos : end - begin);
                    if (define.empty())
                    {
                        error = "shader metadata contains an empty define";
                        return false;
                    }
                    metadata.defines.push_back(define);
                    if (end == std::string::npos)
                        break;
                    begin = end + 1;
                }
            }
        }
        else if (key == "source-fingerprint" && !have_source_fingerprint &&
            parse_shader_fingerprint(value, metadata.source_fingerprint))
            have_source_fingerprint = true;
        else if (key == "spirv-fingerprint" && !have_spirv_fingerprint &&
            parse_shader_fingerprint(value, metadata.spirv_fingerprint))
            have_spirv_fingerprint = true;
        else if (key == "dependency")
        {
            const size_t hash_separator = line.rfind('=');
            const std::string path = hash_separator == std::string::npos || hash_separator <= separator
                ? std::string() : line.substr(separator + 1, hash_separator - separator - 1);
            uint64_t fingerprint = 0;
            if (!safe_shader_metadata_path(path) || path.find('=') != std::string::npos ||
                !parse_shader_fingerprint(hash_separator == std::string::npos ? std::string() :
                    line.substr(hash_separator + 1), fingerprint) || !dependency_paths.insert(path).second)
            {
                error = "shader metadata contains an invalid or duplicate dependency";
                return false;
            }
            metadata.dependencies.emplace_back(path, fingerprint);
        }
        else
        {
            error = "shader metadata contains an unknown, invalid or duplicate field: " + key;
            return false;
        }
    }

    if (!have_source || !have_stage || !have_entry || !have_defines ||
        !have_source_fingerprint || !have_spirv_fingerprint)
    {
        error = "shader metadata is missing a required field";
        return false;
    }
    error.clear();
    return true;
}

bool runtime_shader_metadata_source_matches(const RuntimeShaderMetadata& metadata,
    const uint8_t* source_override, size_t source_override_size,
    const GameShaderResources::Source& source, std::string& error)
{
    std::vector<uint8_t> source_bytes;
    if (source_override && source_override_size)
        source_bytes.assign(source_override, source_override + source_override_size);
    else if (!source(metadata.source, source_bytes, error))
    {
        error = "active HLSL source is unavailable for metadata validation: " + error;
        return false;
    }
    if (shader_bytes_fingerprint(source_bytes.data(), source_bytes.size()) != metadata.source_fingerprint)
    {
        error = "active HLSL source content differs from the precompiled SPIR-V metadata";
        return false;
    }

    for (const auto& dependency : metadata.dependencies)
    {
        std::vector<uint8_t> dependency_bytes;
        if (!source(dependency.first, dependency_bytes, error))
        {
            error = "shader include is unavailable for metadata validation: " + dependency.first;
            return false;
        }
        if (shader_bytes_fingerprint(dependency_bytes.data(), dependency_bytes.size()) != dependency.second)
        {
            error = "shader include content differs from precompiled metadata: " + dependency.first;
            return false;
        }
    }
    error.clear();
    return true;
}

bool runtime_shader_metadata_matches(const RuntimeShaderMetadata& metadata,
    const uint8_t* source_override, size_t source_override_size,
    const std::vector<uint8_t>& spirv, const GameShaderResources::Source& source,
    std::string& error)
{
    if (!runtime_shader_metadata_source_matches(metadata, source_override,
            source_override_size, source, error))
        return false;
    if (shader_bytes_fingerprint(spirv.data(), spirv.size()) != metadata.spirv_fingerprint)
    {
        error = "SPIR-V content differs from its metadata fingerprint";
        return false;
    }
    error.clear();
    return true;
}

bool load_runtime_shader_variant_metadata(const generated::RuntimeGameShaderVariant& variant,
    const std::vector<std::string>& expected_defines, const char* entry,
    const uint8_t* source_override, size_t source_override_size,
    const GameShaderResources::Source& source, const std::string& spirv_name,
    RuntimeShaderMetadata& metadata, std::string& error)
{
    std::vector<uint8_t> metadata_bytes;
    // The compiler and APK asset checker store metadata beside the SPIR-V as
    // `<shader>.spv.meta`; `spirv_name` is the canonical shader stem.
    if (!source(spirv_name + ".spv.meta", metadata_bytes, error))
    {
        error = "precompiled SPIR-V metadata is unavailable: " + error;
        return false;
    }

    if (!parse_runtime_shader_metadata(metadata_bytes, metadata, error))
    {
        error = "precompiled SPIR-V metadata is invalid: " + error;
        return false;
    }
    if (metadata.source != variant.source || metadata.stage != variant.stage ||
        metadata.entry != (entry ? entry : "") || metadata.defines != expected_defines)
    {
        error = "precompiled SPIR-V metadata does not describe the requested shader variant";
        return false;
    }
    return runtime_shader_metadata_source_matches(metadata, source_override,
        source_override_size, source, error);
}

bool runtime_shader_metadata_matches(const generated::RuntimeGameShaderVariant& variant,
    const std::vector<std::string>& expected_defines, const char* entry,
    const uint8_t* source_override, size_t source_override_size,
    const std::vector<uint8_t>& spirv, const GameShaderResources::Source& source,
    const std::string& spirv_name, std::string& error)
{
    RuntimeShaderMetadata metadata;
    if (!load_runtime_shader_variant_metadata(variant, expected_defines, entry,
            source_override, source_override_size, source, spirv_name, metadata, error))
        return false;
    if (shader_bytes_fingerprint(spirv.data(), spirv.size()) != metadata.spirv_fingerprint)
    {
        error = "SPIR-V content differs from its metadata fingerprint";
        return false;
    }
    error.clear();
    return true;
}

std::string game_shader_cache_key(const std::string& key, const char* entry, uint32_t flags,
    bool force_runtime_compile, const std::vector<std::string>& defines,
    const uint8_t* source_override, size_t source_override_size)
{
    std::string result = key + "#" + entry + "#" + std::to_string(flags) +
        (force_runtime_compile ? "#runtime" : "");
    for (const std::string& define : defines)
        result += "#d" + std::to_string(define.size()) + ":" + define;
    if (source_override && source_override_size)
        result += "#s" + std::to_string(source_override_size) + ":" +
            std::to_string(shader_bytes_fingerprint(source_override, source_override_size));
    return result;
}
} // namespace

bool GameShaderResources::merge_defines(const std::vector<std::string>& base,
    const std::vector<std::string>& overrides, std::vector<std::string>& result,
    std::string& error)
{
    result = base;
    return merge_shader_defines(result, overrides, error);
}

bool GameShaderResources::pipeline_defines(const std::string& name, GameShaderStage stage,
    const char* entry, const std::vector<std::string>& overrides,
    std::vector<std::string>& result, std::string& error) const
{
    if (has_manifest_variant(name, stage, entry))
    {
        result = overrides;
        error.clear();
        return merge_shader_defines(result, {}, error);
    }
    return merge_defines(engine_defines_, overrides, result, error);
}

bool GameShaderResources::stage_from_target(const char *target, GameShaderStage &stage)
{
    if (!target || !target[0] || !target[1])
        return false;
    switch (target[0])
    {
    case 'v': if (target[1] == 's') { stage = GameShaderStage::Vertex; return true; } break;
    case 'h': if (target[1] == 's') { stage = GameShaderStage::Hull; return true; } break;
    case 'd': if (target[1] == 's') { stage = GameShaderStage::Domain; return true; } break;
    case 'g': if (target[1] == 's') { stage = GameShaderStage::Geometry; return true; } break;
    case 'p': if (target[1] == 's') { stage = GameShaderStage::Pixel; return true; } break;
    case 'c': if (target[1] == 's') { stage = GameShaderStage::Compute; return true; } break;
    }
    return false;
}

GameShaderResources::GameShaderResources() = default;
GameShaderResources::~GameShaderResources() = default;

bool GameShaderResources::canonical_name(const std::string &name, GameShaderStage stage, std::string &result, std::string &error)
{
    result.clear();
    const char *extension = nullptr;
    switch (stage)
    {
    case GameShaderStage::Vertex: extension = ".vs"; break;
    case GameShaderStage::Hull: extension = ".hs"; break;
    case GameShaderStage::Domain: extension = ".ds"; break;
    case GameShaderStage::Geometry: extension = ".gs"; break;
    case GameShaderStage::Pixel: extension = ".ps"; break;
    case GameShaderStage::Compute: extension = ".cs"; break;
    }
    if (!extension)
    {
        error = "unsupported legacy shader stage: " + name;
        return false;
    }
    std::string normalized = name;
    std::replace(normalized.begin(), normalized.end(), '/', '\\');
    if (normalized.empty() || normalized.front() == '\\' || normalized.find(':') != std::string::npos || normalized.find('\0') != std::string::npos ||
        normalized.find("..") != std::string::npos || normalized.find("\\\\") != std::string::npos || normalized.back() == '\\')
    {
        error = "invalid legacy shader name: " + name;
        return false;
    }
    const auto ends = [&](const char *suffix) {
        const std::string s(suffix);
        return normalized.size() >= s.size() && normalized.compare(normalized.size() - s.size(), s.size(), s) == 0;
    };
    if (ends(".spv"))
        normalized.resize(normalized.size() - 4);
    const bool explicit_stage = ends(".vs") || ends(".hs") || ends(".ds") ||
        ends(".gs") || ends(".ps") || ends(".cs");
    if (explicit_stage && !ends(extension))
    {
        error = "legacy shader stage mismatch: " + name;
        return false;
    }
    if (!explicit_stage)
        normalized += extension;
    result = std::move(normalized);
    error.clear();
    return true;
}

bool GameShaderResources::has_manifest_variant(const std::string &name, GameShaderStage stage, const char *entry)
{
    std::string canonical;
    std::string error;
    return entry && *entry && canonical_name(name, stage, canonical, error) &&
        runtime_shader_variant(canonical, stage, entry) != nullptr;
}

void GameShaderResources::configure(VkDevice device, const ShaderModuleDispatch &dispatch, Source source,
    bool geometry_supported, bool tessellation_supported, Source bundled_source,
    RuntimeCompile runtime_compile)
{
    destroy();
    device_ = device;
    dispatch_ = dispatch;
    source_ = std::move(source);
    bundled_source_ = std::move(bundled_source);
    runtime_compile_ = std::move(runtime_compile);
    runtime_compiler_ = std::make_unique<RuntimeSlangCompiler>();
    geometry_supported_ = geometry_supported;
    tessellation_supported_ = tessellation_supported;
}

bool GameShaderResources::shader(const std::string &name, GameShaderStage stage, const char *entry, uint32_t flags,
                                 GameShaderResource *&result, std::string &error, const uint8_t *source_override,
                                 size_t source_override_size, bool force_runtime_compile,
                                 const std::vector<std::string>& defines)
{
    result = nullptr;
    std::string key;
    if (!canonical_name(name, stage, key, error))
        return false;
    if (!device_ || !source_ || !entry || !*entry)
    {
        error = "invalid legacy shader parameters (device/entry): " + key;
        return false;
    }

    const generated::RuntimeGameShaderVariant* variant = runtime_shader_variant(key, stage, entry);
    RuntimeShaderMetadata mod_metadata;
    bool mod_metadata_present = false;
    bool mod_metadata_valid = false;
    std::string mod_metadata_error;
    if (!variant)
    {
        std::vector<uint8_t> metadata_bytes;
        if (source_(key + ".spv.meta", metadata_bytes, mod_metadata_error) && !metadata_bytes.empty())
        {
            mod_metadata_present = true;
            if (!parse_runtime_shader_metadata(metadata_bytes, mod_metadata, mod_metadata_error))
                mod_metadata_error = "mod shader metadata is invalid: " + mod_metadata_error;
            else if (mod_metadata.stage != shader_stage_name(stage) ||
                mod_metadata.entry != entry)
                mod_metadata_error = "mod shader metadata has a different stage or entry point";
            else
                mod_metadata_valid = runtime_shader_metadata_source_matches(mod_metadata,
                    source_override, source_override_size, source_, mod_metadata_error);
        }
    }
    const std::string compile_path = variant ? variant->source :
        mod_metadata_valid ? mod_metadata.source : key;
    std::vector<std::string> compile_defines = variant ? runtime_shader_defines(variant) :
        mod_metadata_valid ? mod_metadata.defines : std::vector<std::string>{};
    if (!merge_shader_defines(compile_defines, defines, error))
        return false;

    std::string precompiled_error;
    bool variant_source_valid = false;
    if (variant)
    {
        RuntimeShaderMetadata variant_metadata;
        std::string variant_metadata_error;
        variant_source_valid = load_runtime_shader_variant_metadata(*variant, compile_defines, entry,
            source_override, source_override_size, source_, key, variant_metadata,
            variant_metadata_error);
        if (!variant_source_valid)
            precompiled_error = variant_metadata_error;
    }
    const std::string cache_key = game_shader_cache_key(key, entry, flags,
        force_runtime_compile, compile_defines, source_override, source_override_size);
    if (mod_metadata_valid || variant_source_valid)
    {
        const auto cached = resources_.find(cache_key);
        if (cached != resources_.end())
        {
            result = cached->second.get();
            error.clear();
            return true;
        }
    }

    std::vector<uint8_t> bytes;
    bool use_precompiled = !force_runtime_compile && flags == 0 && std::string(entry) == "main";
    if (force_runtime_compile)
        precompiled_error = "runtime HLSL recompilation requested after interface validation";
    if (mod_metadata_present && !mod_metadata_valid)
    {
        use_precompiled = false;
        precompiled_error = mod_metadata_error.empty() ? "mod shader metadata failed validation" : mod_metadata_error;
    }
    if (use_precompiled && !variant &&
        ((mod_metadata_valid && mod_metadata.defines != compile_defines) ||
            (!mod_metadata_present && !defines.empty())))
    {
        use_precompiled = false;
        precompiled_error = "precompiled SPIR-V metadata does not describe the requested shader defines";
    }
    // For non-manifest variants, compare the actual caller source bytes. VFS
    // timestamps are not reliable when Android unpacks engine assets into a
    // separate overlay, and a caller-provided buffer may not come from a VFS
    // file at all.
    if (use_precompiled && !variant && !mod_metadata_valid && source_override && source_override_size)
    {
        std::vector<uint8_t> active_source;
        std::string source_error;
        const bool source_loaded = source_(key, active_source, source_error);
        const bool source_matches = source_loaded && active_source.size() == source_override_size &&
            std::equal(active_source.begin(), active_source.end(), source_override);
        if (!source_matches)
        {
            use_precompiled = false;
            precompiled_error = "precompiled SPIR-V does not match the caller-provided HLSL source bytes";
            if (!source_error.empty())
                precompiled_error += "; VFS source: " + source_error;
        }
    }
    if (use_precompiled)
    {
        if (!source_(key + ".spv", bytes, precompiled_error) || bytes.empty())
            bytes.clear();
        else if (mod_metadata_valid && !runtime_shader_metadata_matches(mod_metadata,
                source_override, source_override_size, bytes, source_, precompiled_error))
            bytes.clear();
        else if (variant && !runtime_shader_metadata_matches(*variant, compile_defines, entry,
                source_override, source_override_size, bytes, source_, key, precompiled_error))
        {
            bytes.clear();
        }
        if (!bytes.empty() && !has_spirv_entry(bytes.data(), bytes.size(), static_cast<uint32_t>(stage), entry))
        {
            precompiled_error = "precompiled SPIR-V stage/entry mismatch: " + key;
            bytes.clear();
        }

        // User-installed game data has higher VFS priority than the APK's
        // engine overlay. It can therefore provide a stale or missing .spv
        // while the active HLSL is still byte-for-byte the source used by the
        // APK. Retry that exact bundled binary, but validate its metadata
        // against the active HLSL/includes so modified shaders still take the
        // runtime compiler path.
        if (bytes.empty() && variant && bundled_source_)
        {
            std::vector<uint8_t> bundled_bytes;
            std::string bundled_error;
            auto bundled_variant_source = [this, &key](const std::string& name,
                std::vector<uint8_t>& source_bytes, std::string& source_error) {
                if (name == key + ".spv.meta")
                    return bundled_source_(name, source_bytes, source_error);
                return source_(name, source_bytes, source_error);
            };
            if (bundled_source_(key + ".spv", bundled_bytes, bundled_error) && !bundled_bytes.empty() &&
                runtime_shader_metadata_matches(*variant, compile_defines, entry,
                    source_override, source_override_size, bundled_bytes, bundled_variant_source, key,
                    bundled_error) &&
                has_spirv_entry(bundled_bytes.data(), bundled_bytes.size(), static_cast<uint32_t>(stage), entry))
            {
                bytes = std::move(bundled_bytes);
                precompiled_error.clear();
            }
            else
            {
                precompiled_error += "; bundled engine SPIR-V is unavailable or does not match the active source";
                if (!bundled_error.empty())
                    precompiled_error += " (" + bundled_error + ")";
            }
        }
    }

    if (bytes.empty())
    {
        std::string compile_error;
        const bool compiled = runtime_compile_ ? runtime_compile_(compile_path, stage, entry, flags,
            source_, source_override, source_override_size, compile_defines, bytes, compile_error) :
            runtime_compiler_ && runtime_compiler_->compile(compile_path, stage, entry, flags, source_,
                source_override, source_override_size, compile_defines, bytes, compile_error);
        if (!compiled)
        {
            error = "legacy shader '" + key + "' has no usable matching precompiled SPIR-V";
            if (!precompiled_error.empty())
                error += " (" + precompiled_error + ")";
            if (!compile_error.empty())
                error += "; runtime HLSL compile failed: " + compile_error;
            return false;
        }
    }

    if (!has_spirv_entry(bytes.data(), bytes.size(), static_cast<uint32_t>(stage), entry))
    {
        error = "compiled SPIR-V stage/entry mismatch: " + key;
        return false;
    }

    const uint64_t content_fingerprint = shader_bytes_fingerprint(bytes.data(), bytes.size());
    const auto found = resources_.find(cache_key);
    if (found != resources_.end())
    {
        if (found->second->content_fingerprint != content_fingerprint)
        {
            error = "legacy shader changed while cached: " + key + "; rebuild after GPU idle";
            return false;
        }
        result = found->second.get();
        error.clear();
        return true;
    }
    auto resource = std::make_unique<GameShaderResource>();
    resource->name = key;
    resource->entry_point = entry;
    resource->stage = stage;
    resource->compile_flags = flags;
    if (source_override && source_override_size)
        resource->runtime_source.assign(source_override, source_override + source_override_size);
    resource->runtime_defines = compile_defines;
    if (!precompiled_error.empty())
        resource->precompiled_rejection = precompiled_error;
    resource->content_fingerprint = content_fingerprint;
    if (!reflect_spirv_resources(bytes.data(), bytes.size(), static_cast<uint32_t>(stage),
            entry, resource->reflection, error))
    {
        error = "legacy shader reflection '" + key + "': " + error;
        return false;
    }
    if (!resource->module.initialize_bytes(device_, dispatch_, bytes.data(), bytes.size(), error))
    {
        error = "legacy shader module '" + key + "': " + error;
        return false;
    }
    result = resource.get();
    resources_.emplace(cache_key, std::move(resource));
    error.clear();
    return true;
}

void GameShaderResources::destroy()
{
    clear();
    engine_defines_.clear();
    device_ = VK_NULL_HANDLE;
    dispatch_ = {};
    source_ = {};
    bundled_source_ = {};
    runtime_compile_ = {};
    runtime_compiler_.reset();
    geometry_supported_ = tessellation_supported_ = false;
}
} // namespace xray::render::vulkan
