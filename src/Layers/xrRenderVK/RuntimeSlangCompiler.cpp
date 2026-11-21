#include "RuntimeSlangCompiler.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <mutex>
#include <sstream>

#if defined(XRAY_HAS_SLANG)
#include <slang-com-ptr.h>
#include <slang.h>
#endif

namespace xray::render::vulkan
{
namespace
{
bool normalize_shader_path(const std::string &path, std::string &normalized)
{
    normalized.clear();
    if (path.empty() || path.find('\0') != std::string::npos || path.find(':') != std::string::npos)
        return false;

    std::vector<std::string> components;
    std::string component;
    for (char ch : path)
    {
        if (ch == '/' || ch == '\\')
        {
            if (component.empty() || component == ".")
            {
                component.clear();
                continue;
            }
            if (component == "..")
            {
                if (components.empty())
                    return false;
                components.pop_back();
            }
            else
                components.push_back(std::move(component));
            component.clear();
            continue;
        }
        component.push_back(ch);
    }
    if (!component.empty() && component != ".")
    {
        if (component == "..")
        {
            if (components.empty())
                return false;
            components.pop_back();
        }
        else
            components.push_back(std::move(component));
    }
    for (const auto &part : components)
    {
        if (part.empty())
            continue;
        if (!normalized.empty())
            normalized.push_back('/');
        normalized += part;
    }
    return !normalized.empty();
}

#if defined(XRAY_HAS_SLANG)
class GameShaderFileSystem final : public ISlangFileSystem
{
  public:
    GameShaderFileSystem(GameShaderResources::Source source, std::string parent)
        : source_(std::move(source)), parent_(std::move(parent))
    {
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(SlangUUID const &uuid, void **outObject) override
    {
        if (!outObject)
            return SLANG_E_INVALID_ARG;
        if (uuid == ISlangFileSystem::getTypeGuid() || uuid == ISlangCastable::getTypeGuid() ||
            uuid == ISlangUnknown::getTypeGuid())
        {
            *outObject = static_cast<ISlangFileSystem *>(this);
            addRef();
            return SLANG_OK;
        }
        *outObject = nullptr;
        return SLANG_E_NO_INTERFACE;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override
    {
        return ++references_;
    }

    SLANG_NO_THROW uint32_t SLANG_MCALL release() override
    {
        const uint32_t count = --references_;
        if (!count)
            delete this;
        return count;
    }

    SLANG_NO_THROW void *SLANG_MCALL castAs(SlangUUID const &uuid) override
    {
        return uuid == ISlangFileSystem::getTypeGuid() || uuid == ISlangCastable::getTypeGuid() ||
                uuid == ISlangUnknown::getTypeGuid()
            ? static_cast<ISlangFileSystem *>(this)
            : nullptr;
    }

    SLANG_NO_THROW SlangResult SLANG_MCALL loadFile(const char *path, ISlangBlob **outBlob) override
    {
        if (!path || !outBlob)
            return SLANG_E_INVALID_ARG;
        *outBlob = nullptr;

        std::string requested;
        if (!normalize_shader_path(path, requested))
            return SLANG_E_NOT_FOUND;

        std::vector<uint8_t> bytes;
        std::string reason;
        const bool already_parented = !parent_.empty() &&
            (requested == parent_ || requested.compare(0, parent_.size() + 1, parent_ + "/") == 0);
        bool loaded = false;
        if (!already_parented && !parent_.empty())
        {
            std::string relative_to_shader;
            if (normalize_shader_path(parent_ + "/" + requested, relative_to_shader) &&
                relative_to_shader != requested)
            {
                loaded = source_(relative_to_shader, bytes, reason);
            }
        }
        // Quoted HLSL includes resolve beside the including shader before the
        // game-shader VFS root. Preserve that precedence so a mod-local include
        // cannot be shadowed by an unrelated file with the same root name.
        if (!loaded)
        {
            bytes.clear();
            loaded = source_(requested, bytes, reason);
        }
        if (!loaded)
            return SLANG_E_NOT_FOUND;

        *outBlob = slang_createBlob(bytes.data(), bytes.size());
        return *outBlob ? SLANG_OK : SLANG_E_OUT_OF_MEMORY;
    }

  private:
    ~GameShaderFileSystem() = default;
    std::atomic<uint32_t> references_{1};
    GameShaderResources::Source source_;
    std::string parent_;
};

SlangStage slang_stage(GameShaderStage stage)
{
    switch (stage)
    {
    case GameShaderStage::Vertex: return SLANG_STAGE_VERTEX;
    case GameShaderStage::Hull: return SLANG_STAGE_HULL;
    case GameShaderStage::Domain: return SLANG_STAGE_DOMAIN;
    case GameShaderStage::Geometry: return SLANG_STAGE_GEOMETRY;
    case GameShaderStage::Pixel: return SLANG_STAGE_FRAGMENT;
    case GameShaderStage::Compute: return SLANG_STAGE_COMPUTE;
    }
    return SLANG_STAGE_NONE;
}

slang::CompilerOptionEntry int_option(slang::CompilerOptionName name, int32_t value0, int32_t value1 = 0)
{
    slang::CompilerOptionEntry result{};
    result.name = name;
    result.value.kind = slang::CompilerOptionValueKind::Int;
    result.value.intValue0 = value0;
    result.value.intValue1 = value1;
    return result;
}

void append_diagnostics(slang::IBlob *diagnostics, std::string &error)
{
    if (!diagnostics || !diagnostics->getBufferPointer() || !diagnostics->getBufferSize())
        return;
    const auto *text = static_cast<const char *>(diagnostics->getBufferPointer());
    if (!error.empty())
        error += ": ";
    error.append(text, diagnostics->getBufferSize());
    while (!error.empty() && error.back() == '\0')
        error.pop_back();
}
#endif
} // namespace

struct RuntimeSlangCompiler::Impl
{
#if defined(XRAY_HAS_SLANG)
    Slang::ComPtr<slang::IGlobalSession> global_session;
    std::mutex mutex;
#endif
};

RuntimeSlangCompiler::RuntimeSlangCompiler() : impl_(std::make_unique<Impl>())
{
}

RuntimeSlangCompiler::~RuntimeSlangCompiler() = default;

bool RuntimeSlangCompiler::compile(const std::string &path, GameShaderStage stage, const char *entry, uint32_t flags,
    const GameShaderResources::Source &source, const uint8_t *source_override, size_t source_override_size,
    const std::vector<std::string> &defines, std::vector<uint8_t> &spirv, std::string &error)
{
    spirv.clear();
#if !defined(XRAY_HAS_SLANG)
    (void)path;
    (void)stage;
    (void)entry;
    (void)flags;
    (void)source;
    (void)source_override;
    (void)source_override_size;
    (void)defines;
    error = "runtime HLSL compilation is unavailable in this build";
    return false;
#else
    if (!entry || !*entry || !source)
    {
        error = "runtime HLSL compilation requires an entry point and game shader VFS";
        return false;
    }
    if ((source_override == nullptr) != (source_override_size == 0))
    {
        error = "runtime HLSL compilation received an invalid caller source buffer";
        return false;
    }

    std::string normalized_path;
    if (!normalize_shader_path(path, normalized_path))
    {
        error = "invalid game shader source path: " + path;
        return false;
    }
    std::vector<uint8_t> source_bytes;
    if (source_override && source_override_size)
        source_bytes.assign(source_override, source_override + source_override_size);
    else if (!source(normalized_path, source_bytes, error))
    {
        if (error.empty())
            error = "could not load game shader source: " + normalized_path;
        return false;
    }
    if (source_bytes.empty())
    {
        if (error.empty())
            error = "game shader source is empty: " + normalized_path;
        return false;
    }
    if (std::find(source_bytes.begin(), source_bytes.end(), uint8_t{0}) != source_bytes.end())
    {
        error = "game shader source contains an embedded NUL: " + normalized_path;
        return false;
    }
    std::string source_text(source_bytes.begin(), source_bytes.end());

    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->global_session && SLANG_FAILED(slang::createGlobalSession(impl_->global_session.writeRef())))
    {
        error = "Slang could not create its global session";
        return false;
    }

    const auto slash = normalized_path.find_last_of('/');
    const std::string parent = slash == std::string::npos ? std::string() : normalized_path.substr(0, slash);
    Slang::ComPtr<ISlangFileSystem> file_system(Slang::INIT_ATTACH, new GameShaderFileSystem(source, parent));

    constexpr uint32_t debug_flag = 1u << 0;
    constexpr uint32_t skip_optimization_flag = 1u << 2;
    constexpr uint32_t row_major_flag = 1u << 3;
    constexpr uint32_t column_major_flag = 1u << 4;
    constexpr uint32_t backwards_compatibility_flag = 1u << 12;
    constexpr uint32_t optimization_mask = 0x0000c000u;
    constexpr uint32_t known_d3d_flags = debug_flag | skip_optimization_flag |
        row_major_flag | column_major_flag | backwards_compatibility_flag | optimization_mask;
    if (flags & ~known_d3d_flags)
    {
        std::ostringstream message;
        message << "unsupported legacy shader compile flags 0x" << std::hex << (flags & ~known_d3d_flags);
        error = message.str();
        return false;
    }
    if ((flags & (row_major_flag | column_major_flag)) == (row_major_flag | column_major_flag))
    {
        error = "legacy shader compile flags request conflicting matrix layouts";
        return false;
    }

    std::vector<slang::CompilerOptionEntry> session_options;
    session_options.reserve(3 + defines.size());
    const uint32_t optimization_flag = flags & optimization_mask;
    const int32_t optimization_level = (flags & skip_optimization_flag) || optimization_flag == 0x00004000u
        ? SLANG_OPTIMIZATION_LEVEL_NONE
        : (optimization_flag == 0x00008000u || optimization_flag == 0x0000c000u)
            ? SLANG_OPTIMIZATION_LEVEL_HIGH : SLANG_OPTIMIZATION_LEVEL_DEFAULT;
    session_options.push_back(int_option(slang::CompilerOptionName::Optimization, optimization_level));
    if (flags & debug_flag)
        session_options.push_back(int_option(slang::CompilerOptionName::DebugInformation,
            SLANG_DEBUG_INFO_LEVEL_STANDARD));
    if (flags & backwards_compatibility_flag)
        session_options.push_back(int_option(
            slang::CompilerOptionName::EnableExtendedHLSLBackwardsCompatibility, 1));

    std::vector<std::string> macro_names;
    std::vector<std::string> macro_values;
    macro_names.reserve(defines.size());
    macro_values.reserve(defines.size());
    for (const auto& define : defines)
    {
        const size_t separator = define.find('=');
        const std::string name = define.substr(0, separator);
        if (name.empty())
        {
            error = "runtime HLSL variant contains an empty preprocessor define";
            return false;
        }
        macro_names.push_back(name);
        macro_values.push_back(separator == std::string::npos ? "1" : define.substr(separator + 1));
    }
    for (size_t i = 0; i < macro_names.size(); ++i)
    {
        slang::CompilerOptionEntry option{};
        option.name = slang::CompilerOptionName::MacroDefine;
        option.value.kind = slang::CompilerOptionValueKind::String;
        option.value.stringValue0 = macro_names[i].c_str();
        option.value.stringValue1 = macro_values[i].c_str();
        session_options.push_back(option);
    }

    slang::CompilerOptionEntry target_options[] = {
        int_option(slang::CompilerOptionName::VulkanUseEntryPointName, 1),
        int_option(slang::CompilerOptionName::VulkanBindShift, 0u << 8, 384), // u-registers
        int_option(slang::CompilerOptionName::VulkanBindShift, 1u << 8, 128), // s-registers
        int_option(slang::CompilerOptionName::VulkanBindShift, 2u << 8, 0),   // t-registers
        int_option(slang::CompilerOptionName::VulkanBindShift, 3u << 8, 256), // b-registers
    };
    slang::TargetDesc target{};
    target.format = SLANG_SPIRV;
    target.profile = impl_->global_session->findProfile("spirv_1_3");
    target.compilerOptionEntries = target_options;
    target.compilerOptionEntryCount = static_cast<uint32_t>(sizeof(target_options) / sizeof(target_options[0]));

    slang::SessionDesc description{};
    description.targets = &target;
    description.targetCount = 1;
    description.defaultMatrixLayoutMode = (flags & column_major_flag) ?
        SLANG_MATRIX_LAYOUT_COLUMN_MAJOR : SLANG_MATRIX_LAYOUT_ROW_MAJOR;
    description.fileSystem = file_system;
    description.compilerOptionEntries = session_options.data();
    description.compilerOptionEntryCount = static_cast<uint32_t>(session_options.size());
    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(impl_->global_session->createSession(description, session.writeRef())))
    {
        error = "Slang could not create a SPIR-V compilation session";
        return false;
    }

    uint64_t module_hash = shader_bytes_fingerprint(source_bytes.data(), source_bytes.size());
    module_hash ^= shader_bytes_fingerprint(reinterpret_cast<const uint8_t *>(normalized_path.data()), normalized_path.size());
    std::string variant_identity;
    for (const auto& define : defines)
    {
        variant_identity.push_back('\0');
        variant_identity += define;
    }
    module_hash ^= shader_bytes_fingerprint(reinterpret_cast<const uint8_t *>(variant_identity.data()), variant_identity.size());
    std::ostringstream module_name;
    module_name << "xray_shader_" << std::hex << module_hash;
    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> module(Slang::INIT_ATTACH,
        session->loadModuleFromSourceString(module_name.str().c_str(), normalized_path.c_str(), source_text.c_str(), diagnostics.writeRef()));
    if (!module)
    {
        error = "Slang failed to load HLSL source '" + normalized_path + "'";
        append_diagnostics(diagnostics, error);
        return false;
    }

    Slang::ComPtr<slang::IEntryPoint> entry_point;
    diagnostics.setNull();
    const SlangResult entry_result = module->findAndCheckEntryPoint(entry, slang_stage(stage), entry_point.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(entry_result) || !entry_point)
    {
        error = "Slang could not find entry point '" + std::string(entry) + "' in '" + normalized_path + "'";
        append_diagnostics(diagnostics, error);
        return false;
    }

    slang::IComponentType *components[] = {module, entry_point};
    Slang::ComPtr<slang::IComponentType> program;
    diagnostics.setNull();
    const SlangResult link_result = session->createCompositeComponentType(components, 2, program.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(link_result) || !program)
    {
        error = "Slang failed to link entry point '" + std::string(entry) + "' in '" + normalized_path + "'";
        append_diagnostics(diagnostics, error);
        return false;
    }

    Slang::ComPtr<slang::IBlob> output;
    diagnostics.setNull();
    const SlangResult compile_result = program->getEntryPointCode(0, 0, output.writeRef(), diagnostics.writeRef());
    if (SLANG_FAILED(compile_result) || !output || !output->getBufferPointer() || !output->getBufferSize())
    {
        error = "Slang failed to compile '" + normalized_path + "' to SPIR-V";
        append_diagnostics(diagnostics, error);
        return false;
    }

    const auto *begin = static_cast<const uint8_t *>(output->getBufferPointer());
    spirv.assign(begin, begin + output->getBufferSize());
    error.clear();
    return true;
#endif
}
} // namespace xray::render::vulkan
