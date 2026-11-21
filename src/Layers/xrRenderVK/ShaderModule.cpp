#include "ShaderModule.h"

#include <cstring>
#include <utility>
#include <vector>

namespace xray::render::vulkan
{
namespace
{
constexpr uint32_t SpirvMagic = 0x07230203;
}

uint64_t shader_bytes_fingerprint(const void* bytes, size_t size)
{
    uint64_t value = 14695981039346656037ull;
    const auto* data = static_cast<const uint8_t*>(bytes);
    for (size_t i = 0; i < size; ++i)
        value = (value ^ data[i]) * 1099511628211ull;
    for (unsigned i = 0; i < 8; ++i)
        value = (value ^ uint8_t(uint64_t(size) >> (i * 8))) * 1099511628211ull;
    return value;
}

bool has_spirv_entry(const void* bytes, size_t size, uint32_t stage, const char* entry)
{
    if (!bytes || !entry || !*entry || size < 20 || size % sizeof(uint32_t))
        return false;
    std::vector<uint32_t> words(size / sizeof(uint32_t));
    std::memcpy(words.data(), bytes, size);
    if (words[0] != SpirvMagic || !words[3] || words[4])
        return false;
    bool found = false;
    for (size_t offset = 5; offset < words.size();)
    {
        const uint32_t count = words[offset] >> 16;
        const uint32_t opcode = words[offset] & 0xffffu;
        if (!count || count > words.size() - offset)
            return false;
        if (opcode == 15 && count >= 4 && words[offset + 1] == stage)
        {
            const char* name = reinterpret_cast<const char*>(words.data() + offset + 3);
            const size_t capacity = (count - 3) * sizeof(uint32_t);
            const void* terminator = std::memchr(name, 0, capacity);
            if (terminator && std::strlen(entry) ==
                static_cast<size_t>(static_cast<const char*>(terminator) - name) &&
                std::memcmp(name, entry, std::strlen(entry)) == 0)
                found = true;
        }
        offset += count;
    }
    return found;
}

ShaderModule::~ShaderModule()
{
    destroy();
}

ShaderModule::ShaderModule(ShaderModule&& other) noexcept
{
    *this = std::move(other);
}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) noexcept
{
    if (this != &other)
    {
        destroy();
        m_device = other.m_device;
        m_module = other.m_module;
        m_destroy = other.m_destroy;
        other.m_device = VK_NULL_HANDLE;
        other.m_module = VK_NULL_HANDLE;
        other.m_destroy = nullptr;
    }
    return *this;
}

bool ShaderModule::initialize(VkDevice device, const ShaderModuleDispatch& dispatch,
    const uint32_t* code, size_t size, std::string& error)
{
    destroy();
    if (!device || !dispatch.create || !dispatch.destroy || !code || size < 5 * sizeof(uint32_t) ||
        size % sizeof(uint32_t) != 0 || code[0] != SpirvMagic)
    {
        error = "invalid SPIR-V shader module input";
        return false;
    }

    VkShaderModuleCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = size;
    info.pCode = code;
    const VkResult result = dispatch.create(device, &info, nullptr, &m_module);
    if (result != VK_SUCCESS)
    {
        m_module = VK_NULL_HANDLE;
        error = "vkCreateShaderModule failed: VkResult=" + std::to_string(static_cast<int>(result));
        return false;
    }

    m_device = device;
    m_destroy = dispatch.destroy;
    error.clear();
    return true;
}

bool ShaderModule::initialize_bytes(VkDevice device, const ShaderModuleDispatch& dispatch,
    const void* bytes, size_t size, std::string& error)
{
    if (!bytes || size < 5 * sizeof(uint32_t) || size % sizeof(uint32_t))
    {
        destroy();
        error = "invalid SPIR-V shader module input";
        return false;
    }
    std::vector<uint32_t> words(size / sizeof(uint32_t));
    std::memcpy(words.data(), bytes, size);
    return initialize(device, dispatch, words.data(), size, error);
}

void ShaderModule::destroy()
{
    if (m_device && m_module && m_destroy)
        m_destroy(m_device, m_module, nullptr);
    m_device = VK_NULL_HANDLE;
    m_module = VK_NULL_HANDLE;
    m_destroy = nullptr;
}

ShaderModule* ShaderModuleCache::find(const std::string& key) const
{
    const auto it = modules_.find(key);
    return it == modules_.end() ? nullptr : it->second.module.get();
}

bool ShaderModuleCache::load(const std::string& key, VkDevice device, const ShaderModuleDispatch& dispatch,
    const void* bytes, size_t size, uint32_t stage, const char* entry,
    ShaderModule*& result, std::string& error)
{
    result = nullptr;
    if (key.empty() || !has_spirv_entry(bytes, size, stage, entry))
    {
        error = "invalid SPIR-V stage or entry for '" + key + "'";
        return false;
    }
    const uint64_t fingerprint = shader_bytes_fingerprint(bytes, size);
    if (const auto found = modules_.find(key); found != modules_.end())
    {
        if (found->second.fingerprint != fingerprint || found->second.device != device)
        {
            error = "cached SPIR-V changed for '" + key + "'; rebuild after GPU idle";
            return false;
        }
        result = found->second.module.get();
        error.clear();
        return true;
    }
    auto module = std::make_unique<ShaderModule>();
    if (!module->initialize_bytes(device, dispatch, bytes, size, error))
        return false;
    result = module.get();
    modules_.emplace(key, Entry{std::move(module), fingerprint, device});
    return true;
}
}
