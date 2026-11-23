#pragma once
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>
#include <vulkan/vulkan.h>

namespace xray::render::vulkan
{
class PipelineCacheFingerprint
{
  public:
    explicit PipelineCacheFingerprint(std::string_view build)
    {
        constexpr std::string_view format = "vulkan-shader-assets-v1";
        append("format", format.data(), format.size());
        append("build", build.data(), build.size());
    }
    void append(std::string_view name, const void *bytes, size_t size)
    {
        append_size(name.size());
        append_bytes(name.data(), name.size());
        append_size(size);
        append_bytes(bytes, size);
    }
    uint64_t value() const { return value_; }

  private:
    void append_size(size_t size)
    {
        for (unsigned i = 0; i < 8; ++i)
            append_byte(uint8_t(uint64_t(size) >> (i * 8)));
    }
    void append_bytes(const void *bytes, size_t size)
    {
        const auto *data = static_cast<const uint8_t *>(bytes);
        for (size_t i = 0; i < size; ++i)
            append_byte(data[i]);
    }
    void append_byte(uint8_t byte)
    {
        value_ = (value_ ^ byte) * 1099511628211ull;
    }
    uint64_t value_{14695981039346656037ull};
};

// The disk wrapper rejects partial writes before the driver sees its opaque
// data. Vulkan's header then limits reuse to a compatible physical device.
class PipelineCache
{
  public:
    static constexpr size_t MaximumBytes = 64u * 1024u * 1024u;
    static constexpr size_t WrapperBytes = 40;
    static constexpr size_t MaximumDiskBytes = MaximumBytes + WrapperBytes;
    ~PipelineCache()
    {
        destroy();
    }
    PipelineCache() = default;
    PipelineCache(const PipelineCache &) = delete;
    PipelineCache &operator=(const PipelineCache &) = delete;

    bool initialize(VkDevice device, PFN_vkGetDeviceProcAddr get, const VkPhysicalDeviceProperties &properties,
                    const std::vector<uint8_t> &disk, uint64_t shader_fingerprint)
    {
        destroy();
        auto create = reinterpret_cast<PFN_vkCreatePipelineCache>(get(device, "vkCreatePipelineCache"));
        destroy_ = reinterpret_cast<PFN_vkDestroyPipelineCache>(get(device, "vkDestroyPipelineCache"));
        data_ = reinterpret_cast<PFN_vkGetPipelineCacheData>(get(device, "vkGetPipelineCacheData"));
        if (!create || !destroy_ || !data_)
            return false;
        device_ = device;
        properties_ = properties;
        VkPipelineCacheCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
        if (compatible(disk, properties, shader_fingerprint))
        {
            info.initialDataSize = disk.size() - WrapperBytes;
            info.pInitialData = disk.data() + WrapperBytes;
        }
        loaded_bytes_ = info.initialDataSize;
        if (create(device_, &info, nullptr, &handle_) != VK_SUCCESS)
        {
            handle_ = VK_NULL_HANDLE;
            info.initialDataSize = 0;
            info.pInitialData = nullptr;
            loaded_bytes_ = 0;
            if (create(device_, &info, nullptr, &handle_) != VK_SUCCESS)
            {
                handle_ = VK_NULL_HANDLE;
                destroy();
                return false;
            }
        }
        return true;
    }
    VkPipelineCache handle() const
    {
        return handle_;
    }
    size_t loaded_bytes() const
    {
        return loaded_bytes_;
    }
    std::vector<uint8_t> snapshot(uint64_t shader_fingerprint) const
    {
        if (!handle_)
            return {};
        size_t size = 0;
        if (data_(device_, handle_, &size, nullptr) != VK_SUCCESS || size > MaximumBytes || size < 32)
            return {};
        std::vector<uint8_t> result(size + WrapperBytes);
        if (data_(device_, handle_, &size, result.data() + WrapperBytes) != VK_SUCCESS || size > MaximumBytes ||
            size < 32)
            return {};
        result.resize(size + WrapperBytes);
        std::memcpy(result.data(), "XRAYVKC3", 8);
        store(result.data() + 8, size);
        store(result.data() + 16, hash(result.data() + WrapperBytes, size));
        store(result.data() + 24, (uint64_t(properties_.apiVersion) << 32) | properties_.driverVersion);
        store(result.data() + 32, shader_fingerprint);
        return result;
    }
    static bool compatible(const std::vector<uint8_t> &disk, const VkPhysicalDeviceProperties &p,
                           uint64_t shader_fingerprint)
    {
        if (disk.size() < WrapperBytes + 32 || disk.size() > MaximumDiskBytes ||
            std::memcmp(disk.data(), "XRAYVKC3", 8) != 0)
            return false;
        const size_t size = disk.size() - WrapperBytes;
        const auto *b = disk.data() + WrapperBytes;
        if (load(disk.data() + 8) != size || load(disk.data() + 16) != hash(b, size) ||
            load(disk.data() + 24) != ((uint64_t(p.apiVersion) << 32) | p.driverVersion) ||
            load(disk.data() + 32) != shader_fingerprint)
            return false;
        uint32_t header[4];
        for (unsigned field = 0; field < 4; ++field)
            header[field] = uint32_t(load(b + field * 4));
        return header[0] >= 32 && header[0] <= size && header[1] == VK_PIPELINE_CACHE_HEADER_VERSION_ONE &&
               header[2] == p.vendorID && header[3] == p.deviceID &&
               std::memcmp(b + 16, p.pipelineCacheUUID, VK_UUID_SIZE) == 0;
    }
    void destroy()
    {
        if (handle_ && destroy_)
            destroy_(device_, handle_, nullptr);
        handle_ = VK_NULL_HANDLE;
        device_ = VK_NULL_HANDLE;
        loaded_bytes_ = 0;
        properties_ = {};
    }

  private:
    static uint64_t hash(const uint8_t *bytes, size_t size)
    {
        uint64_t h = 14695981039346656037ull;
        for (size_t i = 0; i < size; ++i)
        {
            h ^= bytes[i];
            h *= 1099511628211ull;
        }
        return h;
    }
    static uint64_t load(const uint8_t *bytes)
    {
        uint64_t n = 0;
        for (unsigned i = 0; i < 8; ++i)
            n |= uint64_t(bytes[i]) << (8 * i);
        return n;
    }
    static void store(uint8_t *bytes, uint64_t n)
    {
        for (unsigned i = 0; i < 8; ++i)
            bytes[i] = uint8_t(n >> (8 * i));
    }
    VkDevice device_{};
    VkPipelineCache handle_{};
    PFN_vkDestroyPipelineCache destroy_{};
    PFN_vkGetPipelineCacheData data_{};
    size_t loaded_bytes_{};
    VkPhysicalDeviceProperties properties_{};
};
} // namespace xray::render::vulkan
