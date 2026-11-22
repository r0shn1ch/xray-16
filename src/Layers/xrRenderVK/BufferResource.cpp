#include "BufferResource.h"

#include <nvrhi/vulkan.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <utility>
#include <vector>

namespace xray::render::vulkan
{
struct BufferMemoryBlock
{
    VkDevice device{};
    VkDeviceMemory memory{};
    uint32_t memory_type{};
    VkDeviceSize size{};
    std::vector<std::pair<VkDeviceSize, VkDeviceSize>> ranges;
    PFN_vkFreeMemory free_memory{};
    PFN_vkUnmapMemory unmap_memory{};
    void* mapped{};

    ~BufferMemoryBlock()
    {
        if (device && memory && mapped && unmap_memory) unmap_memory(device, memory);
        if (device && memory && free_memory) free_memory(device, memory, nullptr);
    }
};

namespace
{
std::mutex& pool_mutex() { static auto* mutex = new std::mutex; return *mutex; }
std::vector<std::weak_ptr<BufferMemoryBlock>>& pool_blocks()
{
    static auto* blocks = new std::vector<std::weak_ptr<BufferMemoryBlock>>;
    return *blocks;
}

void release_pooled(std::shared_ptr<BufferMemoryBlock>& block, VkDeviceSize offset)
{
    {
        std::lock_guard lock(pool_mutex());
        auto& ranges = block->ranges;
        const auto range = std::lower_bound(ranges.begin(), ranges.end(), offset,
            [](const auto& candidate, VkDeviceSize start) { return candidate.first < start; });
        if (range != ranges.end() && range->first == offset)
            ranges.erase(range);
    }
    block.reset();
}

bool allocate_pooled(VkDevice device, uint32_t memory_type, const VkMemoryRequirements& requirements,
    bool host_visible, const BufferResourceDispatch& vk, std::shared_ptr<BufferMemoryBlock>& block,
    VkDeviceSize& offset, std::string& error, VkResult* allocation_result)
{
    std::lock_guard lock(pool_mutex());
    auto& blocks = pool_blocks();
    const VkDeviceSize alignment = std::max<VkDeviceSize>(requirements.alignment, 1);
    for (auto it = blocks.begin(); it != blocks.end(); )
    {
        auto candidate = it->lock();
        if (!candidate) { it = blocks.erase(it); continue; }
        ++it;
        if (candidate->device != device || candidate->memory_type != memory_type ||
            static_cast<bool>(candidate->mapped) != host_visible) continue;
        VkDeviceSize free_start = 0;
        auto& ranges = candidate->ranges;
        for (auto range = ranges.begin(); ; ++range)
        {
            if (free_start > std::numeric_limits<VkDeviceSize>::max() - (alignment - 1))
                break;
            const VkDeviceSize aligned = (free_start + alignment - 1) / alignment * alignment;
            const VkDeviceSize free_end = range == ranges.end() ? candidate->size : range->first;
            if (aligned <= free_end && requirements.size <= free_end - aligned)
            {
                ranges.emplace(range, aligned, aligned + requirements.size);
                offset = aligned;
                block = std::move(candidate);
                return true;
            }
            if (range == ranges.end())
                break;
            free_start = range->second;
        }
    }
    constexpr VkDeviceSize block_size = 8u * 1024u * 1024u;
    const VkDeviceSize capacity = std::max(block_size, requirements.size);
    VkMemoryAllocateInfo info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    info.allocationSize = capacity;
    info.memoryTypeIndex = memory_type;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult result = vk.allocate_memory(device, &info, nullptr, &memory);
    if (result != VK_SUCCESS && capacity > requirements.size)
    {
        info.allocationSize = requirements.size;
        result = vk.allocate_memory(device, &info, nullptr, &memory);
    }
    if (result != VK_SUCCESS)
    {
        if (allocation_result)
            *allocation_result = result;
        error = "vkAllocateMemory failed for pooled buffer block result=" + std::to_string(result) +
            " bytes=" + std::to_string(info.allocationSize);
        return false;
    }
    void* mapped = nullptr;
    if (host_visible && (vk.map_memory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || !mapped))
    {
        vk.free_memory(device, memory, nullptr);
        error = "vkMapMemory failed for pooled host-visible buffer block";
        return false;
    }
    block = std::make_shared<BufferMemoryBlock>();
    block->device = device;
    block->memory = memory;
    block->memory_type = memory_type;
    block->size = info.allocationSize;
    block->free_memory = vk.free_memory;
    block->unmap_memory = vk.unmap_memory;
    block->mapped = mapped;
    block->ranges.emplace_back(0, requirements.size);
    blocks.push_back(block);
    offset = 0;
    return true;
}

uint32_t find_memory_type(uint32_t type_bits, VkMemoryPropertyFlags required,
    const VkPhysicalDeviceMemoryProperties& properties)
{
    for (uint32_t index = 0; index < properties.memoryTypeCount; ++index)
        if ((type_bits & (1u << index)) &&
            (properties.memoryTypes[index].propertyFlags & required) == required)
            return index;
    return UINT32_MAX;
}

bool complete(const BufferResourceDispatch& vk)
{
    return vk.create_buffer && vk.destroy_buffer && vk.get_buffer_memory_requirements &&
        vk.allocate_memory && vk.free_memory && vk.bind_buffer_memory && vk.map_memory && vk.unmap_memory;
}

uint32_t collect_queue_families(const BufferResourceDispatch& dispatch, VkQueueFlags queue_flags,
    uint32_t (&families)[3])
{
    uint32_t count = 0;
    const auto append = [&](uint32_t family, VkQueueFlags role)
    {
        if ((queue_flags && !(queue_flags & role)) || family == VK_QUEUE_FAMILY_IGNORED ||
            std::find(families, families + count, family) != families + count)
            return;
        families[count++] = family;
    };
    append(dispatch.graphics_queue_family, VK_QUEUE_GRAPHICS_BIT);
    append(dispatch.compute_queue_family, VK_QUEUE_COMPUTE_BIT);
    append(dispatch.transfer_queue_family, VK_QUEUE_TRANSFER_BIT);
    return count;
}

bool nvrhi_buffer_compatible(VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required_memory_properties, bool pooled, bool shared_across_queue_families,
    const BufferResourceDispatch& dispatch)
{
    if (!dispatch.nvrhi_device || !size || pooled || shared_across_queue_families ||
        required_memory_properties != VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        return false;

    constexpr VkBufferUsageFlags unsupported_usage =
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;
    constexpr VkBufferUsageFlags supported_usage =
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    return !(usage & unsupported_usage) && !(usage & ~supported_usage);
}

VkBuffer native_buffer(nvrhi::IBuffer* buffer)
{
    if (!buffer) return VK_NULL_HANDLE;
    const nvrhi::Object object = buffer->getNativeObject(nvrhi::ObjectTypes::VK_Buffer);
#if defined(VK_USE_64_BIT_PTR_DEFINES) && VK_USE_64_BIT_PTR_DEFINES
    return static_cast<VkBuffer>(object.pointer);
#else
    return static_cast<VkBuffer>(object.integer);
#endif
}
}

BufferResource::~BufferResource()
{
    destroy();
}

BufferResource::BufferResource(BufferResource&& other) noexcept
{
    *this = std::move(other);
}

BufferResource& BufferResource::operator=(BufferResource&& other) noexcept
{
    if (this != &other)
    {
        destroy();
        m_device = other.m_device;
        m_buffer = other.m_buffer;
        m_memory = other.m_memory;
        m_sharing_mode = other.m_sharing_mode;
        m_sharing_queue_families = std::move(other.m_sharing_queue_families);
        m_nvrhi_buffer = other.m_nvrhi_buffer;
        m_size = other.m_size;
        m_mapped = other.m_mapped;
        m_pool_block = std::move(other.m_pool_block);
        m_pool_offset = other.m_pool_offset;
        m_vk = other.m_vk;
        other.m_device = VK_NULL_HANDLE;
        other.m_buffer = VK_NULL_HANDLE;
        other.m_memory = VK_NULL_HANDLE;
        other.m_sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
        other.m_sharing_queue_families.clear();
        other.m_nvrhi_buffer = nullptr;
        other.m_size = 0;
        other.m_mapped = nullptr;
        other.m_vk = {};
    }
    return *this;
}

bool BufferResource::initialize(VkDevice device, VkDeviceSize size, VkBufferUsageFlags usage,
    VkMemoryPropertyFlags required_memory_properties,
    const VkPhysicalDeviceMemoryProperties& memory_properties,
    const BufferResourceDispatch& dispatch, std::string& error, bool pooled,
    VkQueueFlags shared_queue_flags, VkResult* allocation_result)
{
    if (allocation_result)
        *allocation_result = VK_SUCCESS;
    destroy();
    if (!device || !size || !usage || !complete(dispatch))
    {
        error = "Vulkan buffer resource requires a device, size, usage and complete procedures";
        return false;
    }

    std::string nvrhi_error;
    constexpr VkQueueFlags supported_queue_flags = VK_QUEUE_GRAPHICS_BIT |
        VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    if (shared_queue_flags & ~supported_queue_flags)
    {
        error = "Vulkan buffer resource has invalid shared queue flags";
        return false;
    }
    uint32_t queue_families[3]{};
    const uint32_t queue_family_count = collect_queue_families(dispatch, shared_queue_flags,
        queue_families);
    const bool needs_concurrent_sharing = shared_queue_flags && queue_family_count > 1;
    const bool use_pooled_memory = pooled || needs_concurrent_sharing;
    if (nvrhi_buffer_compatible(size, usage, required_memory_properties, pooled,
            needs_concurrent_sharing, dispatch))
    {
        nvrhi::BufferDesc desc;
        desc.byteSize = size;
        desc.isVertexBuffer = (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) != 0;
        desc.isIndexBuffer = (usage & VK_BUFFER_USAGE_INDEX_BUFFER_BIT) != 0;
        desc.isConstantBuffer = (usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0;
        desc.canHaveUAVs = (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0;
        desc.isDrawIndirectArgs = (usage & VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT) != 0;
        desc.initialState = nvrhi::ResourceStates::Common;
        desc.debugName = "xrRenderVK BufferResource";

        try
        {
            nvrhi::BufferHandle nvrhi_buffer = dispatch.nvrhi_device->createBuffer(desc);
            if (nvrhi_buffer)
            {
                const VkBuffer buffer = native_buffer(nvrhi_buffer);
                if (buffer)
                {
                    nvrhi_buffer->AddRef();
                    m_device = device;
                    m_buffer = buffer;
                    m_sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
                    m_sharing_queue_families.assign(queue_families,
                        queue_families + queue_family_count);
                    m_size = size;
                    m_nvrhi_buffer = nvrhi_buffer.Get();
                    m_vk = dispatch;
                    error.clear();
                    return true;
                }
                nvrhi_error = "NVRHI buffer does not expose its native VkBuffer";
            }
            else
                nvrhi_error = "NVRHI returned no device-local Vulkan buffer";
        }
        catch (const std::exception& exception)
        {
            nvrhi_error = exception.what();
        }
        catch (...)
        {
            nvrhi_error = "unknown NVRHI buffer creation failure";
        }
    }
    const auto set_vulkan_error = [&](const std::string& message) {
        error = message;
        if (!nvrhi_error.empty())
            error += "; NVRHI buffer creation also failed: " + nvrhi_error;
    };

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    if (queue_family_count > 1)
    {
        buffer_info.sharingMode = VK_SHARING_MODE_CONCURRENT;
        buffer_info.queueFamilyIndexCount = queue_family_count;
        buffer_info.pQueueFamilyIndices = queue_families;
    }
    else
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer buffer = VK_NULL_HANDLE;
    if (dispatch.create_buffer(device, &buffer_info, nullptr, &buffer) != VK_SUCCESS)
    {
        set_vulkan_error("vkCreateBuffer failed");
        return false;
    }

    VkMemoryRequirements requirements{};
    dispatch.get_buffer_memory_requirements(device, buffer, &requirements);
    const VkMemoryPropertyFlags memory_flags = required_memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        ? required_memory_properties | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        : required_memory_properties;
    const uint32_t memory_type = find_memory_type(requirements.memoryTypeBits,
        memory_flags, memory_properties);
    if (memory_type == UINT32_MAX)
    {
        dispatch.destroy_buffer(device, buffer, nullptr);
        set_vulkan_error("no Vulkan memory type satisfies the buffer requirements");
        return false;
    }

    std::shared_ptr<BufferMemoryBlock> pool_block;
    VkDeviceSize binding_offset = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    // Device-address allocations need a memory allocation flag. The general
    // buffer pool does not partition blocks by allocation flags yet.
    const bool device_address = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    if (use_pooled_memory && !device_address)
    {
        if (!allocate_pooled(device, memory_type, requirements,
                (required_memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0,
                dispatch, pool_block, binding_offset, error, allocation_result))
        {
            dispatch.destroy_buffer(device, buffer, nullptr);
            if (!nvrhi_error.empty())
                error += "; NVRHI buffer creation also failed: " + nvrhi_error;
            return false;
        }
        memory = pool_block->memory;
    }
    else
    {
        VkMemoryAllocateInfo allocate_info{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocate_info.allocationSize = requirements.size;
        allocate_info.memoryTypeIndex = memory_type;
        VkMemoryAllocateFlagsInfo address_flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
        if (device_address)
        {
            address_flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
            allocate_info.pNext = &address_flags;
        }
        const VkResult allocated = dispatch.allocate_memory(device, &allocate_info, nullptr, &memory);
        if (allocated != VK_SUCCESS)
        {
            if (allocation_result)
                *allocation_result = allocated;
            dispatch.destroy_buffer(device, buffer, nullptr);
            set_vulkan_error("vkAllocateMemory failed for buffer resource");
            return false;
        }
    }

    if (dispatch.bind_buffer_memory(device, buffer, memory, binding_offset) != VK_SUCCESS)
    {
        dispatch.destroy_buffer(device, buffer, nullptr);
        if (pool_block) release_pooled(pool_block, binding_offset);
        else dispatch.free_memory(device, memory, nullptr);
        set_vulkan_error("vkBindBufferMemory failed for buffer resource");
        return false;
    }

    void* mapped = nullptr;
    if (pool_block && pool_block->mapped)
        mapped = static_cast<unsigned char*>(pool_block->mapped) + binding_offset;
    else if (required_memory_properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
    {
        if (dispatch.map_memory(device, memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || !mapped)
        {
            dispatch.destroy_buffer(device, buffer, nullptr);
            if (pool_block) release_pooled(pool_block, binding_offset);
            else dispatch.free_memory(device, memory, nullptr);
            set_vulkan_error("vkMapMemory failed for host-visible buffer resource");
            return false;
        }
    }

    m_device = device;
    m_buffer = buffer;
    m_sharing_mode = buffer_info.sharingMode;
    m_sharing_queue_families.assign(queue_families, queue_families + queue_family_count);
    m_memory = memory;
    m_size = size;
    m_mapped = mapped;
    m_pool_block = std::move(pool_block);
    m_pool_offset = binding_offset;
    m_vk = dispatch;
    error.clear();
    return true;
}

bool BufferResource::write(VkDeviceSize offset, const void* data, size_t size, std::string& error)
{
    if (!m_buffer || !m_mapped || !data || !size || offset > m_size || size > m_size - offset ||
        offset > std::numeric_limits<size_t>::max())
    {
        error = "buffer write requires mapped storage and a non-empty in-range data span";
        return false;
    }
    std::memcpy(static_cast<unsigned char*>(m_mapped) + offset, data, size);
    error.clear();
    return true;
}

bool BufferResource::read(VkDeviceSize offset, void* data, size_t size, std::string& error) const
{
    if (!m_buffer || !m_mapped || !data || !size || offset > m_size || size > m_size - offset ||
        offset > std::numeric_limits<size_t>::max())
    {
        error = "buffer read requires mapped storage and a non-empty in-range data span";
        return false;
    }
    std::memcpy(data, static_cast<const unsigned char*>(m_mapped) + offset, size);
    error.clear();
    return true;
}

void BufferResource::destroy()
{
    const bool nvrhi_owned = m_nvrhi_buffer != nullptr;
    if (nvrhi_owned)
    {
        m_nvrhi_buffer->Release();
        m_nvrhi_buffer = nullptr;
    }
    else
    {
        if (m_device && m_memory && m_mapped && !m_pool_block && m_vk.unmap_memory)
            m_vk.unmap_memory(m_device, m_memory);
        if (m_device && m_buffer && m_vk.destroy_buffer)
            m_vk.destroy_buffer(m_device, m_buffer, nullptr);
        if (m_pool_block) release_pooled(m_pool_block, m_pool_offset);
        else if (m_device && m_memory && m_vk.free_memory)
            m_vk.free_memory(m_device, m_memory, nullptr);
    }
    m_device = VK_NULL_HANDLE;
    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_sharing_mode = VK_SHARING_MODE_EXCLUSIVE;
    m_sharing_queue_families.clear();
    m_size = 0;
    m_mapped = nullptr;
    m_vk = {};
}
}
