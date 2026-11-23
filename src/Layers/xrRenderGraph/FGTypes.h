#pragma once

#include <atomic>
#include <cstdint>

namespace xray::render::framegraph {

    constexpr uint32_t INVALID_INDEX = 0xFFFFFFFF;
    constexpr uint32_t MAX_RENDER_TARGETS = 8;
    constexpr uint32_t MAX_PASS_DEPENDENCIES = 32;

    struct VirtualResourceHandle {
        uint32_t index = INVALID_INDEX;
        uint64_t generation = 0;
        uint32_t version = 0;

        bool is_valid() const { return index != INVALID_INDEX; }
        bool operator==(const VirtualResourceHandle& other) const {
            return index == other.index && generation == other.generation && version == other.version;
        }
        bool operator!=(const VirtualResourceHandle& other) const { return !(*this == other); }
    };

    inline uint64_t allocate_graph_generation()
    {
        static std::atomic<uint64_t> next{1};
        return next.fetch_add(1, std::memory_order_relaxed);
    }

    template <typename Kind>
    struct TypedResourceHandle : VirtualResourceHandle
    {
        TypedResourceHandle() = default;
        TypedResourceHandle(uint32_t index, uint32_t version, uint64_t generation)
            : VirtualResourceHandle{index, generation, version} {}
    };

    struct PassHandle {
        uint32_t index = INVALID_INDEX;

        bool is_valid() const { return index != INVALID_INDEX; }
        bool operator==(const PassHandle& other) const { return index == other.index; }
        bool operator!=(const PassHandle& other) const { return index != other.index; }
    };

    enum class ResourceState : uint8_t {
        Undefined,
        RenderTarget,
        DepthStencilWrite,
        DepthStencilRead,
        ShaderResource,
        UnorderedAccess,
        IndirectArgument,
        CopySource,
        CopyDest,
        Present,
        Common,
        DepthStencilReadShaderResource,
        AccelStructRead,
        AccelStructWrite,
        AccelStructBuildInput,
        AccelStructBuildBlas,
        VertexBuffer,
        IndexBuffer,
    };

    inline const char* ResourceStateToString(ResourceState state) {
        switch (state) {
        case ResourceState::Undefined: return "Undefined";
        case ResourceState::RenderTarget: return "RenderTarget";
        case ResourceState::DepthStencilWrite: return "DepthStencilWrite";
        case ResourceState::DepthStencilRead: return "DepthStencilRead";
        case ResourceState::DepthStencilReadShaderResource: return "DepthStencilReadShaderResource";
        case ResourceState::ShaderResource: return "ShaderResource";
        case ResourceState::UnorderedAccess: return "UnorderedAccess";
        case ResourceState::IndirectArgument: return "IndirectArgument";
        case ResourceState::CopySource: return "CopySource";
        case ResourceState::CopyDest: return "CopyDest";
        case ResourceState::Present: return "Present";
        case ResourceState::Common: return "Common";
        case ResourceState::AccelStructRead: return "AccelStructRead";
        case ResourceState::AccelStructWrite: return "AccelStructWrite";
        case ResourceState::AccelStructBuildInput: return "AccelStructBuildInput";
        case ResourceState::AccelStructBuildBlas: return "AccelStructBuildBlas";
        case ResourceState::VertexBuffer: return "VertexBuffer";
        case ResourceState::IndexBuffer: return "IndexBuffer";
        default: return "Unknown";
        }
    }

}
