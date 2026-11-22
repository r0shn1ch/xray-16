#pragma once

#include "FGTypes.h"
#include <nvrhi/nvrhi.h>

namespace xray::render::framegraph
{
// The same state vocabulary is used by graph passes on every NVRHI backend.
// Undefined is the upstream graph's common starting state, not a GPU discard.
inline bool to_nvrhi_resource_state(ResourceState state, nvrhi::ResourceStates& result)
{
    using S = nvrhi::ResourceStates;
    switch (state)
    {
    case ResourceState::Undefined:
    case ResourceState::Common: result = S::Common; break;
    case ResourceState::RenderTarget: result = S::RenderTarget; break;
    case ResourceState::DepthStencilWrite: result = S::DepthWrite; break;
    case ResourceState::DepthStencilRead: result = S::DepthRead; break;
    case ResourceState::DepthStencilReadShaderResource: result = S::DepthRead | S::ShaderResource; break;
    case ResourceState::ShaderResource: result = S::ShaderResource; break;
    case ResourceState::UnorderedAccess: result = S::UnorderedAccess; break;
    case ResourceState::IndirectArgument: result = S::IndirectArgument; break;
    case ResourceState::CopySource: result = S::CopySource; break;
    case ResourceState::CopyDest: result = S::CopyDest; break;
    case ResourceState::Present: result = S::Present; break;
    case ResourceState::AccelStructRead: result = S::AccelStructRead; break;
    case ResourceState::AccelStructWrite: result = S::AccelStructWrite; break;
    case ResourceState::AccelStructBuildInput: result = S::AccelStructBuildInput; break;
    case ResourceState::AccelStructBuildBlas: result = S::AccelStructBuildBlas; break;
    case ResourceState::VertexBuffer: result = S::VertexBuffer; break;
    case ResourceState::IndexBuffer: result = S::IndexBuffer; break;
    default: return false;
    }
    return true;
}

enum class NvrhiResourceKind { Texture, Buffer, AccelerationStructure };

inline bool nvrhi_state_allowed(NvrhiResourceKind kind, ResourceState state)
{
    switch (kind)
    {
    case NvrhiResourceKind::AccelerationStructure:
        return state == ResourceState::AccelStructRead || state == ResourceState::AccelStructWrite ||
            state == ResourceState::AccelStructBuildBlas;
    case NvrhiResourceKind::Texture:
        return state == ResourceState::Undefined || state == ResourceState::Common ||
            state == ResourceState::RenderTarget || state == ResourceState::DepthStencilWrite ||
            state == ResourceState::DepthStencilRead || state == ResourceState::DepthStencilReadShaderResource ||
            state == ResourceState::ShaderResource || state == ResourceState::UnorderedAccess ||
            state == ResourceState::CopySource || state == ResourceState::CopyDest || state == ResourceState::Present;
    case NvrhiResourceKind::Buffer:
        return state == ResourceState::Undefined || state == ResourceState::Common ||
            state == ResourceState::ShaderResource || state == ResourceState::UnorderedAccess ||
            state == ResourceState::IndirectArgument || state == ResourceState::CopySource ||
            state == ResourceState::CopyDest || state == ResourceState::AccelStructBuildInput ||
            state == ResourceState::VertexBuffer || state == ResourceState::IndexBuffer;
    }
    return false;
}

template<class CommandList>
bool transition_nvrhi_texture(CommandList& command, nvrhi::ITexture* texture,
    nvrhi::TextureSubresourceSet range, ResourceState state)
{
    nvrhi::ResourceStates native{};
    if (!texture || !nvrhi_state_allowed(NvrhiResourceKind::Texture, state) ||
        !to_nvrhi_resource_state(state, native)) return false;
    command.setTextureState(texture, range, native);
    return true;
}

template<class CommandList>
bool transition_nvrhi_buffer(CommandList& command, nvrhi::IBuffer* buffer, ResourceState state)
{
    nvrhi::ResourceStates native{};
    if (!buffer || !nvrhi_state_allowed(NvrhiResourceKind::Buffer, state) ||
        !to_nvrhi_resource_state(state, native)) return false;
    command.setBufferState(buffer, native);
    return true;
}

template<class CommandList>
bool transition_nvrhi_acceleration_structure(CommandList& command, nvrhi::rt::IAccelStruct* acceleration,
    ResourceState state)
{
    nvrhi::ResourceStates native{};
    if (!acceleration || !nvrhi_state_allowed(NvrhiResourceKind::AccelerationStructure, state) ||
        !to_nvrhi_resource_state(state, native)) return false;
    command.setAccelStructState(acceleration, native);
    return true;
}
}
