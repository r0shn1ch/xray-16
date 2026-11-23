////////////////////////////////////////////////////////////////////////////
//	Module 		: alife_registry_container.cpp
//	Created 	: 01.07.2004
//  Modified 	: 01.07.2004
//	Author		: Dmitriy Iassenev
//	Description : ALife registry container class
////////////////////////////////////////////////////////////////////////////

#include "StdAfx.h"
#include "alife_registry_container.h"
#include "Common/object_interfaces.h"
#include "alife_space.h"
#include "Common/object_type_traits.h"
#include <type_traits>

template <typename TContainer, typename TList>
struct RegistryHelper;

template <typename TContainer>
struct RegistryHelper<TContainer, Loki::NullType>
{
    static void Save(TContainer*, IWriter&) {};
    static bool Load(TContainer*, IReader&) { return true; }
};

template <typename TContainer, typename Head, typename Tail>
struct RegistryHelper<TContainer, Loki::Typelist<Head, Tail>>
{
    static constexpr bool isSerializable =
        object_type_traits::is_base_and_derived<ISerializable, Head>::value;

    static void Save(TContainer* self, IWriter& writer)
    {
        RegistryHelper<TContainer, Tail>::Save(self, writer);
        if constexpr (isSerializable)
            self->Head::save(writer);
    };

    static bool Load(TContainer* self, IReader& reader)
    {
        if (!RegistryHelper<TContainer, Tail>::Load(self, reader))
            return false;
        if constexpr (isSerializable)
            self->Head::load(reader);
        if constexpr (std::is_same_v<Head, CMapLocationRegistry>)
            return self->Head::load_succeeded();
        return true;
    }
};

void CALifeRegistryContainer::load(IReader& file_stream)
{
    IReader* registry_stream = file_stream.open_chunk(REGISTRY_CHUNK_DATA);
    if (!registry_stream)
    {
        Msg("! Can't find chunk REGISTRY_CHUNK_DATA while loading ALife registries");
        return;
    }

    if (!RegistryHelper<CALifeRegistryContainer, TYPE_LIST>::Load(this, *registry_stream))
        Msg("! Remaining ALife registries skipped after invalid map-location data");
    registry_stream->close();
}

void CALifeRegistryContainer::save(IWriter& memory_stream)
{
    memory_stream.open_chunk(REGISTRY_CHUNK_DATA);
    RegistryHelper<CALifeRegistryContainer, TYPE_LIST>::Save(this, memory_stream);
    memory_stream.close_chunk();
}
