#include "xrCore/stdafx.h"
#include "game_news_serialization.h"

#include <cstring>

bool game_news_serialization::skip_legacy_texture_rect(IReader& stream)
{
    constexpr char Marker[] = "$TEXTURE_RECT$";
    constexpr size_t SerializedRectSize = sizeof(float) * 4;
    constexpr size_t SerializedBlockSize = sizeof(Marker) + SerializedRectSize;

    if (stream.elapsed() < static_cast<intptr_t>(SerializedBlockSize))
        return false;

    if (std::memcmp(stream.pointer(), Marker, sizeof(Marker)) != 0)
        return false;

    stream.advance(SerializedBlockSize);
    return true;
}
