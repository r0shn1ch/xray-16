#include "xrCore/stdafx.h"
#include "xrGame/game_news_serialization.h"

#include <array>
#include <cassert>
#include <cstring>

namespace
{
constexpr char TextureRectMarker[] = "$TEXTURE_RECT$";
constexpr u32 NextRecordType = 0x12345678;

void append_bytes(std::array<u8, 64>& bytes, size_t& offset, const void* source, size_t size)
{
    std::memcpy(bytes.data() + offset, source, size);
    offset += size;
}

void test_legacy_rect_is_skipped_before_next_record()
{
    std::array<u8, 64> bytes{};
    size_t size = 0;
    const std::array<float, 4> rect{ 0.f, 0.f, 1.f, 1.f };
    append_bytes(bytes, size, TextureRectMarker, sizeof(TextureRectMarker));
    append_bytes(bytes, size, rect.data(), sizeof(rect));
    append_bytes(bytes, size, &NextRecordType, sizeof(NextRecordType));

    IReader stream(bytes.data(), size);
    assert(game_news_serialization::skip_legacy_texture_rect(stream));
    assert(stream.tell() == sizeof(TextureRectMarker) + sizeof(rect));
    assert(stream.r_u32() == NextRecordType);
}

void test_current_format_does_not_consume_next_record()
{
    std::array<u8, sizeof(u32)> bytes{};
    std::memcpy(bytes.data(), &NextRecordType, sizeof(NextRecordType));

    IReader stream(bytes.data(), bytes.size());
    assert(!game_news_serialization::skip_legacy_texture_rect(stream));
    assert(stream.tell() == 0);
    assert(stream.r_u32() == NextRecordType);
}

void test_truncated_legacy_marker_is_left_untouched()
{
    std::array<u8, sizeof(TextureRectMarker)> bytes{};
    std::memcpy(bytes.data(), TextureRectMarker, bytes.size());

    IReader stream(bytes.data(), bytes.size());
    assert(!game_news_serialization::skip_legacy_texture_rect(stream));
    assert(stream.tell() == 0);
}
} // namespace

int main()
{
    test_legacy_rect_is_skipped_before_next_record();
    test_current_format_does_not_consume_next_record();
    test_truncated_legacy_marker_is_left_untouched();
    return 0;
}
