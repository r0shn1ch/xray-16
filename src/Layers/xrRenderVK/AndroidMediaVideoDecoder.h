#pragma once

#if defined(ANDROID)

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace xray::render::vulkan
{
class AndroidMediaVideoDecoder
{
public:
    AndroidMediaVideoDecoder();
    ~AndroidMediaVideoDecoder();

    AndroidMediaVideoDecoder(const AndroidMediaVideoDecoder&) = delete;
    AndroidMediaVideoDecoder& operator=(const AndroidMediaVideoDecoder&) = delete;

    bool load(const char* path, uint32_t& width, uint32_t& height, std::string& error);
    void play(bool looped, uint32_t time);
    void stop();
    bool playing() const;
    bool update(uint32_t time, std::vector<uint32_t>& pixels);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}

#endif // ANDROID
