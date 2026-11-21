#include "xrEngine/stdafx.h"

#if defined(ANDROID)

#include "AndroidMediaVideoDecoder.h"

#include "xrCore/FS.h"

#include <SDL.h>
#include <SDL_system.h>
#include <jni.h>

#include <cerrno>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace xray::render::vulkan
{
namespace
{
constexpr size_t MaxVideoFileSize = 256ull * 1024 * 1024;
constexpr size_t MaxDecodedFrameBytes = 32ull * 1024 * 1024;

bool ReadVideoFile(const char* path, std::vector<uint8_t>& contents, std::string& error)
{
    IReader* reader = FS.r_open(path);
    if (!reader)
    {
        error = "video file was not found in the game filesystem";
        return false;
    }

    const size_t length = reader->length();
    if (!length || length > MaxVideoFileSize)
    {
        FS.r_close(reader);
        error = "video file is empty or exceeds the 256 MiB Android decoder limit";
        return false;
    }

    contents.resize(length);
    std::memcpy(contents.data(), reader->pointer(), length);
    FS.r_close(reader);
    return true;
}

bool WriteAll(int fd, const uint8_t* bytes, size_t size)
{
    size_t offset = 0;
    while (offset < size)
    {
        const ssize_t written = ::write(fd, bytes + offset, size - offset);
        if (written < 0 && errno == EINTR)
            continue;
        if (written <= 0)
            return false;
        offset += static_cast<size_t>(written);
    }
    return true;
}

struct ActivityAccess
{
    JNIEnv* env{};
    jobject activity{};
    jclass activity_class{};

    bool open(std::string& error)
    {
        env = static_cast<JNIEnv*>(SDL_AndroidGetJNIEnv());
        activity = static_cast<jobject>(SDL_AndroidGetActivity());
        if (!env || !activity)
        {
            error = "Android video decoder could not access the SDL activity";
            close();
            return false;
        }
        activity_class = env->GetObjectClass(activity);
        if (!activity_class || env->ExceptionCheck())
        {
            if (env->ExceptionCheck()) env->ExceptionClear();
            error = "Android video decoder could not inspect the SDL activity";
            close();
            return false;
        }
        return true;
    }

    void close()
    {
        if (env)
        {
            if (activity_class) env->DeleteLocalRef(activity_class);
            if (activity) env->DeleteLocalRef(activity);
        }
        activity_class = nullptr;
        activity = nullptr;
        env = nullptr;
    }

    ~ActivityAccess() { close(); }
};

jmethodID method(ActivityAccess& access, const char* name, const char* signature, std::string& error)
{
    jmethodID result = access.env->GetMethodID(access.activity_class, name, signature);
    if (!result || access.env->ExceptionCheck())
    {
        if (access.env->ExceptionCheck()) access.env->ExceptionClear();
        error = std::string("Android video activity method is unavailable: ") + name;
        return nullptr;
    }
    return result;
}

void clear_exception(ActivityAccess& access, const char* operation)
{
    if (!access.env->ExceptionCheck())
        return;
    access.env->ExceptionDescribe();
    access.env->ExceptionClear();
    Msg("! [renderer-vulkan] Android video JNI call failed: %s", operation);
}
}

struct AndroidMediaVideoDecoder::Impl
{
    std::string source_path;
    jint decoder_id{-1};
    uint32_t width{};
    uint32_t height{};
    bool is_playing{};

    ~Impl() { shutdown(); }

    void shutdown()
    {
        if (decoder_id >= 0)
        {
            ActivityAccess access;
            std::string error;
            if (access.open(error))
            {
                if (jmethodID release = method(access, "releaseAndroidVideoDecoder", "(I)V", error))
                {
                    access.env->CallVoidMethod(access.activity, release, decoder_id);
                    clear_exception(access, "releaseAndroidVideoDecoder");
                }
            }
            decoder_id = -1;
        }
        if (!source_path.empty())
        {
            ::unlink(source_path.c_str());
            source_path.clear();
        }
        width = height = 0;
        is_playing = false;
    }

    bool makeSourceFile(const std::vector<uint8_t>& contents, std::string& error)
    {
        char* preference_path = SDL_GetPrefPath("OpenXRay", "video");
        if (!preference_path)
        {
            error = "could not locate Android app storage for video decoding";
            return false;
        }

        std::string temporary_path(preference_path);
        SDL_free(preference_path);
        temporary_path += "video-XXXXXX";
        std::vector<char> path_buffer(temporary_path.begin(), temporary_path.end());
        path_buffer.push_back('\0');
        const int fd = ::mkstemp(path_buffer.data());
        if (fd < 0)
        {
            error = "could not create a temporary file for Android MediaExtractor";
            return false;
        }

        const bool written = WriteAll(fd, contents.data(), contents.size());
        const int close_result = ::close(fd);
        if (!written || close_result != 0)
        {
            ::unlink(path_buffer.data());
            error = "could not prepare the video file for Android MediaExtractor";
            return false;
        }
        source_path = path_buffer.data();
        return true;
    }

    bool load(const char* path, std::string& error)
    {
        shutdown();
        std::vector<uint8_t> contents;
        if (!ReadVideoFile(path, contents, error) || !makeSourceFile(contents, error))
        {
            shutdown();
            return false;
        }
        contents.clear();
        contents.shrink_to_fit();

        ActivityAccess access;
        if (!access.open(error))
        {
            shutdown();
            return false;
        }
        jmethodID create = method(access, "createAndroidVideoDecoder", "(Ljava/lang/String;)[I", error);
        if (!create)
        {
            shutdown();
            return false;
        }
        jstring java_path = access.env->NewStringUTF(source_path.c_str());
        if (!java_path || access.env->ExceptionCheck())
        {
            if (access.env->ExceptionCheck()) access.env->ExceptionClear();
            error = "could not pass the Android video path to MediaExtractor";
            shutdown();
            return false;
        }
        auto dimensions = static_cast<jintArray>(access.env->CallObjectMethod(
            access.activity, create, java_path));
        access.env->DeleteLocalRef(java_path);
        if (access.env->ExceptionCheck())
        {
            clear_exception(access, "createAndroidVideoDecoder");
            error = "Android MediaExtractor could not open the video container";
            shutdown();
            return false;
        }
        if (!dimensions || access.env->GetArrayLength(dimensions) != 3)
        {
            if (dimensions) access.env->DeleteLocalRef(dimensions);
            error = "Android MediaExtractor could not create a video decoder";
            shutdown();
            return false;
        }
        jint values[3]{};
        access.env->GetIntArrayRegion(dimensions, 0, 3, values);
        access.env->DeleteLocalRef(dimensions);
        if (values[0] > 0)
            decoder_id = values[0];
        if (access.env->ExceptionCheck() || values[0] <= 0 || values[1] <= 0 || values[2] <= 0 ||
            uint64_t(values[1]) * uint64_t(values[2]) * 4 > MaxDecodedFrameBytes)
        {
            if (access.env->ExceptionCheck()) access.env->ExceptionClear();
            error = "Android MediaExtractor returned invalid video dimensions";
            shutdown();
            return false;
        }
        width = static_cast<uint32_t>(values[1]);
        height = static_cast<uint32_t>(values[2]);
        error.clear();
        return true;
    }

    void play(bool looped, uint32_t time)
    {
        if (decoder_id < 0)
            return;
        ActivityAccess access;
        std::string error;
        if (!access.open(error))
        {
            is_playing = false;
            return;
        }
        jmethodID play_method = method(access, "playAndroidVideoDecoder", "(IZI)V", error);
        if (!play_method)
        {
            is_playing = false;
            return;
        }
        access.env->CallVoidMethod(access.activity, play_method, decoder_id,
            looped ? JNI_TRUE : JNI_FALSE, static_cast<jint>(time));
        if (access.env->ExceptionCheck())
        {
            clear_exception(access, "playAndroidVideoDecoder");
            is_playing = false;
            return;
        }
        is_playing = true;
    }

    void stop()
    {
        is_playing = false;
        if (decoder_id < 0)
            return;
        ActivityAccess access;
        std::string error;
        if (!access.open(error)) return;
        if (jmethodID stop_method = method(access, "stopAndroidVideoDecoder", "(I)V", error))
        {
            access.env->CallVoidMethod(access.activity, stop_method, decoder_id);
            clear_exception(access, "stopAndroidVideoDecoder");
        }
    }

    bool playing()
    {
        if (decoder_id < 0 || !is_playing)
            return false;
        ActivityAccess access;
        std::string error;
        if (!access.open(error)) return is_playing;
        jmethodID playing_method = method(access, "isAndroidVideoDecoderPlaying", "(I)Z", error);
        if (!playing_method) return is_playing;
        const jboolean result = access.env->CallBooleanMethod(access.activity, playing_method, decoder_id);
        if (access.env->ExceptionCheck())
        {
            clear_exception(access, "isAndroidVideoDecoderPlaying");
            return false;
        }
        is_playing = result == JNI_TRUE;
        return is_playing;
    }

    bool update(uint32_t time, std::vector<uint32_t>& pixels)
    {
        if (decoder_id < 0 || !is_playing)
            return false;
        ActivityAccess access;
        std::string error;
        if (!access.open(error)) return false;
        jmethodID update_method = method(access, "updateAndroidVideoDecoder", "(II)[B", error);
        if (!update_method) return false;
        auto frame = static_cast<jbyteArray>(access.env->CallObjectMethod(
            access.activity, update_method, decoder_id, static_cast<jint>(time)));
        if (access.env->ExceptionCheck())
        {
            clear_exception(access, "updateAndroidVideoDecoder");
            is_playing = false;
            return false;
        }
        if (!frame)
            return false;
        const jsize byte_count = access.env->GetArrayLength(frame);
        const size_t expected = size_t(width) * height * 4;
        if (byte_count < 0 || static_cast<size_t>(byte_count) != expected)
        {
            access.env->DeleteLocalRef(frame);
            Msg("! [renderer-vulkan] Android video decoder returned %d bytes; expected %zu",
                int(byte_count), expected);
            is_playing = false;
            return false;
        }
        pixels.resize(size_t(width) * height);
        access.env->GetByteArrayRegion(frame, 0, byte_count,
            reinterpret_cast<jbyte*>(pixels.data()));
        access.env->DeleteLocalRef(frame);
        if (access.env->ExceptionCheck())
        {
            clear_exception(access, "copy Android video frame");
            is_playing = false;
            return false;
        }
        return true;
    }
};

AndroidMediaVideoDecoder::AndroidMediaVideoDecoder() : impl_(std::make_unique<Impl>()) {}
AndroidMediaVideoDecoder::~AndroidMediaVideoDecoder() = default;

bool AndroidMediaVideoDecoder::load(const char* path, uint32_t& width, uint32_t& height, std::string& error)
{
    if (!path || !impl_->load(path, error))
        return false;
    width = impl_->width;
    height = impl_->height;
    return true;
}

void AndroidMediaVideoDecoder::play(bool looped, uint32_t time) { impl_->play(looped, time); }
void AndroidMediaVideoDecoder::stop() { impl_->stop(); }
bool AndroidMediaVideoDecoder::playing() const { return const_cast<Impl&>(*impl_).playing(); }
bool AndroidMediaVideoDecoder::update(uint32_t time, std::vector<uint32_t>& pixels)
{
    return impl_->update(time, pixels);
}
}

#endif // ANDROID
