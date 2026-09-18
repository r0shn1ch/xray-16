# OpenXRay on Android (armeabi-v7a)

This directory contains the Android 16 / ARMv7 build entry point. The engine
still uses SDL2 as its platform layer; the Android SDL2 package and all native
codec dependencies must therefore be built for the same NDK ABI and supplied
through `CMAKE_PREFIX_PATH`.

The target ABI is `armeabi-v7a`, which is the Android NDK name for 32-bit ARM
with Thumb-2 and Neon. The minimum platform is API 36 (Android 16).

## Configure and build

```sh
ANDROID_NDK_HOME=/path/to/android-ndk \
ANDROID_DEPS_PREFIX=/path/to/android-deps \
SDL2_DIR=/path/to/android-sdl2/lib/cmake/SDL2 \
./android/build-armv7.sh
```

The script forwards `CMAKE_PREFIX_PATH` from `ANDROID_DEPS_PREFIX` and builds
the regular `xr_3da` target. The prefix must contain Android/armeabi-v7a
builds of SDL2, OpenAL Soft, Ogg, Vorbis, Theora, LZO and JPEG. The script does
not download dependencies implicitly.

LuaJIT also generates ARM32 code with host-side `minilua` and `buildvm`. On a
64-bit Linux build host install the 32-bit compiler runtime (for example
`gcc-multilib` and `libc6-dev-i386`). If the host kernel cannot execute i386
ELF files, the CMake integration uses `qemu-i386` when it is available.

## First validation without game data

After the native target is built, run the executable with:

```sh
./android/run-headless-smoke.sh path/to/xr_3da
```

`-headless-smoke` initializes SDL, xrCore, CPU feature probing, the task
scheduler and the platform locator without creating a window or renderer and
without opening any game files. A successful exit is only the native bootstrap
check; it is not yet evidence that a game can render on a particular GPU.

## Android renderer status

The initial Android path requests an OpenGL ES 3.0 context, disables MSAA
texture allocation and ignores desktop polygon modes. The OpenGL renderer still
has desktop shader and framebuffer assumptions, so game-data rendering needs a
separate GLES compatibility pass and a device test before it can be called
complete.
