# OpenXRay on Android (armeabi-v7a)

This directory contains the Android 16 / ARMv7 build entry point. The engine
still uses SDL2 as its platform layer; the Android SDL2 package and all native
codec dependencies must therefore be built for the same NDK ABI and supplied
through `CMAKE_PREFIX_PATH`.

The target ABI is `armeabi-v7a`, which is the Android NDK name for 32-bit ARM
with Thumb-2 and Neon. The APK currently supports API 26 and targets/compiles
against API 36 (Android 16).

## Configure and build

The recommended entry point is `build-harness.sh`. It applies the Android
patchset automatically (and skips it when the source already contains the
patches), loads a prepared build kit when `XRAY_ANDROID_KIT_ROOT` is set, and
then builds either the APK or the native target:

```sh
XRAY_ANDROID_KIT_ROOT=/path/to/openxray-android-build-kit-v0.5.0 \
./android/build-harness.sh --apk
```

For a checkout that already has the Android toolchain exported, the same
command works without `XRAY_ANDROID_KIT_ROOT`. Use `--native` for the
headless/native target. No patch command is needed before either build.

```sh
ANDROID_NDK_HOME=/path/to/android-ndk \
ANDROID_DEPS_PREFIX=/path/to/android-deps \
SDL2_DIR=/path/to/android-sdl2/lib/cmake/SDL2 \
./android/build-armv7.sh
```

The script forwards `CMAKE_PREFIX_PATH` from `ANDROID_DEPS_PREFIX` and builds
the Android `libmain.so` engine target used by the APK. The prefix must contain
Android/armeabi-v7a builds of SDL2, OpenAL Soft, Ogg, Vorbis, Theora, LZO and
JPEG. The script does not download dependencies implicitly.

`apply-patches.sh` is idempotent. It verifies the null-safe Android core
bootstrap, crash logging, launcher game-root diagnostics and LuaJIT host linker
support before building; on an older clean checkout it applies the numbered
patches in `android/patches/` automatically.

To preserve the installed NDK, SDK, native dependencies, SDL2 Android project
and pinned Gradle distribution for later builds, create the build kit once:

```sh
ANDROID_NDK_HOME=/path/to/android-ndk-r30 \
ANDROID_SDK_ROOT=/path/to/android-sdk \
ANDROID_DEPS_PREFIX=/path/to/android-deps-armv7 \
SDL2_ANDROID_HOME=/path/to/SDL \
GRADLE_BIN=/path/to/gradle-8.1.1/bin/gradle \
./android/create-build-kit.sh
```

The resulting `.tar.zst` contains the harness, patchset and complete pinned
toolchain. Extract it once, point `XRAY_ANDROID_KIT_ROOT` at the extracted
directory, and reuse `build-harness.sh` for subsequent builds.

LuaJIT also generates ARM32 code with host-side `minilua` and `buildvm`. On a
64-bit Linux build host install the 32-bit compiler runtime (for example
`gcc-multilib` and `libc6-dev-i386`). If the host kernel cannot execute i386
ELF files, the CMake integration uses `qemu-i386` when it is available.
The Android build scripts now select the NDK's `i686-linux-android26-clang`
and the build kit's `qemu-i386-static` automatically. Explicit
`LUAJIT_HOST_C_COMPILER` and `LUAJIT_HOST_EXECUTABLE_PREFIX` values still
override those defaults; the host compiler must produce a statically linked
32-bit helper.

## First validation without game data

For a separately built host executable, run the headless smoke check with:

```sh
./android/run-headless-smoke.sh path/to/xr_3da
```

`-headless-smoke` initializes SDL, xrCore, CPU feature probing, the task
scheduler and the platform locator without creating a window or renderer and
without opening any game files. A successful exit is only the native bootstrap
check; it is not yet evidence that a game can render on a particular GPU.

The APK opens a launcher and its **Проверить GLES** action uses the separate
`-renderer-smoke` mode. It creates the Android SDL window and OpenXRay `CHW`,
loads GLES through the same GLAD-backed renderer library, compiles a minimal ES
3.0 shader pair and verifies a pixel readback. This keeps the test independent
of proprietary game data while exercising the actual Android window/context
and shader compiler path. The native activity runs in a separate `:engine`
process, so the launcher can remain visible after a native crash.

## Android renderer status

The Android path requests an OpenGL ES 3.1 context, disables MSAA texture
allocation, keeps SDL's default EGL framebuffer active and ignores desktop
polygon modes. The smoke mode is built and
statically validated in the ARMv7 APK; its runtime device result is deliberately
reported separately because this sandbox has no ARM Android runtime. Game-data
rendering still needs a separate GLES compatibility pass for desktop-only
framebuffer/shader assumptions and a device test with the original game files
before it can be called complete.

The APK displays a Toast after the renderer readback: successful initialization
keeps the smoke window open, while failure shows the error, waits briefly and
exits. The launcher also shows the current log tail and reports whether the
engine process finished successfully or crashed. The native engine log targets
`/storage/emulated/0/openxray/android.log` first and falls back to SDL's
app-specific external/internal directory when Android storage policy denies the
public path.
