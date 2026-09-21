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
XRAY_ANDROID_KIT_ROOT=/path/to/openxray-android-build-kit-v0.9.0 \
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

For a normal git checkout the build prepares recursive submodules itself and
then verifies the LuaJIT, GLI, ImGui, luabind and xrLuaFix source sentinels.
The complete source archive already contains those trees and can therefore be
built offline with the kit. An incomplete source export now fails before CMake
with a direct diagnostic instead of the misleading "unsupported LuaJIT target"
error.

`apply-patches.sh` is idempotent. It verifies the null-safe Android core
bootstrap, crash logging, launcher game-root diagnostics and LuaJIT host linker
support before building; on an older clean checkout it applies the numbered
patches in `android/patches/` automatically.

The standalone harness can also patch a pristine `origin/dev` checkout without
being copied into that checkout first. Keep the extracted harness outside the
source tree and pass the source path explicitly:

```sh
/path/to/openxray-android-build-harness-v0.9.0/android/apply-patches.sh \
  /path/to/clean/xray-16
```

The script reads patches next to itself by default. Set
`XRAY_ANDROID_PATCH_DIR` only when the patch directory is stored elsewhere.

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

The resulting `.tar.zst` contains the harness, patchset, complete pinned
toolchain and the Gradle dependency cache from a successful APK build. Extract
it once, point `XRAY_ANDROID_KIT_ROOT` at the extracted directory, and reuse
`build-harness.sh` for subsequent builds. Builds made through this kit enable
Gradle offline mode, so they do not depend on Maven availability or another
machine's home-directory cache.

Release archives are created by the same checked-in scripts:

```sh
./android/create-harness-archive.sh
./android/create-source-archive.sh
```

The small harness archive contains the patchset and validator. The source
archive copies only files tracked by the main repository and every recursive
submodule, so it is complete for offline builds but excludes `.git`, local
credentials, caches and build products.

The APK build performs a final 16 KiB `zipalign` pass and then signs the
aligned package with the standard Gradle debug key. `ANDROID_DEBUG_KEYSTORE`,
`ANDROID_DEBUG_KEYSTORE_PASS`, `ANDROID_DEBUG_KEY_PASS` and
`ANDROID_DEBUG_KEY_ALIAS` can override that key when a build host uses a
non-default debug setup.

The same kit contains the NDK shader compiler used by the focused GLES
regression check. From the source root run:

```sh
python3 utils/validate-glsl-es.py \
  --ndk "$XRAY_ANDROID_KIT_ROOT/toolchain/android-ndk-r30"
```

The default manifest compiles the startup shader stages that failed in the
Adreno log and checks their vertex/fragment interfaces after applying the same
Android source translation as the engine.

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

The Android path requests an OpenGL ES 3.1 or newer context and keeps the
deferred renderer on an engine-owned framebuffer. `CHW::Present()` explicitly
copies its final color attachment into SDL's EGL framebuffer before the swap.
The GLES source layer assigns MRT output locations, normalizes stage varyings
and makes the desktop shader expressions explicit at runtime. The tracked
`res/gamedata/shaders/gl` directory is identical to upstream; neither original
game files nor mod files are patched on disk. Compatible shader overrides from
a game or mod pass through the same engine-side translation; line matching is
insensitive to formatting whitespace. Texture upload
uses GLES component swizzles and decodes unsupported desktop DDS compression
in memory without rewriting the source DDS file.

The OpenGL renderer in upstream OpenXRay still disables its unfinished MSAA
mode globally. The Android code does not add another feature cut: multisample
texture allocation and resolve have real GLES implementations for the point
where the upstream OpenGL option is enabled.

The ARMv7 build and the startup shaders implicated by the supplied Adreno log
are statically validated as part of this port. That validation is not a claim
that a physical device test has passed; use the generated APK and attach the
new engine/activity logs for any remaining GPU- or data-specific issue.

The audit of prior port changes is in [PORT_AUDIT.md](PORT_AUDIT.md). The
cross-platform Vulkan requirements, upstream prototypes that were rejected,
and implementation gates are in [VULKAN_RENDERER_PLAN.md](VULKAN_RENDERER_PLAN.md).

The APK displays a Toast after the renderer readback: successful initialization
keeps the smoke window open, while failure shows the error, waits briefly and
exits. The launcher also shows the current log tail and reports whether the
engine process finished successfully or crashed. The native engine log targets
`/storage/emulated/0/openxray/android.log` first and falls back to SDL's
app-specific external/internal directory when Android storage policy denies the
public path.
