# OpenXRay Android port

This directory contains the experimental Android build maintained by this
fork. It produces an SDL2 launcher APK and a native OpenXRay engine for the
`armeabi-v7a` ABI.

## Current scope and limitations

- Minimum Android version: Android 8.0 / API 26.
- Target and compile SDK: API 36.
- Packaged ABI: `armeabi-v7a` only. ARM64-only Android systems cannot install
  it; an ARM64 device must retain 32-bit application support.
- The build targets ARMv7-A with NEON. ReleaseMasterGold uses ARM instruction
  mode, `-O3` and ThinLTO by default.
- Gameplay rendering uses OpenGL ES 3.1+. At least four draw buffers and four
  color attachments are required.
- The Vulkan setting is a VK0 capability/presentation probe followed by an
  explicit GLES gameplay fallback. `xrRenderVK` gameplay is not implemented.
- Call of Pripyat is the intended game profile. Launcher entries for SoC and CS
  pass existing compatibility flags; they are not a promise of additional
  game support.
- Proprietary game data is never included in the APK. The selected PC game
  directory is treated as a read-only resource source.

The renderer uses API feature/version checks and contains no GPU vendor/model
allowlist. Devices without usable BC/S3TC support decode affected textures in
memory, which costs loading time, memory and bandwidth. Large levels and mods
can also hit the ARMv7 process address-space limit regardless of physical RAM.

## Build with the prepared kit

The recommended build uses an extracted kit containing the pinned NDK, SDK,
ARMv7 dependencies, SDL2 Android project, Gradle and its offline cache:

```sh
export XRAY_ANDROID_KIT_ROOT=/absolute/path/to/openxray-android-build-kit
./android/build-harness.sh --apk
```

`build-harness.sh --native` builds only the native engine target. The APK is
written to:

```text
build/openxray-armv7-launcher-v<android/PORT_VERSION>-debug.apk
```

The checked-out source must already contain the current Android implementation.
Files under `android/patches/` are retained for older port history and are not
a supported way to upgrade an arbitrary upstream checkout to the current
version.

## Build with an existing Android toolchain

Set these paths before running the APK script:

```sh
export ANDROID_NDK_HOME=/path/to/android-ndk
export ANDROID_SDK_ROOT=/path/to/android-sdk
export ANDROID_DEPS_PREFIX=/path/to/android-deps-armv7
export SDL2_ANDROID_HOME=/path/to/SDL
export GRADLE_BIN=/path/to/gradle/bin/gradle  # optional when SDL gradlew works
./android/build-apk-armv7.sh
```

`ANDROID_DEPS_PREFIX` must contain ARMv7 builds of SDL2, OpenAL Soft, JPEG,
Ogg, Vorbis, Theora and LZO. Recursive source submodules must also be present;
`android/prepare-source.sh` initializes them for a normal git checkout and
validates the required source trees before CMake starts.

Release defaults favor runtime performance. For diagnostic builds only, LTO
can be disabled with `XRAY_ANDROID_ENABLE_LTO=OFF`; compact Thumb mode can be
requested with `XRAY_ANDROID_ARM_MODE=thumb`.

The packaging script strips the staged native libraries, verifies that the APK
contains that exact engine, checks the ABI and version, applies 16 KiB ZIP
alignment and signs with the Gradle debug key. This is a debug-signed test APK,
not a Play Store release.

## Install and run

```sh
adb install -r build/openxray-armv7-launcher-v0.9.12-debug.apk
adb shell am start -n org.openxray.stalker/org.openxray.app.LauncherActivity
```

On Android 11 or newer, grant **All files access** when the PC installation is
stored in shared storage. On Android 8–10, grant the requested storage
permissions. The launcher itself is portrait; the separate engine activity is
landscape.

Choose Call of Pripyat and point the launcher at the installation root that
contains `fsgame.ltx` and the normal resource directories/archives. The
launcher does not download, repair, copy or rewrite the installation.

Launcher settings are passed as explicit engine arguments:

- **Renderer:** Auto and OpenGL ES both use GLES. Vulkan first runs the Vulkan
  probe and then uses GLES for gameplay.
- **Graphics:** Auto selects Minimum. Minimum also disables sun/detail/TSM
  shadows and water reflections. Low through Extreme remain selectable.
- **3D resolution:** Auto uses the native aspect ratio with a width no greater
  than 1280; lower standard modes and native resolution are selectable. The
  Android window remains at the physical landscape size and the final image is
  scaled to it.
- **FPS:** enables the engine counter in opaque red at the top center.
- **Touch controls:** optional overlay including Escape; free screen space
  controls the mouse.

If the engine process is still alive, the launch button becomes **Return to
running game** and a separate stop button can terminate a stuck engine after
confirmation.

## Renderer and performance diagnostics

The GLES smoke test creates the same OpenXRay GLES 3.1 context path, compiles
and links an ES 3.1 shader pair, draws a triangle and validates a pixel
readback. The Vulkan test creates and presents a swapchain image, records
capabilities, and then runs the GLES smoke fallback. Neither test proves that
all game/mod shaders work on a physical device.

During gameplay, `[frame-trace]` records smoothed FPS, frame average/maximum,
update/render/task-wait time, present/swap time, draw calls, polygons, internal
resolution and drawable size every five seconds. This tracing is intentionally
lighter than the full `rs_stats` overlay.

The diagnostics page reads only bounded log tails on a background executor:

- `/storage/emulated/0/openxray/android.log`
- `/storage/emulated/0/openxray/activity.log`
- app-specific external/internal fallbacks when shared storage is unavailable

For a report, reproduce at least 30 seconds of gameplay and collect both logs
plus crash logcat when applicable:

```sh
adb logcat -c
adb shell am force-stop org.openxray.stalker
adb shell am start -n org.openxray.stalker/org.openxray.app.LauncherActivity
adb logcat -d -b all -v threadtime OpenXRay:I DEBUG:E '*:S' > openxray-logcat.txt
adb logcat -d -b crash -v threadtime > openxray-crash.txt
adb pull /sdcard/openxray/android.log openxray-engine.log
adb pull /sdcard/openxray/activity.log openxray-activity.log
```

The unstripped `bin/armv7-a/ReleaseMasterGold/libmain.so` from the same commit
is required for useful native symbolication.

For implementation constraints and Vulkan status, see
[PORT_AUDIT.md](PORT_AUDIT.md) and
[VULKAN_RENDERER_PLAN.md](VULKAN_RENDERER_PLAN.md). Launcher-specific behavior
is summarized in [apk/README.md](apk/README.md).
