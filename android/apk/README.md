# Android launcher APK

`build-apk-armv7.sh` packages the ARMv7 OpenXRay engine, SDL2 activity and
launcher into a debug APK. No proprietary S.T.A.L.K.E.R. resources are
included. Toolchain setup and complete build instructions are in
[../README.md](../README.md).

Build and install:

```sh
export XRAY_ANDROID_KIT_ROOT=/absolute/path/to/openxray-android-build-kit
./android/build-harness.sh --apk
adb install -r build/openxray-armv7-launcher-v0.9.12-debug.apk
```

The generated filename follows `android/PORT_VERSION`. The current application
ID is `org.openxray.stalker`; the main activity is
`org.openxray.app.LauncherActivity`.

## Launcher workflow

1. Grant storage access. Android 11+ uses the system **All files access** page;
   Android 8–10 uses runtime storage permissions.
2. Select Call of Pripyat and its installation root. The folder should contain
   `fsgame.ltx` and the original resource directories/archives. SoC and CS
   entries only select existing engine compatibility flags and remain subject
   to upstream game support.
3. Configure renderer, graphics preset, internal 3D resolution, controls and
   FPS display under **Settings**.
4. Optionally run the GLES smoke test or Vulkan probe without game files.
5. Start the game. The launcher process stays separate from the native engine
   process so logs remain accessible after a native crash.

The launcher is fixed to portrait and the engine activity to landscape. Auto
renderer and explicit OpenGL ES both use the GLES gameplay backend. Selecting
Vulkan runs the Vulkan surface/device/swapchain probe and then explicitly uses
GLES; it does not enable a native Vulkan gameplay renderer.

Graphics choices are applied after `user.ltx` for that engine session without
rewriting the file. Auto graphics maps to Minimum; the completed Android
Minimum profile disables sun/detail/TSM shadows and water reflections. Auto
resolution preserves the physical display aspect ratio and caps internal width
at 1280 unless the display is smaller. The physical Android surface remains
native and the final frame is scaled to it.

The optional engine FPS counter is opaque red at the top center. Touch
controls include Escape as well as movement, fire, interaction and inventory
actions.

## Running engine controls

While the `:engine` process exists, **Start game** changes to **Return to
running game**. The launcher asks the existing engine activity to return to the
foreground instead of starting a second engine. The adjacent stop button
force-terminates the engine process after confirmation and is intended for a
stuck load or renderer.

These controls depend on Android still reporting the engine process as alive.
They cannot restore an engine that has already crashed or been killed by the
system.

## Logs

The diagnostics page polls bounded 32 KiB tails from each engine/activity log
on a background executor, so it does not load entire growing log files on the
UI thread. Sharing can include a larger bounded tail.

Primary paths:

```text
/storage/emulated/0/openxray/android.log
/storage/emulated/0/openxray/activity.log
```

App-specific external and internal paths are used as fallbacks when shared
storage is unavailable. **Clear** removes these diagnostic files; it does not
touch game files, saves, mods, `fsgame.ltx` or `user.ltx`.

For performance reports, enable FPS and include at least 30 seconds of
gameplay. `[frame-trace]` reports update/render/wait, present/swap, draw calls,
polygons and both internal and physical resolutions every five seconds.

Useful collection commands:

```sh
adb logcat -c
adb shell am force-stop org.openxray.stalker
adb shell am start -n org.openxray.stalker/org.openxray.app.LauncherActivity
adb logcat -d -b all -v threadtime OpenXRay:I DEBUG:E '*:S' > openxray-logcat.txt
adb logcat -d -b crash -v threadtime > openxray-crash.txt
adb pull /sdcard/openxray/android.log openxray-engine.log
adb pull /sdcard/openxray/activity.log openxray-activity.log
```

Native crashes must be symbolicated against the unstripped `libmain.so` built
from the same commit as the APK.
