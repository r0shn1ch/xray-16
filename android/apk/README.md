# Android ARMv7 APK bring-up

`build-apk-armv7.sh` packages the Android shared target as an SDL2 APK for
`armeabi-v7a`. The APK opens an OpenXRay launcher first. It contains the
complete native `xr_3da`/`xrRender_GL` engine link but no proprietary STALKER
game data. The launcher can run a renderer smoke test, choose a game root and
start the native engine in a separate process so a native crash does not take
the launcher or its visible log with it.

The renderer smoke path creates a real OpenXRay `CHW` GLES 3.1 context,
compiles an ES 3.1 vertex/fragment shader pair, draws a triangle and validates
a framebuffer readback. A successful or failed load is shown in a toast and in
the launcher's status line.

The build uses the SDL2 `android-project` Java activity as an external source
dependency. Set `SDL2_ANDROID_HOME` to an SDL2 source tree (the same SDL2
version used for the native dependency prefix), and set `ANDROID_SDK_ROOT`,
`ANDROID_NDK_HOME` and `ANDROID_DEPS_PREFIX` as for `build-armv7.sh`.

Example:

```sh
ANDROID_SDK_ROOT=/path/to/android-sdk \
ANDROID_NDK_HOME=/path/to/android-ndk \
ANDROID_DEPS_PREFIX=/path/to/android-deps-armv7 \
SDL2_ANDROID_HOME=/path/to/SDL \
./android/build-apk-armv7.sh
```

The APK build disables LTO by default so the ARMv7 shared-library link stays
bounded on ordinary build hosts. Set `XRAY_ANDROID_ENABLE_LTO=ON` when a long
LTO link is desired.

The APK also carries the OpenXRay engine `res/gamedata` tree. The launcher
extracts this engine-only data into its private app storage and exposes it to
the native virtual file system as a read-only fallback overlay. It never copies,
rewrites or creates files in the selected game directory. The selected path is
passed directly to the native engine, together with the user's own
`fsgame.ltx`; the engine performs the same resource discovery and validation as
on desktop.

Put the original Call of Pripyat installation in a user-accessible directory
without restructuring or editing it. Its normal top-level resource
directories include `levels`, `localization`, `mp`, `patches` and `resources`.
The launcher does not reject an incomplete folder: missing `fsgame.ltx` or
resources are reported by the engine in the log.

The resulting debug APK is
`build/openxray-armv7-launcher-v0.8.0-debug.apk`. Proprietary game data is not
bundled. The APK is a device-test candidate: a successful build and offline
shader validation do not substitute for running the exact game/mod and GPU.

## Launcher and diagnostics on a phone

1. Remove the old pre-launcher APK once before installing version 0.8.0. The
   old package registered `XRayActivity` itself as the launcher, so a pinned
   old icon can bypass the launcher entirely:

   ```sh
   adb uninstall org.openxray.stalker
   adb install -r build/openxray-armv7-launcher-v0.8.0-debug.apk
   adb shell am start -n org.openxray.stalker/org.openxray.app.LauncherActivity
   ```

2. On Android 11 and newer the launcher immediately shows an explanation and
   opens the system page for **Allow access to manage all files** on first
   start. This is a special Android settings grant, not a normal install
   permission dialog. If it was dismissed, press **Доступ к памяти**.
   On Android 6–10 it requests both `READ_EXTERNAL_STORAGE` and
   `WRITE_EXTERNAL_STORAGE` through the normal system permission dialog.
3. On the **Игра** page first select Shadow of Chernobyl, Clear Sky, Call of
   Pripyat or automatic detection, then press **Выбрать папку** and select the STALKER
   installation directory, or enter
   its direct path manually (for example `/storage/emulated/0/STALKER`). Keep
   the original Call of Pripyat directory layout and files unchanged.
4. Each game profile remembers its own installation path, so switching between
   all three games does not require retyping folders. On **Параметры**, optional
   arguments are tokenized without a shell and cannot replace the selected root
   or profile. The reset action clears launcher preferences only and explicitly
   leaves game files, mods, `fsgame.ltx` and `user.ltx` untouched.
5. Press **Проверить GLES без игровых файлов** to test the Android renderer,
   or **Запустить игру** to run the selected installation. The launcher passes
   that directory to OpenXRay as-is; all missing or unreadable resources are
   logged by the engine.

The launcher separates game selection, settings and diagnostics into pages.
It continuously displays the tail of the engine and activity logs, can share
them through Android, and changes the launch action to **Вернуться в
запущенную игру** while the engine process exists. **Очистить** starts the next
test with an empty log.

The native engine log is written to
`/storage/emulated/0/openxray/android.log`. Java lifecycle and exception
records go to `/storage/emulated/0/openxray/activity.log`. If Android refuses
shared-storage access, both writers fall back to the app-specific external or
internal directory; the launcher reads all of these locations.

Android 11 and newer may require enabling **All files access** for this debug
APK. With USB debugging enabled, grant it before launch:

```sh
adb shell appops set --uid org.openxray.stalker MANAGE_EXTERNAL_STORAGE allow
```

The renderer smoke reports its final status with an Android Toast: success
keeps the smoke window open; failure displays the error, waits briefly, and
then exits. Collect both file and logcat diagnostics with:

```sh
adb logcat -c
adb shell am force-stop org.openxray.stalker
adb shell am start -n org.openxray.stalker/org.openxray.app.LauncherActivity
adb logcat -d -b all -v threadtime OpenXRay:I DEBUG:E '*:S' > openxray-logcat.txt
adb pull /sdcard/openxray/android.log openxray-engine.log
adb pull /sdcard/openxray/activity.log openxray-activity.log
```

If the native process crashes, preserve the complete `adb logcat -b crash` output
and symbolicate it with the unstripped `armeabi-v7a` library using `ndk-stack`.
