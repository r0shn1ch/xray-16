# Android ARMv7 APK bring-up

`build-apk-armv7.sh` packages the Android shared target as an SDL2 APK for
`armeabi-v7a`. The debug APK starts with `-renderer-smoke -nogame`, so it
contains the complete native `xr_3da`/`xrRender_GL` engine link but no STALKER
game data. It creates a real OpenXRay `CHW` GLES 3.0 context, compiles an ES
3.0 vertex/fragment shader pair, draws a triangle and validates a framebuffer
readback. The renderer smoke screen stays open until the user closes it.

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

The resulting debug APK is `build/openxray-armv7-debug.apk`. It is a
bring-up artifact, not a playable release: proprietary game data and touch
controls are not bundled. This test proves the Android window/context and
shader path; it does not claim that every original desktop shader or every
game-data render feature is already GLES-compatible.

## Diagnostics on a phone

The native engine log is written to
`/storage/emulated/0/openxray/android.log`. Java lifecycle and exception
records go to `/storage/emulated/0/openxray/activity.log`. If Android refuses
shared-storage access, both writers fall back to the app-specific external
directory so a failure is still diagnosable.

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
adb shell monkey -p org.openxray.stalker 1
adb logcat -d -b all -v threadtime OpenXRay:I DEBUG:E '*:S' > openxray-logcat.txt
adb pull /sdcard/openxray/android.log openxray-engine.log
adb pull /sdcard/openxray/activity.log openxray-activity.log
```

If the native process crashes, preserve the complete `adb logcat -b crash` output
and symbolicate it with the unstripped `armeabi-v7a` library using `ndk-stack`.
