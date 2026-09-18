# Android APK bring-up

`build-apk-armv7.sh` packages the Android shared target as an SDL2 APK for
`armeabi-v7a`. The first APK starts with `-headless-smoke`, so it contains no
STALKER game data and exits after validating SDL/JNI, the native loader and the
OpenXRay filesystem bootstrap.

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
bring-up artifact, not a playable release: game data, touch controls and the
remaining GLES renderer compatibility work are intentionally out of scope for
this first launch check.
