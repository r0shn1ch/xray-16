#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
ndk_dir=${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}
sdk_dir=${ANDROID_SDK_ROOT:-${ANDROID_HOME:-}}
deps_prefix=${ANDROID_DEPS_PREFIX:-}
sdl_dir=${SDL2_ANDROID_HOME:-}
gradle_bin=${GRADLE_BIN:-}
build_dir=${XRAY_ANDROID_APK_BUILD_DIR:-"$repo_dir/build/android-apk-armv7"}

if [ -z "$ndk_dir" ] || [ ! -f "$ndk_dir/build/cmake/android.toolchain.cmake" ]; then
    echo "ANDROID_NDK_HOME must point to an installed Android NDK" >&2
    exit 2
fi
if [ -z "$sdk_dir" ] || [ ! -x "$sdk_dir/platform-tools/adb" ]; then
    echo "ANDROID_SDK_ROOT must point to an installed Android SDK" >&2
    exit 2
fi
if [ -z "$deps_prefix" ] || [ ! -f "$deps_prefix/lib/cmake/SDL2/SDL2Config.cmake" ]; then
    echo "ANDROID_DEPS_PREFIX must contain the Android/armeabi-v7a dependency prefix" >&2
    exit 2
fi
if [ -z "$sdl_dir" ] || [ ! -f "$sdl_dir/android-project/gradlew" ]; then
    echo "SDL2_ANDROID_HOME must point to an SDL2 source tree with android-project" >&2
    exit 2
fi

native_build_dir="$build_dir/native"
ANDROID_NDK_HOME="$ndk_dir" \
ANDROID_DEPS_PREFIX="$deps_prefix" \
XRAY_ANDROID_BUILD_DIR="$native_build_dir" \
XRAY_ANDROID_SHARED=ON \
"$script_dir/build-armv7.sh" "$@"

native_lib="$repo_dir/bin/armv7-a/ReleaseMasterGold/libmain.so"
if [ ! -f "$native_lib" ]; then
    echo "native APK library was not produced: $native_lib" >&2
    exit 1
fi

project_dir="$build_dir/gradle-project"
mkdir -p "$project_dir"
cp -R "$sdl_dir/android-project/." "$project_dir/"
cp "$repo_dir/android/apk/app/build.gradle" "$project_dir/app/build.gradle"
cp "$repo_dir/android/apk/app/src/main/AndroidManifest.xml" "$project_dir/app/src/main/AndroidManifest.xml"
mkdir -p "$project_dir/app/src/main/java/org/openxray/app"
cp "$repo_dir/android/apk/app/src/main/java/org/openxray/app/XRayActivity.java" \
    "$project_dir/app/src/main/java/org/openxray/app/XRayActivity.java"
mkdir -p "$project_dir/app/src/main/res/values"
cp "$repo_dir/android/apk/app/src/main/res/values/strings.xml" "$project_dir/app/src/main/res/values/strings.xml"
cp "$repo_dir/android/apk/app/src/main/res/values/styles.xml" "$project_dir/app/src/main/res/values/styles.xml"

native_lib_dir="$project_dir/app/src/main/jniLibs/armeabi-v7a"
mkdir -p "$native_lib_dir"
cp "$native_lib" "$native_lib_dir/libmain.so"
cp "$deps_prefix/lib/libopenal.so" "$native_lib_dir/libopenal.so"
cp "$ndk_dir/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/arm-linux-androideabi/libc++_shared.so" \
    "$native_lib_dir/libc++_shared.so"

chmod +x "$project_dir/gradlew"
(
    cd "$project_dir"
    if [ -n "$gradle_bin" ]; then
        "$gradle_bin" --no-daemon clean assembleDebug
    else
        ./gradlew --no-daemon clean assembleDebug
    fi
)

apk="$project_dir/app/build/outputs/apk/debug/app-debug.apk"
mkdir -p "$repo_dir/build"
cp "$apk" "$repo_dir/build/openxray-armv7-debug.apk"
printf '%s\n' "$repo_dir/build/openxray-armv7-debug.apk"
