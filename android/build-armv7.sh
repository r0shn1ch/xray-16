#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
ndk_dir=${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}
build_dir=${XRAY_ANDROID_BUILD_DIR:-"$repo_dir/build/android-armv7"}
deps_prefix=${ANDROID_DEPS_PREFIX:-}

if [ -z "$ndk_dir" ] || [ ! -f "$ndk_dir/build/cmake/android.toolchain.cmake" ]; then
    echo "ANDROID_NDK_HOME must point to an installed Android NDK" >&2
    exit 2
fi

set -- "$@"
if [ -n "$deps_prefix" ]; then
    set -- "$@" "-DCMAKE_PREFIX_PATH=$deps_prefix"
fi
if [ -n "${SDL2_DIR:-}" ]; then
    set -- "$@" "-DSDL2_DIR=$SDL2_DIR"
fi

cmake -S "$repo_dir" -B "$build_dir" -G "${CMAKE_GENERATOR:-Ninja}" \
    -DCMAKE_TOOLCHAIN_FILE="$ndk_dir/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=armeabi-v7a \
    -DANDROID_PLATFORM=android-36 \
    -DANDROID_STL=c++_shared \
    -DANDROID_ARM_MODE=thumb \
    -DBUILD_SHARED_LIBS=OFF \
    -DXRAY_USE_LUAJIT=ON \
    -DXRAY_ENABLE_TRACY=OFF \
    -DMEMORY_ALLOCATOR=standard \
    "$@"

cmake --build "$build_dir" --target xr_3da
