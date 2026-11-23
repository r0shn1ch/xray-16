#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
android_abi=${1:-${XRAY_ANDROID_ABI:-arm64-v8a}}
case "$android_abi" in
    arm64-v8a) slang_preset=android-arm64 ;;
    x86_64) slang_preset=android-x86_64 ;;
    *) echo "Slang runtime compilation supports Android arm64-v8a and x86_64" >&2; exit 2 ;;
esac

ndk_dir=${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}
if [ -z "$ndk_dir" ] || [ ! -f "$ndk_dir/build/cmake/android.toolchain.cmake" ]; then
    echo "ANDROID_NDK_HOME must point to an installed Android NDK" >&2
    exit 2
fi
export ANDROID_NDK_HOME="$ndk_dir"

cache_root=${XRAY_SLANG_BUILD_ROOT:-"${XDG_CACHE_HOME:-/tmp}/openxray-slang"}
version=v2026.22
source_dir=${XRAY_SLANG_SOURCE_DIR:-"$cache_root/source/slang-$version"}
install_root=${XRAY_SLANG_INSTALL_ROOT:-"$cache_root/install/$version/$android_abi"}
jobs=${XRAY_SLANG_BUILD_JOBS:-8}

if [ ! -f "$source_dir/CMakePresets.json" ]; then
    mkdir -p "$(dirname -- "$source_dir")"
    git clone --depth 1 --branch "$version" --recurse-submodules --shallow-submodules \
        https://github.com/shader-slang/slang.git "$source_dir" >&2
fi

if [ ! -x "$source_dir/generators/bin/slang-lookup-generator" ]; then
    (
        cd "$source_dir"
        cmake --workflow --preset generators >&2
        cmake --install build --prefix generators --component generators --config Release >&2
    )
fi

binary_dir="$source_dir/build-android-$android_abi"
if [ ! -f "$install_root/lib/libslang-compiler.so" ]; then
    (
        cd "$source_dir"
        cmake --preset "$slang_preset" \
            "-DSLANG_GENERATORS_PATH=$source_dir/generators/bin" \
            -DSLANG_ENABLE_SLANGC=OFF \
            -DSLANG_ENABLE_SLANGD=OFF \
            -DSLANG_ENABLE_SLANGI=OFF \
            -DSLANG_ENABLE_REPLAYER=OFF \
            -DSLANG_ENABLE_DXIL=OFF \
            -DSLANG_ENABLE_EXAMPLES=OFF \
            -DSLANG_ENABLE_TESTS=OFF \
            -DSLANG_ENABLE_GFX=OFF \
            -DSLANG_ENABLE_SLANG_RHI=OFF \
            -DSLANG_ENABLE_SLANGRT=OFF \
            -DSLANG_ENABLE_SLANG_PROXY=OFF \
            -DSLANG_SLANG_LLVM_FLAVOR=DISABLE >&2
        cmake --build --preset "$slang_preset-release" --target slang --parallel "$jobs" >&2
    )
fi

if [ ! -f "$binary_dir/Release/lib/libslang-compiler.so" ] || [ ! -d "$binary_dir/Release/include" ]; then
    echo "Slang Android compiler artifacts are missing from $binary_dir" >&2
    exit 1
fi

mkdir -p "$install_root/include" "$install_root/lib/cmake/slang"
cp -R "$binary_dir/Release/include/." "$install_root/include/"
cp "$binary_dir/Release/lib/libslang-compiler.so" "$install_root/lib/libslang-compiler.so"
cat > "$install_root/lib/cmake/slang/slangConfig.cmake" <<'CMAKE'
if(NOT TARGET slang::slang)
    add_library(slang::slang SHARED IMPORTED GLOBAL)
    set_target_properties(slang::slang PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../../../lib/libslang-compiler.so"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include"
    )
endif()
set(slang_FOUND TRUE)
CMAKE

if [ ! -f "$install_root/lib/cmake/slang/slangConfig.cmake" ] || [ ! -f "$install_root/lib/libslang-compiler.so" ]; then
    echo "Slang Android package was not installed correctly at $install_root" >&2
    exit 1
fi

printf '%s\n' "$install_root"
