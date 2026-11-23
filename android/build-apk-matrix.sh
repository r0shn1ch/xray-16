#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/.." && pwd)
matrix_root=${XRAY_ANDROID_MATRIX_DIR:-"$repo_dir/build/android-matrix"}
# Both configurations use the same checkout; each APK contains both ABIs.
export XRAY_ANDROID_ARM64_ONLY=OFF
SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH:-$(git -C "$repo_dir" show -s --format=%ct HEAD)}
export SOURCE_DATE_EPOCH
for config in ReleaseMasterGold Debug; do
    XRAY_ANDROID_NATIVE_CONFIG="$config" \
    XRAY_ANDROID_APK_BUILD_DIR="$matrix_root/$config" \
        "$script_dir/build-apk-dual.sh" "$@"
    port_version=$(sed -n '1p' "$script_dir/PORT_VERSION")
    cp "$repo_dir/build/openxray-universal-launcher-v$port_version-debug.apk" \
        "$matrix_root/openxray-universal-v$port_version-$config.apk"
done
