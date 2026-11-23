#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
XRAY_ANDROID_ABI=x86_64 exec "$script_dir/build-android-native.sh" "$@"
