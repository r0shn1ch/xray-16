#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=${1:-$(CDPATH= cd -- "$script_dir/.." && pwd)}
patch_dir="$repo_dir/android/patches"

if grep -Fq 'pw->pw_gecos && pw->pw_gecos[0]' "$repo_dir/src/xrCore/xrCore.cpp" \
    && grep -Fq 'STALKER folder not found' "$repo_dir/android/apk/app/src/main/java/org/openxray/app/LauncherActivity.java" \
    && grep -Fq 'android_native_crash_handler' "$repo_dir/src/xrEngine/x_ray.cpp" \
    && grep -Fq 'LUAJIT_HOST_EXTRA_LDFLAGS' "$repo_dir/Externals/LuaJIT-proj/CMakeLists.txt"; then
    echo "Android patchset: already present"
    exit 0
fi

if [ ! -d "$patch_dir" ]; then
    echo "Android patchset is missing: $patch_dir" >&2
    exit 2
fi

if ! git -C "$repo_dir" diff --quiet || ! git -C "$repo_dir" diff --cached --quiet; then
    echo "Android patchset is incomplete, but the repository has local changes; commit or stash them before applying patches" >&2
    exit 2
fi

found_patch=false
for patch in "$patch_dir"/*.patch; do
    [ -f "$patch" ] || continue
    found_patch=true
    if git -C "$repo_dir" apply --check --3way "$patch" >/dev/null 2>&1; then
        git -C "$repo_dir" apply --3way "$patch"
        echo "Android patchset: applied $(basename "$patch")"
    elif git -C "$repo_dir" apply --reverse --check "$patch" >/dev/null 2>&1; then
        echo "Android patchset: $(basename "$patch") is already applied"
    else
        echo "Android patchset cannot be applied cleanly: $patch" >&2
        exit 2
    fi
done

if [ "$found_patch" = false ]; then
    echo "Android patchset directory is empty: $patch_dir" >&2
    exit 2
fi
