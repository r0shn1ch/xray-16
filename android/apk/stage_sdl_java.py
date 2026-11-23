#!/usr/bin/env python3
"""Stage the SDL Java classes matching the SDL installed for every APK ABI."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]


def native_version(header):
    text = header.read_text(encoding="utf-8-sig")
    values = []
    for field in ("MAJOR_VERSION", "MINOR_VERSION", "PATCHLEVEL"):
        match = re.search(r"^\s*#\s*define\s+SDL_" + field + r"\s+(\d+)\b", text, re.M)
        if not match:
            raise ValueError(f"Missing SDL_{field} in {header}")
        values.append(int(match[1]))
    return tuple(values)


def java_version(activity):
    text = activity.read_text(encoding="utf-8-sig")
    values = []
    for field in ("MAJOR", "MINOR", "MICRO"):
        match = re.search(r"\bSDL_" + field + r"_VERSION\s*=\s*(\d+)\s*;", text)
        if not match:
            raise ValueError(f"Missing SDL_{field}_VERSION in {activity}")
        values.append(int(match[1]))
    return tuple(values)


def host_path(value):
    # The same native CMake cache can be consumed by Windows and WSL packagers.
    value = str(value).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", value):
        value = "/mnt/" + value[0].lower() + value[2:]
    return Path(value)


def stage(project, headers, sources=(), builds=(), repo=ROOT):
    versions = {native_version(header) for header in headers}
    if len(versions) != 1:
        raise ValueError(f"APK ABIs use different SDL versions: {sorted(versions)}")
    expected = next(iter(versions))
    if expected[0] != 2:
        raise ValueError(f"This launcher requires SDL 2, got {expected}")
    candidates = []
    for build in builds:
        cache = build if build.is_file() else build / "CMakeCache.txt"
        if cache.is_file():
            match = re.search(r"^CMAKE_HOME_DIRECTORY:[^=]*=(.+)$",
                cache.read_text(encoding="utf-8-sig"), re.M)
            if match:
                candidates.append(host_path(match[1].strip()))
    candidates.extend(host_path(source) for source in sources)
    if os.environ.get("SDL2_ANDROID_HOME"):
        candidates.append(host_path(os.environ["SDL2_ANDROID_HOME"]))
    version = ".".join(map(str, expected))
    candidates.append(repo / "build" / "renderer-deps" / f"SDL-{version}")
    candidates.append(repo / "build" / "sdl-java-cache" / f"SDL-{version}")
    selected = None
    for source in dict.fromkeys(path.resolve() for path in candidates):
        activity = source / "android-project/app/src/main/java/org/libsdl/app/SDLActivity.java"
        header = source / "include/SDL_version.h"
        if header.is_file() and activity.is_file() and native_version(header) == expected and java_version(activity) == expected:
            selected = activity.parent
            break
    if selected is None:
        raise ValueError(f"No SDL {version} Java source matches the native dependency. "
            "Supply its source via --sdl-source or its CMake build via --sdl-build. "
            "APK packaging stopped before replacing Java classes.")
    project = project.resolve()
    destination = (project / "app/src/main/java/org/libsdl/app").resolve()
    destination.relative_to(project)
    if selected == destination or selected.is_relative_to(destination) or destination.is_relative_to(selected):
        raise ValueError("SDL source and APK staging directories must be separate")
    if not (project / "app").is_dir():
        raise ValueError(f"Not an APK staging project: {project}")
    files = sorted(selected.rglob("*.java"))
    if not files or any(file.is_symlink() for file in files):
        raise ValueError(f"Invalid SDL Java source tree: {selected}")
    contents = {file.relative_to(selected): file.read_bytes() for file in files}
    # Only replace the explicitly named SDL package in the staging project.
    # The source dependency and the engine's org.openxray classes are untouched.
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    for relative, data in contents.items():
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    if java_version(destination / "SDLActivity.java") != expected:
        raise ValueError("Staged SDL Java version differs from the native dependency")
    manifest = {"version": version, "source": str(selected),
        "native_headers": [str(header.resolve()) for header in headers],
        "java_sha256": {path.as_posix(): hashlib.sha256(data).hexdigest() for path, data in contents.items()}}
    (project / "sdl-java-provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--version-header", type=Path, action="append", required=True)
    parser.add_argument("--sdl-source", type=Path, action="append", default=[])
    parser.add_argument("--sdl-build", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        result = stage(args.project, args.version_header, args.sdl_source, args.sdl_build)
    except (OSError, ValueError) as error:
        parser.exit(1, f"SDL Java/native pairing failed: {error}\n")
    print(f"SDL Java/native pair {result['version']}: staged {len(result['java_sha256'])} Java classes from {result['source']}")


if __name__ == "__main__":
    sys.exit(main())
