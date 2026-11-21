#!/usr/bin/env python3
"""Fail APK staging when a Vulkan shader binary or its source metadata is stale."""

import argparse
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

if __package__:
    from .compile_vulkan_shader import (
        _fingerprint, _manifest, _shader_metadata, _shader_source_metadata, _valid_spirv,
    )
else:
    from compile_vulkan_shader import (
        _fingerprint, _manifest, _shader_metadata, _shader_source_metadata, _valid_spirv,
    )


def _vfs_shader_asset(shader_root, path):
    relative = PurePosixPath(path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts or "." in relative.parts:
        return None
    resolved_root = shader_root.resolve()
    resolved = resolved_root.joinpath(*relative.parts).resolve()
    try:
        return resolved.relative_to(resolved_root).as_posix()
    except ValueError:
        return None


def check_assets(shader_root, manifest):
    variants = _manifest(manifest, shader_root)
    problems = []
    for variant in variants:
        path = variant["output"]
        if not path.is_file():
            problems.append(f"missing {path}")
            continue
        elif not _valid_spirv(path.read_bytes(), variant["stage"], variant.get("entry", "main")):
            problems.append(f"invalid SPIR-V stage/entry: {path}")
            continue
        metadata_path = Path(str(path) + ".meta")
        if not metadata_path.is_file():
            problems.append(f"missing shader metadata: {metadata_path}")
        elif metadata_path.read_bytes() != _shader_metadata(variant, path.read_bytes()):
            problems.append(f"stale or invalid shader metadata: {metadata_path}")
    return problems


def check_apk(apk, manifest):
    shader_root = manifest.parent.parent
    variants = _manifest(manifest, shader_root)
    problems = []
    try:
        with ZipFile(apk) as archive:
            if not any("lib/" + abi + "/libmain.so" in archive.namelist()
                       for abi in ("armeabi-v7a", "arm64-v8a")):
                problems.append("missing Android gameplay libmain.so")
            for variant in variants:
                relative_output = variant["output"].relative_to(shader_root).as_posix()
                name = "assets/gamedata/shaders/" + relative_output
                try:
                    data = archive.read(name)
                except KeyError:
                    problems.append(f"missing {name}")
                    continue
                if not _valid_spirv(data, variant["stage"], variant.get("entry", "main")):
                    problems.append(f"invalid SPIR-V stage/entry: {name}")
                    continue
                metadata_name = name + ".meta"
                try:
                    metadata = archive.read(metadata_name)
                except KeyError:
                    problems.append(f"missing {metadata_name}")
                    continue
                if metadata != _shader_metadata(variant, data):
                    problems.append(f"stale or invalid shader metadata: {metadata_name}")

                source_metadata = _shader_source_metadata(variant)
                source_path = next((line.split("=", 1)[1] for line in source_metadata
                                    if line.startswith("source=")), "")
                source_asset = _vfs_shader_asset(shader_root, source_path)
                if source_asset is not None:
                    source_name = "assets/gamedata/shaders/" + source_asset
                    try:
                        source_data = archive.read(source_name)
                    except KeyError:
                        problems.append(f"missing runtime shader source: {source_name}")
                    else:
                        expected = next((line.split("=", 1)[1] for line in source_metadata
                                         if line.startswith("source-fingerprint=")), "")
                        if f"{_fingerprint(source_data):016x}" != expected:
                            problems.append(f"stale runtime shader source: {source_name}")

                for line in source_metadata:
                    if not line.startswith("dependency="):
                        continue
                    dependency, separator, expected = line[len("dependency="):].rpartition("=")
                    dependency_asset = _vfs_shader_asset(shader_root, dependency)
                    if not separator or dependency_asset is None:
                        problems.append(f"invalid runtime shader dependency: {line}")
                        continue
                    dependency_name = "assets/gamedata/shaders/" + dependency_asset
                    try:
                        dependency_data = archive.read(dependency_name)
                    except KeyError:
                        problems.append(f"missing runtime shader dependency: {dependency_name}")
                    else:
                        if f"{_fingerprint(dependency_data):016x}" != expected:
                            problems.append(f"stale runtime shader dependency: {dependency_name}")
    except (BadZipFile, OSError) as error:
        problems.append(f"cannot inspect APK {apk}: {error}")
    return problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shader-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--apk", type=Path,
                        help="also verify the final APK contains libmain.so and every shader")
    args = parser.parse_args()
    errors = check_assets(args.shader_root, args.manifest)
    if args.apk:
        errors.extend(check_apk(args.apk, args.manifest))
    for error in errors:
        print(error)
    raise SystemExit(1 if errors else 0)
