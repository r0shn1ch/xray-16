#!/usr/bin/env python3
"""Compile game HLSL variants to SPIR-V with DXC, preserving game sources.

Single shader: --source FILE --stage ps --entry main --output FILE
Variant set:   --manifest FILE --output-dir DIR
"""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import struct
import sys
import tempfile

STAGES = ("vs", "ps", "cs", "gs", "hs", "ds")
SPIRV_MAGIC = b"\x03\x02\x23\x07"
EXECUTION_MODEL = {"vs": 0, "hs": 1, "ds": 2, "gs": 3, "ps": 4, "cs": 5}
SHADER_METADATA_MAGIC = "XRAY_VULKAN_SHADER_METADATA_V1"
INCLUDE_DIRECTIVE = re.compile(r'^\s*#\s*include\s*["<]([^">]+)[">]', re.MULTILINE)


def _valid_spirv(data, stage, entry):
    if len(data) < 20 or len(data) % 4 or data[:4] != SPIRV_MAGIC:
        return False
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[3] == 0 or words[4] != 0:
        return False
    cursor = 5
    found = False
    while cursor < len(words):
        instruction = words[cursor]
        size, opcode = instruction >> 16, instruction & 0xFFFF
        if size == 0 or cursor + size > len(words):
            return False
        if opcode == 15 and size >= 4:
            name = struct.pack(f"<{size - 3}I", *words[cursor + 3:cursor + size]).split(b"\0", 1)[0]
            found |= words[cursor + 1] == EXECUTION_MODEL[stage] and name == entry.encode("utf-8")
        cursor += size
    return found


def _fingerprint(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    for shift in range(0, 64, 8):
        value = ((value ^ ((len(data) >> shift) & 0xFF)) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value


def _variant(source, stage, entry, output, includes, defines, runtime_source=None, shader_root=None):
    if stage not in STAGES or not entry or not source.is_file():
        raise ValueError(f"invalid shader source/stage/entry: {source} ({stage}, {entry})")
    if not isinstance(defines, list) or not all(isinstance(x, str) and x for x in defines):
        raise ValueError(f"invalid defines for {source}")
    if any("|" in definition or "\n" in definition or "\r" in definition for definition in defines):
        raise ValueError(f"shader defines contain reserved metadata characters: {source}")
    return dict(source=source, stage=stage, entry=entry, output=output,
                includes=includes, defines=defines,
                runtime_source=runtime_source or source.as_posix(), shader_root=shader_root)


def _manifest(path, output_dir):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("variants"), list) or not data["variants"]:
        raise ValueError("manifest requires a nonempty variants array")
    root = path.parent.resolve()
    common = data.get("include", [])
    if not isinstance(common, list) or not all(isinstance(x, str) for x in common):
        raise ValueError("manifest include must be an array of paths")
    variants = []
    outputs = set()
    for item in data["variants"]:
        if not isinstance(item, dict) or not isinstance(item.get("output"), str):
            raise ValueError("each variant requires an output path")
        relative = Path(item["output"])
        if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
            raise ValueError(f"variant output must stay inside output-dir: {relative}")
        output = output_dir / relative
        if output in outputs:
            raise ValueError(f"duplicate shader variant output: {relative}")
        outputs.add(output)
        source = root / item["source"]
        includes = [root / x for x in [*common, *item.get("include", [])]]
        variants.append(_variant(source, item["stage"], item.get("entry", "main"),
                                 output, includes, item.get("define", []),
                                 f"vk/{Path(item['source']).as_posix()}", root.parent))
    return variants


def _shader_source_metadata(variant):
    source_data = variant["source"].read_bytes()
    lines = [
        f"source={variant['runtime_source']}",
        f"stage={variant['stage']}",
        f"entry={variant['entry']}",
        f"defines={'|'.join(variant['defines'])}",
        f"source-fingerprint={_fingerprint(source_data):016x}",
    ]

    shader_root = variant.get("shader_root")
    if shader_root:
        root = Path(shader_root).resolve()
        visited = set()
        dependencies = {}

        def visit(path):
            path = path.resolve()
            if path in visited:
                return
            visited.add(path)
            try:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            except OSError:
                return
            for include in INCLUDE_DIRECTIVE.findall(text):
                resolved = None
                for directory in [path.parent, *variant["includes"]]:
                    candidate = (directory / include).resolve()
                    if candidate.is_file():
                        resolved = candidate
                        break
                if resolved is None:
                    continue
                try:
                    relative = resolved.relative_to(root).as_posix()
                except ValueError:
                    relative = None
                if relative and relative != variant["runtime_source"][3:]:
                    dependencies[relative] = _fingerprint(resolved.read_bytes())
                visit(resolved)

        visit(variant["source"])
        for path, fingerprint in sorted(dependencies.items()):
            # `relative` is already rooted at the game shader directory, so
            # retain that VFS path verbatim (`vk/LightingLimits.h`, not
            # `vk/vk/LightingLimits.h`).
            lines.append(f"dependency={path}={fingerprint:016x}")

    return lines


def _shader_metadata(variant, spirv_data, source_metadata=None):
    lines = [SHADER_METADATA_MAGIC, *(source_metadata or _shader_source_metadata(variant))]
    lines.append(f"spirv-fingerprint={_fingerprint(spirv_data):016x}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _compile(variant, dxc, temporary, compiler="dxc"):
    source = variant["source"]
    if compiler == "glslc":
        stage = {"vs": "vertex", "ps": "fragment", "cs": "compute",
                 "gs": "geometry", "hs": "tesscontrol", "ds": "tesseval"}[variant["stage"]]
        command = [dxc, "-x", "hlsl", "-fauto-combined-image-sampler", f"-fshader-stage={stage}",
                   f"-fentry-point={variant['entry']}", "--target-env=vulkan1.0",
                   "-o", str(temporary), str(source)]
        for directory in [source.parent, *variant["includes"]]:
            command += ["-I", str(directory)]
        for definition in variant["defines"]:
            command += [f"-D{definition}"]
    else:
        command = [dxc, "-spirv", "-fspv-target-env=vulkan1.0", "-Zpr",
                   "-T", f"{variant['stage']}_6_0", "-E", variant["entry"],
                   "-Fo", str(temporary), str(source)]
        for directory in [source.parent, *variant["includes"]]:
            command += ["-I", str(directory)]
        for definition in variant["defines"]:
            command += ["-D", definition]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        print(f"Shader compiler not found: {dxc}", file=sys.stderr)
        return 1
    if result.returncode:
        print(f"Shader compilation failed for {source} [{variant['stage']}, "
              f"{', '.join(variant['defines']) or 'default'}] (exit {result.returncode}):\n"
              f"{result.stdout}{result.stderr}", file=sys.stderr)
        return result.returncode
    data = temporary.read_bytes()
    if not _valid_spirv(data, variant["stage"], variant["entry"]):
        print(f"Compiler produced invalid SPIR-V or missing {variant['stage']} entry "
              f"'{variant['entry']}' for {source}", file=sys.stderr)
        return 1
    return 0


def compile_shader(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dxc", default=os.environ.get("DXC", "dxc"))
    parser.add_argument("--compiler", choices=("dxc", "glslc"), default="dxc")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--entry")
    parser.add_argument("--include", action="append", default=[], type=Path)
    parser.add_argument("--define", action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.compiler == "glslc" and args.dxc == "dxc":
        args.dxc = "glslc"
    if args.manifest:
        if not args.output_dir or any((args.source, args.output, args.stage, args.entry,
                                       args.include, args.define)):
            parser.error("--manifest requires --output-dir and no single-shader options")
        try:
            variants = _manifest(args.manifest, args.output_dir)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"invalid shader manifest: {exc}")
    else:
        if not args.source or not args.output or not args.stage or not args.entry:
            parser.error("single shader requires --source, --output, --stage and --entry")
        try:
            variants = [_variant(args.source, args.stage, args.entry, args.output,
                                 args.include, args.define)]
        except ValueError as exc:
            parser.error(str(exc))

    # Compile the complete set before replacing any existing output. Failed
    # variants cannot leave a mixture of new and old shader permutations.
    pending = []
    temporary_paths = []
    try:
        for variant in variants:
            output = variant["output"]
            output.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".spv", delete=False) as tmp:
                temporary = Path(tmp.name)
            with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".spv.meta", delete=False) as tmp:
                metadata_temporary = Path(tmp.name)
            temporary_paths.extend((temporary, metadata_temporary))
            pending.append((temporary, output, metadata_temporary, Path(str(output) + ".meta")))
            source_metadata = _shader_source_metadata(variant)
            status = _compile(variant, args.dxc, temporary, args.compiler)
            if status:
                return status
            if source_metadata != _shader_source_metadata(variant):
                print(f"Shader source or include changed during compilation: {variant['source']}",
                      file=sys.stderr)
                return 1
            metadata_temporary.write_bytes(_shader_metadata(
                variant, temporary.read_bytes(), source_metadata))
        for temporary, output, metadata_temporary, metadata_output in pending:
            temporary.replace(output)
            metadata_temporary.replace(metadata_output)
        return 0
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(compile_shader())
