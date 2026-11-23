#!/usr/bin/env python3
"""Compile and embed the Vulkan diffuse reference path tracing shader."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = root / "src/Layers/xrRenderVK/smoke/reference_path_trace.comp"
header = root / "src/Layers/xrRenderVK/ReferencePathTraceShaders.h"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", default="glslangValidator")
parser.add_argument("--glslc")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    binary = Path(directory) / "reference_path_trace.spv"
    if args.glslc:
        command = [args.glslc, "--target-env=vulkan1.1", "--target-spv=spv1.4", "-fshader-stage=compute",
                   "-I", str(source.parent), "-o", str(binary), str(source)]
    else:
        command = [args.compiler, "-V", "--target-env", "vulkan1.1", "--target-env", "spirv1.4", "-S", "comp",
                   "-I" + str(source.parent), "-o", str(binary), str(source)]
    subprocess.run(command, check=True, capture_output=True)
    data = binary.read_bytes()
if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
    raise ValueError("invalid reference path trace SPIR-V")
words = struct.unpack("<" + "I" * (len(data) // 4), data)
lines = ["#pragma once", "#include <cstdint>",
         "namespace xray::render::vulkan::reference_path_trace_shaders", "{",
         "inline constexpr uint32_t Trace[] = {"]
for index in range(0, len(words), 8):
    lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[index:index + 8]) + ",")
result = "\n".join(lines + ["};", "}", ""])
if args.check:
    if not header.exists() or header.read_text() != result:
        parser.error("ReferencePathTraceShaders.h is stale")
else:
    header.write_text(result)
