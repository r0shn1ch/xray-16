#!/usr/bin/env python3
"""Compile and embed the Vulkan world radiance cache update shader."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = root / "src/Layers/xrRenderVK/smoke/world_radiance_cache_update.comp"
header = root / "src/Layers/xrRenderVK/WorldRadianceCacheShaders.h"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", default="glslangValidator")
parser.add_argument("--glslc")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
with tempfile.TemporaryDirectory() as directory:
    binary = Path(directory) / "world_radiance_cache_update.spv"
    if args.glslc:
        command = [args.glslc, "--target-env=vulkan1.0", "-fshader-stage=compute",
                   "-o", str(binary), str(source)]
    else:
        command = [args.compiler, "-V", "--target-env", "vulkan1.0", "-S", "comp",
                   "-o", str(binary), str(source)]
    subprocess.run(command, check=True, capture_output=True)
    data = binary.read_bytes()
if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
    raise ValueError("invalid world radiance cache SPIR-V")
words = struct.unpack("<" + "I" * (len(data) // 4), data)
lines = ["#pragma once", "#include <cstdint>",
         "namespace xray::render::vulkan::world_radiance_cache_shaders", "{",
         "inline constexpr uint32_t Update[] = {"]
for index in range(0, len(words), 8):
    lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[index:index + 8]) + ",")
result = "\n".join(lines + ["};", "}", ""])
if args.check:
    if not header.exists() or header.read_text() != result:
        parser.error("WorldRadianceCacheShaders.h is stale")
else:
    header.write_text(result)
