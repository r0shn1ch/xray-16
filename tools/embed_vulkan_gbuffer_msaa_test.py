#!/usr/bin/env python3
"""Compile the GPU multisample-coverage regression shaders."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
header = root / "tests/vulkan_gbuffer_msaa_shaders.h"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--glslc", default="glslc")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
lines = ["#pragma once", "#include <cstdint>", "namespace gbuffer_msaa_test_shaders", "{"]
with tempfile.TemporaryDirectory() as tmp:
    for name, stage in (("Vertex", "vert"), ("Fragment", "frag")):
        binary = Path(tmp) / (stage + ".spv")
        source = root / "tests/shaders" / ("gbuffer_msaa." + stage)
        subprocess.run([args.glslc, "--target-env=vulkan1.0", "-o", str(binary), str(source)], check=True)
        data = binary.read_bytes()
        if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
            raise ValueError("invalid MSAA test SPIR-V")
        words = struct.unpack("<" + "I" * (len(data) // 4), data)
        lines.append(f"inline constexpr uint32_t {name}[] = {{")
        for offset in range(0, len(words), 8):
            lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[offset:offset + 8]) + ",")
        lines.append("};")
result = "\n".join(lines + ["}", ""])
if args.check:
    if header.read_text() != result:
        parser.error("MSAA test shaders are stale")
else:
    header.write_text(result)
