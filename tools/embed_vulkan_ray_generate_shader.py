#!/usr/bin/env python3
"""Compile the G-buffer ray generation compute shader into a C++ header."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
source = root / "src/Layers/xrRenderVK/smoke/ray_generate.comp"
header = root / "src/Layers/xrRenderVK/RayGenerateShaders.h"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", default="glslangValidator")
parser.add_argument("--glslc")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
lines = ["#pragma once", "#include <cstdint>",
         "namespace xray::render::vulkan::ray_generate_shaders", "{"]
with tempfile.TemporaryDirectory() as directory:
    for name, defines in (("Generate", []), ("Indirect", ["-DINDIRECT=1"])):
        binary = Path(directory) / (name + ".spv")
        if args.glslc:
            command = [args.glslc, "--target-env=vulkan1.0", "-fshader-stage=compute",
                       *defines, "-o", str(binary), str(source)]
        else:
            command = [args.compiler, "-V", "--target-env", "vulkan1.0", "-S", "comp",
                       *defines, "-o", str(binary), str(source)]
        subprocess.run(command, check=True, capture_output=True)
        data = binary.read_bytes()
        if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
            raise ValueError("invalid ray generation SPIR-V")
        words = struct.unpack("<" + "I" * (len(data) // 4), data)
        lines.append(f"inline constexpr uint32_t {name}[] = {{")
        for index in range(0, len(words), 8):
            lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[index:index + 8]) + ",")
        lines.append("};")
result = "\n".join(lines + ["}", ""])
if args.check:
    if header.read_text() != result:
        parser.error("RayGenerateShaders.h is stale")
else:
    header.write_text(result)
