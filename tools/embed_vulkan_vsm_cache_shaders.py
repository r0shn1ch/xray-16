#!/usr/bin/env python3
"""Compile Vulkan static virtual-shadow cache and publication shaders."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
sources = root / "src/Layers/xrRenderVK/smoke"
header = root / "src/Layers/xrRenderVK/VsmCacheShaders.h"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", default="glslangValidator")
parser.add_argument("--check", action="store_true")
args = parser.parse_args()
lines = ["#pragma once", "#include <cstdint>", "namespace xray::render::vulkan::vsm_cache_shaders", "{"]
with tempfile.TemporaryDirectory() as directory:
    for source, name in (("vsm_static_cache.comp", "Prepare"), ("vsm_static_publish.comp", "Publish")):
        binary = Path(directory) / (source + ".spv")
        subprocess.run([args.compiler, "-V", "--target-env", "vulkan1.0", "-S", "comp",
                        "-o", str(binary), str(sources / source)], check=True)
        data = binary.read_bytes()
        if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
            raise ValueError("invalid VSM cache SPIR-V")
        words = struct.unpack("<" + "I" * (len(data) // 4), data)
        lines.append(f"inline constexpr uint32_t {name}[] = {{")
        for index in range(0, len(words), 8):
            lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[index:index + 8]) + ",")
        lines.append("};")
result = "\n".join(lines + ["}", ""])
if args.check:
    if header.read_text() != result:
        parser.error("VsmCacheShaders.h is stale")
else:
    header.write_text(result)
