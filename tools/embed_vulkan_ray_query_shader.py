#!/usr/bin/env python3
"""Compile and embed the ray-query batch compute shader."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/Layers/xrRenderVK/smoke/ray_query_batch.comp"
HEADER = ROOT / "src/Layers/xrRenderVK/RayQueryBatchShaders.h"


def generate(compiler, glslc=None):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "ray_query_batch.spv"
        command = [glslc, "--target-env=vulkan1.1", "--target-spv=spv1.4", "-fshader-stage=compute",
                   "-o", str(output), str(SOURCE)] if glslc else [
            compiler, "-V", "--target-env", "vulkan1.1", "--target-env", "spirv1.4", "-S", "comp",
            "-o", str(output), str(SOURCE)]
        subprocess.run(command, check=True, capture_output=True)
        data = output.read_bytes()
    if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
        raise ValueError("invalid ray-query SPIR-V")
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    lines = ["#pragma once", "#include <cstdint>",
             "namespace xray::render::vulkan::ray_query_batch_shaders", "{",
             "inline constexpr uint32_t Query[] = {"]
    for index in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[index:index + 8]) + ",")
    lines += ["};", "}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="glslangValidator")
    parser.add_argument("--glslc")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    result = generate(arguments.compiler, arguments.glslc)
    if arguments.check:
        if HEADER.read_text() != result:
            parser.error("RayQueryBatchShaders.h is stale")
    else:
        HEADER.write_text(result)
