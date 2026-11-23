#!/usr/bin/env python3
"""Embed the AO GPU fixture shader, which includes the production AO function."""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/shaders/vulkan_ao_gpu.comp"
HEADER = ROOT / "tests/vulkan_ao_gpu_shader.h"


def generate(compiler):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "ao.spv"
        options = (["-V", "--target-env", "vulkan1.0"] if "glslang" in Path(compiler).name.lower()
                   else ["--target-env=vulkan1.0"])
        subprocess.run([compiler, *options, "-o", str(output), str(SOURCE)], check=True)
        data = output.read_bytes()
    if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
        raise ValueError("invalid AO GPU test SPIR-V")
    words = struct.unpack("<" + "I" * (len(data) // 4), data)
    lines = ["#pragma once", "#include <cstdint>", "namespace ao_gpu_shader {",
             "inline constexpr uint32_t Compute[] = {"]
    for start in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[start:start + 8]) + ",")
    lines += ["};", "}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generated = generate(arguments.glslc)
    if arguments.check:
        if HEADER.read_text() != generated:
            parser.error("AO GPU shader header is stale")
    else:
        HEADER.write_text(generated)
