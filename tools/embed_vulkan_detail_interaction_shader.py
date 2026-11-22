#!/usr/bin/env python3
"""Embed the GPU grass contact/spring field; --check validates the shipped words."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/Layers/xrRenderVK/smoke/detail_interaction.comp"
HEADER = ROOT / "src/Layers/xrRenderVK/DetailInteractionShaders.h"


def generate(compiler, source=SOURCE, namespace="detail_interaction_shaders"):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "interaction.spv"
        if "glslang" in Path(compiler).name.lower():
            command = [compiler, "-V", "--target-env", "vulkan1.0", "-o", str(output), str(source)]
        else:
            command = [compiler, "--target-env=vulkan1.0", "-o", str(output), str(source)]
        subprocess.run(command, check=True)
        data = output.read_bytes()
        if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
            raise ValueError("invalid detail interaction SPIR-V")
        words = struct.unpack(f"<{len(data)//4}I", data)
    lines = ["#pragma once", "#include <cstdint>",
             f"namespace xray::render::vulkan::{namespace}", "{",
             "inline constexpr uint32_t Compute[] = {"]
    for first in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[first:first+8]) + ",")
    return "\n".join(lines + ["};", "}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for source, header, namespace in [(SOURCE, HEADER, "detail_interaction_shaders"),
            (SOURCE.with_name("detail_ground_height.comp"), HEADER.with_name("DetailGroundHeightShaders.h"),
             "detail_ground_height_shaders"),
            (SOURCE.with_name("detail_bend.comp"), HEADER.with_name("DetailBendShaders.h"),
             "detail_bend_shaders"),
            (SOURCE.with_name("detail_cull.comp"), HEADER.with_name("DetailCullShaders.h"),
             "detail_cull_shaders")]:
        result = generate(args.glslc, source, namespace)
        if args.check:
            if header.read_text() != result: parser.error(f"{header.name} is stale")
        else:
            header.write_text(result)
