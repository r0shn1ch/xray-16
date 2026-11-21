#!/usr/bin/env python3
"""Compile local deferred lighting for the Vulkan gameplay renderer."""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/Layers/xrRenderVK/smoke/local_light.frag"
HEADER = ROOT / "src/Layers/xrRenderVK/LocalLightShaders.h"


def generate(glslc):
    with tempfile.TemporaryDirectory() as directory:
        variants = (("local_light.spv", "Fragment", ()),
                    ("local_light_msaa.spv", "FragmentMsaa", ("GBUFFER_MSAA=1",)))
        glslang = "glslang" in Path(glslc).name.lower()
        options = ["-V", "--target-env", "vulkan1.0"] if glslang else ["--target-env=vulkan1.0"]
        binaries = []
        for file_name, name, defines in variants:
            output = Path(directory) / file_name
            subprocess.run([glslc, *options, *(f"-D{define}" for define in defines),
                            "-o", str(output), str(SOURCE)], check=True, capture_output=True)
            data = output.read_bytes()
            if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
                raise ValueError(f"invalid local light SPIR-V: {name}")
            binaries.append((name, struct.unpack(f"<{len(data) // 4}I", data)))
    lines = ["#pragma once", "", "#include <cstdint>", "",
             "namespace xray::render::vulkan::local_light_shaders", "{"]
    for name, words in binaries:
        lines.append(f"inline constexpr uint32_t {name}[] = {{")
        for start in range(0, len(words), 8):
            lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[start:start + 8]) + ",")
        lines.extend(["};", ""])
    return "\n".join(lines + ["}", ""])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = generate(args.glslc)
    if args.check:
        if HEADER.read_text() != result:
            parser.error("LocalLightShaders.h is stale")
    else:
        HEADER.write_text(result)
