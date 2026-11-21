#!/usr/bin/env python3
"""Build the Vulkan weather/fog fragment and embed its validated SPIR-V."""

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/Layers/xrRenderVK/smoke/weather.frag"
HEADER = ROOT / "src/Layers/xrRenderVK/WeatherShaders.h"


def generate(compiler):
    with tempfile.TemporaryDirectory() as temp:
        variants = (("weather.spv", "Fragment", ()),
                    ("weather_msaa.spv", "FragmentMsaa", ("GBUFFER_MSAA=1",)))
        binaries = []
        glslang = "glslang" in Path(compiler).name.lower()
        options = ["-V", "--target-env", "vulkan1.0"] if glslang else ["--target-env=vulkan1.0"]
        for file_name, name, defines in variants:
            output = Path(temp) / file_name
            subprocess.run([compiler, *options, *(f"-D{define}" for define in defines),
                            "-o", str(output), str(SOURCE)], check=True, capture_output=True)
            data = output.read_bytes()
            if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
                raise ValueError(f"invalid weather SPIR-V: {name}")
            binaries.append((name, struct.unpack(f"<{len(data) // 4}I", data)))
    lines = ["#pragma once", "", "#include <cstdint>", "",
             "namespace xray::render::vulkan::weather_shaders", "{"]
    for name, words in binaries:
        lines.append(f"inline constexpr uint32_t {name}[] = {{")
        for start in range(0, len(words), 8):
            lines.append("    " + ", ".join(f"0x{word:08x}u" for word in words[start:start + 8]) + ",")
        lines.extend(["};", ""])
    lines += ["}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = generate(args.glslc)
    if args.check:
        if HEADER.read_text() != content:
            parser.error("WeatherShaders.h is stale")
    else:
        HEADER.write_text(content)
