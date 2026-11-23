#!/usr/bin/env python3
"""Compile the complete visibility pipeline before replacing any shader assets."""

import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/Layers/xrRenderVK/smoke"
OUTPUT = ROOT / "res/gamedata/shaders/vk"
VARIANTS = (
    ("visibility_software_raster.comp", "visibility_software_raster.cs.spv", ()),
    ("visibility_software_raster.comp", "visibility_software_raster_alpha.cs.spv", ("VISIBILITY_ALPHA_TEST=1",)),
    ("visibility_material_resolve.comp", "visibility_material_resolve.cs.spv", ()),
    ("visibility_material_resolve.comp", "visibility_material_resolve_cube.cs.spv", ("VISIBILITY_ENVIRONMENT_CUBE=1",)),
    ("visibility_gbuffer.vert", "visibility_gbuffer.vs.spv", ()),
    ("visibility_gbuffer.frag", "visibility_gbuffer.ps.spv", ()),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true", help="report stale assets without replacing them")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as directory:
        staged = []
        for source, output, defines in VARIANTS:
            temporary = Path(directory) / output
            command = [args.glslc, "--target-env=vulkan1.0", "-O"]
            command.extend("-D" + define for define in defines)
            command.extend([str(SOURCE / source), "-o", str(temporary)])
            subprocess.run(command, check=True)
            data = temporary.read_bytes()
            if len(data) < 20 or len(data) % 4 or data[:4] != b"\x03\x02\x23\x07":
                raise ValueError(f"compiler produced invalid SPIR-V: {source}")
            destination = args.output_dir / output
            if args.check:
                if not destination.is_file() or destination.read_bytes() != data:
                    parser.error(f"visibility shader is missing or stale: {destination}")
            else:
                staged.append((temporary, destination))
        for temporary, destination in staged:
            temporary.replace(destination)


if __name__ == "__main__":
    main()
