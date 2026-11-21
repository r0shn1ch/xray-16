#!/usr/bin/env python3
"""Compile the upstream-derived GPU PAPI compute kernels for Vulkan."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/Layers/xrRenderVK/gpu_particles"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--spirv-val", default="spirv-val")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--collision-test-output", type=Path,
                        help="also compile the standalone collision GPU test shader")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("init", "simulate", "compact", "motion_capture"):
        output = args.output_dir / f"gpu_particle_{name}.spv"
        subprocess.run([args.glslc, "-x", "hlsl", "-fshader-stage=compute", "-fentry-point=main",
                        "--target-env=vulkan1.1", "-o", str(output),
                        str(SOURCE / f"gpu_particle_{name}.cs")], check=True)
        subprocess.run([args.spirv_val, "--target-env", "vulkan1.1", str(output)], check=True)
    if args.collision_test_output:
        args.collision_test_output.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([args.glslc, "-x", "hlsl", "-fshader-stage=compute", "-fentry-point=main",
                        "--target-env=vulkan1.1", "-I", str(SOURCE), "-o", str(args.collision_test_output),
                        str(ROOT / "tests/shaders/gpu_particle_collision_test.cs")], check=True)
        subprocess.run([args.spirv_val, "--target-env", "vulkan1.1", str(args.collision_test_output)], check=True)


if __name__ == "__main__":
    main()
