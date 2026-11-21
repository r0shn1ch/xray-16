#!/usr/bin/env python3
"""Embed the static-object velocity raster shaders."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / 'src/Layers/xrRenderVK/smoke'
HEADER = ROOT / 'src/Layers/xrRenderVK/ObjectVelocityShaders.h'


def generate(compiler):
    lines = ['#pragma once', '#include <cstdint>',
             'namespace xray::render::vulkan::object_velocity_shaders', '{']
    with tempfile.TemporaryDirectory() as directory:
        for name, source_name, suffix, defines in [
            ('Vertex', 'object_velocity.vert', 'vert', []),
            ('SkinnedVertex', 'object_velocity_skinned.vert', 'vert', []),
            ('LodVertex', 'object_velocity_lod.vert', 'vert', []),
            ('LodFragment', 'object_velocity_lod.frag', 'frag', []),
            ('ParticleVelocityVertex', 'object_velocity_particle.vert', 'vert', []),
            ('ParticleVelocityFragment', 'object_velocity_particle.frag', 'frag', []),
            ('Fragment', 'object_velocity.frag', 'frag', []),
            ('CutoutFragment', 'object_velocity.frag', 'frag', ['-DXRAY_OBJECT_VELOCITY_CUTOUT'])]:
            source = SHADERS / source_name
            output = Path(directory) / f'{name}.spv'
            if 'glslang' in Path(compiler).name.lower():
                command = [compiler, '--target-env', 'vulkan1.0', '-V', '-S', suffix, *defines,
                           '-o', str(output), str(source)]
            else:
                command = [compiler, '--target-env=vulkan1.0', *defines,
                           '-o', str(output), str(source)]
            subprocess.run(command, check=True)
            data = output.read_bytes()
            if len(data) < 20 or len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
                raise ValueError(f'invalid {name} SPIR-V')
            words = struct.unpack('<' + 'I' * (len(data) // 4), data)
            lines.append(f'inline constexpr uint32_t {name}[] = {{')
            for index in range(0, len(words), 8):
                lines.append('    ' + ', '.join(f'0x{word:08x}u' for word in words[index:index + 8]) + ',')
            lines.append('};')
    return '\n'.join(lines + ['}', ''])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glslc', default='glslc')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = generate(args.glslc)
    if args.check:
        if HEADER.read_text() != result:
            parser.error('ObjectVelocityShaders.h is stale')
    else:
        HEADER.write_text(result)
