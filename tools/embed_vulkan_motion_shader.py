#!/usr/bin/env python3
"""Embed the Vulkan depth-reprojected motion-vector compute shader."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/Layers/xrRenderVK/smoke/motion_vectors.comp'
HEADER = ROOT / 'src/Layers/xrRenderVK/MotionVectorShaders.h'


def generate(glslc):
    lines = ['#pragma once', '#include <cstdint>',
             'namespace xray::render::vulkan::motion_vector_shaders', '{']
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'motion.spv'
        for name, defines in [('FillRg', []), ('FillRgba', ['-DXRAY_MOTION_RGBA'])]:
            command = [glslc, '--target-env=vulkan1.0', *defines,
                       '-o', str(output), str(SOURCE)]
            if 'glslang' in Path(glslc).name.lower():
                command = [glslc, '--target-env', 'vulkan1.0', '-V', *defines,
                           '-o', str(output), str(SOURCE)]
            subprocess.run(command, check=True)
            data = output.read_bytes()
            if len(data) < 20 or len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
                raise ValueError('invalid motion-vector SPIR-V')
            words = struct.unpack('<' + 'I' * (len(data) // 4), data)
            lines.append('inline constexpr uint32_t ' + name + '[] = {')
            for index in range(0, len(words), 8):
                lines.append('    ' + ', '.join(f'0x{word:08x}u' for word in words[index:index + 8]) + ',')
            lines.append('};')
    lines += ['}', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glslc', default='glslc')
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    result = generate(arguments.glslc)
    if arguments.check:
        if HEADER.read_text() != result:
            parser.error('MotionVectorShaders.h is stale')
    else:
        HEADER.write_text(result)
