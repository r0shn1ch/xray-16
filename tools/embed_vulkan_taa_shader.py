#!/usr/bin/env python3
"""Embed the temporal color resolve compute shader."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/Layers/xrRenderVK/smoke/temporal_resolve.comp'
HEADER = ROOT / 'src/Layers/xrRenderVK/TemporalResolveShaders.h'


def generate(compiler):
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / 'temporal_resolve.spv'
        command = [compiler, '--target-env=vulkan1.0', '-o', str(output)]
        if 'glslang' in Path(compiler).name.lower():
            command = [compiler, '--target-env', 'vulkan1.0', '-V', '-o', str(output)]
        subprocess.run(command + [str(SOURCE)], check=True)
        data = output.read_bytes()
    if len(data) < 20 or len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
        raise ValueError('invalid temporal resolve SPIR-V')
    words = struct.unpack('<' + 'I' * (len(data) // 4), data)
    lines = ['#pragma once', '#include <cstdint>',
             'namespace xray::render::vulkan::temporal_resolve_shaders', '{',
             'inline constexpr uint32_t Resolve[] = {']
    for index in range(0, len(words), 8):
        lines.append('    ' + ', '.join(f'0x{word:08x}u' for word in words[index:index + 8]) + ',')
    lines += ['};', '}', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glslc', default='glslc')
    parser.add_argument('--check', action='store_true')
    arguments = parser.parse_args()
    result = generate(arguments.glslc)
    if arguments.check:
        if HEADER.read_text() != result:
            parser.error('TemporalResolveShaders.h is stale')
    else:
        HEADER.write_text(result)
