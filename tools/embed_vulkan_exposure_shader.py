#!/usr/bin/env python3
"""Embed the full-scene temporal Vulkan exposure shaders."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SHADERS = {
    'Histogram': ROOT / 'src/Layers/xrRenderVK/smoke/temporal_exposure_histogram.comp',
    'Adapt': ROOT / 'src/Layers/xrRenderVK/smoke/temporal_exposure_adapt.comp',
}
HEADER = ROOT / 'src/Layers/xrRenderVK/TemporalExposureShaders.h'


def compile_shader(glslc, source, output):
    command = [glslc, '--target-env=vulkan1.0', '-o', str(output)]
    if 'glslang' in Path(glslc).name.lower():
        command = [glslc, '--target-env', 'vulkan1.0', '-V', '-o', str(output)]
    subprocess.run(command + [str(source)], check=True)
    data = output.read_bytes()
    if len(data) < 20 or len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
        raise ValueError(f'invalid temporal exposure SPIR-V: {source.name}')
    return struct.unpack('<' + 'I' * (len(data) // 4), data)


def generate(glslc):
    with tempfile.TemporaryDirectory() as directory:
        arrays = {
            name: compile_shader(glslc, source, Path(directory) / f'{name.lower()}.spv')
            for name, source in SHADERS.items()
        }
    lines = ['#pragma once', '#include <cstdint>',
             'namespace xray::render::vulkan::temporal_exposure_shaders', '{']
    for name, words in arrays.items():
        lines.append(f'inline constexpr uint32_t {name}[] = {{')
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
            parser.error('TemporalExposureShaders.h is stale')
    else:
        HEADER.write_text(result)
