#!/usr/bin/env python3
"""Build the core Vulkan compute culling and batched point-light shaders."""
import argparse
from pathlib import Path
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'src/Layers/xrRenderVK/LightGridShaders.h'


def generate(glslc):
    lines=['#pragma once', '#include <cstdint>', 'namespace xray::render::vulkan::light_grid_shaders', '{']
    with tempfile.TemporaryDirectory() as d:
        for name,stage in [('Cull','comp'),('Shade','frag')]:
            output=Path(d)/'shader.spv'
            subprocess.run([glslc,'--target-env=vulkan1.0','-o',str(output),str(ROOT/'src/Layers/xrRenderVK/smoke'/('light_grid.'+stage))],check=True)
            data=output.read_bytes()
            if len(data) < 20 or len(data) % 4 or data[:4] != b'\x03\x02\x23\x07':
                raise ValueError('invalid light grid SPIR-V')
            words=struct.unpack('<'+'I'*(len(data)//4),data)
            lines.append('inline constexpr uint32_t '+name+'[] = {')
            for i in range(0,len(words),8): lines.append('    '+', '.join(f'0x{x:08x}u' for x in words[i:i+8])+',')
            lines.append('};')
    lines.append('}')
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--glslc',default='glslc')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = generate(args.glslc)
    if args.check:
        if HEADER.read_text() != result:
            parser.error('LightGridShaders.h is stale')
    else:
        HEADER.write_text(result)
