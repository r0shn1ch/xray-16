# Vulkan DDS texture contract

The Vulkan DDS loader accepts these pixel encodings:

| Encoding | Upload with BC support | Upload without BC support |
|---|---|---|
| RGBA8 / BGRA8, including DX10 sRGB variants | Uncompressed | Uncompressed |
| Legacy 8/16/24/32-bit RGB, luminance and alpha with valid contiguous, non-overlapping masks | Expanded to RGBA8 | Expanded to RGBA8 |
| BC1 / DXT1 | BC1 | Expanded to RGBA8 |
| BC2 / DXT3 | BC2 | Expanded to RGBA8 |
| BC3 / DXT5 | BC3 | Expanded to RGBA8 |
| BC7, all eight modes | BC7 | Expanded to RGBA8 |

The BC path follows the selected device's `textureCompressionBC` feature.
BC7 is accepted through DX10 formats 98 (UNORM) and 99 (sRGB). BC4, BC5,
BC6H, texture arrays, and volume textures are currently rejected with an error.

DX10 sRGB metadata remains attached to the Vulkan image. Software BC expansion
preserves encoded channel bytes and uses an RGBA8 sRGB image for sRGB input;
it does not apply an additional CPU gamma conversion. Legacy DDS encodings
without an sRGB format declaration use UNORM. This describes loader behavior;
the renderer's broader material and lighting color-policy review is separate.

Supported shapes are a single 2D image or a six-face square cubemap, including
their mip chains. The loader validates dimensions, mip counts, complete face
payloads and exact input length before calling GLI. Legacy uncompressed rows
may be byte-aligned or DWORD-aligned when their layout matches the payload.
World texture mip selection occurs before BC expansion; cubemaps retain their
complete face/mip layout. For large textures, RGBA8 fallback uses approximately
four times the storage of BC2/BC3/BC7, or eight times the storage of BC1.

`vulkan_dds_test` covers DDS parsing, compressed upload and RGBA fallback,
including partial blocks, mip tails and cubemaps. `vulkan_bc7_gpu` compares the
production decoder against Vulkan's native BC7 decoding for 512 blocks in
UNORM and sRGB: all modes, all partition IDs, rotations and index selectors.
It allows one byte of rounding difference after sRGB conversion and reports
an explicit CTest skip when the device lacks the required BC7/blit formats.
