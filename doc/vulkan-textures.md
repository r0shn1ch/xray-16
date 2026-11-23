# Vulkan DDS texture contract

The Vulkan DDS loader accepts these pixel encodings:

| Encoding | Upload with BC support | Upload without BC support |
|---|---|---|
| RGBA8 / BGRA8, including DX10 sRGB variants | Uncompressed | Uncompressed |
| DX10 and numeric FourCC R16F, RG16F, RGBA16F, R32F, RG32F, RGBA32F | Native float format | Native float format |
| Legacy 8/16/24/32-bit RGB, luminance and alpha with valid contiguous, non-overlapping masks | Expanded to RGBA8 | Expanded to RGBA8 |
| BC1 / DXT1 | BC1 | Expanded to RGBA8 |
| BC2 / DXT3 | BC2 | Expanded to RGBA8 |
| BC3 / DXT5 | BC3 | Expanded to RGBA8 |
| BC4 / ATI1 | BC4 | Expanded to RGBA8 (SNORM retained for signed input) |
| BC5 / ATI2 | BC5 | Expanded to RGBA8 (SNORM retained for signed input) |
| BC6H / DXGI 95–96 | BC6H | Expanded to RGBA16F, preserving HDR values |
| BC7, all eight modes | BC7 | Expanded to RGBA8 |

The BC path follows the selected device's `textureCompressionBC` feature.
BC7 is accepted through DX10 formats 98 (UNORM) and 99 (sRGB). BC4 and BC5
are accepted through DX10 UNORM/SNORM formats and legacy ATI/BC FourCC values.
BC6H is accepted through DX10 formats 95 (unsigned float) and 96 (signed float).
Devices with BC support upload the compressed image natively; other devices use
the bcdec half-float decoder and an RGBA16F image, preserving signed and HDR
values. DX10 2D arrays and cube arrays preserve every layer and mip through
decoding and upload. Legacy and DX10 3D volume images preserve their depth at
each mip and upload as Vulkan 3D images.

DX10 sRGB metadata remains attached to the Vulkan image. Software BC expansion
preserves encoded channel bytes and uses an RGBA8 sRGB image for sRGB input;
it does not apply an additional CPU gamma conversion. Legacy DDS encodings
without an sRGB format declaration use UNORM. This describes loader behavior;
the renderer's broader material and lighting color-policy review is separate.

Supported shapes are a single 2D image, a DX10 2D array, a six-face square
cubemap, a DX10 cube array whose layer count is a multiple of six, or a 3D
volume image, including their mip chains. The loader validates dimensions, mip
counts, complete face, array and volume payloads, and exact input length before
calling GLI. Float DDS payloads retain
their 16-bit or 32-bit Vulkan format and are validated using their actual
bytes-per-pixel. Legacy uncompressed rows
may be byte-aligned or DWORD-aligned when their layout matches the payload.
World texture mip selection occurs before BC expansion; cubemaps retain their
complete face/mip layout. For large textures, RGBA8 fallback uses approximately
four times the storage of BC2/BC3/BC5/BC7, or eight times the storage of
BC1/BC4/BC6H.

`vulkan_dds_test` covers DDS parsing, compressed upload and RGBA fallback,
including partial blocks, mip tails and cubemaps. `vulkan_bc7_gpu` compares the
production decoder against Vulkan's native BC7 decoding for 512 blocks in
UNORM and sRGB: all modes, all partition IDs, rotations and index selectors.
It allows one byte of rounding difference after sRGB conversion and reports
an explicit CTest skip when the device lacks the required BC7/blit formats.
