#!/usr/bin/env python3
"""Bake an ONNX material model into the engine's linear *_pbr.dds format.

The model accepts float32 NCHW, batch one, with RGB diffuse or RGB diffuse
followed by RGB normal. Its output is float32 [1,3,H,W], in metallic,
roughness, ambient occlusion order. Inputs and outputs use the range [0,1].
Requires numpy, Pillow and onnxruntime.

Alternatively pass the OpenXRay ONNX model directory and an RGB normal map.
This mode packs parallax into alpha and can export albedo and parallax separately.
Pass several diffuse images to reuse the loaded models and write one PBR DDS per image.
"""

import argparse
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import onnxruntime as ort
from PIL import Image


def read_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def model_session(model, sessions):
    if sessions is not None and model in sessions:
        return sessions[model]
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    if sessions is not None:
        sessions[model] = session
    return session


def run_model(model, diffuse, normal, output_name, sessions=None):
    session = model_session(model, sessions)
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].type != "tensor(float)" or len(inputs[0].shape) != 4:
        raise ValueError("model must have one rank-four float32 NCHW input")

    rgb = read_rgb(diffuse)
    if normal is not None:
        normals = read_rgb(normal)
        if normals.shape != rgb.shape:
            raise ValueError("normal and diffuse dimensions must match")
        rgb = np.concatenate((rgb, normals), axis=2)
    tensor = np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])
    for expected, actual in zip(inputs[0].shape, tensor.shape):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(f"model input dimensions {inputs[0].shape} do not match {tensor.shape}")

    outputs = session.get_outputs()
    if output_name is None:
        if len(outputs) != 1:
            raise ValueError("pass --output-name for a model with multiple outputs")
        output_name = outputs[0].name
    if all(output.name != output_name for output in outputs):
        raise ValueError(f"unknown model output {output_name!r}")
    packed, = session.run([output_name], {inputs[0].name: tensor})
    if packed.dtype != np.float32 or packed.shape != (1, 3, *rgb.shape[:2]) or not np.isfinite(packed).all():
        raise ValueError("model must output finite float32 [1,3,H,W] at the input resolution")
    return pack_pbr(packed)


def pack_pbr(packed):
    rgb8 = np.rint(np.clip(packed[0].transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
    if rgb8.shape[2] == 3:
        alpha = np.full((*rgb8.shape[:2], 1), 255, dtype=np.uint8)
        rgb8 = np.concatenate((rgb8, alpha), axis=2)
    return Image.fromarray(rgb8)


def normal_features(normal):
    height, width = normal.shape[-2:]
    blurred = np.zeros_like(normal[0])
    if min(height, width) >= 3:
        neighborhoods = np.lib.stride_tricks.sliding_window_view(normal[0], (3, 3), axis=(1, 2))
        blurred[:, 1:-1, 1:-1] = np.median(neighborhoods, axis=(-2, -1))
    curvature = np.zeros((height, width), dtype=np.float32)
    if min(height, width) >= 3:
        nx, ny = np.lib.stride_tricks.sliding_window_view(blurred[:2], (3, 3), axis=(1, 2))
        sobel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        divergence = np.einsum('...ij,ij->...', nx, sobel) + np.einsum('...ij,ij->...', ny, sobel.T)
        curvature[1:-1, 1:-1] = np.abs(0.5 * divergence)
    rank99 = min(curvature.size - 1, int(curvature.size * 0.99))
    upper = max(np.partition(curvature.ravel(), rank99)[rank99], 1e-6)
    curvature = np.clip(curvature / upper, 0, 1) * 2 - 1

    x, y, z = blurred * 2 - 1
    gradients = np.clip(np.stack((x, y)) / np.maximum(z, 1e-3), -10, 10)
    relief = np.zeros_like(curvature)
    for row in range(1, height):
        relief[row, 0] = (relief[row - 1, 0] + gradients[1, row, 0]) * 0.5
    for col in range(1, width):
        relief[:, col] = (relief[:, col - 1] + gradients[0, :, col]) * 0.5
    span = np.ptp(relief)
    if span > 1e-6:
        relief = (relief - relief.min()) / span
    ranks = (int(relief.size * 0.01), rank99)
    low, high = np.partition(relief.ravel(), ranks)[list(ranks)]
    relief = (np.clip(relief, low, high) - low) / (high - low + 1e-6)
    smooth = np.zeros_like(relief)
    if min(height, width) >= 15:
        kernel = np.exp(-np.arange(-7, 8, dtype=np.float32) ** 2 / 50)
        kernel /= kernel.sum()
        horizontal = np.einsum('...i,i->...', np.lib.stride_tricks.sliding_window_view(relief, 15, axis=1), kernel)
        smooth[7:-7, 7:-7] = np.einsum('...i,i->...',
            np.lib.stride_tricks.sliding_window_view(horizontal, 15, axis=0), kernel)
    return np.stack((curvature, smooth * 2 - 1))[None]


def resize_mask(mask, height, width):
    rows, cols = mask.shape[-2:]
    y = np.arange(height, dtype=np.float32) * (rows / height)
    x = np.arange(width, dtype=np.float32) * (cols / width)
    y0, x0 = y.astype(int), x.astype(int)
    uy, ux = (y - y0).astype(np.float32)[:, None], (x - x0).astype(np.float32)[None, :]
    y1, x1 = np.minimum(y0 + 1, rows - 1), np.minimum(x0 + 1, cols - 1)
    top = mask[..., y0[:, None], x0] * (1 - ux) + mask[..., y0[:, None], x1] * ux
    bottom = mask[..., y1[:, None], x0] * (1 - ux) + mask[..., y1[:, None], x1] * ux
    return top * (1 - uy) + bottom * uy


def run_openxray_models(directory, diffuse, normal, sessions=None):
    if normal is None:
        raise ValueError("the OpenXRay model set requires --normal with an RGB tangent-space normal map")
    rgb, normals = read_rgb(diffuse), read_rgb(normal)
    if rgb.shape != normals.shape:
        raise ValueError("normal and diffuse dimensions must match")
    height, width = rgb.shape[:2]
    diffuse_nchw = rgb.transpose(2, 0, 1)[None]
    normal_nchw = normals.transpose(2, 0, 1)[None] * 2 - 1
    image = np.concatenate((diffuse_nchw * 2 - 1, normal_nchw), axis=1)
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]

    def infer(name, outputs, **inputs):
        session = model_session(directory / (name + ".onnx"), sessions)
        data = session.run(outputs, {key: np.ascontiguousarray(value) for key, value in inputs.items()})
        if any(tensor.dtype != np.float32 or not np.isfinite(tensor).all() for tensor in data):
            raise ValueError(f"{name} returned invalid float32 data")
        return data

    dirty, = infer("unet_albedo_uncond", ["albedo"], image=image)
    if dirty.shape != (1, 3, height, width):
        raise ValueError("unconditional albedo must match the diffuse resolution and RGB channels")
    logits, features = infer("segformer", ["logits", "features"],
                            input=np.concatenate(((dirty - mean) / std, normal_nchw), axis=1))
    if logits.ndim != 4 or logits.shape[:2] != (1, 10):
        raise ValueError("SegFormer must output ten material classes")
    albedo, = infer("unet_albedo", ["albedo"], image=image, seg_features=features)
    if albedo.shape != (1, 3, height, width):
        raise ValueError("albedo model must output three color channels at the input resolution")
    mask = np.exp(logits - logits.max(axis=1, keepdims=True))
    mask /= mask.sum(axis=1, keepdims=True)
    appearance = np.concatenate(((albedo - mean) / std, normal_nchw, resize_mask(mask, height, width)), axis=1)
    geometry = np.concatenate((normal_nchw, normal_features(normal_nchw)), axis=1)

    channels = []
    for name, inputs in (("metallic", appearance), ("roughness", appearance), ("ao", geometry), ("parallax", geometry)):
        values, = infer("unet_" + name, ["output"], image=inputs, seg_features=features)
        if values.shape != (1, 1, height, width):
            raise ValueError(f"{name} model must output [1,1,{height},{width}]")
        channels.append(1 / (1 + np.exp(-np.clip(values, -80, 80))))
    packed = pack_pbr(np.concatenate(channels, axis=1))
    with Image.open(diffuse) as original:
        alpha = original.convert("RGBA").getchannel("A")
    albedo8 = pack_pbr(albedo)
    albedo8.putalpha(alpha)
    parallax = packed.getchannel("A").convert("RGBA")
    return packed, albedo8, parallax


def write_dds(file, image, srgb=False):
    width, height = image.size
    mip_count = max(width, height).bit_length()
    # DDS + DX10, R8G8B8A8_UNORM, one 2D image. PBR channels stay linear.
    flags = 0x100F | (0x20000 if mip_count > 1 else 0)
    caps = 0x1000 | (0x400008 if mip_count > 1 else 0)
    dx10 = int.from_bytes(b"DX10", "little")
    header = [124, flags, height, width, width * 4, 0, mip_count, *([0] * 11),
              32, 4, dx10, 0, 0, 0, 0, 0, caps, 0, 0, 0, 0]
    file.write(b"DDS " + struct.pack("<31I", *header) + struct.pack("<5I", 29 if srgb else 28, 3, 0, 1, 0))
    linear_rgb = None
    if srgb and mip_count > 1:
        encoded = np.asarray(image, dtype=np.float32)[..., :3] / 255.0
        linear_rgb = np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)
    for level in range(mip_count):
        if level:
            size = (max(1, image.width // 2), max(1, image.height // 2))
            if linear_rgb is None:
                # Packed PBR channels, including parallax alpha, are already linear.
                image = Image.merge("RGBA", tuple(channel.resize(size, Image.Resampling.BOX) for channel in image.split()))
            else:
                linear_rgb = np.stack([
                    np.asarray(Image.fromarray(channel).resize(size, Image.Resampling.BOX))
                    for channel in linear_rgb.transpose(2, 0, 1)
                ], axis=-1)
                encoded = np.where(linear_rgb <= 0.0031308, linear_rgb * 12.92,
                                   1.055 * linear_rgb ** (1 / 2.4) - 0.055)
                alpha = image.getchannel("A").resize(size, Image.Resampling.BOX)
                image = Image.fromarray(np.rint(np.clip(encoded, 0, 1) * 255).astype(np.uint8))
                image.putalpha(alpha)
        file.write(image.tobytes())


def save_outputs(outputs, overwrite):
    staged = []
    published = []
    try:
        for destination, image, srgb in outputs:
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
                staged.append((Path(file.name), destination))
                write_dds(file, image, srgb)
        for temporary, destination in staged:
            if overwrite:
                os.replace(temporary, destination)
            else:
                os.link(temporary, destination)
                published.append((temporary, destination))
    except OSError:
        for temporary, destination in published:
            try:
                if os.path.samefile(temporary, destination):
                    destination.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("diffuse", type=Path, nargs="+")
    parser.add_argument("--normal", type=Path, action="append", help="repeat in diffuse input order for batches")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--albedo-output", type=Path)
    parser.add_argument("--parallax-output", type=Path, help="use <material>_parallax.dds for engine discovery")
    parser.add_argument("--export-albedo", action="store_true", help="write <material>_albedo.dds next to each diffuse input")
    parser.add_argument("--export-parallax", action="store_true", help="write <material>_parallax.dds next to each diffuse input")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    model_set = args.model.is_dir()
    if len(args.diffuse) > 1 and (args.output or args.albedo_output or args.parallax_output):
        parser.error("explicit output paths require a single diffuse input")
    if (model_set or args.normal) and len(args.normal or []) != len(args.diffuse):
        parser.error("provide one --normal per diffuse input")
    normals = args.normal or [None] * len(args.diffuse)
    destinations = []
    for diffuse in args.diffuse:
        base = diffuse.stem
        if len(base) > 2 and base.endswith("_d"):
            base = base[:-2]
        destinations.append((
            args.output or diffuse.with_name(base + "_pbr.dds"),
            args.albedo_output or (diffuse.with_name(base + "_albedo.dds") if args.export_albedo else None),
            args.parallax_output or (diffuse.with_name(base + "_parallax.dds") if args.export_parallax else None),
        ))
    if model_set and args.output_name is not None:
        parser.error("--output-name applies to single-model conversion")
    if not model_set and (args.albedo_output or args.parallax_output or args.export_albedo or args.export_parallax):
        parser.error("separate albedo and parallax exports require a model directory")
    paths = [path for outputs in destinations for path in outputs if path is not None]
    inputs = {path.resolve() for path in (args.model, *args.diffuse, *normals) if path is not None}
    for path in paths:
        if path.suffix.lower() != ".dds":
            parser.error("outputs must have a .dds extension")
        if path.resolve() in inputs:
            parser.error("outputs must be distinct and differ from every input")
        inputs.add(path.resolve())
        if path.exists() and not args.overwrite:
            parser.error(f"output exists: {path}")

    sessions = {} if len(args.diffuse) > 1 else None
    for diffuse, normal, (destination, albedo_path, parallax_path) in zip(args.diffuse, normals, destinations):
        outputs = []
        if model_set:
            packed, albedo, parallax = run_openxray_models(args.model, diffuse, normal, sessions)
            if albedo_path:
                outputs.append((albedo_path, albedo, True))
            if parallax_path:
                outputs.append((parallax_path, parallax, False))
        else:
            packed = run_model(args.model, diffuse, normal, args.output_name, sessions)
        outputs.append((destination, packed, False))
        save_outputs(outputs, args.overwrite)


if __name__ == "__main__":
    main()
