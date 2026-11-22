#!/usr/bin/env python3
"""Exercise batch publication with real ONNX inference and injected I/O errors."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import onnx
from onnx import TensorProto, helper
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "convert_pbr_onnx.py"
spec = importlib.util.spec_from_file_location("convert_pbr_onnx", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class BatchConversionTest(unittest.TestCase):
    def test_openxray_packing_truncates_model_outputs(self):
        channels = np.array([0.5, 0.999, 0.0], dtype=np.float32).reshape(1, 3, 1, 1)
        self.assertEqual(converter.pack_pbr(channels, round_to_nearest=False).getpixel((0, 0)), (127, 254, 0, 255))
        self.assertEqual(converter.pack_pbr(channels).getpixel((0, 0)), (128, 255, 0, 255))

    def test_openxray_input_resize_preserves_sampling_and_alpha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffuse = np.zeros((2, 3, 4), dtype=np.uint8)
            diffuse[..., 0] = [[10, 20, 30], [40, 50, 60]]
            diffuse[..., 3] = [[1, 2, 3], [4, 5, 6]]
            normal = np.zeros((4, 2, 3), dtype=np.uint8)
            normal[..., 1] = np.arange(8, dtype=np.uint8).reshape(4, 2)
            Image.fromarray(diffuse).save(root / "diffuse.png")
            Image.fromarray(normal).save(root / "normal.png")
            rgb, normal_rgb, alpha = converter.prepare_openxray_inputs(root / "diffuse.png", root / "normal.png")
            self.assertEqual(rgb.shape, (32, 32, 3))
            for y, x in ((0, 0), (15, 10), (16, 11), (31, 31)):
                sy, sx = y * 2 // 32, x * 3 // 32
                ny, nx = y * 4 // 32, x * 2 // 32
                self.assertAlmostEqual(float(rgb[y, x, 0]), float(diffuse[sy, sx, 0]) / 255, places=6)
                self.assertAlmostEqual(float(normal_rgb[y, x, 1]), float(normal[ny, nx, 1]) / 255, places=6)
                self.assertEqual(alpha[y, x], diffuse[sy, sx, 3])

            # The diffuse extent is retained when it exceeds the model floor;
            # the normal texture is resized even if it is larger.
            Image.new("RGBA", (40, 35), (0, 0, 0, 127)).save(root / "diffuse.png")
            Image.new("RGB", (60, 50), (128, 128, 255)).save(root / "normal.png")
            rgb, normal_rgb, alpha = converter.prepare_openxray_inputs(root / "diffuse.png", root / "normal.png")
            self.assertEqual(rgb.shape, (35, 40, 3))
            self.assertEqual(normal_rgb.shape, rgb.shape)
            self.assertTrue(np.all(alpha == 127))

    def test_openxray_model_pipeline_uses_prepared_dimensions_and_alpha(self):
        seen = []

        class Session:
            def __init__(self, name):
                self.name = name

            def run(self, outputs, inputs):
                image = next(iter(inputs.values()))
                seen.append((self.name, image.shape))
                height, width = image.shape[-2:]
                if self.name == "segformer":
                    return [np.zeros((1, 10, 1, 1), np.float32), np.zeros((1, 512, 1, 1), np.float32)]
                channels = 3 if self.name.startswith("unet_albedo") else 1
                value = .5 if channels == 3 else 0.
                return [np.full((1, channels, height, width), value, dtype=np.float32)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            diffuse = np.zeros((2, 3, 4), dtype=np.uint8)
            diffuse[..., 3] = [[1, 2, 3], [4, 5, 6]]
            Image.fromarray(diffuse).save(root / "diffuse.png")
            Image.new("RGB", (4, 5), (128, 128, 255)).save(root / "normal.png")
            with mock.patch.object(converter, "model_session", side_effect=lambda path, _: Session(path.stem)):
                packed, albedo, parallax = converter.run_openxray_models(root, root / "diffuse.png", root / "normal.png")
            self.assertEqual(packed.size, (32, 32))
            self.assertEqual(albedo.getpixel((31, 31)), (127, 127, 127, 6))
            self.assertEqual(packed.getpixel((0, 0)), (127, 127, 127, 127))
            self.assertEqual(parallax.getpixel((0, 0)), (127, 127, 127, 255))
            self.assertEqual(len(seen), 7)
            self.assertTrue(all(shape[-2:] == (32, 32) for _, shape in seen))

    def test_default_normal_features_match_openxray_flat_normal_contract(self):
        normal = np.zeros((1, 3, 32, 48), dtype=np.float32)
        normal[:, 2] = 1
        features = converter.normal_features(normal)
        # OpenXRay's Poisson input is normalized twice and its Gaussian
        # output leaves a seven-texel border at zero before mapping to [-1,1].
        np.testing.assert_allclose(features[0, :, 0, 0], [-1, -1], atol=1e-6)
        self.assertGreater(float(features[0, 1, 16, 24]), .99)
        self.assertLess(float(converter.normal_features_corrected(normal)[0, 1, 16, 24]), -.99)

    def test_flat_signed_normals_have_no_artificial_curvature_or_relief(self):
        for height, width in ((1, 1), (2, 7), (32, 48)):
            normal = np.zeros((1, 3, height, width), dtype=np.float32)
            normal[:, 2] = 1
            features = converter.normal_features_corrected(normal)
            self.assertEqual(features.shape, (1, 2, height, width))
            np.testing.assert_allclose(features, -1, atol=1e-6)

    def test_tilted_normals_produce_relief_without_zeroed_borders(self):
        normal = np.zeros((1, 3, 32, 48), dtype=np.float32)
        normal[:, 0] = .25
        normal[:, 2] = np.sqrt(1 - .25 ** 2)
        features = converter.normal_features_corrected(normal)[0]
        self.assertTrue(np.isfinite(features).all())
        np.testing.assert_allclose(features[0], -1, atol=1e-5)
        self.assertGreater(float(features[1, :, -1].mean()), .5)
        self.assertGreater(float(features[1, :, -1].mean()), float(features[1, :, 0].mean()) + 1)

    def test_later_inference_failure_does_not_publish_or_replace_earlier_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 2, 2])
            result = helper.make_tensor_value_info("pbr", TensorProto.FLOAT, [1, 3, 2, 2])
            graph = helper.make_graph([helper.make_node("Identity", ["image"], ["pbr"])],
                                      "identity", [value], [result])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            model.ir_version = 8
            onnx.save(model, root / "model.onnx")
            Image.fromarray(np.full((2, 2, 3), 127, dtype=np.uint8)).save(root / "one.png")
            Image.fromarray(np.full((3, 3, 3), 127, dtype=np.uint8)).save(root / "two.png")
            first = root / "one_pbr.dds"
            first.write_bytes(b"original")
            command = [sys.executable, str(SCRIPT), str(root / "model.onnx"),
                       str(root / "one.png"), str(root / "two.png"), "--overwrite"]
            failure = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("do not match", failure.stderr)
            self.assertEqual(first.read_bytes(), b"original")
            self.assertFalse((root / "two_pbr.dds").exists())

            Image.fromarray(np.full((2, 2, 3), 255, dtype=np.uint8)).save(root / "two.png")
            success = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(first.read_bytes()[:4], b"DDS ")
            self.assertEqual((root / "two_pbr.dds").read_bytes()[:4], b"DDS ")

    def test_publication_failure_restores_overwritten_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "one.dds", root / "two.dds"
            first.write_bytes(b"first original")
            second.write_bytes(b"second original")
            image = Image.new("RGBA", (2, 2), (1, 2, 3, 4))
            real_replace = converter.os.replace

            def fail_second(source, destination):
                if Path(destination) == second:
                    raise OSError("injected publication failure")
                real_replace(source, destination)

            with mock.patch.object(converter.os, "replace", side_effect=fail_second):
                with self.assertRaisesRegex(OSError, "injected"):
                    converter.save_outputs([(first, image, False), (second, image, False)], True)
            self.assertEqual(first.read_bytes(), b"first original")
            self.assertEqual(second.read_bytes(), b"second original")


if __name__ == "__main__":
    unittest.main()
