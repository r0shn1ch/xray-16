"""Check the shader binaries against the Vulkan game pipeline interfaces."""

import json
from pathlib import Path
import struct
import tempfile
import unittest
from zipfile import ZipFile

from tools.check_vulkan_shader_assets import check_apk, check_assets
from tools.compile_vulkan_shader import _manifest, _shader_metadata, _shader_source_metadata


ROOT = Path(__file__).resolve().parents[1]
SHADERS = ROOT / "res/gamedata/shaders"
MANIFEST = SHADERS / "vk/opaque-variants.json"


def instructions(path):
    data = path.read_bytes()
    if len(data) < 20 or len(data) % 4:
        raise ValueError(f"invalid SPIR-V length: {path}")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError(f"invalid SPIR-V magic: {path}")
    offset = 5
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 0xffff
        if not count or offset + count > len(words):
            raise ValueError(f"invalid instruction: {path}")
        yield opcode, words[offset + 1:offset + count]
        offset += count


class GameShaderVariantsTest(unittest.TestCase):
    def test_precompiled_assets_match_current_sources(self):
        self.assertEqual(check_assets(SHADERS, MANIFEST), [])

    def test_gpu_particle_vertex_bindings_and_outputs(self):
        code = list(instructions(SHADERS / "vk/gpu_particle_draw.vs.spv"))
        decorations = {(args[0], args[1]): args[2] for op, args in code if op == 71 and len(args) >= 3}
        bindings = {(decorations.get((identifier, 34)), value)
                    for (identifier, kind), value in decorations.items() if kind == 33}
        self.assertEqual(bindings, {(1, binding) for binding in range(6)})
        output_ids = {args[1] for op, args in code if op == 59 and args[2] == 3}
        locations = {decorations[(identifier, 30)] for identifier in output_ids
                     if (identifier, 30) in decorations}
        # Current motion-vector output carries current/previous clip positions
        # and a validity flag in addition to the original material varyings.
        self.assertEqual(locations, set(range(11)))
        fragment = list(instructions(SHADERS / "vk/gpu_particle_world_blended.ps.spv"))
        fragment_decorations = {(args[0], args[1]): args[2]
                                for op, args in fragment if op == 71 and len(args) >= 3}
        fragment_bindings = {(fragment_decorations.get((identifier, 34)), value)
                             for (identifier, kind), value in fragment_decorations.items() if kind == 33}
        self.assertIn((1, 6), fragment_bindings)
        self.assertFalse(any(op == 59 and args[2] == 9 for op, args in fragment))

    def test_manifest_entries_and_pipeline_interfaces(self):
        variants = json.loads(MANIFEST.read_text())["variants"]
        outputs = {v["output"]: v for v in variants}
        self.assertEqual(len(outputs), len(variants))
        expected = {
            "vk/level_opaque.vs.spv", "vk/level_opaque.ps.spv",
            "vk/object_opaque.vs.spv", "vk/object_opaque.ps.spv",
            "vk/level_cutout.ps.spv", "vk/object_cutout.ps.spv",
            "vk/object_blended.ps.spv", "vk/skinned_blended.ps.spv",
            "vk/object_priority_opaque.ps.spv", "vk/skinned_priority_opaque.ps.spv",
            "vk/hud_blended.ps.spv", "vk/object_double_sided.ps.spv",
            *{f"vk/skinned_{n}.vs.spv" for n in range(1, 5)},
            "vk/hud_skinned_1.vs.spv", "vk/hud_skinned_2.vs.spv",
            "vk/hud_skinned_3.vs.spv", "vk/hud_skinned_4.vs.spv", "vk/tree_opaque.vs.spv",
            "vk/progressive_opaque.vs.spv", "vk/level_tree_s.vs.spv",
            "vk/object_tree_s.vs.spv",
        }
        self.assertTrue(expected <= outputs.keys())
        for name, variant in outputs.items():
            with self.subTest(name=name):
                code = list(instructions(SHADERS / name))
                model = {"vs": 0, "ps": 4, "cs": 5}[variant["stage"]]
                self.assertTrue(any(op == 15 and args[0] == model for op, args in code))
                decorations = {(args[0], args[1]): args[2] for op, args in code
                               if op == 71 and len(args) >= 3}
                bindings = {identifier: value for (identifier, kind), value in decorations.items() if kind == 33}
                outputs_ids = {args[1] for op, args in code if op == 59 and args[2] == 3}
                output_locations = {decorations[(identifier, 30)] for identifier in outputs_ids
                                    if (identifier, 30) in decorations}
                locations = {value for (identifier, kind), value in decorations.items() if kind == 30}
                if name.endswith(".ps.spv"):
                    self.assertIn(0, output_locations)
                    if name not in ("vk/smoke_trail.ps.spv", "vk/trail.ps.spv"):
                        self.assertIn(0, bindings.values())
                    if name in ("vk/particle_world_deferred.ps.spv", "vk/gpu_particle_world_deferred.ps.spv"):
                        self.assertEqual(output_locations, {0, 1, 2})
                    elif (name in ("vk/smoke_trail.ps.spv", "vk/trail.ps.spv") or "blended" in name or "particle_" in name or "world_set_" in name or
                            "model_set_" in name or
                            "alpha_add" in name or
                            "environment.ps" in name and name.startswith(("vk/object_", "vk/skinned_")) or
                            name in ("vk/wallmark_multiply_2x.ps.spv", "vk/level_fog.ps.spv",
                                     "vk/level_priority_opaque.ps.spv",
                                     "vk/level_priority_detail.ps.spv",
                                     "vk/level_priority_detail_lightmap.ps.spv",
                                     "vk/level_priority_lightmap.ps.spv",
                                     "vk/object_priority_opaque.ps.spv",
                                     "vk/skinned_priority_opaque.ps.spv",
                                     "vk/hud_model_environment.ps.spv")):
                        # The forward pass has one color attachment.
                        self.assertEqual(output_locations, {0})
                    else:
                        self.assertEqual(output_locations, {0, 1, 2})
                if "skinned" in name and name.endswith(".vs.spv"):
                    self.assertTrue({0, 1, 2, 3, 4} <= locations)
                    self.assertIn(0, bindings.values())
                    self.assertIn(1, {value for (identifier, kind), value in decorations.items()
                                      if kind == 34})  # descriptor set for pose
                if name in ("vk/level_tree_s.vs.spv", "vk/object_tree_s.vs.spv"):
                    self.assertTrue({0, 1, 4} <= output_locations)
                    inputs = {args[1] for op, args in code if op == 59 and args[2] == 1}
                    input_locations = {decorations[(identifier, 30)] for identifier in inputs
                                       if (identifier, 30) in decorations}
                    self.assertTrue({0, 1, 2} <= input_locations)
                    if name.startswith("vk/level_"):
                        self.assertTrue({2, 3} <= output_locations)
                        self.assertTrue({5, 6} <= input_locations)

    def test_cutout_discards_but_blended_does_not(self):
        for variant in ("level_cutout", "object_cutout"):
            code = list(instructions(SHADERS / f"vk/{variant}.ps.spv"))
            opcodes = {op for op, _ in code}
            self.assertIn(252, opcodes)  # OpKill
            offsets = {args[3] for op, args in code
                       if op == 72 and len(args) >= 4 and args[2] == 35}
            self.assertIn(64, offsets)  # The material cutoff follows the MVP.
        blended = {op for op, _ in instructions(SHADERS / "vk/object_blended.ps.spv")}
        self.assertNotIn(252, blended)
        for name in ("particle_world_set", "particle_hud_set"):
            code = list(instructions(SHADERS / f"vk/{name}.ps.spv"))
            self.assertIn(252, {op for op, _ in code})
            offsets = {args[3] for op, args in code
                       if op == 72 and len(args) >= 4 and args[2] == 35}
            self.assertIn(64, offsets)

    def test_forward_transparent_layouts(self):
        for name, expected_set in (("object_blended", 1), ("skinned_blended", 2),
                ("object_priority_opaque", 1), ("skinned_priority_opaque", 2),
                ("level_detail_blended", 1), ("level_environment_blended", 1)):
            code = list(instructions(SHADERS / f"vk/{name}.ps.spv"))
            sets = {args[2] for op, args in code if op == 71 and len(args) >= 3 and args[1] == 34}
            self.assertIn(expected_set, sets)
            offsets = {args[3] for op, args in code if op == 72 and len(args) >= 4 and args[2] == 35}
            self.assertTrue({0, 64, 80, 96, 112, 128, 144, 160, 176} <= offsets)
            decorations = {(args[0], args[1]): args[2] for op, args in code
                           if op == 71 and len(args) >= 3}
            weather_bindings = {binding for (identifier, kind), binding in decorations.items()
                                if kind == 33 and decorations.get((identifier, 34)) == expected_set + 1}
            self.assertTrue({4, 5} <= weather_bindings)
        hud = list(instructions(SHADERS / "vk/hud_blended.ps.spv"))
        self.assertFalse(any(op == 71 and len(args) >= 3 and args[1] == 34 and args[2] != 0
                             for op, args in hud))

    def test_model_hemi_cube_survives_vertex_to_fragment_contract(self):
        for vertex_name, fragment_name in (("object_blended.vs.spv", "object_blended.ps.spv"),
                                           ("skinned_4.vs.spv", "skinned_blended.ps.spv")):
            vertex = list(instructions(SHADERS / "vk" / vertex_name))
            fragment = list(instructions(SHADERS / "vk" / fragment_name))
            for code in (vertex, fragment):
                locations = {args[2] for op, args in code
                             if op == 71 and len(args) >= 3 and args[1] == 30}
                self.assertIn(5, locations)
            offsets = {args[3] for op, args in vertex
                       if op == 72 and len(args) >= 4 and args[2] == 35}
            if vertex_name.startswith("skinned"):
                self.assertEqual(offsets, {0})  # model/hemi data live in the pose SSBO
                self.assertIn(68, {op for op, _ in vertex})  # OpArrayLength finds the trailer
            else:
                self.assertIn(128, offsets)
                self.assertIn(240, offsets)  # quaternion cannot overlap fragment parameters

    def test_opaque_model_environment_uses_cube_and_surface_attachment(self):
        for name in ("object_environment_opaque", "skinned_environment_opaque"):
            with self.subTest(name=name):
                code = list(instructions(SHADERS / f"vk/{name}.ps.spv"))
                bindings = {args[2] for op, args in code
                            if op == 71 and len(args) >= 3 and args[1] == 33}
                offsets = {args[3] for op, args in code
                           if op == 72 and len(args) >= 4 and args[2] == 35}
                locations = {args[2] for op, args in code
                             if op == 71 and len(args) >= 3 and args[1] == 30}
                self.assertIn(2, bindings)  # The environment descriptor.
                self.assertTrue(any(op == 25 and args[2] == 3 for op, args in code))  # Cube image.
                self.assertTrue({0, 1, 2} <= locations)  # Albedo, normal and surface.
                self.assertTrue({64, 96, 112} <= offsets)  # Camera and material channels.

    def test_world_set_alpha_reference_and_color_scale(self):
        for name in ("world_set_blended", "world_set_set_2r", "world_set_blend_2r",
                     "world_set_blend_4r", "glow_alpha_add"):
            with self.subTest(name=name):
                code = list(instructions(SHADERS / f"vk/{name}.ps.spv"))
                self.assertIn(252, {op for op, _ in code})  # Alpha test discards.
                offsets = {args[3] for op, args in code
                           if op == 72 and len(args) >= 4 and args[2] == 35}
                self.assertIn(64, offsets)  # Cutoff set by the draw recorder.
        self.assertNotEqual((SHADERS / "vk/world_set_blend_2r.ps.spv").read_bytes(),
                            (SHADERS / "vk/world_set_blend_4r.ps.spv").read_bytes())
        fog = list(instructions(SHADERS / "vk/level_fog.ps.spv"))
        decorations = {(args[0], args[1]): args[2] for op, args in fog
                       if op == 71 and len(args) >= 3}
        self.assertTrue(any(decorations.get((identifier, 33)) == 0 and set_id == 1
                            for (identifier, kind), set_id in decorations.items() if kind == 34))
        self.assertIn(252, {op for op, _ in fog})  # S_SET BLEND keeps serialized alpha test.
        self.assertNotEqual((SHADERS / "vk/level_fog.ps.spv").read_bytes(),
                            (SHADERS / "vk/world_set_blended.ps.spv").read_bytes())

    def test_model_set_has_no_level_only_fragment_inputs(self):
        for variant in json.loads(MANIFEST.read_text())["variants"]:
            if not variant["output"].startswith(("vk/model_set_", "vk/hud_model_set_")):
                continue
            with self.subTest(name=variant["output"]):
                code = list(instructions(SHADERS / variant["output"]))
                locations = {args[0]: args[2] for op, args in code
                             if op == 71 and len(args) >= 3 and args[1] == 30}
                inputs = {args[1] for op, args in code
                          if op == 59 and len(args) >= 3 and args[2] == 1}
                self.assertFalse({2, 3} & {locations[i] for i in inputs if i in locations})

    def test_apk_contains_every_game_pipeline_asset(self):
        variants = _manifest(MANIFEST, SHADERS)

        def add_pipeline_assets(archive, include_sources, include_dependencies):
            written = set()
            for variant in variants:
                output = variant["output"]
                data = output.read_bytes()
                output_name = "assets/gamedata/shaders/" + output.relative_to(SHADERS).as_posix()
                archive.writestr(output_name, data)
                archive.writestr(output_name + ".meta", _shader_metadata(variant, data))
                if not include_sources:
                    continue
                for line in _shader_source_metadata(variant):
                    if line.startswith("source="):
                        vfs_path = line.split("=", 1)[1]
                    elif include_dependencies and line.startswith("dependency="):
                        vfs_path = line[len("dependency="):].rsplit("=", 1)[0]
                    else:
                        continue
                    path = Path(vfs_path)
                    if path.is_absolute() or ".." in path.parts or "." in path.parts:
                        continue  # Engine-internal sources are loaded from the binary.
                    asset = "assets/gamedata/shaders/" + path.as_posix()
                    if asset not in written:
                        archive.write(SHADERS.joinpath(*path.parts), asset)
                        written.add(asset)

        with tempfile.TemporaryDirectory() as directory:
            apk = Path(directory) / "game.apk"
            with ZipFile(apk, "w") as archive:
                archive.writestr("lib/armeabi-v7a/libmain.so", b"test native library")
                add_pipeline_assets(archive, include_sources=True, include_dependencies=True)
            self.assertEqual(check_apk(apk, MANIFEST), [])

            with ZipFile(apk, "w") as archive:
                archive.writestr("lib/armeabi-v7a/libmain.so", b"test native library")
                add_pipeline_assets(archive, include_sources=False, include_dependencies=False)
            missing_runtime = check_apk(apk, MANIFEST)
            self.assertIn("missing runtime shader source: assets/gamedata/shaders/vk/opaque.ps",
                          missing_runtime)
            self.assertIn("missing runtime shader dependency: assets/gamedata/shaders/vk/LightingLimits.h",
                          missing_runtime)

            with ZipFile(apk, "w") as archive:
                archive.writestr("lib/armeabi-v7a/libmain.so", b"test native library")
            self.assertIn("missing assets/gamedata/shaders/vk/level_opaque.vs.spv",
                          check_apk(apk, MANIFEST))


if __name__ == "__main__":
    unittest.main()
