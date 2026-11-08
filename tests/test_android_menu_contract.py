"""Check the wiring between the shipped launcher, Lua menus, and engine options."""

from pathlib import Path
import json
import re
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
JAVA = ROOT / "android/apk/app/src/main/java/org/openxray/app"
ENGINE = ROOT / "src/Layers/xrRenderVK"


class AndroidMenuContractTest(unittest.TestCase):
    def test_vulkan_fxaa_menu_has_live_persisted_token_in_all_shipped_layouts(self):
        module = (ENGINE / "VulkanRendererModule.cpp").read_text()
        device = (ENGINE / "VulkanGameDevice.cpp").read_text()
        script = (ROOT / "res/gamedata/scripts/ui_mm_opt_video_adv.script").read_text()
        self.assertIn('aa("r_vk_aa", &antialiasing, aa_tokens)', module)
        self.assertIn('Console->AddCommand(&aa)', module)
        self.assertIn('render_world && antialiasing == 1', device)
        self.assertIn('pass->set_constants(frame_postprocess)', device)
        self.assertIn('xml:HasNode("video_adv:combo_vk_aa")', script)
        layouts = list((ROOT / "res/gamedata/configs/ui").rglob("ui_mm_opt*.xml"))
        found = 0
        for path in layouts:
            # Some legacy style files contain comments that strict XML parsers
            # reject. Check the added control as its own XML fragment.
            text = path.read_bytes().decode("latin1")
            if '<combo_ssample ' not in text:
                continue
            start = text.index('<combo_vk_aa ')
            end = text.index('</combo_vk_aa>', start) + len('</combo_vk_aa>')
            control = ET.fromstring(text[start:end])
            option = control.find("options_item")
            self.assertEqual(option.get("entry"), "r_vk_aa")
            self.assertIsNone(option.get("depend"))
            self.assertEqual(control.get("list_length"), "2")
            found += 1
        self.assertGreaterEqual(found, 4)

    def test_launcher_retains_display_controls_and_leaves_quality_to_the_game(self):
        launcher = (JAVA / "LauncherActivity.java").read_text()
        activity = (JAVA / "XRayActivity.java").read_text()
        for pref in ("PREF_RENDERER_MODE", "PREF_RENDER_RESOLUTION", "PREF_SHOW_FPS"):
            self.assertRegex(launcher, rf"\.put\w+\({pref},")
            self.assertRegex(launcher, rf"preferences\.get\w+\({pref},")
        self.assertIn('args.add("-android-render-width")', activity)
        self.assertNotIn('args.add("-android-mobile-preset")', activity)
        self.assertNotIn('args.add("-vk-detail-distance")', activity)
        self.assertNotIn('actionButton("Проверка Vulkan"', launcher)
        catalog = json.loads((ROOT / "android/apk/launcher-options.json").read_text())
        self.assertEqual(catalog["renderer"][2], {"label": "Vulkan", "arg": "-renderer-vulkan"})

    def test_vulkan_anisotropy_has_persisted_console_value_and_uses_sampler(self):
        module = (ENGINE / "VulkanRendererModule.cpp").read_text()
        device = (ENGINE / "VulkanGameDevice.cpp").read_text()
        self.assertIn('anisotropy("r__tf_aniso", &texture_anisotropy, 1, 16)', module)
        self.assertIn("Console->AddCommand(&anisotropy)", module)
        self.assertIn("float(texture_anisotropy)", device)
        self.assertNotIn("apply_vulkan_launch_settings", device)
        for filename in ("ui_mm_opt.xml", "ui_mm_opt_16.xml"):
            ui = ET.parse(ROOT / "res/gamedata/configs/ui" / filename)
            self.assertEqual(ui.find(".//options_item[@entry='r__tf_aniso']").get("depend"), "restart")

    def test_vulkan_advanced_menu_only_exposes_effective_controls(self):
        script = (ROOT / "res/gamedata/scripts/ui_mm_opt_video_adv.script").read_text()
        functions = {
            name: body
            for name, body in re.findall(r"function (\w+)\(render_id\)\s+(.*?)\s+end", script, re.DOTALL)
        }
        self.assertIn("render_id~=7", functions["desktop_modes"])
        self.assertIn("render_id==7", functions["vulkan_mode"])
        for name in ("mode_ge_2a", "mode_ge_2", "mode_ge_25", "mode_ge_3", "mode_ge_4"):
            self.assertIn("render_id~=7", functions[name])

        def precondition(control):
            match = re.search(
                rf'Init\w+\s*\("video_adv:{control}".*?table\.insert\(handler\.m_preconditions, '
                r'{func=(\w+)',
                script,
                re.DOTALL,
            )
            self.assertIsNotNone(match, control)
            return match.group(1)

        visible = {
            "track_vis_dist": "all_modes",
            "track_texture_lod": "all_modes",
            "track_aniso": "all_modes",
            "combo_vk_aa": "vulkan_mode",
            "track_detail_density": "all_modes",
            "track_detail_radius": "all_modes",
            "check_npc_torch": "all_modes",
            "check_always_active": "all_modes",
            "track_fps_limit": "all_modes",
            "track_fps_limit_in_menu": "all_modes",
        }
        for control, expected in visible.items():
            self.assertEqual(precondition(control), expected, control)

        hidden = {
            "track_geometry_lod": "desktop_modes",
            "track_ssample": "mode_less_3",
            "combo_ssample": "mode_ge_3",
            "track_light_distance": "mode_ge_2a",
            "track_particles_distance": "desktop_modes",
            "list_r2_smap_size": "desktop_modes",
            "check_r2_detail_bump": "mode_ge_2",
            "check_r2_steep_parallax": "mode_ge_25",
            "list_r2_sun_quality": "mode_ge_25",
            "combo_sun_shafts": "mode_ge_25",
            "radio_tab_ao_options": "mode_ge_25",
            "combo_ssao": "mode_ge_25",
            "check_soft_water": "mode_ge_25",
            "check_soft_particles": "mode_ge_25",
            "check_dof": "mode_ge_25",
            "check_volumetric_light": "mode_ge_25",
            "check_r3_dynamic_wet_surfaces": "mode_ge_3",
            "check_r3_volumetric_smoke": "mode_ge_3",
            "check_r3_msaa_opt": "mode_3",
            "check_vsync": "desktop_modes",
        }
        for control, expected in hidden.items():
            self.assertEqual(precondition(control), expected, control)

        self.assertRegex(
            script,
            r'(?s)InitCheck\("video_adv:check_r2_sun".*?return id>=2.*?control=_st',
        )
        self.assertRegex(
            script,
            r'(?s)InitCheck\("video_adv:check_r4_tessellation".*?mode_4\(id\) or id == 7.*?control=_st',
        )

    def test_vulkan_presets_cover_detail_distance_textures_and_shadows(self):
        settings = (ENGINE / "VulkanSettings.h").read_text()
        policy = (ENGINE / "RenderPolicy.h").read_text()
        module = (ENGINE / "VulkanRendererModule.cpp").read_text()
        self.assertIn("PresetDetailSpacing[]{1.f, .75f, .5f, .375f, .25f}", settings)
        self.assertIn("detail_spacing = PresetDetailSpacing[quality_preset]", settings)
        for assignment in (
            "detail_radius = quality.detail_distance",
            "detail_update_cells = quality.detail_updates",
            "point_shadow_budget = quality.point_shadows",
            "spot_shadow_budget = quality.spot_shadows",
            "texture_anisotropy = PresetAnisotropy[quality_preset]",
        ):
            self.assertIn(assignment, settings)
        self.assertIn("unsigned texture_limit", policy)
        self.assertIn("texture-limit=%u", module)
        self.assertIn("detail-spacing=%.3f", module)
        self.assertIn("point-shadows=%d", module)

    def test_combo_selection_and_restore_use_the_same_token_id(self):
        combo = (ROOT / "src/xrUICore/ComboBox/UIComboBox.cpp").read_text()
        self.assertIn("itm->SetTAG(_data)", combo)
        self.assertIn("get_token_id(tokens, current)", combo)
        self.assertIn("m_list_box.GetIdxByTAG(m_itoken_id)", combo)
        self.assertNotIn("m_list_box.SetSelectedText", combo)


if __name__ == "__main__":
    unittest.main()
