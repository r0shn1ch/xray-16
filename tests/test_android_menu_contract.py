"""Check the wiring between the shipped launcher, Lua menus, and engine options."""

from pathlib import Path
import json
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

    def test_combo_selection_and_restore_use_the_same_token_id(self):
        combo = (ROOT / "src/xrUICore/ComboBox/UIComboBox.cpp").read_text()
        self.assertIn("itm->SetTAG(_data)", combo)
        self.assertIn("get_token_id(tokens, current)", combo)
        self.assertIn("m_list_box.GetIdxByTAG(m_itoken_id)", combo)
        self.assertNotIn("m_list_box.SetSelectedText", combo)


if __name__ == "__main__":
    unittest.main()
