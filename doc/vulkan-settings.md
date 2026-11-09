# Vulkan graphics settings contract

DDS encodings and software texture fallback are described in
[the Vulkan texture contract](vulkan-textures.md).

The Vulkan renderer uses renderer ID `7`. The advanced graphics menu builds a
renderer-specific list in `ui_mm_opt_video_adv.script`; controls not listed as
supported below are hidden rather than presented as working options.

| Menu setting | Console value | Application point | Apply/persistence contract |
|---|---|---|---|
| Quality preset | `_preset` | `set_vulkan_quality`: texture limit and anisotropy, detail density/distance/update budget, local-shadow budgets and sun-shadow span | Restart; saved by `cfg_save` |
| Brightness / contrast / gamma | `rs_c_brightness`, `rs_c_contrast`, `rs_c_gamma` | Vulkan post-process constants | Live; saved by `cfg_save` |
| Visibility distance | `rs_vis_distance` | Environment far plane and particle distance | Next frame; saved by `cfg_save` |
| Texture quality | `texture_lod` | World-texture mip selection during upload | Restart; saved by `cfg_save` |
| Anisotropic filtering | `r__tf_aniso` | Vulkan sampler creation | Restart; saved by `cfg_save` |
| Anti-aliasing | `r_vk_aa` | Full-screen FXAA pass | Next accepted frame; saved by `cfg_save` |
| Detail density | `r__detail_density` | Detail instance grid spacing | Detail grid rebuild; saved by `cfg_save` |
| Detail distance | `r__detail_radius` | Per-frame detail preparation radius | Next frame; saved by `cfg_save` |
| Sun | `r2_sun` | Sun light intensity and shadow submission | Next frame; saved by `cfg_save` |
| NPC dynamic torches | `ai_use_torch_dynamic_lights` | AI inventory-owner torch light creation | Runtime game state; saved by `cfg_save` |
| Tessellation | `r4_enable_tessellation` | Tessellated level/model geometry selection during loading | Restart; saved by `cfg_save` |
| Always active / FPS limits | `rs_always_active`, `rs_fps_limit`, `rs_fps_limit_in_menu` | Shared engine scheduling | Runtime; saved by `cfg_save` |
| Resolution / window mode | `vid_mode`, `vid_window_mode` | Shared SDL device | Desktop only; launcher owns the Android render size |

On Android, the launcher resolution selects the 3D scene size. The G-buffer,
world water/refraction targets and post-process input use that size, including
the 3D weapon HUD and world-space UI. The swapchain and engine UI coordinate
system use the native surface size. Ordinary 2D UI is drawn after scene
postprocessing, with its own native-size snapshot for UI distortion. Screenshot
readback includes that final UI and uses the presentation size. The Java touch
controls also use the Android display size.

Tonemapping and scene effects precede native 2D UI. Gamma, brightness and
contrast apply once to the completed scene and UI, after UI distortion and
before screenshot readback. Neutral display correction skips the extra copy
and draw. The shared color-policy review remains open; this rendering split
alone does not validate linear/sRGB behavior or the appearance of every menu,
video, font and UI distortion effect.

The Vulkan preset levels are intentionally monotonic. Minimum through Extreme
increase the texture dimension limit, anisotropy, detail density, detail
distance and per-frame detail update budget, local-shadow budgets, and the
sun-shadow coverage span. Applying a preset logs all of these effective values
under the `[renderer-vulkan] quality` tag so a run can be audited without
inferring state from the menu.

Legacy desktop options that have no Vulkan implementation are hidden: desktop
MSAA, supersampling, geometry LOD, shadow-map size/quality, SSAO/HDAO/HBAO,
sun shafts, depth of field, volumetric lights and smoke, soft water/particles,
dynamic wet surfaces, light/particle distance, and the desktop VSync control.
Their console commands may still exist for other renderer modules, but changing
them is not advertised as a Vulkan feature.
