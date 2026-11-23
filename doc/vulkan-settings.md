# Vulkan graphics settings contract

DDS encodings and software texture fallback are described in
[the Vulkan texture contract](vulkan-textures.md).

The Vulkan renderer uses renderer ID `7`. The advanced graphics menu builds a
renderer-specific list in `ui_mm_opt_video_adv.script`; controls not listed as
supported below are hidden rather than presented as working options.

Desktop builds register the Vulkan module alongside the existing renderers.
Select it by saving `renderer renderer_vulkan` in the configuration passed to
`-ltx` before starting the engine. Confirm `Selected renderer: renderer_vulkan`
and `[renderer-vulkan] module.selected` in the startup log before using a run
as Vulkan evidence. Keep a separate configuration for OpenGL comparisons.

| Setting | Console value | Application point | Apply/persistence contract |
|---|---|---|---|
| Quality preset | `_preset` | `set_vulkan_quality`: texture limit and anisotropy, detail density/distance/update budget, local-shadow budgets, sun-shadow span, ambient-occlusion directions (4/8/8/12/16), and sun-shaft depth samples (8/12/16/24/32); after the command runs, the advanced controls reread their effective console values | Restart; saved by `cfg_save` |
| Brightness / contrast / gamma | `rs_c_brightness`, `rs_c_contrast`, `rs_c_gamma` | Vulkan post-process constants | Live; saved by `cfg_save` |
| Visibility distance | `rs_vis_distance` | Environment far plane and particle distance | Next frame; saved by `cfg_save` |
| Texture quality | `texture_lod` | World-texture mip selection during upload; runtime changes stage replacement uploads for resident world textures and refresh material/color-map/weather/ray descriptors | Next safe resource refresh; saved by `cfg_save` |
| Anisotropic filtering | `r__tf_aniso` | Vulkan world sampler creation | Safe renderer reset; saved by `cfg_save` |
| Texture mip bias | `r_vk_texture_mip_bias`, `r1_tf_mipbias`, `r2_tf_mipbias` | World sampler LOD bias, clamped by the physical device; UI sampling keeps zero bias | Safe renderer reset; saved by `cfg_save` |
| Anti-aliasing | `r_vk_aa` | `off`, `FXAA`, or temporal color resolve (`TAA`); falls back to FXAA if temporal resources are unavailable | Next accepted frame; saved by `cfg_save` |
| Bloom strength, threshold, radius and soft knee | `r_vk_bloom_intensity`, `r_vk_bloom_threshold`, `r_vk_bloom_mip`, `r_vk_bloom_knee` | Vulkan scene postprocess; bright pass precedes downsampling and radius controls the upsample filter | Next gameplay frame; saved by `cfg_save` |
| Legacy bloom aliases | `r2_ls_bloom_threshold`, `r2_ls_bloom_kernel_scale`, `r2_ls_bloom_kernel_g`, `r2_ls_bloom_kernel_b`, `r2_ls_bloom_speed`, `r2_ls_bloom_fast` | Map legacy threshold and speed into the Vulkan bloom pass; kernel scale controls intensity, the two kernel values jointly adjust upsample radius, and `fast` selects five-tap upsampling | Next gameplay frame; saved by `cfg_save` |
| Tonemap adaptation rate | `r_vk_tonemap_adaptation`, `r2_tonemap_adaptation`, `r2_tonemap_lowlum` | GPU exposure from a trimmed log-luminance histogram; legacy aliases control adaptation speed and the luminance floor | Next gameplay frame; saved by `cfg_save` |
| Dynamic model detail and culling | `r__ssaDISCARD`, `r__geometry_lod`, `r2_ssa_lod_a` | Uses scene-resolution, FOV-adjusted screen-space area to discard tiny world models and select external `OGF_S_LODS` models for distant skeletons | Next gameplay frame; saved by `cfg_save` |
| Wallmark lifetime, fade and duplicate replacement | `r__wallmark_ttl` | Wallmarks fade in over their lifetime and consume TTL at one tenth speed while inside the camera frustum; a static mark with the same texture within 2 cm replaces the earlier mark | New wallmarks use the current value; saved by `cfg_save` |
| Ambient occlusion strength | `r_vk_ao_strength` | Screen-space depth/normal samples attenuate ambient and diffuse indirect light in both lighting paths; zero disables AO | Next gameplay frame; saved by `cfg_save` |
| Experimental RT indirect strength | `r_vk_rtgi_strength` | Scales the current diffuse RT bounce on supported devices; zero skips its dispatch | Next gameplay frame; saved by `cfg_save`. The menu exposes it only while level RT resources are ready |
| Depth of field strength | `r_vk_dof_strength` | Scene compositor samples copied G-buffer depth; zero disables the eight-tap blur and ordinary 2D UI is drawn afterward | Next gameplay frame; saved by `cfg_save` |
| Depth of field focus distance (console) | `r_vk_dof_focus` | World-space distance in metres projected through the current view-projection matrix | Next gameplay frame; saved by `cfg_save` |
| Motion blur strength (console) | `r_vk_motion_blur` | Samples current camera/object motion vectors along a depth-rejected, capped screen-space path; requires the motion-vector pass, and zero disables the extra samples | Next gameplay frame; saved by `cfg_save` |
| Sun shafts strength | `r_vk_sun_shafts_strength` | Additive screen-space rays from the projected sun, masked by copied scene depth; zero skips the pass | Next gameplay frame; saved by `cfg_save` |
| Upscaling | `r_vk_upscaler` | Bilinear, clamped cubic, experimental temporal reconstruction or Android/Vulkan FSR 3.1 when the scene is smaller than the presentation surface; UI remains native resolution | `vid_restart`; saved by `cfg_save` |
| Frame generation | `r_vk_frame_generation` | Menu control is shown only in Android FSR builds when the selected device exposes the required extra queues. Opt-in FSR 3 interpolation also requires `r_vk_upscaler FSR 3.1` and compatible surface image usages; failed activation falls back to off. | `vid_restart`; saved by `cfg_save` |
| Experimental temporal upscaling launch override | `-vk-temporal-upscale` with `r_vk_aa TAA` | Allocates presentation-resolution TAA history when the scene is smaller; falls back to scene-resolution TAA if allocation fails | Launch flag |
| Volumetric local lights | `r2_volumetric_lights` | Enables scattering for lights marked volumetric; direct light remains when disabled | Next gameplay frame; saved by `cfg_save` |
| Rain-driven wet surfaces | `r3_dynamic_wet_surfaces`, `r3_dynamic_wet_surfaces_near`, `r3_dynamic_wet_surfaces_far`, `r3_dynamic_wet_surfaces_sm_res` | Vulkan samples the stabilized rain shadow map for wet darkening and surface response. Near/far and map resolution controls are available in the advanced menu; resolution changes rebuild rain-shadow targets after waiting for GPU work to finish | Toggle and falloff apply on the next frame; resolution rebuilds at render entry; saved by `cfg_save` |
| Detail density | `r__detail_density` | Detail instance grid spacing | Detail grid rebuild; saved by `cfg_save` |
| Detail distance | `r__detail_radius` | Per-frame detail preparation radius | Next frame; saved by `cfg_save` |
| Sun | `r2_sun` | Sun light intensity and shadow submission | Next frame; saved by `cfg_save` |
| NPC dynamic torches | `ai_use_torch_dynamic_lights` | AI inventory-owner torch light creation | Runtime game state; saved by `cfg_save` |
| Tessellation | `r4_enable_tessellation` | Tessellated level/model geometry selection during loading | Restart; saved by `cfg_save` |
| Always active / FPS limits | `rs_always_active`, `rs_fps_limit`, `rs_fps_limit_in_menu` | Shared engine scheduling | Runtime; saved by `cfg_save` |
| Resolution / window mode | `vid_mode`, `vid_window_mode` | Shared SDL device | Desktop only; launcher owns the Android render size |

GPU detail generation selects resident cells by range and frustum, then builds
an indirect generation dispatch and indirect instance draws without a CPU
counter readback. Empty selections clear stale placements. Each frame slot owns
its selection and dispatch buffers. Ground-triangle streaming and transparent
detail sorting still use the CPU; this does not complete the upstream grass
system, including its wind and interaction paths.

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
sun-shadow coverage span. Sun-shaft depth samples increase from 8 to 32, and
ambient-occlusion directions increase from 4 to 16 while the Default preset
keeps the existing 8-direction cost. Applying a preset logs its effective
values under the `[renderer-vulkan] quality` tag so a run can be audited
without inferring state from the menu.

Legacy desktop options that have no Vulkan implementation are hidden: desktop
MSAA, supersampling, shadow-map size/quality, legacy SSAO/HDAO/HBAO controls,
legacy sun-shaft and depth-of-field controls, volumetric smoke, soft water/particles,
light/particle distance, and the desktop VSync control.
Their console commands may still exist for other renderer modules, but changing
them is not advertised as a Vulkan feature.
