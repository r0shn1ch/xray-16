# Android port audit

This audit covers the Android port through version 0.9.14. The comparison base
is upstream OpenXRay `dev` at `247d72764`. The Android branch already contains
the two upstream UI commits that follow the fork's older `origin/dev`, so they
are not Android-specific replacements.

## Compatibility invariants

- Original game and mod resources are read-only inputs. The launcher and engine
  must not rewrite `fsgame.ltx`, shaders, textures or archives. The standard
  `<STALKER>/_appdata_` subtree is deliberately writable and owns `user.ltx`,
  saves, screenshots and normal engine logs, as it does on desktop.
- The tracked `res/gamedata` resource tree stays identical to upstream. Android
  compatibility is implemented in engine code and in-memory conversion only.
- Disk formats used by the three PC games remain unchanged. ARM fixes copy
  unaligned packed values instead of changing serialized layouts.
- Hardware selection is capability-based. There are no GPU model, vendor, or
  driver-version allowlists in the Android renderer.
- Platform code is guarded by `XR_PLATFORM_ANDROID`; desktop behavior is not
  silently replaced.

## Result

No tracked upstream file was deleted by the port, and `res/` has no difference
from upstream. The audit found no unexplained feature removal to restore.

The retained changes fall into these categories:

| Area | Reason retained | Resource impact |
|---|---|---|
| ARMv7 build and dependency fixes | Required to compile the existing engine and LuaJIT with the NDK | None |
| Android filesystem bootstrap | Mounts the selected PC installation and verifies its normal `_appdata_` path | Resources read-only; `_appdata_` writable |
| SDL Activity and diagnostics | Owns the native surface, lifecycle, logs, orientation, and process isolation | None |
| GLES context and framebuffer presentation | Adapts SDL's EGL framebuffer to the existing deferred renderer | None |
| Texture upload compatibility | Decodes unsupported desktop compression in memory when required | Source DDS files unchanged |
| Shader compatibility | Transforms the in-memory compiler input; original and mod shader files are not edited | Source files unchanged |
| Touch input | Emits normal engine keyboard/mouse actions and is optional | No UI asset copies |

## Findings addressed in 0.8.0

- Reattach intents explicitly select game mode instead of inheriting the old
  renderer-smoke default.
- The engine Activity is landscape before SDL creates its first surface.
- Background/foreground events now reach the engine, and the GLES context is
  rebound to SDL's recreated EGL surface before presentation resumes.
- `game.graph` and embedded cross-table records are treated as packed disk
  data. Multi-byte values are copied out before use, preventing ARMv7
  `SIGBUS/BUS_ADRALN` without changing the PC file format.
- GLES startup checks the reported ES version, draw-buffer count, and color
  attachment count. Unsupported devices receive an explicit error instead of
  a vendor-specific workaround.
- Shader compiler logs are length-bounded and NUL-terminated; future failures
  no longer appear as corrupted binary text.
- Build scripts derive archive versions from `PORT_VERSION` and discover a
  reusable build kit instead of requiring a hard-coded historical version.

## Findings addressed in 0.9.0

- The touch overlay exposes an `ESC` control and routes it through the same
  `kQUIT` binding as a physical keyboard Escape key.
- Android game launches disable object prefetch, compact memory after the
  previous session is unloaded, and redraw the loading surface during large
  ALife spawn passes.
- The launcher exposes an Android Vulkan surface/device/swapchain/present
  probe with automatic fallback to the existing GLES smoke path. This is a
  VK0 bring-up check, not yet the complete gameplay renderer.

## Findings addressed in 0.9.8

- Android rendering can use an internal resolution independent from the EGL
  drawable. The default 50% scale quarters pixel load, presentation uses a
  filtered upscale, and input coordinates follow the internal size.
- Launcher-controlled low/default mobile presets are applied after `user.ltx`,
  making an accidentally retained desktop high preset unable to override the
  selected mobile performance mode.
- The FPS toggle uses OpenXRay's frame statistics, while periodic lightweight
  frame traces make CPU/render bottlenecks visible in device logs.
- Vulkan selection now runs the real VK0 probe during a game launch, records
  limits/features/deferred attachment formats, and explicitly identifies the
  subsequent GLES gameplay fallback.

## Findings addressed in 0.9.9

- Graphics preset and internal resolution are independent launcher settings.
  Auto uses the minimum preset and an aspect-correct 1280-pixel-wide target on
  larger displays instead of inheriting desktop High/Extreme at native phone
  resolution.
- The launcher offers concrete low-through-native resolution choices. The EGL
  drawable remains native and landscape while the engine-owned 3D targets use
  the chosen size, preserving SDL lifecycle and orientation behavior.
- Launcher overrides are applied after reading `user.ltx` and before game
  startup; resource files stay unchanged and normal engine persistence remains
  under `_appdata_`.

## Findings addressed in 0.9.10

- The Android shader-cache key now includes the GLES compatibility-pass
  revision, preventing program binaries from older APKs from being linked
  against newly translated shader stages.
- Minimum-quality shaders explicitly define zero-valued quality macros for
  strict Adreno preprocessors instead of accidentally entering unsupported
  ultra-shadow code and falling back to an interface-incompatible stub.
- Software-decoded 1K and 2K mipmapped textures start two mips lower on
  32-bit Android. This preserves UV and atlas geometry while reducing the
  first gameplay frame's decode time and RGBA working set.
- Compact crash records now include the actual shared-object name and
  module-relative program-counter offset when the Android linker can resolve
  them, allowing subsequent native faults to be symbolized from user logs.

## Findings addressed in 0.9.11

- BC1/BC2/BC3 fallback decoding now expands each compressed block once instead
  of invoking GLI's per-texel decoder sixteen times. This removes the roughly
  one-second stalls previously seen for individual 1024x1024 textures on ARM.
- Android lazy texture loading once again runs the engine's normal rotating
  visible-set precache behind the loading screen. It does not return to the
  unsafe all-resource deferred upload, but it prepares the current level before
  exposing gameplay and restores the stock new-game intro trigger.
- The legacy Minimum preset is completed in memory for old CoP configurations
  that do not reset newer SSR settings. Other presets still expose all renderer
  features, and original configuration files remain read-only.
- Invalid GLES sampler-object parameters for desktop LOD bias/max-level state
  are no longer submitted. Frame diagnostics now state when detailed engine
  timers are disabled instead of silently reporting misleading zero values.
- Zero-quality sun shafts no longer define `SUN_SHAFTS_QUALITY=0`, which entered
  the volumetric ray-march without a `RAY_SAMPLES` definition and broke the
  shader on Adreno.

## Findings addressed in 0.9.12

- ReleaseMasterGold received explicit compiler optimization instead of the
  accidental `-O0` Android build. Version 0.9.14 refines the Android baseline
  to `-O2` with LTO opt-in while startup stability is validated.
- Minimum explicitly disables sun, detail and TSM shadows and uses the correct
  `r2_smap_size` command. Other presets retain the complete shadow feature set.
- The launcher FPS overlay is opaque red and centered at the top. Frame traces
  now expose real update/render/wait, present/swap, draw-call and polygon data.
- Disabled SSR is explicitly compiled as quality zero on Android, avoiding the
  strict-preprocessor failure in the water shader without any vendor check.

## Findings addressed in 0.9.13

- The focused GLES validation set now compiles and links the soft-water and
  soft-water-depth programs, including the strict `SSR_QUALITY=0` path seen in
  physical-device logs.
- Activity diagnostics record the APK version and version code at startup so a
  log can be matched to its exact installed package before engine bootstrap.

## Findings addressed in 0.9.14

- The temporary app-private `$app_data_root$` override is removed. Android once
  again follows `fsgame.ltx`, so `user.ltx`, saves, screenshots and normal logs
  live under `<STALKER>/_appdata_`.
- The launcher creates the standard writable subdirectories and performs a real
  write/delete probe before native startup. A scoped-storage denial is reported
  in the launcher instead of surfacing later as a save failure.
- Renderer-wide experiments added after the last confirmed playable build are
  removed from the Android baseline: GLES uses the established
  `glVertexAttribPointer` state path, retains the established final postprocess
  transition, and does not invalidate the just-presented framebuffer.
- Android ReleaseMasterGold uses `-O2` with LTO disabled by default. This keeps
  the large gain over the accidental `-O0` build while avoiding two aggressive
  compiler changes in the startup-regression range. ARM mode and optional LTO
  remain explicit, logged build choices.

## Remaining renderer debt

`AndroidGlslCompatRules.inl` is an engine-side bridge for strict GLSL ES
compilers. It preserves resources on disk, but exact source-line rewrites are
not a sufficiently general contract for arbitrary graphics mods. It is kept
for the working GLES path until a source compiler with type-aware HLSL-to-SPIR-V
translation replaces it. New fixes must prefer capability checks and general
syntax/interface normalization; new GPU-model rules or asset patches are not
acceptable.

The Vulkan work and its acceptance gates are documented in
`VULKAN_RENDERER_PLAN.md`. A renderer is not considered implemented merely
because a stub module, Vulkan instance, or clear-screen test exists.
