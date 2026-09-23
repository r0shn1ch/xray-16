# Android launcher and renderer recovery plan

Baseline: fork `dev` at `e3e891ff`, upstream OpenXRay `dev` at `a7055a4b`;
common ancestor `247d7276` (49 fork commits, eight upstream commits). Both sides
changed 11 of the same paths, especially CMake configuration and
`src/xrCore/LocatorAPI.cpp`; these need an explicit merge test.
Device log: 2026-09-23, Adreno 710, CoP, 1280×576 internal resolution.
The existing [Vulkan plan](VULKAN_RENDERER_PLAN.md) defines VK0–VK6.

## Evidence and acceptance gates

| Work | Evidence from the attached run | Required acceptance |
| --- | --- | --- |
| Process lifecycle | `activity onDestroy` precedes `Destroying Render...`; the launcher signals PID 26030 after shutdown stalls | 20 cycles of game → launcher → game and 20 normal quits; no second native entry point and no `:engine` PID after each normal quit. Force stop terminates a deliberately stalled engine; do not report success until the PID actually disappears. |
| Configuration | Launcher repeats labels, preset tokens and command-line switches in Java; engine overrides Minimum with console commands | A single catalog generates labels and arguments; Android-specific renderer adjustments live in an APK-owned `.ltx` and do not modify original resources or `user.ltx`. |
| Image correctness | Minimum explicitly issued `r2_sun off`; GL reports repeated `0x500` errors before Present; floors and walls disappear intermittently | Capture identical camera/time frames on PC and Android. Verify sunlight and every static surface in a scene with portals. Locate the *first* GLES error via per-pass diagnostics and fix its generating operation; preserve a fallback build for comparison. |
| Frame pacing | 16–25 FPS gameplay; `update` commonly 24–50 ms and `render` 9–27 ms, with 75–456 ms worst frames | Record median/p95/p99 frame times, update/render time, allocations and lazy texture uploads across a fixed 60-second route. Profile update hot spots and remove stalls only after correctness parity. Compare same game data and resolution. |
| Vulkan | VK0 surface/device/swapchain/present passes, gameplay selects GLES | VK1–VK6 pass independently using Vulkan objects only; validation layers have no errors. No renderer selection claims Vulkan gameplay until VK3 works. |
| Upstream | Current tip differs from upstream in 128 paths (+17,250/−308 lines); much of this is old patch snapshots | Keep Android build, UI and preset files under `android/`; isolate unavoidable platform hooks behind `XR_PLATFORM_ANDROID`. Test an upstream merge in a temporary branch before advancing `dev`; measure real conflicts and avoid editing upstream game assets. |

## Sequence

1. **Lifecycle and configuration (started in this PR).** Give the engine process ownership of its own normal termination when native `main` completes; route both Android Back and SDL's manual back hook into the same task transition; bring the existing SDL Activity forward through Android task reuse; retry forced shutdown only for the same session. Generate launcher option tables from JSON. Load Android-only Minimum adjustments from the APK staging tree, keeping the sun enabled.
2. **GLES correctness (started in this PR).** Preserve the final render attachment until its actual last consumer. Test the sun and missing-surface reports on the phone. In 0.9.19 the HOM rasterizer visits only the accepted prefix after `remove_if`, sector detection keeps the collision-model identity separate from its numeric triangle index, and color clears restore the previous render attachment, draw buffers and write masks. Compare the same scene again and locate the first remaining GLES error with the debug callback. Do not ship a broad culling disable as a substitute.
   Launch with optional `-android-gl-debug` to get a synchronous GL debug callback on devices that support it; disable the option when measuring FPS.
   The 0.9.19 callback reports `query target 35092 is an invalid enum`: the shared OpenGL query helper uses desktop `GL_SAMPLES_PASSED` in GLES. Version 0.9.20 uses `GL_ANY_SAMPLES_PASSED` and reads its boolean result with the GLES query API; light visibility now accepts a nonzero result and failed occlusion queries fail open. This is a confirmed source of GL errors and missing lights, but the log cannot establish whether it also caused the missing static surfaces. Recheck the same portal scenes and the first remaining GL error on a device.
   The 0.9.20 report narrows the missing-surfaces symptom: walls and NPCs disappear together. They share sector traversal before individual visibility checks. Version 0.9.21 keeps Android portal traversal based on clipped portal frustums and reserves HOM for individual static visuals and entities. Both portal HOM and portal SSA discard entire sectors on approximate screen-space tests, so neither may remove sector topology. `-android-portal-hom` restores the previous portal HOM/SSA/fade tests for A/B comparisons. `[sector-trace]` records how many sectors are visited; compare the same route and frame time because extra sectors can increase CPU work. Portal and sector runtime markers now have defined initial values.
   Version 0.9.21 did not fix the missing groups. Its unrestricted traversal visited as many as 1635 sectors in 99 calls and made rendering slower. In 0.9.22, restore the portal HOM/SSA/fade policy and retain `[sector-trace]`. Correct the camera sector refresh guard, which compared the saved *direction* to the current *position*, and retry while the current sector is invalid. `[sector-detect]` records the camera sector and failed detections. HOM visibility now always accepts a hierarchy or visual whose bounding box contains the camera. `[object-visibility]` counts renderable rejects by invalid sector, inactive sector or individual HOM, so the next report can distinguish them without another broad culling change. These invariants need an on-device comparison before claiming they fix the disappearance.
   In 0.9.22 the outdoor scene still shows large missing ground patches while the main sector (115) remains traversed, and the expensive scene runs at 9–12 FPS. This rules out the previously assumed whole-sector portal drop as a complete explanation. In 0.9.23, retain portal pruning and the fast HOM test on each static aggregate when it passes. Only if that approximate test rejects a hierarchy, descend to its children and test them individually. A false result on one parent must not discard all its geometry; genuine negative results remain cheap to prune at the leaves. `[object-visibility]` now includes static hierarchy fallbacks and rejected leaves; `-android-hom-hierarchy` selects the original parent-level rejection for same-scene comparison. Neither restored surfaces nor faster rendering is established before device testing. The log also shows an unbound `s_base` material sampler at stage 20 early in gameplay, which requires a separate render-pass investigation.
3. **Frame pacing.** Obtain p95/p99 and CPU hot-path traces on the same route after image parity. Separate per-frame simulation cost from on-demand texture decode and shader compilation. Move one-time work out of the gameplay frame and make any cache bounded by an explicit memory budget; repeat the route at two internal resolutions to distinguish CPU and GPU ceilings.
4. **Vulkan implementation.** Replace the Android-only probe with a portable SDL surface bootstrap owned by `xrRenderVK`. Next build the HLSL→SPIR-V compile/reflection/cache path (including mods), then resource/barrier primitives, UI and static geometry, deferred lighting/shadows, post-process and lifecycle. Retain independent GLES for users while incomplete. Each gate has a validation-layer test and PC/Android screenshot comparison.
   In 0.9.18 the Android smoke path creates a render pass, image views and
   framebuffers, records one command buffer for each swapchain image, submits a
   color clear and presents the submitted image. Its smoke mode does not load
   GLES. It is not wired to gameplay or the cross-platform `xrRenderVK` module.
5. **Upstream integration.** Keep this PR small; split later renderer work by backend capability and add Android CI with a reproducible toolchain. Replay each logical patch on a fresh upstream `dev` branch; record merge conflicts and actual platform build results before updating the main fork branch.

The Android SDK/NDK build kit is available and APK/native compilation has passed.
Native gameplay, screen comparisons and lifecycle acceptance remain pending
physical-device verification; compilation cannot establish visual correctness or FPS gains.
The 0.9.19 run also fails to save `u0_a482 - начало игры.scop`: the
engine passes CP1251 filename bytes to Android's UTF-8 filesystem, and its
writer reports apparent success despite `fopen` failing. Filesystem encoding
must be handled symmetrically for creation, enumeration, rename and loading;
the gameplay UI must not report success before a valid writer is closed.
