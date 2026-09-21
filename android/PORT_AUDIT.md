# Android port audit

This audit covers the Android port through version 0.8.0. The comparison base
is upstream OpenXRay `dev` at `247d72764`. The Android branch already contains
the two upstream UI commits that follow the fork's older `origin/dev`, so they
are not Android-specific replacements.

## Compatibility invariants

- Original game and mod installations are read-only inputs. The launcher and
  engine must not rewrite `fsgame.ltx`, `user.ltx`, shaders, textures, archives,
  or other resources in a selected installation.
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
| Android filesystem bootstrap | Mounts the user-selected PC installation without copying or rewriting it | Read-only |
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
