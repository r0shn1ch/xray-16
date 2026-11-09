# Reproducible Vulkan frame captures

Use `r_vk_benchmark 600` in the console to arm a capture of 600 gameplay frame
intervals (range 1–36000). Close the console and return to unpaused gameplay.
The first completed game frame primes a steady wall clock; subsequent frame
intervals are measured independently of the engine's `time_factor`.

The renderer writes `frame-benchmark.json` under SDL's application preference
directory when the capture completes. On Android this is normally
`/data/user/0/org.openxray.stalker/files/frame-benchmark.json`. Look for
`benchmark.finished complete=1 saved=1` in the engine log and check the JSON's
`capture.complete` and exact requested/sample counts before accepting a run.
The ordinary `frame-metrics.json` remains a whole-session diagnostic and
includes loading, menus and background gaps; do not use it as a gameplay run.

Opening the console/menu, pausing, losing application focus, resetting the
surface, unloading the level or failing to present interrupts an active run.
Partial results are saved with `complete=false` and an interruption reason;
the capture does not silently resume. `r_vk_benchmark cancel` also cancels an
armed or active run. A new command is rejected while another capture is active.
An armed capture waits for gameplay in the loaded level, so it may also be
armed while its menu is open. A level change or device reset cancels an armed
capture too; arm it after loading the desired save, rather than before loading.

The report includes source SHA, GPU/driver, scene and presentation dimensions,
quality preset and several effective settings. Save the full user config,
APK hash, save-game fixture, camera/actor route and host/device description
alongside the JSON. Use the same route and settings for both sides of a
comparison, with a separate warm-up run. A stationary capture is a useful
baseline, but does not replace the required movement, combat and weather runs.
Emulator timings describe the emulator and host translation stack, not a phone.

Frame percentiles use every measured interval, including long frames. CPU time
currently covers the renderer's recording/submission/present call; it excludes
game simulation and is not total CPU-active frame time. GPU timestamps cover
submitted renderer commands and arrive two frames later. The first two delayed
results are omitted to exclude pre-capture work; the final pending queries are
not awaited. GPU percentiles therefore use fewer samples, and individual JSON
rows must not be interpreted as paired CPU/GPU measurements of the same frame.
Unavailable GPU timestamps are reported as null, rather than zero.
