# Paperboat VR

World-locked, stereoscopic "3D TV" rendering to an OpenXR headset. Windows +
DirectX 11 only. The rest of the port talks to this module only through
`Vr.h`; all OpenXR/D3D11 details are confined to `Vr.cpp`.

## How it works

- **Display:** OpenXR **composition quad layers** (`XrCompositionLayerQuad`).
  The runtime handles world-locking, head tracking, lens distortion, reprojection
  and compositing. We hand it a texture, a pose and a size — one quad per eye for
  stereo. Screen size/distance/recenter just change the quad pose/size.
- **Stereo:** Each game frame is rendered twice (once per eye) by driving
  `Fast::Interpreter` directly (`StartFrame()` + `Run()`), as the port's
  `RunCommands` already does. We force rendering offscreen into a reuse of
  `mGameFb` and clear colour between eyes (libultraship's `Run` only clears
  depth).
- **Eye offset:** The game bakes the view into the combined projection matrix
  (`src/cam_main.c`). In row-major convention a translated eye is an off-axis
  projection, so we edit that matrix directly:
  `Pc[r][0] += Pc[r][3] * tx`, `tx = ±half-IPD` in game units. It is injected via
  the per-subframe `mtx_replacements` map keyed by
  `&gDisplayContext->camPerspMatrix[cam]` for the perspective cameras (0,1,2);
  the HUD camera (3, orthographic) is left mono.

### Decoupled frame delivery (game rate ≠ display rate)

A 60 fps game on a 120 Hz headset must still be *delivered* at the display rate,
or the compositor stutters. So rendering and presentation are decoupled:

1. `RenderGameFrame` (called from `Engine.cpp` `RunCommands` once per sub-frame)
   renders each eye into an **intermediate eye texture** (`eyeTex[e]`,
   `R8G8B8A8_UNORM`) and mirrors the desktop window. It does **not** touch the
   XR swapchains.
2. `PresentFrame` (called once per main-loop iteration from `Game.cpp`) runs the
   OpenXR frame: `xrWaitFrame` → `xrBeginFrame` → acquire both swapchains →
   blit `eyeTex[e]` into the acquired image → release → `xrEndFrame`. It reuses
   the latest render, so the headset gets a fresh submission every display tick
   even though the game only renders 60 times a second.

The blit is a fullscreen draw (`Blitter`, shaders compiled from
`d3dcompiler_47.dll`) because the runtime usually only offers an *sRGB*
swapchain format while the game FB is linear RGBA8, and the eye texture size may
differ from the swapchain size. `mGfxFrameBuffer` is copied into `eyeTex[e]`
first (a `CopyResource` when sizes/formats match, else a blit).

Measured on a Quest 3-class setup: game render ≈ 0.6–2 ms per eye-pair, blit
≈ 0.03 ms, `xrWaitFrame` ≈ 7 ms (paces to 120 Hz), `xrEndFrame` ≈ 0.4 ms — i.e.
presentation comfortably holds the display rate.

### Frame pacing / latency notes

- `GetTargetFpsOverride()` returns **60** (the *game* content rate), so the port
  interpolates 2 sub-frames per 30 Hz logic tick. `PresentFrame` still submits at
  the full display rate.
- The port's own **desktop present** (`Interpreter::EndFrame`) contains a frame
  limiter (`gfx_dxgi.cpp` `SwapBuffersBegin`) that blocks ~12 ms to hit the
  window's target FPS. During VR it would starve the headset loop, so it is
  presented only every Nth game frame (a cheap spectator mirror); the other
  frames use the cheap `EndFrame`/`FinishRender` path.
- `VrDebugEnabled()` also forces the render path when the session is not
  focused, for headset-free testing.

### Settings persistence

Screen size / distance / 3D strength / the VR enable flag are stored as console
variables (`gVR.ScreenSize`, `gVR.ScreenDistance`, `gVR.StereoStrength`,
`gVR.Enabled`) and so persist in `paperboat.cfg.json` like every other port
setting. They are loaded lazily (`LoadSettings`) the first time a getter is
called, so they survive restarts and work even with no headset attached.

### Headset sleep / disconnect

- **Sleep / standby** (take the headset off): the runtime sends
  `STOPPING → IDLE`. We `xrEndSession`, stop pumping the frame loop, and the game
  keeps running on the desktop at full speed. On wake the session goes `READY`
  again and we `xrBeginSession` + resume. The session is kept, so no relaunch is
  needed.
- **Runtime exit / hard disconnect**: the runtime sends `EXITING`. We destroy the
  OpenXR objects immediately (leaving a half-dead session around makes the
  loader/runtime fault a few seconds later). The game continues on the desktop.
- **Watchdog / backoff**: if a single XR frame takes > 250 ms, or the swapchains
  stop being handed over, we assume the runtime stopped presenting and pause the
  pump. We then resume only when the session state actually changes (e.g. the
  headset wakes) — never on a timer, because a retry would block in `xrWaitFrame`
  for as long as the runtime is throttling (which is what turned "headset asleep"
  into 1 fps). While paused the game runs on the desktop at full speed.

## Key files / hooks

| Where | What |
| - | - |
| `src/port/vr/Vr.h` | Engine-facing API (no OpenXR types). |
| `src/port/vr/Vr.cpp` | OpenXR session, swapchains, quad layers, per-eye render, eye textures, blitter. |
| `src/port/Engine.cpp` `RunCommands` | Calls `VR::RenderGameFrame`; falls back to the normal draw. |
| `src/port/Game.cpp` | `VR::Initialize()` at launch, `VR::PollEvents()` + `VR::PollHotkeys()` + `VR::PresentFrame()` per loop, `VR::Shutdown()` on exit. |
| `src/port/ui/PaperboatMenuSettings.cpp` | Settings → VR section. |
| `CMakeLists.txt` | Fetches OpenXR-SDK, links the loader, copies the DLL. |

## Environment variables (debug / tuning)

| Variable | Effect |
| - | - |
| `PAPERBOAT_VR_DEBUG=1` | Force the VR render path even when the session isn't focused (headset-free testing). |
| `PAPERBOAT_VR_PERF=1` | Emit a `[VR] perf[scene] …` line every 2 s (also enabled by `PAPERBOAT_VR_DEBUG`). |
| `PAPERBOAT_VR_FPS=N` | Force the game's interpolation target (content) rate. |
| `PAPERBOAT_VR_RESSCALE=F` | Scale the per-eye render-resolution cap. |
| `PAPERBOAT_VR_MAXW=N` | Absolute per-eye render-width cap (px). |
| `PAPERBOAT_VR_MIRROR=N` | Desktop spectator-mirror period in game frames (default 2 = ~30 Hz; 0 disables). Do **not** use 1: presenting every game frame re-engages the ~12 ms present limiter and pins the headset to the game rate. |

## Tuning / gotchas

- **Render size**: `kMaxEyeRenderWidth` (default 1280 px per eye) caps the game
  render resolution; the blit upscales it to the HMD swapchain. Raise it for
  sharpness (there is headroom), lower it if VR performance suffers.
- **Stereo sign/axis:** if depth looks inverted or left/right are swapped, flip
  the sign of the offset passed in `RenderGameFrame` (`+half` / `-half`). The
  horizontal eye axis is eye-space X; a canted-display HMD may need a different
  axis term in `ApplyEyeOffset`.
- **Image orientation:** if the screen appears vertically flipped, invert the V
  coordinate in the blitter's vertex shader.
- **Key-frame interpolation:** on key frames the replacement map is empty, so the
  baked matrix is converted from fixed point (`MatrixL2F`) and offset.
- **Side effects run twice:** display lists that capture/read the framebuffer
  (previous-frame capture, flame FB) execute once per eye. If something looks
  wrong only in VR, that is the first place to look.
- **Session loss:** on `EXITING`/`LOSS_PENDING`/instance loss the session is
  stopped, so the game falls back to the desktop instead of calling into a dead
  session.
