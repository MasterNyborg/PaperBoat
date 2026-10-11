# Paperboat VR

World-locked, stereoscopic "3D TV" rendering to an OpenXR headset. Windows +
DirectX 11 only. The rest of the port talks to this module only through
`Vr.h`; all OpenXR/D3D11 details are confined to `Vr.cpp`.

## How it works

- **Display:** OpenXR **composition quad layers** (`XrCompositionLayerQuad`).
  The runtime handles world-locking, head tracking, lens distortion and
  compositing. We hand it a texture, a pose and a size — one quad per eye for
  stereo. Screen size/distance/recenter just change the quad pose/size.
- **Stereo:** The game is rendered twice per presented frame (once per eye) by
  driving `Fast::Interpreter` directly (`StartFrame()` + `Run()`), as the port's
  `RunCommands` already does. We force rendering offscreen into the reuse of
  `mGameFb` and clear colour between eyes (libultraship's `Run` only clears
  depth).
- **Eye offset:** The game bakes the view into the combined projection matrix
  (`src/cam_main.c`). In row-major convention a translated eye is an off-axis
  projection, so we edit that matrix directly:
  `Pc[r][0] += Pc[r][3] * tx`, `tx = ±half-IPD` in game units. It is injected via
  the per-subframe `mtx_replacements` map keyed by
  `&gDisplayContext->camPerspMatrix[cam]` for the perspective cameras (0,1,2);
  the HUD camera (3, orthographic) is left mono.
- **Handoff to the HMD:** `interpreter->mGfxFrameBuffer` is the game FB's
  `ID3D11ShaderResourceView`. Runtimes often only offer an *sRGB* swapchain
  format while the game FB is linear RGBA8, so a tiny D3D11 fullscreen blit
  (`Blitter`, shaders compiled from `d3dcompiler_47.dll`) copies it into the eye
  swapchain. If `d3dcompiler` is missing it falls back to `CopyResource` when the
  formats happen to match.

Nothing in the vendored `external/libultraship` is modified; the integration
relies on the interpreter members that are currently public (they carry a
`// private: TODO` marker upstream).

## Key files / hooks

| Where | What |
| - | - |
| `src/port/vr/Vr.h` | Engine-facing API (no OpenXR types). |
| `src/port/vr/Vr.cpp` | OpenXR session, swapchains, quad layers, per-eye render, blitter. |
| `src/port/Engine.cpp` `RunCommands` | Calls `VR::RenderGameFrame`; falls back to the normal draw. |
| `src/port/Game.cpp` | `VR::Initialize()` at launch, `VR::PollHotkeys()` per loop, `VR::RenderIdleFrame()` when idle, `VR::Shutdown()` on exit. |
| `src/port/ui/PaperboatMenuSettings.cpp` | Settings → VR section. |
| `CMakeLists.txt` | Fetches OpenXR-SDK, links the loader, copies the DLL. |

## Tuning / gotchas

- **Render size** (`kEyeRenderWidth` in `Vr.cpp`) is the per-eye game render
  width; height follows the game aspect. Lower it if VR performance suffers.
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
