#pragma once

// Engine-facing VR interface for Paperboat.
//
// This header intentionally exposes no OpenXR types: the rest of the port only
// sees plain functions and POD settings. All OpenXR/D3D11 details live in
// Vr.cpp. The implementation is compiled only when PAPERBOAT_ENABLE_OPENXR is
// defined (Windows + D3D11); otherwise every function is a no-op stub so the
// other platforms keep building unchanged.

#include <cstdint>
#include <unordered_map>

// Gfx / Mtx / MtxF / Interpreter
#include <fast/interpreter.h>

namespace Paperboat::VR {

// Creates the OpenXR session (if a runtime + HMD are present). Safe to call
// when no runtime exists: it just fails and VR stays inactive. Call once, after
// the window/interpreter exist.
bool Initialize();

// Tears the session down.
void Shutdown();

// True when an OpenXR session was created and is running (i.e. rendering to the
// headset is possible). Independent of focus.
bool IsSessionRunning();

// Pumps OpenXR events (session state transitions, instance/profile changes).
// Must be called every main-loop iteration, unconditionally: the session only
// advances to READY/RUNNING in response to events, so gating this on an already
// running session would deadlock startup.
void PollEvents();

// User toggle. When false, VR rendering is skipped entirely even if the session
// is alive (handy for A/B testing / falling back to a plain desktop window).
bool GetEnabled();
void SetEnabled(bool enabled);

// Current VR-controller state mapped to N64 pad semantics: `buttons` uses the
// CONT_* bits and the stick is in the N64 range (-80..80). Returns false unless
// a VR session is running.
bool GetVrPad(uint16_t& buttons, int8_t& stickX, int8_t& stickY);

// Returns the headset refresh rate (Hz) while VR is active, else 0, so the port
// can pace its frame interpolation to match the frames the runtime presents.
uint32_t GetTargetFpsOverride();

// Caps the interpreter's render resolution while VR is active: a large desktop
// window otherwise makes each eye render at full desktop resolution
// (e.g. 2560x1299 x2), which drops frames and causes judder. Call every frame;
// it restores the user's setting when VR is inactive.
void UpdateResolutionCap(Fast::Interpreter* interpreter);

// Renders one presented frame to the headset in stereo, and mirrors the left eye
// to the desktop window. `mtxReplacements` is the per-subframe interpolation map
// the port would normally hand to Fast3dWindow::DrawAndRunGraphicsCommands.
//
// Returns true if VR handled the frame (caller must NOT also draw normally).
// Returns false when VR is inactive/unfocused so the caller falls back to the
// regular desktop path.
bool RenderGameFrame(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtxReplacements);

// Submits one frame to the headset at the display rate, reusing the most recent
// game render. Call once per main-loop iteration so the headset always receives
// a frame every display tick, in order, regardless of the game's render rate.
void PresentFrame();

// Keyboard shortcuts (recenter / size / distance / stereo / toggle). Call once
// per main-loop iteration.
void PollHotkeys();

// --- Screen controls -------------------------------------------------------
void Recenter();

float GetScreenSize();            // width in metres (height follows aspect)
void SetScreenSize(float metres);
float GetScreenDistance();        // metres in front of the head
void SetScreenDistance(float metres);
float GetStereoStrength();        // half-IPD in game world units
void SetStereoStrength(float strength);
float GetConvergence();           // zero-parallax trim; fixed at 0 in this build
void SetConvergence(float ndc);   // no-op (trim is fixed at 0)

// Re-reads the persisted settings (gVR.* CVars) and applies them live. The
// settings menu widgets write the CVars directly and call this.
void ReloadSettings();

} // namespace Paperboat::VR
