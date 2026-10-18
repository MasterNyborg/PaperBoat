// Paperboat VR: world-locked stereoscopic "3D TV" rendered to an OpenXR headset.
//
// Design (see also the plan discussed with the user):
//   * OpenXR composition *quad layers* do the hard work: the runtime world-locks
//     the screen, tracks the head, applies lens distortion and composites. We
//     only supply a per-eye texture + a pose + a size.
//   * The game is rendered twice per game frame (once per eye) through
//     libultraship's public Interpreter API. We force the frame offscreen into
//     the game FBO (mGameFb), reuse it for both eyes (clearing colour between
//     them), and copy each result into an intermediate eye texture.
//   * Delivery is decoupled from rendering: RenderGameFrame fills the eye
//     textures at the game's rate; PresentFrame (once per main-loop iteration)
//     runs the OpenXR frame and blits the latest eye textures into the swapchains
//     at the display rate. A 60 fps game therefore still gets a fresh submission
//     every 120 Hz display tick.
//   * Stereo is produced by shifting the game's combined projection*view matrix
//     by a small eye-space translation. The game bakes view into the projection
//     matrix (cam_main.c), so in row-major convention an eye offset is:
//         Pc_eye[r][0] += Pc[r][3] * tx     (tx = half-IPD in game units)
//     The interpreter consumes that matrix through the per-subframe replacement
//     map, which is where we inject it.
//
// No changes are made to external/libultraship.

#include "Vr.h"

#if defined(PAPERBOAT_ENABLE_OPENXR)

#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef XR_USE_PLATFORM_WIN32
#define XR_USE_PLATFORM_WIN32
#endif
#ifndef XR_USE_GRAPHICS_API_D3D11
#define XR_USE_GRAPHICS_API_D3D11
#endif
#include <windows.h>
#include <unknwn.h>
#include <d3d11.h>
#include <dxgi.h>
#include <dxgiformat.h>
#include <wrl/client.h>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include <fast/Fast3dWindow.h>
#include <fast/backends/gfx_direct3d_common.h>
#include <fast/backends/gfx_window_manager_api.h>
#include <fast/backends/gfx_dxgi.h>
#include <fast/interpreter.h>
#include <ship/Context.h>
#include <ship/config/ConsoleVariable.h>
#include <ship/window/Window.h>
#include <ship/window/gui/Gui.h>

#include <imgui.h>
#include <imgui_impl_dx11.h>

#include <spdlog/spdlog.h>

#include "Engine.h" // OTRGetGameRenderWidth/Height / OTRGetAspectRatio
#include "port/interpolation/FrameInterpolation.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>
#include <vector>

// Game globals (defined in src/cam_main.c).
#include "common_structs.h"
extern "C" {
extern Camera gCameras[4];
extern s16 gCurrentCamID;
// Double-buffered display contexts (main_loop.c). The display list bakes in the
// projection matrix of one buffer while gDisplayContext may point at the other,
// so the eye offset must be applied to the projection in BOTH.
extern DisplayContext DisplayContexts[2];
extern s32 gCurrentDisplayContextIndex;
}

namespace Paperboat::VR {
namespace {

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Tunables / defaults
// ---------------------------------------------------------------------------
constexpr uint32_t kEyeRenderWidth = 1280; // per-eye render width; height from aspect
constexpr float kDefaultScreenSize = 1.6f; // metres, width
constexpr float kDefaultScreenDistance = 2.0f;
constexpr float kDefaultScreenVerticalOffset = 0.0f;
constexpr float kDefaultStereoStrength = 3.0f; // full IPD in game world units
constexpr float kMaxEyeRenderWidth = 1280.0f;  // per-eye render width cap
constexpr float kScreenSizeStep = 0.1f;
constexpr float kScreenDistanceStep = 0.1f;
constexpr float kStereoStep = 0.25f;

// The three perspective gameplay cameras (CAM_DEFAULT/CAM_BATTLE/CAM_TATTLE).
// CAM_HUD (3, orthographic) is intentionally left mono.
constexpr int kPerspectiveCameras[] = { 0, 1, 2 };

bool Failed(XrResult r, const char* what) {
    if (XR_FAILED(r)) {
        SPDLOG_ERROR("[VR] OpenXR call failed ({}): {}", what, (int)r);
        return true;
    }
    return false;
}

struct EyeSwapchain {
    XrSwapchain handle = XR_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
    int64_t format = 0;
    std::vector<XrSwapchainImageD3D11KHR> images;
    uint32_t acquiredIndex = 0;
    bool acquired = false;
};

// True for swapchain formats whose values are sRGB-encoded (display-referred).
bool IsSrgbFormat(int64_t format) {
    return format == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB || format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
}

// Minimal D3D11 fullscreen blitter. The HMD swapchain format often doesn't
// match the game's framebuffer format (e.g. sRGB vs linear RGBA8), and the
// sizes may differ, so we can't CopyResource. This draws the game texture onto
// the swapchain image with a passthrough shader.
class Blitter {
  public:
    bool Init(ID3D11Device* device) {
        HMODULE compiler = LoadLibraryA("d3dcompiler_47.dll");
        if (!compiler) {
            SPDLOG_WARN("[VR] d3dcompiler_47.dll not found; falling back to CopyResource.");
            return false;
        }
        auto compile = reinterpret_cast<PFN_D3DCOMPILE>(GetProcAddress(compiler, "D3DCompile"));
        if (!compile) {
            return false;
        }

        static const char* kVs = "struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };\n"
                                 "VSOut main(uint id : SV_VertexID) {\n"
                                 "  VSOut o;\n"
                                 "  float2 p[3] = { float2(-1,-1), float2(-1,3), float2(3,-1) };\n"
                                 "  float2 t = p[id];\n"
                                 "  o.pos = float4(t, 0, 1);\n"
                                 "  o.uv = float2((t.x+1)*0.5, 1.0-(t.y+1)*0.5);\n"
                                 "  return o;\n"
                                 "}";
        static const char* kPs = "Texture2D tex : register(t0);\n"
                                 "SamplerState smp : register(s0);\n"
                                 "float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {\n"
                                 "  return tex.Sample(smp, uv);\n"
                                 "}";
        // For a linear (UNORM) swapchain: decode the game's sRGB bytes so the
        // compositor's linear pipeline reproduces the intended brightness.
        static const char* kPsDecode = "Texture2D tex : register(t0);\n"
                                       "SamplerState smp : register(s0);\n"
                                       "float3 toLinear(float3 c) {\n"
                                       "  float3 lo = c / 12.92;\n"
                                       "  float3 hi = pow(saturate((c + 0.055) / 1.055), 2.4);\n"
                                       "  return lerp(hi, lo, step(c, 0.04045));\n"
                                       "}\n"
                                       "float4 main(float4 pos : SV_Position, float2 uv : TEXCOORD0) : SV_Target {\n"
                                       "  float4 c = tex.Sample(smp, uv);\n"
                                       "  return float4(toLinear(c.rgb), c.a);\n"
                                       "}\n";

        ComPtr<ID3DBlob> vsBlob, psBlob, psDecodeBlob, err;
        if (FAILED(compile(kVs, strlen(kVs), nullptr, nullptr, nullptr, "main", "vs_5_0", 0, 0, &vsBlob, &err)) ||
            FAILED(compile(kPs, strlen(kPs), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0, &psBlob, &err)) ||
            FAILED(compile(kPsDecode, strlen(kPsDecode), nullptr, nullptr, nullptr, "main", "ps_5_0", 0, 0,
                           &psDecodeBlob, &err))) {
            SPDLOG_WARN("[VR] Shader compilation failed; falling back to CopyResource.");
            return false;
        }
        if (FAILED(device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs)) ||
            FAILED(device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps)) ||
            FAILED(device->CreatePixelShader(psDecodeBlob->GetBufferPointer(), psDecodeBlob->GetBufferSize(), nullptr,
                                             &psDecode))) {
            return false;
        }

        D3D11_SAMPLER_DESC sd{};
        sd.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
        sd.AddressU = sd.AddressV = sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        sd.MaxLOD = D3D11_FLOAT32_MAX;
        if (FAILED(device->CreateSamplerState(&sd, &sampler))) {
            return false;
        }

        D3D11_RASTERIZER_DESC rd{};
        rd.FillMode = D3D11_FILL_SOLID;
        rd.CullMode = D3D11_CULL_NONE;
        if (FAILED(device->CreateRasterizerState(&rd, &raster))) {
            return false;
        }

        D3D11_DEPTH_STENCIL_DESC dsd{};
        dsd.DepthEnable = FALSE;
        dsd.StencilEnable = FALSE;
        if (FAILED(device->CreateDepthStencilState(&dsd, &depthOff))) {
            return false;
        }

        D3D11_BLEND_DESC bd{};
        bd.RenderTarget[0].BlendEnable = FALSE;
        bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (FAILED(device->CreateBlendState(&bd, &blendOpaque))) {
            return false;
        }

        // "over" blend for premultiplied-alpha sources (the ImGui menu texture):
        //   out = src + dst * (1 - src.a)
        D3D11_BLEND_DESC pb{};
        pb.RenderTarget[0].BlendEnable = TRUE;
        pb.RenderTarget[0].SrcBlend = D3D11_BLEND_ONE;
        pb.RenderTarget[0].DestBlend = D3D11_BLEND_INV_SRC_ALPHA;
        pb.RenderTarget[0].BlendOp = D3D11_BLEND_OP_ADD;
        pb.RenderTarget[0].SrcBlendAlpha = D3D11_BLEND_ONE;
        pb.RenderTarget[0].DestBlendAlpha = D3D11_BLEND_INV_SRC_ALPHA;
        pb.RenderTarget[0].BlendOpAlpha = D3D11_BLEND_OP_ADD;
        pb.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
        if (FAILED(device->CreateBlendState(&pb, &blendPremul))) {
            return false;
        }

        ready = true;
        return true;
    }

    bool ready = false;

    void Blit(ID3D11Device* device, ID3D11DeviceContext* ctx, ID3D11ShaderResourceView* srv,
              ID3D11Texture2D* dst, DXGI_FORMAT dstFormat, uint32_t dstW, uint32_t dstH, bool decodeSrgb = false,
              bool premul = false) {
        if (!ready || !srv || !dst) {
            return;
        }
        // The XR swapchain textures are created TYPELESS, so a default (nullptr)
        // RTV desc would fail; the format must be given explicitly.
        ComPtr<ID3D11RenderTargetView> rtv;
        D3D11_RENDER_TARGET_VIEW_DESC rtvDesc{};
        rtvDesc.Format = dstFormat;
        rtvDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
        if (FAILED(device->CreateRenderTargetView(dst, &rtvDesc, &rtv))) {
            static bool warned = false;
            if (!warned) {
                SPDLOG_WARN("[VR] CreateRenderTargetView on swapchain (fmt {}) failed.", (int)dstFormat);
                warned = true;
            }
            return;
        }

        D3D11_VIEWPORT vp{};
        vp.Width = (float)dstW;
        vp.Height = (float)dstH;
        vp.MaxDepth = 1.0f;

        ID3D11RenderTargetView* rtvRaw = rtv.Get();
        ctx->OMSetRenderTargets(1, &rtvRaw, nullptr);
        ctx->OMSetDepthStencilState(depthOff.Get(), 0);
        ctx->OMSetBlendState(premul ? blendPremul.Get() : blendOpaque.Get(), nullptr, 0xffffffff);
        ctx->RSSetState(raster.Get());
        ctx->RSSetViewports(1, &vp);
        ctx->IASetInputLayout(nullptr);
        ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        ctx->VSSetShader(vs.Get(), nullptr, 0);
        ctx->PSSetShader(decodeSrgb ? psDecode.Get() : ps.Get(), nullptr, 0);
        ctx->PSSetShaderResources(0, 1, &srv);
        ctx->PSSetSamplers(0, 1, sampler.GetAddressOf());
        ctx->Draw(3, 0);

        ID3D11ShaderResourceView* nullSrv = nullptr;
        ctx->PSSetShaderResources(0, 1, &nullSrv);
    }

  private:
    typedef HRESULT(WINAPI* PFN_D3DCOMPILE)(LPCVOID, SIZE_T, LPCSTR, const D3D_SHADER_MACRO*, ID3DInclude*, LPCSTR,
                                            LPCSTR, UINT, UINT, ID3DBlob**, ID3DBlob**);
    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    ComPtr<ID3D11PixelShader> psDecode;
    ComPtr<ID3D11SamplerState> sampler;
    ComPtr<ID3D11RasterizerState> raster;
    ComPtr<ID3D11DepthStencilState> depthOff;
    ComPtr<ID3D11BlendState> blendOpaque;
    ComPtr<ID3D11BlendState> blendPremul;
};

class Context {
  public:
    bool Initialize();
    void Shutdown();

    // Starts an OpenXR frame (wait/begin/acquire). Returns false if no frame
    // could be started (session not running / call failed).
    bool BeginFrame();
    // Ends the current OpenXR frame. `submitContent` false submits no layers.
    void EndFrame(bool submitContent);

    void PollEvents();

    // Syncs controller state without starting an XR frame (used while the pump is
    // paused so VR buttons/stick keep working instead of freezing).
    void SyncInput() {
        SyncActions();
    }

    XrSession session = XR_NULL_HANDLE;
    XrInstance instance = XR_NULL_HANDLE;
    XrSystemId systemId = XR_NULL_SYSTEM_ID;
    XrSpace localSpace = XR_NULL_HANDLE;
    XrSpace viewSpace = XR_NULL_HANDLE;
    XrSessionState sessionState = XR_SESSION_STATE_UNKNOWN;
    XrFrameState frameState{ XR_TYPE_FRAME_STATE };
    LUID requiredAdapterLuid{}; // adapter the runtime requires (from OpenXR)
    bool sessionRunning = false;
    bool initialized = false;
    bool haveContent = false;
    bool exitRequested = false;
    bool didInitialRecenter = false;
    bool loggedFirstContent = false;

    // --- OpenXR input (VR controllers -> N64 pad) ---
    XrActionSet actionSet = XR_NULL_HANDLE;
    XrAction actA = XR_NULL_HANDLE;
    XrAction actB = XR_NULL_HANDLE;
    XrAction actZ = XR_NULL_HANDLE;
    XrAction actStart = XR_NULL_HANDLE;
    XrAction actStick = XR_NULL_HANDLE;
    XrAction actCStick = XR_NULL_HANDLE;
    XrAction actMenu = XR_NULL_HANDLE;
    uint16_t vrButtons = 0;
    int8_t vrStickX = 0;
    int8_t vrStickY = 0;
    float hmdHz = 0.0f; // headset refresh rate (from predictedDisplayPeriod)
    uint32_t targetFps = 0; // latched, stabilised refresh for frame pacing

    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    float renderAspect = 0.0f; // captured once from the game camera viewport
    EyeSwapchain eyes[2];
    bool eyeValid[2] = { false, false }; // successfully acquired+waited this frame
    Blitter blitter;
    // The game renders into these at its own rate; the presentation step copies
    // them into the XR swapchains at the headset rate (decoupled frame delivery).
    ComPtr<ID3D11Texture2D> eyeTex[2];
    ComPtr<ID3D11ShaderResourceView> eyeSrv[2];
    // Offscreen ImGui frame. Rendering the port menu at the desktop window
    // resolution keeps the DX11 backend's viewport/ortho consistent; it is then
    // scaled over both (smaller) eye textures for the headset.
    ComPtr<ID3D11Texture2D> menuTex;
    ComPtr<ID3D11ShaderResourceView> menuSrv;
    ComPtr<ID3D11RenderTargetView> menuRtv;
    uint32_t menuWidth = 0;
    uint32_t menuHeight = 0;
    XrPosef screenPose{ { 0, 0, 0, 1 }, { 0, 0, -kDefaultScreenDistance } };
    XrExtent2Df screenSize{ kDefaultScreenSize, kDefaultScreenSize * 0.75f };
    // The screen sits on the ray (anchor + dir * distance), captured at recenter.
    // Using the stored anchor (rather than a live head pose) means the distance
    // control always moves the screen deterministically.
    XrVector3f screenAnchor{ 0.0f, 0.0f, 0.0f };
    XrVector3f screenDir{ 0.0f, 0.0f, -1.0f };

    bool CreateSwapchains(uint32_t width, uint32_t height);
    void CreateEyeTextures(uint32_t width, uint32_t height);
    bool EnsureMenuTexture(uint32_t w, uint32_t h);
    void CompositeUiToEyes();
    ID3D11DeviceContext* DeviceContext() const {
        return context;
    }
    ID3D11Device* Device() const {
        return device;
    }

  private:
    bool CreateInstance();
    bool CreateSession();
    bool CreateActions();
    void SyncActions();
    void DestroySwapchains();
    void AcquireEye(int eye);
    void ReleaseEye(int eye);
    void SubmitLayers(bool submitContent);

    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    bool frameBegun = false;
};

Context* g_ctx = nullptr;

// Settings (kept here; surfaced through the getters/setters below and persisted
// as console variables so they survive a restart).
bool g_enabled = true;
float g_screenWidth = kDefaultScreenSize; // metres
float g_screenDistance = kDefaultScreenDistance;
// Vertical offset from the anchor/eye line, in metres. Positive moves the screen up.
float g_screenVerticalOffset = kDefaultScreenVerticalOffset;
// When true the screen yaws every frame so its face always points at the viewer.
bool g_facePlayer = true;
float g_stereoStrength = kDefaultStereoStrength;
// User trim (NDC) added to the auto zero-parallax shift. Fixed at 0: the menu
// control was removed, so the zero-parallax plane stays on the camera target.
float g_convergence = 0.0f;
// Per-eye horizontal shift (pixels) applied to the background so it reads as
// distant. Left eye gets -g_bgParallaxPx, right eye +g_bgParallaxPx (a point
// behind the zero-parallax plane has its left view shifted left, right view
// right). 0 = background stays mono.
int g_bgParallaxPx = 2;

// ---------------------------------------------------------------------------
// In-headset ImGui menu
// ---------------------------------------------------------------------------
constexpr const char* kCvarControlNav = "gSettings.ControlNav";
constexpr int32_t kVrMenuInputBlockId = 95237930;

// True while the port menu is open in VR. Read by GetVrPad on the SI thread, so
// it must be atomic.
std::atomic<bool> g_vrMenuInputBlock{ false };

bool PortMenuOpen(Ship::Gui* gui) {
    return gui != nullptr && (gui->GetMenuOrMenubarVisible() ||
                              ImGui::IsPopupOpen(nullptr, ImGuiPopupFlags_AnyPopupId | ImGuiPopupFlags_AnyPopupLevel));
}

void UpdateVrInputBlock(bool block);

// Toggles the port's ImGui menu (same target the keyboard/controller toggle hits).
void TogglePortMenu() {
    auto ship = Ship::Context::GetRawInstance();
    if (ship == nullptr || ship->GetWindow() == nullptr) {
        return;
    }
    auto gui = ship->GetWindow()->GetGui();
    if (gui == nullptr) {
        return;
    }
    if (gui->GetMenu()) {
        gui->GetMenu()->ToggleVisibility();
    } else if (gui->GetMenuBar()) {
        gui->GetMenuBar()->ToggleVisibility();
    }
    gui->UpdateGamepadNavigation();
    // Block/unblock immediately so the enabling grip press cannot leak into the
    // game before the next render frame.
    UpdateVrInputBlock(gui->GetMenuOrMenubarVisible());
}

// Blocks (or unblocks) game input while the VR menu is being used, so menu
// button presses and stick movement cannot leak into the game underneath.
void UpdateVrInputBlock(bool block) {
    g_vrMenuInputBlock.store(block, std::memory_order_relaxed);
    static bool applied = false;
    if (block == applied) {
        return;
    }
    applied = block;
    auto ship = Ship::Context::GetRawInstance();
    if (ship == nullptr || ship->GetControlDeck() == nullptr) {
        return;
    }
    if (block) {
        ship->GetControlDeck()->BlockGameInput(kVrMenuInputBlockId);
    } else {
        ship->GetControlDeck()->UnblockGameInput(kVrMenuInputBlockId);
    }
}

struct VrImGuiKeys {
    bool up = false;
    bool down = false;
    bool left = false;
    bool right = false;
    bool activate = false;
    bool cancel = false;
    bool start = false;
};
VrImGuiKeys g_vrImGuiKeys;

void SendVrImGuiKey(ImGuiKey key, bool& previous, bool next) {
    if (previous == next) {
        return;
    }
    previous = next;
    ImGui::GetIO().AddKeyEvent(key, next);
}

// Feeds the VR controllers into ImGui as a virtual gamepad. Must run before the
// ImGui frame is built (Gui::StartDraw). Gamepad navigation is forced on while
// the port menu is visible, independent of the desktop "Menu Controller
// Navigation" setting.
void InjectVrImGuiInput(bool menuOpen) {
    ImGuiIO& io = ImGui::GetIO();
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
    if (menuOpen) {
        // Set it before ImGui::NewFrame so nav is live on the same frame the
        // menu opens, not one frame later.
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    }

    bool up = false, down = false, left = false, right = false;
    bool activate = false, cancel = false, start = false;
    if (menuOpen && g_ctx != nullptr && g_ctx->sessionRunning) {
        constexpr int kDeadzone = 40; // 0.5 * the -80..80 N64 stick range
        up = g_ctx->vrStickY > kDeadzone;
        down = g_ctx->vrStickY < -kDeadzone;
        left = g_ctx->vrStickX < -kDeadzone;
        right = g_ctx->vrStickX > kDeadzone;
        activate = (g_ctx->vrButtons & 0x8000) != 0; // A
        cancel = (g_ctx->vrButtons & 0x4000) != 0;   // B
        start = (g_ctx->vrButtons & 0x1000) != 0;    // Start
    }
    SendVrImGuiKey(ImGuiKey_GamepadDpadUp, g_vrImGuiKeys.up, up);
    SendVrImGuiKey(ImGuiKey_GamepadDpadDown, g_vrImGuiKeys.down, down);
    SendVrImGuiKey(ImGuiKey_GamepadDpadLeft, g_vrImGuiKeys.left, left);
    SendVrImGuiKey(ImGuiKey_GamepadDpadRight, g_vrImGuiKeys.right, right);
    SendVrImGuiKey(ImGuiKey_GamepadFaceDown, g_vrImGuiKeys.activate, activate);
    SendVrImGuiKey(ImGuiKey_GamepadFaceRight, g_vrImGuiKeys.cancel, cancel);
    SendVrImGuiKey(ImGuiKey_GamepadStart, g_vrImGuiKeys.start, start);
}

// Watchdog for a runtime that stopped presenting (headset asleep/disconnected).
// If an XR frame takes pathologically long, or the swapchains stop being handed
// over, we stop pumping the XR frame loop so the game keeps running on the
// desktop at full speed. We only resume when the session state changes (which is
// what happens when the headset wakes) -- never on a timer, because a retry
// itself blocks in xrWaitFrame for as long as the runtime is throttling.
bool g_pumpPaused = false;

// Watchdog: the OpenXR frame pump runs on the main thread (the same thread that
// polls input and runs the game loop). If a blocking XR call hangs -- e.g. the
// headset/link stalls while the runtime keeps the session "focused" -- the whole
// loop freezes and the last input stays latched (Mario walks forever). Watch the
// main-loop heartbeat from a helper thread and ask the runtime to end the
// session, which releases the blocked call so the game falls back to the desktop.
std::atomic<double> g_mainLoopBeatMs{ 0.0 };
std::atomic<bool> g_stopWatchdog{ false };
std::thread g_watchdogThread;

// ---------------------------------------------------------------------------
// Temporary frame-timing instrumentation (removed once pacing is verified).
// ---------------------------------------------------------------------------
double NowMs() {
    static const auto start = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}
struct FramePerf {
    double lastLog = NowMs();
    double waitMs = 0, blitMs = 0, endMs = 0, renderMs = 0;
    double waitMax = 0, blitMax = 0, endMax = 0, renderMax = 0;
    double prepMs = 0, eye1Ms = 0, eye0Ms = 0, guiMs = 0, deskMs = 0;
    int present = 0, renders = 0, shouldRender = 0, beginFail = 0, endFail = 0;
};
FramePerf g_perf;

// True when VR should own rendering this frame: enabled, session running, and
// not currently backing off from a stalled/absent runtime.
bool VrActive() {
    return g_enabled && g_ctx != nullptr && g_ctx->sessionRunning && !g_pumpPaused;
}

// Pauses the XR frame pump (the runtime stopped presenting).
void PausePump(const char* why) {
    if (!g_pumpPaused) {
        SPDLOG_WARN("[VR] Runtime not presenting ({}); pausing headset submission, running on desktop.", why);
    }
    g_pumpPaused = true;
}

// Clears a pause (the session state changed, e.g. the headset woke).
void ResumePumpIfPaused() {
    if (g_pumpPaused) {
        g_pumpPaused = false;
        SPDLOG_INFO("[VR] Session state changed; resuming headset submission.");
    }
}

// --- main-loop stall watchdog ------------------------------------------------
void WatchdogThreadMain() {
    constexpr double kStallMs = 10000.0;
    while (!g_stopWatchdog.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const double beat = g_mainLoopBeatMs.load(std::memory_order_relaxed);
        if (beat <= 0.0) {
            continue;
        }
        const double now = NowMs();
        if (now - beat > kStallMs) {
            if (g_ctx != nullptr && g_ctx->sessionRunning && g_ctx->session != XR_NULL_HANDLE) {
                SPDLOG_WARN("[VR] Main loop stalled {:.1f}s; requesting session exit to recover.",
                            (now - beat) / 1000.0);
                xrRequestExitSession(g_ctx->session);
            } else {
                SPDLOG_WARN("[VR] Main loop stalled {:.1f}s.", (now - beat) / 1000.0);
            }
            g_mainLoopBeatMs.store(now, std::memory_order_relaxed); // don't spam
        }
    }
}

void StartWatchdog() {
    if (g_watchdogThread.joinable()) {
        return;
    }
    g_stopWatchdog.store(false, std::memory_order_relaxed);
    g_watchdogThread = std::thread(WatchdogThreadMain);
}

void StopWatchdog() {
    g_stopWatchdog.store(true, std::memory_order_relaxed);
    if (g_watchdogThread.joinable()) {
        g_watchdogThread.join();
    }
}

void BeatMainLoop() {
    g_mainLoopBeatMs.store(NowMs(), std::memory_order_relaxed);
}

void LogPerfIfNeeded(const char* phase) {
    static const bool enabled =
        std::getenv("PAPERBOAT_VR_PERF") != nullptr || std::getenv("PAPERBOAT_VR_DEBUG") != nullptr;
    if (!enabled) {
        return;
    }
    const double now = NowMs();
    const double dt = now - g_perf.lastLog;
    if (dt < 2000.0 || g_perf.present == 0) {
        return;
    }
    const double n = (double)g_perf.present;
    const double r = (double)std::max(1, g_perf.renders);
    SPDLOG_INFO(
        "[VR] perf[{}] present_fps={:.1f} game_fps={:.1f} | wait {:.2f} blit {:.2f} end {:.2f} | render {:.2f} "
        "(prep {:.2f} eye1 {:.2f} eye0 {:.2f} gui {:.2f} desk {:.2f}) max {:.1f}/{:.1f} | shouldRender={}/{} beginFail={}",
        phase, n * 1000.0 / dt, (double)g_perf.renders * 1000.0 / dt, g_perf.waitMs / n, g_perf.blitMs / n,
        g_perf.endMs / n, g_perf.renderMs / r, g_perf.prepMs / r, g_perf.eye1Ms / r, g_perf.eye0Ms / r, g_perf.guiMs / r,
        g_perf.deskMs / r, g_perf.renderMax, g_perf.waitMax, g_perf.shouldRender, g_perf.present, g_perf.beginFail);
    g_perf = FramePerf{};
    g_perf.lastLog = now;
}

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------
void RotateByQuat(const XrQuaternionf& q, float vx, float vy, float vz, float out[3]) {
    const float x = q.x, y = q.y, z = q.z, w = q.w;
    const float xx = x * x, yy = y * y, zz = z * z;
    const float xy = x * y, xz = x * z, yz = y * z;
    const float wx = w * x, wy = w * y, wz = w * z;
    out[0] = (1 - 2 * (yy + zz)) * vx + 2 * (xy - wz) * vy + 2 * (xz + wy) * vz;
    out[1] = 2 * (xy + wz) * vx + (1 - 2 * (xx + zz)) * vy + 2 * (yz - wx) * vz;
    out[2] = 2 * (xz - wy) * vx + 2 * (yz + wx) * vy + (1 - 2 * (xx + yy)) * vz;
}

XrQuaternionf YawQuat(float radians) {
    XrQuaternionf q{};
    q.y = std::sin(radians * 0.5f);
    q.w = std::cos(radians * 0.5f);
    return q;
}

// ---------------------------------------------------------------------------
// Multi-GPU (hybrid graphics) handling
//
// OpenXR requires the D3D11 device to be the adapter the runtime asks for
// (xrGetD3D11GraphicsRequirementsKHR). PaperBoat's DXGI device uses the OS
// default adapter, which on hybrid laptops is often the iGPU, producing
// "App submitted texture from wrong device" and a black headset. We detect the
// mismatch, set a per-app Windows GPU preference so DXGI picks the right GPU,
// then relaunch once (the preference only takes effect in a fresh process).
// ---------------------------------------------------------------------------
bool GetDeviceLuid(ID3D11Device* device, LUID& out) {
    ComPtr<IDXGIDevice> dxgiDev;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&dxgiDev)))) {
        return false;
    }
    ComPtr<IDXGIAdapter> adapter;
    if (FAILED(dxgiDev->GetAdapter(&adapter))) {
        return false;
    }
    DXGI_ADAPTER_DESC desc{};
    if (FAILED(adapter->GetDesc(&desc))) {
        return false;
    }
    out = desc.AdapterLuid;
    return true;
}

bool SameLuid(const LUID& a, const LUID& b) {
    return a.HighPart == b.HighPart && a.LowPart == b.LowPart;
}

bool ReadExeGpuPreference(const std::string& exe, std::string& value) {
    char buf[256] = "";
    DWORD size = sizeof(buf);
    if (RegGetValueA(HKEY_CURRENT_USER, "Software\\Microsoft\\DirectX\\UserGpuPreferences", exe.c_str(), RRF_RT_REG_SZ,
                     nullptr, buf, &size) == ERROR_SUCCESS) {
        value = buf;
        return true;
    }
    return false;
}

void WriteExeGpuPreference(const std::string& exe, const std::string& value) {
    HKEY key = nullptr;
    if (RegCreateKeyExA(HKEY_CURRENT_USER, "Software\\Microsoft\\DirectX\\UserGpuPreferences", 0, nullptr,
                        REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, nullptr, &key, nullptr) == ERROR_SUCCESS) {
        RegSetValueExA(key, exe.c_str(), 0, REG_SZ, (const BYTE*)value.c_str(), (DWORD)value.size() + 1);
        RegCloseKey(key);
    }
}

std::string CurrentExePath() {
    char buf[MAX_PATH] = "";
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return buf;
}

// High-performance (discrete) vs power-saving (integrated): whichever matches
// the runtime's adapter.
std::string PreferenceForRequiredAdapter(ID3D11Device* device, const LUID& required) {
    ComPtr<IDXGIDevice> dxgiDev;
    ComPtr<IDXGIAdapter> curAdapter;
    DXGI_ADAPTER_DESC curDesc{};
    if (SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&dxgiDev))) && SUCCEEDED(dxgiDev->GetAdapter(&curAdapter)) &&
        SUCCEEDED(curAdapter->GetDesc(&curDesc))) {
        ComPtr<IDXGIFactory1> factory;
        if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
            ComPtr<IDXGIAdapter1> a;
            for (UINT i = 0; factory->EnumAdapters1(i, a.ReleaseAndGetAddressOf()) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 d{};
                a->GetDesc1(&d);
                if (d.AdapterLuid.HighPart == required.HighPart && d.AdapterLuid.LowPart == required.LowPart) {
                    return d.DedicatedVideoMemory >= curDesc.DedicatedVideoMemory ? "GpuPreference=2;"
                                                                                  : "GpuPreference=1;";
                }
            }
        }
    }
    return "GpuPreference=2;"; // best guess: prefer the discrete GPU
}

[[noreturn]] void RelaunchSelfAndExit() {
    char exe[MAX_PATH] = "";
    GetModuleFileNameA(nullptr, exe, MAX_PATH);
    std::string cmd = GetCommandLineA();
    _putenv_s("PAPERBOAT_VR_GPU_RELAUNCH", "1");
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (CreateProcessA(exe, cmd.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &si, &pi)) {
        CloseHandle(pi.hThread);
        CloseHandle(pi.hProcess);
    }
    ExitProcess(0);
}

// Returns false when a relaunch was started (caller must not continue).
bool EnsureRuntimeAdapter(Context* c) {
    LUID deviceLuid{};
    if (!GetDeviceLuid(c->Device(), deviceLuid) || SameLuid(deviceLuid, c->requiredAdapterLuid)) {
        return true;
    }

    const std::string exe = CurrentExePath();
    const std::string wanted = PreferenceForRequiredAdapter(c->Device(), c->requiredAdapterLuid);
    std::string current;
    const bool haveCurrent = ReadExeGpuPreference(exe, current);

    SPDLOG_WARN(
        "[VR] D3D device LUID {}:{} != runtime adapter {}:{}; setting Windows GPU preference for '{}' to '{}'.",
        deviceLuid.HighPart, deviceLuid.LowPart, c->requiredAdapterLuid.HighPart, c->requiredAdapterLuid.LowPart, exe,
        wanted);

    if (haveCurrent && current == wanted) {
        SPDLOG_ERROR("[VR] GPU preference already correct but device still mismatched; VR may not display.");
        return true;
    }

    WriteExeGpuPreference(exe, wanted);

    if (std::getenv("PAPERBOAT_VR_GPU_RELAUNCH") != nullptr) {
        SPDLOG_ERROR("[VR] Already relaunched once for GPU selection; continuing (VR may not display).");
        return true;
    }

    SPDLOG_INFO("[VR] Relaunching PaperBoat on the correct GPU...");
    RelaunchSelfAndExit();
    return false;
}

// ---------------------------------------------------------------------------
// Context implementation
// ---------------------------------------------------------------------------
bool Context::CreateInstance() {
    const char* extensions[] = { XR_KHR_D3D11_ENABLE_EXTENSION_NAME };

    XrInstanceCreateInfo ci{ XR_TYPE_INSTANCE_CREATE_INFO };
    std::strncpy(ci.applicationInfo.applicationName, "Paperboat", XR_MAX_APPLICATION_NAME_SIZE - 1);
    ci.applicationInfo.applicationVersion = 1;
    std::strncpy(ci.applicationInfo.engineName, "libultraship", XR_MAX_ENGINE_NAME_SIZE - 1);
    ci.applicationInfo.engineVersion = 1;
    ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
    ci.enabledExtensionCount = 1;
    ci.enabledExtensionNames = extensions;

    XrResult createResult = xrCreateInstance(&ci, &instance);
    if (createResult == XR_ERROR_API_VERSION_UNSUPPORTED) {
        // Some runtimes (e.g. older SteamVR) only advertise OpenXR 1.0.
        SPDLOG_INFO("[VR] Runtime rejected OpenXR 1.1; retrying with 1.0.");
        ci.applicationInfo.apiVersion = XR_API_VERSION_1_0;
        createResult = xrCreateInstance(&ci, &instance);
    }
    if (Failed(createResult, "xrCreateInstance")) {
        instance = XR_NULL_HANDLE;
        return false;
    }

    XrSystemGetInfo si{ XR_TYPE_SYSTEM_GET_INFO };
    si.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
    if (Failed(xrGetSystem(instance, &si, &systemId), "xrGetSystem")) {
        return false;
    }

    // Log the adapter the runtime wants (multi-GPU correctness aid).
    PFN_xrVoidFunction getReqsFn = nullptr;
    xrGetInstanceProcAddr(instance, "xrGetD3D11GraphicsRequirementsKHR", &getReqsFn);
    auto getReqs = reinterpret_cast<PFN_xrGetD3D11GraphicsRequirementsKHR>(getReqsFn);
    if (getReqs) {
        XrGraphicsRequirementsD3D11KHR reqs{ XR_TYPE_GRAPHICS_REQUIREMENTS_D3D11_KHR };
        if (XR_SUCCEEDED(getReqs(instance, systemId, &reqs))) {
            requiredAdapterLuid = reqs.adapterLuid;
            SPDLOG_INFO("[VR] Runtime adapter LUID: {}:{} (min feature level {})", reqs.adapterLuid.HighPart,
                        reqs.adapterLuid.LowPart, (int)reqs.minFeatureLevel);
        }
    }

    return true;
}

bool Context::CreateSession() {
    XrGraphicsBindingD3D11KHR binding{ XR_TYPE_GRAPHICS_BINDING_D3D11_KHR };
    binding.device = device;

    XrSessionCreateInfo sci{ XR_TYPE_SESSION_CREATE_INFO };
    sci.next = &binding;
    sci.systemId = systemId;
    if (Failed(xrCreateSession(instance, &sci, &session), "xrCreateSession")) {
        session = XR_NULL_HANDLE;
        return false;
    }

    XrReferenceSpaceCreateInfo rsci{ XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
    rsci.poseInReferenceSpace = { { 0, 0, 0, 1 }, { 0, 0, 0 } };
    rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
    if (Failed(xrCreateReferenceSpace(session, &rsci, &localSpace), "xrCreateReferenceSpace(local)")) {
        return false;
    }
    rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    if (Failed(xrCreateReferenceSpace(session, &rsci, &viewSpace), "xrCreateReferenceSpace(view)")) {
        return false;
    }

    return true;
}

// Creates the action set mapping VR controllers onto the N64 pad and suggests
// bindings for the common interaction profiles.
bool Context::CreateActions() {
    XrActionSetCreateInfo asci{ XR_TYPE_ACTION_SET_CREATE_INFO };
    std::strncpy(asci.actionSetName, "gameplay", XR_MAX_ACTION_SET_NAME_SIZE - 1);
    std::strncpy(asci.localizedActionSetName, "Gameplay", XR_MAX_LOCALIZED_ACTION_SET_NAME_SIZE - 1);
    asci.priority = 0;
    if (Failed(xrCreateActionSet(instance, &asci, &actionSet), "xrCreateActionSet")) {
        actionSet = XR_NULL_HANDLE;
        return false;
    }

    auto makeAction = [this](const char* name, const char* localized, XrActionType type) -> XrAction {
        XrActionCreateInfo ci{ XR_TYPE_ACTION_CREATE_INFO };
        std::strncpy(ci.actionName, name, XR_MAX_ACTION_NAME_SIZE - 1);
        std::strncpy(ci.localizedActionName, localized, XR_MAX_LOCALIZED_ACTION_NAME_SIZE - 1);
        ci.actionType = type;
        ci.countSubactionPaths = 0;
        XrAction a = XR_NULL_HANDLE;
        xrCreateAction(actionSet, &ci, &a);
        return a;
    };

    actA = makeAction("a", "A", XR_ACTION_TYPE_BOOLEAN_INPUT);
    actB = makeAction("b", "B", XR_ACTION_TYPE_BOOLEAN_INPUT);
    actZ = makeAction("z", "Z", XR_ACTION_TYPE_BOOLEAN_INPUT);
    actStart = makeAction("start", "Start", XR_ACTION_TYPE_BOOLEAN_INPUT);
    actStick = makeAction("stick", "Stick", XR_ACTION_TYPE_VECTOR2F_INPUT);
    actCStick = makeAction("cstick", "C-Stick", XR_ACTION_TYPE_VECTOR2F_INPUT);
    actMenu = makeAction("menu", "Menu", XR_ACTION_TYPE_BOOLEAN_INPUT);

    auto pathOf = [this](const char* s) -> XrPath {
        XrPath p = XR_NULL_PATH;
        xrStringToPath(instance, s, &p);
        return p;
    };
    auto suggest = [&](const char* profile, const std::vector<std::pair<XrAction, const char*>>& binds) {
        std::vector<XrActionSuggestedBinding> v;
        v.reserve(binds.size());
        for (const auto& b : binds) {
            XrActionSuggestedBinding sb{};
            sb.action = b.first;
            sb.binding = pathOf(b.second);
            v.push_back(sb);
        }
        XrInteractionProfileSuggestedBinding sip{ XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
        sip.interactionProfile = pathOf(profile);
        sip.countSuggestedBindings = (uint32_t)v.size();
        sip.suggestedBindings = v.data();
        xrSuggestInteractionProfileBindings(instance, &sip);
    };

    // Oculus Touch (Quest via Virtual Desktop / Link). Right hand = A/B + C-stick,
    // left hand = trigger (Z), menu (Start) + analog stick.
    suggest("/interaction_profiles/oculus/touch_controller",
            { { actA, "/user/hand/right/input/a/click" },
              { actB, "/user/hand/right/input/b/click" },
              { actZ, "/user/hand/left/input/trigger/click" },
              { actStart, "/user/hand/left/input/menu/click" },
              { actStick, "/user/hand/left/input/thumbstick" },
              { actCStick, "/user/hand/right/input/thumbstick" },
              { actMenu, "/user/hand/left/input/squeeze/click" } });

    // Valve Index
    suggest("/interaction_profiles/valve/index_controller",
            { { actA, "/user/hand/right/input/a/click" },
              { actB, "/user/hand/right/input/b/click" },
              { actZ, "/user/hand/left/input/trigger/click" },
              { actStart, "/user/hand/left/input/thumbstick/click" },
              { actStick, "/user/hand/left/input/thumbstick" },
              { actCStick, "/user/hand/right/input/thumbstick" },
              { actMenu, "/user/hand/left/input/squeeze/click" } });

    // Microsoft Mixed Reality
    suggest("/interaction_profiles/microsoft/motion_controller",
            { { actA, "/user/hand/right/input/trigger/click" },
              { actB, "/user/hand/right/input/squeeze/click" },
              { actZ, "/user/hand/left/input/trigger/click" },
              { actStart, "/user/hand/left/input/menu/click" },
              { actStick, "/user/hand/left/input/thumbstick" },
              { actCStick, "/user/hand/right/input/thumbstick" },
              { actMenu, "/user/hand/left/input/squeeze/click" } });

    // KHR Simple (fallback: buttons only, no sticks)
    suggest("/interaction_profiles/khr/simple_controller",
            { { actA, "/user/hand/right/input/select/click" },
              { actB, "/user/hand/left/input/select/click" },
              { actStart, "/user/hand/left/input/menu/click" },
              { actMenu, "/user/hand/right/input/menu/click" } });

    XrSessionActionSetsAttachInfo attach{ XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
    attach.countActionSets = 1;
    attach.actionSets = &actionSet;
    if (Failed(xrAttachSessionActionSets(session, &attach), "xrAttachSessionActionSets")) {
        return false;
    }
    SPDLOG_INFO("[VR] Controller action set ready.");
    return true;
}

void Context::SyncActions() {
    if (actionSet == XR_NULL_HANDLE || !sessionRunning) {
        return;
    }
    XrActiveActionSet active{};
    active.actionSet = actionSet;
    active.subactionPath = XR_NULL_PATH;
    XrActionsSyncInfo si{ XR_TYPE_ACTIONS_SYNC_INFO };
    si.countActiveActionSets = 1;
    si.activeActionSets = &active;
    if (XR_FAILED(xrSyncActions(session, &si))) {
        return;
    }

    auto boolState = [&](XrAction a) -> bool {
        if (a == XR_NULL_HANDLE) {
            return false;
        }
        XrActionStateGetInfo gi{ XR_TYPE_ACTION_STATE_GET_INFO };
        gi.action = a;
        XrActionStateBoolean s{ XR_TYPE_ACTION_STATE_BOOLEAN };
        if (XR_FAILED(xrGetActionStateBoolean(session, &gi, &s))) {
            return false;
        }
        return s.isActive && s.currentState;
    };
    auto vecState = [&](XrAction a, XrVector2f& out) -> bool {
        if (a == XR_NULL_HANDLE) {
            return false;
        }
        XrActionStateGetInfo gi{ XR_TYPE_ACTION_STATE_GET_INFO };
        gi.action = a;
        XrActionStateVector2f s{ XR_TYPE_ACTION_STATE_VECTOR2F };
        if (XR_FAILED(xrGetActionStateVector2f(session, &gi, &s))) {
            return false;
        }
        out = s.currentState;
        return s.isActive;
    };

    uint16_t b = 0;
    if (boolState(actA)) {
        b |= 0x8000; // A_BUTTON
    }
    if (boolState(actB)) {
        b |= 0x4000; // B_BUTTON
    }
    if (boolState(actZ)) {
        b |= 0x2000; // Z_TRIG
    }
    if (boolState(actStart)) {
        b |= 0x1000; // START_BUTTON
    }

    // Left grip toggles the port's ImGui menu. Edge-triggered so holding the grip
    // does not reopen the menu after it is closed from inside.
    static bool menuPrev = false;
    const bool menuNow = boolState(actMenu);
    if (menuNow && !menuPrev) {
        TogglePortMenu();
    }
    menuPrev = menuNow;

    XrVector2f ls{}, rs{};
    const bool lsActive = vecState(actStick, ls);
    const bool rsActive = vecState(actCStick, rs);

    int8_t sx = 0, sy = 0;
    if (lsActive) {
        sx = (int8_t)std::clamp(ls.x * 80.0f, -80.0f, 80.0f);
        sy = (int8_t)std::clamp(ls.y * 80.0f, -80.0f, 80.0f);
    }
    if (rsActive) {
        const float dz = 0.5f;
        if (rs.y > dz) {
            b |= 0x0008; // C up
        }
        if (rs.y < -dz) {
            b |= 0x0004; // C down
        }
        if (rs.x < -dz) {
            b |= 0x0002; // C left
        }
        if (rs.x > dz) {
            b |= 0x0001; // C right
        }
    }

    vrButtons = b;
    vrStickX = sx;
    vrStickY = sy;
}

bool Context::CreateSwapchains(uint32_t width, uint32_t height) {
    DestroySwapchains();

    uint32_t fmtCount = 0;
    xrEnumerateSwapchainFormats(session, 0, &fmtCount, nullptr);
    std::vector<int64_t> formats(fmtCount);
    if (fmtCount > 0) {
        xrEnumerateSwapchainFormats(session, fmtCount, &fmtCount, formats.data());
    }

    // The game framebuffer holds display-referred (sRGB-encoded) bytes. Prefer an
    // sRGB swapchain so those bytes can be stored verbatim (the blit uses the
    // UNORM view to avoid a second encode). A linear UNORM swapchain is the
    // fallback: the blit then decodes sRGB -> linear, because the compositor
    // treats UNORM values as linear and would otherwise show them too bright and
    // washed out.
    int64_t chosenFormat = 0;
    const int64_t kSrgbFormats[] = { DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB };
    const int64_t kLinearFormats[] = { DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM };
    bool found = false;
    for (int64_t want : kSrgbFormats) {
        for (int64_t f : formats) {
            if (f == want) {
                chosenFormat = f;
                found = true;
                break;
            }
        }
        if (found) {
            break;
        }
    }
    if (!found) {
        for (int64_t want : kLinearFormats) {
            for (int64_t f : formats) {
                if (f == want) {
                    chosenFormat = f;
                    found = true;
                    break;
                }
            }
            if (found) {
                break;
            }
        }
    }
    if (!found) {
        chosenFormat = formats.empty() ? DXGI_FORMAT_R8G8B8A8_UNORM_SRGB : formats[0];
        SPDLOG_WARN("[VR] Runtime offered no R8G8B8A8/B8G8R8A8 format; using format {}. Colour may be off.",
                    chosenFormat);
    }

    for (int e = 0; e < 2; ++e) {
        EyeSwapchain& sc = eyes[e];
        sc.width = width;
        sc.height = height;
        sc.format = chosenFormat;

        XrSwapchainCreateInfo sci{ XR_TYPE_SWAPCHAIN_CREATE_INFO };
        sci.usageFlags =
            XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
        sci.format = chosenFormat;
        sci.sampleCount = 1;
        sci.width = width;
        sci.height = height;
        sci.faceCount = 1;
        sci.arraySize = 1;
        sci.mipCount = 1;

        if (Failed(xrCreateSwapchain(session, &sci, &sc.handle), "xrCreateSwapchain")) {
            return false;
        }

        uint32_t imgCount = 0;
        xrEnumerateSwapchainImages(sc.handle, 0, &imgCount, nullptr);
        XrSwapchainImageD3D11KHR proto{ XR_TYPE_SWAPCHAIN_IMAGE_D3D11_KHR };
        sc.images.assign(imgCount, proto);
        if (Failed(xrEnumerateSwapchainImages(sc.handle, imgCount, &imgCount,
                                             reinterpret_cast<XrSwapchainImageBaseHeader*>(sc.images.data())),
                   "xrEnumerateSwapchainImages")) {
            return false;
        }
    }

    renderWidth = width;
    renderHeight = height;
    CreateEyeTextures(width, height);
    SPDLOG_INFO("[VR] Swapchains created {}x{} (format {}).", width, height, chosenFormat);
    return true;
}

// Intermediate eye targets the game renders into. The presentation step copies
// these into the XR swapchains, decoupling the game's render rate from the
// headset's frame rate.
void Context::CreateEyeTextures(uint32_t width, uint32_t height) {
    for (int e = 0; e < 2; ++e) {
        eyeSrv[e].Reset();
        eyeTex[e].Reset();
        if (!device) {
            continue;
        }
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM; // matches the game framebuffer
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_DEFAULT;
        // Render target so the ImGui menu can be composited onto the eye texture.
        desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
        if (SUCCEEDED(device->CreateTexture2D(&desc, nullptr, eyeTex[e].GetAddressOf()))) {
            device->CreateShaderResourceView(eyeTex[e].Get(), nullptr, eyeSrv[e].GetAddressOf());
        }
    }
}

// (Re)creates the offscreen menu texture at the desktop window resolution.
bool Context::EnsureMenuTexture(uint32_t w, uint32_t h) {
    if (w < 2 || h < 2 || !device) {
        return false;
    }
    if (menuTex && menuWidth == w && menuHeight == h) {
        return menuRtv != nullptr && menuSrv != nullptr;
    }
    menuSrv.Reset();
    menuRtv.Reset();
    menuTex.Reset();
    menuWidth = w;
    menuHeight = h;
    D3D11_TEXTURE2D_DESC desc{};
    desc.Width = w;
    desc.Height = h;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    if (FAILED(device->CreateTexture2D(&desc, nullptr, menuTex.GetAddressOf()))) {
        return false;
    }
    device->CreateShaderResourceView(menuTex.Get(), nullptr, menuSrv.GetAddressOf());
    device->CreateRenderTargetView(menuTex.Get(), nullptr, menuRtv.GetAddressOf());
    return menuRtv != nullptr && menuSrv != nullptr;
}

// Composites the offscreen ImGui menu (premultiplied alpha) over both eye
// textures, scaling it from the window resolution to the eye resolution.
void Context::CompositeUiToEyes() {
    if (!menuSrv || !eyeTex[0] || !eyeTex[1] || renderWidth < 2 || renderHeight < 2) {
        return;
    }
    for (int e = 0; e < 2; ++e) {
        blitter.Blit(device, context, menuSrv.Get(), eyeTex[e].Get(), DXGI_FORMAT_R8G8B8A8_UNORM, renderWidth,
                     renderHeight, /*decodeSrgb=*/false, /*premul=*/true);
    }
}

void Context::DestroySwapchains() {
    for (auto& sc : eyes) {
        sc.images.clear();
        if (sc.handle != XR_NULL_HANDLE) {
            xrDestroySwapchain(sc.handle);
            sc.handle = XR_NULL_HANDLE;
        }
        sc.acquired = false;
    }
}

bool Context::Initialize() {
    if (initialized) {
        return true;
    }

    auto ship = Ship::Context::GetRawInstance();
    if (!ship) {
        return false;
    }
    auto wnd = std::dynamic_pointer_cast<Fast::Fast3dWindow>(ship->GetWindow());
    if (!wnd) {
        return false;
    }
    auto interp = wnd->GetInterpreterWeak().lock();
    if (!interp) {
        return false;
    }

    auto* rapi = dynamic_cast<Fast::GfxRenderingAPIDX11*>(interp->GetCurrentRenderingAPI());
    if (rapi == nullptr) {
        SPDLOG_WARN("[VR] Only the DirectX 11 backend supports VR; VR disabled.");
        return false;
    }
    device = rapi->mDevice.Get();
    context = rapi->mContext.Get();
    if (!device || !context) {
        return false;
    }
    blitter.Init(device);

    if (!CreateInstance()) {
        return false;
    }
    if (!EnsureRuntimeAdapter(this)) {
        return false; // a relaunch was started
    }
    if (!CreateSession()) {
        return false;
    }

    // Controller input is optional: if it fails, VR still works (keyboard/gamepad).
    CreateActions();

    // Initial swapchain size; the real aspect is captured from the game camera
    // on the first rendered frame (see EnsureRenderTargets).
    uint32_t w = kEyeRenderWidth;
    uint32_t h = (uint32_t)std::lround(kEyeRenderWidth * 0.5625); // 16:9 placeholder
    if (!CreateSwapchains(w, h)) {
        return false;
    }

    initialized = true;
    SPDLOG_INFO("[VR] OpenXR session initialised.");
    return true;
}

void Context::Shutdown() {
    if (!initialized && instance == XR_NULL_HANDLE) {
        return;
    }
    DestroySwapchains();
    menuSrv.Reset();
    menuRtv.Reset();
    menuTex.Reset();
    menuWidth = 0;
    menuHeight = 0;
    if (viewSpace != XR_NULL_HANDLE) {
        xrDestroySpace(viewSpace);
        viewSpace = XR_NULL_HANDLE;
    }
    if (localSpace != XR_NULL_HANDLE) {
        xrDestroySpace(localSpace);
        localSpace = XR_NULL_HANDLE;
    }
    if (session != XR_NULL_HANDLE) {
        xrDestroySession(session);
        session = XR_NULL_HANDLE;
    }
    if (instance != XR_NULL_HANDLE) {
        xrDestroyInstance(instance);
        instance = XR_NULL_HANDLE;
    }
    sessionRunning = false;
    initialized = false;
}

void Context::PollEvents() {
    if (instance == XR_NULL_HANDLE) {
        return;
    }
    XrEventDataBuffer ev{ XR_TYPE_EVENT_DATA_BUFFER };
    while (xrPollEvent(instance, &ev) == XR_SUCCESS) {
        if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
            auto* changed = reinterpret_cast<XrEventDataSessionStateChanged*>(&ev);
            sessionState = changed->state;
            SPDLOG_INFO("[VR] Session state -> {}", (int)sessionState);
            // Any real state change is a good moment to stop backing off.
            ResumePumpIfPaused();
            switch (sessionState) {
                case XR_SESSION_STATE_READY: {
                    XrSessionBeginInfo bi{ XR_TYPE_SESSION_BEGIN_INFO };
                    bi.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                    if (XR_SUCCEEDED(xrBeginSession(session, &bi))) {
                        sessionRunning = true;
                        SPDLOG_INFO("[VR] Session running.");
                    }
                    break;
                }
                case XR_SESSION_STATE_STOPPING:
                    xrEndSession(session);
                    sessionRunning = false;
                    break;
                case XR_SESSION_STATE_EXITING:
                case XR_SESSION_STATE_LOSS_PENDING:
                    // The runtime is going away (headset removed / SteamVR quit).
                    // Stop feeding it so the game falls back to the desktop and
                    // never calls into a dead session.
                    exitRequested = true;
                    sessionRunning = false;
                    break;
                default:
                    break;
            }
        } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
            exitRequested = true;
            sessionRunning = false;
        } else if (ev.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
            // The runtime recentered/reconfigured tracking; re-anchor the screen.
            didInitialRecenter = false;
            SPDLOG_INFO("[VR] Reference space changed; re-centering the screen.");
        } else if (ev.type == XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED) {
            // Log which controller profile the runtime bound (useful when a
            // headset streams through e.g. Virtual Desktop + SteamVR).
            for (const char* hand : { "/user/hand/left", "/user/hand/right" }) {
                XrPath handPath = XR_NULL_PATH;
                xrStringToPath(instance, hand, &handPath);
                XrInteractionProfileState st{ XR_TYPE_INTERACTION_PROFILE_STATE };
                if (XR_SUCCEEDED(xrGetCurrentInteractionProfile(session, handPath, &st)) &&
                    st.interactionProfile != XR_NULL_PATH) {
                    char buf[XR_MAX_PATH_LENGTH] = "";
                    uint32_t len = 0;
                    xrPathToString(instance, st.interactionProfile, sizeof(buf), &len, buf);
                    SPDLOG_INFO("[VR] {} interaction profile: {}", hand, buf);
                }
            }
        }
        ev = { XR_TYPE_EVENT_DATA_BUFFER };
    }
}

void Context::AcquireEye(int e) {
    EyeSwapchain& sc = eyes[e];
    eyeValid[e] = false;
    if (sc.handle == XR_NULL_HANDLE || sc.acquired) {
        return;
    }
    XrSwapchainImageAcquireInfo ai{ XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
    if (Failed(xrAcquireSwapchainImage(sc.handle, &ai, &sc.acquiredIndex), "xrAcquireSwapchainImage")) {
        return;
    }
    // A finite timeout: if the runtime can't hand over the image in time (e.g.
    // the compositor stopped consuming because the headset went to sleep), give
    // up on this eye instead of blocking the whole loop.
    XrSwapchainImageWaitInfo wi{ XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
    wi.timeout = 10 * 1000 * 1000; // 10 ms
    const XrResult r = xrWaitSwapchainImage(sc.handle, &wi);
    if (r == XR_TIMEOUT_EXPIRED) {
        XrSwapchainImageReleaseInfo ri{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
        xrReleaseSwapchainImage(sc.handle, &ri);
        static bool warned = false;
        if (!warned) {
            SPDLOG_WARN("[VR] xrWaitSwapchainImage timed out (eye {}); skipping frame.", e);
            warned = true;
        }
        return;
    }
    if (Failed(r, "xrWaitSwapchainImage")) {
        return;
    }
    sc.acquired = true;
    eyeValid[e] = true;
}

void Context::ReleaseEye(int e) {
    EyeSwapchain& sc = eyes[e];
    if (sc.handle == XR_NULL_HANDLE || !sc.acquired) {
        return;
    }
    XrSwapchainImageReleaseInfo ri{ XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
    xrReleaseSwapchainImage(sc.handle, &ri);
    sc.acquired = false;
}

bool Context::BeginFrame() {
    if (!sessionRunning || frameBegun) {
        return false;
    }
    XrFrameWaitInfo wi{ XR_TYPE_FRAME_WAIT_INFO };
    frameState = { XR_TYPE_FRAME_STATE };
    if (Failed(xrWaitFrame(session, &wi, &frameState), "xrWaitFrame")) {
        return false;
    }
    XrFrameBeginInfo bi{ XR_TYPE_FRAME_BEGIN_INFO };
    if (Failed(xrBeginFrame(session, &bi), "xrBeginFrame")) {
        return false;
    }
    if (frameState.predictedDisplayPeriod > 0) {
        hmdHz = 1.0e9f / (float)frameState.predictedDisplayPeriod;
        // Take the highest rate the runtime reports: it under-reports while
        // starting/idle (30/60), and the port RESETS its interpolation clock
        // whenever the target FPS changes, so we lock onto the peak and hold it.
        const uint32_t hz = (uint32_t)std::lround(hmdHz);
        if (hz >= 60u && hz > targetFps) {
            targetFps = hz;
            SPDLOG_INFO("[VR] Frame pacing target {} Hz.", targetFps);
        }
    }
    SyncActions();
    frameBegun = true;
    AcquireEye(0);
    AcquireEye(1);
    return true;
}

void Context::SubmitLayers(bool submitContent) {
    XrCompositionLayerQuad quads[2];
    const XrCompositionLayerBaseHeader* layers[2] = { nullptr, nullptr };
    uint32_t layerCount = 0;

    if (submitContent) {
        const XrEyeVisibility vis[2] = { XR_EYE_VISIBILITY_LEFT, XR_EYE_VISIBILITY_RIGHT };
        for (int e = 0; e < 2; ++e) {
            if (!eyeValid[e] || eyes[e].handle == XR_NULL_HANDLE) {
                continue; // never submit an image we did not acquire+release
            }
            XrCompositionLayerQuad& q = quads[e];
            q = { XR_TYPE_COMPOSITION_LAYER_QUAD };
            q.layerFlags = 0;
            q.space = localSpace;
            q.eyeVisibility = vis[e];
            q.subImage.swapchain = eyes[e].handle;
            q.subImage.imageRect.offset = { 0, 0 };
            q.subImage.imageRect.extent.width = (int32_t)eyes[e].width;
            q.subImage.imageRect.extent.height = (int32_t)eyes[e].height;
            q.pose = screenPose;
            q.size = screenSize;
            layers[layerCount++] = reinterpret_cast<const XrCompositionLayerBaseHeader*>(&q);
        }
    }

    XrFrameEndInfo fei{ XR_TYPE_FRAME_END_INFO };
    fei.displayTime = frameState.predictedDisplayTime;
    fei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
    fei.layerCount = layerCount;
    fei.layers = layerCount > 0 ? layers : nullptr;
    Failed(xrEndFrame(session, &fei), "xrEndFrame");
}

void Context::EndFrame(bool submitContent) {
    if (!frameBegun) {
        return;
    }
    ReleaseEye(0);
    ReleaseEye(1);
    SubmitLayers(submitContent && haveContent);
    frameBegun = false;
}

// ---------------------------------------------------------------------------
// Engine integration
// ---------------------------------------------------------------------------

// The game framebuffer holds display-referred (already sRGB-encoded) bytes, and
// the desktop path shows them verbatim on a UNORM swapchain. On an sRGB XR
// swapchain, rendering with an sRGB target view would encode them a second time
// (washed out), so use the UNORM view of the (typeless) swapchain texture and
// store the bytes verbatim. On a linear UNORM XR swapchain the blit decodes
// sRGB -> linear instead; see BlitEyeTexToSwapchain.
DXGI_FORMAT UnormViewFormat(int64_t fmt) {
    switch (fmt) {
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
            return DXGI_FORMAT_R8G8B8A8_UNORM;
        case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
            return DXGI_FORMAT_B8G8R8A8_UNORM;
        default:
            return (DXGI_FORMAT)fmt;
    }
}

// Copies the game's current framebuffer into this eye's intermediate texture.
void CopyGameFbToEyeTex(Fast::Interpreter* interp, Context* c, int eye) {
    uintptr_t fb = interp->mGfxFrameBuffer;
    if (fb == 0 || !c->eyeTex[eye]) {
        return;
    }
    auto* srv = reinterpret_cast<ID3D11ShaderResourceView*>(fb);
    ComPtr<ID3D11Resource> res;
    srv->GetResource(res.GetAddressOf());
    ComPtr<ID3D11Texture2D> src;
    if (res) {
        res.As(&src);
    }
    if (!src) {
        return;
    }
    D3D11_TEXTURE2D_DESC sd{}, dd{};
    src->GetDesc(&sd);
    c->eyeTex[eye]->GetDesc(&dd);
    ID3D11DeviceContext* ctx = c->DeviceContext();
    if (sd.Width == dd.Width && sd.Height == dd.Height && sd.Format == dd.Format) {
        ctx->CopyResource(c->eyeTex[eye].Get(), src.Get());
    } else if (c->blitter.ready) {
        c->blitter.Blit(c->Device(), ctx, srv, c->eyeTex[eye].Get(), DXGI_FORMAT_R8G8B8A8_UNORM, c->renderWidth,
                        c->renderHeight);
    }
}

// Copies an eye's intermediate texture into its acquired XR swapchain image.
void BlitEyeTexToSwapchain(Context* c, int eye) {
    EyeSwapchain& sc = c->eyes[eye];
    if (!sc.acquired || !c->eyeSrv[eye] || !c->blitter.ready) {
        return;
    }
    ID3D11Texture2D* dst = sc.images[sc.acquiredIndex].texture;
    if (!dst) {
        return;
    }
    // sRGB swapchain: write the game's display-referred bytes verbatim through
    // its UNORM view (no double encode). Linear UNORM swapchain: decode them so
    // the compositor sees linear light instead of washed-out sRGB values.
    c->blitter.Blit(c->Device(), c->DeviceContext(), c->eyeSrv[eye].Get(), dst, UnormViewFormat(sc.format), sc.width,
                    sc.height, !IsSrgbFormat(sc.format));
}

// Converts a packed N64 fixed-point matrix to floats, using the exact layout the
// interpreter reads (see Interpreter::GfxSpMatrix). Avoids depending on the
// game's guMtxL2F and its Mtx typedef.
void MatrixL2F(const Mtx* m, float out[4][4]) {
    const int32_t* addr = reinterpret_cast<const int32_t*>(m);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j += 2) {
            int32_t intPart = addr[i * 2 + j / 2];
            uint32_t fracPart = addr[8 + i * 2 + j / 2];
            out[i][j] = (int32_t)((intPart & 0xffff0000) | (fracPart >> 16)) / 65536.0f;
            out[i][j + 1] = (int32_t)((intPart << 16) | (fracPart & 0xffff)) / 65536.0f;
        }
    }
}

// Fills in the perspective cameras not present in the map (key frames) by
// reading their baked matrices. Called once per frame so both eyes share an
// identical base camera (the game thread rewrites camPerspMatrix, so reading it
// twice, once per eye, can make the eyes disagree and the stereo jump).
void EnsureCameraMatrices(std::unordered_map<Mtx*, MtxF>& map) {
    for (int ctx = 0; ctx < 2; ++ctx) {
        for (int cam : kPerspectiveCameras) {
            if (gCameras[cam].flags == 0) {
                continue;
            }
            Mtx* key = reinterpret_cast<Mtx*>(&DisplayContexts[ctx].camPerspMatrix[cam]);
            if (map.find(key) == map.end()) {
                MtxF value{};
                MatrixL2F(key, value.mf);
                map[key] = value;
            }
        }
    }
}

// Moves one eye by `worldOffset` game units along the camera's right axis.
//
// The game stores the combined matrix Pc = view * projection (camPerspMatrix).
// A world-space eye translation is a pre-multiplication by a translation matrix
// T (row-vector convention), which only changes ROW 3 of Pc:
//     Pc'[3][j] = Pc[3][j] + t0*Pc[0][j] + t1*Pc[1][j] + t2*Pc[2][j]
// with t = -offset. This yields true parallax (depth-dependent), unlike a
// post-projection shift which just translates the whole image.
void ApplyEyeOffset(MtxF& value, float worldOffset) {
    // Camera right axis in world is proportional to column 0 of Pc (= V*P;
    // P's column 0 is (cot/aspect,0,0,0)). Using Pc itself keeps the baseline
    // consistent with the exact matrix being rendered.
    float rx = value.mf[0][0], ry = value.mf[1][0], rz = value.mf[2][0];
    float len = std::sqrt(rx * rx + ry * ry + rz * rz);
    if (len > 1e-6f) {
        rx /= len;
        ry /= len;
        rz /= len;
    } else {
        rx = 1.0f;
        ry = 0.0f;
        rz = 0.0f;
    }
    const float tx = -worldOffset * rx;
    const float ty = -worldOffset * ry;
    const float tz = -worldOffset * rz;
    for (int j = 0; j < 4; ++j) {
        value.mf[3][j] += tx * value.mf[0][j] + ty * value.mf[1][j] + tz * value.mf[2][j];
    }
}

void ApplyEyeOffset(std::unordered_map<Mtx*, MtxF>& map, float worldOffset) {
    for (int ctx = 0; ctx < 2; ++ctx) {
        for (int cam : kPerspectiveCameras) {
            Mtx* key = reinterpret_cast<Mtx*>(&DisplayContexts[ctx].camPerspMatrix[cam]);
            auto it = map.find(key);
            if (it != map.end()) {
                ApplyEyeOffset(it->second, worldOffset);
            }
        }
    }
}

// NDC x of a world point under a combined matrix (clip = p * Pc).
float NdcX(const MtxF& m, float x, float y, float z) {
    const float cx = x * m.mf[0][0] + y * m.mf[1][0] + z * m.mf[2][0] + m.mf[3][0];
    const float cw = x * m.mf[0][3] + y * m.mf[1][3] + z * m.mf[2][3] + m.mf[3][3];
    return (cw != 0.0f) ? cx / cw : 0.0f;
}

// Applies a constant NDC x shift by shearing clip.x proportionally to clip.w
// (mf[i][0] += ndc * mf[i][3]). This moves the stereo zero-parallax plane: a
// per-eye shift s places zero disparity at the depth that previously had
// disparity -s, instead of at infinity. Unlike a uniform clip-space translation
// it is the same shift at every depth, which is what sets a convergence plane.
void ApplyConvergence(MtxF& value, float ndc) {
    if (ndc == 0.0f) {
        return;
    }
    for (int i = 0; i < 4; ++i) {
        value.mf[i][0] += ndc * value.mf[i][3];
    }
}

// The per-eye NDC shift that puts the stereo zero-parallax plane on the active
// camera's look-at target, so the subject sits on the screen plane and the world
// mostly recedes behind the on-screen UI instead of crossing it. Outer cameras
// share the same shift (a constant NDC shift is a constant visual angle).
float ComputeZeroParallaxNdc(const std::unordered_map<Mtx*, MtxF>& base, float half) {
    int cam = gCurrentCamID;
    if (cam < 0 || cam > 2) {
        cam = 0; // CAM_DEFAULT
    }
    if (gCameras[cam].flags == 0) {
        return 0.0f;
    }
    Mtx* key = reinterpret_cast<Mtx*>(&DisplayContexts[gCurrentDisplayContextIndex].camPerspMatrix[cam]);
    auto it = base.find(key);
    if (it == base.end()) {
        return 0.0f;
    }
    MtxF right = it->second;
    MtxF left = it->second;
    ApplyEyeOffset(right, +half);
    ApplyEyeOffset(left, -half);
    const Vec3f& p = gCameras[cam].lookAt_obj;
    const float ndcR = NdcX(right, p.x, p.y, p.z);
    const float ndcL = NdcX(left, p.x, p.y, p.z);
    // Applied +to eye 1 and -to eye 0, this cancels the target's disparity.
    const float ndc = (ndcL - ndcR) * 0.5f;
    if (!std::isfinite(ndc)) {
        return 0.0f;
    }
    return std::clamp(ndc, -0.5f, 0.5f);
}

// Renders one eye into mGameFb (reused) and copies it to this eye's intermediate
// texture. The presentation step later copies that into the headset swapchain.
void RenderEyeToTex(Fast::Interpreter* interp, Context* c, int eye, Gfx* commands,
                    const std::unordered_map<Mtx*, MtxF>& baseMap, float eyeOffset, float convergence) {
    std::unordered_map<Mtx*, MtxF> map = baseMap;
    ApplyEyeOffset(map, eyeOffset);
    if (convergence != 0.0f) {
        for (int ctx = 0; ctx < 2; ++ctx) {
            for (int cam : kPerspectiveCameras) {
                Mtx* key = reinterpret_cast<Mtx*>(&DisplayContexts[ctx].camPerspMatrix[cam]);
                auto it = map.find(key);
                if (it != map.end()) {
                    ApplyConvergence(it->second, convergence);
                }
            }
        }
    }

    // Run() only clears depth, so clear colour ourselves to avoid the previous
    // eye leaking through.
    auto* rapi = interp->GetCurrentRenderingAPI();
    rapi->StartDrawToFramebuffer(interp->mGameFb, 1.0f);
    rapi->ClearFramebuffer(true, true);

    // The background sits at infinity, so it gets effectively no camera parallax
    // (and is drawn as 2D rects that bypass the camera matrices entirely). Give
    // it a small per-eye screen-space shift so it recedes behind the
    // zero-parallax plane: left eye left, right eye right. The interpreter
    // applies this only to rects in the registered background display list.
    interp->mBackgroundRectOffsetX = (eye == 0 ? -g_bgParallaxPx : g_bgParallaxPx) * 4;

    interp->Run(commands, map, {});

    CopyGameFbToEyeTex(interp, c, eye);
}

// Makes the next Run() render offscreen into mGameFb at the interpreter's
// current dimensions. We deliberately do NOT change mCurDimensions: the game
// derives its widescreen aspect from it, and mutating it fought the game's own
// logic (aspect flickering between 4:3/16:9).
void PrepareOffscreen(Fast::Interpreter* interp) {
    interp->mMsaaLevel = 1;
    interp->StartFrame();
    // StartFrame may have decided not to render offscreen (window == native
    // resolution). Force it: size mGameFb to the game's dimensions and target it.
    const uint32_t w = interp->mCurDimensions.width;
    const uint32_t h = interp->mCurDimensions.height;
    auto* rapi = interp->GetCurrentRenderingAPI();
    rapi->UpdateFramebufferParameters(interp->mGameFb, w, h, 1, true, true, true, true);
    interp->mRendersToFb = true;

    rapi->StartDrawToFramebuffer(interp->mGameFb, 1.0f);
    rapi->ClearFramebuffer(true, true);
}

// Re-creates the swapchains (and eye targets) when VR's render size changes.
//
// The game derives its render size from the desktop window every frame
// (Fast3dGui: size * dpiScale * internal_mul), so while the window is being
// resized - or the internal-resolution slider is dragged, or the game eases the
// widescreen aspect - mCurDimensions changes on every single frame. Rebuilding
// the cross-process XR swapchains and the game framebuffer on each of those
// frames costs tens of ms and shows up as frame drops, so only commit a new size
// once it has been stable for a short while. While a change is pending we keep
// rendering at the last committed size; the resolution cap keeps the aspect
// fixed, so nothing visibly changes during the hold.
void EnsureRenderTargets(Context* c, Fast::Interpreter* interp) {
    uint32_t w = interp->mCurDimensions.width;
    uint32_t h = interp->mCurDimensions.height;
    if (w < 2 || h < 2) {
        return;
    }

    static uint32_t s_rtPendingW = 0;
    static uint32_t s_rtPendingH = 0;
    static int s_rtStableFrames = 0;

    auto commit = [&](uint32_t cw, uint32_t ch) {
        c->CreateSwapchains(cw, ch);
        // Screen aspect follows the game render aspect.
        c->renderAspect = (float)cw / (float)ch;
        c->screenSize.height = c->screenSize.width * (float)ch / (float)cw;
    };

    // First setup always commits immediately.
    if (c->renderWidth == 0 || c->renderHeight == 0) {
        commit(w, h);
        s_rtPendingW = s_rtPendingH = 0;
        s_rtStableFrames = 0;
        return;
    }

    if (w == c->renderWidth && h == c->renderHeight) {
        s_rtPendingW = s_rtPendingH = 0;
        s_rtStableFrames = 0;
        return;
    }

    // Ignore small tweaks, e.g. the couple-of-percent height wobble the game's
    // aspect handling produces. Rebuilding the swapchains and eye textures for a
    // 1-2% aspect change is expensive and invisible (the blit scales anyway), so
    // hold the committed size and keep drawing at it.
    constexpr float kSizeTolerance = 0.05f;
    const float dw = std::abs((float)w - (float)c->renderWidth) / (float)c->renderWidth;
    const float dh = std::abs((float)h - (float)c->renderHeight) / (float)c->renderHeight;
    if (dw < kSizeTolerance && dh < kSizeTolerance) {
        interp->mCurDimensions.width = c->renderWidth;
        interp->mCurDimensions.height = c->renderHeight;
        s_rtPendingW = s_rtPendingH = 0;
        s_rtStableFrames = 0;
        return;
    }

    // RenderGameFrame runs once per interpolated sub-frame, so this is a handful
    // of sub-frames of stability, not seconds.
    constexpr int kStableFrames = 6;
    if (w == s_rtPendingW && h == s_rtPendingH) {
        ++s_rtStableFrames;
    } else {
        s_rtPendingW = w;
        s_rtPendingH = h;
        s_rtStableFrames = 0;
    }

    if (s_rtStableFrames >= kStableFrames) {
        commit(w, h);
        s_rtPendingW = s_rtPendingH = 0;
        s_rtStableFrames = 0;
        return;
    }

    // Hold the last committed size so the game framebuffer isn't reallocated
    // mid-transition (StartFrame resizes mGameFb whenever mCurDimensions moves).
    interp->mCurDimensions.width = c->renderWidth;
    interp->mCurDimensions.height = c->renderHeight;
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------
bool KeyEdge(int vk) {
    static bool down[256] = { false };
    const bool now = (GetAsyncKeyState(vk) & 0x8000) != 0;
    const bool edge = now && !down[vk & 0xFF];
    down[vk & 0xFF] = now;
    return edge;
}

bool CtrlAlt() {
    return (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_MENU) & 0x8000);
}

// PAPERBOAT_VR_DEBUG=1 forces the VR render path even when the headset isn't
// worn/focused, so the frame loop and perf logging can be exercised without a
// headset (no more repeated in-headset testing while debugging).
bool VrDebugEnabled() {
    static const bool enabled = std::getenv("PAPERBOAT_VR_DEBUG") != nullptr;
    return enabled;
}

// Optional integer env override (0 == unset).
int EnvInt(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? std::atoi(v) : 0;
}

// Optional float env override (0 == unset).
float EnvFloat(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr ? (float)std::atof(v) : 0.0f;
}

// Places the screen in front of the head. Returns false if the head pose is not
// tracked yet (retry later).
bool TryRecenter() {
    if (g_ctx == nullptr || !g_ctx->initialized || g_ctx->viewSpace == XR_NULL_HANDLE) {
        return false;
    }
    XrSpaceLocation loc{ XR_TYPE_SPACE_LOCATION };
    if (XR_FAILED(xrLocateSpace(g_ctx->viewSpace, g_ctx->localSpace, 0, &loc)) ||
        !(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) ||
        !(loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT)) {
        return false;
    }
    float fwd[3];
    RotateByQuat(loc.pose.orientation, 0, 0, -1, fwd);
    float hx = fwd[0], hz = fwd[2];
    float len = std::sqrt(hx * hx + hz * hz);
    if (len < 1e-4f) {
        hx = 0.0f;
        hz = -1.0f;
        len = 1.0f;
    }
    hx /= len;
    hz /= len;

    const float dist = GetScreenDistance();
    g_ctx->screenAnchor = { loc.pose.position.x, loc.pose.position.y, loc.pose.position.z };
    g_ctx->screenDir = { hx, 0.0f, hz };
    XrPosef pose{};
    pose.position.x = loc.pose.position.x + hx * dist;
    pose.position.y = loc.pose.position.y + g_screenVerticalOffset;
    pose.position.z = loc.pose.position.z + hz * dist;
    pose.orientation = YawQuat(std::atan2(-hx, -hz));
    g_ctx->screenPose = pose;
    SPDLOG_INFO(
        "[VR] Recenter: head=({:.2f},{:.2f},{:.2f}) -> screen=({:.2f},{:.2f},{:.2f}) yaw={:.3f}", loc.pose.position.x,
        loc.pose.position.y, loc.pose.position.z, pose.position.x, pose.position.y, pose.position.z,
        std::atan2(-hx, -hz));
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// Settings persistence (console variables)
// ---------------------------------------------------------------------------
constexpr const char* kCvarEnabled = "gVR.Enabled";
constexpr const char* kCvarScreenSize = "gVR.ScreenSize";
constexpr const char* kCvarScreenDistance = "gVR.ScreenDistance";
constexpr const char* kCvarScreenVerticalOffset = "gVR.ScreenVerticalOffset";
constexpr const char* kCvarFacePlayer = "gVR.FacePlayer";
constexpr const char* kCvarStereoStrength = "gVR.StereoStrength";
constexpr const char* kCvarBackgroundParallax = "gVR.BackgroundParallax";

// Reads a float CVar in a type-tolerant way. The port's CVarGetFloat returns the
// DEFAULT whenever the stored value's type isn't Float -- which happens when a
// whole-number float (e.g. 8.0) gets serialised as a JSON integer and reloaded as
// an Integer CVar. Reading the raw entry avoids silently dropping the user's
// saved value (which made the screen always render at the default distance).
float CvarFloat(const char* name, float fallback) {
    auto ship = Ship::Context::GetRawInstance();
    if (ship == nullptr) {
        return fallback;
    }
    auto cvars = ship->GetConsoleVariables();
    if (cvars == nullptr) {
        return fallback;
    }
    auto var = cvars->Get(name);
    if (var == nullptr) {
        return fallback;
    }
    if (var->Type == Ship::ConsoleVariableType::Float) {
        return var->Float;
    }
    if (var->Type == Ship::ConsoleVariableType::Integer) {
        return (float)var->Integer;
    }
    return fallback;
}

// Loads persisted settings into the module globals. Lazy + idempotent so it
// works no matter whether the settings menu is built before or after the OpenXR
// session is created (and even when there is no runtime/headset at all).
void LoadSettings() {
    static bool loaded = false;
    if (loaded || Ship::Context::GetRawInstance() == nullptr) {
        return;
    }
    loaded = true;
    g_enabled = CVarGetInteger(kCvarEnabled, 1) != 0;
    g_screenWidth = CvarFloat(kCvarScreenSize, kDefaultScreenSize);
    g_screenDistance = CvarFloat(kCvarScreenDistance, kDefaultScreenDistance);
    g_screenVerticalOffset = std::clamp(CvarFloat(kCvarScreenVerticalOffset, kDefaultScreenVerticalOffset), -5.0f, 5.0f);
    g_facePlayer = CVarGetInteger(kCvarFacePlayer, 1) != 0;
    g_stereoStrength = CvarFloat(kCvarStereoStrength, kDefaultStereoStrength);
    g_bgParallaxPx = (int)CvarFloat(kCvarBackgroundParallax, 2.0f);
    SPDLOG_INFO("[VR] Settings loaded: enabled={} screenSize={:.2f}m distance={:.2f}m offset={:.2f}m facePlayer={} "
                "stereo={:.2f}",
                g_enabled, g_screenWidth, g_screenDistance, g_screenVerticalOffset, g_facePlayer, g_stereoStrength);
}

// Pushes the loaded settings into the live session (screen size + distance).
void ApplySettingsToContext() {
    if (g_ctx == nullptr) {
        return;
    }
    const float aspect = g_ctx->renderWidth > 0 ? (float)g_ctx->renderHeight / (float)g_ctx->renderWidth : 0.75f;
    g_ctx->screenSize.width = g_screenWidth;
    g_ctx->screenSize.height = g_screenWidth * aspect;
    g_ctx->screenPose.position.y = g_screenVerticalOffset;
    g_ctx->screenPose.position.z = -g_screenDistance;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
bool Initialize() {
    LoadSettings();
    StartWatchdog();
    if (g_ctx != nullptr && g_ctx->initialized) {
        return true;
    }
    if (g_ctx == nullptr) {
        g_ctx = new Context();
    }
    if (!g_ctx->Initialize()) {
        g_ctx->Shutdown();
        delete g_ctx;
        g_ctx = nullptr;
        return false;
    }
    ApplySettingsToContext();
    return true;
}

void Shutdown() {
    StopWatchdog();
    UpdateVrInputBlock(false);
    if (g_ctx != nullptr) {
        g_ctx->Shutdown();
        delete g_ctx;
        g_ctx = nullptr;
    }
}

bool IsSessionRunning() {
    return g_ctx != nullptr && g_ctx->sessionRunning;
}

void PollEvents() {
    BeatMainLoop();
    if (g_ctx == nullptr) {
        return;
    }
    g_ctx->PollEvents();
    // Keep VR-controller input live even while the pump is paused for a stall,
    // otherwise the actions stop syncing and the last value stays latched.
    if (g_pumpPaused) {
        g_ctx->SyncInput();
    }
    // The runtime told us it is exiting (SteamVR quit / HMD link dropped for
    // good): tear the OpenXR objects down immediately. Leaving a half-dead
    // session around while the runtime unloads makes the loader/runtime fall over
    // a few seconds later. (A mere sleep sends STOPPING/IDLE, not EXITING, so the
    // session is kept and resumes on wake.)
    if (g_ctx->exitRequested) {
        SPDLOG_INFO("[VR] Runtime exiting; shutting down OpenXR.");
        Shutdown();
    }
}

bool GetEnabled() {
    LoadSettings();
    return g_enabled;
}

void SetEnabled(bool enabled) {
    LoadSettings();
    g_enabled = enabled;
    CVarSetInteger(kCvarEnabled, enabled ? 1 : 0);
    if (!enabled) {
        // VR is off: make sure we are not still blocking game input for the menu.
        UpdateVrInputBlock(false);
    }
}

bool GetVrPad(uint16_t& buttons, int8_t& stickX, int8_t& stickY) {
    if (g_ctx == nullptr || !g_ctx->sessionRunning) {
        return false;
    }
    // While the VR menu is open the controllers drive it, so report no game input.
    if (g_vrMenuInputBlock.load(std::memory_order_relaxed)) {
        buttons = 0;
        stickX = 0;
        stickY = 0;
        return true;
    }
    buttons = g_ctx->vrButtons;
    stickX = g_ctx->vrStickX;
    stickY = g_ctx->vrStickY;
    return true;
}

uint32_t GetTargetFpsOverride() {
    // This is the GAME's render rate (interpolated sub-frames per second), not
    // the headset's: PresentFrame() still submits a frame every display tick
    // regardless. Match the headset refresh so the game produces a fresh frame
    // for (or faster than) every displayed frame instead of a fixed 60 - lower
    // latency, and interpolation always has a sub-frame ready for the compositor.
    // PAPERBOAT_VR_FPS forces a value.
    if (!VrActive()) {
        return 0;
    }
    if (int forced = EnvInt("PAPERBOAT_VR_FPS"); forced > 0) {
        return (uint32_t)forced;
    }
    // targetFps is latched from the runtime's predicted display period (BeginFrame)
    // and is 0 until the first frame, so fall back to 60 until the rate is known.
    if (g_ctx != nullptr && g_ctx->targetFps >= 60) {
        return g_ctx->targetFps;
    }
    return 60;
}

void UpdateResolutionCap(Fast::Interpreter* interp) {
    if (interp == nullptr) {
        return;
    }
    static float s_savedMul = -1.0f;
    static int s_savedVsync = -1;
    const bool vrActive = VrActive();

    // The desktop window's present (vsync + its own frame-interval pacing) runs
    // inside the XR frame loop and throttles the headset, so force it off while
    // VR is active and restore the user's setting afterwards.
    if (vrActive) {
        if (s_savedVsync < 0) {
            s_savedVsync = CVarGetInteger(CVAR_VSYNC_ENABLED, 1);
            CVarSetInteger(CVAR_VSYNC_ENABLED, 0);
        }
    } else if (s_savedVsync >= 0) {
        CVarSetInteger(CVAR_VSYNC_ENABLED, s_savedVsync);
        s_savedVsync = -1;
    }

    if (vrActive) {
        // The port sets the desktop window's max frame latency to 1 each frame,
        // which makes its present block on the monitor's flip queue and throttle
        // the XR loop. Raise it so the present doesn't stall the headset.
        if (auto wnd = std::dynamic_pointer_cast<Fast::Fast3dWindow>(Ship::Context::GetRawInstance()->GetWindow())) {
            wnd->SetMaximumFrameLatency(8);
        }
        if (s_savedMul < 0.0f) {
            s_savedMul = interp->mCurDimensions.internal_mul;
        }
        const float oldMul = interp->mCurDimensions.internal_mul;
        const float curW = (float)interp->mCurDimensions.width;
        float scale = EnvFloat("PAPERBOAT_VR_RESSCALE");
        if (scale <= 0.0f) {
            scale = 1.0f;
        }
        float cap = kMaxEyeRenderWidth * scale;
        if (int forced = EnvInt("PAPERBOAT_VR_MAXW"); forced > 0) {
            cap = (float)forced;
        }
        // Only clamp when clearly over the cap. Rewriting internal_mul every
        // frame makes the render size (and the framebuffer/swapchains that follow
        // it) jitter, which is far more expensive than the few extra pixels.
        if (oldMul > 0.0f && curW > cap * 1.02f) {
            float newMul = oldMul * (cap / curW);
            if (newMul < 0.05f) {
                newMul = 0.05f;
            }
            interp->mCurDimensions.internal_mul = newMul;
        }
    } else if (s_savedMul >= 0.0f) {
        interp->mCurDimensions.internal_mul = s_savedMul;
        s_savedMul = -1.0f;
    }
}

void Recenter() {
    TryRecenter();
}

float GetScreenSize() {
    LoadSettings();
    return g_screenWidth;
}

void SetScreenSize(float metres) {
    LoadSettings();
    metres = std::clamp(metres, 0.25f, 12.0f);
    g_screenWidth = metres;
    CVarSetFloat(kCvarScreenSize, metres);
    if (g_ctx) {
        const float aspect = g_ctx->renderWidth > 0 ? (float)g_ctx->renderHeight / (float)g_ctx->renderWidth : 0.75f;
        g_ctx->screenSize.width = metres;
        g_ctx->screenSize.height = metres * aspect;
    }
}

float GetScreenDistance() {
    LoadSettings();
    return g_screenDistance;
}

// Places the screen `g_screenDistance` along the anchored ray captured at the
// last recenter. Deterministic and independent of live tracking, so the distance
// control always moves the screen and a given value always lands in the same
// place (consistent with the recenter, which also puts it anchor + dir * dist).
void ApplyDistance() {
    if (g_ctx == nullptr) {
        return;
    }
    const float d = g_screenDistance;
    g_ctx->screenPose.position.x = g_ctx->screenAnchor.x + g_ctx->screenDir.x * d;
    g_ctx->screenPose.position.y = g_ctx->screenAnchor.y + g_ctx->screenDir.y * d + g_screenVerticalOffset;
    g_ctx->screenPose.position.z = g_ctx->screenAnchor.z + g_ctx->screenDir.z * d;
}

// Keeps the quad yawed toward the viewer when "face player" is enabled. The
// screen stays world-locked at the anchor (only its orientation changes), so the
// distance/size controls keep behaving exactly as before.
void UpdateScreenPose() {
    if (!g_facePlayer || g_ctx == nullptr || !g_ctx->initialized || g_ctx->viewSpace == XR_NULL_HANDLE) {
        return;
    }
    XrSpaceLocation loc{ XR_TYPE_SPACE_LOCATION };
    if (XR_FAILED(xrLocateSpace(g_ctx->viewSpace, g_ctx->localSpace, g_ctx->frameState.predictedDisplayTime, &loc)) ||
        !(loc.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) ||
        !(loc.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT)) {
        return;
    }
    const float dx = g_ctx->screenPose.position.x - loc.pose.position.x;
    const float dz = g_ctx->screenPose.position.z - loc.pose.position.z;
    const float len = std::sqrt(dx * dx + dz * dz);
    if (len < 1e-4f) {
        return;
    }
    const float hx = dx / len;
    const float hz = dz / len;
    g_ctx->screenPose.orientation = YawQuat(std::atan2(-hx, -hz));
}

void SetScreenDistance(float metres) {
    LoadSettings();
    metres = std::clamp(metres, 0.5f, 15.0f);
    g_screenDistance = metres;
    CVarSetFloat(kCvarScreenDistance, metres);
    ApplyDistance();
}

// Re-reads the persisted VR settings from their CVars and applies them live.
// Called by the settings menu (whose widgets write the CVars directly), so the
// UI and the running session can never disagree.
void ReloadSettings() {
    if (Ship::Context::GetRawInstance() == nullptr) {
        return;
    }
    const bool enabled = CVarGetInteger(kCvarEnabled, 1) != 0;
    const float size = CvarFloat(kCvarScreenSize, kDefaultScreenSize);
    const float dist = CvarFloat(kCvarScreenDistance, kDefaultScreenDistance);
    const float vertical = CvarFloat(kCvarScreenVerticalOffset, kDefaultScreenVerticalOffset);
    const bool facePlayer = CVarGetInteger(kCvarFacePlayer, 1) != 0;
    const float stereo = CvarFloat(kCvarStereoStrength, kDefaultStereoStrength);
    const int bgParallax = (int)CvarFloat(kCvarBackgroundParallax, 2.0f);
    // The settings menu fires its callback every frame, so only apply (and log)
    // when something actually changed.
    if (enabled == g_enabled && size == g_screenWidth && dist == g_screenDistance &&
        vertical == g_screenVerticalOffset && facePlayer == g_facePlayer && stereo == g_stereoStrength &&
        bgParallax == g_bgParallaxPx) {
        return;
    }
    g_enabled = enabled;
    g_screenWidth = size;
    g_screenDistance = dist;
    g_screenVerticalOffset = std::clamp(vertical, -5.0f, 5.0f);
    g_facePlayer = facePlayer;
    g_stereoStrength = stereo;
    g_bgParallaxPx = std::clamp(bgParallax, 0, 64);
    if (g_ctx != nullptr) {
        const float aspect = g_ctx->renderWidth > 0 ? (float)g_ctx->renderHeight / (float)g_ctx->renderWidth : 0.75f;
        g_ctx->screenSize.width = g_screenWidth;
        g_ctx->screenSize.height = g_screenWidth * aspect;
        ApplyDistance();
    }
    SPDLOG_INFO("[VR] ReloadSettings: enabled={} size={:.2f} dist={:.2f} offset={:.2f} facePlayer={} stereo={:.2f}",
                g_enabled, g_screenWidth, g_screenDistance, g_screenVerticalOffset, g_facePlayer, g_stereoStrength);
}

float GetStereoStrength() {
    LoadSettings();
    return g_stereoStrength;
}
void SetStereoStrength(float strength) {
    LoadSettings();
    g_stereoStrength = std::clamp(strength, 0.0f, 20.0f);
    CVarSetFloat(kCvarStereoStrength, g_stereoStrength);
}

float GetConvergence() {
    return g_convergence;
}
void SetConvergence(float) {
    // Zero-parallax trim is intentionally fixed at 0; the menu control was
    // removed. Kept as a no-op for API compatibility.
}

void PollHotkeys() {
    if (!CtrlAlt()) {
        return;
    }
    if (KeyEdge('V')) {
        SetEnabled(!g_enabled);
        SPDLOG_INFO("[VR] VR rendering {}.", g_enabled ? "enabled" : "disabled");
    }
    if (KeyEdge('R')) {
        Recenter();
        SPDLOG_INFO("[VR] Recentered.");
    }
    if (KeyEdge(VK_UP)) {
        SetScreenDistance(GetScreenDistance() - kScreenDistanceStep);
    }
    if (KeyEdge(VK_DOWN)) {
        SetScreenDistance(GetScreenDistance() + kScreenDistanceStep);
    }
    if (KeyEdge(VK_LEFT)) {
        SetScreenSize(GetScreenSize() - kScreenSizeStep);
    }
    if (KeyEdge(VK_RIGHT)) {
        SetScreenSize(GetScreenSize() + kScreenSizeStep);
    }
    if (KeyEdge(VK_OEM_COMMA)) {
        SetStereoStrength(GetStereoStrength() - kStereoStep);
        SPDLOG_INFO("[VR] Stereo strength {}.", GetStereoStrength());
    }
    if (KeyEdge(VK_OEM_PERIOD)) {
        SetStereoStrength(GetStereoStrength() + kStereoStep);
        SPDLOG_INFO("[VR] Stereo strength {}.", GetStereoStrength());
    }
}

// Presents the desktop window's backbuffer without ever blocking. The normal
// present path (Fast::Interpreter::EndFrame) sleeps to the window's target FPS
// and waits on the swapchain's frame-latency object; during VR the game tick and
// the headset pump share this thread, so either wait can starve them and drop the
// game framerate while the headset (fed from cached textures) stays smooth. The
// desktop window is only a spectator mirror, so present the swapchain directly
// with DXGI_PRESENT_DO_NOT_WAIT (a dropped mirror frame is harmless) and skip
// both waits.
void PresentDesktopMirror(Fast::Interpreter* interp) {
#if defined(ENABLE_DX11) || defined(ENABLE_DX12)
    if (interp == nullptr) {
        return;
    }
    auto* dxgi = dynamic_cast<Fast::GfxWindowBackendDXGI*>(interp->mWapi);
    if (dxgi == nullptr) {
        return;
    }
    IDXGISwapChain1* sc = dxgi->GetSwapChain();
    if (sc == nullptr) {
        return;
    }
    UINT flags = DXGI_PRESENT_DO_NOT_WAIT;
    if (dxgi->CanDisableVsync()) {
        flags |= DXGI_PRESENT_ALLOW_TEARING;
    }
    HRESULT hr = sc->Present(0, flags);
    if (hr == DXGI_ERROR_WAS_STILL_DRAWING) {
        return; // mirror frame dropped; the queued frame stays on screen
    }
    if (FAILED(hr) && (flags & DXGI_PRESENT_ALLOW_TEARING)) {
        // ALLOW_TEARING is rejected in some window modes; retry without it so the
        // mirror keeps updating instead of silently freezing.
        sc->Present(0, DXGI_PRESENT_DO_NOT_WAIT);
        return;
    }
    if (FAILED(hr)) {
        static bool s_warned = false;
        if (!s_warned) {
            s_warned = true;
            SPDLOG_WARN("[VR] Desktop mirror present failed (0x{:08X}); mirror may not update.", (uint32_t)hr);
        }
    }
#else
    (void) interp;
#endif
}

bool RenderGameFrame(Gfx* commands, const std::unordered_map<Mtx*, MtxF>& mtxReplacements) {
    if (!g_enabled || g_ctx == nullptr || !g_ctx->initialized) {
        return false;
    }

    g_ctx->PollEvents();
    if (!g_ctx->sessionRunning) {
        return false;
    }

    // Render whenever the session is running (and we are not backing off from a
    // stalled runtime). Some runtimes (SteamVR) only promote the app out of READY
    // once it starts submitting frames, so waiting for VISIBLE/FOCUSED before
    // rendering deadlocks startup. Rendering early is cheap.
    const bool visible = VrActive();
    static bool lastVisible = false;
    if (visible != lastVisible) {
        SPDLOG_INFO("[VR] render-active={} (state {})", visible, (int)g_ctx->sessionState);
        lastVisible = visible;
    }
    if (!visible) {
        // Let the normal desktop path draw; PresentFrame() keeps the session fed.
        UpdateVrInputBlock(false);
        InjectVrImGuiInput(false);
        return false;
    }

    auto wnd = std::dynamic_pointer_cast<Fast::Fast3dWindow>(Ship::Context::GetRawInstance()->GetWindow());
    if (!wnd) {
        return false;
    }
    auto interp = wnd->GetInterpreterWeak().lock().get();
    auto gui = wnd->GetGui();
    if (!interp || !gui) {
        return false;
    }

    // Keep trying to recenter until the head pose is tracked.
    if (!g_ctx->didInitialRecenter) {
        g_ctx->didInitialRecenter = TryRecenter();
    }

    // StartDraw sets mCurDimensions from the window; size the eye targets to match.
    const double t0 = NowMs();
    const bool menuOpen = PortMenuOpen(gui.get());
    UpdateVrInputBlock(menuOpen);
    // Feed the VR controllers to ImGui before the frame is built. The gamepad
    // navigation CVar is forced on for this frame only, then restored, so the
    // desktop setting is untouched (and the menu UI still agrees with the value
    // it was drawn with).
    InjectVrImGuiInput(menuOpen);
    const int navOriginal = CVarGetInteger(kCvarControlNav, 0);
    if (menuOpen) {
        CVarSetInteger(kCvarControlNav, 1);
    }
    gui->StartDraw();
    CVarSetInteger(kCvarControlNav, navOriginal);
    EnsureRenderTargets(g_ctx, interp);
    PrepareOffscreen(interp);
    const double tPrep = NowMs();

    const float half = g_stereoStrength * 0.5f;
    // Snapshot the camera matrices once so both eyes share an identical base.
    std::unordered_map<Mtx*, MtxF> base = mtxReplacements;
    EnsureCameraMatrices(base);
    // Zero-parallax plane: auto-placed on the active camera's look-at target
    // (plus the user trim). This keeps the subject on the screen plane so the
    // scene mostly recedes behind the UI instead of crossing in front of it.
    const float convergence = ComputeZeroParallaxNdc(base, half) + g_convergence;
    // Render right first, left last, so the desktop mirror (which reads the last
    // mGfxFrameBuffer) shows the left eye.
    RenderEyeToTex(interp, g_ctx, 1, commands, base, +half, +convergence);
    const double tEye1 = NowMs();
    RenderEyeToTex(interp, g_ctx, 0, commands, base, -half, -convergence);
    const double tEye0 = NowMs();

    // Our blitter leaves an intermediate texture bound; ImGui's DX11 backend does
    // not set its own render target, so the target must be bound explicitly.
    auto* rapi = interp->GetCurrentRenderingAPI();

    if (!g_ctx->loggedFirstContent) {
        g_ctx->loggedFirstContent = true;
        SPDLOG_INFO("[VR] first content: render {}x{}, swapchain {}x{}; eye textures", g_ctx->renderWidth,
                    g_ctx->renderHeight, g_ctx->eyes[0].width, g_ctx->eyes[0].height);
    }

    // Present the desktop mirror (once per game frame; the headset is fed
    // separately by PresentFrame at the display rate).
    if (menuOpen) {
        const ImGuiIO& io = ImGui::GetIO();
        const uint32_t winW = (uint32_t)std::max(2.0f, io.DisplaySize.x);
        const uint32_t winH = (uint32_t)std::max(2.0f, io.DisplaySize.y);
        if (g_ctx->EnsureMenuTexture(winW, winH)) {
            // Render only the ImGui overlay (not the game image) into the
            // offscreen texture. Clearing the game framebuffer pointer makes
            // Fast3dGui::DrawGame skip its ImGui::Image; the game itself is
            // already in the eye textures, so this keeps stereo intact.
            const float clear[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            g_ctx->DeviceContext()->ClearRenderTargetView(g_ctx->menuRtv.Get(), clear);
            ID3D11RenderTargetView* mrtv = g_ctx->menuRtv.Get();
            g_ctx->DeviceContext()->OMSetRenderTargets(1, &mrtv, nullptr);
            const uintptr_t savedFb = interp->mGfxFrameBuffer;
            interp->mGfxFrameBuffer = 0;
            gui->EndDraw();
            interp->mGfxFrameBuffer = savedFb;

            // Composite the menu over both stereo eye textures for the headset.
            g_ctx->CompositeUiToEyes();

            // Desktop mirror: the menu over black (the game returns when the
            // menu is closed). The same draw data is still valid after EndDraw.
            rapi->StartDrawToFramebuffer(0, 1.0f);
            rapi->ClearFramebuffer(true, true);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
        } else {
            // Fallback: no offscreen texture, so the menu only shows on the desktop.
            rapi->StartDrawToFramebuffer(0, 1.0f);
            gui->EndDraw();
        }
    } else {
        rapi->StartDrawToFramebuffer(0, 1.0f);
        gui->EndDraw();
    }
    const double tGui = NowMs();
    // The desktop window mirrors the left eye. It is only a spectator view, so
    // present it every Nth game frame: default 2 (~30 Hz mirror) so the desktop
    // view is visible; set PAPERBOAT_VR_MIRROR=0 to turn the mirror off entirely.
    //
    // Flush the game+GUI draw commands that target the window backbuffer, then
    // present the swapchain directly (see PresentDesktopMirror). We deliberately
    // avoid Interpreter::EndFrame here: its frame-rate limiter and frame-latency
    // wait would stall this thread, which also drives the game tick and the
    // headset pump, dropping the game framerate while the headset stays smooth.
    rapi->EndFrame();
    rapi->FinishRender();

    static const int s_mirrorPeriod = [] {
        const char* v = std::getenv("PAPERBOAT_VR_MIRROR");
        return v == nullptr ? 2 : std::max(0, std::atoi(v));
    }();
    static int s_mirrorCounter = 0;
    if (s_mirrorPeriod > 0 && (++s_mirrorCounter % s_mirrorPeriod) == 0) {
        PresentDesktopMirror(interp);
    }
    const double tEnd = NowMs();

    g_ctx->haveContent = true;

    g_perf.prepMs += tPrep - t0;
    g_perf.eye1Ms += tEye1 - tPrep;
    g_perf.eye0Ms += tEye0 - tEye1;
    g_perf.guiMs += tGui - tEye0;
    g_perf.deskMs += tEnd - tGui;
    const double rt = tEnd - t0;
    g_perf.renderMs += rt;
    g_perf.renderMax = std::max(g_perf.renderMax, rt);
    ++g_perf.renders;
    return true;
}

// Feeds the headset one frame at the display rate, reusing the most recent game
// render. Called every main-loop iteration so frames are delivered in order and
// on time regardless of how fast the game is rendering.
void PresentFrame() {
    if (!g_enabled || g_ctx == nullptr || !g_ctx->initialized) {
        return;
    }
    g_ctx->PollEvents();
    if (!g_ctx->sessionRunning) {
        return;
    }
    // If the runtime stopped presenting, do nothing until the session state
    // changes (a retry would block for as long as the runtime throttles).
    if (g_pumpPaused) {
        return;
    }
    const bool visible = VrDebugEnabled()
                             ? true
                             : (g_ctx->sessionState == XR_SESSION_STATE_FOCUSED ||
                                g_ctx->sessionState == XR_SESSION_STATE_VISIBLE);
    // NOTE: do NOT gate the frame loop on `visible`. The runtime only advances
    // the session state (READY -> SYNCHRONIZED -> VISIBLE -> FOCUSED) while the
    // app is driving xrWaitFrame/xrEndFrame; if we waited for VISIBLE before
    // pumping, the session would stall at READY forever (and SteamVR would sit on
    // its "launching" overlay). So pump from READY onward regardless of focus.

    const double t0 = NowMs();
    if (!g_ctx->BeginFrame()) {
        ++g_perf.beginFail;
        return;
    }
    const double t1 = NowMs();

    UpdateScreenPose();

    static int s_lastShould = -1;
    if ((int)g_ctx->frameState.shouldRender != s_lastShould) {
        s_lastShould = (int)g_ctx->frameState.shouldRender;
        SPDLOG_INFO("[VR] shouldRender={} (state {} visible={})", (bool)g_ctx->frameState.shouldRender,
                    (int)g_ctx->sessionState, visible);
    }

    // Always (re)submit the most recent game render: it is a cached texture and
    // the copy is cheap, so the headset never shows stale content even when the
    // runtime sets shouldRender=false.
    const bool submit = g_ctx->haveContent;
    if (submit) {
        BlitEyeTexToSwapchain(g_ctx, 0);
        BlitEyeTexToSwapchain(g_ctx, 1);
        // Submit the swapchain writes to the GPU *before* releasing the images /
        // handing them to the compositor. The XR swapchains are cross-process
        // shared textures, so if the drawing is still only queued the compositor's
        // sync times out (SteamVR: "AcquireSync FAILED with WAIT_TIMEOUT") and the
        // headset freezes even though the game keeps running.
        g_ctx->DeviceContext()->Flush();
    }
    const double t2 = NowMs();

    g_ctx->EndFrame(submit);
    const double t3 = NowMs();

    // Watchdog: only back off when the runtime is *sustainedly* not presenting
    // (headset asleep / link dropped) -- never on a one-off game hitch, which
    // would otherwise freeze headset delivery and input mid-play. Require several
    // consecutive pathologically long frames, and use the runtime's own predicted
    // frame period as the signal (it reflects throttling, not our render cost).
    const bool presenting = g_ctx->sessionState == XR_SESSION_STATE_VISIBLE ||
                            g_ctx->sessionState == XR_SESSION_STATE_FOCUSED;
    const bool runtimeThrottled = g_ctx->frameState.predictedDisplayPeriod > 100 * 1000 * 1000; // < 10 Hz
    static int s_stallFrames = 0;
    if (presenting && runtimeThrottled) {
        if (++s_stallFrames >= 3) { // a few consecutive throttled frames, not a hitch
            PausePump("runtime stopped presenting (throttled)");
            s_stallFrames = 0;
        }
    } else {
        s_stallFrames = 0;
    }

    g_perf.waitMs += t1 - t0;
    g_perf.blitMs += t2 - t1;
    g_perf.endMs += t3 - t2;
    g_perf.waitMax = std::max(g_perf.waitMax, t1 - t0);
    g_perf.blitMax = std::max(g_perf.blitMax, t2 - t1);
    g_perf.endMax = std::max(g_perf.endMax, t3 - t2);
    ++g_perf.present;
    if (g_ctx->frameState.shouldRender) {
        ++g_perf.shouldRender;
    }
    LogPerfIfNeeded("scene");
}

} // namespace Paperboat::VR

#else // !PAPERBOAT_ENABLE_OPENXR

namespace Paperboat::VR {
bool Initialize() {
    return false;
}
void Shutdown() {}
bool IsSessionRunning() {
    return false;
}
void PollEvents() {}
bool GetEnabled() {
    return false;
}
void SetEnabled(bool) {}
bool GetVrPad(uint16_t&, int8_t&, int8_t&) {
    return false;
}
uint32_t GetTargetFpsOverride() {
    return 0;
}
void UpdateResolutionCap(Fast::Interpreter*) {}
bool RenderGameFrame(Gfx*, const std::unordered_map<Mtx*, MtxF>&) {
    return false;
}
void PresentFrame() {}
void PollHotkeys() {}
void Recenter() {}
float GetScreenSize() {
    return 0.0f;
}
void SetScreenSize(float) {}
float GetScreenDistance() {
    return 0.0f;
}
void SetScreenDistance(float) {}
float GetStereoStrength() {
    return 0.0f;
}
void SetStereoStrength(float) {}
float GetConvergence() {
    return 0.0f;
}
void SetConvergence(float) {}
void ReloadSettings() {}
} // namespace Paperboat::VR

#endif // PAPERBOAT_ENABLE_OPENXR
