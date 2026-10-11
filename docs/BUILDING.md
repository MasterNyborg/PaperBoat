# Paperboat

### Prerequisites

#### Windows

Install **Visual Studio 2022** (Community edition or higher) with these
workloads/components:

- `Desktop development with C++`
- `C++ Clang tools for Windows` (provides `clang-cl`)
- `CMake tools for Windows`

The Ninja build system is bundled with the CMake tools component.

> **Note:** If you are building from a standalone PowerShell (outside of VS Code
> or Visual Studio), you need to run from a **Developer PowerShell for VS 2022**
> so that `clang-cl`, `ninja`, and the Windows SDK are on your PATH.
> Open one from the Start Menu, or run this in any PowerShell first:
> ```powershell
> & "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64
> ```
> (adjust the path for your VS edition if needed, e.g. `Enterprise` or `Professional`)

#### macOS / Linux

Install `cmake`, `ninja`, and a C/C++ compiler (clang or gcc) via your
package manager.

### Building

Clone with submodules:

```
git clone --recurse-submodules <repo-url>
cd PaperBoat
```

If you already cloned without `--recurse-submodules`:

```
git submodule update --init
```

#### Windows (from a Developer PowerShell)

```
cmake --preset windows-debug
cmake --build --preset windows-debug -j
```

#### Windows quick loop (wrapper scripts)

`port-build.ps1` and `port-run.ps1` in the repo root wrap the Windows build so
you don't have to open a Developer PowerShell by hand. `port-build.ps1` enters
the VS developer shell for you and clears the `VCPKG_ROOT` shim that the dev
shell otherwise points at `VC\vcpkg` (which has no ports and breaks the vcpkg
bootstrap). Subsequent builds are incremental.

```powershell
.\port-build.ps1                 # configure (first run) + build Debug
.\port-build.ps1 -Config Release # Release build
.\port-run.ps1                   # launch build\windows-debug\Paperboat.exe
.\port-run.ps1 -Rom <rom.z64>    # launch straight into ROM import
```

#### macOS / Linux

```
cmake --preset ninja-debug
cmake --build --preset ninja-debug -j
```

### Windows compiler support

Windows builds use `clang-cl` via the CMake presets and the repo-local
toolchain file in [cmake/toolchains/windows-clang-cl.cmake](cmake/toolchains/windows-clang-cl.cmake).

The plain MSVC C compiler (`cl.exe`) is **not** supported for the C99 decomp
sources. The project relies on C99-era constructs that would require large
direct edits to decomp files, so the supported Windows path is
`Ninja + clang-cl`.

# Configuration

### Default keyboard configuration
| N64 | A | B | Z | Start | Analog stick | C buttons | D-Pad |
| - | - | - | - | - | - | - | - |
| Keyboard | X | C | Z | Space | WASD | Arrow keys | TFGH |

### Other shortcuts
| Keys | Action |
| - | - |
| Esc | Toggle menu |
| Ctrl+R | Reset (inside levels) |
| F11 | Fullscreen |
| Tab | Toggle Alternate assets |

# VR (OpenXR)

PaperBoat can render the game as a **world-locked, stereoscopic screen** inside a
headset (a "3D TV"), while the desktop window mirrors the left eye. It is not a
full VR conversion: the game camera, input and gameplay are unchanged — only the
image is shown on a virtual screen you can resize and move.

Requirements:

- Windows with the **DirectX 11** renderer (Settings → Renderer API).
- An **OpenXR runtime** (SteamVR, Oculus, Virtual Desktop/VDXR, …). The active
  runtime is whatever is registered under
  `HKLM\SOFTWARE\Khronos\OpenXR\1\ActiveRuntime`.
- A supported Paper Mario (US) ROM, imported as usual.

The OpenXR loader is fetched and built automatically (Khronos OpenXR-SDK, pinned
in `CMakeLists.txt`); `openxr_loader.dll` is copied next to `Paperboat.exe`.

## Controls

Adjustable at runtime with the keyboard (works while wearing the headset, since
the game window keeps focus) and from **Settings → VR** in the menu.

| Keys | Action |
| - | - |
| Ctrl+Alt+V | Toggle VR rendering on/off |
| Ctrl+Alt+R | Recenter the screen in front of you |
| Ctrl+Alt+Up / Down | Move the screen closer / farther |
| Ctrl+Alt+Left / Right | Shrink / grow the screen |
| Ctrl+Alt+, / . | Decrease / increase 3D strength (0 = flat) |

If no headset/runtime is present, VR silently stays off and the game runs
normally. See `src/port/vr/README.md` for implementation notes.
