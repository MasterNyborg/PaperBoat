#Requires -Version 5.1
<#
.SYNOPSIS
    Configure and build the Windows Paperboat port (paperboat.exe).

.DESCRIPTION
    Wraps the two fiddly parts of the supported Windows build:
      1. Entering the "Developer PowerShell for VS 2022" so cmake/ninja/clang-cl
         and the Windows SDK are on PATH.
      2. Clearing VCPKG_ROOT, which the VS dev shell points at its bundled
         vcpkg shim (no ports/git), so CMake clones a real vcpkg into the build
         directory instead.

    First run bootstraps vcpkg and builds the static dependencies; later runs
    are incremental.

.PARAMETER Config
    Debug (default) or Release.

.PARAMETER Reconfigure
    Force a fresh CMake configure even if the build dir already exists.

.PARAMETER Clean
    Delete the build directory first (slow: rebuilds everything).

.EXAMPLE
    .\port-build.ps1
    .\port-build.ps1 -Config Release
    .\port-build.ps1 -Reconfigure
#>
param(
    [ValidateSet('Debug', 'Release')]
    [string]$Config = 'Debug',
    [switch]$Reconfigure,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path -LiteralPath $vswhere)) {
    throw "vswhere.exe not found. Install Visual Studio 2022 (Build Tools) first."
}

$vsPath = & $vswhere -latest -products * -property installationPath
if (-not $vsPath) {
    throw "No Visual Studio 2022 installation found."
}

$devShell = Join-Path $vsPath 'Common7\Tools\Launch-VsDevShell.ps1'
Write-Host "Entering VS developer shell: $vsPath" -ForegroundColor Cyan
& $devShell -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null

# See .DESCRIPTION above.
Remove-Item Env:VCPKG_ROOT -ErrorAction SilentlyContinue

$preset = if ($Config -eq 'Release') { 'windows-release' } else { 'windows-debug' }
$buildDir = Join-Path $root "build\$preset"

if ($Clean -and (Test-Path -LiteralPath $buildDir)) {
    Write-Host "Removing $buildDir ..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force -LiteralPath $buildDir
}

if ($Reconfigure -or -not (Test-Path -LiteralPath (Join-Path $buildDir 'CMakeCache.txt'))) {
    Write-Host "Configuring ($preset)..." -ForegroundColor Cyan
    cmake --preset $preset
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

$cores = [Environment]::ProcessorCount
Write-Host "Building ($preset, $cores jobs)..." -ForegroundColor Cyan
cmake --build --preset $preset --parallel $cores
exit $LASTEXITCODE
