#Requires -Version 5.1
<#
.SYNOPSIS
    Run the built Windows Paperboat port.

.DESCRIPTION
    Launches build\<preset>\Paperboat.exe. If -Rom is given, the path is passed
    as the first argument, which makes Paperboat go straight to the ROM import
    step (GameEngine::RunExtract / GameExtractor::GenerateOTR) instead of the
    normal "pick a ROM" UI.

.PARAMETER Rom
    Optional path to a supported Paper Mario (US) ROM (.z64/.n64/.v64).
    Triggers asset import/extraction at launch.

.PARAMETER Config
    Debug (default) or Release; selects which build output to run.

.EXAMPLE
    .\port-run.ps1
    .\port-run.ps1 -Rom "D:\roms\papermario.z64"
#>
param(
    [string]$Rom,
    [ValidateSet('Debug', 'Release')]
    [string]$Config = 'Debug'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$preset = if ($Config -eq 'Release') { 'windows-release' } else { 'windows-debug' }
$exe = Join-Path $root "build\$preset\Paperboat.exe"

if (-not (Test-Path -LiteralPath $exe)) {
    throw "Paperboat.exe not found at $exe. Run .\port-build.ps1 first."
}

if ($Rom) {
    if (-not (Test-Path -LiteralPath $Rom)) {
        throw "ROM not found at $Rom"
    }
    Write-Host "Launching with ROM import: $Rom" -ForegroundColor Cyan
    & $exe $Rom
} else {
    & $exe
}
exit $LASTEXITCODE
