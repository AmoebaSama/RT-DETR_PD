param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
$distDir = Join-Path $root 'dist'
$deployDir = Join-Path $root 'deploy\RTDETR_Solder_GUI'

if (-not (Test-Path $python)) {
    throw "Workspace virtual environment not found at $python"
}

if ($Clean) {
    if (Test-Path (Join-Path $root 'build')) {
        Remove-Item (Join-Path $root 'build') -Recurse -Force
    }
    if (Test-Path $distDir) {
        Remove-Item $distDir -Recurse -Force
    }
    if (Test-Path (Join-Path $root 'deploy')) {
        Remove-Item (Join-Path $root 'deploy') -Recurse -Force
    }
}

Set-Location $root

$portableDir = Join-Path $root 'deploy\RTDETR_Solder_GUI_Portable'

& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name RTDETR_Solder_GUI `
    --add-data "rtdetr\web\templates;rtdetr\web\templates" `
    --add-data "rtdetr\web\static;rtdetr\web\static" `
    --add-data "rtdetr\runs\solder_defects_rtdetr\weights\best.pt;rtdetr\runs\solder_defects_rtdetr\weights" `
    --add-data "rtdetr\runs\smoke_test\weights\best.pt;rtdetr\runs\smoke_test\weights" `
    launch_rtdetr_app.py

if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $distDir 'RTDETR_Solder_GUI'))) {
    Write-Warning "Standalone EXE build did not complete. If disk space is low, use the portable launcher instead: $portableDir"
    exit 1
}

if (Test-Path $deployDir) {
    Remove-Item $deployDir -Recurse -Force
}

New-Item -ItemType Directory -Path (Split-Path $deployDir) -Force | Out-Null
Copy-Item (Join-Path $distDir 'RTDETR_Solder_GUI') $deployDir -Recurse -Force

$launcher = Join-Path $deployDir 'Start RT-DETR GUI.bat'
@"
@echo off
cd /d "%~dp0"
start "" "RTDETR_Solder_GUI.exe"
"@ | Set-Content -Path $launcher -Encoding ASCII

Write-Output "Deployed app folder created at: $deployDir"