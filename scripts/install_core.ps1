param (
    [string]$Module = "all"
)

Write-Host "============================================================"
Write-Host "  NumJa Install Tool - Module: $Module"
Write-Host "============================================================"

$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName + "\"
$InstallDir = Join-Path $env:USERPROFILE ".numja"
$JarFile = Join-Path $ProjectDir "build\numja-$Module.jar"

if (-not (Test-Path $JarFile)) {
    Write-Host "[ERROR] Could not find build\numja-$Module.jar." -ForegroundColor Red
    Write-Host "Please run scripts\build\build_$Module.bat first!" -ForegroundColor Yellow
    exit 1
}

Write-Host "[1/2] Copying $Module library to $InstallDir..."
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
}
Copy-Item -Path $JarFile -Destination (Join-Path $InstallDir "numja-$Module.jar") -Force

Write-Host "[2/2] Updating Global CLASSPATH..."
$currentClasspath = [Environment]::GetEnvironmentVariable("CLASSPATH", "User")
$newJarPath = "$InstallDir\numja-$Module.jar"

if ($currentClasspath -notmatch [regex]::Escape($newJarPath)) {
    if ([string]::IsNullOrWhiteSpace($currentClasspath)) {
        $newClasspath = ".;$newJarPath"
    } else {
        $newClasspath = "$currentClasspath;$newJarPath"
    }
    [Environment]::SetEnvironmentVariable("CLASSPATH", $newClasspath, "User")
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "INSTALLATION SUCCESSFUL for $Module!" -ForegroundColor Green
Write-Host "Please RESTART your terminal to use the new module natively." -ForegroundColor Yellow
Write-Host "============================================================"
