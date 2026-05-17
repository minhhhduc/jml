<#
install_from_zip.ps1

Run on client machine after extracting the zip. Copies all jars in the current folder into
%USERPROFILE%\libs\numja\0.1.0 so projects can reference them without Maven.
#>

$ErrorActionPreference = 'Stop'

$version = '0.1.0'
$artifact = 'numja'
$dest = Join-Path $env:USERPROFILE "libs\numja\$version"
if (-not (Test-Path $dest)) { New-Item -ItemType Directory -Path $dest -Force | Out-Null }

Get-ChildItem -File -Filter "*.jar" | ForEach-Object {
    Copy-Item $_.FullName -Destination $dest -Force
}

Write-Host "Installed to: $dest" -ForegroundColor Green
