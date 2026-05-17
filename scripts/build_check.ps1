param(
    [string]$Module = "all"
)

Write-Host "Checking if build is needed..."

$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName
$BuildDir = Join-Path $ProjectDir "build"
$DistDir = Join-Path $ProjectDir "dist"

# Find newest source modification time
$srcFiles = Get-ChildItem -Path (Join-Path $ProjectDir 'modules') -Recurse -Include *.java -ErrorAction SilentlyContinue | Where-Object { $_.FullName -notmatch '\\src\\test\\' }
if ($srcFiles.Count -eq 0) {
    Write-Host "No source files found. Will run full build." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File "$ProjectDir\scripts\build_core.ps1" -Module $Module
    exit $LASTEXITCODE
}

$newestSrc = ($srcFiles | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime

# Find newest jar in dist
if (-not (Test-Path $DistDir)) { 
    Write-Host "Dist folder missing — build required." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File "$ProjectDir\scripts\build_core.ps1" -Module $Module
    exit $LASTEXITCODE
}

$distJars = Get-ChildItem -Path $DistDir -Recurse -Filter *.jar -ErrorAction SilentlyContinue
if ($distJars.Count -eq 0) {
    Write-Host "No jars in dist — build required." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File "$ProjectDir\scripts\build_core.ps1" -Module $Module
    exit $LASTEXITCODE
}

$newestJar = ($distJars | Sort-Object LastWriteTime -Descending | Select-Object -First 1).LastWriteTime

Write-Host "Newest source time: $newestSrc"
Write-Host "Newest dist jar time: $newestJar"

if ($newestJar -lt $newestSrc) {
    Write-Host "Sources are newer than jars — running build..." -ForegroundColor Yellow
    & powershell -ExecutionPolicy Bypass -File "$ProjectDir\scripts\build_core.ps1" -Module $Module
    exit $LASTEXITCODE
} else {
    Write-Host "Build not required — jars are up-to-date." -ForegroundColor Green
}
