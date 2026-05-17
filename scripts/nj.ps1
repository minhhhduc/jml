param (
    [Parameter(Mandatory=$true)]
    [string]$File
)

$PSScriptRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName

# Handle relative paths
if ($File -notmatch ":") {
    $File = Join-Path $ProjectDir $File
}

if (-not (Test-Path $File)) {
    Write-Host "[ERROR] Target file not found: $File" -ForegroundColor Red
    exit 1
}

$DistDir = Join-Path $ProjectDir "dist"
$ModuleJars = Get-ChildItem -Path $DistDir -Filter "*.jar" | Select-Object -ExpandProperty FullName
$LibJars = Get-ChildItem -Path "$DistDir\libs" -Filter "*.jar" | Select-Object -ExpandProperty FullName
$FullCP = ($ModuleJars + $LibJars) -join ";"

# Create temporary output directory
$OutDir = Join-Path $ProjectDir "build\nj_out"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

Write-Host "[1/2] Compiling $File..." -ForegroundColor Cyan
$javacArgs = @("-cp", $FullCP, "-d", $OutDir, $File)
$process = Start-Process -FilePath "javac" -ArgumentList $javacArgs -Wait -NoNewWindow -PassThru
if ($process.ExitCode -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[2/2] Running..." -ForegroundColor Green
$className = [System.IO.Path]::GetFileNameWithoutExtension($File)
$content = Get-Content $File
$packageLine = $content | Where-Object { $_ -match "^package\s+([\w\.]+);" }
if ($packageLine) {
    $packageName = $packageLine -replace "^package\s+([\w\.]+);", '$1'
    $fullClassName = "$packageName.$className"
} else {
    $fullClassName = $className
}

$javaArgs = @("-cp", "$FullCP;$OutDir", $fullClassName)
Start-Process -FilePath "java" -ArgumentList $javaArgs -Wait -NoNewWindow
