param (
    [string]$ExamplePath = "examples\plot\TestMatplotlib.java"
)

$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName
if ($ExamplePath -notmatch ":") {
    $ExamplePath = Join-Path $ProjectDir $ExamplePath
}

$DistDir = Join-Path $ProjectDir "dist"
$ModuleJars = Get-ChildItem -Path $DistDir -Filter "*.jar" | Where-Object { 
    $_.Name -match "^(numja|pandas|matplotlib|seaborn|sklearn)\.jar$" 
} | Select-Object -ExpandProperty FullName
$LibJars = Get-ChildItem -Path "$DistDir\libs" -Filter "*.jar" | Select-Object -ExpandProperty FullName

if ($ModuleJars.Count -eq 0) {
    Write-Host "[ERROR] Module jars not found in $DistDir. Please run build_core.ps1 first." -ForegroundColor Red
    exit 1
}

$FullCP = ($ModuleJars + $LibJars) -join ";"

Write-Host "============================================================"
Write-Host "  NumJa Example Runner (Separate Jars - Offline)"
Write-Host "============================================================"
Write-Host "Target: $ExamplePath"

# Create a temporary output dir for the example
$OutDir = Join-Path $ProjectDir "build\example_out"
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# Compile the example
Write-Host "[1/2] Compiling example..."
$javacArgs = @("-cp", $FullCP, "-d", $OutDir, $ExamplePath)
$process = Start-Process -FilePath "javac" -ArgumentList $javacArgs -Wait -NoNewWindow -PassThru
if ($process.ExitCode -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit 1
}

# Run the example
Write-Host "[2/2] Running example..."
$className = [System.IO.Path]::GetFileNameWithoutExtension($ExamplePath)
$content = Get-Content $ExamplePath
$packageLine = $content | Where-Object { $_ -match "^package\s+([\w\.]+);" }
if ($packageLine) {
    $packageName = $packageLine -replace "^package\s+([\w\.]+);", '$1'
    $fullClassName = "$packageName.$className"
} else {
    $fullClassName = $className
}

$javaArgs = @("-cp", "$FullCP;$OutDir", $fullClassName)
Start-Process -FilePath "java" -ArgumentList $javaArgs -Wait -NoNewWindow
