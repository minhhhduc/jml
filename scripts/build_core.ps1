param (
    [string]$Module = "all"
)

Write-Host "============================================================"
Write-Host "  NumJa Build Tool"
Write-Host "============================================================"

$ProjectDir = (Get-Item $PSScriptRoot).Parent.FullName
$BuildDir = Join-Path $ProjectDir "build"
$LibsDir = Join-Path $ProjectDir "libs"

if (-not (Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }
if (-not (Test-Path "$BuildDir\classes")) { New-Item -ItemType Directory -Path "$BuildDir\classes" | Out-Null }

# Delete old classes to ensure clean build
Remove-Item -Recurse -Force "$BuildDir\classes\*" -ErrorAction SilentlyContinue

# Include all module sources and exclude tests
$sources = @(Get-ChildItem -Path $ProjectDir\modules -Recurse -Filter *.java | Where-Object { $_.FullName -notmatch "src\\test" -and $_.FullName -notmatch "target" } | Select-Object -ExpandProperty FullName)

if ($sources.Count -eq 0) {
    Write-Host "[ERROR] No sources found" -ForegroundColor Red
    exit 1
}

$sourcesFile = Join-Path $BuildDir "sources.txt"
$sources -join "`n" | Out-File -FilePath $sourcesFile -Encoding ASCII

# Lấy các thư viện cần thiết, loại bỏ thư viện cũ
$jars = Get-ChildItem -Path $LibsDir -Filter *.jar | Where-Object { $_.Name -notmatch "numja-core" -and $_.Name -notmatch "matplotlib" -and $_.Name -notmatch "seaborn" -and $_.Name -notmatch "plot" } | Select-Object -ExpandProperty FullName
$classpath = $jars -join ";"

# ... (existing setup code) ...

Write-Host "[1/4] Compiling sources..."
$javacArgs = @("--release", "11", "-d", "$BuildDir\classes", "-cp", $classpath, "@$sourcesFile")
$process = Start-Process -FilePath "javac" -ArgumentList $javacArgs -Wait -NoNewWindow -PassThru
if ($process.ExitCode -ne 0) {
    Write-Host "[ERROR] Compilation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[2/4] Preparing separate module JARs..."
$TempDir = Join-Path $BuildDir "temp_jars"
if (-not (Test-Path $TempDir)) { New-Item -ItemType Directory -Path $TempDir | Out-Null }

# Helper to create jar from package
function Create-ModuleJar($name, $packagePath) {
    $jarPath = Join-Path $TempDir "$name-temp.jar"
    Write-Host "    -> Creating $name-temp.jar"
    jar cf $jarPath -C "$BuildDir\classes" $packagePath
    return $jarPath
}

$coreJar = Create-ModuleJar "numja" "numja"
$pandasJar = Create-ModuleJar "pandas" "pandas"
$matplotJar = Create-ModuleJar "matplotlib" "matplotlib"
$seabornJar = Create-ModuleJar "seaborn" "seaborn"
$sklearnJar = Create-ModuleJar "sklearn" "sklearn"

# Re-create Core Jar without the sub-modules
$coreJarPath = Join-Path $TempDir "numja-temp.jar"
jar cf $coreJarPath -C "$BuildDir\classes" "numja"

Write-Host "[3/4] Running ProGuard Obfuscation (Multi-Output)..."
$finalCore = Join-Path $BuildDir "numja.jar"
$finalPandas = Join-Path $BuildDir "pandas.jar"
$finalMatplot = Join-Path $BuildDir "matplotlib.jar"
$finalSeaborn = Join-Path $BuildDir "seaborn.jar"
$finalSklearn = Join-Path $BuildDir "sklearn.jar"

$proguardJar = Join-Path $ProjectDir "libs\proguard-7.3.2\lib\proguard.jar"
$proguardConfig = Join-Path $ProjectDir "proguard.pro"

# Find Java home
$javaHome = $env:JAVA_HOME
if (-not $javaHome) { $javaHome = (Get-Command java).Source.Replace("\bin\java.exe", "") }
$jmods = Join-Path $javaHome "jmods"

$proguardArgs = @(
    "-jar", "`"$proguardJar`"",
    "-injars", "`"$coreJarPath`"", "-outjars", "`"$finalCore`"",
    "-injars", "`"$pandasJar`"", "-outjars", "`"$finalPandas`"",
    "-injars", "`"$matplotJar`"", "-outjars", "`"$finalMatplot`"",
    "-injars", "`"$seabornJar`"", "-outjars", "`"$finalSeaborn`"",
    "-injars", "`"$sklearnJar`"", "-outjars", "`"$finalSklearn`"",
    "-libraryjars", "`"$jmods\java.base.jmod(!**.jar;!module-info.class)`"",
    "-libraryjars", "`"$jmods\java.desktop.jmod(!**.jar;!module-info.class)`"",
    "-libraryjars", "`"$ProjectDir\libs\commons-math3-3.6.1.jar`"",
    "-libraryjars", "`"$ProjectDir\libs\ejml-core-0.43.1.jar`"",
    "-libraryjars", "`"$ProjectDir\libs\ejml-ddense-0.43.1.jar`"",
    "-libraryjars", "`"$ProjectDir\libs\jfreechart-1.5.3.jar`"",
    "-include", "`"$proguardConfig`""
)

$process = Start-Process -FilePath "java" -ArgumentList $proguardArgs -Wait -NoNewWindow -PassThru
if ($process.ExitCode -ne 0) {
    Write-Host "[ERROR] Obfuscation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "[4/4] Finalizing Distribution Folder (dist/)..."
if (-not (Test-Path "$ProjectDir\dist")) { New-Item -ItemType Directory -Path "$ProjectDir\dist" | Out-Null }
if (-not (Test-Path "$ProjectDir\dist\libs")) { New-Item -ItemType Directory -Path "$ProjectDir\dist\libs" | Out-Null }

# Clear old dist files
Remove-Item "$ProjectDir\dist\*.jar" -Force -ErrorAction SilentlyContinue
Remove-Item "$ProjectDir\dist\libs\*.jar" -Force -ErrorAction SilentlyContinue

# Copy protected jars
Copy-Item $finalCore, $finalPandas, $finalMatplot, $finalSeaborn, $finalSklearn "$ProjectDir\dist\" -Force

# Copy datasets
if (Test-Path "$ProjectDir\datasets") {
    Copy-Item "$ProjectDir\datasets" "$ProjectDir\dist\" -Recurse -Force
}

# Copy ONLY third-party jars (exclude our own modules from libs)
Get-ChildItem -Path "$ProjectDir\libs" -Filter "*.jar" | Where-Object { 
    $_.Name -notmatch "numja-core" -and $_.Name -notmatch "matplotlib" -and $_.Name -notmatch "seaborn" -and $_.Name -notmatch "plot" 
} | Copy-Item -Destination "$ProjectDir\dist\libs\" -Force

# Remove any accidental source or test artifacts from dist (no .java/.class allowed)
Get-ChildItem -Path "$ProjectDir\dist" -Recurse -Include *.java,*.class -ErrorAction SilentlyContinue | ForEach-Object {
    Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
}

# Cleanup
Remove-Item $TempDir -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $finalCore, $finalPandas, $finalMatplot, $finalSeaborn, $finalSklearn -Force -ErrorAction SilentlyContinue

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  BUILD SUCCESSFUL! -> dist\" -ForegroundColor Green
Write-Host "  Modules: core, pandas, matplotlib, seaborn, sklearn" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
