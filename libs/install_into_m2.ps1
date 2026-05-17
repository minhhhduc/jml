<#
install_into_m2.ps1

Install all jars in the current folder into the local Maven repository so other
Maven projects can depend on `com.numja:numja:0.1.0` without network access.

Usage: open PowerShell in the folder containing the jars (extracted from the zip),
then run:
  powershell -ExecutionPolicy Bypass -File .\install_into_m2.ps1

Requires: `mvn` on PATH.
#>

$ErrorActionPreference = 'Stop'

function Install-Jar($file, $groupId, $artifactId, $version) {
    Write-Host "Installing $artifactId:$version from $file"
    & mvn org.apache.maven.plugins:maven-install-plugin:3.1.0:install-file -Dfile="$file" -DgroupId="$groupId" -DartifactId="$artifactId" -Dversion="$version" -Dpackaging=jar
    if ($LASTEXITCODE -ne 0) { throw "mvn install-file failed for $file" }
}

$version = '0.1.0'

# Known mappings for the distribution
$mappings = @{
    'numja' = @{group='com.numja'; artifact='numja'; version=$version}
    'ejml-core' = @{group='org.ejml'; artifact='ejml-core'; version='0.43.1'}
    'ejml-ddense' = @{group='org.ejml'; artifact='ejml-ddense'; version='0.43.1'}
    'commons-math3' = @{group='org.apache.commons'; artifact='commons-math3'; version='3.6.1'}
    'jfreechart' = @{group='org.jfree'; artifact='jfreechart'; version='1.5.3'}
}

Get-ChildItem -Filter "*.jar" | ForEach-Object {
    $name = $_.BaseName
    # Normalize name (strip classifier if present)
    $parts = $name -split '-' 
    if ($parts.Length -ge 2) {
        $key = $parts[0]
        if ($mappings.ContainsKey($key)) {
            $m = $mappings[$key]
            Install-Jar $_.FullName $m.group $m.artifact $m.version
            return
        }
    }
    # Fallback: install under group com.numja with artifact = filename (no ext)
    $fallbackArtifact = $name
    Install-Jar $_.FullName 'com.numja' $fallbackArtifact $version
}

Write-Host "All jars installed into local Maven repository." -ForegroundColor Green
