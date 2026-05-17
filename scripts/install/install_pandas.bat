@echo off
cd ..
powershell -ExecutionPolicy Bypass -File scripts\install_core.ps1 -Module "pandas"
pause
