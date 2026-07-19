@echo off
setlocal
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo Requesting Administrator permission...
  powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)

echo [1/2] Enabling Windows Subsystem for Linux...
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
if %errorlevel% neq 0 goto error

echo [2/2] Enabling Virtual Machine Platform...
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
if %errorlevel% neq 0 goto error

echo.
echo WSL2 features are enabled.
echo Restart Windows, then double-click start.bat.
pause
exit /b 0

:error
echo Failed to enable Windows features.
echo Make sure this computer supports WSL2.
pause
exit /b 1
