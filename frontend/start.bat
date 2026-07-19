@echo off
setlocal
if not exist "%~dp0build\release\AIInfraQA.exe" call "%~dp0build.bat"
if errorlevel 1 pause & exit /b 1
start "AIInfraQA" "%~dp0build\release\AIInfraQA.exe"
endlocal
