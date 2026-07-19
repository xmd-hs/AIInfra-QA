@echo off
setlocal
if not exist "%~dp0build\release\cpp_backend.exe" call "%~dp0build.bat"
if errorlevel 1 pause & exit /b 1
start "CppBackend" /D "%~dp0" "%~dp0build\release\cpp_backend.exe"
endlocal
