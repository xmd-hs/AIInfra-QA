@echo off
setlocal
set "QT=C:\Qt\5.14.2\5.14.2\msvc2017_64"
set "VS=C:\Program Files\Microsoft Visual Studio\2022\Community"
call "%VS%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1
if not exist build mkdir build
cd build
"%QT%\bin\qmake.exe" ..\AIInfraQA.pro -spec win32-msvc
if errorlevel 1 exit /b 1
nmake release
if errorlevel 1 exit /b 1
"%QT%\bin\windeployqt.exe" --release release\AIInfraQA.exe
endlocal
