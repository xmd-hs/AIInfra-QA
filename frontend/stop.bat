@echo off
setlocal
cd /d "%~dp0"
docker.exe compose down
echo Zhiwen has stopped. Model and knowledge data are preserved.
pause
