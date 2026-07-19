@echo off
setlocal
cd /d "%~dp0"
docker.exe compose ps
echo.
echo AI Infra:  http://127.0.0.1:8080/health
echo Python API: http://127.0.0.1:8000/api/health
echo Web:        http://127.0.0.1:3000
pause
