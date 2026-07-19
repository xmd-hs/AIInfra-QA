@echo off
setlocal
cd /d "%~dp0"
title Zhiwen Launcher

where docker.exe >nul 2>&1
if %errorlevel% neq 0 goto no_docker

docker.exe version --format "{{.Server.Version}}" >nul 2>&1
if %errorlevel% equ 0 goto docker_ready

echo Starting Docker Desktop...
if not exist "C:\Program Files\Docker\Docker\Docker Desktop.exe" goto no_docker
start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
echo Waiting for Docker Linux Engine...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ready=$false; for($i=0;$i -lt 60;$i++){docker.exe version --format '{{.Server.Version}}' 2>$null; if($LASTEXITCODE -eq 0){$ready=$true;break}; Start-Sleep -Seconds 2}; if(-not $ready){exit 1}" >nul
if %errorlevel% neq 0 goto docker_timeout

:docker_ready
echo Docker is ready.
echo Building and starting Zhiwen...
echo The first run downloads DeepSeek-R1 and may take several minutes.
docker.exe compose up --build -d
if %errorlevel% neq 0 goto compose_error

echo Waiting for the web application...
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$ready=$false; for($i=0;$i -lt 100;$i++){try{$r=Invoke-WebRequest 'http://127.0.0.1:3000' -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -eq 200){$ready=$true;break}}catch{}; Start-Sleep -Seconds 3}; if(-not $ready){exit 1}" >nul
if %errorlevel% neq 0 goto app_timeout

echo Zhiwen is running at http://127.0.0.1:3000
start "" "http://127.0.0.1:3000"
exit /b 0

:no_docker
echo Docker Desktop was not found.
echo Install Docker Desktop before running this launcher.
pause
exit /b 1

:docker_timeout
echo Docker Linux Engine did not start.
echo Run setup-windows.bat as Administrator, restart Windows, and try again.
pause
exit /b 1

:compose_error
echo Docker Compose failed. Run status.bat and inspect the container logs.
pause
exit /b 1

:app_timeout
echo The application did not become healthy in time.
echo Run status.bat and inspect the container logs.
pause
exit /b 1
