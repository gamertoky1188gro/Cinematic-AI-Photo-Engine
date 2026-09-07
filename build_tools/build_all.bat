@echo off
setlocal

echo ============================================
echo   CinematicAI - Multi-Platform Build Script
echo ============================================
echo.

set PROJECT=%~dp0..
set DIST=%PROJECT%\dist

:: Clean previous builds
if exist "%DIST%" rmdir /s /q "%DIST%"
if exist "%PROJECT%\build" rmdir /s /q "%PROJECT%\build"

:: ---- Windows 64-bit (native) ----
echo [1/5] Building Windows x64...
cd /d "%PROJECT%"
call :build_windows_x64
if errorlevel 1 echo WARN: Windows x64 build failed
echo.

:: ---- Linux amd64 ----
echo [2/5] Building Linux amd64 (Docker)...
cd /d "%PROJECT%"
call :build_linux amd64
if errorlevel 1 echo WARN: Linux amd64 build failed
echo.

:: ---- Linux arm64 ----
echo [3/5] Building Linux arm64 (Docker QEMU)...
cd /d "%PROJECT%"
call :build_linux arm64
if errorlevel 1 echo WARN: Linux arm64 build failed
echo.

:: ---- Linux armhf ----
echo [4/5] Building Linux armhf (Docker QEMU)...
cd /d "%PROJECT%"
call :build_linux armhf
if errorlevel 1 echo WARN: Linux armhf build failed
echo.

echo [5/5] Build summary:
echo.
dir /b "%DIST%" 2>nul
echo.
echo ============================================
echo   Done. Check dist\ for output folders.
echo ============================================
pause
exit /b 0

:: ============================================================
:: FUNCTIONS
:: ============================================================

:build_windows_x64
echo   - Activating PyInstaller...
cd /d "%PROJECT%"
pyinstaller binary.spec --noconfirm
if errorlevel 1 exit /b 1
echo   - Windows x64: OK
exit /b 0

:build_linux
set ARCH=%1
echo   - Building Docker image for linux-%ARCH%...

docker buildx create --name cinematicai-%ARCH% --driver docker-container --use 2>nul
docker buildx inspect --bootstrap 2>nul

docker buildx build ^
    --file build_tools/Dockerfile.linux-%ARCH% ^
    --platform linux/%ARCH% ^
    --load ^
    --tag cinematicai:%ARCH% ^
    .

if errorlevel 1 (
    echo   - Docker build failed for linux-%ARCH%
    exit /b 1
)

echo   - Extracting binary from container...
docker create --name cinematicai-%ARCH%-extract cinematicai:%ARCH% 2>nul
if not exist "%DIST%\CinematicAI-linux-%ARCH%" mkdir "%DIST%\CinematicAI-linux-%ARCH%"
docker cp cinematicai-%ARCH%-extract:/app/. "%DIST%\CinematicAI-linux-%ARCH%\"
docker rm cinematicai-%ARCH%-extract 2>nul

echo   - linux-%ARCH%: OK
exit /b 0
