@echo off
setlocal
echo ============================================
echo   CinematicAI - Windows 64-bit Build
echo ============================================
echo.
cd /d "%~dp0.."

echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build

echo Building Windows x64...
pyinstaller binary.spec --noconfirm

if errorlevel 1 (
    echo BUILD FAILED
    pause
    exit /b 1
)

echo.
echo BUILD SUCCESSFUL
echo Output: dist\CinematicAI\CinematicAI.exe
echo.
pause
