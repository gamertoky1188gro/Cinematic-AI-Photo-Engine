# CinematicAI - Build Tools

## Prerequisites

### Windows 64-bit (native)
- Python 3.11 with all dependencies installed
- PyInstaller (`pip install pyinstaller`)

### Linux / ARM builds (Docker required)
- Docker Desktop for Windows (with WSL 2 backend)
- Docker Buildx enabled

Install Docker: https://docs.docker.com/desktop/install/windows-install/

Enable buildx:
```powershell
docker buildx create --name multiarch --driver docker-container --use
docker buildx inspect --bootstrap
```

## Build Commands

### Windows 64-bit only
```cmd
build_tools\build_windows.bat
```

### All platforms (requires Docker)
```cmd
build_tools\build_all.bat
```

### Manual Docker builds
```bash
# Linux amd64
docker buildx build --file build_tools/Dockerfile.linux-amd64 --platform linux/amd64 --load --tag cinematicai:amd64 .

# Linux arm64
docker buildx build --file build_tools/Dockerfile.linux-arm64 --platform linux/arm64 --load --tag cinematicai:arm64 .

# Linux armhf (32-bit ARM)
docker buildx build --file build_tools/Dockerfile.linux-armhf --platform linux/arm/v7 --load --tag cinematicai:armhf .
```

## Output

```
dist\
├── CinematicAI\              Windows x64
│   ├── CinematicAI.exe
│   └── _internal\
├── CinematicAI-linux-amd64\  Linux x86_64
│   ├── CinematicAI
│   └── _internal\
├── CinematicAI-linux-arm64\  Linux aarch64
│   ├── CinematicAI
│   └── _internal\
└── CinematicAI-linux-armhf\  Linux armv7l
    ├── CinematicAI
    └── _internal\
```

## Notes
- Linux builds use CPU-only PyTorch (no CUDA in containers)
- Windows build uses CUDA 12.6 PyTorch
- Total size per platform: ~3-4 GB
- Each folder is self-contained — zip and distribute
