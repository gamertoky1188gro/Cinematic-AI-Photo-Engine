# Cinematic AI Photo Engine

Scene-aware cinematic photo transformation engine with a React frontend. Transforms ordinary photos into cinematic images through a 5-stage pipeline combining adaptive 3D LUTs, film response curves, optical effects, and geometry rendering.

## Overview

CinematicAI analyzes each image's scene properties (brightness, contrast, warmth, shadows, highlights) and generates a tailored RenderPlan with 13 normalized parameters. A hybrid intelligence system blends heuristic rules with a trained neural network (MobileNet V3 Small) to produce cinematic results while preserving naturalness.

The engine supports batch processing of JPG, PNG, and DNG/RAW files with concurrent download and processing, real-time SSE progress streaming, before/after comparison, and one-click ZIP export.

## Features

- **5-stage rendering pipeline**: Adaptive 3D LUT, film transform, optical effects, geometry rendering
- **Hybrid intelligence**: Heuristic + trained neural predictor with confidence-weighted blending
- **13 adjustable parameters**: `film_strength`, `highlight_rolloff`, `shadow_tint`, `color_separation`, `halation`, `bloom`, `grain`, `vignette`, `lens_distortion`, `chromatic_aberration`, `edge_softness`, `perspective`, `lut_strength`
- **RAW format support**: DNG, CR2, CR3, NEF, ARW, ORF, RW2, RAF, PEF, SRW
- **Batch processing**: Multi-file upload or URL input with concurrency control (1-6x)
- **Real-time progress**: SSE streaming with per-file status updates
- **Before/after comparison**: Drag divider viewer for processed vs original
- **Client-side ZIP export**: Download all results without server storage
- **Three processing modes**: Automatic, Full Power, Custom (manual sliders)
- **Deterministic mode**: Seed parameter for reproducible grain generation
- **Max image size control**: Resize images via LANCZOS (512-8192px)

## Tech Stack

| Area | Technology |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| Frontend | React 19, Vite 8, Tailwind CSS 4 |
| ML/Inference | PyTorch, MobileNet V3 Small |
| Image Processing | OpenCV, Pillow, NumPy, SciPy |
| RAW Support | rawpy |
| 3D LUT | Image-Adaptive-3DLUT (git submodule) |
| Icons | Lucide React |
| ZIP Export | JSZip |

## Project Structure

```
CinematicAI/
├── server.py                  # FastAPI HTTP server with SSE streaming
├── test.py                    # CLI test/evaluation tool
├── train.py                   # Parameter predictor training script
├── engine/
│   ├── pipeline.py            # Main CinematicPipeline class
│   ├── analyzer.py            # SceneAnalyzer
│   ├── intelligence.py        # Heuristic controller
│   ├── hybrid_intelligence.py # Blends heuristic + learned predictions
│   ├── learned_intelligence.py
│   ├── render_plan.py         # RenderPlan dataclass (13 params)
│   ├── evaluate.py            # PSNR, SSIM metrics
│   ├── quality_evaluator.py   # Naturalness/cinematicity scoring
│   └── stages/
│       ├── adaptive_lut.py    # 3D LUT color transform
│       ├── film_transform.py  # S-curve, rolloff, tint, separation
│       ├── optical_renderer.py # Halation, bloom, grain, vignette
│       └── geometry_renderer.py # Perspective, distortion, CA, softness
├── models/
│   ├── parameter_predictor.py # MobileNet V3 Small multi-head model
│   └── checkpoints/           # Trained .pth files (gitignored)
├── frontend/
│   ├── src/App.jsx            # React SPA
│   ├── vite.config.js
│   └── package.json
├── scripts/
│   ├── build_dataset.py
│   ├── build_references.py
│   ├── download_unsplash.py
│   ├── retrain.py
│   └── test_server.py
├── Image-Adaptive-3DLUT/      # Git submodule (Apache 2.0)
│   ├── pretrained_models/     # LUT + classifier weights
│   └── ...
└── dataset/                   # Training images (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js (for frontend build)
- CUDA-capable GPU (optional, falls back to CPU)

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/gamertoky1188gro/Cinematic-AI-Photo-Engine.git
cd Cinematic-AI-Photo-Engine

# Create Python virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install Python dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install fastapi uvicorn pillow numpy scipy opencv-python rawpy pydantic

# Install frontend dependencies
cd frontend
npm install
cd ..

# Build frontend
cd frontend && npm run build && cd ..
```

### Running the Server

```bash
# Start the API server (serves frontend from frontend/dist/)
python server.py --port 8000

# Optional: limit max image dimensions
python server.py --port 8000 --max-size 2048
```

Open `http://localhost:8000` in your browser.

### Frontend Development

```bash
cd frontend
npm run dev    # Vite dev server on port 3000
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/process` | Process image, return JPEG |
| `POST` | `/process/json` | Process image, return JSON with base64 |
| `POST` | `/process/stream` | Process image with SSE progress |
| `POST` | `/process/stream/batch` | Batch process uploaded files (SSE) |
| `POST` | `/process/stream/batch/urls` | Batch process from URLs (SSE) |
| `GET` | `/download/zip` | Download all results as ZIP |
| `GET` | `/original/{batch_id}/{filename}` | Serve original image |

### Form Parameters

All parameters are optional (0.0-1.0 range):

`film_strength`, `highlight_rolloff`, `shadow_tint`, `color_separation`, `halation`, `bloom`, `grain`, `vignette`, `lens_distortion`, `chromatic_aberration`, `edge_softness`, `perspective`, `lut_strength`, `seed`, `max_size`, `concurrency`

## CLI Usage

```bash
# Process a single image
python test.py --input photo.jpg --output result.jpg

# Full regression report
python test.py --input photo.jpg --report

# A/B/C comparison (heuristic vs learned vs hybrid)
python test.py --input photo.jpg --compare

# Deterministic output
python test.py --input photo.jpg --output result.jpg --seed 42
```

## Training

```bash
# Train the parameter predictor
python train.py --dataset dataset/ --output models/checkpoints --epochs 50 --lr 1e-3

# Retrain with 100 epochs
python scripts/retrain_100.py
```

## Testing

```bash
# Server integration test
python scripts/test_server.py

# Frontend lint
cd frontend && npm run lint
```

## License

- **Image-Adaptive-3DLUT** submodule: [Apache License 2.0](https://github.com/HuiZeng/Image-Adaptive-3DLUT/blob/master/LICENSE)
- Root project: `TODO: Add license`
