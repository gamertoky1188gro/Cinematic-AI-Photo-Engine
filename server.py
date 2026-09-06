"""Cinematic AI Photo Engine — HTTP Server.

Usage:
    python server.py --port 8000

Client sends image binary + parameters, server processes and returns result.
"""

import argparse
import sys
import uuid
import os
import json
import shutil
import time
import queue
import threading
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Setup path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from engine.pipeline import CinematicPipeline
from engine.evaluate import compute_metrics, measure

# Global pipeline
pipeline: CinematicPipeline = None
gpu_lock = threading.Lock()
temp_dir = Path(project_root / "temp_uploads")
# In-memory originals: {batch_id: {filename: bytes}}
originals_store: dict[str, dict[str, bytes]] = {}
store_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    temp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = project_root / "models" / "checkpoints" / "best_predictor.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = CinematicPipeline(
        predictor_checkpoint=str(checkpoint) if checkpoint.exists() else None,
        device=device,
    )
    print(f"Pipeline ready on {device}", flush=True)

    # Background cleanup: delete batch dirs and originals older than 2 hours
    def _cleanup_old_batches():
        while True:
            time.sleep(300)  # check every 5 min
            try:
                now = time.time()
                for d in list(temp_dir.iterdir()):
                    if d.is_dir() and (now - d.stat().st_mtime) > 7200:
                        batch_name = d.name
                        shutil.rmtree(d, ignore_errors=True)
                        with store_lock:
                            originals_store.pop(batch_name, None)
            except Exception:
                pass
    cleanup_thread = threading.Thread(target=_cleanup_old_batches, daemon=True)
    cleanup_thread.start()

    yield
    print("Shutting down...", flush=True)

app = FastAPI(title="Cinematic AI Photo Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process")
async def process_image(
    file: UploadFile = File(...),
    film_strength: Optional[float] = Form(None),
    highlight_rolloff: Optional[float] = Form(None),
    shadow_tint: Optional[float] = Form(None),
    color_separation: Optional[float] = Form(None),
    halation: Optional[float] = Form(None),
    bloom: Optional[float] = Form(None),
    grain: Optional[float] = Form(None),
    vignette: Optional[float] = Form(None),
    lens_distortion: Optional[float] = Form(None),
    chromatic_aberration: Optional[float] = Form(None),
    edge_softness: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    report: Optional[bool] = Form(False),
):
    """Process an image through the cinematic pipeline.

    Send image as multipart file upload. All parameters are optional.
    If omitted, scene-adaptive defaults are used.

    Returns:
        - Processed image (JPEG) if report=False
        - JSON with metrics + image path if report=True
    """
    job_id = str(uuid.uuid4())[:8]
    job_dir = temp_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Save uploaded image
        ext = Path(file.filename).suffix or ".jpg"
        input_path = job_dir / f"input{ext}"
        output_path = job_dir / f"output.jpg"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Build overrides dict (only non-None values)
        overrides = {}
        param_map = {
            "film_strength": film_strength,
            "highlight_rolloff": highlight_rolloff,
            "shadow_tint": shadow_tint,
            "color_separation": color_separation,
            "halation": halation,
            "bloom": bloom,
            "grain": grain,
            "vignette": vignette,
            "lens_distortion": lens_distortion,
            "chromatic_aberration": chromatic_aberration,
            "edge_softness": edge_softness,
        }
        for key, val in param_map.items():
            if val is not None:
                overrides[key] = val

        # Set seed on pipeline
        if seed is not None:
            pipeline.seed = seed

        # Run pipeline
        with gpu_lock:
            result = pipeline.run(str(input_path), str(output_path), overrides=overrides or None)

        # Read output image bytes
        with open(output_path, "rb") as f:
            image_bytes = f.read()

        # Cleanup temp files
        shutil.rmtree(job_dir, ignore_errors=True)

        if report:
            # Return JSON with metrics
            original = Image.open(input_path).convert("RGB") if input_path.exists() else None
            return JSONResponse(content={
                "status": "ok",
                "scene": result["scene"],
                "plan": result["plan"],
                "overrides": overrides,
            })
        else:
            # Return processed image
            from fastapi.responses import Response
            return Response(content=image_bytes, media_type="image/jpeg")

    except Exception as e:
        # Cleanup on error
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/json")
async def process_image_json(
    file: UploadFile = File(...),
    film_strength: Optional[float] = Form(None),
    highlight_rolloff: Optional[float] = Form(None),
    shadow_tint: Optional[float] = Form(None),
    color_separation: Optional[float] = Form(None),
    halation: Optional[float] = Form(None),
    bloom: Optional[float] = Form(None),
    grain: Optional[float] = Form(None),
    vignette: Optional[float] = Form(None),
    lens_distortion: Optional[float] = Form(None),
    chromatic_aberration: Optional[float] = Form(None),
    edge_softness: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
):
    """Process image and return JSON with scene, plan, and base64 output."""
    import base64

    job_id = str(uuid.uuid4())[:8]
    job_dir = temp_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        ext = Path(file.filename).suffix or ".jpg"
        input_path = job_dir / f"input{ext}"
        output_path = job_dir / f"output.jpg"

        with open(input_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        overrides = {}
        param_map = {
            "film_strength": film_strength,
            "highlight_rolloff": highlight_rolloff,
            "shadow_tint": shadow_tint,
            "color_separation": color_separation,
            "halation": halation,
            "bloom": bloom,
            "grain": grain,
            "vignette": vignette,
            "lens_distortion": lens_distortion,
            "chromatic_aberration": chromatic_aberration,
            "edge_softness": edge_softness,
        }
        for key, val in param_map.items():
            if val is not None:
                overrides[key] = val

        if seed is not None:
            pipeline.seed = seed

        with gpu_lock:
            result = pipeline.run(str(input_path), str(output_path), overrides=overrides or None)

        with open(output_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        shutil.rmtree(job_dir, ignore_errors=True)

        return JSONResponse(content={
            "status": "ok",
            "scene": result["scene"],
            "plan": result["plan"],
            "overrides": overrides,
            "image_base64": img_b64,
        })

    except Exception as e:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok", "pipeline": "ready" if pipeline else "not ready"}


@app.post("/process/stream")
async def process_image_stream(
    file: UploadFile = File(...),
    film_strength: Optional[float] = Form(None),
    highlight_rolloff: Optional[float] = Form(None),
    shadow_tint: Optional[float] = Form(None),
    color_separation: Optional[float] = Form(None),
    halation: Optional[float] = Form(None),
    bloom: Optional[float] = Form(None),
    grain: Optional[float] = Form(None),
    vignette: Optional[float] = Form(None),
    lens_distortion: Optional[float] = Form(None),
    chromatic_aberration: Optional[float] = Form(None),
    edge_softness: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
):
    """Process image with SSE progress stream. Returns NDJSON events."""
    job_id = str(uuid.uuid4())[:8]
    job_dir = temp_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Save file first
    ext = Path(file.filename).suffix or ".jpg"
    input_path = job_dir / f"input{ext}"
    output_path = job_dir / f"output.jpg"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Build overrides
    overrides = {}
    param_map = {
        "film_strength": film_strength, "highlight_rolloff": highlight_rolloff,
        "shadow_tint": shadow_tint, "color_separation": color_separation,
        "halation": halation, "bloom": bloom, "grain": grain,
        "vignette": vignette, "lens_distortion": lens_distortion,
        "chromatic_aberration": chromatic_aberration, "edge_softness": edge_softness,
    }
    for key, val in param_map.items():
        if val is not None:
            overrides[key] = val

    if seed is not None:
        pipeline.seed = seed

    progress_queue = queue.Queue()

    def on_progress(data):
        progress_queue.put(data)

    def run_pipeline():
        try:
            with gpu_lock:
                result = pipeline.run(
                    str(input_path), str(output_path),
                    overrides=overrides or None,
                    on_progress=on_progress,
                )
            with open(output_path, "rb") as f:
                image_bytes = f.read()
            progress_queue.put({"stage": "done", "percent": 100, "result": result["plan"]})
        except Exception as e:
            progress_queue.put({"stage": "error", "percent": -1, "detail": str(e)})

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    def event_generator():
        while True:
            try:
                data = progress_queue.get(timeout=30)
            except queue.Empty:
                yield f'data: {json.dumps({"stage": "timeout", "percent": -1, "detail": "Timed out"})}\n\n'
                break

            yield f'data: {json.dumps(data)}\n\n'

            if data.get("stage") in ("done", "error", "timeout"):
                # Send the image bytes as base64 in the final event
                if data.get("stage") == "done" and output_path.exists():
                    import base64
                    with open(output_path, "rb") as f:
                        img_b64 = base64.b64encode(f.read()).decode()
                    yield f'data: {json.dumps({"stage": "image", "percent": 100, "image_base64": img_b64})}\n\n'
                shutil.rmtree(job_dir, ignore_errors=True)
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/process/stream/batch")
async def process_batch_stream(
    files: List[UploadFile] = File(...),
    film_strength: Optional[float] = Form(None),
    highlight_rolloff: Optional[float] = Form(None),
    shadow_tint: Optional[float] = Form(None),
    color_separation: Optional[float] = Form(None),
    halation: Optional[float] = Form(None),
    bloom: Optional[float] = Form(None),
    grain: Optional[float] = Form(None),
    vignette: Optional[float] = Form(None),
    lens_distortion: Optional[float] = Form(None),
    chromatic_aberration: Optional[float] = Form(None),
    edge_softness: Optional[float] = Form(None),
    seed: Optional[int] = Form(None),
    max_size: Optional[int] = Form(None),
    concurrency: Optional[int] = Form(1),
):
    """Process multiple images concurrently with SSE progress stream per file."""
    import base64
    from concurrent.futures import ThreadPoolExecutor, as_completed

    concurrency = max(1, min(6, concurrency or 1))
    batch_id = str(uuid.uuid4())[:8]
    batch_dir = temp_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    progress_queue = queue.Queue()
    total = len(files)

    overrides = {}
    param_map = {
        "film_strength": film_strength, "highlight_rolloff": highlight_rolloff,
        "shadow_tint": shadow_tint, "color_separation": color_separation,
        "halation": halation, "bloom": bloom, "grain": grain,
        "vignette": vignette, "lens_distortion": lens_distortion,
        "chromatic_aberration": chromatic_aberration, "edge_softness": edge_softness,
    }
    for key, val in param_map.items():
        if val is not None:
            overrides[key] = val

    if seed is not None:
        pipeline.seed = seed

    def save_one(i, f):
        ext = Path(f.filename).suffix or ".jpg"
        input_path = batch_dir / f"f{i}_input{ext}"
        output_path = batch_dir / f"f{i}_output.jpg"
        try:
            f.file.seek(0, 2)
            file_size = f.file.tell()
            f.file.seek(0)
        except Exception:
            file_size = 0

        progress_queue.put({"type": "file_saving", "file_index": i, "file_name": f.filename, "total_files": total, "file_size": file_size})
        bytes_read = 0
        last_emit = 0
        with open(input_path, "wb") as out:
            while True:
                chunk = f.file.read(256 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                bytes_read += len(chunk)
                if file_size > 0 and bytes_read - last_emit >= 1024 * 1024:
                    pct = int(bytes_read * 100 / file_size)
                    progress_queue.put({"type": "file_saving_progress", "file_index": i, "file_name": f.filename, "total_files": total, "bytes_read": bytes_read, "file_size": file_size, "percent": pct})
                    last_emit = bytes_read
        progress_queue.put({"type": "file_saved", "file_index": i, "file_name": f.filename, "total_files": total, "file_size": file_size})
        return {"index": i, "name": f.filename, "input": str(input_path), "output": str(output_path)}

    def run_batch():
        try:
            save_queue = queue.Queue()
            processed_count = [0]
            process_lock = threading.Lock()

            def save_worker():
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(save_one, i, f) for i, f in enumerate(files)]
                    for fut in as_completed(futures):
                        try:
                            entry = fut.result()
                            if entry and not entry.get("error"):
                                save_queue.put(entry)
                        except Exception:
                            pass

            save_thread = threading.Thread(target=save_worker, daemon=True)
            save_thread.start()

            def process_one(entry):
                idx = entry["index"]
                name = entry["name"]
                progress_queue.put({"type": "file_start", "file_index": idx, "file_name": name, "total_files": total})
                def on_progress(data, _idx=idx, _name=name):
                    progress_queue.put({"type": "file_progress", "file_index": _idx, "file_name": _name, "total_files": total, **data})
                try:
                    result = pipeline.run(entry["input"], entry["output"], overrides=overrides or None, on_progress=on_progress, max_size=max_size)
                    with open(entry["output"], "rb") as rf:
                        img_b64 = base64.b64encode(rf.read()).decode()
                    try:
                        os.unlink(entry["input"])
                    except Exception:
                        pass
                    try:
                        os.unlink(entry["output"])
                    except Exception:
                        pass
                    progress_queue.put({"type": "file_done", "file_index": idx, "file_name": name, "total_files": total, "image_base64": img_b64, "plan": result["plan"]})
                except Exception as e:
                    progress_queue.put({"type": "file_error", "file_index": idx, "file_name": name, "total_files": total, "detail": str(e)})
                with process_lock:
                    processed_count[0] += 1

            with ThreadPoolExecutor(max_workers=concurrency) as process_pool:
                futures = []
                while processed_count[0] < total:
                    try:
                        entry = save_queue.get(timeout=300)
                    except queue.Empty:
                        break
                    futures.append(process_pool.submit(process_one, entry))
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception:
                        pass

            progress_queue.put({"type": "batch_done", "total_files": total})
        except Exception as e:
            progress_queue.put({"type": "batch_error", "detail": str(e)})

    thread = threading.Thread(target=run_batch, daemon=True)
    thread.start()

    def event_generator():
        while True:
            try:
                data = progress_queue.get(timeout=300)
            except queue.Empty:
                yield f'data: {json.dumps({"type": "timeout"})}\n\n'
                break
            yield f'data: {json.dumps(data)}\n\n'
            if data.get("type") in ("batch_done", "batch_error", "timeout"):
                shutil.rmtree(batch_dir, ignore_errors=True)
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



class BatchURLRequest(BaseModel):
    urls: List[str]
    film_strength: Optional[float] = None
    highlight_rolloff: Optional[float] = None
    shadow_tint: Optional[float] = None
    color_separation: Optional[float] = None
    halation: Optional[float] = None
    bloom: Optional[float] = None
    grain: Optional[float] = None
    vignette: Optional[float] = None
    lens_distortion: Optional[float] = None
    chromatic_aberration: Optional[float] = None
    edge_softness: Optional[float] = None
    seed: Optional[int] = None
    max_size: Optional[int] = None
    concurrency: Optional[int] = 1


@app.post("/process/stream/batch/urls")
async def process_batch_urls(request: BatchURLRequest):
    """Download images from URLs concurrently, process them, return results, delete temp files."""
    import base64
    import urllib.request
    import ssl
    from concurrent.futures import ThreadPoolExecutor, as_completed

    urls = request.urls
    if not urls:
        raise HTTPException(status_code=400, detail="No URLs provided")

    concurrency = max(1, min(6, getattr(request, "concurrency", 1) or 1))

    overrides = {}
    param_map = {
        "film_strength": request.film_strength, "highlight_rolloff": request.highlight_rolloff,
        "shadow_tint": request.shadow_tint, "color_separation": request.color_separation,
        "halation": request.halation, "bloom": request.bloom, "grain": request.grain,
        "vignette": request.vignette, "lens_distortion": request.lens_distortion,
        "chromatic_aberration": request.chromatic_aberration, "edge_softness": request.edge_softness,
    }
    for key, val in param_map.items():
        if val is not None:
            overrides[key] = val

    if request.seed is not None:
        pipeline.seed = request.seed

    batch_id = str(uuid.uuid4())[:8]
    batch_dir = temp_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    progress_queue = queue.Queue()
    total = len(urls)

    def download_one(i, url):
        try:
            progress_queue.put({"type": "file_downloading", "file_index": i, "file_name": url.split("/")[-1].split("?")[0], "total_files": total, "detail": url})
            ext = Path(url).suffix.split("?")[0] or ".jpg"
            if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".dng", ".cr2", ".cr3", ".nef", ".arw"):
                ext = ".jpg"
            input_path = batch_dir / f"f{i}_input{ext}"
            output_path = batch_dir / f"f{i}_output.jpg"

            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE

            req = urllib.request.Request(url, headers={"User-Agent": "CinematicAI/1.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                total_bytes = int(resp.headers.get("Content-Length", 0) or 0)
                progress_queue.put({"type": "file_downloading", "file_index": i, "file_name": url.split("/")[-1].split("?")[0], "total_files": total, "file_size": total_bytes})
                bytes_read = 0
                last_emit = 0
                with open(input_path, "wb") as out:
                    while True:
                        chunk = resp.read(256 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        bytes_read += len(chunk)
                        if total_bytes > 0 and bytes_read - last_emit >= 1024 * 1024:
                            pct = int(bytes_read * 100 / total_bytes)
                            progress_queue.put({"type": "file_downloading_progress", "file_index": i, "file_name": url.split("/")[-1].split("?")[0], "total_files": total, "bytes_read": bytes_read, "file_size": total_bytes, "percent": pct})
                            last_emit = bytes_read

            from engine.pipeline import RAW_EXTENSIONS
            is_raw = input_path.suffix.lower() in RAW_EXTENSIONS
            if not is_raw:
                Image.open(input_path).verify()

            progress_queue.put({"type": "file_downloaded", "file_index": i, "file_name": Path(url).stem + ext, "total_files": total})
            return {"index": i, "name": Path(url).stem + ext, "url": url, "input": str(input_path), "output": str(output_path)}
        except Exception as e:
            progress_queue.put({"type": "file_error", "file_index": i, "file_name": url.split("/")[-1].split("?")[0], "total_files": total, "detail": str(e)})
            return {"index": i, "name": url, "url": url, "input": None, "output": None, "error": str(e)}

    def run_batch():
        try:
            download_queue = queue.Queue()
            completed = [0]
            total_count = len(urls)
            processed_count = [0]
            process_lock = threading.Lock()

            def download_worker():
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = [pool.submit(download_one, i, url) for i, url in enumerate(urls)]
                    for fut in as_completed(futures):
                        entry = fut.result()
                        if not entry.get("error") and entry.get("input"):
                            download_queue.put(entry)
                        completed[0] += 1

            dl_thread = threading.Thread(target=download_worker, daemon=True)
            dl_thread.start()

            def process_one(entry):
                idx = entry["index"]
                name = entry["name"]
                progress_queue.put({"type": "file_start", "file_index": idx, "file_name": name, "total_files": total})
                def on_progress(data, _idx=idx, _name=name):
                    progress_queue.put({"type": "file_progress", "file_index": _idx, "file_name": _name, "total_files": total, **data})
                try:
                    result = pipeline.run(entry["input"], entry["output"], overrides=overrides or None, on_progress=on_progress, max_size=request.max_size)
                    with open(entry["output"], "rb") as rf:
                        img_b64 = base64.b64encode(rf.read()).decode()
                    try:
                        from engine.pipeline import load_image
                        orig_img = load_image(Path(entry["input"]))
                        import io
                        buf = io.BytesIO()
                        orig_img.save(buf, format="JPEG", quality=90)
                        orig_jpeg = buf.getvalue()
                    except Exception:
                        with open(entry["input"], "rb") as rf:
                            orig_jpeg = rf.read()
                    with store_lock:
                        originals_store.setdefault(batch_id, {})[name] = orig_jpeg
                    try:
                        os.unlink(entry["input"])
                    except Exception:
                        pass
                    try:
                        os.unlink(entry["output"])
                    except Exception:
                        pass
                    progress_queue.put({"type": "file_done", "file_index": idx, "file_name": name, "total_files": total, "image_base64": img_b64, "batch_id": batch_id, "plan": result["plan"]})
                except Exception as e:
                    progress_queue.put({"type": "file_error", "file_index": idx, "file_name": name, "total_files": total, "detail": str(e)})
                with process_lock:
                    processed_count[0] += 1

            with ThreadPoolExecutor(max_workers=concurrency) as process_pool:
                futures = []
                while True:
                    try:
                        entry = download_queue.get(timeout=300)
                    except queue.Empty:
                        break
                    futures.append(process_pool.submit(process_one, entry))
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception:
                        pass

            progress_queue.put({"type": "batch_done", "total_files": total})
        except Exception as e:
            progress_queue.put({"type": "batch_error", "detail": str(e)})

    thread = threading.Thread(target=run_batch, daemon=True)
    thread.start()

    def event_generator():
        while True:
            try:
                data = progress_queue.get(timeout=300)
            except queue.Empty:
                yield f'data: {json.dumps({"type": "timeout"})}\n\n'
                break
            yield f'data: {json.dumps(data)}\n\n'
            if data.get("type") in ("batch_done", "batch_error", "timeout"):
                shutil.rmtree(batch_dir, ignore_errors=True)
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )



# Serve React frontend (production build)
frontend_dist = project_root / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(frontend_dist / "assets")), name="static-assets")

    @app.get("/download/zip")
    async def download_zip():
        import io, zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for batch_dir_path in temp_dir.iterdir():
                if batch_dir_path.is_dir():
                    for f in batch_dir_path.iterdir():
                        if f.name.endswith("_output.jpg"):
                            zf.write(f, f.name)
        buf.seek(0)
        from starlette.responses import StreamingResponse as SR
        return SR(buf, media_type="application/zip", headers={"Content-Disposition": "attachment; filename=cinematic-results.zip"})

    @app.get("/original/{batch_id}/{filename}")
    async def serve_original(batch_id: str, filename: str):
        """Serve original files on-demand for compare viewer (from memory, always JPEG)."""
        import re as _re
        if not _re.match(r'^[a-f0-9]{8}$', batch_id):
            raise HTTPException(status_code=400, detail="Invalid batch ID")
        safe_name = Path(filename).name
        if ".." in safe_name or "/" in safe_name:
            raise HTTPException(status_code=400, detail="Invalid filename")
        with store_lock:
            entry = originals_store.get(batch_id, {}).get(safe_name)
        if entry is None:
            raise HTTPException(status_code=404, detail="File not found or expired")
        return Response(content=entry, media_type="image/jpeg")

    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str):
        # Serve static files if they exist, otherwise return index.html (SPA fallback)
        file_path = frontend_dist / full_path
        if full_path and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(frontend_dist / "index.html"))


def main():
    import logging
    logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Cinematic AI Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--max-size", type=int, default=None, help="Max image dimension in pixels (e.g. 2048). Reduces processing time for large DNGs.")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
