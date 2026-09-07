"""Cinematic AI Photo Engine - Desktop Binary (pywebview).

Serverless desktop app: Python backend + React frontend via pywebview bridge.
No HTTP server, no temp files on disk - all in-memory.

Usage:
    python binary.py
"""

import sys
import os
import json
import base64
import io
import uuid
import threading
import queue
import time
import urllib.request
import ssl
import shutil
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
import numpy as np
import webview

from engine.pipeline import CinematicPipeline
from engine.evaluate import compute_metrics, measure

pipeline = None
gpu_lock = threading.Lock()
originals_store = {}
store_lock = threading.Lock()
progress_store = {}
progress_lock = threading.Lock()

PARAM_KEYS = (
    "film_strength", "highlight_rolloff", "shadow_tint",
    "color_separation", "halation", "bloom", "grain",
    "vignette", "lens_distortion", "chromatic_aberration",
    "edge_softness",
)


def _emit(job_id, event):
    with progress_lock:
        progress_store.setdefault(job_id, []).append(event)


def _build_overrides(params):
    return {k: v for k, v in params.items() if k in PARAM_KEYS}


class CinematicAPI:
    def __init__(self, window=None):
        self._window = window

    def health(self):
        return {"status": "ok", "pipeline": "ready" if pipeline else "not ready"}

    def process_image(self, file_data, filename, params=None):
        params = params or {}
        job_id = str(uuid.uuid4())[:8]
        overrides = _build_overrides(params)

        try:
            image_bytes = base64.b64decode(file_data)
        except Exception as e:
            return {"error": f"Invalid base64 data: {e}"}

        ext = Path(filename).suffix or ".jpg"

        def _process():
            job_dir = project_root / "temp_uploads" / job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            try:
                input_path = job_dir / f"input{ext}"
                output_path = job_dir / "output.jpg"
                with open(input_path, "wb") as f:
                    f.write(image_bytes)

                seed = params.get("seed")
                if seed is not None:
                    pipeline.seed = int(seed)

                _emit(job_id, {"stage": "analyzing", "percent": 10})

                with gpu_lock:
                    result = pipeline.run(
                        str(input_path), str(output_path),
                        overrides=overrides or None,
                        on_progress=lambda d, _jid=job_id: _emit(_jid, d),
                    )

                with open(output_path, "rb") as f:
                    out_b64 = base64.b64encode(f.read()).decode()

                _emit(job_id, {"stage": "done", "percent": 100, "result": result["plan"]})
                _emit(job_id, {"stage": "image", "image_base64": out_b64,
                               "scene": result["scene"], "plan": result["plan"],
                               "overrides": overrides})
            except Exception as e:
                _emit(job_id, {"stage": "error", "detail": str(e)})
            finally:
                try:
                    shutil.rmtree(job_dir, ignore_errors=True)
                except Exception:
                    pass

        threading.Thread(target=_process, daemon=True).start()
        return {"job_id": job_id}

    def start_batch(self, files_data, params=None):
        params = params or {}
        concurrency = max(1, min(6, params.get("concurrency", 1) or 1))
        max_size = params.get("max_size")
        batch_id = str(uuid.uuid4())[:8]
        total = len(files_data)
        overrides = _build_overrides(params)

        seed = params.get("seed")
        if seed is not None:
            pipeline.seed = int(seed)

        with progress_lock:
            progress_store[batch_id] = []

        def _process_one(idx, item):
            name = item["filename"]
            try:
                _emit(batch_id, {"type": "file_start", "file_index": idx,
                                 "file_name": name, "total_files": total})

                image_bytes = base64.b64decode(item["data"])
                ext = Path(name).suffix or ".jpg"
                job_dir = project_root / "temp_uploads" / batch_id
                job_dir.mkdir(parents=True, exist_ok=True)
                input_path = job_dir / f"f{idx}_input{ext}"
                output_path = job_dir / f"f{idx}_output.jpg"

                with open(input_path, "wb") as f:
                    f.write(image_bytes)

                def on_progress(d, _idx=idx, _name=name):
                    _emit(batch_id, {"type": "file_progress",
                                     "file_index": _idx, "file_name": _name,
                                     "total_files": total, **d})

                with gpu_lock:
                    result = pipeline.run(
                        str(input_path), str(output_path),
                        overrides=overrides or None,
                        on_progress=on_progress,
                        max_size=max_size,
                    )

                with open(output_path, "rb") as rf:
                    img_b64 = base64.b64encode(rf.read()).decode()

                with store_lock:
                    originals_store.setdefault(batch_id, {})[name] = image_bytes

                try:
                    os.unlink(input_path)
                except Exception:
                    pass
                try:
                    os.unlink(output_path)
                except Exception:
                    pass

                _emit(batch_id, {"type": "file_done", "file_index": idx,
                                 "file_name": name, "total_files": total,
                                 "image_base64": img_b64, "plan": result["plan"]})
            except Exception as e:
                _emit(batch_id, {"type": "file_error", "file_index": idx,
                                 "file_name": name, "total_files": total,
                                 "detail": str(e)})

        def _run():
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_process_one, i, item)
                           for i, item in enumerate(files_data)]
                for f in as_completed(futures):
                    try:
                        f.result()
                    except Exception:
                        pass
            _emit(batch_id, {"type": "batch_done", "total_files": total})

        threading.Thread(target=_run, daemon=True).start()
        return {"batch_id": batch_id, "total": total}

    def start_batch_urls(self, urls, params=None):
        params = params or {}
        concurrency = max(1, min(6, params.get("concurrency", 1) or 1))
        max_size = params.get("max_size")
        batch_id = str(uuid.uuid4())[:8]
        total = len(urls)
        overrides = _build_overrides(params)

        seed = params.get("seed")
        if seed is not None:
            pipeline.seed = int(seed)

        with progress_lock:
            progress_store[batch_id] = []

        def _download_one(idx, url):
            try:
                _emit(batch_id, {"type": "file_downloading", "file_index": idx,
                                 "file_name": url.split("/")[-1].split("?")[0],
                                 "total_files": total, "detail": url})

                ext = Path(url).suffix.split("?")[0] or ".jpg"
                if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp",
                                        ".tiff", ".tif", ".dng", ".cr2", ".cr3",
                                        ".nef", ".arw"):
                    ext = ".jpg"

                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE

                req = urllib.request.Request(url, headers={"User-Agent": "CinematicAI/1.0"})
                with urllib.request.urlopen(req, context=ctx, timeout=60) as resp:
                    data = resp.read()

                name = Path(url).stem + ext
                _emit(batch_id, {"type": "file_downloaded", "file_index": idx,
                                 "file_name": name, "total_files": total})
                return {"index": idx, "name": name, "data": data}
            except Exception as e:
                _emit(batch_id, {"type": "file_error", "file_index": idx,
                                 "file_name": url.split("/")[-1].split("?")[0],
                                 "total_files": total, "detail": str(e)})
                return None

        def _process_one(idx, name, image_data):
            try:
                _emit(batch_id, {"type": "file_start", "file_index": idx,
                                 "file_name": name, "total_files": total})

                ext = Path(name).suffix or ".jpg"
                job_dir = project_root / "temp_uploads" / batch_id
                job_dir.mkdir(parents=True, exist_ok=True)
                input_path = job_dir / f"f{idx}_input{ext}"
                output_path = job_dir / f"f{idx}_output.jpg"

                with open(input_path, "wb") as f:
                    f.write(image_data)

                def on_progress(d, _idx=idx, _name=name):
                    _emit(batch_id, {"type": "file_progress",
                                     "file_index": _idx, "file_name": _name,
                                     "total_files": total, **d})

                with gpu_lock:
                    result = pipeline.run(
                        str(input_path), str(output_path),
                        overrides=overrides or None,
                        on_progress=on_progress,
                        max_size=max_size,
                    )

                with open(output_path, "rb") as rf:
                    img_b64 = base64.b64encode(rf.read()).decode()

                with store_lock:
                    originals_store.setdefault(batch_id, {})[name] = image_data

                try:
                    os.unlink(input_path)
                except Exception:
                    pass
                try:
                    os.unlink(output_path)
                except Exception:
                    pass

                _emit(batch_id, {"type": "file_done", "file_index": idx,
                                 "file_name": name, "total_files": total,
                                 "image_base64": img_b64, "batch_id": batch_id,
                                 "plan": result["plan"]})
            except Exception as e:
                _emit(batch_id, {"type": "file_error", "file_index": idx,
                                 "file_name": name, "total_files": total,
                                 "detail": str(e)})

        def _run():
            try:
                download_queue = queue.Queue()

                def _download_worker():
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futures = [pool.submit(_download_one, i, url)
                                   for i, url in enumerate(urls)]
                        for fut in as_completed(futures):
                            result = fut.result()
                            if result:
                                download_queue.put(result)

                dl_thread = threading.Thread(target=_download_worker, daemon=True)
                dl_thread.start()

                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=concurrency) as pool:
                    futures = []
                    while True:
                        try:
                            entry = download_queue.get(timeout=300)
                        except queue.Empty:
                            break
                        futures.append(pool.submit(_process_one,
                                                   entry["index"], entry["name"],
                                                   entry["data"]))
                    for f in as_completed(futures):
                        try:
                            f.result()
                        except Exception:
                            pass

                _emit(batch_id, {"type": "batch_done", "total_files": total})
            except Exception as e:
                _emit(batch_id, {"type": "batch_error", "detail": str(e)})

        threading.Thread(target=_run, daemon=True).start()
        return {"batch_id": batch_id, "total": total}

    def get_progress(self, job_id):
        with progress_lock:
            events = progress_store.pop(job_id, [])
        return events

    def get_original(self, batch_id, filename):
        with store_lock:
            data = originals_store.get(batch_id, {}).get(filename)
        if data is None:
            return {"error": "Not found"}
        return {"data": base64.b64encode(data).decode(), "content_type": "image/jpeg"}


def main():
    global pipeline

    import logging
    logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.WARNING)

    checkpoint = project_root / "models" / "checkpoints" / "best_predictor.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = CinematicPipeline(
        predictor_checkpoint=str(checkpoint) if checkpoint.exists() else None,
        device=device,
    )
    print(f"Pipeline ready on {device}", flush=True)

    frontend_dist = project_root / "frontend (binary)" / "dist"
    if not frontend_dist.exists():
        print(f"ERROR: Frontend build not found at {frontend_dist}", file=sys.stderr)
        print("Run: cd 'frontend (binary)' && npm install && npm run build", file=sys.stderr)
        sys.exit(1)

    index_html = str(frontend_dist / "index.html")
    api = CinematicAPI()

    window = webview.create_window(
        title="Cinematic AI Photo Engine",
        url=index_html,
        js_api=api,
        width=1400,
        height=900,
        min_size=(900, 600),
        background_color="#07040d",
        text_select=True,
    )

    print("Launching desktop window...", flush=True)
    webview.start(debug=False)
    print("Shutting down.", flush=True)


if __name__ == "__main__":
    main()
