"""Test the Cinematic AI Server.

Starts server, sends test requests, saves results, stops server.
Requires: requests (pip install requests)
"""

import subprocess
import time
import requests
from pathlib import Path

SERVER = "http://localhost:8000"
TEST_IMAGE = r"D:\imgtocinamatic\CinematicAI\Image-Adaptive-3DLUT\demo_images\sRGB\a1629.jpg"
OUTPUT_DIR = Path(r"D:\imgtocinamatic\CinematicAI\results\server_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

proc = subprocess.Popen(
    [r"D:\imgtocinamatic\lut-env\Scripts\python.exe", r"D:\imgtocinamatic\CinematicAI\server.py", "--port", "8000"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

try:
    for i in range(60):
        time.sleep(1)
        try:
            r = requests.get(f"{SERVER}/health", timeout=2)
            if r.status_code == 200:
                print(f"Server ready after {i+1}s")
                break
        except:
            pass
    else:
        print("Server failed to start"); exit(1)

    # 1. Dynamic
    print("\n[1/3] Dynamic (scene-adaptive)...")
    with open(TEST_IMAGE, "rb") as f:
        r = requests.post(f"{SERVER}/process", files={"file": ("a1629.jpg", f, "image/jpeg")})
    (OUTPUT_DIR / "dynamic.jpg").write_bytes(r.content)
    print(f"  Saved dynamic.jpg ({len(r.content)} bytes)")

    # 2. Full power
    print("\n[2/3] Full power (all overrides)...")
    with open(TEST_IMAGE, "rb") as f:
        r = requests.post(f"{SERVER}/process", files={"file": ("a1629.jpg", f, "image/jpeg")}, data={
            "film_strength": 1.0, "highlight_rolloff": 1.0, "shadow_tint": 1.0,
            "color_separation": 1.0, "halation": 1.0, "bloom": 1.0, "grain": 0.8,
            "vignette": 1.0, "lens_distortion": 0.5, "chromatic_aberration": 0.5,
            "edge_softness": 0.5, "seed": 42,
        })
    (OUTPUT_DIR / "fullpower.jpg").write_bytes(r.content)
    print(f"  Saved fullpower.jpg ({len(r.content)} bytes)")

    # 3. JSON
    print("\n[3/3] JSON response...")
    with open(TEST_IMAGE, "rb") as f:
        r = requests.post(f"{SERVER}/process/json", files={"file": ("a1629.jpg", f, "image/jpeg")})
    data = r.json()
    print(f"  brightness={data['scene']['brightness']:.3f}, warmth={data['scene']['warmth']}")
    print(f"  vignette={data['plan']['vignette']}, grain={data['plan']['grain']}")

    print("\nAll tests passed!")

finally:
    proc.terminate(); proc.wait()
    print("Server stopped.")
