"""Download diverse images from Unsplash Lite dataset for CinematicAI training."""

import csv
import json
import random
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Target keywords for diverse cinematic scenes
TARGET_KEYWORDS = {
    "portrait": 80,
    "person": 70,
    "human": 50,
    "face": 40,
    "urban": 50,
    "city": 50,
    "night": 50,
    "sunset": 50,
    "sunrise": 40,
    "building": 40,
    "architecture": 40,
    "silhouette": 30,
    "landscape": 30,
    "mountain": 20,
    "water": 20,
    "forest": 20,
}

UNSUPSLASH_TSV = r"D:\imgtocinamatic\unsplash-lite\photos.tsv000"
KEYWORDS_TSV = r"D:\imgtocinamatic\unsplash-lite\keywords.tsv000"
OUTPUT_DIR = Path(r"D:\imgtocinamatic\CinematicAI\dataset\unsplash_images")
TARGET_COUNT = 600
IMAGE_WIDTH = 1024
MAX_WORKERS = 8


def load_photo_metadata(tsv_path: str) -> dict:
    """Load photo metadata from TSV."""
    photos = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["photo_id"]
            url = row.get("photo_image_url", "")
            if url:
                photos[pid] = {
                    "photo_id": pid,
                    "url": url,
                    "width": int(row.get("photo_width", 0)),
                    "height": int(row.get("photo_height", 0)),
                    "description": row.get("photo_description", ""),
                    "ai_description": row.get("ai_description", ""),
                }
    return photos


def load_photo_keywords(tsv_path: str) -> dict:
    """Load photo-keyword mappings."""
    photo_kws = {}
    with open(tsv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            pid = row["photo_id"]
            kw = row["keyword"]
            if pid not in photo_kws:
                photo_kws[pid] = set()
            photo_kws[pid].add(kw.lower())
    return photo_kws


def select_diverse_photos(photos: dict, photo_kws: dict, target_count: int) -> list:
    """Select diverse photos based on target keywords."""
    selected = {}
    scores = {}

    # Score each photo by keyword coverage
    for pid, kws in photo_kws.items():
        if pid not in photos:
            continue
        score = 0
        matched_kws = []
        for target_kw, weight in TARGET_KEYWORDS.items():
            if target_kw in kws:
                score += weight
                matched_kws.append(target_kw)
        if score > 0:
            scores[pid] = (score, matched_kws)

    # Sort by score (descending) and select top photos
    sorted_pids = sorted(scores.keys(), key=lambda x: -scores[x][0])

    # Ensure diversity by limiting per-keyword
    keyword_counts = {kw: 0 for kw in TARGET_KEYWORDS}
    for pid in sorted_pids:
        if len(selected) >= target_count:
            break
        score, matched_kws = scores[pid]
        # Check if adding this photo would exceed keyword limits
        can_add = True
        for kw in matched_kws:
            if keyword_counts[kw] >= TARGET_KEYWORDS[kw]:
                can_add = False
                break
        if can_add:
            selected[pid] = photos[pid]
            for kw in matched_kws:
                keyword_counts[kw] += 1

    # Fill remaining with random photos if needed
    remaining = [pid for pid in photos if pid not in selected]
    random.shuffle(remaining)
    for pid in remaining:
        if len(selected) >= target_count:
            break
        selected[pid] = photos[pid]

    return list(selected.values())


def download_image(photo: dict, output_dir: Path, width: int) -> tuple:
    """Download a single image."""
    pid = photo["photo_id"]
    url = photo["url"]

    # Add width parameter to Unsplash URL
    if "?" in url:
        download_url = f"{url}&w={width}&q=80"
    else:
        download_url = f"{url}?w={width}&q=80"

    output_path = output_dir / f"{pid}.jpg"

    if output_path.exists():
        return pid, True, "exists"

    try:
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
            with open(output_path, "wb") as f:
                f.write(data)
        return pid, True, "downloaded"
    except Exception as e:
        return pid, False, str(e)


def main():
    print("Loading photo metadata...")
    photos = load_photo_metadata(UNSUPSLASH_TSV)
    print(f"  {len(photos)} photos loaded")

    print("Loading keywords...")
    photo_kws = load_photo_keywords(KEYWORDS_TSV)
    print(f"  {len(photo_kws)} photo-keyword mappings loaded")

    print(f"\nSelecting {TARGET_COUNT} diverse photos...")
    selected = select_diverse_photos(photos, photo_kws, TARGET_COUNT)
    print(f"  {len(selected)} photos selected")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {len(selected)} images at {IMAGE_WIDTH}px width...")
    downloaded = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_image, photo, OUTPUT_DIR, IMAGE_WIDTH): photo
            for photo in selected
        }

        with tqdm(total=len(futures), desc="Downloading") as pbar:
            for future in as_completed(futures):
                pid, success, msg = future.result()
                if success:
                    if msg == "exists":
                        skipped += 1
                    else:
                        downloaded += 1
                else:
                    failed += 1
                pbar.update(1)

    print(f"\nDownload complete:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped (exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {downloaded + skipped}")

    # Save selection metadata
    selection_meta = {
        "target_count": TARGET_COUNT,
        "actual_count": len(selected),
        "downloaded": downloaded,
        "skipped": skipped,
        "failed": failed,
        "photos": [
            {
                "photo_id": p["photo_id"],
                "url": p["url"],
                "description": p["description"],
                "ai_description": p["ai_description"],
            }
            for p in selected
        ],
    }

    meta_path = OUTPUT_DIR / "selection_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(selection_meta, f, indent=2)
    print(f"\nSelection metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
