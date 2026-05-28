"""
Fetch sample skin-lesion images for the demo.
Tries HuggingFace dataset first, falls back to procedural generation.

Usage:
    python -m scripts.fetch_samples --n 6
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = ROOT / "assets" / "sample_images"


def _try_huggingface(n: int) -> int:
    try:
        from datasets import load_dataset
    except Exception:
        return 0

    candidates = [
        ("marmal88/skin_cancer", "train"),
    ]
    for name, split in candidates:
        try:
            ds = load_dataset(name, split=f"{split}[:{n}]")
            saved = 0
            for i, row in enumerate(ds):
                img = row.get("image") or row.get("img")
                if img is None:
                    continue
                label = row.get("dx") or row.get("label") or f"sample{i}"
                fname = SAMPLE_DIR / f"{i:02d}_{label}.jpg"
                img.convert("RGB").save(fname, quality=92)
                saved += 1
            if saved:
                print(f"Saved {saved} images from {name}.")
                return saved
        except Exception as exc:
            print(f"  -> {name} failed: {exc}", file=sys.stderr)
    return 0


def _procedural_patch(seed: int, size: int = 320) -> Image.Image:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    base_palette = [
        (236, 196, 166),
        (224, 178, 140),
        (198, 144, 105),
        (151, 102, 73),
        (102, 68, 52),
    ]
    base = base_palette[seed % len(base_palette)]
    arr = np.zeros((size, size, 3), dtype=np.float32)
    for c in range(3):
        arr[..., c] = base[c]
    grad = np.linspace(-15, 15, size, dtype=np.float32)
    arr += grad[:, None, None]
    arr += np_rng.normal(0, 4, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    points = []
    for theta in np.linspace(0, 2 * math.pi, 18, endpoint=False):
        r = rng.uniform(40, 75)
        points.append((cx + r * math.cos(theta), cy + r * math.sin(theta)))
    lesion_color = (
        max(base[0] - 90, 20) + rng.randint(-10, 10),
        max(base[1] - 70, 15) + rng.randint(-10, 10),
        max(base[2] - 60, 10) + rng.randint(-10, 10),
    )
    draw.polygon(points, fill=lesion_color)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.2))
    return img


def _procedural_set(n: int) -> int:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = _procedural_patch(seed=i)
        img.save(SAMPLE_DIR / f"synthetic_{i:02d}.png")
    print(f"Saved {n} procedural placeholder images.")
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--force-procedural", action="store_true")
    args = ap.parse_args()

    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    if not args.force_procedural:
        saved = _try_huggingface(args.n)

    if saved == 0:
        print("Falling back to procedural placeholder images.")
        saved = _procedural_set(args.n)

    print(f"Sample images in {SAMPLE_DIR.relative_to(ROOT)}/")
    return 0 if saved else 1


if __name__ == "__main__":
    raise SystemExit(main())
