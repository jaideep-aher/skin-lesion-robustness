"""
Perturbations that simulate real-world variation for skin lesion images.
Each function takes a PIL Image (RGB) and returns a PIL Image (RGB).
Implemented with numpy + PIL only to keep the HF Space dependency-light.
"""

from __future__ import annotations

import io
from typing import Callable, Dict

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _rgb_to_lin(arr: np.ndarray) -> np.ndarray:
    return np.power(np.clip(arr / 255.0, 0.0, 1.0), 2.2)


def _lin_to_rgb(arr: np.ndarray) -> np.ndarray:
    return (np.clip(np.power(arr, 1 / 2.2), 0.0, 1.0) * 255.0).astype(np.uint8)


def shift_skin_tone(img: Image.Image, factor: float) -> Image.Image:
    """Scale luminance in linear RGB to approximate a skin tone shift.
    factor < 1 = darker, factor > 1 = lighter.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    lin = _rgb_to_lin(arr)
    out = lin * factor
    return Image.fromarray(_lin_to_rgb(out))


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(factor)


def warm_lighting(img: Image.Image, strength: float = 0.15) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * (1 + strength), 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * (1 - strength), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def cool_lighting(img: Image.Image, strength: float = 0.15) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    arr[..., 0] = np.clip(arr[..., 0] * (1 - strength), 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * (1 + strength), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def add_gaussian_noise(img: Image.Image, sigma: float = 15.0) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    noise = np.random.normal(0.0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))


def gaussian_blur(img: Image.Image, radius: float = 2.0) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def motion_blur(img: Image.Image, kernel: int = 9) -> Image.Image:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    k = max(3, kernel | 1)
    pad = k // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for i in range(k):
        out += padded[:, i : i + arr.shape[1], :]
    out /= k
    return Image.fromarray(out.astype(np.uint8))


def jpeg_compress(img: Image.Image, quality: int = 25) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def rotate(img: Image.Image, degrees: float = 12.0) -> Image.Image:
    return img.rotate(degrees, resample=Image.BILINEAR, fillcolor=(0, 0, 0))


PerturbationFn = Callable[[Image.Image], Image.Image]


def _named(name: str, fn: PerturbationFn) -> PerturbationFn:
    fn.__name__ = name
    return fn


PERTURBATIONS: Dict[str, PerturbationFn] = {
    "darker_skin":    _named("darker_skin",    lambda im: shift_skin_tone(im, 0.55)),
    "lighter_skin":   _named("lighter_skin",   lambda im: shift_skin_tone(im, 1.45)),
    "low_light":      _named("low_light",      lambda im: adjust_brightness(im, 0.55)),
    "harsh_light":    _named("harsh_light",    lambda im: adjust_brightness(adjust_contrast(im, 1.4), 1.25)),
    "warm_white_bal": _named("warm_white_bal", lambda im: warm_lighting(im, 0.20)),
    "cool_white_bal": _named("cool_white_bal", lambda im: cool_lighting(im, 0.20)),
    "motion_blur":    _named("motion_blur",    lambda im: motion_blur(im, 11)),
    "out_of_focus":   _named("out_of_focus",   lambda im: gaussian_blur(im, 2.5)),
    "sensor_noise":   _named("sensor_noise",   lambda im: add_gaussian_noise(im, 18.0)),
    "jpeg_artifacts": _named("jpeg_artifacts", lambda im: jpeg_compress(im, 20)),
    "off_axis":       _named("off_axis",       lambda im: rotate(im, 15.0)),
}


def apply_named(img: Image.Image, name: str) -> Image.Image:
    if name not in PERTURBATIONS:
        raise KeyError(f"Unknown perturbation '{name}'. Available: {sorted(PERTURBATIONS.keys())}")
    return PERTURBATIONS[name](img)
