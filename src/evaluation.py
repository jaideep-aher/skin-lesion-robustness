"""
Robustness evaluation -- measures how much the classifier drifts
when we perturb images in ways that shouldn't change the diagnosis.
Compares baseline (single view) vs TTA.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .model import SkinLesionClassifier
from .perturbations import PERTURBATIONS


@dataclass
class PerturbationResult:
    name: str
    top1_agreement: float
    mean_conf_drop: float
    n_images: int


def load_demo_images(folder: Path) -> List[Image.Image]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
    return [Image.open(p).convert("RGB") for p in paths]


def evaluate(
    clf: SkinLesionClassifier,
    images: List[Image.Image],
) -> Dict[str, PerturbationResult]:
    if not images:
        raise ValueError("No demo images provided.")

    clean_top1: List[str] = []
    clean_top1_conf: List[float] = []
    for img in images:
        probs = clf.predict_proba(img)
        label = max(probs, key=probs.get)
        clean_top1.append(label)
        clean_top1_conf.append(probs[label])

    results: Dict[str, PerturbationResult] = {}
    for name, fn in PERTURBATIONS.items():
        agree = 0
        drops: List[float] = []
        for img, clean_lbl, clean_conf in zip(images, clean_top1, clean_top1_conf):
            try:
                perturbed = fn(img)
            except Exception:
                continue
            probs = clf.predict_proba(perturbed)
            top = max(probs, key=probs.get)
            if top == clean_lbl:
                agree += 1
            drops.append(clean_conf - probs.get(clean_lbl, 0.0))
        n = len(images)
        results[name] = PerturbationResult(
            name=name,
            top1_agreement=agree / n if n else 0.0,
            mean_conf_drop=float(np.mean(drops)) if drops else 0.0,
            n_images=n,
        )
    return results


def evaluate_tta(
    clf: SkinLesionClassifier,
    images: List[Image.Image],
    tta_views: List[str],
) -> Dict[str, PerturbationResult]:
    augmenters = [PERTURBATIONS[name] for name in tta_views]

    def tta_probs(img: Image.Image) -> Dict[str, float]:
        avg, _ = clf.predict_tta(img, augmenters)
        return avg

    clean_top1: List[str] = []
    clean_top1_conf: List[float] = []
    for img in images:
        probs = tta_probs(img)
        label = max(probs, key=probs.get)
        clean_top1.append(label)
        clean_top1_conf.append(probs[label])

    results: Dict[str, PerturbationResult] = {}
    for name, fn in PERTURBATIONS.items():
        agree = 0
        drops: List[float] = []
        for img, clean_lbl, clean_conf in zip(images, clean_top1, clean_top1_conf):
            try:
                perturbed = fn(img)
            except Exception:
                continue
            probs = tta_probs(perturbed)
            top = max(probs, key=probs.get)
            if top == clean_lbl:
                agree += 1
            drops.append(clean_conf - probs.get(clean_lbl, 0.0))
        n = len(images)
        results[name] = PerturbationResult(
            name=name,
            top1_agreement=agree / n if n else 0.0,
            mean_conf_drop=float(np.mean(drops)) if drops else 0.0,
            n_images=n,
        )
    return results


def plot_comparison(
    baseline: Dict[str, PerturbationResult],
    tta: Dict[str, PerturbationResult],
    out_path: Path,
    title: str = "Robustness: baseline vs test-time augmentation",
) -> None:
    names = list(baseline.keys())
    base_vals = [baseline[n].top1_agreement for n in names]
    tta_vals = [tta[n].top1_agreement for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, base_vals, width, label="Baseline (single view)")
    ax.bar(x + width / 2, tta_vals, width, label="With TTA")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 agreement with clean prediction")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confidence_drop(
    baseline: Dict[str, PerturbationResult],
    tta: Dict[str, PerturbationResult],
    out_path: Path,
) -> None:
    names = list(baseline.keys())
    base = [baseline[n].mean_conf_drop for n in names]
    aug = [tta[n].mean_conf_drop for n in names]
    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, base, width, label="Baseline")
    ax.bar(x + width / 2, aug, width, label="With TTA")
    ax.set_ylabel("Mean confidence drop on clean top-1 label")
    ax.set_title("Lower is more robust")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def dump_json(
    baseline: Dict[str, PerturbationResult],
    tta: Dict[str, PerturbationResult],
    out_path: Path,
) -> None:
    payload = {
        "baseline": {k: asdict(v) for k, v in baseline.items()},
        "tta": {k: asdict(v) for k, v in tta.items()},
    }
    out_path.write_text(json.dumps(payload, indent=2))
