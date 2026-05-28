"""
Wrapper around a pretrained skin-lesion classifier from HuggingFace Hub.
Supports standard prediction and test-time augmentation (TTA).
Falls back to a torchvision MobileNetV2 if the hub model can't be downloaded.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

_PRIMARY_MODEL = "Anwarkh1/Skin_Cancer-Image_Classification"
_FALLBACK_MODELS = [
    "syaha/skin_cancer_detection_model",
]


@dataclass
class Prediction:
    label: str
    score: float


class SkinLesionClassifier:

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or _PRIMARY_MODEL
        self._pipe = None
        self._fallback_used = False
        self._labels: List[str] = []

    def load(self) -> None:
        if self._pipe is not None:
            return

        from transformers import pipeline

        candidates = [self.model_name] + [
            m for m in _FALLBACK_MODELS if m != self.model_name
        ]

        last_err: Optional[Exception] = None
        for name in candidates:
            try:
                log.info("Loading model: %s", name)
                self._pipe = pipeline(
                    task="image-classification",
                    model=name,
                    device=-1,
                )
                self.model_name = name
                cfg = getattr(self._pipe.model, "config", None)
                if cfg is not None and getattr(cfg, "id2label", None):
                    self._labels = [
                        cfg.id2label[i] for i in sorted(cfg.id2label.keys())
                    ]
                return
            except Exception as exc:
                log.warning("Failed to load %s: %s", name, exc)
                last_err = exc

        log.warning("Falling back to torchvision MobileNetV2.")
        self._build_torchvision_fallback()
        if self._pipe is None and last_err is not None:
            log.warning("Original errors: %s", last_err)

    def _build_torchvision_fallback(self) -> None:
        import torch.nn as nn
        from torchvision import models, transforms

        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        net.classifier[1] = nn.Linear(net.last_channel, 2)
        net.eval()

        tfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        labels = ["benign", "malignant"]

        @torch.no_grad()
        def _stub_pipe(img: Image.Image, top_k: int = 5):
            x = tfm(img.convert("RGB")).unsqueeze(0)
            logits = net(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            order = np.argsort(probs)[::-1]
            return [{"label": labels[i], "score": float(probs[i])} for i in order[:top_k]]

        self._pipe = _stub_pipe
        self._fallback_used = True
        self._labels = labels
        self.model_name = "torchvision/mobilenet_v2 (untrained head, demo fallback)"

    @property
    def labels(self) -> List[str]:
        if not self._labels:
            self.load()
        return list(self._labels)

    @property
    def using_fallback(self) -> bool:
        return self._fallback_used

    def _raw_predict(self, img: Image.Image, top_k: int = 5) -> List[Dict]:
        self.load()
        assert self._pipe is not None
        if callable(self._pipe) and not hasattr(self._pipe, "model"):
            return self._pipe(img, top_k=top_k)
        return self._pipe(img, top_k=top_k)

    def predict(self, img: Image.Image, top_k: int = 5) -> List[Prediction]:
        raw = self._raw_predict(img, top_k=top_k)
        return [Prediction(label=str(r["label"]), score=float(r["score"])) for r in raw]

    def predict_proba(self, img: Image.Image) -> Dict[str, float]:
        k = max(len(self.labels), 1)
        preds = self.predict(img, top_k=k)
        return {p.label: p.score for p in preds}

    def predict_tta(
        self,
        img: Image.Image,
        augmenters: List[Callable[[Image.Image], Image.Image]],
    ) -> Tuple[Dict[str, float], List[Tuple[str, Dict[str, float]]]]:
        """Run model on original + each augmented version, average the softmax."""
        views: List[Tuple[str, Image.Image]] = [("original", img)]
        for fn in augmenters:
            name = getattr(fn, "__name__", "aug")
            try:
                views.append((name, fn(img)))
            except Exception as exc:
                log.warning("Skipping augmenter %s: %s", name, exc)

        per_view: List[Tuple[str, Dict[str, float]]] = []
        accumulator: Dict[str, float] = {}
        for name, view in views:
            probs = self.predict_proba(view)
            per_view.append((name, probs))
            for label, p in probs.items():
                accumulator[label] = accumulator.get(label, 0.0) + p

        n = max(len(views), 1)
        averaged = {label: score / n for label, score in accumulator.items()}
        return averaged, per_view


_CLASSIFIER: Optional[SkinLesionClassifier] = None


def get_classifier() -> SkinLesionClassifier:
    global _CLASSIFIER
    if _CLASSIFIER is None:
        _CLASSIFIER = SkinLesionClassifier()
        _CLASSIFIER.load()
    return _CLASSIFIER
