"""
Gradio app for the skin lesion robustness demo.
Upload a dermoscopic/phone photo, pick perturbations, and see how
the classifier holds up + how TTA helps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from src.model import get_classifier
from src.perturbations import PERTURBATIONS

ASSETS = Path(__file__).parent / "assets"
SAMPLE_DIR = ASSETS / "sample_images"


def _format_probs(probs: Dict[str, float], top_k: int = 5) -> Dict[str, float]:
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return {k: float(v) for k, v in items}


def _top1(probs: Dict[str, float]) -> Tuple[str, float]:
    label = max(probs, key=probs.get)
    return label, float(probs[label])


def run_demo(
    image: Image.Image,
    selected: List[str],
) -> Tuple[Dict[str, float], List[Tuple[Image.Image, str]], Dict[str, float], str]:
    if image is None:
        return {}, [], {}, "Upload an image to begin."

    clf = get_classifier()
    image = image.convert("RGB")

    clean_probs = clf.predict_proba(image)
    clean_label, clean_conf = _top1(clean_probs)

    if not selected:
        selected = list(PERTURBATIONS.keys())[:6]
    augmenters = [PERTURBATIONS[name] for name in selected]

    tta_probs, per_view = clf.predict_tta(image, augmenters)
    tta_label, tta_conf = _top1(tta_probs)

    gallery: List[Tuple[Image.Image, str]] = []
    flips = 0
    for (view_name, probs), aug_name in zip(per_view[1:], selected):
        top, conf = _top1(probs)
        match = "OK same" if top == clean_label else "FLIP"
        if top != clean_label:
            flips += 1
        perturbed = PERTURBATIONS[aug_name](image)
        gallery.append(
            (
                perturbed,
                f"{aug_name}\n-> {top} ({conf*100:.1f}%)  [{match}]",
            )
        )

    n_views = max(len(selected), 1)
    flip_rate = flips / n_views
    fallback_note = ""
    if clf.using_fallback:
        fallback_note = (
            "\n\n> Note: running on torchvision MobileNetV2 fallback "
            "(HF Hub download failed). Augmentation pipeline still works, "
            "just a different underlying classifier."
        )

    summary = (
        f"### Robustness summary\n"
        f"- Clean prediction: **{clean_label}** ({clean_conf*100:.1f}%)\n"
        f"- TTA prediction (mean over {n_views+1} views): **{tta_label}** "
        f"({tta_conf*100:.1f}%)\n"
        f"- Perturbations that flipped top-1: **{flips} / {n_views}** "
        f"({flip_rate*100:.0f}%)\n"
        f"- Model: `{clf.model_name}`"
        f"{fallback_note}"
    )

    return (
        _format_probs(clean_probs),
        gallery,
        _format_probs(tta_probs),
        summary,
    )


def list_examples() -> List[List]:
    if not SAMPLE_DIR.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths = sorted(p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in exts)
    default_selection = [
        "darker_skin",
        "lighter_skin",
        "low_light",
        "motion_blur",
        "sensor_noise",
        "off_axis",
    ]
    return [[str(p), default_selection] for p in paths]


DESCRIPTION = """
# See What Matters: Skin Lesion Robustness

Skin lesion classifiers assume production images will look like the training data --
mostly fair-skinned, well-lit, dermoscope shots. Real world is different.

This demo stress-tests a pretrained classifier (ImageNet -> HAM10000 transfer learning)
with augmentations tied to actual deployment variation:

* **Skin-tone shifts** (darker/lighter) -- fairness across Fitzpatrick types
* **Lighting** -- low light, warm/cool white balance
* **Phone camera** -- motion blur, defocus, sensor noise, JPEG artifacts
* **Geometry** -- off-axis rotation

Then applies **Test-Time Augmentation (TTA)**: averages the softmax across the
original + all perturbations. Buys real robustness with zero retraining.
"""


def build_ui() -> gr.Blocks:
    pert_names = list(PERTURBATIONS.keys())
    default_selection = [
        "darker_skin",
        "lighter_skin",
        "low_light",
        "motion_blur",
        "sensor_noise",
        "off_axis",
    ]

    with gr.Blocks(title="See What Matters - Skin Lesion Robustness") as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=1):
                image_in = gr.Image(
                    type="pil",
                    label="Upload a skin lesion image",
                    height=320,
                )
                pert_select = gr.CheckboxGroup(
                    choices=pert_names,
                    value=default_selection,
                    label="Perturbations to apply",
                )
                run_btn = gr.Button("Run robustness analysis", variant="primary")

            with gr.Column(scale=1):
                clean_out = gr.Label(
                    label="Clean prediction (top-5)",
                    num_top_classes=5,
                )
                tta_out = gr.Label(
                    label="TTA prediction (top-5)",
                    num_top_classes=5,
                )
                summary_out = gr.Markdown()

        gallery_out = gr.Gallery(
            label="Perturbed views + predictions",
            columns=3,
            height=380,
        )

        run_btn.click(
            run_demo,
            inputs=[image_in, pert_select],
            outputs=[clean_out, gallery_out, tta_out, summary_out],
        )

        examples = list_examples()
        if examples:
            gr.Examples(
                examples=examples,
                inputs=[image_in, pert_select],
                label="Sample images",
            )

        gr.Markdown(
            "---\n"
            "Module 1 Mini Hackathon -- *How Can Machines See What Matters?*"
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0")
