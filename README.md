---
title: See What Matters - Skin Lesion Robustness
emoji: 🩺
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.15.1
app_file: app.py
pinned: false
license: mit
---

# See What Matters: Skin Lesion Robustness

Module 1 Mini Hackathon -- "How Can Machines See What Matters?"

By **Jaideep Aher**

## Problem

Skin lesion classifiers (like the ones used in tele-dermatology apps) are mostly trained on bright, well-lit, dermoscope-captured images of lighter skin tones. When you take the same model and run it on a phone photo from a dimly lit room, or on a patient with a darker skin tone, the accuracy tanks. That's a fairness problem *and* a robustness problem.

Instead of collecting and labeling more data (which is expensive and slow), I wanted to see how far you can push robustness using just the model you already have and a smart augmentation pipeline.

## What I built

1. Took a **pretrained skin lesion classifier** from HuggingFace Hub ([Anwarkh1/Skin_Cancer-Image_Classification](https://huggingface.co/Anwarkh1/Skin_Cancer-Image_Classification)) -- this is a ViT that was fine-tuned on HAM10000 data (so it's already a transfer learning artifact: ImageNet -> skin lesions).

2. Built an **augmentation pipeline** (`src/perturbations.py`) with 11 perturbations that each map to a real deployment scenario:

| Augmentation | What it simulates |
|---|---|
| `darker_skin` | Fitzpatrick V/VI patients (underrepresented in training data) |
| `lighter_skin` | Fitzpatrick I/II, very bright lighting |
| `low_light` | Dim exam room, phone photo |
| `harsh_light` | Overhead clinic lamp, high contrast |
| `warm_white_bal` | Incandescent / sunset lighting |
| `cool_white_bal` | Fluorescent / hospital LED |
| `motion_blur` | Shaky hand, patient moves |
| `out_of_focus` | Wrong focal distance on phone |
| `sensor_noise` | Low-light phone sensor noise |
| `jpeg_artifacts` | Chat app compresses the image before model sees it |
| `off_axis` | Camera tilted, not perpendicular to skin |

3. Implemented **Test-Time Augmentation (TTA)** -- at inference time, run the model on the original image plus a bunch of perturbed versions, then average the softmax probabilities. This smooths out the prediction and makes it less sensitive to any single "bad view" of the image.

4. Built a **Gradio web app** that lets you upload a lesion image, pick which perturbations to apply, and see:
   - The model's clean prediction
   - What happens to the prediction under each perturbation
   - How TTA stabilizes the output

5. Ran an **evaluation** comparing baseline (single view) vs TTA robustness across all perturbations. Results are in `assets/results/`.

## Live demo

**Deployed app:** https://huggingface.co/spaces/jaydeep123423/skin-lesion-robustness

## Repo layout

```
.
├── app.py                  # gradio web app
├── requirements.txt
├── src/
│   ├── perturbations.py    # all 11 augmentations
│   ├── model.py            # classifier wrapper + TTA logic
│   └── evaluation.py       # metrics + chart generation
├── scripts/
│   ├── run_evaluation.py   # reproduce the result charts
│   └── fetch_samples.py    # grab demo images
└── assets/
    ├── sample_images/      # demo inputs
    └── results/            # generated charts + json
```

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# grab some demo images
python -m scripts.fetch_samples --n 6

# run the evaluation
python -m scripts.run_evaluation

# start the app
python app.py
```

## Limitations

- The classifier is not retrained here -- this is a prototype showing that TTA + targeted augmentations can improve robustness without retraining.
- The skin-tone shift is a luminance-space approximation, not a true perceptual Fitzpatrick mapping.
- The eval set is small (demo-sized). A real study would need way more images and proper statistical testing.

## License

MIT
