"""
Run the robustness evaluation end-to-end and produce charts.

Usage:
    python -m scripts.run_evaluation --images assets/sample_images --out assets/results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import (
    dump_json,
    evaluate,
    evaluate_tta,
    load_demo_images,
    plot_comparison,
    plot_confidence_drop,
)
from src.model import get_classifier

DEFAULT_TTA_VIEWS = [
    "darker_skin",
    "lighter_skin",
    "low_light",
    "warm_white_bal",
    "cool_white_bal",
    "out_of_focus",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", type=Path, default=Path("assets/sample_images"))
    ap.add_argument("--out", type=Path, default=Path("assets/results"))
    ap.add_argument("--tta-views", nargs="*", default=DEFAULT_TTA_VIEWS)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    images = load_demo_images(args.images)
    if not images:
        print(f"No images found under {args.images}", file=sys.stderr)
        return 1
    print(f"Loaded {len(images)} demo images from {args.images}")

    clf = get_classifier()
    print(f"Using model: {clf.model_name}")

    print("Running baseline evaluation...")
    baseline = evaluate(clf, images)
    print("Running TTA evaluation...")
    tta = evaluate_tta(clf, images, args.tta_views)

    plot_comparison(baseline, tta, args.out / "robustness_top1.png")
    plot_confidence_drop(baseline, tta, args.out / "confidence_drop.png")
    dump_json(baseline, tta, args.out / "results.json")

    print(f"Wrote charts + JSON to {args.out}")
    print(f"\n{'perturbation':<18}  baseline   tta")
    for name in baseline:
        b = baseline[name].top1_agreement
        t = tta[name].top1_agreement
        delta = t - b
        print(f"{name:<18}  {b:>7.2f}  {t:>7.2f}   ({delta:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
