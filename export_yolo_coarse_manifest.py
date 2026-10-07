"""Export YOLO bbox detections as weak 2D contact hypotheses.

The output is for inspection, ray-casting experiments and later ROI training
only after class semantics have been verified. It is never merged into the
existing ``p_fire`` label JSON. For the project convention, the original
dataset's class ``1`` (fire) is remapped to model class ``0`` by the one-class
training script; this exporter therefore receives ``--model-fire-class 0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from home_fire_detector_adapter import HomeFireYOLO, bottom_band_points
from project_paths import D_FIRE_ROOT


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path, split: str):
    image_root = root / split / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    return sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=D_FIRE_ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root / "working" / "home_fire_yolo_coarse.jsonl")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument(
        "--model-fire-class",
        "--fire-class",
        dest="model_fire_class",
        type=int,
        required=True,
        help=(
            "Fire class id in the checkpoint predictions. In a checkpoint "
            "trained with --single-fire-class this is 0."
        ),
    )
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--columns", type=int, default=5)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser().resolve()
    images = iter_images(dataset_root, args.split)
    if args.max_images > 0:
        images = images[: args.max_images]
    detector = HomeFireYOLO(
        args.checkpoint,
        fire_class=args.model_fire_class,
        threshold=args.threshold,
        device=args.device,
        imgsz=args.imgsz,
        contact_columns=args.columns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detected = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for index, image_path in enumerate(images, 1):
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                result = detector.detect(image)
            row = {
                "image_path": str(image_path),
                "split": args.split,
                "confidence": float(result.confidence),
                "bbox_xyxy": None if result.bbox is None else [float(value) for value in result.bbox],
                "point": None if result.point is None else [float(value) for value in result.point],
                "bottom_band_points": (
                    [] if result.bbox is None else bottom_band_points(result.bbox, args.columns).round(3).tolist()
                ),
                "detected": bool(result.detected),
                "latency_ms": float(result.latency_ms),
                "source": "home_fire_yolo",
                "class_id": args.model_fire_class,
                "label_quality": "weak_bbox_bottom",
                "weight": 0.10,
            }
            detected += int(result.detected)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            if index % 50 == 0 or index == len(images):
                print(f"processed={index}/{len(images)} detected={detected}")
    print(f"saved={args.output} images={len(images)} detected={detected}")


if __name__ == "__main__":
    main()
