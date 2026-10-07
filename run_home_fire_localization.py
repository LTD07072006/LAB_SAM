"""Run the independent Home Fire YOLO branch through calibrated 3D geometry.

This is deliberately a separate entry point from ``main_localization.py``.
It requires a measured camera calibration and a metric triangle mesh, then
uses the existing robust multi-ray/local uncertainty/tracking implementation.
The optional ROI checkpoint is applied only when explicitly requested.

Example::

    python run_home_fire_localization.py \
      --checkpoint working/home_fire_yolo/bbox640/weights/best.pt \
      --fire-class 0 \
      --samples home-fire-dataset/test/images \
      --calibration measured_camera.json \
      --mesh measured_room_mesh.json \
      --output output/home_fire_yolo_3d/results.json \
      --max-images 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from camera_calibration import CameraCalibration
from home_fire_detector_adapter import HomeFireYOLO, bottom_band_points
from home_fire_3d_pipeline import Fire3DLocalizationPipeline, _json_value
from mesh_loader import load_triangle_mesh
from project_paths import D_FIRE_ROOT


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def select_images(root: Path, max_images: int, selection: str) -> list[Path]:
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images <= 0 or len(images) <= max_images:
        return images
    if selection == "head":
        return images[:max_images]
    if selection == "even":
        indexes = np.linspace(0, len(images) - 1, max_images, dtype=int)
        return [images[int(index)] for index in indexes]

    # ``diverse`` is deliberately deterministic and does not assume that the
    # dataset has labels or that filenames contain a usable frame number.  It
    # anchors the subset at the beginning, middle and end of the sequence,
    # then fills the remaining slots evenly.  This is useful for a folder of
    # CCTV frames where adjacent files are often near-duplicates.
    selected: set[int] = set()
    for index in (0, len(images) // 2, len(images) - 1):
        if len(selected) < max_images:
            selected.add(index)
    indexes = np.linspace(0, len(images) - 1, max_images * 2, dtype=int)
    for index in indexes:
        if len(selected) >= max_images:
            break
        selected.add(int(index))
    if len(selected) < max_images:
        for index in range(len(images)):
            if len(selected) >= max_images:
                break
            selected.add(index)
    return [images[index] for index in sorted(selected)[:max_images]]


def draw_preview(image: Image.Image, result: dict[str, Any], output: Path, columns: int) -> None:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    detection = result.get("detection") or {}
    bbox = detection.get("bbox")
    point = detection.get("point")
    confidence = float(detection.get("confidence", 0.0))
    if bbox is not None:
        box = tuple(float(value) for value in bbox)
        draw.rectangle(box, outline=(231, 76, 60), width=3)
        for x, y in bottom_band_points(box, columns=columns):
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(241, 156, 18))
    if point is not None:
        x, y = (float(point[0]), float(point[1]))
        draw.line((x - 10, y, x + 10, y), fill=(52, 152, 219), width=2)
        draw.line((x, y - 10, x, y + 10), fill=(52, 152, 219), width=2)
    location = result.get("location") or {}
    status = location.get("status", "not_localized")
    draw.rectangle((0, 0, min(canvas.width, 1100), 32), fill=(0, 0, 0))
    draw.text((6, 8), f"p={confidence:.3f} status={status}", fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fire-class",
        type=int,
        required=True,
        help="Fire class id in the model checkpoint; use 0 for the one-class checkpoint",
    )
    parser.add_argument("--samples", type=Path, default=D_FIRE_ROOT / "test" / "images")
    parser.add_argument("--calibration", type=Path, required=True, help="Measured camera calibration JSON")
    parser.add_argument("--mesh", type=Path, required=True, help="Metric room triangle mesh JSON")
    parser.add_argument("--roi-checkpoint", type=Path, default=None, help="Optional ROI refiner; omitted for YOLO->ray baseline")
    parser.add_argument("--output", type=Path, default=root / "output" / "home_fire_yolo_3d" / "results.json")
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--selection", choices=("even", "diverse", "head"), default="diverse")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-uncertainty", action="store_true")
    parser.add_argument("--sequence", action="store_true")
    parser.add_argument(
        "--max-refine-shift-px",
        type=float,
        default=80.0,
        help="Reject ROI output when it is farther than this from YOLO bottom-center; <0 disables the guard",
    )
    parser.add_argument(
        "--refine-blend",
        type=float,
        default=0.75,
        help="Maximum ROI contribution in the coarse/refined weighted blend",
    )
    parser.add_argument(
        "--min-refine-confidence",
        type=float,
        default=0.15,
        help="Minimum ROI heatmap confidence required to use a refined point",
    )
    args = parser.parse_args()

    for path, label in ((args.checkpoint, "YOLO checkpoint"), (args.samples, "sample directory"), (args.calibration, "calibration"), (args.mesh, "mesh")):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    if args.max_refine_shift_px == 0.0:
        raise ValueError("--max-refine-shift-px must be negative to disable the guard, or positive")
    if not 0.0 <= args.refine_blend <= 1.0:
        raise ValueError("--refine-blend must be in [0, 1]")
    if not 0.0 <= args.min_refine_confidence <= 1.0:
        raise ValueError("--min-refine-confidence must be in [0, 1]")

    image_paths = select_images(args.samples, args.max_images, args.selection)
    if not image_paths:
        raise RuntimeError(f"No images found in {args.samples}")
    calibration = CameraCalibration.from_json(args.calibration)
    mesh = load_triangle_mesh(args.mesh)
    detector = HomeFireYOLO(
        args.checkpoint,
        fire_class=args.fire_class,
        threshold=args.threshold,
        device=args.device,
        imgsz=args.imgsz,
        contact_columns=args.columns,
    )
    refiner = None
    if args.roi_checkpoint is not None:
        if not args.roi_checkpoint.is_file():
            raise FileNotFoundError(args.roi_checkpoint)
        from narrow_localizer import ROIRefinerInference
        refiner = ROIRefinerInference(args.roi_checkpoint, device=args.device)
    tracker = None
    temporal = None
    if args.sequence:
        from config import DEFAULT_CONFIG
        from temporal_filter import DetectionSmoother
        from tracking_3d import Fire3DTracker
        tracker = Fire3DTracker(DEFAULT_CONFIG.tracker_alpha, DEFAULT_CONFIG.tracker_gate_m, DEFAULT_CONFIG.tracker_max_missed)
        temporal = DetectionSmoother(
            DEFAULT_CONFIG.temporal_alpha,
            DEFAULT_CONFIG.temporal_window,
            DEFAULT_CONFIG.temporal_min_hits,
            threshold=args.threshold,
        )
    pipeline = Fire3DLocalizationPipeline(
        detector,
        calibration,
        grid_map=mesh,
        refiner=refiner,
        threshold=args.threshold,
        use_uncertainty=not args.no_uncertainty,
        tracker=tracker,
        temporal=temporal,
        ray_kwargs={"columns": args.columns},
        max_refine_shift_px=(
            None if args.max_refine_shift_px < 0.0 else args.max_refine_shift_px
        ),
        refine_blend=args.refine_blend,
        min_refine_confidence=args.min_refine_confidence,
    )
    output_rows = []
    for index, image_path in enumerate(image_paths, 1):
        result = pipeline.process(image_path)
        result["branch"] = "home_fire_yolo_bbox_to_3d"
        result["fire_class"] = args.fire_class
        output_rows.append(result)
        location = result.get("location") or {}
        point = location.get("point")
        print(
            f"{image_path.name:24} p={result['detection']['confidence']:.3f} "
            f"status={location.get('status', 'not_localized'):22} "
            f"3d={point} latency={result['latency_ms']:.1f}ms"
        )
        preview_dir = args.preview_dir or args.output.parent / "previews"
        with Image.open(image_path) as image:
            draw_preview(image.convert("RGB"), result, preview_dir / f"sample_{index:04d}.jpg", args.columns)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "branch": "home_fire_yolo_bbox_to_3d",
        "images": len(output_rows),
        "fire_class": args.fire_class,
        "threshold": args.threshold,
        "calibration": str(args.calibration.resolve()),
        "mesh": str(args.mesh.resolve()),
        "roi_enabled": refiner is not None,
        "refinement_guard": {
            "max_refine_shift_px": args.max_refine_shift_px,
            "refine_blend": args.refine_blend,
            "min_refine_confidence": args.min_refine_confidence,
        },
        "results": _json_value(output_rows),
        "note": "Metric 3D error still requires 3D ground truth in the same world frame.",
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
