"""Run the existing detector + ROI pipeline on real fire images and videos.

This script is intentionally an inference/evidence tool, not a training
script.  It accepts ordinary image folders and videos that do not contain
``p_fire`` labels, so it reports confidence, detection coverage and latency
plus visual overlays.  It does not invent pixel or 3D ground truth.

If a measured calibration/mesh/XYZ annotation is supplied later, use
``paper_workflow_3d.py`` for the metric 3D benchmark.  The outputs here are
therefore domain-shift diagnostics for real media only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps

from fire_detector import FireDetector
from narrow_localizer import ROIRefinerInference


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
COLOURS = {
    "coarse": (52, 152, 219),
    "roi": (241, 156, 18),
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _select(paths: list[Path], maximum: int, selection: str) -> list[Path]:
    paths = sorted(paths)
    if maximum <= 0 or len(paths) <= maximum:
        return paths
    if selection == "head":
        return paths[:maximum]
    if selection == "diverse":
        indexes = np.linspace(0, len(paths) - 1, maximum, dtype=int)
    else:
        indexes = np.linspace(0, len(paths) - 1, maximum, dtype=int)
    return [paths[int(index)] for index in indexes]


def _point(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(values) < 2 or not np.all(np.isfinite(values[:2])):
        return None
    return [float(values[0]), float(values[1])]


def _marker(draw: ImageDraw.ImageDraw, point: list[float] | None, colour: tuple[int, int, int], label: str) -> None:
    if point is None:
        return
    x, y = point
    radius = 7
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=3)
    draw.line((x - 2 * radius, y, x + 2 * radius, y), fill=colour, width=2)
    draw.line((x, y - 2 * radius, x, y + 2 * radius), fill=colour, width=2)
    draw.text((x + radius + 3, y - radius - 3), label, fill=colour)


def _overlay(image: Image.Image, result: dict[str, Any]) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    coarse = result.get("coarse_pixel")
    refined = result.get("roi_pixel")
    _marker(draw, coarse, COLOURS["coarse"], "coarse")
    _marker(draw, refined, COLOURS["roi"], "ROI")
    title = (
        f"p={result['confidence']:.3f} detected={result['detected']} "
        f"refined={result['roi_used']} latency={result['latency_ms']:.1f}ms"
    )
    draw.rectangle((0, 0, min(canvas.width, 1200), 34), fill=(0, 0, 0))
    draw.text((7, 9), title, fill=(255, 255, 255))
    return canvas


def _write_contact_sheet(paths: list[Path], output: Path, columns: int = 4) -> None:
    if not paths:
        return
    tile_width, tile_height = 420, 320
    rows = math.ceil(len(paths) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    for index, path in enumerate(paths):
        with Image.open(path) as source:
            tile = ImageOps.contain(source.convert("RGB"), (tile_width, tile_height))
        x = (index % columns) * tile_width + (tile_width - tile.width) // 2
        y = (index // columns) * tile_height + (tile_height - tile.height) // 2
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _infer_one(
    detector: FireDetector,
    refiner: ROIRefinerInference,
    image: Image.Image,
    threshold: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    detection = detector.detect(image)
    coarse = _point(detection.pixel)
    refined = None
    roi_confidence = None
    roi_used = False
    if detection.detected and coarse is not None:
        refined_result = refiner.refine(image, coarse)
        refined = _point(refined_result.point)
        roi_confidence = float(refined_result.confidence)
        roi_used = refined is not None
    return {
        "confidence": float(detection.confidence),
        "detected": bool(detection.confidence >= threshold and detection.detected),
        "coarse_pixel": coarse,
        "roi_pixel": refined,
        "roi_used": roi_used,
        "roi_confidence": roi_confidence,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
    }


def _run_images(
    detector: FireDetector,
    refiner: ROIRefinerInference,
    roots: list[Path],
    output_dir: Path,
    maximum_per_root: int,
    selection: str,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    preview_paths: list[Path] = []
    for root in roots:
        paths = _select(
            [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
            maximum_per_root,
            selection,
        )
        class_name = root.name
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                image = source.convert("RGB")
            result = _infer_one(detector, refiner, image, threshold)
            result.update({"media_type": "image", "source": str(path), "group": class_name})
            rows.append(result)
            preview = output_dir / "images" / class_name / f"sample_{index:04d}_{path.stem}.jpg"
            preview.parent.mkdir(parents=True, exist_ok=True)
            _overlay(image, result).save(preview, quality=92)
            preview_paths.append(preview)
    _write_contact_sheet(preview_paths, output_dir / "image_contact_sheet.jpg")
    return rows


def _frame_indices(frame_count: int, maximum: int) -> list[int]:
    if frame_count <= 0:
        return []
    if maximum <= 0 or frame_count <= maximum:
        return list(range(frame_count))
    return [int(index) for index in np.linspace(0, frame_count - 1, maximum, dtype=int)]


def _run_videos(
    detector: FireDetector,
    refiner: ROIRefinerInference,
    videos: list[Path],
    output_dir: Path,
    maximum_frames: int,
    threshold: float,
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video inference requires opencv-python") from exc

    rows: list[dict[str, Any]] = []
    for video_path in videos:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            rows.append({"media_type": "video", "source": str(video_path), "error": "open_failed"})
            continue
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_rows: list[dict[str, Any]] = []
        preview_paths: list[Path] = []
        for frame_index in _frame_indices(count, maximum_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).convert("RGB")
            result = _infer_one(detector, refiner, image, threshold)
            result.update({"frame_index": frame_index, "timestamp_s": frame_index / fps if fps > 0 else None})
            frame_rows.append(result)
            preview = output_dir / "videos" / video_path.stem / f"frame_{frame_index:06d}.jpg"
            preview.parent.mkdir(parents=True, exist_ok=True)
            _overlay(image, result).save(preview, quality=92)
            preview_paths.append(preview)
        capture.release()
        _write_contact_sheet(
            preview_paths,
            output_dir / "videos" / video_path.stem / "contact_sheet.jpg",
            columns=4,
        )
        rows.append(
            {
                "media_type": "video",
                "source": str(video_path),
                "frame_count": count,
                "fps": fps,
                "sampled_frames": frame_rows,
            }
        )
    return rows


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-root",
        action="append",
        type=Path,
        default=[],
        help="Image folder; repeat for Fire/Smoke/Neutral groups",
    )
    parser.add_argument("--video-root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=root / "fire-model-data" / "best.pth")
    parser.add_argument("--roi", type=Path, default=root / "week6_roi_result" / "best_roi.pth")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "real_fire_inference")
    parser.add_argument("--max-images-per-root", type=int, default=20)
    parser.add_argument("--max-frames-per-video", type=int, default=8)
    parser.add_argument("--max-videos", type=int, default=3)
    parser.add_argument("--selection", choices=("head", "diverse"), default="diverse")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for path, label in ((args.model, "detector checkpoint"), (args.roi, "ROI checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    image_roots = [path.expanduser().resolve() for path in args.image_root]
    video_root = args.video_root.expanduser().resolve() if args.video_root else None
    if not image_roots and video_root is None:
        raise ValueError("Provide at least one --image-root or --video-root")
    if any(not path.is_dir() for path in image_roots):
        raise FileNotFoundError(f"Image folder not found: {image_roots}")
    if video_root is not None and not video_root.is_dir():
        raise FileNotFoundError(f"Video folder not found: {video_root}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device={device}")
    print(f"image_roots={image_roots}")
    detector = FireDetector(args.model, device=device, threshold=args.threshold)
    refiner = ROIRefinerInference(args.roi, device=str(device))
    detector.warmup(repeats=1)

    image_rows = _run_images(
        detector,
        refiner,
        image_roots,
        output_dir,
        args.max_images_per_root,
        args.selection,
        args.threshold,
    ) if image_roots else []
    video_rows: list[dict[str, Any]] = []
    if video_root is not None:
        videos = _select(
            [p for p in video_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS],
            args.max_videos,
            args.selection,
        )
        video_rows = _run_videos(
            detector,
            refiner,
            videos,
            output_dir,
            args.max_frames_per_video,
            args.threshold,
        )

    all_image_rows = [row for row in image_rows if "error" not in row]
    image_conf = np.asarray([row["confidence"] for row in all_image_rows], dtype=np.float64)
    image_summary = {
        "images": len(all_image_rows),
        "detected": int(sum(bool(row["detected"]) for row in all_image_rows)),
        "detection_rate": float(np.mean([row["detected"] for row in all_image_rows])) if all_image_rows else None,
        "mean_confidence": float(image_conf.mean()) if len(image_conf) else None,
        "median_confidence": float(np.median(image_conf)) if len(image_conf) else None,
        "mean_latency_ms": float(np.mean([row["latency_ms"] for row in all_image_rows])) if all_image_rows else None,
        "p95_latency_ms": float(np.percentile([row["latency_ms"] for row in all_image_rows], 95)) if all_image_rows else None,
    }
    payload = {
        "protocol": {
            "kind": "real_media_2d_domain_shift",
            "detector": str(args.model.resolve()),
            "roi": str(args.roi.resolve()),
            "threshold": args.threshold,
            "3d_metrics": "not computed; real media has no matching measured calibration/mesh/XYZ labels in this run",
        },
        "image_summary": image_summary,
        "images": image_rows,
        "videos": video_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(_json_value(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"image_summary={image_summary}")
    print(f"videos={len(video_rows)}")
    print(f"saved_summary={output_dir / 'summary.json'}")
    print(f"saved_contact_sheet={output_dir / 'image_contact_sheet.jpg'}")


if __name__ == "__main__":
    main()
