"""Compare coarse, ROI-refined, and v3 2D fire points visually.

The comparison is deliberately a 2D evaluation:

    labels -> upstream detector (coarse) -> ROIRefiner
                                  \-> v3 detector

The script draws the labelled point and every available prediction on each
image, writes a contact sheet, and prints pixel MAE/PCK statistics.  It does
not claim a metre-level 3D result; ray casting is a separate stage and needs
measured calibration, a metric mesh, and 3D ground truth.

The v3 branch is loaded through ``v3_detector.py`` when that adapter exists.
As a compatibility fallback, a checkpoint that is understood by the local
``FireDetector`` can be used, but the script prints a warning because that is
not evidence that the checkpoint is an FPN model.
"""
from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
COLOURS = {
    "gt": (46, 204, 113),
    "coarse": (52, 152, 219),
    "roi": (243, 156, 18),
    "v3": (231, 76, 60),
}


def _configure_console() -> None:
    """Keep Windows console output from failing on Vietnamese messages."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="replace")


def default_v3_checkpoint() -> Path:
    """Return the first v3 checkpoint layout present in this checkout."""
    candidates = (
        ROOT / "v3_outputs" / "best_v3_fpn.pth",
        ROOT / "v3_outputs" / "best" / "best_v3_fpn.pth",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


@dataclass
class Prediction:
    """A model point in original-image pixel coordinates."""

    confidence: float = 0.0
    point: Optional[tuple[float, float]] = None
    detected: bool = False
    latency_ms: float = 0.0


@dataclass
class ComparisonRow:
    record: Any
    image: Image.Image
    gt: tuple[float, float]
    coarse: Prediction
    roi: Prediction
    v3: Prediction


def _get(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _point(value: Any, image_size: tuple[int, int]) -> Optional[tuple[float, float]]:
    """Read either pixel or normalised coordinates from a detector result."""
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(array) < 2 or not np.all(np.isfinite(array[:2])):
        return None
    x, y = float(array[0]), float(array[1])
    width, height = image_size
    # Local FireDetector returns pixels. Most custom v3 adapters return either
    # pixels or [0, 1] coordinates; this convention handles both.
    if abs(x) <= 1.5 and abs(y) <= 1.5 and (width > 4 or height > 4):
        x, y = x * width, y * height
    return float(np.clip(x, 0.0, max(0, width - 1))), float(np.clip(y, 0.0, max(0, height - 1)))


def _normalise_prediction(raw: Any, image_size: tuple[int, int], threshold: float) -> Prediction:
    if raw is None:
        return Prediction()
    confidence = _get(raw, "confidence", "p_fire", "score", "probability", default=0.0)
    try:
        confidence = float(np.clip(float(confidence), 0.0, 1.0))
    except (TypeError, ValueError):
        confidence = 0.0
    value = _get(raw, "point", "pixel", "center", "coord", "coordinates", default=None)
    point = _point(value, image_size)
    detected_value = _get(raw, "detected", "is_fire", default=None)
    detected = confidence >= threshold if detected_value is None else bool(detected_value)
    detected = bool(detected and confidence >= threshold and point is not None)
    return Prediction(confidence=confidence, point=point if detected else None, detected=detected)


def _records_for_split(records: Sequence[Any], split: str, seed: int = 42) -> list[Any]:
    positives = [record for record in records if int(record.has_fire) == 1]
    if split == "all":
        return sorted(positives, key=lambda item: str(item.image_path))
    source = [record for record in positives if record.source_split == split]
    if source:
        return sorted(source, key=lambda item: str(item.image_path))
    # Labels from some exports do not encode a source split. Reuse the same
    # leakage-safe splitter as training in that case.
    from train_week6 import split_records

    generated = split_records(records, seed=seed)
    return sorted(
        [record for record in generated.get(split, []) if int(record.has_fire) == 1],
        key=lambda item: str(item.image_path),
    )


def _even_selection(records: Sequence[Any], maximum: int) -> list[Any]:
    records = list(records)
    if maximum <= 0 or maximum >= len(records):
        return records
    indices = np.linspace(0, len(records) - 1, maximum, dtype=int)
    return [records[int(index)] for index in np.unique(indices)]


def _metric(errors: Sequence[float], total: int, threshold_values=(10.0, 25.0)) -> dict[str, Any]:
    values = np.asarray(list(errors), dtype=np.float64)
    result: dict[str, Any] = {
        "images": int(total),
        "detected": int(len(values)),
        "coverage": float(len(values) / total) if total else 0.0,
        "mae_px": None,
        "median_px": None,
        "p95_px": None,
        "pck10": None,
        "pck25": None,
    }
    if len(values):
        result.update(
            mae_px=float(values.mean()),
            median_px=float(np.median(values)),
            p95_px=float(np.percentile(values, 95)),
            pck10=float(np.mean(values <= threshold_values[0])),
            pck25=float(np.mean(values <= threshold_values[1])),
        )
    return result


def _prediction_error(prediction: Prediction, gt: tuple[float, float]) -> Optional[float]:
    if prediction.point is None:
        return None
    return float(np.linalg.norm(np.asarray(prediction.point) - np.asarray(gt)))


def _instantiate_v3_class(cls: type, checkpoint: Path, device: Optional[str], threshold: float) -> Any:
    """Instantiate common local v3 adapter signatures without hard-coding one."""
    attempts = [
        ((checkpoint,), {"device": device, "threshold": threshold}),
        ((checkpoint,), {"device": device}),
        ((), {"model_path": checkpoint, "device": device, "threshold": threshold}),
        ((), {"checkpoint_path": checkpoint, "device": device, "threshold": threshold}),
        ((checkpoint,), {}),
    ]
    errors = []
    for args, kwargs in attempts:
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            return cls(*args, **kwargs)
        except (TypeError, ValueError, FileNotFoundError, RuntimeError) as exc:
            errors.append(str(exc))
    signature = "unknown"
    try:
        signature = str(inspect.signature(cls))
    except (TypeError, ValueError):
        pass
    raise RuntimeError(f"Không khởi tạo được {cls.__name__}{signature}: {' | '.join(errors[-3:])}")


class V3Adapter:
    """Load a restored v3 detector adapter or fail with an actionable error."""

    def __init__(self, checkpoint: Path, device: Optional[str], threshold: float):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint v3: {self.checkpoint}. "
                "Hãy khôi phục v3_outputs/best/best_v3_fpn.pth trước khi so sánh."
            )
        self.threshold = float(threshold)
        self.backend = None
        self.adapter_name = ""
        errors: list[str] = []

        # Keep this defined even when v3_detector.py is absent.  The fallback
        # branch below must not turn a missing adapter into an UnboundLocalError.
        module = None
        adapter_path = ROOT / "v3_detector.py"
        if adapter_path.is_file():
            spec = importlib.util.spec_from_file_location("_local_v3_detector", adapter_path)
            module = None if spec is None or spec.loader is None else importlib.util.module_from_spec(spec)
            if module is not None:
                try:
                    sys.modules["_local_v3_detector"] = module
                    spec.loader.exec_module(module)
                    for name in ("V3Detector", "V3FireDetector", "FireDetectorV3", "FPNFireDetector"):
                        cls = getattr(module, name, None)
                        if cls is not None:
                            self.backend = _instantiate_v3_class(cls, self.checkpoint, device, threshold)
                            self.adapter_name = f"v3_detector.{name}"
                            break
                except Exception as exc:  # adapter code is external to this runner
                    errors.append(f"{adapter_path.name}: {exc}")

        if self.backend is None:
            # A few local experiments exposed a factory instead of a class.
            if adapter_path.is_file() and module is not None:
                factory = getattr(module, "load_detector", None)
                if callable(factory):
                    try:
                        self.backend = factory(self.checkpoint, device=device, threshold=threshold)
                        self.adapter_name = "v3_detector.load_detector"
                    except Exception as exc:
                        errors.append(f"load_detector: {exc}")

        if self.backend is None and not adapter_path.is_file():
            # Compatibility fallback only when the v3 adapter is absent. If an
            # adapter exists but cannot load the checkpoint, failing is safer
            # than labelling a baseline result as a v3 result.
            try:
                from fire_detector import FireDetector

                self.backend = FireDetector(self.checkpoint, device=device, threshold=threshold)
                self.adapter_name = "fire_detector.FireDetector (fallback)"
                print("WARNING: v3_detector.py chưa có; đang dùng FireDetector fallback cho checkpoint v3.")
            except Exception as exc:
                errors.append(f"FireDetector fallback: {exc}")

        if self.backend is None:
            detail = "\n  ".join(errors[-5:])
            if adapter_path.is_file():
                hint = "Checkpoint/kiến trúc v3 không khớp v3_detector.py."
            else:
                hint = "Cần checkpoint v3 thật và v3_detector.py."
            raise RuntimeError(f"Không thể tải nhánh v3. {hint} Chi tiết:\n  {detail}")

    def detect(self, image: Image.Image) -> Prediction:
        start = time.perf_counter()
        if hasattr(self.backend, "detect"):
            try:
                raw = self.backend.detect(image, warmup=False)
            except TypeError:
                # Keep compatibility with small user-written adapters whose
                # detect() method does not expose the optional warmup flag.
                raw = self.backend.detect(image)
        elif hasattr(self.backend, "predict"):
            raw = self.backend.predict(image)
        elif callable(self.backend):
            raw = self.backend(image)
        else:
            raise TypeError("v3 adapter cần detect(), predict(), hoặc callable")
        result = _normalise_prediction(raw, image.size, self.threshold)
        result.latency_ms = (time.perf_counter() - start) * 1000.0
        return result


def _run_detector(detector: Any, image: Image.Image, threshold: float) -> Prediction:
    start = time.perf_counter()
    raw = detector.detect(image, warmup=False)
    result = _normalise_prediction(raw, image.size, threshold)
    result.latency_ms = (time.perf_counter() - start) * 1000.0
    return result


def _draw_marker(draw: ImageDraw.ImageDraw, point: Optional[tuple[float, float]], colour: tuple[int, int, int], label: str, radius: int) -> None:
    if point is None:
        return
    x, y = point
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=max(2, radius // 3))
    draw.line((x - radius * 2, y, x + radius * 2, y), fill=colour, width=max(1, radius // 4))
    draw.line((x, y - radius * 2, x, y + radius * 2), fill=colour, width=max(1, radius // 4))
    draw.text((x + radius + 2, y + radius + 2), label, fill=colour)


def _annotate(row: ComparisonRow) -> Image.Image:
    canvas = row.image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    radius = max(4, int(round(max(canvas.size) / 90.0)))
    _draw_marker(draw, row.gt, COLOURS["gt"], "GT", radius)
    _draw_marker(draw, row.coarse.point, COLOURS["coarse"], "coarse", radius)
    _draw_marker(draw, row.roi.point, COLOURS["roi"], "ROI", radius)
    _draw_marker(draw, row.v3.point, COLOURS["v3"], "v3", radius)
    header = (
        f"{Path(row.record.image_path).name}  "
        f"coarse={row.coarse.confidence:.2f}  v3={row.v3.confidence:.2f}"
    )
    draw.rectangle((0, 0, canvas.width, max(24, radius * 5)), fill=(0, 0, 0))
    draw.text((6, 6), header, fill=(255, 255, 255))
    return canvas


def _contact_sheet(images: Sequence[tuple[str, Image.Image]], output: Path, columns: int = 2) -> None:
    if not images:
        raise RuntimeError("Không có ảnh để tạo contact sheet")
    thumb_w, thumb_h = 520, 390
    caption_h = 28
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * thumb_w, rows * (thumb_h + caption_h)), (35, 35, 35))
    for index, (name, image) in enumerate(images):
        thumb = image.copy()
        thumb.thumbnail((thumb_w - 12, thumb_h - 12), Image.Resampling.LANCZOS)
        x = (index % columns) * thumb_w + (thumb_w - thumb.width) // 2
        y = (index // columns) * (thumb_h + caption_h) + (thumb_h - thumb.height) // 2
        sheet.paste(thumb, (x, y))
        ImageDraw.Draw(sheet).text(
            ((index % columns) * thumb_w + 6, (index // columns) * (thumb_h + caption_h) + thumb_h + 5),
            name,
            fill=(255, 255, 255),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def _print_summary(rows: Sequence[ComparisonRow]) -> None:
    total = len(rows)
    print("\n2D pixel comparison (same images and labels)")
    print("branch    detected/total   MAE(px)   median(px)   P95(px)   PCK@10   PCK@25")
    for name in ("coarse", "roi", "v3"):
        errors = []
        for row in rows:
            prediction = getattr(row, name)
            error = _prediction_error(prediction, row.gt)
            if error is not None:
                errors.append(error)
        metrics = _metric(errors, total)
        def fmt(value: Any) -> str:
            return "-" if value is None else f"{float(value):.2f}"
        print(
            f"{name:<8} {metrics['detected']:>3}/{total:<10} "
            f"{fmt(metrics['mae_px']):>8} {fmt(metrics['median_px']):>11} "
            f"{fmt(metrics['p95_px']):>9} {fmt(metrics['pck10']):>8} {fmt(metrics['pck25']):>8}"
        )
    print("Note: đây là sai số pixel 2D; chưa phải sai số mét 3D vật lý.")


def _select_diverse(rows: Sequence[ComparisonRow], maximum: int) -> list[ComparisonRow]:
    rows = list(rows)
    if maximum <= 0 or maximum >= len(rows):
        return rows

    def score(row: ComparisonRow) -> float:
        coarse_error = _prediction_error(row.coarse, row.gt) or 0.0
        roi_error = _prediction_error(row.roi, row.gt) or coarse_error
        v3_error = _prediction_error(row.v3, row.gt) or 0.0
        disagreement = 0.0
        if row.coarse.point is not None and row.roi.point is not None:
            disagreement += float(np.linalg.norm(np.asarray(row.coarse.point) - row.roi.point))
        if row.coarse.point is not None and row.v3.point is not None:
            disagreement += float(np.linalg.norm(np.asarray(row.coarse.point) - row.v3.point))
        low_confidence = (1.0 - row.coarse.confidence) + (1.0 - row.v3.confidence)
        return coarse_error + roi_error + v3_error + disagreement + 10.0 * low_confidence

    ranked = sorted(rows, key=score, reverse=True)
    selected: list[ComparisonRow] = []
    groups: set[str] = set()
    for row in ranked:
        group = str(getattr(row.record, "group", "unknown"))
        if group not in groups:
            selected.append(row)
            groups.add(group)
            if len(selected) >= maximum:
                return sorted(selected, key=lambda item: str(item.record.image_path))
    for row in ranked:
        if row not in selected:
            selected.append(row)
        if len(selected) >= maximum:
            break
    return sorted(selected, key=lambda item: str(item.record.image_path))


def evaluate(args: argparse.Namespace) -> list[ComparisonRow]:
    from fire_detector import FireDetector
    from narrow_localizer import ROIRefinerInference
    from train_week6 import load_records

    records, stats = load_records(args.labels, args.dataset)
    print(f"loaded_records={stats.get('loaded', 0)} unresolved={stats.get('unresolved', 0)}")
    records = _records_for_split(records, args.split, args.seed)
    if not records:
        raise RuntimeError(f"Không có fire record nào trong split={args.split}")

    detector = FireDetector(args.baseline, device=args.device, threshold=args.threshold)
    refiner = ROIRefinerInference(args.roi, device=args.device)
    v3 = V3Adapter(args.v3, args.device, args.threshold)
    if hasattr(detector, "warmup"):
        detector.warmup(repeats=1)

    rows: list[ComparisonRow] = []
    run_records = _even_selection(records, args.max_images) if args.selection == "even" else records
    for record in run_records:
        image = Image.open(record.image_path).convert("RGB")
        width, height = image.size
        gt = (float(record.x_norm * width), float(record.y_norm * height))
        coarse = _run_detector(detector, image, args.threshold)

        roi = Prediction()
        if coarse.point is not None:
            start = time.perf_counter()
            refined = refiner.refine(image, coarse.point)
            roi = Prediction(
                confidence=float(getattr(refined, "confidence", 0.0)),
                point=_point(getattr(refined, "point", None), image.size),
                detected=True,
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )
        v3_prediction = v3.detect(image)
        rows.append(ComparisonRow(record, image, gt, coarse, roi, v3_prediction))

    if args.selection == "diverse":
        rows = _select_diverse(rows, args.max_images)
    _print_summary(rows)
    return rows


def main() -> None:
    _configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=ROOT / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset", type=Path, default=ROOT / "fire-detection-from-cctv")
    parser.add_argument("--baseline", type=Path, default=ROOT / "fire-model-data" / "best.pth")
    parser.add_argument("--roi", type=Path, default=ROOT / "week6_roi_result" / "best_roi.pth")
    parser.add_argument("--v3", type=Path, default=default_v3_checkpoint())
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "week6_run" / "compare_v3_roi")
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--max-images", type=int, default=10, help="0 means all selected images")
    parser.add_argument("--selection", choices=("even", "diverse"), default="even")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = evaluate(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated: list[tuple[str, Image.Image]] = []
    for row in rows:
        image = _annotate(row)
        name = Path(row.record.image_path).stem + "_comparison.png"
        image.save(args.output_dir / name)
        annotated.append((Path(row.record.image_path).name, image))
    contact_path = args.output_dir / "comparison_contact_sheet.png"
    _contact_sheet(annotated, contact_path)
    print(f"saved_contact_sheet={contact_path}")
    print(f"saved_images={len(annotated)} output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
