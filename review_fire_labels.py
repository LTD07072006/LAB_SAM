"""Human-in-the-loop audit and relabelling tool for ``p_fire`` labels.

The model is used only to prioritise suspicious records.  It is *not* treated
as ground truth.  A person must confirm every change in the review window.

The original labels JSON is never overwritten.  The tool writes:

    working/label_review/dataset_labels_reviewed.json
    working/label_review/audit_report.json
    working/label_review/predictions.json
    working/label_review/review_log.json

Typical local usage::

    .venv\\Scripts\\python.exe review_fire_labels.py \\
        --labels "fire-model-data\\dataset_labels (1).json" \\
        --dataset-root fire-detection-from-cctv \\
        --checkpoint fire-model-data\\best.pth \\
        --roi-checkpoint week6_roi_result\\best_roi.pth \\
        --mode candidates

Useful modes:

``candidates`` (default)
    Review records with invalid labels, missing images, model/class disagreement,
    or a large model-vs-label point error.
``all``
    Review every resolvable image.  Use this when a model is not trustworthy or
    when the complete dataset needs a visual annotation pass.
``audit-only``
    Write the reports and exit without opening Tkinter.

The detector checkpoint is an assistant for triage.  A high-confidence model
disagreement may be a detector failure rather than a label failure, so the
reviewer must inspect the image before saving a correction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from project_paths import CCTV_DATASET


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = {"train", "val", "test"}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON through a sibling temporary file to avoid partial output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_labels(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Labels JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Labels JSON must contain a list of records")
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Record {index} is not a JSON object")
    return payload


def strict_binary(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number not in (0.0, 1.0):
        return None
    return int(number)


def point_norm(value: Any) -> Optional[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
        return None
    return x, y


def normalise_path_text(value: str) -> str:
    return str(value).replace("\\", "/").lower()


def normalise_device(value: Optional[str]) -> Optional[str]:
    """Accept ``auto`` and Ultralytics-style numeric GPU names."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text == "auto":
        return None
    if text.isdigit():
        return f"cuda:{text}"
    return text


@dataclass
class ImageIndex:
    """Portable lookup index for paths generated on Kaggle or Windows."""

    root: Path
    files: list[Path]

    @classmethod
    def build(cls, root: Path) -> "ImageIndex":
        if not root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {root}")
        files = [
            path.resolve()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        return cls(root.resolve(), files)

    def resolve(self, raw_path: str) -> Optional[Path]:
        """Resolve absolute Kaggle paths and nested local ``data/data`` paths."""
        raw_text = str(raw_path or "")
        normalised = normalise_path_text(raw_text)
        raw = Path(raw_text)
        direct_candidates = [raw]

        match = re.search(
            r"(?:^|/)img_data/(train|val|test)/(.+)$",
            normalised,
            flags=re.IGNORECASE,
        )
        if match:
            split, relative = match.groups()
            relative_path = Path(split) / Path(relative)
            direct_candidates.extend(
                [
                    self.root / relative_path,
                    self.root / "img_data" / relative_path,
                    self.root / "data" / relative_path,
                    self.root / "data" / "data" / "img_data" / relative_path,
                    self.root / "data" / "img_data" / relative_path,
                ]
            )

        direct_candidates.extend(
            [self.root / raw.name, self.root / "data" / "data" / "img_data" / raw.name]
        )
        for candidate in direct_candidates:
            try:
                if candidate.is_file():
                    return candidate.resolve()
            except OSError:
                continue

        # Prefer the exact split/class/filename suffix.  A basename-only match
        # is used only if it is unique; this avoids attaching test/img_103.jpg
        # to a different split when duplicate filenames exist.
        suffix = None
        if match:
            suffix = "/" + "/".join(match.groups())
        basename = raw.name.lower()
        matches = [path for path in self.files if path.name.lower() == basename]
        if suffix:
            suffix_matches = [
                path for path in matches if normalise_path_text(str(path)).endswith(suffix)
            ]
            if len(suffix_matches) == 1:
                return suffix_matches[0]
            if suffix_matches:
                matches = suffix_matches
        return matches[0] if len(matches) == 1 else None


def safe_image_size(path: Optional[Path]) -> tuple[Optional[int], Optional[int], Optional[str]]:
    if path is None:
        return None, None, "unresolved_image"
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height), None
    except Exception as exc:  # Pillow reports several format-specific errors.
        return None, None, f"image_unreadable:{type(exc).__name__}"


def pixel_error(
    predicted_norm: Optional[list[float]],
    truth_norm: Optional[tuple[float, float]],
    width: Optional[int],
    height: Optional[int],
) -> Optional[float]:
    if predicted_norm is None or truth_norm is None or not width or not height:
        return None
    dx = (float(predicted_norm[0]) - truth_norm[0]) * width
    dy = (float(predicted_norm[1]) - truth_norm[1]) * height
    return math.hypot(dx, dy)


def structural_audit(item: dict[str, Any], index: int, image_index: ImageIndex) -> dict[str, Any]:
    raw_path = str(item.get("image_path", ""))
    resolved = image_index.resolve(raw_path)
    label = strict_binary(item.get("has_fire"))
    truth = point_norm(item.get("p_fire"))
    width, height, image_error = safe_image_size(resolved)
    flags: list[str] = []

    if not raw_path:
        flags.append("missing_image_path")
    if label is None:
        flags.append("invalid_has_fire")
    if resolved is None:
        flags.append("unresolved_image")
    if image_error and image_error != "unresolved_image":
        flags.append(image_error)
    if label == 1 and truth is None:
        flags.append("fire_without_valid_p_fire")
    if label == 0 and truth is not None and any(abs(value) > 1e-12 for value in truth):
        flags.append("no_fire_with_nonzero_p_fire")

    return {
        "record_index": index,
        "image_path": raw_path,
        "resolved_image": str(resolved) if resolved else None,
        "width": width,
        "height": height,
        "has_fire": label,
        "p_fire": list(truth) if truth is not None else None,
        "flags": flags,
        "score": float(len(flags) * 100),
        "prediction": None,
    }


def mark_duplicate_images(entries: list[dict[str, Any]]) -> None:
    """Record path aliases without treating them as 1034 separate images."""
    by_path: dict[str, list[int]] = {}
    for entry in entries:
        resolved = entry.get("resolved_image")
        if resolved:
            by_path.setdefault(str(Path(resolved).resolve()).lower(), []).append(
                int(entry["record_index"])
            )
    for indices in by_path.values():
        if len(indices) <= 1:
            continue
        # The current JSON has two path aliases per image.  Keep the aliases
        # in the output for compatibility, but do not make every alias a
        # suspicious label.  A disagreement in has_fire/p_fire is genuinely
        # suspicious and is still promoted to the GUI.
        signatures = {
            (
                entries[index].get("has_fire"),
                tuple(entries[index].get("p_fire") or []),
            )
            for index in indices
        }
        for index in indices:
            entry = entries[index]
            entry["duplicate_record_indices"] = indices
            if len(signatures) > 1:
                entry["flags"] = list(dict.fromkeys([*entry["flags"], "duplicate_label_conflict"]))
                entry["score"] = float(len(entry["flags"]) * 100)


class ModelPredictor:
    """Lazy adapter around the project's existing detector and ROI refiner."""

    def __init__(
        self,
        checkpoint: Path,
        roi_checkpoint: Optional[Path],
        device: Optional[str],
        threshold: float,
        batch_size: int,
    ) -> None:
        # Imports are deliberately lazy so ``--audit-only`` can work without
        # torch/timm and so syntax/data validation does not load a 34 MB model.
        from fire_detector import FireDetector

        self.detector = FireDetector(
            checkpoint,
            device=device,
            threshold=threshold,
            use_amp=True,
        )
        self.roi = None
        if roi_checkpoint:
            from narrow_localizer import ROIRefinerInference

            self.roi = ROIRefinerInference(roi_checkpoint, device=device)
        self.batch_size = max(1, int(batch_size))

    @staticmethod
    def _prediction_from_result(result: Any, width: int, height: int) -> dict[str, Any]:
        output: dict[str, Any] = {
            "confidence": float(result.confidence),
            "detected": bool(result.detected),
            "coarse_pixel": None,
            "coarse_norm": None,
            "roi_confidence": None,
            "roi_pixel": None,
            "roi_norm": None,
            "latency_ms": float(getattr(result, "latency_ms", 0.0)),
        }
        if result.pixel is not None:
            x, y = float(result.pixel[0]), float(result.pixel[1])
            output["coarse_pixel"] = [x, y]
            output["coarse_norm"] = [x / max(1, width), y / max(1, height)]
        return output

    def predict(self, entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        results: dict[int, dict[str, Any]] = {}
        usable = [
            entry
            for entry in entries
            if entry.get("resolved_image") and entry.get("width") and entry.get("height")
        ]
        for start in range(0, len(usable), self.batch_size):
            batch = usable[start : start + self.batch_size]
            paths = [entry["resolved_image"] for entry in batch]
            try:
                detections = self.detector.detect_many(paths, warmup=(start == 0))
            except Exception as exc:
                # One damaged file should not discard predictions for the rest
                # of the dataset. Retry one-by-one and record the error.
                print(
                    f"warning: batch inference failed at records "
                    f"{batch[0]['record_index']}..{batch[-1]['record_index']}: {exc}",
                    file=sys.stderr,
                )
                detections = []
                for path in paths:
                    try:
                        detections.append(self.detector.detect(path))
                    except Exception as single_exc:
                        detections.append(single_exc)

            for entry, detection in zip(batch, detections):
                record_index = int(entry["record_index"])
                if isinstance(detection, Exception):
                    results[record_index] = {"error": f"inference:{detection}"}
                    continue
                width, height = int(entry["width"]), int(entry["height"])
                prediction = self._prediction_from_result(detection, width, height)
                if self.roi is not None and prediction["coarse_pixel"] is not None:
                    try:
                        from PIL import Image

                        with Image.open(entry["resolved_image"]) as image:
                            refined = self.roi.refine(image.convert("RGB"), prediction["coarse_pixel"])
                        rx, ry = refined.point
                        prediction["roi_confidence"] = float(refined.confidence)
                        prediction["roi_pixel"] = [float(rx), float(ry)]
                        prediction["roi_norm"] = [
                            float(rx) / max(1, width),
                            float(ry) / max(1, height),
                        ]
                    except Exception as exc:
                        prediction["roi_error"] = str(exc)
                results[record_index] = prediction
        return results


def enrich_audit(
    entries: list[dict[str, Any]],
    predictions: dict[int, dict[str, Any]],
    threshold: float,
    point_error_px_threshold: float,
    roi_shift_px_threshold: float,
) -> None:
    for entry in entries:
        prediction = predictions.get(int(entry["record_index"]))
        entry["prediction"] = prediction
        flags = list(entry["flags"])
        if not prediction or prediction.get("error"):
            entry["flags"] = flags
            entry["score"] = float(len(flags) * 100)
            continue

        confidence = float(prediction.get("confidence", 0.0))
        label = entry.get("has_fire")
        if label == 1 and confidence < threshold:
            flags.append("model_misses_fire")
        if label == 0 and confidence >= threshold:
            flags.append("model_false_positive")

        error = pixel_error(
            prediction.get("coarse_norm"),
            tuple(entry["p_fire"]) if entry.get("p_fire") else None,
            entry.get("width"),
            entry.get("height"),
        )
        if error is not None:
            prediction["coarse_gt_error_px"] = float(error)
            if error >= point_error_px_threshold:
                flags.append("large_coarse_gt_error")

        roi_error = pixel_error(
            prediction.get("roi_norm"),
            tuple(entry["p_fire"]) if entry.get("p_fire") else None,
            entry.get("width"),
            entry.get("height"),
        )
        if roi_error is not None:
            prediction["roi_gt_error_px"] = float(roi_error)

        coarse = prediction.get("coarse_pixel")
        roi = prediction.get("roi_pixel")
        if coarse and roi:
            shift = math.hypot(float(roi[0]) - float(coarse[0]), float(roi[1]) - float(coarse[1]))
            prediction["roi_shift_px"] = float(shift)
            if shift >= roi_shift_px_threshold:
                flags.append("large_roi_shift")

        # Larger values put high-risk cases near the front of the GUI.  This
        # is a review priority, not a probability that the label is wrong.
        score = float(len(set(flags)) * 100)
        if "model_misses_fire" in flags or "model_false_positive" in flags:
            score += 50.0
        if "large_coarse_gt_error" in flags:
            score += min(100.0, float(prediction.get("coarse_gt_error_px", 0.0)))
        if "large_roi_shift" in flags:
            score += min(50.0, float(prediction.get("roi_shift_px", 0.0)))
        entry["flags"] = list(dict.fromkeys(flags))
        entry["score"] = score


def _record_before(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "has_fire": item.get("has_fire"),
        "p_fire": copy.deepcopy(item.get("p_fire")),
    }


def _normalised_change(item: dict[str, Any], has_fire: int, x: float, y: float) -> None:
    item["has_fire"] = int(has_fire)
    item["p_fire"] = [0.0, 0.0] if int(has_fire) == 0 else [float(x), float(y)]


class LabelReviewApp:
    """Small Tkinter reviewer with mouse and keyboard annotation controls."""

    def __init__(
        self,
        tk: Any,
        ttk: Any,
        image_tk: Any,
        items: list[dict[str, Any]],
        original_items: list[dict[str, Any]],
        entries_by_index: dict[int, dict[str, Any]],
        selected_indices: list[int],
        output_labels: Path,
        output_log: Path,
    ) -> None:
        self.tk = tk
        self.ttk = ttk
        self.image_tk = image_tk
        self.items = items
        self.original_items = original_items
        self.entries = entries_by_index
        self.indices = selected_indices
        self.output_labels = output_labels
        self.output_log = output_log
        self.position = 0
        self.log: list[dict[str, Any]] = []
        self.photo = None
        self.source_image = None
        self.display_scale = 1.0
        self.display_origin = (0.0, 0.0)
        self.display_size = (1, 1)

        self.root = tk.Tk()
        self.root.title("Fire label review — human confirmation required")
        self.root.geometry("1320x820")
        self.root.minsize(1050, 700)
        self._build_widgets()
        self.root.bind("<KeyPress>", self._key_press)
        self.canvas.bind("<Button-1>", self._canvas_click)
        self.root.protocol("WM_DELETE_WINDOW", self._finish)
        self._show_current()

    def _build_widgets(self) -> None:
        main = self.ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        self.canvas = self.tk.Canvas(main, width=920, height=700, background="#202020", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        side = self.ttk.Frame(main, width=360)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)

        self.title_var = self.tk.StringVar()
        self.info_var = self.tk.StringVar()
        self.status_var = self.tk.StringVar()
        self.has_fire_var = self.tk.StringVar()
        self.x_var = self.tk.StringVar()
        self.y_var = self.tk.StringVar()

        self.ttk.Label(side, textvariable=self.title_var, wraplength=340).pack(anchor="w", pady=(0, 6))
        self.ttk.Label(side, textvariable=self.info_var, justify="left", wraplength=340).pack(anchor="w", pady=(0, 10))

        form = self.ttk.LabelFrame(side, text="Nhãn sẽ lưu")
        form.pack(fill="x", pady=4)
        for row, (label, variable) in enumerate(
            [("has_fire (0/1)", self.has_fire_var), ("p_fire x [0,1]", self.x_var), ("p_fire y [0,1]", self.y_var)]
        ):
            self.ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=4)
            self.ttk.Entry(form, textvariable=variable, width=18).grid(row=row, column=1, sticky="e", padx=6, pady=4)

        buttons = self.ttk.LabelFrame(side, text="Thao tác")
        buttons.pack(fill="x", pady=8)
        button_specs = [
            ("Lưu thay đổi + tiếp [S]", self._save_next),
            ("Lưu hiện tại", self._save_current),
            ("Bỏ qua [K]", self._skip),
            ("Ảnh trước [B]", self._previous),
            ("Khôi phục nhãn cũ [R]", self._reset),
            ("Đánh dấu không lửa [N]", self._set_no_fire),
            ("Dùng coarse dự đoán", self._use_coarse),
            ("Dùng ROI dự đoán", self._use_roi),
            ("Hoàn tất", self._finish),
        ]
        for text, command in button_specs:
            self.ttk.Button(buttons, text=text, command=command).pack(fill="x", padx=6, pady=2)

        self.ttk.Label(
            side,
            text=(
                "Click trực tiếp lên ảnh để đặt điểm chân lửa và tự đặt has_fire=1.\n"
                "Màu: GT hiện tại xanh lá | coarse xanh dương | ROI cam.\n"
                "Model chỉ gợi ý ứng viên; hãy nhìn ảnh trước khi lưu."
            ),
            justify="left",
            wraplength=340,
        ).pack(anchor="w", pady=8)
        self.ttk.Label(side, textvariable=self.status_var, foreground="#9a5b00", wraplength=340).pack(anchor="w")

    def _current_index(self) -> Optional[int]:
        if not self.indices or self.position < 0 or self.position >= len(self.indices):
            return None
        return self.indices[self.position]

    def _show_current(self) -> None:
        record_index = self._current_index()
        if record_index is None:
            self.title_var.set("Không còn ảnh cần review")
            self.info_var.set("Bạn có thể đóng cửa sổ. Output đã được lưu sau mỗi thao tác.")
            self.canvas.delete("all")
            return
        item = self.items[record_index]
        entry = self.entries[record_index]
        self._load_form(item)
        flags = ", ".join(entry.get("flags", [])) or "không có cờ nghi vấn"
        self.title_var.set(f"{self.position + 1}/{len(self.indices)} — record {record_index}\n{item.get('image_path', '')}")
        prediction = entry.get("prediction") or {}
        pred_text = (
            f"confidence={prediction.get('confidence', 'n/a')}\n"
            f"coarse_gt_error={prediction.get('coarse_gt_error_px', 'n/a')} px\n"
            f"roi_shift={prediction.get('roi_shift_px', 'n/a')} px"
        )
        self.info_var.set(f"Flags: {flags}\n{pred_text}")
        self.status_var.set("Click ảnh để đặt lại p_fire; S để lưu và sang ảnh kế.")
        self._draw_image()

    def _load_form(self, item: dict[str, Any]) -> None:
        label = strict_binary(item.get("has_fire"))
        point = point_norm(item.get("p_fire")) or (0.0, 0.0)
        self.has_fire_var.set(str(label if label is not None else 1))
        self.x_var.set(f"{point[0]:.6f}")
        self.y_var.set(f"{point[1]:.6f}")

    def _draw_image(self) -> None:
        record_index = self._current_index()
        self.canvas.delete("all")
        if record_index is None:
            return
        entry = self.entries[record_index]
        path = entry.get("resolved_image")
        if not path:
            self.canvas.create_text(30, 30, anchor="nw", fill="white", text="Ảnh không resolve được")
            return
        try:
            from PIL import Image, ImageDraw

            with Image.open(path) as opened:
                image = opened.convert("RGB")
            self.source_image = image
            canvas_width = max(100, int(self.canvas.winfo_width()))
            canvas_height = max(100, int(self.canvas.winfo_height()))
            scale = min((canvas_width - 20) / image.width, (canvas_height - 20) / image.height)
            scale = max(0.01, scale)
            display_width = max(1, int(round(image.width * scale)))
            display_height = max(1, int(round(image.height * scale)))
            display = image.resize((display_width, display_height), Image.Resampling.LANCZOS)
            drawer = ImageDraw.Draw(display)

            def draw_point(point: Optional[list[float]], colour: str, label: str) -> None:
                if point is None:
                    return
                px = float(point[0]) * image.width if max(abs(float(point[0])), abs(float(point[1]))) <= 1.5 else float(point[0])
                py = float(point[1]) * image.height if max(abs(float(point[0])), abs(float(point[1]))) <= 1.5 else float(point[1])
                x, y = px * scale, py * scale
                radius = max(5, int(round(7 * scale)))
                drawer.ellipse((x - radius, y - radius, x + radius, y + radius), outline=colour, width=max(2, int(round(2 * scale))))
                drawer.line((x - radius * 2, y, x + radius * 2, y), fill=colour, width=2)
                drawer.line((x, y - radius * 2, x, y + radius * 2), fill=colour, width=2)
                drawer.text((x + radius + 2, y + 2), label, fill=colour)

            item = self.items[record_index]
            current = point_norm(item.get("p_fire"))
            draw_point(list(current) if current and strict_binary(item.get("has_fire")) == 1 else None, "#39d353", "GT")
            prediction = entry.get("prediction") or {}
            draw_point(prediction.get("coarse_norm"), "#268bd2", "coarse")
            draw_point(prediction.get("roi_norm"), "#ff9f1c", "ROI")

            origin_x = (canvas_width - display_width) / 2.0
            origin_y = (canvas_height - display_height) / 2.0
            self.display_scale = scale
            self.display_origin = (origin_x, origin_y)
            self.display_size = (display_width, display_height)
            self.photo = self.image_tk.PhotoImage(display)
            self.canvas.create_image(origin_x, origin_y, anchor="nw", image=self.photo)
        except Exception as exc:
            self.canvas.create_text(30, 30, anchor="nw", fill="white", text=f"Không hiển thị được ảnh: {exc}")

    def _canvas_click(self, event: Any) -> None:
        record_index = self._current_index()
        if record_index is None or self.source_image is None:
            return
        origin_x, origin_y = self.display_origin
        display_width, display_height = self.display_size
        if not (origin_x <= event.x <= origin_x + display_width and origin_y <= event.y <= origin_y + display_height):
            return
        x = (event.x - origin_x) / self.display_scale
        y = (event.y - origin_y) / self.display_scale
        self.has_fire_var.set("1")
        self.x_var.set(f"{min(1.0, max(0.0, x / self.source_image.width)):.6f}")
        self.y_var.set(f"{min(1.0, max(0.0, y / self.source_image.height)):.6f}")
        self._apply_form_preview()
        self.status_var.set(f"Đã chọn pixel ({x:.1f}, {y:.1f}); nhấn S để lưu.")

    def _apply_form_preview(self) -> None:
        record_index = self._current_index()
        if record_index is None:
            return
        try:
            label = int(self.has_fire_var.get())
            x, y = float(self.x_var.get()), float(self.y_var.get())
            if label not in (0, 1) or not 0 <= x <= 1 or not 0 <= y <= 1:
                return
            preview = copy.deepcopy(self.items[record_index])
            _normalised_change(preview, label, x, y)
            old = self.items[record_index]
            self.items[record_index] = preview
            self._draw_image()
            self.items[record_index] = old
        except (TypeError, ValueError):
            return

    def _read_form(self) -> tuple[int, float, float]:
        label = int(self.has_fire_var.get())
        x, y = float(self.x_var.get()), float(self.y_var.get())
        if label not in (0, 1):
            raise ValueError("has_fire must be 0 or 1")
        if not 0 <= x <= 1 or not 0 <= y <= 1:
            raise ValueError("p_fire coordinates must be in [0, 1]")
        return label, x, y

    def _save_current(self, move_next: bool = False) -> None:
        record_index = self._current_index()
        if record_index is None:
            return
        try:
            label, x, y = self._read_form()
        except ValueError as exc:
            self.status_var.set(f"Lỗi nhãn: {exc}")
            return
        affected = self.entries[record_index].get("duplicate_record_indices", [record_index])
        before = {
            str(index): _record_before(self.items[index])
            for index in affected
        }
        for index in affected:
            _normalised_change(self.items[index], label, x, y)
        after = {
            str(index): _record_before(self.items[index])
            for index in affected
        }
        self.log.append({
            "time": now_iso(),
            "record_index": record_index,
            "image_path": self.items[record_index].get("image_path"),
            "action": "save",
            "affected_record_indices": affected,
            "before": before,
            "after": after,
        })
        self._persist()
        self.status_var.set("Đã lưu bản review; file gốc không bị sửa.")
        if move_next:
            self._next()
        else:
            self._show_current()

    def _save_next(self) -> None:
        self._save_current(move_next=True)

    def _skip(self) -> None:
        record_index = self._current_index()
        if record_index is None:
            return
        self.log.append({
            "time": now_iso(),
            "record_index": record_index,
            "image_path": self.items[record_index].get("image_path"),
            "action": "skip",
        })
        self._persist()
        self._next()

    def _next(self) -> None:
        if self.position < len(self.indices) - 1:
            self.position += 1
            self._show_current()
        else:
            self._show_current()
            self.status_var.set("Đã tới ảnh cuối. Nhấn Hoàn tất để đóng.")

    def _previous(self) -> None:
        if self.position > 0:
            self.position -= 1
            self._show_current()

    def _reset(self) -> None:
        record_index = self._current_index()
        if record_index is None:
            return
        affected = self.entries[record_index].get("duplicate_record_indices", [record_index])
        for index in affected:
            self.items[index] = copy.deepcopy(self.original_items[index])
        self._load_form(self.items[record_index])
        self._draw_image()
        self.status_var.set("Đã khôi phục nhóm alias về nhãn gốc; chưa ghi thay đổi.")

    def _set_no_fire(self) -> None:
        self.has_fire_var.set("0")
        self.x_var.set("0.000000")
        self.y_var.set("0.000000")
        self._apply_form_preview()

    def _use_prediction(self, key: str) -> None:
        record_index = self._current_index()
        if record_index is None:
            return
        point = (self.entries[record_index].get("prediction") or {}).get(key)
        if point is None:
            self.status_var.set(f"Không có {key} cho ảnh này.")
            return
        self.has_fire_var.set("1")
        self.x_var.set(f"{float(point[0]):.6f}")
        self.y_var.set(f"{float(point[1]):.6f}")
        self._apply_form_preview()

    def _use_coarse(self) -> None:
        self._use_prediction("coarse_norm")

    def _use_roi(self) -> None:
        self._use_prediction("roi_norm")

    def _persist(self) -> None:
        write_json_atomic(self.output_labels, self.items)
        write_json_atomic(self.output_log, self.log)

    def _key_press(self, event: Any) -> None:
        key = str(event.keysym).lower()
        if key == "s":
            self._save_next()
        elif key == "k":
            self._skip()
        elif key == "b":
            self._previous()
        elif key == "r":
            self._reset()
        elif key == "n":
            self._set_no_fire()

    def _finish(self) -> None:
        self._persist()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(
    items: list[dict[str, Any]],
    original_items: list[dict[str, Any]],
    entries: list[dict[str, Any]],
    selected_indices: list[int],
    output_labels: Path,
    output_log: Path,
) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk
    except Exception as exc:
        raise RuntimeError(
            "Không mở được GUI. Cài/đảm bảo Tkinter và Pillow, "
            "hoặc dùng --audit-only để chỉ tạo báo cáo."
        ) from exc
    app = LabelReviewApp(
        tk,
        ttk,
        ImageTk,
        items,
        original_items,
        {int(entry["record_index"]): entry for entry in entries},
        selected_indices,
        output_labels,
        output_log,
    )
    app.run()


def unique_review_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Choose one representative entry per resolved physical image."""
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        resolved = entry.get("resolved_image")
        if resolved:
            key = str(Path(resolved).resolve()).lower()
        else:
            key = f"record:{entry['record_index']}"
        if key in seen:
            continue
        seen.add(key)
        selected.append(entry)
    return selected


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, default=root / "fire-model-data" / "dataset_labels (1).json")
    parser.add_argument("--dataset-root", type=Path, default=CCTV_DATASET)
    parser.add_argument("--checkpoint", type=Path, default=root / "fire-model-data" / "best.pth")
    parser.add_argument("--roi-checkpoint", type=Path, default=None)
    parser.add_argument("--no-model", action="store_true", help="Không chạy detector; chỉ audit cấu trúc và review thủ công")
    parser.add_argument("--strict-model", action="store_true", help="Dừng nếu checkpoint/dependencies không load được")
    parser.add_argument("--device", default=None, help="cpu, cuda, cuda:0 hoặc auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--point-error-px", type=float, default=30.0)
    parser.add_argument("--roi-shift-px", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--mode", choices=("candidates", "all", "audit-only"), default="candidates")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số ảnh mở trong GUI; 0 là không giới hạn")
    parser.add_argument("--output-dir", type=Path, default=root / "working" / "label_review")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ dataset_labels_reviewed.json đã có")
    parser.add_argument("--refresh-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels_path = args.labels.resolve()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_labels = output_dir / "dataset_labels_reviewed.json"
    output_report = output_dir / "audit_report.json"
    output_predictions = output_dir / "predictions.json"
    output_log = output_dir / "review_log.json"

    if output_labels == labels_path:
        raise ValueError("Refuse to overwrite the original labels JSON")
    original_items = load_labels(labels_path)
    working_items = copy.deepcopy(original_items)
    if args.resume and output_labels.is_file():
        working_items = load_labels(output_labels)
        if len(working_items) != len(original_items):
            raise ValueError("Resume labels has a different record count than the original JSON")

    image_index = ImageIndex.build(dataset_root)
    entries = [structural_audit(item, index, image_index) for index, item in enumerate(working_items)]
    mark_duplicate_images(entries)

    predictions: dict[int, dict[str, Any]] = {}
    checkpoint_available = args.checkpoint.is_file() and not args.no_model
    if checkpoint_available:
        cache_ok = output_predictions.is_file() and not args.refresh_predictions
        if cache_ok:
            try:
                cached = json.loads(output_predictions.read_text(encoding="utf-8"))
                predictions = {int(key): value for key, value in cached.get("records", {}).items()}
                print(f"loaded_prediction_cache={output_predictions}")
            except Exception as exc:
                print(f"warning: prediction cache ignored: {exc}", file=sys.stderr)
        if not predictions:
            try:
                print(f"loading_checkpoint={args.checkpoint}")
                predictor = ModelPredictor(
                    args.checkpoint.resolve(),
                    args.roi_checkpoint.resolve() if args.roi_checkpoint else None,
                    normalise_device(args.device),
                    args.threshold,
                    args.batch_size,
                )
                predictions = predictor.predict(entries)
                write_json_atomic(
                    output_predictions,
                    {
                        "created_at": now_iso(),
                        "checkpoint": str(args.checkpoint.resolve()),
                        "roi_checkpoint": str(args.roi_checkpoint.resolve()) if args.roi_checkpoint else None,
                        "threshold": args.threshold,
                        "records": {str(key): value for key, value in predictions.items()},
                    },
                )
            except Exception as exc:
                message = (
                    f"Không load/chạy được model: {exc}\n"
                    "Có thể chạy lại với --no-model để review thủ công, "
                    "hoặc dùng đúng .venv có torch/timm/Pillow."
                )
                if args.strict_model:
                    raise RuntimeError(message) from exc
                print("warning: " + message, file=sys.stderr)
    elif not args.no_model:
        print(f"warning: checkpoint không tồn tại hoặc không dùng: {args.checkpoint}; chuyển sang audit thủ công", file=sys.stderr)

    enrich_audit(entries, predictions, args.threshold, args.point_error_px, args.roi_shift_px)
    flagged = [entry for entry in entries if entry.get("flags")]
    flagged.sort(key=lambda entry: (-float(entry.get("score", 0.0)), int(entry["record_index"])))
    if args.mode == "all":
        selected = [entry for entry in entries if entry.get("resolved_image")]
        selected.sort(key=lambda entry: int(entry["record_index"]))
    else:
        # Unresolved files remain visible in audit_report.json, but cannot be
        # corrected through an image GUI.  Review only drawable candidates.
        selected = [entry for entry in flagged if entry.get("resolved_image")]
    selected = unique_review_entries(selected)
    if args.limit > 0:
        selected = selected[: args.limit]
    selected_indices = [int(entry["record_index"]) for entry in selected]

    report = {
        "created_at": now_iso(),
        "source_labels": str(labels_path),
        "dataset_root": str(dataset_root),
        "checkpoint": str(args.checkpoint.resolve()) if checkpoint_available else None,
        "roi_checkpoint": str(args.roi_checkpoint.resolve()) if args.roi_checkpoint else None,
        "mode": args.mode,
        "thresholds": {
            "classification": args.threshold,
            "point_error_px": args.point_error_px,
            "roi_shift_px": args.roi_shift_px,
        },
        "summary": {
            "records": len(entries),
            "resolved_images": sum(bool(entry.get("resolved_image")) for entry in entries),
            "unique_resolved_images": len({entry.get("resolved_image") for entry in entries if entry.get("resolved_image")}),
            "duplicate_alias_records": sum(bool(entry.get("duplicate_record_indices")) for entry in entries),
            "duplicate_label_conflict_records": sum("duplicate_label_conflict" in entry.get("flags", []) for entry in entries),
            "flagged_records": len(flagged),
            "selected_for_review": len(selected),
            "class_1": sum(entry.get("has_fire") == 1 for entry in entries),
            "class_0": sum(entry.get("has_fire") == 0 for entry in entries),
        },
        "records": entries,
    }
    write_json_atomic(output_report, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"saved_audit={output_report}")
    if args.mode == "audit-only":
        return
    if not selected_indices:
        print("Không có record bị gắn cờ. Dùng --mode all nếu muốn xem toàn bộ ảnh.")
        return

    print(f"opening_review_gui={len(selected_indices)} images")
    launch_gui(
        working_items,
        original_items,
        entries,
        selected_indices,
        output_labels,
        output_log,
    )
    print(f"saved_reviewed_labels={output_labels}")
    print(f"saved_review_log={output_log}")


if __name__ == "__main__":
    main()
