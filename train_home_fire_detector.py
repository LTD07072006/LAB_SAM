"""Train an independent YOLO branch on the 1.9 GB Home Fire dataset.

YOLO learns bounding boxes first. Downstream code converts a selected box into
a bottom-contact point or a bottom band for calibrated ray casting. This
branch never overwrites the existing point-detector or ROI checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _has_yolo_layout(path: Path) -> bool:
    return all((path / split / "images").is_dir() and (path / split / "labels").is_dir() for split in SPLITS)


def resolve_dataset_root(value: Path) -> Path:
    """Resolve a direct or nested YOLO dataset directory."""
    root = value.expanduser().resolve()
    if _has_yolo_layout(root):
        return root
    candidates = [path for path in root.rglob("*") if path.is_dir() and _has_yolo_layout(path)]
    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy YOLO layout train/val/test dưới {root}. "
            "Cần <split>/images và <split>/labels."
        )
    candidates.sort(key=lambda path: (len(path.parts), str(path).lower()))
    return candidates[0]


def discover_class_ids(dataset_root: Path) -> list[int]:
    """Read class ids without assuming which id means fire."""
    class_ids: set[int] = set()
    for label_path in dataset_root.rglob("*.txt"):
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if not fields:
                continue
            try:
                class_id = int(fields[0])
            except ValueError:
                continue
            if class_id >= 0:
                class_ids.add(class_id)
    if not class_ids:
        raise ValueError(f"Không tìm thấy class id hợp lệ trong {dataset_root}")
    expected = list(range(max(class_ids) + 1))
    if sorted(class_ids) != expected:
        raise ValueError(f"Class ids phải liên tục từ 0; đã thấy {sorted(class_ids)}")
    return expected


def write_data_yaml(dataset_root: Path, class_ids: list[int], names: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    names_text = ", ".join(json.dumps(name, ensure_ascii=False) for name in names)
    content = (
        f"path: {json.dumps(dataset_root.as_posix(), ensure_ascii=False)}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"names: [{names_text}]\n"
    )
    output.write_text(content, encoding="utf-8")


def _default_names(class_ids: list[int], class_names: str) -> list[str]:
    supplied = [item.strip() for item in class_names.split(",") if item.strip()]
    if supplied and len(supplied) != len(class_ids):
        raise ValueError(f"--class-names cần {len(class_ids)} tên, nhưng nhận {len(supplied)}")
    return supplied or [f"class_{class_id}" for class_id in class_ids]


def _link_or_copy(source: Path, destination: Path) -> str:
    """Materialise an image without duplicating bytes when hard links work."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return "existing"
    try:
        # The dataset and working directory are normally on the same NTFS
        # volume, so this is effectively free in disk space.
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _remap_label_file(source: Path, destination: Path, source_class: int) -> int:
    """Keep only one source class and remap it to YOLO class 0."""
    kept = 0
    output_lines: list[str] = []
    if source.is_file():
        for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) != 5:
                continue
            try:
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
            except ValueError:
                continue
            if class_id != source_class:
                continue
            if not all(value == value and abs(value) != float("inf") for value in values):
                continue
            if values[2] <= 0.0 or values[3] <= 0.0:
                continue
            output_lines.append("0 " + " ".join(fields[1:]))
            kept += 1
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")
    return kept


def build_single_class_dataset(
    dataset_root: Path,
    output_root: Path,
    source_fire_class: int,
) -> dict[str, int | str]:
    """Create a temporary single-class view without touching the source data.

    Ultralytics resolves labels from the image path (``images`` -> ``labels``),
    therefore a small materialised view is safer than trying to override label
    directories in a YAML file. Images are hard-linked whenever possible.
    """
    image_count = 0
    box_count = 0
    hardlinks = 0
    copies = 0
    for split in SPLITS:
        source_images = dataset_root / split / "images"
        source_labels = dataset_root / split / "labels"
        target_images = output_root / split / "images"
        target_labels = output_root / split / "labels"
        if not source_images.is_dir() or not source_labels.is_dir():
            raise FileNotFoundError(f"Thiếu split {split}: {source_images} hoặc {source_labels}")
        for image_path in source_images.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative = image_path.relative_to(source_images)
            result = _link_or_copy(image_path, target_images / relative)
            if result == "hardlink":
                hardlinks += 1
            elif result == "copy":
                copies += 1
            image_count += 1
            source_label = source_labels / relative.with_suffix(".txt")
            box_count += _remap_label_file(
                source_label,
                target_labels / relative.with_suffix(".txt"),
                source_fire_class,
            )
    if image_count == 0:
        raise ValueError(f"Không tìm thấy ảnh để tạo single-class view từ {dataset_root}")
    return {
        "image_count": image_count,
        "box_count": box_count,
        "hardlinks": hardlinks,
        "copies": copies,
        "source_fire_class": source_fire_class,
        "output_root": str(output_root),
    }


def train(args: argparse.Namespace) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài Ultralytics. Hãy chạy trong đúng .venv: python -m pip install ultralytics"
        ) from exc

    project = args.project.expanduser().resolve()
    run_dir = project / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root(args.dataset_root)
    source_class_ids = discover_class_ids(dataset_root)
    if args.single_fire_class is not None:
        if args.single_fire_class not in source_class_ids:
            raise ValueError(
                f"--single-fire-class={args.single_fire_class} không tồn tại; "
                f"class ids={source_class_ids}"
            )
        train_dataset_root = run_dir / "single_class_dataset"
        remap_stats = build_single_class_dataset(
            dataset_root,
            train_dataset_root,
            args.single_fire_class,
        )
        class_ids = [0]
        names = [args.single_class_name]
        fire_class_id = 0
    else:
        train_dataset_root = dataset_root
        class_ids = source_class_ids
        names = _default_names(class_ids, args.class_names)
        if args.fire_class is not None and args.fire_class not in class_ids:
            raise ValueError(
                f"--fire-class={args.fire_class} không tồn tại; class ids={class_ids}"
            )
        remap_stats = None
        fire_class_id = args.fire_class

    data_yaml = run_dir / "home_fire_data.yaml"
    write_data_yaml(train_dataset_root, class_ids, names, data_yaml)
    metadata = {
        "dataset_root": str(train_dataset_root),
        "original_dataset_root": str(dataset_root),
        "data_yaml": str(data_yaml),
        "class_ids": class_ids,
        "class_names": names,
        "fire_class_id": fire_class_id,
        "source_fire_class_id": args.single_fire_class,
        "fire_class_mapping_verified": fire_class_id is not None,
        "single_class_remap": remap_stats,
        "model": args.model,
        "imgsz": args.imgsz, "epochs": args.epochs, "batch": args.batch, "device": args.device,
    }
    (run_dir / "training_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"dataset_root={dataset_root}")
    print(f"class_ids={class_ids} class_names={names}")
    if remap_stats is not None:
        print(
            "single_class_view="
            f"{train_dataset_root} images={remap_stats['image_count']} "
            f"boxes={remap_stats['box_count']} hardlinks={remap_stats['hardlinks']} "
            f"copies={remap_stats['copies']}"
        )
    if fire_class_id is None:
        print("warning=fire_class_unset; verify class semantics before localization")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml), "epochs": int(args.epochs), "imgsz": int(args.imgsz),
        "batch": int(args.batch), "project": str(project), "name": args.name,
        "exist_ok": True, "workers": int(args.workers), "patience": int(args.patience),
        "cache": bool(args.cache), "verbose": True,
    }
    if args.device is not None:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)
    print(f"best_checkpoint={run_dir / 'weights' / 'best.pt'}")
    print(f"last_checkpoint={run_dir / 'weights' / 'last.pt'}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "home-fire-dataset")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--class-names", default="")
    parser.add_argument(
        "--fire-class",
        type=int,
        default=None,
        help="Verified fire class id in two-class mode; does not alter labels",
    )
    parser.add_argument(
        "--single-fire-class",
        type=int,
        default=None,
        help="Source class to keep and remap to class 0 in a temporary view",
    )
    parser.add_argument(
        "--single-class-name",
        default="fire",
        help="Name used for class 0 in --single-fire-class mode",
    )
    parser.add_argument("--project", type=Path, default=root / "working" / "home_fire_yolo")
    parser.add_argument("--name", default="bbox640")
    return parser


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
