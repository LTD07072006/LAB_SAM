"""Kaggle T4 launcher for the 640x640 detector training stage.

The Kaggle input filesystem is read-only and generated directory names depend
on how files were uploaded. This launcher discovers the project files below
``/kaggle/input`` and writes checkpoints to ``/kaggle/working``.

The Home Fire YOLO dataset is not needed here: it is a separate bbox branch.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional
import zipfile


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _has_cctv_layout(path: Path) -> bool:
    return any(
        (path / suffix).is_dir()
        for suffix in ("data/data/img_data", "data/img_data", "img_data")
    )


def _zip_image_member(name: str) -> Optional[str]:
    """Return a safe staged path for an image inside the CCTV ZIP.

    The original archive contains ``data/img_data/...`` and also contains
    ``data/video_data``.  Only the image tree is copied to the writable
    Kaggle working directory; videos are deliberately ignored.
    """
    normalized = str(name).replace("\\", "/").lstrip("/")
    lowered = normalized.lower()
    marker = "data/img_data/"
    position = lowered.find(marker)
    if position >= 0:
        relative = normalized[position:]
    else:
        # Be tolerant of a ZIP made from the contents of ``data`` rather
        # than from the parent directory.
        marker = "img_data/"
        position = lowered.find(marker)
        if position < 0:
            return None
        relative = "data/" + normalized[position:]

    parts = relative.split("/")
    if len(parts) < 5 or parts[0].lower() != "data" or parts[1].lower() != "img_data":
        return None
    if parts[2].lower() not in {"train", "test", "val"}:
        return None
    if Path(parts[-1]).suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    return relative


def unpack_data_zip(zip_path: Path, output_root: Path) -> Path:
    """Extract only ``data/img_data`` from ``data.zip``.

    ``/kaggle/input`` is read-only, so the trainer cannot point directly at a
    ZIP.  The extracted root keeps the same relative layout expected by
    ``train_week6.load_records``.  The path check prevents a malicious ZIP
    member from escaping ``output_root``.
    """
    zip_path = Path(zip_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"data.zip không tồn tại: {zip_path}")

    train_dir = output_root / "data" / "img_data" / "train"
    test_dir = output_root / "data" / "img_data" / "test"
    if _has_cctv_layout(output_root) and any(train_dir.rglob("*")) and any(test_dir.rglob("*")):
        print(f"Đã dùng dữ liệu đã giải nén: {output_root}")
        return output_root

    output_root.mkdir(parents=True, exist_ok=True)
    count = 0
    root_resolved = output_root.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = _zip_image_member(info.filename)
            if relative is None:
                continue
            destination = (output_root / Path(*relative.split("/"))).resolve()
            try:
                destination.relative_to(root_resolved)
            except ValueError as exc:
                raise RuntimeError(f"ZIP member không an toàn: {info.filename}") from exc
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            count += 1

    if count == 0 or not train_dir.is_dir() or not test_dir.is_dir():
        raise RuntimeError(
            "data.zip không có cấu trúc data/img_data/train và data/img_data/test "
            "với ảnh jpg/png hợp lệ."
        )
    print(f"Đã giải nén {count} ảnh từ {zip_path.name} vào {output_root}")
    return output_root


def find_code_root(input_root: Path, explicit: Optional[Path]) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser().resolve())
    candidates.extend([Path.cwd().resolve(), Path(__file__).resolve().parent])
    if input_root.is_dir():
        candidates.extend(path.parent for path in input_root.rglob("detector_2d.py"))
    for candidate in _unique_paths(candidates):
        if (candidate / "detector_2d.py").is_file() and (candidate / "train_week6.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Không tìm thấy code root có detector_2d.py và train_week6.py. "
        "Hãy truyền --code-root tới thư mục code đã upload lên Kaggle."
    )


def find_labels(input_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Labels không tồn tại: {path}")
        return path
    candidates = _unique_paths(input_root.rglob("dataset_labels (1).json"))
    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy dataset_labels (1).json dưới /kaggle/input. "
            "Truyền --labels /kaggle/input/.../dataset_labels (1).json."
        )
    candidates.sort(key=lambda path: ("fire-model-data" not in str(path).lower(), len(path.parts), str(path).lower()))
    return candidates[0]


def find_data_zip(input_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    """Find the uploaded ``data.zip`` when no extracted image tree exists."""
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"data.zip không tồn tại: {path}")
        return path
    candidates = _unique_paths(input_root.rglob("data.zip"))
    if not candidates:
        return None
    candidates.sort(
        key=lambda path: (
            "fire-detection-from-cctv" not in str(path).lower(),
            len(path.parts),
            str(path).lower(),
        )
    )
    return candidates[0]


def find_cctv_root(input_root: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Dataset root không tồn tại: {path}")
        return path
    candidates = [path for path in input_root.rglob("*") if path.is_dir() and _has_cctv_layout(path)]
    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy fire-detection-from-cctv dưới /kaggle/input. "
            "Truyền --dataset-root tới thư mục chứa data/img_data."
        )
    candidates.sort(key=lambda path: ("fire-detection-from-cctv" not in str(path).lower(), len(path.parts), str(path).lower()))
    return candidates[0]


def resolve_dataset_root(
    input_root: Path,
    explicit_root: Optional[Path],
    explicit_zip: Optional[Path],
    unpack_root: Path,
) -> tuple[Path, Optional[Path]]:
    """Resolve either an extracted dataset directory or an uploaded ZIP."""
    if explicit_root is not None:
        candidate = explicit_root.expanduser().resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return unpack_data_zip(candidate, unpack_root), candidate
        if not candidate.is_dir():
            raise FileNotFoundError(f"Dataset root không tồn tại: {candidate}")
        if not _has_cctv_layout(candidate):
            raise FileNotFoundError(
                f"Dataset root không có data/img_data: {candidate}. "
                "Nếu chỉ upload data.zip, dùng --data-zip."
            )
        return candidate, None

    data_zip = find_data_zip(input_root, explicit_zip)
    if data_zip is not None:
        return unpack_data_zip(data_zip, unpack_root), data_zip
    return find_cctv_root(input_root, None), None


def find_checkpoint(input_root: Path, explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint không tồn tại: {path}")
        return path
    candidates = _unique_paths(input_root.rglob("best_spatial.pth"))
    if not candidates:
        candidates = _unique_paths(input_root.rglob("best.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("week6_spatial" not in str(path).lower(), len(path.parts), str(path).lower()))
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--code-root", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--data-zip",
        type=Path,
        default=None,
        help="uploaded data.zip; only data/img_data is extracted to --unpack-root",
    )
    parser.add_argument(
        "--unpack-root",
        type=Path,
        default=Path("/kaggle/working/fire_detection_cctv_data"),
        help="writable directory used for ZIP extraction",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/detector_2d_v2"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--freeze-epochs", type=int, default=3)
    parser.add_argument("--image-size", "--imgsz", dest="image_size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--backbone", default="mobilenetv4_conv_medium")
    parser.add_argument("--fpn-channels", type=int, default=96)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--backbone-lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_root = args.input_root.expanduser().resolve()
    code_root = find_code_root(input_root, args.code_root)
    labels = find_labels(input_root, args.labels)
    dataset_root, data_zip = resolve_dataset_root(
        input_root,
        args.dataset_root,
        args.data_zip,
        args.unpack_root,
    )
    checkpoint = find_checkpoint(input_root, args.init_checkpoint)

    sys.path.insert(0, str(code_root))
    from detector_2d import main as detector_train_main

    forwarded = [
        str(code_root / "train_detector_2d.py"),
        "--labels", str(labels),
        "--dataset-root", str(dataset_root),
        "--output-dir", str(args.output_dir),
        "--backbone", args.backbone,
        "--image-size", str(args.image_size),
        "--fpn-channels", str(args.fpn_channels),
        "--epochs", str(args.epochs),
        "--freeze-epochs", str(args.freeze_epochs),
        "--batch-size", str(args.batch_size),
        "--grad-accumulation", str(args.grad_accumulation),
        "--workers", str(args.workers),
        "--lr", str(args.lr),
        "--backbone-lr", str(args.backbone_lr),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--threshold", str(args.threshold),
    ]
    if checkpoint is not None:
        forwarded.extend(["--init-checkpoint", str(checkpoint)])
    else:
        forwarded.append("--no-pretrained")
    if args.no_pretrained:
        forwarded.append("--no-pretrained")

    print(f"code_root={code_root}")
    print(f"labels={labels}")
    print(f"dataset_root={dataset_root}")
    print(f"data_zip={data_zip}")
    print(f"init_checkpoint={checkpoint}")
    print(f"output_dir={args.output_dir}")
    old_argv = sys.argv
    try:
        sys.argv = forwarded
        detector_train_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
