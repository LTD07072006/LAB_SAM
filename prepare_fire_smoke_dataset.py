"""Prepare FIRE-SMOKE-DATASET.zip as an isolated binary detector add-on.

This dataset has three folders: ``Fire``, ``Smoke`` and ``Neutral``.  The
project's binary classifier convention is:

    Fire -> class 1
    Smoke/Neutral -> class 0

The script extracts only to the new ``datasets/fire_smoke_dataset`` folder and
writes a manifest under ``working/fire_smoke_binary``.  It does not modify
``archive.zip`` or any existing dataset/output/checkpoint.  The generated
records intentionally contain no ``p_fire`` or 3D coordinates: this add-on is
for classification/domain pretraining, not ROI or physical 3D evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

from project_paths import FIRE_SMOKE_ZIP


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _safe_member_path(root: Path, member: str) -> Path:
    target = (root / member).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"Unsafe ZIP member path: {member}")
    return target


def _class_from_parts(parts: tuple[str, ...]) -> tuple[str, int] | None:
    lowered = {part.lower() for part in parts}
    if "fire" in lowered:
        return "fire", 1
    if "smoke" in lowered:
        return "smoke", 0
    if "neutral" in lowered:
        return "neutral", 0
    return None


def prepare(zip_path: Path, extract_root: Path, output_root: Path) -> dict[str, Any]:
    zip_path = zip_path.expanduser().resolve()
    extract_root = extract_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(f"Dataset ZIP not found: {zip_path}")
    if zip_path.suffix.lower() != ".zip":
        raise ValueError(f"Expected a completed .zip file, got: {zip_path}")

    records: list[dict[str, Any]] = []
    counts = {"fire": 0, "smoke": 0, "neutral": 0}
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            path = Path(member.filename)
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parsed = _class_from_parts(tuple(path.parts))
            if parsed is None:
                continue
            source_class, class_id = parsed
            target = _safe_member_path(extract_root, member.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size != member.file_size:
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            counts[source_class] += 1
            relative = target.relative_to(extract_root).as_posix()
            split = "train" if "/train/" in f"/{relative.lower()}" else "test"
            records.append({
                "sample_id": f"fire_smoke_{len(records):07d}",
                "image_path": str(target),
                "relative_path": relative,
                "split": split,
                "source_class": source_class,
                "class_id": class_id,
                "has_fire": class_id,
                "p_fire": None,
                "fire_xyz_world": None,
                "annotation_source": "FIRE-SMOKE-DATASET folder label",
            })

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "dataset": "FIRE-SMOKE-DATASET",
        "source_zip": str(zip_path),
        "extracted_root": str(extract_root),
        "manifest": str(manifest),
        "records": len(records),
        "class_mapping": {"Fire": 1, "Smoke": 0, "Neutral": 0},
        "source_counts": counts,
        "p_fire_records": 0,
        "xyz_records": 0,
        "warning": "Classification/domain pretraining only; no ROI or physical 3D labels.",
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=FIRE_SMOKE_ZIP,
    )
    parser.add_argument("--extract-root", type=Path, default=root / "datasets" / "fire_smoke_dataset")
    parser.add_argument("--output", type=Path, default=root / "working" / "fire_smoke_binary")
    args = parser.parse_args()
    print(json.dumps(prepare(args.zip, args.extract_root, args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
