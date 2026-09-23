"""Build a complete manifest for images stored in fire-samples."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "fire-samples"
OUT = SAMPLES / "test_manifest.json"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(name: str):
    if name.startswith("dataset_test_img_"):
        return "fire", "local_test_fire"
    if name.startswith("dataset_test_smoke_"):
        return "smoke", "local_test_smoke"
    if name.startswith("dataset_test_default_"):
        return "default", "local_test_default"
    if name.startswith("external_fire_"):
        return "fire", "external_fire"
    if name.startswith("external_smoke_"):
        return "smoke", "external_smoke"
    if name.startswith("fire"):
        return "fire", "manual_fire_samples"
    return "unknown", "unclassified"


def main():
    rows = []
    seen = {}
    for path in sorted(SAMPLES.iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTS:
            continue
        file_hash = sha256(path)
        duplicate_of = seen.get(file_hash)
        if duplicate_of is None:
            seen[file_hash] = path.name
        label, source = classify(path.name)
        rows.append({
            "file": path.name,
            "label": label,
            "source": source,
            "sha256": file_hash,
            "duplicate_of": duplicate_of,
        })
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"images={len(rows)}")
    print(f"unique_hashes={len(seen)}")
    for label in ("fire", "smoke", "default", "unknown"):
        print(f"{label}={sum(row['label'] == label for row in rows)}")


if __name__ == "__main__":
    main()
