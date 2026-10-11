from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument('--zip', type=Path, default=root / 'datasets' / 'FLAME' / 'Training.zip')
    parser.add_argument('--output', type=Path, default=root / 'datasets' / 'FLAME')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive_copy = args.output / 'Training.zip'
    # A completed extraction is a valid source after the archive is removed.
    extracted_root = args.output / 'Training'
    if not args.zip.is_file() and extracted_root.is_dir():
        members = [p for p in extracted_root.rglob('*') if p.is_file()]
        summary = {'format': 'LAB_SAM.FLAME.v1', 'members': len(members),
                   'classes': {name: len(list((extracted_root / name).glob('*')))
                               for name in ('Fire', 'No_Fire')}}
        (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(json.dumps(summary, indent=2))
        return
    if args.zip.resolve() != archive_copy.resolve() and args.zip.is_file() and not archive_copy.exists():
        archive_copy.write_bytes(args.zip.read_bytes())
    source = archive_copy if archive_copy.is_file() else args.zip
    with zipfile.ZipFile(source) as archive:
        members = [name for name in archive.namelist() if name.startswith('Training/') and not name.endswith('/')]
        for index, name in enumerate(members, 1):
            target = args.output / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size == 0:
                with archive.open(name) as src, target.open('wb') as dst:
                    dst.write(src.read())
            if index % 2000 == 0:
                print(f'extracted={index}/{len(members)}', flush=True)
    summary = {'format': 'LAB_SAM.FLAME.v1', 'members': len(members),
               'classes': {name: len(list((args.output / 'Training' / name).glob('*')))
                           for name in ('Fire', 'No_Fire')}}
    (args.output / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
