"""Generate printable calibration targets for the measured-room experiment.

The defaults match ``aruco_tagsgen_config.json`` and
``checkerboard_config.json``.  ArUco generation needs an OpenCV build with
``cv2.aruco`` (normally ``opencv-contrib-python``); checkerboard generation
only needs the Python standard library.

This utility writes only to ``--output-dir``.  It does not inspect, modify or
delete datasets, checkpoints or existing benchmark output.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


PAGE_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}


def _svg_open(width: float, height: float, title: str) -> list[str]:
    safe_title = html.escape(title)
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}mm" height="{height:g}mm" viewBox="0 0 {width:g} {height:g}">',
        f"<title>{safe_title}</title>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]


def _svg_close(lines: list[str]) -> str:
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def write_checkerboard(path: Path, inner_cols: int, inner_rows: int, square_mm: float, page: str) -> None:
    page_w, page_h = PAGE_MM[page]
    squares_x, squares_y = inner_cols + 1, inner_rows + 1
    board_w, board_h = squares_x * square_mm, squares_y * square_mm
    left, top = (page_w - board_w) / 2.0, (page_h - board_h) / 2.0
    lines = _svg_open(page_w, page_h, f"Checkerboard {inner_cols}x{inner_rows} inner corners")
    lines.append(f'<g id="checkerboard" transform="translate({left:g} {top:g})">')
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                x, y = col * square_mm, row * square_mm
                lines.append(f'<rect x="{x:g}" y="{y:g}" width="{square_mm:g}" height="{square_mm:g}" fill="black"/>')
    lines.append("</g>")
    lines.append(f'<text x="10" y="{page_h - 10:g}" font-family="Arial" font-size="4">{inner_cols}x{inner_rows} inner corners; square={square_mm:g} mm; print at 100%</text>')
    path.write_text(_svg_close(lines), encoding="utf-8")


def _aruco_cells(dictionary_name: str, marker_id: int) -> list[list[int]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("ArUco generation requires opencv-contrib-python") from exc
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build has no cv2.aruco; install opencv-contrib-python")
    dictionary_id = getattr(cv2.aruco, dictionary_name, None)
    if dictionary_id is None:
        raise ValueError(f"Unknown OpenCV ArUco dictionary: {dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    # An ArUco marker with a 6x6 payload and a one-cell black border is an
    # 8x8 logical grid.  Generate a tiny raster only to recover those logical
    # cells, then write the SVG as 64 vector rectangles instead of hundreds of
    # thousands of individual raster pixels.
    cells = int(getattr(dictionary, "markerSize", 6)) + 2
    pixels = cells * 32
    if hasattr(cv2.aruco, "generateImageMarker"):
        image = cv2.aruco.generateImageMarker(dictionary, marker_id, pixels, borderBits=1)
    else:
        image = cv2.aruco.drawMarker(dictionary, marker_id, pixels, borderBits=1)
    return [
        [int(image[min(pixels - 1, int((row + 0.5) * pixels / cells)), min(pixels - 1, int((col + 0.5) * pixels / cells))]) for col in range(cells)]
        for row in range(cells)
    ]


def write_aruco(path: Path, dictionary: str, marker_id: int, size_mm: float, margin_mm: float, page: str) -> None:
    page_w, page_h = PAGE_MM[page]
    if size_mm + 2 * margin_mm > min(page_w, page_h):
        raise ValueError("ArUco tag plus margins does not fit on the selected page")
    left, top = (page_w - size_mm) / 2.0, (page_h - size_mm) / 2.0
    bitmap = _aruco_cells(dictionary, marker_id)
    lines = _svg_open(page_w, page_h, f"{dictionary} marker {marker_id}")
    cell = size_mm / len(bitmap)
    lines.append(f'<g id="aruco-{marker_id}" transform="translate({left:g} {top:g})">')
    lines.append(f'<rect width="{size_mm:g}" height="{size_mm:g}" fill="white"/>')
    for row, values in enumerate(bitmap):
        for col, value in enumerate(values):
            if int(value) < 128:
                lines.append(f'<rect x="{col * cell:g}" y="{row * cell:g}" width="{cell:g}" height="{cell:g}" fill="black"/>')
    lines.append("</g>")
    lines.append(f'<text x="10" y="{page_h - 10:g}" font-family="Arial" font-size="4">{dictionary} ID={marker_id}; black square={size_mm:g} mm; print at 100%</text>')
    path.write_text(_svg_close(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("generated"))
    parser.add_argument("--dictionary", default="DICT_6X6_1000")
    parser.add_argument("--marker-ids", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--tag-size-mm", type=float, default=120.0)
    parser.add_argument("--margin-mm", type=float, default=20.0)
    parser.add_argument("--page", choices=sorted(PAGE_MM), default="A4")
    parser.add_argument("--inner-corners", nargs=2, type=int, default=[9, 6], metavar=("COLS", "ROWS"))
    parser.add_argument("--square-size-mm", type=float, default=25.0)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_checkerboard(output / "checkerboard_9x6_inner_25mm.svg", args.inner_corners[0], args.inner_corners[1], args.square_size_mm, args.page)
    for marker_id in args.marker_ids:
        write_aruco(output / f"aruco_{args.dictionary.lower()}_id_{marker_id:02d}.svg", args.dictionary, marker_id, args.tag_size_mm, args.margin_mm, args.page)
    manifest = {
        "dictionary": args.dictionary,
        "marker_ids": args.marker_ids,
        "tag_size_mm": args.tag_size_mm,
        "margin_mm": args.margin_mm,
        "page": args.page,
        "checkerboard_inner_corners": args.inner_corners,
        "checkerboard_square_size_mm": args.square_size_mm,
        "print_scale": "100_percent_actual_size",
    }
    (output / "calibration_targets_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved={output}")


if __name__ == "__main__":
    main()

