"""Export static and interactive 3D views from ``paper_workflow_3d`` output.

The script reads ``summary.json`` and reconstructs the room scene from the
mesh recorded in the summary. It exports:

* perspective, top and side PNG views;
* an interactive Plotly HTML scene that can be rotated and zoomed;
* a compact JSON manifest of the plotted points;
* optionally, an ASCII PLY containing the mesh and colour-coded points.

For a synthetic summary, camera calibration is recovered from the matching
synthetic manifest row. For a measured-room summary, pass ``--calibration``
with the measured camera JSON. The visualizer never changes metric values;
it only renders the coordinates already present in the evaluator output.

Example::

    .venv\\Scripts\\python.exe visualize_3d_results.py ^
      --summary output\\paper_workflow_synthetic\\summary.json ^
      --output-dir output\\paper_workflow_synthetic\\visualization

The interactive result is ``scene_3d_interactive.html``. Open it in a browser
or drag it into a presentation. Plotly is optional for the Python package,
but is recommended for a fully self-contained HTML export::

    .venv\\Scripts\\python.exe -m pip install matplotlib plotly
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from camera_calibration import CameraCalibration


BRANCHES = ("coarse", "roi_raw", "roi_blend", "ipm")
COLORS = {
    "mesh": "#8c8c8c",
    "camera": "#7b2cbf",
    "gt": "#1f9d55",
    "coarse": "#2878d0",
    "roi_raw": "#f28e2b",
    "roi_blend": "#d62728",
    "ipm": "#17a2b8",
}
PLY_COLORS = {
    "camera": (123, 44, 191),
    "gt": (31, 157, 85),
    "coarse": (40, 120, 208),
    "roi_raw": (242, 142, 43),
    "roi_blend": (214, 39, 40),
    "ipm": (23, 162, 184),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)!r}")


def _point(value: Any, dimensions: int = 3) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if len(array) < dimensions or not np.all(np.isfinite(array[:dimensions])):
        return None
    return array[:dimensions].copy()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(value: Any, *bases: Path) -> Optional[Path]:
    if value is None:
        return None
    candidate = Path(str(value)).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    for base in bases:
        resolved = (base / candidate).resolve()
        if resolved.is_file():
            return resolved
    return candidate.resolve()


def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = _load_json(path)
    vertices = np.asarray(data["vertices"], dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(data["faces"], dtype=np.int64).reshape(-1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Mesh is empty: {path}")
    if faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"Mesh face index out of range: {path}")
    return vertices, faces


def _manifest_aliases(value: str) -> list[str]:
    text = str(value).replace("\\", "/").lower()
    aliases = [text, Path(text).name]
    parts = text.split("/")
    for marker in ("images/", "img_data/"):
        if marker in text:
            aliases.append(text.split(marker, 1)[1])
    if "images/" in text:
        aliases.append("images/" + text.split("images/", 1)[1])
    return list(dict.fromkeys(aliases))


def _load_manifest_index(dataset: Optional[Path]) -> dict[str, dict[str, Any]]:
    if dataset is None:
        return {}
    manifest = dataset / "manifest.jsonl"
    if not manifest.is_file():
        return {}
    index: dict[str, dict[str, Any]] = {}
    with manifest.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            keys = []
            for value in (
                row.get("sample_id"),
                row.get("image_path"),
                row.get("_image_path"),
            ):
                if value is not None:
                    keys.extend(_manifest_aliases(str(value)))
            for key in keys:
                index.setdefault(key, row)
    return index


def _row_manifest(row: dict[str, Any], index: dict[str, dict[str, Any]]) -> Optional[dict[str, Any]]:
    for value in (row.get("sample_id"), row.get("image_path"), Path(str(row.get("image_path", ""))).name):
        if value is None:
            continue
        for alias in _manifest_aliases(str(value)):
            if alias in index:
                return index[alias]
    return None


def _calibration_from_camera(camera: dict[str, Any], image_size: Any = None) -> CameraCalibration:
    return CameraCalibration.from_dict(
        {
            "image_size": image_size,
            "intrinsics": {
                "K": camera.get("K"),
                "dist_coeffs": camera.get("dist_coeffs", []),
            },
            "extrinsics": {
                "R": camera.get("R_world_to_camera", camera.get("R")),
                "camera_position": camera.get("camera_position"),
            },
        }
    )


def _external_calibration(path: Path) -> CameraCalibration:
    data = _load_json(path)
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    status = str(metadata.get("status", data.get("status", "unknown"))).lower()
    if any(token in status for token in ("provisional", "synthetic", "template")):
        raise ValueError(f"Refusing non-measured calibration in external mode: {path}")
    return CameraCalibration.from_dict(data)


def _calibration_for_record(
    row: dict[str, Any],
    manifest_row: Optional[dict[str, Any]],
    source: str,
    external: Optional[CameraCalibration],
) -> Optional[CameraCalibration]:
    if source == "external":
        return external
    source_row = manifest_row or row
    camera_key = "camera_estimated" if source == "estimated" else "camera"
    camera = source_row.get(camera_key)
    if not isinstance(camera, dict):
        return None
    return _calibration_from_camera(camera, source_row.get("image_size", row.get("image_size")))


def _mesh_path(summary_path: Path, summary: dict[str, Any], dataset: Optional[Path], override: Optional[Path]) -> Path:
    if override is not None:
        return override.resolve()
    method = summary.get("method", {})
    value = method.get("mesh")
    resolved = _resolve_path(value, summary_path.parent, dataset or summary_path.parent)
    if resolved is not None and resolved.is_file():
        return resolved
    if dataset is not None and (dataset / "room_mesh.json").is_file():
        return (dataset / "room_mesh.json").resolve()
    raise FileNotFoundError("Could not resolve the room mesh from summary.json")


def _select_rows(rows: list[dict[str, Any]], maximum: int, selection: str) -> list[dict[str, Any]]:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    if selection == "even":
        indices = np.linspace(0, len(rows) - 1, int(maximum), dtype=int)
        return [rows[int(index)] for index in indices]
    if selection == "diverse":
        points = np.asarray([_point(row.get("gt_xyz")) for row in rows], dtype=np.float64)
        chosen = [0]
        distances = np.full(len(rows), np.inf, dtype=np.float64)
        for _ in range(1, int(maximum)):
            distances = np.minimum(
                distances,
                np.linalg.norm(points - points[chosen[-1]], axis=1),
            )
            distances[chosen] = -1.0
            chosen.append(int(np.argmax(distances)))
        return [rows[index] for index in sorted(chosen)]
    raise ValueError(f"Unknown selection policy: {selection}")


def _extract_scene(
    summary_path: Path,
    dataset: Optional[Path],
    calibration_path: Optional[Path],
    calibration_source: str,
    max_records: int,
    selection: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    summary = _load_json(summary_path)
    rows = list(summary.get("records", []))
    if not rows:
        raise ValueError(f"No records in {summary_path}")
    rows = _select_rows(rows, max_records, selection)
    mesh_path = _mesh_path(summary_path, summary, dataset, None)
    vertices, faces = _load_mesh(mesh_path)
    manifest_index = _load_manifest_index(dataset)
    external = _external_calibration(calibration_path) if calibration_source == "external" else None
    scene_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for row in rows:
        manifest_row = _row_manifest(row, manifest_index)
        calibration = _calibration_for_record(row, manifest_row, calibration_source, external)
        record = {
            "sample_id": row.get("sample_id", Path(str(row.get("image_path", ""))).stem),
            "scene_id": row.get("scene_id", row.get("sequence_id", "unknown")),
            "frame_index": row.get("frame_index", 0),
            "gt": _point(row.get("gt_xyz")),
            "camera": None if calibration is None else calibration.camera_position(),
            "branches": {},
        }
        if calibration is None:
            warnings.append(f"No calibration for {record['sample_id']}; rays will be omitted")
        for branch in BRANCHES:
            branch_data = (row.get("branches", {}) or {}).get(branch, {}) or {}
            location = branch_data.get("location", {}) or {}
            record["branches"][branch] = {
                "point": _point(location.get("point")),
                "pixel": _point(branch_data.get("pixel"), dimensions=2),
                "confidence": location.get("confidence"),
                "status": location.get("status"),
            }
        scene_rows.append(record)
    metadata = {
        "summary": str(summary_path),
        "mesh": str(mesh_path),
        "calibration_source": calibration_source,
        "records": len(scene_rows),
        "warnings": sorted(set(warnings)),
        "synthetic_metric_geometry": bool(summary.get("dataset", {}).get("synthetic_metric_geometry", False)),
    }
    return summary, vertices, faces, scene_rows, metadata


def _all_points(scene_rows: list[dict[str, Any]]) -> np.ndarray:
    points: list[np.ndarray] = []
    for row in scene_rows:
        for value in [row.get("gt"), row.get("camera")]:
            if value is not None:
                points.append(value)
        for branch in BRANCHES:
            value = (row.get("branches", {}).get(branch, {}) or {}).get("point")
            if value is not None:
                points.append(value)
    return np.asarray(points, dtype=np.float64).reshape(-1, 3) if points else np.empty((0, 3))


def _limits(vertices: np.ndarray, scene_rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    cloud = np.vstack([vertices, _all_points(scene_rows)]) if len(_all_points(scene_rows)) else vertices
    low = cloud.min(axis=0)
    high = cloud.max(axis=0)
    span = np.maximum(high - low, 1.0)
    margin = 0.08 * span
    return low - margin, high + margin


def _plot_static(
    vertices: np.ndarray,
    faces: np.ndarray,
    scene_rows: list[dict[str, Any]],
    output: Path,
    title: str,
    elev: float,
    azim: float,
    max_rays: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError as exc:
        raise RuntimeError("Static PNG export requires matplotlib") from exc

    fig = plt.figure(figsize=(11, 8), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    triangles = [vertices[face] for face in faces]
    mesh = Poly3DCollection(triangles, alpha=0.20, facecolor=COLORS["mesh"], edgecolor="#666666", linewidth=0.35)
    ax.add_collection3d(mesh)
    branch_points: dict[str, list[np.ndarray]] = {branch: [] for branch in BRANCHES}
    gt_points: list[np.ndarray] = []
    camera_points: list[np.ndarray] = []
    for row in scene_rows:
        if row.get("gt") is not None:
            gt_points.append(row["gt"])
        if row.get("camera") is not None:
            camera_points.append(row["camera"])
        for branch in BRANCHES:
            value = (row.get("branches", {}).get(branch, {}) or {}).get("point")
            if value is not None:
                branch_points[branch].append(value)

    if camera_points:
        camera_array = np.asarray(camera_points)
        ax.scatter(camera_array[:, 0], camera_array[:, 1], camera_array[:, 2], color=COLORS["camera"], marker="^", s=60, depthshade=False)
    if gt_points:
        array = np.asarray(gt_points)
        ax.scatter(array[:, 0], array[:, 1], array[:, 2], color=COLORS["gt"], marker="o", s=38, depthshade=False)
    for branch in BRANCHES:
        if not branch_points[branch]:
            continue
        array = np.asarray(branch_points[branch])
        ax.scatter(array[:, 0], array[:, 1], array[:, 2], color=COLORS[branch], marker="x", s=42, depthshade=False)

    ray_count = 0
    for row in scene_rows:
        camera = row.get("camera")
        if camera is None:
            continue
        for branch in BRANCHES:
            point = (row.get("branches", {}).get(branch, {}) or {}).get("point")
            if point is None or ray_count >= max_rays:
                continue
            ax.plot(
                [camera[0], point[0]],
                [camera[1], point[1]],
                [camera[2], point[2]],
                color=COLORS[branch],
                alpha=0.22,
                linewidth=0.8,
            )
            ray_count += 1

    low, high = _limits(vertices, scene_rows)
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])
    ax.set_zlim(low[2], high[2])
    ax.set_box_aspect(np.maximum(high - low, 1e-6))
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["gt"], label="Ground truth 3D", markersize=8),
        Line2D([0], [0], marker="x", color=COLORS["coarse"], label="Coarse", markersize=8),
        Line2D([0], [0], marker="x", color=COLORS["roi_raw"], label="ROI", markersize=8),
        Line2D([0], [0], marker="x", color=COLORS["roi_blend"], label="Weighted blend", markersize=8),
        Line2D([0], [0], marker="x", color=COLORS["ipm"], label="IPM floor", markersize=8),
        Line2D([0], [0], marker="^", color=COLORS["camera"], label="Camera", markersize=8),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _plotly_figure(vertices: np.ndarray, faces: np.ndarray, scene_rows: list[dict[str, Any]], title: str, max_rays: int):
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError("Interactive HTML export requires plotly") from exc

    figure = go.Figure()
    figure.add_trace(
        go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            color=COLORS["mesh"], opacity=0.26, name="Room mesh",
            hoverinfo="skip", flatshading=True,
        )
    )
    # Scatter3d supports a smaller symbol set than 2D Scatter.  Keep the
    # symbols in Plotly's portable 3D set; triangle-up is not supported
    # consistently across installed Plotly releases.
    for branch, label, marker in (("gt", "Ground truth 3D", "circle"), ("coarse", "Coarse", "x"), ("roi_raw", "ROI", "diamond"), ("roi_blend", "Weighted blend", "square"), ("ipm", "IPM floor", "circle-open"), ("camera", "Camera", "diamond-open")):
        points = []
        ids = []
        for row in scene_rows:
            value = row.get(branch) if branch in {"gt", "camera"} else (row.get("branches", {}).get(branch, {}) or {}).get("point")
            if value is not None:
                points.append(value)
                ids.append(str(row["sample_id"]))
        if not points:
            continue
        array = np.asarray(points)
        figure.add_trace(
            go.Scatter3d(
                x=array[:, 0], y=array[:, 1], z=array[:, 2],
                mode="markers", name=label,
                marker={"size": 5 if branch != "camera" else 6, "symbol": marker, "color": COLORS[branch]},
                text=ids, hovertemplate="%{text}<br>X=%{x:.3f} m<br>Y=%{y:.3f} m<br>Z=%{z:.3f} m<extra>" + label + "</extra>",
            )
        )

    ray_x: dict[str, list[Optional[float]]] = {branch: [] for branch in BRANCHES}
    ray_y: dict[str, list[Optional[float]]] = {branch: [] for branch in BRANCHES}
    ray_z: dict[str, list[Optional[float]]] = {branch: [] for branch in BRANCHES}
    count = 0
    for row in scene_rows:
        camera = row.get("camera")
        if camera is None:
            continue
        for branch in BRANCHES:
            point = (row.get("branches", {}).get(branch, {}) or {}).get("point")
            if point is None or count >= max_rays:
                continue
            ray_x[branch].extend([float(camera[0]), float(point[0]), None])
            ray_y[branch].extend([float(camera[1]), float(point[1]), None])
            ray_z[branch].extend([float(camera[2]), float(point[2]), None])
            count += 1
    for branch in BRANCHES:
        if not ray_x[branch]:
            continue
        figure.add_trace(
            go.Scatter3d(
                x=ray_x[branch], y=ray_y[branch], z=ray_z[branch],
                mode="lines", name=f"{branch} rays", legendgroup=branch,
                line={"color": COLORS[branch], "width": 2}, opacity=0.30,
                hoverinfo="skip",
            )
        )
    figure.update_layout(
        title=title,
        template="plotly_white",
        scene={
            "xaxis_title": "X (m)", "yaxis_title": "Y (m)", "zaxis_title": "Z (m)",
            "aspectmode": "data",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 0, "r": 0, "b": 0, "t": 60},
    )
    return figure


def _write_ply(vertices: np.ndarray, faces: np.ndarray, scene_rows: list[dict[str, Any]], output: Path) -> None:
    points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for row in scene_rows:
        for branch in ("camera", "gt", *BRANCHES):
            value = row.get(branch) if branch in {"camera", "gt"} else (row.get("branches", {}).get(branch, {}) or {}).get("point")
            if value is not None:
                points.append((np.asarray(value, dtype=np.float64), PLY_COLORS[branch]))
    mesh_color = (150, 150, 150)
    total_vertices = len(vertices) + len(points)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {total_vertices}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header",
    ]
    for vertex in vertices:
        lines.append(f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} {mesh_color[0]} {mesh_color[1]} {mesh_color[2]}")
    for point, color in points:
        lines.append(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {color[0]} {color[1]} {color[2]}")
    lines.extend(f"3 {int(face[0])} {int(face[1])} {int(face[2])}" for face in faces)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.summary.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve() if args.dataset else None
    mesh_path = args.mesh.expanduser().resolve() if args.mesh else None
    summary, vertices, faces, scene_rows, metadata = _extract_scene(
        summary_path,
        dataset,
        args.calibration.expanduser().resolve() if args.calibration else None,
        args.calibration_source,
        args.max_records,
        args.selection,
    )
    if mesh_path is not None:
        vertices, faces = _load_mesh(mesh_path)
        metadata["mesh"] = str(mesh_path)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    warning_title = "Synthetic metric geometry" if metadata["synthetic_metric_geometry"] else "Measured room geometry"
    title = f"Fire 2D-to-3D localisation - {warning_title}"
    _plot_static(vertices, faces, scene_rows, output_dir / "scene_3d_overview.png", title, 28, -58, args.max_rays)
    _plot_static(vertices, faces, scene_rows, output_dir / "scene_3d_top_view.png", title + " - top view", 88, -90, args.max_rays)
    _plot_static(vertices, faces, scene_rows, output_dir / "scene_3d_side_view.png", title + " - side view", 5, -90, args.max_rays)
    interactive_path = output_dir / "scene_3d_interactive.html"
    try:
        figure = _plotly_figure(vertices, faces, scene_rows, title, args.max_rays)
        figure.write_html(interactive_path, include_plotlyjs=True, full_html=True, auto_open=False)
        interactive_mode = "plotly_embedded"
    except RuntimeError as exc:
        # Preserve a useful artifact even when plotly is not installed. The
        # browser can load Plotly from the approved CDN in this fallback.
        payload = {"vertices": vertices, "faces": faces, "records": scene_rows, "colors": COLORS}
        interactive_path.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><title>3D fire localisation</title>"
            "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head><body>"
            "<div id='scene' style='width:100%;height:95vh'></div><script>"
            "const data=" + json.dumps(payload, default=_json_default) + ";"
            "const traces=[{type:'mesh3d',x:data.vertices.map(v=>v[0]),y:data.vertices.map(v=>v[1]),z:data.vertices.map(v=>v[2]),"
            "i:data.faces.map(f=>f[0]),j:data.faces.map(f=>f[1]),k:data.faces.map(f=>f[2]),opacity:.25,color:'#8c8c8c',name:'Room mesh'}];"
            "const branches={gt:'Ground truth 3D',coarse:'Coarse',roi_raw:'ROI',roi_blend:'Weighted blend',ipm:'IPM floor',camera:'Camera'};"
            "for(const [key,label] of Object.entries(branches)){const p=[];for(const r of data.records){const v=key==='gt'||key==='camera'?r[key]:r.branches[key].point;if(v)p.push(v);}if(p.length)traces.push({type:'scatter3d',mode:'markers',x:p.map(v=>v[0]),y:p.map(v=>v[1]),z:p.map(v=>v[2]),name:label,marker:{size:5}});}"
            "Plotly.newPlot('scene',traces,{title:'" + title.replace("'", "\\'") + "',scene:{aspectmode:'data',xaxis:{title:'X (m)'},yaxis:{title:'Y (m)'},zaxis:{title:'Z (m)'}}});"
            "</script></body></html>",
            encoding="utf-8",
        )
        interactive_mode = "plotly_cdn_fallback"
        metadata["interactive_warning"] = str(exc)
    if args.write_ply:
        _write_ply(vertices, faces, scene_rows, output_dir / "scene_3d_annotations.ply")
    manifest = {
        **metadata,
        "title": title,
        "outputs": {
            "perspective_png": str(output_dir / "scene_3d_overview.png"),
            "top_png": str(output_dir / "scene_3d_top_view.png"),
            "side_png": str(output_dir / "scene_3d_side_view.png"),
            "interactive_html": str(interactive_path),
            "ply": str(output_dir / "scene_3d_annotations.ply") if args.write_ply else None,
        },
        "summary_metrics": summary.get("branches", {}),
    }
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=root / "output" / "paper_workflow_synthetic" / "summary.json")
    parser.add_argument("--dataset", type=Path, default=None, help="Dataset root used to recover per-frame camera calibration")
    parser.add_argument("--mesh", type=Path, default=None, help="Optional mesh override")
    parser.add_argument("--calibration", type=Path, default=None, help="Measured camera JSON for external calibration")
    parser.add_argument("--calibration-source", choices=("true", "estimated", "external"), default="true")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "paper_workflow_synthetic" / "visualization")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--selection", choices=("even", "diverse"), default="even")
    parser.add_argument("--max-rays", type=int, default=120)
    parser.add_argument("--write-ply", action="store_true", help="Also write a coloured ASCII PLY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_records < 0 or args.max_rays < 0:
        raise ValueError("--max-records and --max-rays must be >= 0")
    manifest = run(args)
    print(json.dumps({"outputs": manifest["outputs"], "warnings": manifest.get("warnings", [])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
