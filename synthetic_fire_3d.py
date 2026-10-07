"""Generate a metric synthetic fire-localisation dataset.

This is a dependency-light generator intended for the LAB_SAM geometry
workflow, not a replacement for a photorealistic renderer.  It deliberately
keeps the geometric contract exact:

    world fire contact point -> calibrated camera projection -> image point
    image point + camera pose + room mesh -> metric 3D ray-casting target

Each record contains both clean ground truth and a noisy observation.  The
noisy observation is useful for testing detector/ROI error propagation without
pretending that a pseudo-label is a real annotation.

The generator follows ideas used by the supplied papers:

* store complete 2D/3D/camera correspondence rather than only an image;
* split by sequence/scene, not by adjacent frames;
* inject image, point and confidence noise explicitly;
* keep calibration and metric geometry in the same coordinate frame.

Example:

    .venv\\Scripts\\python.exe synthetic_fire_3d.py \\
        --output-dir working\\synthetic_fire_3d_v3 \\
        --scenes 240 --frames-per-scene 4 --image-size 640 640 \\
        --mesh-profile obstacles --seed 17 --preview-count 12

The output can be evaluated with ``evaluate_synthetic_fire_3d.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from locator import TriangleMesh


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_ROOM = (8.0, 14.0, 4.0)  # width, depth, height in metres


def default_obstacles(room_width: float, room_depth: float) -> list[dict[str, Any]]:
    """Return deterministic low-poly obstacles in the room coordinate frame.

    The camera enters from negative Y, so these boxes create realistic
    foreground/background occlusion without closing the front of the room.
    They are deliberately simple cuboids so the same JSON mesh can be loaded
    by the dependency-free ray caster and by Blender later.
    """
    return [
        {
            "name": "central_cabinet",
            "min": [-1.15, 4.15, 0.0],
            "max": [1.15, 5.05, 1.35],
            "surface_type": "furniture",
        },
        {
            "name": "left_table",
            "min": [-3.10, 7.0, 0.0],
            "max": [-1.35, 8.15, 0.95],
            "surface_type": "furniture",
        },
        {
            "name": "right_column",
            "min": [2.05, 10.0, 0.0],
            "max": [2.85, 11.15, 2.15],
            "surface_type": "obstacle",
        },
    ]


def cuboid_triangles(box: dict[str, Any]) -> tuple[list[list[float]], list[list[int]], list[str]]:
    """Build a closed cuboid and per-face labels for a JSON triangle mesh."""
    minimum = np.asarray(box["min"], dtype=np.float64)
    maximum = np.asarray(box["max"], dtype=np.float64)
    x0, y0, z0 = minimum.tolist()
    x1, y1, z1 = maximum.tolist()
    vertices = [
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ]
    # Counter-clockwise winding is useful to downstream renderers, while the
    # local ray caster intentionally remains double-sided.
    faces = [
        [0, 2, 1], [0, 3, 2],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7],  # left
    ]
    label = str(box.get("surface_type", "obstacle"))
    return vertices, faces, [label] * len(faces)


def _append_quad(
    vertices: list[list[float]],
    faces: list[list[int]],
    labels: list[str],
    corners: Sequence[Sequence[float]],
    label: str,
) -> None:
    """Append a planar quad as two triangles to the mesh buffers."""
    start = len(vertices)
    vertices.extend([list(map(float, corner)) for corner in corners])
    faces.extend([[start, start + 1, start + 2], [start, start + 2, start + 3]])
    labels.extend([label, label])


def build_room_mesh(
    room_width: float,
    room_depth: float,
    room_height: float,
    profile: str,
) -> dict[str, Any]:
    """Build a small open-room mesh with optional walls and furniture.

    ``floor`` is the compatibility profile used by v1/v2. ``obstacles``
    adds furniture but leaves the sides open so bad rays can genuinely miss.
    ``room`` adds walls/ceiling, and ``room_obstacles`` combines both. The
    mesh is intentionally represented as JSON triangles so the same artifact
    can be loaded by this project, Blender, or an external renderer.
    """
    allowed = {"floor", "obstacles", "room", "room_obstacles"}
    profile = str(profile).lower().strip()
    if profile not in allowed:
        raise ValueError(f"mesh profile must be one of {sorted(allowed)}, got {profile!r}")

    half_width = float(room_width) * 0.5
    depth = float(room_depth)
    height = float(room_height)
    vertices: list[list[float]] = [
        [-half_width, 0.0, 0.0],
        [half_width, 0.0, 0.0],
        [half_width, depth, 0.0],
        [-half_width, depth, 0.0],
    ]
    faces: list[list[int]] = [[0, 1, 2], [0, 2, 3]]
    labels: list[str] = ["floor", "floor"]

    if profile in {"room", "room_obstacles"}:
        _append_quad(
            vertices, faces, labels,
            [[-half_width, depth, 0.0], [half_width, depth, 0.0],
             [half_width, depth, height], [-half_width, depth, height]],
            "back_wall",
        )
        _append_quad(
            vertices, faces, labels,
            [[-half_width, 0.0, 0.0], [-half_width, depth, 0.0],
             [-half_width, depth, height], [-half_width, 0.0, height]],
            "left_wall",
        )
        _append_quad(
            vertices, faces, labels,
            [[half_width, depth, 0.0], [half_width, 0.0, 0.0],
             [half_width, 0.0, height], [half_width, depth, height]],
            "right_wall",
        )
        _append_quad(
            vertices, faces, labels,
            [[-half_width, 0.0, height], [-half_width, depth, height],
             [half_width, depth, height], [half_width, 0.0, height]],
            "ceiling",
        )

    obstacles = default_obstacles(room_width, room_depth) if profile in {"obstacles", "room_obstacles"} else []
    obstacle_ranges: list[dict[str, Any]] = []
    for obstacle in obstacles:
        obstacle_vertices, obstacle_faces, obstacle_labels = cuboid_triangles(obstacle)
        start = len(vertices)
        vertices.extend(obstacle_vertices)
        faces.extend([[start + index for index in face] for face in obstacle_faces])
        labels.extend(obstacle_labels)
        obstacle_ranges.append(
            {
                "name": obstacle["name"],
                "surface_type": obstacle.get("surface_type", "obstacle"),
                "min": list(obstacle["min"]),
                "max": list(obstacle["max"]),
                "face_start": len(faces) - len(obstacle_faces),
                "face_count": len(obstacle_faces),
            }
        )

    return {
        "metadata": {
            "units": "metres",
            "coordinate_system": "world: X lateral, Y forward, Z up",
            "mesh_profile": profile,
            "room_size_m": [float(room_width), float(room_depth), float(room_height)],
            "open_front": True,
            "surface_labels": labels,
        },
        "vertices": vertices,
        "faces": faces,
        "face_labels": labels,
        "obstacles": obstacle_ranges,
    }


def point_inside_obstacle(point: Sequence[float], obstacle: dict[str, Any], margin: float = 0.0) -> bool:
    values = np.asarray(point, dtype=np.float64)
    lower = np.asarray(obstacle["min"], dtype=np.float64) - float(margin)
    upper = np.asarray(obstacle["max"], dtype=np.float64) + float(margin)
    return bool(np.all(values >= lower) and np.all(values <= upper))


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(json_safe(row), ensure_ascii=False, allow_nan=False))
            stream.write("\n")
    temporary.replace(path)


def normalise_size(value: Sequence[int]) -> tuple[int, int]:
    values = tuple(int(item) for item in value)
    if len(values) != 2 or min(values) <= 0:
        raise ValueError(f"image size must be two positive integers: {value!r}")
    return values


def look_at_rotation(
    camera_position: np.ndarray,
    target: np.ndarray,
    up: np.ndarray | None = None,
) -> np.ndarray:
    """Return R that maps world coordinates to camera coordinates.

    Camera coordinates use x=right, y=down, z=forward, matching the pinhole
    convention used by ``locator.CameraGeometry``.
    """
    camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up is None else np.asarray(up, dtype=np.float64).reshape(3)
    forward = target - camera_position
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-12)
    return np.vstack([right, down, forward])


def project_world(
    points_world: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    camera_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points; return pixels and a positive-depth mask."""
    points = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
    camera_position = np.asarray(camera_position, dtype=np.float64).reshape(3)
    camera_points = (np.asarray(R, dtype=np.float64).reshape(3, 3) @ (points - camera_position).T).T
    depth = camera_points[:, 2]
    safe_depth = np.where(np.abs(depth) > 1e-9, depth, 1e-9)
    pixels = np.column_stack(
        [
            K[0, 0] * camera_points[:, 0] / safe_depth + K[0, 2],
            K[1, 1] * camera_points[:, 1] / safe_depth + K[1, 2],
        ]
    )
    return pixels, depth > 1e-6


def _axis_angle_rotation(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for a small axis-angle perturbation."""
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12 or abs(float(angle_rad)) < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    return identity + math.sin(angle_rad) * skew + (1.0 - math.cos(angle_rad)) * (skew @ skew)


def perturb_camera(
    K: np.ndarray,
    R_world_to_camera: np.ndarray,
    camera_position: np.ndarray,
    rng: np.random.Generator,
    focal_noise_pct: float,
    principal_noise_px: float,
    rotation_noise_deg: float,
    position_noise_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a calibrated camera estimate with controlled errors.

    Rendering always uses the true camera. The returned camera is what a
    localisation module would receive after imperfect calibration/pose
    estimation, so calibration error can be measured separately from point
    error.
    """
    estimated_K = np.asarray(K, dtype=np.float64).copy()
    focal_scale = 1.0 + float(rng.normal(0.0, max(0.0, focal_noise_pct)))
    estimated_K[0, 0] *= focal_scale
    estimated_K[1, 1] *= focal_scale
    estimated_K[0, 2] += float(rng.normal(0.0, max(0.0, principal_noise_px)))
    estimated_K[1, 2] += float(rng.normal(0.0, max(0.0, principal_noise_px)))

    angle = math.radians(float(rng.normal(0.0, max(0.0, rotation_noise_deg))))
    delta_R = _axis_angle_rotation(rng.normal(0.0, 1.0, size=3), angle)
    estimated_R = delta_R @ np.asarray(R_world_to_camera, dtype=np.float64)
    estimated_position = np.asarray(camera_position, dtype=np.float64).reshape(3) + rng.normal(
        0.0, max(0.0, position_noise_m), size=3
    )
    return estimated_K, estimated_R, estimated_position


def clamp_norm_pixel(pixel: Sequence[float], width: int, height: int) -> list[float]:
    return [
        float(np.clip(float(pixel[0]) / max(width - 1, 1), 0.0, 1.0)),
        float(np.clip(float(pixel[1]) / max(height - 1, 1), 0.0, 1.0)),
    ]


def pixel_to_norm(pixel: Sequence[float], width: int, height: int) -> list[float]:
    # p_fire in the existing project is normalised by image width/height.
    return [
        float(np.clip(float(pixel[0]) / max(width, 1), 0.0, 1.0)),
        float(np.clip(float(pixel[1]) / max(height, 1), 0.0, 1.0)),
    ]


def project_bbox(
    point_world: np.ndarray,
    height: float,
    radius: float,
    K: np.ndarray,
    R: np.ndarray,
    camera_position: np.ndarray,
    width: int,
    image_height: int,
) -> tuple[list[float], list[list[float]]]:
    """Project a coarse 3D flame envelope and return xyxy + polygon points."""
    center = np.asarray(point_world, dtype=np.float64).reshape(3)
    samples: list[np.ndarray] = []
    # Several rings make the 2D box respond to depth and camera pose instead
    # of being an arbitrary 2D rectangle.
    for fraction in np.linspace(0.0, 1.0, 9):
        z = height * fraction
        ring_radius = radius * (1.0 - fraction) ** 0.65
        if fraction >= 0.98:
            ring_radius = radius * 0.08
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            samples.append(
                center
                + np.array(
                    [
                        ring_radius * math.cos(angle),
                        ring_radius * math.sin(angle),
                        z,
                    ],
                    dtype=np.float64,
                )
            )
    projected, valid = project_world(np.asarray(samples), K, R, camera_position)
    projected = projected[valid]
    projected = projected[
        (projected[:, 0] >= -width)
        & (projected[:, 0] <= 2 * width)
        & (projected[:, 1] >= -image_height)
        & (projected[:, 1] <= 2 * image_height)
    ]
    if len(projected) == 0:
        return [0.0, 0.0, 0.0, 0.0], []
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    bbox = [
        float(np.clip(x1, 0.0, width - 1)),
        float(np.clip(y1, 0.0, image_height - 1)),
        float(np.clip(x2, 0.0, width - 1)),
        float(np.clip(y2, 0.0, image_height - 1)),
    ]
    return bbox, projected.tolist()


def _room_background(
    size: tuple[int, int],
    floor_y: int,
    rng: np.random.Generator,
    dark: bool = False,
) -> Image.Image:
    width, height = size
    y = np.arange(height, dtype=np.float32)[:, None]
    top = np.array([15.0, 20.0, 28.0] if dark else [30.0, 38.0, 50.0])
    bottom = np.array([55.0, 48.0, 39.0] if dark else [100.0, 89.0, 66.0])
    alpha = np.clip(y / max(floor_y, 1), 0.0, 1.0)
    wall = top[None, None, :] * (1.0 - alpha[:, :, None]) + bottom[None, None, :] * alpha[:, :, None]
    wall = np.repeat(wall, width, axis=1)
    floor_colour = np.array([43.0, 39.0, 35.0] if dark else [91.0, 78.0, 56.0])
    wall[floor_y:, :, :] = floor_colour
    grain = rng.normal(0.0, 2.0, size=(height, width, 1))
    array = np.clip(wall + grain, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


def _draw_room_details(
    image: Image.Image,
    floor_y: int,
    rng: np.random.Generator,
) -> None:
    drawer = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    drawer.line((0, floor_y, width, floor_y), fill=(12, 12, 12, 180), width=max(1, width // 320))
    # Perspective-like floor lines are visual context only; the mesh is the
    # actual geometry used by the evaluator.
    for fraction in np.linspace(0.08, 0.92, 7):
        x = int(round(width * fraction))
        drawer.line((width // 2, floor_y, x, height), fill=(210, 180, 120, 28), width=1)
    for fraction in np.linspace(0.06, 0.90, 7):
        y = int(round(floor_y + (height - floor_y) * fraction))
        drawer.line((0, y, width, y), fill=(210, 180, 120, 20), width=1)
    # Random low-contrast wall rectangles provide domain variation without
    # changing the known floor-plane geometry.
    for _ in range(int(rng.integers(2, 6))):
        x1 = int(rng.integers(0, max(1, width - width // 5)))
        y1 = int(rng.integers(height // 10, max(height // 10 + 1, floor_y - height // 12)))
        x2 = min(width, x1 + int(rng.integers(width // 18, width // 5)))
        y2 = min(floor_y, y1 + int(rng.integers(height // 18, height // 7)))
        colour = tuple(int(v) for v in rng.integers(80, 160, size=3)) + (rng.integers(15, 48),)
        drawer.rectangle((x1, y1, x2, y2), outline=colour, width=1)


def _draw_projected_obstacles(
    image: Image.Image,
    obstacles: Sequence[dict[str, Any]],
    K: np.ndarray,
    R: np.ndarray,
    camera_position: np.ndarray,
) -> None:
    """Draw low-contrast projected furniture before the flame overlay."""
    if not obstacles:
        return
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(overlay, "RGBA")
    colours = [
        (28, 36, 44, 150),
        (62, 49, 35, 170),
        (45, 53, 58, 150),
    ]
    for index, obstacle in enumerate(obstacles):
        lower = np.asarray(obstacle["min"], dtype=np.float64)
        upper = np.asarray(obstacle["max"], dtype=np.float64)
        corners = np.asarray(
            [
                [lower[0], lower[1], lower[2]],
                [upper[0], lower[1], lower[2]],
                [upper[0], upper[1], lower[2]],
                [lower[0], upper[1], lower[2]],
                [lower[0], lower[1], upper[2]],
                [upper[0], lower[1], upper[2]],
                [upper[0], upper[1], upper[2]],
                [lower[0], upper[1], upper[2]],
            ],
            dtype=np.float64,
        )
        projected, valid = project_world(corners, K, R, camera_position)
        projected = projected[valid]
        if len(projected) < 3:
            continue
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        box = (
            float(np.clip(x1, 0.0, image.width - 1)),
            float(np.clip(y1, 0.0, image.height - 1)),
            float(np.clip(x2, 0.0, image.width - 1)),
            float(np.clip(y2, 0.0, image.height - 1)),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        colour = colours[index % len(colours)]
        drawer.rectangle(box, fill=colour, outline=(170, 145, 105, 100), width=max(1, image.width // 420))
        drawer.line((box[0], box[3], box[2], box[3]), fill=(220, 190, 125, 90), width=1)
    image.paste(overlay, (0, 0), overlay)


def _polygon_from_projected(
    points: list[list[float]],
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    polygon = []
    for x, y in points:
        if math.isfinite(x) and math.isfinite(y):
            polygon.append((int(round(np.clip(x, -width, 2 * width))), int(round(np.clip(y, -height, 2 * height)))))
    return polygon


def _draw_fire(
    image: Image.Image,
    polygon: list[list[float]],
    bbox: Sequence[float],
    rng: np.random.Generator,
    flame_scale: float,
    smoke_probability: float,
) -> None:
    """Draw a stylised translucent fire over the projected 3D silhouette."""
    if not polygon:
        return
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    drawer = ImageDraw.Draw(overlay, "RGBA")
    outer = _polygon_from_projected(polygon, image.width, image.height)
    drawer.polygon(outer, fill=(214, 48, 12, 238), outline=(255, 92, 16, 245))

    x1, y1, x2, y2 = [float(v) for v in bbox]
    box_w = max(3.0, x2 - x1)
    box_h = max(3.0, y2 - y1)
    cx = (x1 + x2) * 0.5
    bottom = y2
    # Nested irregular flame tongues approximate high-frequency flame texture.
    for index, colour in enumerate(((255, 142, 18, 235), (255, 202, 44, 230), (255, 239, 132, 225))):
        scale = (0.66 - 0.14 * index) * float(flame_scale)
        tongue_count = int(rng.integers(2, 5))
        for tongue in range(tongue_count):
            tx = cx + float(rng.normal(0.0, box_w * 0.13))
            tw = box_w * scale * float(rng.uniform(0.22, 0.48))
            th = box_h * scale * float(rng.uniform(0.42, 0.86))
            ty_bottom = bottom - box_h * float(rng.uniform(0.01, 0.20))
            points = [
                (tx - tw, ty_bottom),
                (tx - tw * 0.82, ty_bottom - th * 0.28),
                (tx - tw * 0.25, ty_bottom - th * 0.74),
                (tx + tw * float(rng.uniform(-0.12, 0.12)), ty_bottom - th),
                (tx + tw * 0.42, ty_bottom - th * 0.66),
                (tx + tw, ty_bottom - th * 0.20),
                (tx + tw * 0.88, ty_bottom),
            ]
            drawer.polygon(points, fill=colour)

    if rng.random() < float(smoke_probability):
        smoke = Image.new("RGBA", image.size, (0, 0, 0, 0))
        smoke_draw = ImageDraw.Draw(smoke, "RGBA")
        for _ in range(int(rng.integers(2, 5))):
            sx = cx + float(rng.normal(0.0, box_w * 0.13))
            sy = y1 - float(rng.uniform(0.0, box_h * 0.18))
            radius = max(2.0, box_w * float(rng.uniform(0.08, 0.20)))
            smoke_draw.ellipse(
                (sx - radius, sy - radius, sx + radius, sy + radius),
                fill=(110, 112, 116, int(rng.integers(22, 70))),
            )
        smoke = smoke.filter(ImageFilter.GaussianBlur(max(1, image.width // 320)))
        overlay = Image.alpha_composite(overlay, smoke)

    image.paste(overlay, (0, 0), overlay)


def _add_image_noise(
    image: Image.Image,
    rng: np.random.Generator,
    gaussian_sigma: float,
    blur_probability: float,
    brightness_jitter: float,
    jpeg_quality: int,
) -> Image.Image:
    result = image.convert("RGB")
    if brightness_jitter > 0.0:
        factor = float(rng.uniform(1.0 - brightness_jitter, 1.0 + brightness_jitter))
        array = np.asarray(result, dtype=np.float32) * factor
    else:
        array = np.asarray(result, dtype=np.float32)
    if gaussian_sigma > 0.0:
        array += rng.normal(0.0, gaussian_sigma, size=array.shape)
    result = Image.fromarray(np.clip(array, 0.0, 255.0).astype(np.uint8), mode="RGB")
    if blur_probability > 0.0 and rng.random() < blur_probability:
        result = result.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.2, 1.2))))
    # JPEG round-trip is a useful, controlled compression nuisance.  It is
    # skipped when a lossless output is requested with quality=100.
    if int(jpeg_quality) < 100:
        from io import BytesIO

        buffer = BytesIO()
        result.save(buffer, format="JPEG", quality=int(np.clip(jpeg_quality, 20, 100)))
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
    return result


def _noise_point(
    clean_pixel: Sequence[float],
    rng: np.random.Generator,
    sigma_px: float,
    miss_probability: float,
    width: int,
    height: int,
) -> tuple[Optional[list[float]], float, bool]:
    if rng.random() < float(miss_probability):
        return None, 0.0, False
    noisy = np.asarray(clean_pixel, dtype=np.float64) + rng.normal(0.0, float(sigma_px), size=2)
    noisy[0] = np.clip(noisy[0], 0.0, width - 1)
    noisy[1] = np.clip(noisy[1], 0.0, height - 1)
    error = float(np.linalg.norm(noisy - np.asarray(clean_pixel, dtype=np.float64)))
    confidence = float(np.exp(-0.5 * (error / max(float(sigma_px) * 2.0, 1.0)) ** 2))
    return noisy.tolist(), confidence, True


@dataclass
class GeneratorConfig:
    output_dir: Path
    scenes: int = 60
    frames_per_scene: int = 4
    image_size: tuple[int, int] = (640, 640)
    seed: int = 7
    fire_probability: float = 0.82
    point_noise_px: float = 3.0
    miss_probability: float = 0.04
    image_noise_sigma: float = 2.2
    blur_probability: float = 0.12
    brightness_jitter: float = 0.18
    jpeg_quality: int = 92
    calibration_focal_noise_pct: float = 0.01
    calibration_principal_noise_px: float = 1.5
    calibration_rotation_noise_deg: float = 0.25
    calibration_position_noise_m: float = 0.02
    room_width: float = DEFAULT_ROOM[0]
    room_depth: float = DEFAULT_ROOM[1]
    room_height: float = DEFAULT_ROOM[2]
    mesh_profile: str = "floor"
    fire_surface: str = "floor"
    smoke_probability: float = 0.45
    preview_count: int = 12


class SyntheticFireGenerator:
    def __init__(self, config: GeneratorConfig):
        self.config = config
        self.rng = np.random.default_rng(int(config.seed))
        self.py_rng = random.Random(int(config.seed))
        self.width, self.height = config.image_size
        self.output = Path(config.output_dir)
        self.image_dir = self.output / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.mesh_path = self.output / "room_mesh.json"
        self.mesh_data = self._write_mesh()
        self.mesh_object = TriangleMesh(
            np.asarray(self.mesh_data["vertices"], dtype=np.float64),
            np.asarray(self.mesh_data["faces"], dtype=np.int64),
        )
        obstacle_faces = np.asarray(
            [
                face
                for obstacle in self.mesh_data.get("obstacles", [])
                for face in self.mesh_data["faces"][
                    int(obstacle["face_start"]): int(obstacle["face_start"]) + int(obstacle["face_count"])
                ]
            ],
            dtype=np.int64,
        ).reshape(-1, 3)
        self.obstacle_mesh = (
            TriangleMesh(np.asarray(self.mesh_data["vertices"], dtype=np.float64), obstacle_faces)
            if len(obstacle_faces)
            else None
        )

    def intrinsics(self) -> np.ndarray:
        width, height = self.width, self.height
        # Vary focal length by scene, but keep the principal point fixed at the
        # optical centre. The chosen range is plausible for a narrow CCTV view.
        focal = float(self.rng.uniform(0.90, 1.16) * max(width, height))
        return np.array(
            [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def _write_mesh(self) -> dict[str, Any]:
        mesh = build_room_mesh(
            self.config.room_width,
            self.config.room_depth,
            self.config.room_height,
            self.config.mesh_profile,
        )
        write_json(self.mesh_path, mesh)
        return mesh

    def _camera_for_scene(self, scene_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        K = self.intrinsics()
        # Cameras stay outside the floor rectangle and look into the room.
        camera = np.array(
            [
                float(self.rng.uniform(-1.5, 1.5)),
                float(self.rng.uniform(-7.5, -5.0)),
                float(self.rng.uniform(2.2, 3.3)),
            ],
            dtype=np.float64,
        )
        target = np.array(
            [
                float(self.rng.uniform(-0.8, 0.8)),
                float(self.rng.uniform(3.2, 6.5)),
                float(self.rng.uniform(0.8, 1.5)),
            ],
            dtype=np.float64,
        )
        R = look_at_rotation(camera, target)
        return K, R, camera

    def _sample_fire(self) -> tuple[np.ndarray, float, float, str]:
        margin = 0.8
        obstacles = self.mesh_data.get("obstacles", [])
        fire_surface = str(self.config.fire_surface).lower()
        use_obstacle_top = fire_surface in {"mixed", "obstacle_top"} and bool(obstacles)
        if use_obstacle_top and self.rng.random() < 0.30:
            obstacle = obstacles[int(self.rng.integers(0, len(obstacles)))]
            lower = np.asarray(obstacle["min"], dtype=np.float64)
            upper = np.asarray(obstacle["max"], dtype=np.float64)
            point = np.array(
                [
                    float(self.rng.uniform(lower[0] + 0.18, upper[0] - 0.18)),
                    float(self.rng.uniform(lower[1] + 0.18, upper[1] - 0.18)),
                    float(upper[2]),
                ],
                dtype=np.float64,
            )
            surface_name = str(obstacle.get("name", "obstacle_top"))
        else:
            point = None
            for _ in range(100):
                candidate = np.array(
                    [
                        float(self.rng.uniform(-self.config.room_width / 2 + margin, self.config.room_width / 2 - margin)),
                        float(self.rng.uniform(1.0 + margin, self.config.room_depth - margin)),
                        0.0,
                    ],
                    dtype=np.float64,
                )
                if not any(point_inside_obstacle(candidate, obstacle, margin=0.04) for obstacle in obstacles):
                    point = candidate
                    break
            if point is None:
                point = np.array([0.0, max(1.0 + margin, self.config.room_depth * 0.5), 0.0], dtype=np.float64)
            surface_name = "floor"
        height = float(self.rng.uniform(0.7, 2.4))
        radius = float(self.rng.uniform(0.20, 0.62))
        return point, height, radius, surface_name

    def _fire_visible(
        self,
        fire_point: np.ndarray,
        camera_position: np.ndarray,
        frame_R: np.ndarray,
        K: np.ndarray,
    ) -> tuple[bool, Optional[str]]:
        """Approximate visibility by testing the camera ray against obstacles.

        A fire is considered occluded only when an obstacle face is strictly
        closer than the fire point. The room walls are target surfaces, not
        occluders for a fire located inside the room.
        """
        obstacles = self.mesh_data.get("obstacles", [])
        if not obstacles:
            return True, None
        projected, valid = project_world(fire_point.reshape(1, 3), K, frame_R, camera_position)
        if not bool(valid[0]):
            return False, "behind_camera"
        origin = np.asarray(camera_position, dtype=np.float64)
        target = np.asarray(fire_point, dtype=np.float64)
        direction = target - origin
        distance = float(np.linalg.norm(direction))
        if distance < 1e-8:
            return False, "degenerate_visibility_ray"
        direction /= distance
        hit = (
            self.obstacle_mesh.intersect_ray(origin, direction, max_dist=max(0.0, distance - 1e-3))
            if self.obstacle_mesh is not None
            else None
        )
        if hit is None:
            return True, None
        _, hit_distance = hit
        # The floor beneath the fire is behind the target; only an obstacle
        # hit before the target indicates true visual occlusion.
        if float(hit_distance) < distance - 1e-3:
            return False, "occluded_by_mesh"
        return True, None

    def _floor_horizon(self, camera_position: np.ndarray, R: np.ndarray, K: np.ndarray) -> int:
        points = np.array(
            [[-self.config.room_width / 2, 0.0, 0.0], [self.config.room_width / 2, 0.0, 0.0]],
            dtype=np.float64,
        )
        projected, valid = project_world(points, K, R, camera_position)
        if valid.any():
            return int(np.clip(np.median(projected[valid, 1]), self.height * 0.35, self.height * 0.82))
        return int(self.height * 0.62)

    def _split_for_scene(self, scene_index: int) -> str:
        # Sequence-level split: adjacent frames from one scene never cross a
        # split, avoiding the most common synthetic leakage failure.
        ratio = scene_index / max(1, self.config.scenes)
        if ratio < 0.70:
            return "train"
        if ratio < 0.85:
            return "val"
        return "test"

    def generate(self) -> dict[str, Any]:
        records: list[dict[str, Any]] = []
        previews: list[tuple[Image.Image, str]] = []
        scene_rows: list[dict[str, Any]] = []

        for scene_index in range(int(self.config.scenes)):
            split = self._split_for_scene(scene_index)
            K, R, camera = self._camera_for_scene(scene_index)
            fire_present = bool(self.rng.random() < self.config.fire_probability)
            fire_point, flame_height, flame_radius, fire_surface = self._sample_fire()
            sequence_id = f"scene_{scene_index:05d}"
            scene_rows.append(
                {
                    "scene_id": sequence_id,
                    "split": split,
                    "camera_position": camera,
                    "R_world_to_camera": R,
                    "K": K,
                    "fire_present": fire_present,
                    "fire_xyz_world": fire_point if fire_present else None,
                    "fire_surface": fire_surface if fire_present else None,
                }
            )

            for frame_index in range(int(self.config.frames_per_scene)):
                # Small camera jitter emulates a short CCTV sequence while the
                # sequence-level split remains fixed.
                frame_camera = camera + np.array(
                    [
                        float(self.rng.normal(0.0, 0.035)),
                        float(self.rng.normal(0.0, 0.035)),
                        float(self.rng.normal(0.0, 0.018)),
                    ],
                    dtype=np.float64,
                )
                frame_target = np.array(
                    [
                        float(self.rng.normal(0.0, 0.025)),
                        float(self.rng.normal(4.8, 0.08)),
                        float(self.rng.normal(1.1, 0.025)),
                    ],
                    dtype=np.float64,
                )
                frame_R = look_at_rotation(frame_camera, frame_target)
                estimated_K, estimated_R, estimated_camera = perturb_camera(
                    K,
                    frame_R,
                    frame_camera,
                    self.rng,
                    focal_noise_pct=self.config.calibration_focal_noise_pct,
                    principal_noise_px=self.config.calibration_principal_noise_px,
                    rotation_noise_deg=self.config.calibration_rotation_noise_deg,
                    position_noise_m=self.config.calibration_position_noise_m,
                )
                horizon = self._floor_horizon(frame_camera, frame_R, K)
                image = _room_background(
                    self.config.image_size,
                    horizon,
                    self.rng,
                    dark=bool(self.rng.random() < 0.35),
                )
                _draw_room_details(image, horizon, self.rng)
                if self.mesh_data.get("obstacles"):
                    _draw_projected_obstacles(
                        image,
                        self.mesh_data["obstacles"],
                        K,
                        frame_R,
                        frame_camera,
                    )

                clean_pixel: Optional[np.ndarray] = None
                clean_norm: Optional[list[float]] = None
                bbox: Optional[list[float]] = None
                polygon: list[list[float]] = []
                fire_visible = False
                occlusion_reason: Optional[str] = None
                if fire_present:
                    projected_base, valid_base = project_world(
                        fire_point.reshape(1, 3), K, frame_R, frame_camera
                    )
                    fire_visible, occlusion_reason = self._fire_visible(
                        fire_point,
                        frame_camera,
                        frame_R,
                        K,
                    )
                    in_frame = bool(
                        valid_base[0]
                        and 0.0 <= float(projected_base[0, 0]) < float(self.width)
                        and 0.0 <= float(projected_base[0, 1]) < float(self.height)
                    )
                    if fire_visible and not in_frame:
                        fire_visible = False
                        occlusion_reason = "outside_image"
                    if bool(valid_base[0]) and fire_visible and in_frame:
                        clean_pixel = projected_base[0]
                        clean_norm = pixel_to_norm(clean_pixel, self.width, self.height)
                        bbox, polygon = project_bbox(
                            fire_point,
                            flame_height,
                            flame_radius,
                            K,
                            frame_R,
                            frame_camera,
                            self.width,
                            self.height,
                        )
                        _draw_fire(
                            image,
                            polygon,
                            bbox,
                            self.rng,
                            flame_scale=float(self.rng.uniform(0.8, 1.15)),
                            smoke_probability=self.config.smoke_probability,
                        )

                image = _add_image_noise(
                    image,
                    self.rng,
                    gaussian_sigma=self.config.image_noise_sigma,
                    blur_probability=self.config.blur_probability,
                    brightness_jitter=self.config.brightness_jitter,
                    jpeg_quality=self.config.jpeg_quality,
                )

                noisy_pixel = None
                noisy_confidence = 0.0
                detected = False
                if clean_pixel is not None:
                    noisy_pixel, noisy_confidence, detected = _noise_point(
                        clean_pixel,
                        self.rng,
                        sigma_px=self.config.point_noise_px,
                        miss_probability=self.config.miss_probability,
                        width=self.width,
                        height=self.height,
                    )

                sample_id = f"{sequence_id}_frame_{frame_index:03d}"
                image_path = self.image_dir / split / f"{sample_id}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(image_path, format="PNG")
                record: dict[str, Any] = {
                    "sample_id": sample_id,
                    "scene_id": sequence_id,
                    "frame_index": frame_index,
                    "split": split,
                    "image_path": str(image_path.relative_to(self.output)).replace("\\", "/"),
                    "image_size": [self.width, self.height],
                    # ``has_fire`` is the observable 2D class used by the
                    # detector. A physically present but fully occluded fire
                    # is retained as ``fire_event=1`` and ``has_fire=0`` so it
                    # is not silently treated as a valid 2D annotation.
                    "has_fire": int(fire_present and fire_visible and clean_pixel is not None),
                    "fire_event": int(fire_present),
                    "fire_visible": int(fire_visible),
                    "occlusion_reason": occlusion_reason,
                    "fire_surface": fire_surface if fire_present else None,
                    "fire_xyz_world": fire_point.tolist() if fire_present else None,
                    "fire_height_m": flame_height if fire_present else None,
                    "fire_radius_m": flame_radius if fire_present else None,
                    "p_fire": clean_norm if clean_norm is not None else [0.0, 0.0],
                    "p_fire_pixel": clean_pixel.tolist() if clean_pixel is not None else None,
                    "p_fire_noisy": pixel_to_norm(noisy_pixel, self.width, self.height)
                    if noisy_pixel is not None
                    else [0.0, 0.0],
                    "p_fire_noisy_pixel": noisy_pixel,
                    "bbox_xyxy": bbox,
                    "bbox_yolo": None
                    if bbox is None
                    else [
                        1,
                        float(((bbox[0] + bbox[2]) * 0.5) / self.width),
                        float(((bbox[1] + bbox[3]) * 0.5) / self.height),
                        float(max(0.0, bbox[2] - bbox[0]) / self.width),
                        float(max(0.0, bbox[3] - bbox[1]) / self.height),
                    ],
                    "camera": {
                        "K": K,
                        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "R_world_to_camera": frame_R,
                        "camera_position": frame_camera,
                    },
                    "camera_estimated": {
                        "K": estimated_K,
                        "dist_coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
                        "R_world_to_camera": estimated_R,
                        "camera_position": estimated_camera,
                    },
                    "noise": {
                        "image_gaussian_sigma": self.config.image_noise_sigma,
                        "image_blur_probability": self.config.blur_probability,
                        "image_brightness_jitter": self.config.brightness_jitter,
                        "jpeg_quality": self.config.jpeg_quality,
                        "point_sigma_px": self.config.point_noise_px,
                        "miss_probability": self.config.miss_probability,
                        "detected_observation": detected,
                        "observation_confidence": noisy_confidence,
                        "calibration_focal_noise_pct": self.config.calibration_focal_noise_pct,
                        "calibration_principal_noise_px": self.config.calibration_principal_noise_px,
                        "calibration_rotation_noise_deg": self.config.calibration_rotation_noise_deg,
                        "calibration_position_noise_m": self.config.calibration_position_noise_m,
                    },
                    "annotation_source": "procedural_metric_synthetic",
                    "ground_truth_is_metric": True,
                }
                records.append(record)
                if len(previews) < max(0, int(self.config.preview_count)):
                    previews.append((image.copy(), sample_id))

        write_jsonl(self.output / "manifest.jsonl", records)
        write_jsonl(self.output / "train.jsonl", (row for row in records if row["split"] == "train"))
        write_jsonl(self.output / "val.jsonl", (row for row in records if row["split"] == "val"))
        write_jsonl(self.output / "test.jsonl", (row for row in records if row["split"] == "test"))
        write_json(self.output / "scenes.json", scene_rows)
        write_json(
            self.output / "dataset_info.json",
            {
                "format": "LAB_SAM.synthetic_fire_3d.v2",
                "created_with": "synthetic_fire_3d.py",
                "image_size": [self.width, self.height],
                "room_size_m": [self.config.room_width, self.config.room_depth, self.config.room_height],
                "mesh_profile": self.config.mesh_profile,
                "coordinate_system": "X lateral, Y forward, Z up; fire contact lies on floor or an obstacle top",
                "label_contract": {
                    "has_fire": "1=fire, 0=no fire",
                    "fire_event": "physical fire exists in the scene, including an occluded fire",
                    "fire_visible": "1 when the clean contact point is visible from the camera",
                    "p_fire": "clean contact point normalised by original image width/height",
                    "p_fire_noisy": "simulated detector/ROI observation; never ground truth",
                    "fire_xyz_world": "metric 3D contact point in room frame",
                },
                "split_policy": "sequence/scene-level split: adjacent frames stay together",
                "occlusion_policy": "occluded events keep fire_event=1 but has_fire=0 and no 2D point",
                "noise_policy": {
                    "image_noise": "Gaussian sensor noise, blur, brightness and JPEG compression",
                    "label_noise": "pixel Gaussian perturbation plus miss probability",
                    "calibration_noise": "focal/principal-point, rotation and camera-position perturbations",
                },
                "counts": {
                    "records": len(records),
                    "scenes": int(self.config.scenes),
                    "train": sum(row["split"] == "train" for row in records),
                    "val": sum(row["split"] == "val" for row in records),
                    "test": sum(row["split"] == "test" for row in records),
                    "fire": sum(row["has_fire"] == 1 for row in records),
                    "no_fire": sum(row["has_fire"] == 0 for row in records),
                    "fire_events": sum(row["fire_event"] == 1 for row in records),
                    "visible_fire": sum(row["fire_visible"] == 1 for row in records),
                    "occluded_fire": sum(row["fire_event"] == 1 and row["fire_visible"] == 0 for row in records),
                },
            },
        )
        self._write_preview(previews)
        return {
            "output_dir": str(self.output),
            "records": len(records),
            "fire": sum(row["has_fire"] == 1 for row in records),
            "no_fire": sum(row["has_fire"] == 0 for row in records),
            "fire_events": sum(row["fire_event"] == 1 for row in records),
            "visible_fire": sum(row["fire_visible"] == 1 for row in records),
            "occluded_fire": sum(row["fire_event"] == 1 and row["fire_visible"] == 0 for row in records),
            "train": sum(row["split"] == "train" for row in records),
            "val": sum(row["split"] == "val" for row in records),
            "test": sum(row["split"] == "test" for row in records),
            "preview": str(self.output / "preview_contact_sheet.png"),
            "mesh": str(self.mesh_path),
        }

    def _write_preview(self, previews: list[tuple[Image.Image, str]]) -> None:
        if not previews:
            return
        thumb_width = 240
        thumb_height = int(round(self.height * thumb_width / self.width))
        columns = min(4, len(previews))
        rows = int(math.ceil(len(previews) / columns))
        sheet = Image.new("RGB", (columns * thumb_width, rows * (thumb_height + 22)), (18, 18, 18))
        drawer = ImageDraw.Draw(sheet)
        for index, (image, name) in enumerate(previews):
            x = (index % columns) * thumb_width
            y = (index // columns) * (thumb_height + 22)
            sheet.paste(image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS), (x, y))
            drawer.text((x + 4, y + thumb_height + 3), name, fill=(240, 240, 240))
        sheet.save(self.output / "preview_contact_sheet.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("working/synthetic_fire_3d_v3"))
    parser.add_argument("--scenes", type=int, default=240)
    parser.add_argument("--frames-per-scene", type=int, default=4)
    parser.add_argument("--image-size", type=int, nargs=2, default=[640, 640], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fire-probability", type=float, default=0.82)
    parser.add_argument(
        "--mesh-profile",
        choices=("floor", "obstacles", "room", "room_obstacles"),
        default="obstacles",
        help="geometry profile: floor compatibility, furniture, room shell, or both",
    )
    parser.add_argument(
        "--fire-surface",
        choices=("floor", "mixed", "obstacle_top"),
        default="mixed",
        help="sample metric fire contacts from the floor and optionally obstacle tops",
    )
    parser.add_argument("--point-noise-px", type=float, default=3.0)
    parser.add_argument("--miss-probability", type=float, default=0.04)
    parser.add_argument("--image-noise-sigma", type=float, default=2.2)
    parser.add_argument("--blur-probability", type=float, default=0.12)
    parser.add_argument("--brightness-jitter", type=float, default=0.18)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--calibration-focal-noise-pct", type=float, default=0.01)
    parser.add_argument("--calibration-principal-noise-px", type=float, default=1.5)
    parser.add_argument("--calibration-rotation-noise-deg", type=float, default=0.25)
    parser.add_argument("--calibration-position-noise-m", type=float, default=0.02)
    parser.add_argument("--smoke-probability", type=float, default=0.45)
    parser.add_argument("--preview-count", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GeneratorConfig(
        output_dir=args.output_dir,
        scenes=max(1, int(args.scenes)),
        frames_per_scene=max(1, int(args.frames_per_scene)),
        image_size=normalise_size(args.image_size),
        seed=int(args.seed),
        fire_probability=float(np.clip(args.fire_probability, 0.0, 1.0)),
        mesh_profile=str(args.mesh_profile),
        fire_surface=str(args.fire_surface),
        point_noise_px=max(0.0, float(args.point_noise_px)),
        miss_probability=float(np.clip(args.miss_probability, 0.0, 1.0)),
        image_noise_sigma=max(0.0, float(args.image_noise_sigma)),
        blur_probability=float(np.clip(args.blur_probability, 0.0, 1.0)),
        brightness_jitter=max(0.0, float(args.brightness_jitter)),
        jpeg_quality=int(np.clip(args.jpeg_quality, 20, 100)),
        calibration_focal_noise_pct=max(0.0, float(args.calibration_focal_noise_pct)),
        calibration_principal_noise_px=max(0.0, float(args.calibration_principal_noise_px)),
        calibration_rotation_noise_deg=max(0.0, float(args.calibration_rotation_noise_deg)),
        calibration_position_noise_m=max(0.0, float(args.calibration_position_noise_m)),
        smoke_probability=float(np.clip(args.smoke_probability, 0.0, 1.0)),
        preview_count=max(0, int(args.preview_count)),
    )
    summary = SyntheticFireGenerator(config).generate()
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
