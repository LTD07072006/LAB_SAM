"""Benchmark multi-plane homography/IPM for metric fire-room manifests.

This is a homography-only complement to ``benchmark_homography.py``.  The
floor-only IPM baseline is correct when the fire contact is on ``Z=0``.  It is
not correct for a fire contact on a table, cabinet or column.  This script
builds a small bank of plane-specific homographies directly from the scene
mesh and evaluates three ways of selecting a plane:

``floor_ipm``
    Always use the floor plane.  This is the existing planar baseline.
``surface_oracle``
    Select the plane from the manifest's ``fire_surface`` label.  This is an
    upper bound for a future semantic surface classifier.
``plane_bank_nearest``
    Try every visible horizontal plane, keep candidates whose intersection is
    inside the plane patch, and select the candidate nearest to the camera.
    This is a deployable geometry-only selector, although it can fail when an
    occluding surface is in front of the true contact.
``plane_bank_visible``
    Cast one ray only to identify the first visible horizontal mesh patch, then
    invert that patch's homography.  The ray is used for visibility selection;
    the returned estimate is still produced by homography/IPM.
``plane_bank_visible_floor_fallback``
    Use ``plane_bank_visible`` first and fall back to the continuous floor
    homography when the asset mesh reports an occluding/non-horizontal face.
    This is useful for asset-backed scenes whose fire annotation is a semantic
    floor contact even when the downloaded mesh contains unrelated occluders.

For each raw method an EMA version is also evaluated per scene sequence.  The
script never changes the source dataset and writes all diagnostics and 3D
visualisations into a new output directory.

Important: a homography maps a pixel to a nominated plane.  It is not a
general 3D reconstruction method.  ``surface_oracle`` uses ground-truth
surface metadata and must not be reported as an end-to-end test result.
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import benchmark_homography as base
from camera_calibration import CameraCalibration
from mesh_loader import load_triangle_mesh


METHODS = (
    "floor_ipm",
    "surface_oracle",
    "plane_bank_nearest",
    "plane_bank_visible",
    "plane_bank_visible_floor_fallback",
    "floor_ipm_ema",
    "surface_oracle_ema",
    "plane_bank_nearest_ema",
    "plane_bank_visible_ema",
    "plane_bank_visible_floor_fallback_ema",
)
COLORS: dict[str, tuple[int, int, int]] = {
    "gt": (31, 157, 85),
    "input": (230, 230, 230),
    "floor_ipm": (40, 120, 208),
    "surface_oracle": (242, 142, 43),
    "plane_bank_nearest": (148, 103, 189),
    "plane_bank_visible": (0, 150, 136),
    "plane_bank_visible_floor_fallback": (255, 127, 14),
    "floor_ipm_ema": (23, 162, 184),
    "surface_oracle_ema": (214, 39, 40),
    "plane_bank_nearest_ema": (120, 70, 170),
    "plane_bank_visible_ema": (0, 105, 92),
    "plane_bank_visible_floor_fallback_ema": (196, 78, 0),
}
FLOOR_NAMES = {"floor", "ground", "floor_plane", "floor_ipm"}


def json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def as_point(value: Any, dimensions: int = 2) -> Optional[np.ndarray]:
    return base.point(value, dimensions)


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def surface_matches(surface: Any, patch_name: str, patch_label: str) -> bool:
    target = normalize_name(surface)
    if not target:
        return False
    name = normalize_name(patch_name)
    label = normalize_name(patch_label)
    if target in {"ground", "floor_plane"}:
        target = "floor"
    if target == "floor":
        return name == "floor" or label == "floor"
    return target == name or target == label or target in name or name in target


def triangle_area(triangle: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))


def point_in_triangle(point: np.ndarray, triangle: np.ndarray, tolerance: float = 1e-6) -> bool:
    a, b, c = triangle
    v0 = c - a
    v1 = b - a
    v2 = point - a
    dot00 = float(np.dot(v0, v0))
    dot01 = float(np.dot(v0, v1))
    dot02 = float(np.dot(v0, v2))
    dot11 = float(np.dot(v1, v1))
    dot12 = float(np.dot(v1, v2))
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(denominator) < 1e-12:
        return False
    inverse = 1.0 / denominator
    u = (dot11 * dot02 - dot01 * dot12) * inverse
    v = (dot00 * dot12 - dot01 * dot02) * inverse
    return bool(u >= -tolerance and v >= -tolerance and u + v <= 1.0 + tolerance)


@dataclass
class PlanePatch:
    """A connected approximately planar mesh patch with a 2D local frame."""

    patch_id: str
    name: str
    label: str
    face_indices: np.ndarray
    origin: np.ndarray
    basis_u: np.ndarray
    basis_v: np.ndarray
    normal: np.ndarray
    triangles_uv: np.ndarray
    area_m2: float
    nominal_z: float

    def contains_uv(self, uv: np.ndarray) -> bool:
        return any(point_in_triangle(uv, triangle) for triangle in self.triangles_uv)

    def world_from_uv(self, uv: np.ndarray) -> np.ndarray:
        return self.origin + float(uv[0]) * self.basis_u + float(uv[1]) * self.basis_v

    def uv_from_world(self, xyz: np.ndarray) -> np.ndarray:
        delta = np.asarray(xyz, dtype=np.float64).reshape(3) - self.origin
        return np.asarray([np.dot(delta, self.basis_u), np.dot(delta, self.basis_v)], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        all_points = self.triangles_uv.reshape(-1, 2)
        return {
            "patch_id": self.patch_id,
            "name": self.name,
            "label": self.label,
            "face_indices": self.face_indices,
            "origin": self.origin,
            "basis_u": self.basis_u,
            "basis_v": self.basis_v,
            "normal": self.normal,
            "nominal_z": self.nominal_z,
            "area_m2": self.area_m2,
            "uv_bounds": [all_points[:, 0].min(), all_points[:, 0].max(), all_points[:, 1].min(), all_points[:, 1].max()],
        }


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def obstacle_face_names(mesh_data: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for obstacle in mesh_data.get("obstacles", []) or []:
        try:
            start = int(obstacle["face_start"])
            count = int(obstacle["face_count"])
            name = str(obstacle.get("name", obstacle.get("surface_type", "obstacle")))
        except (KeyError, TypeError, ValueError):
            continue
        for index in range(start, start + count):
            mapping[index] = name
    return mapping


def load_mesh_data(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Mesh JSON must be an object: {path}")
    return data


def extract_horizontal_patches(
    mesh_path: Path,
    mesh: Any,
    z_tolerance: float = 0.025,
    angle_degrees: float = 7.0,
    min_area_m2: float = 0.02,
) -> list[PlanePatch]:
    """Extract connected horizontal mesh patches from a JSON triangle mesh."""

    data = load_mesh_data(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    labels = data.get("face_labels", [])
    face_names = obstacle_face_names(data)
    triangles = vertices[faces]
    raw_normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(raw_normals, axis=1)
    normals = raw_normals / np.maximum(lengths[:, None], 1e-12)
    centers = triangles.mean(axis=1)
    horizontal = (lengths > 1e-10) & (np.abs(normals[:, 2]) >= math.cos(math.radians(angle_degrees)))
    candidates = np.flatnonzero(horizontal)
    if not len(candidates):
        raise RuntimeError(f"No horizontal planes found in {mesh_path}")

    # Separate different height layers first.  Connected components then keep
    # two triangles of a rectangle together without merging disconnected tops.
    layer_groups: list[list[int]] = []
    for face_index in candidates:
        z = float(centers[face_index, 2])
        destination = next((group for group in layer_groups if abs(z - float(np.mean([centers[i, 2] for i in group]))) <= z_tolerance), None)
        if destination is None:
            layer_groups.append([int(face_index)])
        else:
            destination.append(int(face_index))

    patches: list[PlanePatch] = []
    for layer_index, layer in enumerate(layer_groups):
        uf = UnionFind(len(layer))
        vertex_owner: dict[int, int] = {}
        for local_index, face_index in enumerate(layer):
            for vertex_index in faces[face_index]:
                if int(vertex_index) in vertex_owner:
                    uf.union(local_index, vertex_owner[int(vertex_index)])
                else:
                    vertex_owner[int(vertex_index)] = local_index
        components: dict[int, list[int]] = defaultdict(list)
        for local_index, face_index in enumerate(layer):
            components[uf.find(local_index)].append(int(face_index))

        for component_index, face_indices in enumerate(components.values()):
            area = sum(triangle_area(triangles[i]) for i in face_indices)
            if area < min_area_m2:
                continue
            points = triangles[face_indices].reshape(-1, 3)
            origin = points.mean(axis=0)
            _, _, vh = np.linalg.svd(points - origin, full_matrices=False)
            basis_u = vh[0]
            normal = np.mean(normals[face_indices], axis=0)
            normal /= max(float(np.linalg.norm(normal)), 1e-12)
            if float(np.dot(np.cross(basis_u, vh[1]), normal)) < 0.0:
                basis_u = -basis_u
            basis_u -= normal * float(np.dot(basis_u, normal))
            basis_u /= max(float(np.linalg.norm(basis_u)), 1e-12)
            basis_v = np.cross(normal, basis_u)
            basis_v /= max(float(np.linalg.norm(basis_v)), 1e-12)
            triangles_uv = np.asarray(
                [
                    [[np.dot(point - origin, basis_u), np.dot(point - origin, basis_v)] for point in triangles[index]]
                    for index in face_indices
                ],
                dtype=np.float64,
            )
            labels_for_faces = [str(labels[i]) if i < len(labels) else "horizontal" for i in face_indices]
            label = Counter(labels_for_faces).most_common(1)[0][0]
            named = [face_names[i] for i in face_indices if i in face_names]
            name = Counter(named).most_common(1)[0][0] if named else ("floor" if abs(float(origin[2])) < z_tolerance else label)
            patch_id = f"{normalize_name(name) or 'plane'}_z{float(origin[2]):.3f}_{layer_index}_{component_index}"
            patches.append(
                PlanePatch(
                    patch_id=patch_id,
                    name=name,
                    label=label,
                    face_indices=np.asarray(face_indices, dtype=np.int64),
                    origin=origin,
                    basis_u=basis_u,
                    basis_v=basis_v,
                    normal=normal,
                    triangles_uv=triangles_uv,
                    area_m2=float(area),
                    nominal_z=float(origin[2]),
                )
            )
    if not patches:
        raise RuntimeError(f"No horizontal patch above {min_area_m2} m2 found in {mesh_path}")
    patches.sort(key=lambda patch: (patch.nominal_z, patch.name, patch.patch_id))
    return patches


def merge_floor_tiles(
    patches: list[PlanePatch],
    z_tolerance: float = 0.025,
) -> list[PlanePatch]:
    """Merge tessellated floor tiles into one continuous floor patch.

    Asset meshes such as ReplicaCAD commonly duplicate vertices at tile
    boundaries.  The connected-component extractor therefore sees hundreds
    of small floor patches even though they are one physical plane.  The
    homography is defined on the plane, so keeping those tiles separate makes
    a geometry-only plane selector reject valid pixels near tile boundaries.
    Only patches labelled as floor (or the lowest layer when no floor label is
    available) are merged; furniture/table/cabinet patches remain separate.
    """

    if len(patches) <= 1:
        return list(patches)
    named_floor = [
        patch
        for patch in patches
        if surface_matches("floor", patch.name, patch.label)
    ]
    if not named_floor:
        lowest_z = min(float(patch.nominal_z) for patch in patches)
        named_floor = [
            patch
            for patch in patches
            if abs(float(patch.nominal_z) - lowest_z) <= z_tolerance
        ]
    if len(named_floor) <= 1:
        return list(patches)

    floor_z = float(np.median([patch.nominal_z for patch in named_floor]))
    origin = np.asarray([0.0, 0.0, floor_z], dtype=np.float64)
    basis_u = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    basis_v = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    merged_triangles: list[np.ndarray] = []
    merged_faces: list[np.ndarray] = []
    area = 0.0
    for patch in named_floor:
        world_triangles = np.asarray(
            [
                [patch.world_from_uv(uv) for uv in triangle]
                for triangle in patch.triangles_uv
            ],
            dtype=np.float64,
        )
        merged_triangles.append(
            np.asarray(
                [
                    [
                        [
                            float(np.dot(point - origin, basis_u)),
                            float(np.dot(point - origin, basis_v)),
                        ]
                        for point in triangle
                    ]
                    for triangle in world_triangles
                ],
                dtype=np.float64,
            )
        )
        merged_faces.append(patch.face_indices)
        area += float(patch.area_m2)

    merged = PlanePatch(
        patch_id=f"floor_merged_z{floor_z:.3f}",
        name="floor",
        label="floor",
        face_indices=np.concatenate(merged_faces).astype(np.int64, copy=False),
        origin=origin,
        basis_u=basis_u,
        basis_v=basis_v,
        normal=normal,
        triangles_uv=np.concatenate(merged_triangles, axis=0),
        area_m2=area,
        nominal_z=floor_z,
    )
    floor_ids = {id(patch) for patch in named_floor}
    remaining = [patch for patch in patches if id(patch) not in floor_ids]
    return sorted(remaining + [merged], key=lambda patch: (patch.nominal_z, patch.name, patch.patch_id))


def build_face_patch_index(patches: list[PlanePatch]) -> dict[int, PlanePatch]:
    """Map every mesh face retained by a horizontal patch to that patch."""

    index: dict[int, PlanePatch] = {}
    for patch in patches:
        for face_index in patch.face_indices:
            index[int(face_index)] = patch
    return index


def intersect_mesh_face(
    mesh: Any,
    origin: np.ndarray,
    direction: np.ndarray,
    max_dist: float = 100.0,
) -> Optional[tuple[np.ndarray, float, int]]:
    """Vectorised nearest ray/triangle hit with the face index included."""

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    triangles = vertices[faces]
    ray_origin = np.asarray(origin, dtype=np.float64).reshape(3)
    ray_direction = np.asarray(direction, dtype=np.float64).reshape(3)
    edge_1 = triangles[:, 1] - triangles[:, 0]
    edge_2 = triangles[:, 2] - triangles[:, 0]
    cross_1 = np.cross(ray_direction[None, :], edge_2)
    determinant = np.einsum("ij,ij->i", edge_1, cross_1)
    valid = np.abs(determinant) > 1e-10
    inverse = np.zeros_like(determinant)
    inverse[valid] = 1.0 / determinant[valid]
    delta = ray_origin[None, :] - triangles[:, 0]
    bary_u = inverse * np.einsum("ij,ij->i", delta, cross_1)
    cross_2 = np.cross(delta, edge_1)
    bary_v = inverse * np.einsum("j,ij->i", ray_direction, cross_2)
    distance = inverse * np.einsum("ij,ij->i", edge_2, cross_2)
    valid &= bary_u >= 0.0
    valid &= bary_v >= 0.0
    valid &= bary_u + bary_v <= 1.0
    valid &= distance > 1e-8
    valid &= distance <= float(max_dist)
    if not np.any(valid):
        return None
    candidate_indices = np.flatnonzero(valid)
    face_index = int(candidate_indices[np.argmin(distance[candidate_indices])])
    hit_distance = float(distance[face_index])
    return ray_origin + hit_distance * ray_direction, hit_distance, face_index


def homography_for_patch(calibration: CameraCalibration, patch: PlanePatch) -> np.ndarray:
    camera_origin = calibration.R @ patch.origin.reshape(3) + calibration.t.reshape(3)
    h = calibration.K @ np.column_stack(
        (calibration.R @ patch.basis_u, calibration.R @ patch.basis_v, camera_origin)
    )
    if not np.all(np.isfinite(h)) or abs(float(np.linalg.det(h))) < 1e-12:
        raise ValueError(f"Singular homography for patch {patch.patch_id}")
    return h / h[2, 2]


def unproject_patch(
    calibration: CameraCalibration,
    patch: PlanePatch,
    pixel: Optional[np.ndarray],
    strict_patch: bool = True,
) -> tuple[Optional[np.ndarray], str, Optional[float], Optional[np.ndarray]]:
    if pixel is None:
        return None, "no_pixel", None, None
    try:
        ideal = calibration.undistort_pixels(np.asarray(pixel, dtype=np.float64).reshape(1, 2))[0]
        h = homography_for_patch(calibration, patch)
        homogeneous = np.asarray([ideal[0], ideal[1], 1.0], dtype=np.float64) @ np.linalg.inv(h).T
        if abs(float(homogeneous[2])) < 1e-12:
            return None, "horizon", None, None
        uv = homogeneous[:2] / homogeneous[2]
        # A real scanned floor is often tessellated into many disconnected
        # tiles with duplicated boundary vertices.  The floor homography is
        # defined on the continuous floor plane, so callers may deliberately
        # skip the per-tile containment test.  Object/table patches keep the
        # strict test to avoid accepting points outside their physical top.
        if strict_patch and not patch.contains_uv(uv):
            return None, "outside_patch", None, uv
        xyz = patch.world_from_uv(uv)
        camera_point = calibration.R @ xyz + calibration.t.reshape(3)
        depth = float(camera_point[2])
        if depth <= 1e-7:
            return None, "behind_camera", depth, uv
        return xyz, "valid_homography", depth, uv
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        return None, str(exc), None, None


def patch_by_surface(patches: list[PlanePatch], surface: Any) -> Optional[PlanePatch]:
    matching = [patch for patch in patches if surface_matches(surface, patch.name, patch.label)]
    if not matching:
        return None
    target = normalize_name(surface)
    if target == "floor":
        return min(matching, key=lambda patch: abs(patch.nominal_z))
    # A box contributes both a bottom face at Z=0 and a contact/top face at
    # its actual height.  The manifest's object-level surface label does not
    # distinguish those faces, so the oracle selector takes the highest
    # horizontal patch with that object name.  This is still explicitly an
    # upper-bound diagnostic; a deployable selector must infer this from
    # semantics/visibility rather than from ground-truth surface metadata.
    return max(matching, key=lambda patch: (patch.nominal_z, patch.area_m2))


def floor_patch(patches: list[PlanePatch]) -> PlanePatch:
    candidates = [patch for patch in patches if surface_matches("floor", patch.name, patch.label)]
    return min(candidates or patches, key=lambda patch: abs(patch.nominal_z))


def nearest_patch(
    calibration: CameraCalibration,
    patches: list[PlanePatch],
    pixel: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], str, Optional[PlanePatch], Optional[float], Optional[np.ndarray]]:
    candidates: list[tuple[float, PlanePatch, np.ndarray, np.ndarray]] = []
    for patch in patches:
        xyz, status, depth, uv = unproject_patch(
            calibration,
            patch,
            pixel,
            strict_patch=not surface_matches("floor", patch.name, patch.label),
        )
        if xyz is not None and depth is not None and uv is not None:
            candidates.append((depth, patch, xyz, uv))
    if not candidates:
        return None, "no_plane_candidate", None, None, None
    depth, patch, xyz, uv = min(candidates, key=lambda item: item[0])
    return xyz, "valid_plane_bank", patch, depth, uv


def visible_patch(
    calibration: CameraCalibration,
    mesh: Any,
    patches: list[PlanePatch],
    pixel: Optional[np.ndarray],
    face_patch_index: Optional[dict[int, PlanePatch]] = None,
    patch_tolerance_m: float = 0.03,
) -> tuple[Optional[np.ndarray], str, Optional[PlanePatch], Optional[float], Optional[np.ndarray]]:
    """Select the horizontal patch at the first visible mesh intersection.

    This is the geometry-only selector for the deployable multi-plane branch.
    It does not use ``fire_surface``.  A ray is cast through the input pixel,
    the nearest mesh hit is found, and the horizontal patch containing that hit
    is used for the homography inversion.  The returned point is recomputed by
    homography so that this branch remains a genuine IPM/homography estimate,
    not a hidden ray-casting result.
    """

    if pixel is None:
        return None, "no_pixel", None, None, None
    try:
        ideal = calibration.undistort_pixels(np.asarray(pixel, dtype=np.float64).reshape(1, 2))[0]
        origin, direction = calibration.geometry().pixel_to_ray(float(ideal[0]), float(ideal[1]))
        hit_with_face = intersect_mesh_face(mesh, origin, direction, max_dist=100.0)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return None, "invalid_ray", None, None, None
    if hit_with_face is None:
        return None, "no_mesh_intersection", None, None, None
    hit_point, depth, face_index = hit_with_face
    hit_point = np.asarray(hit_point, dtype=np.float64).reshape(3)

    if face_patch_index is not None:
        direct_patch = face_patch_index.get(int(face_index))
        if direct_patch is not None:
            xyz, status, homography_depth, uv = unproject_patch(
                calibration,
                direct_patch,
                pixel,
                strict_patch=not surface_matches("floor", direct_patch.name, direct_patch.label),
            )
            if xyz is not None:
                return xyz, "valid_visible_plane", direct_patch, homography_depth, uv

    candidates: list[tuple[float, PlanePatch, np.ndarray]] = []
    for patch in patches:
        distance_to_plane = abs(float(np.dot(hit_point - patch.origin, patch.normal)))
        if distance_to_plane > patch_tolerance_m:
            continue
        uv = patch.uv_from_world(hit_point)
        if patch.contains_uv(uv):
            candidates.append((distance_to_plane, patch, uv))
    if not candidates:
        return None, "visible_hit_not_in_horizontal_patch", None, float(depth), None

    _, patch, hit_uv = min(candidates, key=lambda item: item[0])
    xyz, status, homography_depth, uv = unproject_patch(
        calibration,
        patch,
        pixel,
        strict_patch=True,
    )
    if xyz is None:
        return None, f"visible_{status}", patch, homography_depth, uv
    # Keep a diagnostic of the actual hit-vs-homography consistency.  The
    # homography point is the returned estimate; hit_uv is not used to alter it.
    return xyz, "valid_visible_plane", patch, homography_depth, uv


def source_points(row: dict[str, Any], prediction: Optional[dict[str, Any]]) -> dict[str, Optional[np.ndarray]]:
    coarse = as_point(row.get("p_fire_noisy_pixel"))
    if prediction is not None:
        prediction_coarse = as_point(prediction.get("coarse_pixel"))
        if prediction_coarse is not None:
            coarse = prediction_coarse
    return {
        "coarse": coarse,
        "roi_blend": as_point(prediction.get("blend_pixel")) if prediction is not None else None,
        "oracle": as_point(row.get("p_fire_pixel")),
    }


def load_prediction_index(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(row["sample_id"]): row for row in payload.get("records", []) if row.get("sample_id") is not None}


def select_rows(rows: list[dict[str, Any]], maximum: int, policy: str) -> list[dict[str, Any]]:
    if maximum <= 0 or len(rows) <= maximum:
        return list(rows)
    if policy == "head":
        return list(rows[:maximum])
    indices = np.linspace(0, len(rows) - 1, maximum, dtype=int)
    return [rows[int(index)] for index in indices]


def load_rows(dataset: Path, split: str) -> list[dict[str, Any]]:
    rows = base.load_rows(dataset, split)
    return rows


def patch_summary(patches: list[PlanePatch]) -> list[dict[str, Any]]:
    return [patch.to_dict() for patch in patches]


def evaluate_row(
    row: dict[str, Any],
    patches: list[PlanePatch],
    mesh: Any,
    face_patch_index: dict[int, PlanePatch],
    prediction: Optional[dict[str, Any]],
    ema_alpha: float,
) -> dict[str, Any]:
    calibration = base.calibration_from_row(row)
    sources = source_points(row, prediction)
    floor = floor_patch(patches)
    surface = patch_by_surface(patches, row.get("fire_surface", row.get("surface_type")))
    detail: dict[str, Any] = {
        "sample_id": str(row.get("sample_id")),
        "scene_id": str(row.get("scene_id", "unknown")),
        "frame_index": int(row.get("frame_index", 0)),
        "image_path": row.get("_image_path"),
        # Keep the source calibration in the derived record.  The visual
        # exporters use it to project 3D estimates back onto the RGB image.
        "image_size": row.get("image_size"),
        "camera": row.get("camera"),
        "surface": row.get("fire_surface", row.get("surface_type")),
        "gt_pixel": as_point(row.get("p_fire_pixel")),
        "gt_xyz": as_point(row.get("fire_xyz_world"), 3),
        "sources": sources,
        "floor_patch": floor.patch_id,
        "surface_patch": None if surface is None else surface.patch_id,
        "methods": {},
    }
    for source_name, pixel in sources.items():
        started = time.perf_counter()
        floor_xyz, floor_status, floor_depth, floor_uv = unproject_patch(
            calibration, floor, pixel, strict_patch=False
        )
        detail["methods"][f"{source_name}:floor_ipm"] = {
            "point": floor_xyz,
            "status": floor_status,
            "patch_id": floor.patch_id,
            "patch_name": floor.name,
            "depth_m": floor_depth,
            "uv": floor_uv,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        started = time.perf_counter()
        if surface is None:
            oracle_xyz, oracle_status, oracle_depth, oracle_uv = None, "surface_patch_not_found", None, None
        else:
            oracle_xyz, oracle_status, oracle_depth, oracle_uv = unproject_patch(
                calibration,
                surface,
                pixel,
                strict_patch=not surface_matches("floor", surface.name, surface.label),
            )
        detail["methods"][f"{source_name}:surface_oracle"] = {
            "point": oracle_xyz,
            "status": oracle_status,
            "patch_id": None if surface is None else surface.patch_id,
            "patch_name": None if surface is None else surface.name,
            "depth_m": oracle_depth,
            "uv": oracle_uv,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        started = time.perf_counter()
        bank_xyz, bank_status, bank, bank_depth, bank_uv = nearest_patch(calibration, patches, pixel)
        detail["methods"][f"{source_name}:plane_bank_nearest"] = {
            "point": bank_xyz,
            "status": bank_status,
            "patch_id": None if bank is None else bank.patch_id,
            "patch_name": None if bank is None else bank.name,
            "depth_m": bank_depth,
            "uv": bank_uv,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        started = time.perf_counter()
        visible_xyz, visible_status, visible, visible_depth, visible_uv = visible_patch(
            calibration,
            mesh,
            patches,
            pixel,
            face_patch_index,
        )
        detail["methods"][f"{source_name}:plane_bank_visible"] = {
            "point": visible_xyz,
            "status": visible_status,
            "patch_id": None if visible is None else visible.patch_id,
            "patch_name": None if visible is None else visible.name,
            "depth_m": visible_depth,
            "uv": visible_uv,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
        started = time.perf_counter()
        if visible_xyz is not None:
            fallback_xyz, fallback_status, fallback_patch, fallback_depth, fallback_uv = (
                visible_xyz,
                "valid_visible_plane",
                visible,
                visible_depth,
                visible_uv,
            )
        else:
            fallback_xyz, fallback_status, fallback_depth, fallback_uv = unproject_patch(
                calibration,
                floor,
                pixel,
                strict_patch=False,
            )
            fallback_patch = floor
            if fallback_xyz is not None:
                fallback_status = f"{fallback_status}_floor_fallback"
            else:
                fallback_status = f"{visible_status}_and_{fallback_status}"
        detail["methods"][f"{source_name}:plane_bank_visible_floor_fallback"] = {
            "point": fallback_xyz,
            "status": fallback_status,
            "patch_id": None if fallback_patch is None else fallback_patch.patch_id,
            "patch_name": None if fallback_patch is None else fallback_patch.name,
            "depth_m": fallback_depth,
            "uv": fallback_uv,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
        }
    return detail


def apply_ema(rows: list[dict[str, Any]], source_name: str, method: str, alpha: float, gate_m: float) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    raw_method = method.replace("_ema", "")
    for scene_rows in grouped.values():
        scene_rows.sort(key=lambda item: item["frame_index"])
        state: Optional[np.ndarray] = None
        for row in scene_rows:
            raw = row["methods"].get(f"{source_name}:{raw_method}", {})
            raw_point = as_point(raw.get("point"), 3)
            if raw_point is None:
                estimate, status = None, "ema_no_input"
            elif state is None:
                state, estimate, status = raw_point.copy(), raw_point.copy(), "valid_ema_init"
            elif float(np.linalg.norm(raw_point - state)) > gate_m:
                estimate, status = state.copy(), "ema_gate_reject"
            else:
                state = alpha * raw_point + (1.0 - alpha) * state
                estimate, status = state.copy(), "valid_ema"
            row["methods"][f"{source_name}:{method}"] = {
                "point": estimate,
                "status": status,
                "patch_id": raw.get("patch_id"),
                "patch_name": raw.get("patch_name"),
                "depth_m": raw.get("depth_m"),
                "uv": raw.get("uv"),
                "latency_ms": raw.get("latency_ms", 0.0),
            }


def metric_for(rows: list[dict[str, Any]], source: str, method: str, surface: Optional[str] = None) -> dict[str, Any]:
    selected = rows
    if surface is not None:
        selected = [row for row in rows if normalize_name(row.get("surface")) == normalize_name(surface)]
    errors: list[float] = []
    deltas: list[np.ndarray] = []
    latencies: list[float] = []
    statuses: Counter[str] = Counter()
    patch_names: Counter[str] = Counter()
    for row in selected:
        result = row["methods"].get(f"{source}:{method}", {})
        statuses[str(result.get("status", "missing"))] += 1
        if result.get("patch_name"):
            patch_names[str(result["patch_name"])] += 1
        estimate = as_point(result.get("point"), 3)
        truth = as_point(row.get("gt_xyz"), 3)
        if estimate is not None and truth is not None:
            delta = estimate - truth
            deltas.append(delta)
            errors.append(float(np.linalg.norm(delta)))
        if result.get("latency_ms") is not None:
            latencies.append(float(result["latency_ms"]))
    values = np.asarray(errors, dtype=np.float64)
    xyz = np.asarray(deltas, dtype=np.float64).reshape(-1, 3) if deltas else np.empty((0, 3))
    return {
        "records": len(selected),
        "valid": int(len(values)),
        "success_rate": float(len(values) / max(1, len(selected))),
        "mae_m": None if not len(values) else float(values.mean()),
        "median_m": None if not len(values) else float(np.median(values)),
        "p95_m": None if not len(values) else float(np.percentile(values, 95)),
        "under_0.10m": None if not len(values) else float(np.mean(values <= 0.10)),
        "under_0.25m": None if not len(values) else float(np.mean(values <= 0.25)),
        "under_0.50m": None if not len(values) else float(np.mean(values <= 0.50)),
        "under_1.00m": None if not len(values) else float(np.mean(values <= 1.00)),
        "xyz_mae_m": None if not len(xyz) else np.mean(np.abs(xyz), axis=0),
        "mean_latency_ms": None if not latencies else float(np.mean(latencies)),
        "p95_latency_ms": None if not latencies else float(np.percentile(latencies, 95)),
        "statuses": dict(statuses),
        "selected_patch_names": dict(patch_names),
    }


def all_metrics(rows: list[dict[str, Any]], sources: Iterable[str]) -> dict[str, Any]:
    surfaces = sorted({str(row.get("surface")) for row in rows if row.get("surface") is not None})
    result: dict[str, Any] = {}
    for source in sources:
        for method in METHODS:
            result[f"{source}:{method}"] = {
                "all_surfaces": metric_for(rows, source, method),
                "by_surface": {surface: metric_for(rows, source, method, surface) for surface in surfaces},
            }
    return result


def project_world_to_pixel(value: Any, calibration: CameraCalibration) -> Optional[np.ndarray]:
    xyz = as_point(value, 3)
    if xyz is None:
        return None
    pixels, valid = base.project_world(xyz.reshape(1, 3), calibration)
    return pixels[0] if bool(valid[0]) else None


def draw_marker(draw: ImageDraw.ImageDraw, value: Any, color: tuple[int, int, int], label: str) -> None:
    point = as_point(value)
    if point is None:
        return
    x, y = float(point[0]), float(point[1])
    radius = 5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=2)
    draw.text((x + radius + 2, y - radius - 2), label, fill=color)


def contact_sheet(rows: list[dict[str, Any]], output: Path, maximum: int) -> None:
    tiles: list[Image.Image] = []
    for row in rows[:maximum] if maximum > 0 else rows:
        image_path = Path(str(row["image_path"]))
        if not image_path.is_file():
            continue
        with Image.open(image_path) as opened:
            image = opened.convert("RGB").copy()
        draw = ImageDraw.Draw(image)
        draw_marker(draw, row.get("gt_pixel"), COLORS["gt"], "GT")
        source_name = "roi_blend" if row["sources"].get("roi_blend") is not None else "coarse"
        draw_marker(draw, row["sources"].get(source_name), COLORS["input"], "input")
        calibration = base.calibration_from_row(row)
        for method in (
            "floor_ipm",
            "surface_oracle",
            "plane_bank_nearest",
            "plane_bank_visible",
            "plane_bank_visible_floor_fallback",
            "plane_bank_nearest_ema",
            "plane_bank_visible_ema",
            "plane_bank_visible_floor_fallback_ema",
        ):
            value = row["methods"].get(f"{source_name}:{method}", {}).get("point")
            draw_marker(draw, project_world_to_pixel(value, calibration), COLORS[method], method.replace("_", " "))
        tiles.append(ImageOps.contain(image, (520, 360)))
    if not tiles:
        return
    columns = 3
    sheet = Image.new("RGB", (columns * 520, math.ceil(len(tiles) / columns) * 360), "white")
    for index, tile in enumerate(tiles):
        x = (index % columns) * 520 + (520 - tile.width) // 2
        y = (index // columns) * 360 + (360 - tile.height) // 2
        sheet.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=94)


def static_plots(rows: list[dict[str, Any]], metrics: dict[str, Any], mesh: Any, patches: list[PlanePatch], output: Path, maximum: int) -> dict[str, str]:
    outputs: dict[str, str] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        return outputs

    source_name = "roi_blend" if any(row["sources"].get("roi_blend") is not None for row in rows) else "coarse"
    names = (
        "floor_ipm",
        "surface_oracle",
        "plane_bank_nearest",
        "plane_bank_visible",
        "plane_bank_visible_floor_fallback",
        "floor_ipm_ema",
        "surface_oracle_ema",
        "plane_bank_nearest_ema",
        "plane_bank_visible_ema",
        "plane_bank_visible_floor_fallback_ema",
    )
    values = [metrics[f"{source_name}:{name}"]["all_surfaces"]["mae_m"] for name in names]
    figure, axis = plt.subplots(figsize=(11, 5.5), dpi=150)
    axis.bar(list(names), [np.nan if value is None else value for value in values], color=[np.asarray(COLORS[name]) / 255.0 for name in names])
    axis.set_ylabel("3D MAE (m)")
    axis.set_title(f"Multi-plane homography/IPM ({source_name})")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    path = output / "multiplane_homography_error_comparison.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    outputs["error_comparison_png"] = str(path)

    figure, axis = plt.subplots(figsize=(9, 7), dpi=150)
    for patch in patches:
        points = np.asarray([patch.world_from_uv(uv) for uv in patch.triangles_uv.reshape(-1, 2)])
        axis.scatter(points[:, 0], points[:, 1], s=5, alpha=0.25)
        center = points.mean(axis=0)
        axis.text(center[0], center[1], patch.name, fontsize=8)
    selected = rows if maximum <= 0 else rows[:maximum]
    for row in selected:
        truth = as_point(row.get("gt_xyz"), 3)
        if truth is not None:
            axis.scatter(truth[0], truth[1], c=[np.asarray(COLORS["gt"]) / 255.0], marker="o", s=26)
        for name in (
            "floor_ipm",
            "surface_oracle",
            "plane_bank_nearest",
            "plane_bank_visible",
            "plane_bank_visible_floor_fallback",
            "plane_bank_nearest_ema",
            "plane_bank_visible_ema",
            "plane_bank_visible_floor_fallback_ema",
        ):
            value = as_point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                axis.scatter(value[0], value[1], c=[np.asarray(COLORS[name]) / 255.0], marker="x", s=24)
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_title("Top view: plane patches and estimates")
    axis.axis("equal")
    figure.tight_layout()
    path = output / "multiplane_floor_map.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    outputs["floor_map_png"] = str(path)

    figure = plt.figure(figsize=(11, 8), dpi=150)
    axis = figure.add_subplot(111, projection="3d")
    triangles = mesh.vertices[mesh.faces]
    axis.add_collection3d(Poly3DCollection(triangles, alpha=0.12, facecolor="#8c8c8c", edgecolor="#777777", linewidth=0.25))
    for row in selected:
        truth = as_point(row.get("gt_xyz"), 3)
        if truth is not None:
            axis.scatter(*truth, c=[np.asarray(COLORS["gt"]) / 255.0], marker="o", s=28)
        for name in (
            "floor_ipm",
            "surface_oracle",
            "plane_bank_nearest",
            "plane_bank_visible",
            "plane_bank_visible_floor_fallback",
            "plane_bank_nearest_ema",
            "plane_bank_visible_ema",
            "plane_bank_visible_floor_fallback_ema",
        ):
            value = as_point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                axis.scatter(*value, c=[np.asarray(COLORS[name]) / 255.0], marker="x", s=30)
    low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_xlabel("X (m)")
    axis.set_ylabel("Y (m)")
    axis.set_zlabel("Z (m)")
    axis.set_title("Multi-plane homography estimates on the room mesh")
    figure.tight_layout()
    path = output / "multiplane_3d_overview.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    outputs["overview_3d_png"] = str(path)
    return outputs


def write_ply(rows: list[dict[str, Any]], mesh: Any, output: Path, maximum: int, source_name: str) -> None:
    selected = rows if maximum <= 0 else rows[:maximum]
    points: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    for row in selected:
        truth = as_point(row.get("gt_xyz"), 3)
        if truth is not None:
            points.append((truth, COLORS["gt"]))
        for name in (
            "floor_ipm",
            "surface_oracle",
            "plane_bank_nearest",
            "plane_bank_visible",
            "plane_bank_visible_floor_fallback",
            "floor_ipm_ema",
            "surface_oracle_ema",
            "plane_bank_nearest_ema",
            "plane_bank_visible_ema",
            "plane_bank_visible_floor_fallback_ema",
        ):
            value = as_point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                points.append((value, COLORS[name]))
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(vertices) + len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element face {len(faces)}", "property list uchar int vertex_indices", "end_header",
    ]
    lines.extend(f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f} 150 150 150" for v in vertices)
    lines.extend(f"{p[0]:.8f} {p[1]:.8f} {p[2]:.8f} {c[0]} {c[1]} {c[2]}" for p, c in points)
    lines.extend(f"3 {int(f[0])} {int(f[1])} {int(f[2])}" for f in faces)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def write_html(rows: list[dict[str, Any]], mesh: Any, output: Path, maximum: int, source_name: str) -> None:
    selected = rows if maximum <= 0 else rows[:maximum]
    traces: list[dict[str, Any]] = [{
        "type": "mesh3d", "x": mesh.vertices[:, 0].tolist(), "y": mesh.vertices[:, 1].tolist(), "z": mesh.vertices[:, 2].tolist(),
        "i": mesh.faces[:, 0].tolist(), "j": mesh.faces[:, 1].tolist(), "k": mesh.faces[:, 2].tolist(),
        "name": "Room mesh", "opacity": 0.20, "color": "#8c8c8c",
    }]
    for name in (
        "gt",
        "floor_ipm",
        "surface_oracle",
        "plane_bank_nearest",
        "plane_bank_visible",
        "plane_bank_visible_floor_fallback",
        "floor_ipm_ema",
        "surface_oracle_ema",
        "plane_bank_nearest_ema",
        "plane_bank_visible_ema",
        "plane_bank_visible_floor_fallback_ema",
    ):
        values: list[np.ndarray] = []
        labels: list[str] = []
        for row in selected:
            value = as_point(row.get("gt_xyz"), 3) if name == "gt" else as_point(row["methods"].get(f"{source_name}:{name}", {}).get("point"), 3)
            if value is not None:
                values.append(value)
                labels.append(str(row["sample_id"]))
        if not values:
            continue
        array = np.asarray(values)
        color = COLORS["gt"] if name == "gt" else COLORS[name]
        traces.append({
            "type": "scatter3d", "mode": "markers", "x": array[:, 0].tolist(), "y": array[:, 1].tolist(), "z": array[:, 2].tolist(),
            "text": labels, "name": name, "marker": {"size": 5, "color": "rgb(%d,%d,%d)" % color},
        })
    payload = json.dumps(traces, ensure_ascii=False)
    output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Multi-plane homography benchmark</title>"
        "<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script></head>"
        "<body><div id='scene' style='width:100%;height:95vh'></div><script>const traces=" + payload
        + ";Plotly.newPlot('scene',traces,{title:'Multi-plane homography/IPM',scene:{aspectmode:'data',xaxis:{title:'X (m)'},yaxis:{title:'Y (m)'},zaxis:{title:'Z (m)'}},margin:{l:0,r:0,b:0,t:45}});</script></body></html>",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = select_rows(load_rows(dataset, args.split), args.max_records, args.selection)
    mesh_path = args.mesh.expanduser().resolve() if args.mesh else (dataset / "room_mesh.json").resolve()
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")
    mesh = load_triangle_mesh(mesh_path)
    patches = extract_horizontal_patches(mesh_path, mesh, args.z_tolerance, args.horizontal_angle, args.min_patch_area)
    patches = merge_floor_tiles(patches, args.z_tolerance)
    face_patch_index = build_face_patch_index(patches)
    prediction_path = args.prediction_summary.expanduser().resolve() if args.prediction_summary else None
    prediction_index = load_prediction_index(prediction_path)
    processed = [
        evaluate_row(
            row,
            patches,
            mesh,
            face_patch_index,
            prediction_index.get(str(row.get("sample_id"))),
            args.ema_alpha,
        )
        for row in rows
    ]
    sources = [source for source in ("coarse", "roi_blend", "oracle") if any(row["sources"].get(source) is not None for row in processed)]
    for source in sources:
        apply_ema(processed, source, "floor_ipm_ema", args.ema_alpha, args.ema_gate_m)
        apply_ema(processed, source, "surface_oracle_ema", args.ema_alpha, args.ema_gate_m)
        apply_ema(processed, source, "plane_bank_nearest_ema", args.ema_alpha, args.ema_gate_m)
        apply_ema(processed, source, "plane_bank_visible_ema", args.ema_alpha, args.ema_gate_m)
        apply_ema(
            processed,
            source,
            "plane_bank_visible_floor_fallback_ema",
            args.ema_alpha,
            args.ema_gate_m,
        )
    metrics = all_metrics(processed, sources)
    source_for_visuals = "roi_blend" if "roi_blend" in sources else "coarse"
    visuals = static_plots(processed, metrics, mesh, patches, output, args.max_visual_records)
    contact_path = output / "multiplane_homography_contact_sheet.jpg"
    contact_sheet(processed, contact_path, args.max_contact_images)
    visuals["contact_sheet"] = str(contact_path)
    ply_path = output / "multiplane_homography_annotations.ply"
    write_ply(processed, mesh, ply_path, args.max_visual_records, source_for_visuals)
    visuals["ply"] = str(ply_path)
    html_path = output / "multiplane_homography_3d_interactive.html"
    write_html(processed, mesh, html_path, args.max_visual_records, source_for_visuals)
    visuals["interactive_html"] = str(html_path)
    summary = {
        "format": "LAB_SAM.benchmark_multiplane_homography.v1",
        "dataset": str(dataset),
        "mesh": str(mesh_path),
        "split": args.split,
        "records": len(processed),
        "prediction_summary": None if prediction_path is None else str(prediction_path),
        "sources": sources,
        "protocol": {
            "floor_ipm": "one analytic homography for the lowest floor patch",
            "surface_oracle": "plane selected from manifest fire_surface; upper bound, not deployable evaluation",
            "plane_bank_nearest": "all connected horizontal patches; choose valid candidate with nearest camera depth",
            "plane_bank_visible": "cast one visibility ray, map the first horizontal mesh face to a plane patch, then invert its homography",
            "plane_bank_visible_floor_fallback": "use visible-plane homography when available; otherwise use continuous floor IPM",
            "ema": {"alpha": args.ema_alpha, "gate_m": args.ema_gate_m},
            "warning": "Homography is valid only on the selected planar patch; vertical/non-planar surfaces are not evaluated.",
        },
        "plane_extraction": {
            "horizontal_angle_degrees": args.horizontal_angle,
            "z_tolerance_m": args.z_tolerance,
            "min_patch_area_m2": args.min_patch_area,
            "patches": patch_summary(patches),
        },
        "metrics": metrics,
        "records_detail": processed,
        "visual_outputs": visuals,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "plane_inventory.json", {"mesh": str(mesh_path), "patches": patch_summary(patches)})
    print(f"dataset={dataset}")
    print(f"records={len(processed)} patches={len(patches)} sources={','.join(sources)}")
    print("source/method                         MAE(m)   median(m)   P95(m)  success")
    for source in sources:
        for method in METHODS:
            metric = metrics[f"{source}:{method}"]["all_surfaces"]
            print(f"{source + ':' + method:36} {str(metric['mae_m']):>8} {str(metric['median_m']):>10} {str(metric['p95_m']):>9} {metric['success_rate']:.3f}")
    print(f"saved_summary={output / 'summary.json'}")
    return summary


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument("--prediction-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--selection", choices=("head", "even"), default="even")
    parser.add_argument("--max-visual-records", type=int, default=40)
    parser.add_argument("--max-contact-images", type=int, default=16)
    parser.add_argument("--horizontal-angle", type=float, default=7.0)
    parser.add_argument("--z-tolerance", type=float, default=0.025)
    parser.add_argument("--min-patch-area", type=float, default=0.02)
    parser.add_argument("--ema-alpha", type=float, default=0.35)
    parser.add_argument("--ema-gate-m", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_records < 0 or args.max_visual_records < 0 or args.max_contact_images < 0:
        raise ValueError("max values must be non-negative")
    if args.horizontal_angle <= 0 or args.horizontal_angle >= 45 or args.z_tolerance <= 0 or args.min_patch_area <= 0:
        raise ValueError("invalid plane extraction settings")
    if not 0.0 < args.ema_alpha <= 1.0 or args.ema_gate_m <= 0:
        raise ValueError("invalid EMA settings")
    run(args)


if __name__ == "__main__":
    main()
