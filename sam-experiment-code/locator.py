"""Camera geometry and robust ray-to-height-map intersection."""
from dataclasses import dataclass
import numpy as np

@dataclass
class RayHit:
    hit: bool
    point: object = None
    distance: float = 0.0
    status: str = "no_hit"

class CameraGeometry:
    def __init__(self, K, R, t):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.R = np.asarray(R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(t, dtype=np.float64).reshape(3, 1)
        self.K_inv = np.linalg.inv(self.K)
        self.C = (-self.R.T @ self.t).ravel()

    def pixel_to_ray(self, u, v):
        p_cam = self.K_inv @ np.array([float(u), float(v), 1.0])
        ray_dir = self.R.T @ p_cam
        norm = np.linalg.norm(ray_dir)
        if norm < 1e-12: raise ValueError("Degenerate camera ray")
        return self.C.copy(), ray_dir / norm

    def pixels_to_rays(self, pixels):
        points = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        rays_cam = np.column_stack([points, np.ones(len(points))]) @ self.K_inv.T
        rays_world = rays_cam @ self.R
        rays_world /= np.linalg.norm(rays_world, axis=1, keepdims=True)
        return np.repeat(self.C[None, :], len(points), axis=0), rays_world

class GridMap:
    def get_elevation(self, X, Y):
        return 5.0 * np.sin(np.asarray(X) / 10.0) * np.cos(np.asarray(Y) / 10.0)


class TriangleMesh:
    """Small dependency-free triangle-surface backend for ray casting.

    ``vertices`` is ``(N, 3)`` and ``faces`` is ``(M, 3)`` integer indices.
    It is intended for a narrow room/scene mesh; callers can load OBJ/PLY
    files with their preferred library and pass the resulting arrays here.
    """

    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if len(self.vertices) == 0 or len(self.faces) == 0:
            raise ValueError("TriangleMesh requires non-empty vertices and faces")
        if self.faces.min() < 0 or self.faces.max() >= len(self.vertices):
            raise ValueError("TriangleMesh face index is out of range")

    def intersect_ray(self, origin, direction, max_dist=1000.0):
        """Return the nearest positive ray/triangle hit or ``None``."""
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        direction = np.asarray(direction, dtype=np.float64).reshape(3)
        nearest_t, nearest_point = float(max_dist), None
        epsilon = 1e-9
        for face in self.faces:
            v0, v1, v2 = self.vertices[face]
            edge1, edge2 = v1 - v0, v2 - v0
            h = np.cross(direction, edge2)
            determinant = float(np.dot(edge1, h))
            if abs(determinant) < epsilon:
                continue
            inv_det = 1.0 / determinant
            s = origin - v0
            u = inv_det * float(np.dot(s, h))
            if u < 0.0 or u > 1.0:
                continue
            q = np.cross(s, edge1)
            v = inv_det * float(np.dot(direction, q))
            if v < 0.0 or u + v > 1.0:
                continue
            distance = inv_det * float(np.dot(edge2, q))
            if epsilon < distance <= nearest_t:
                nearest_t = distance
                nearest_point = origin + distance * direction
        return None if nearest_point is None else (nearest_point, nearest_t)

def _signed_height(cam_pos, ray_dir, distance, grid_map):
    p = cam_pos + distance * ray_dir
    return float(p[2] - grid_map.get_elevation(p[0], p[1]))

def intersect_ray_with_grid_result(cam_pos, ray_dir, grid_map, max_dist=1000.0,
                                    step=2.0, bisection_iterations=24):
    cam_pos = np.asarray(cam_pos, dtype=np.float64).reshape(3)
    ray_dir = np.asarray(ray_dir, dtype=np.float64).reshape(3)
    norm = np.linalg.norm(ray_dir)
    if norm < 1e-12: return RayHit(False, status="degenerate_ray")
    ray_dir /= norm
    if hasattr(grid_map, "intersect_ray"):
        hit = grid_map.intersect_ray(cam_pos, ray_dir, max_dist=max_dist)
        if hit is None:
            return RayHit(False, status="no_mesh_intersection")
        point, distance = hit
        return RayHit(True, np.asarray(point, dtype=np.float64), float(distance), "valid_mesh")
    initial = _signed_height(cam_pos, ray_dir, 0.0, grid_map)
    if ray_dir[2] >= 0 and initial > 0: return RayHit(False, status="ray_points_up")
    if initial <= 0: return RayHit(True, cam_pos.copy(), 0.0, "camera_below_surface")
    previous_t, previous_diff = 0.0, initial
    distance = float(step)
    while distance <= max_dist:
        current_diff = _signed_height(cam_pos, ray_dir, distance, grid_map)
        if current_diff <= 0:
            lo, hi = previous_t, distance
            for _ in range(max(1, int(bisection_iterations))):
                mid = 0.5 * (lo + hi)
                if _signed_height(cam_pos, ray_dir, mid, grid_map) > 0: lo = mid
                else: hi = mid
            hit_distance = 0.5 * (lo + hi)
            return RayHit(True, cam_pos + hit_distance * ray_dir, hit_distance, "valid")
        previous_t, previous_diff = distance, current_diff
        distance += step
    return RayHit(False, status="no_intersection")

def intersect_ray_with_grid(cam_pos, ray_dir, grid_map, max_dist=1000.0,
                            step=2.0, bisection_iterations=24):
    result = intersect_ray_with_grid_result(cam_pos, ray_dir, grid_map, max_dist, step, bisection_iterations)
    return result.point if result.hit else None

def bcc_drop_filter(points_3d, dc_ratio=0.15):
    from point_filter import bcc_drop_filter as robust_wrapper
    return robust_wrapper(points_3d, dc_ratio)
