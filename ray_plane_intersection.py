"""Vectorized ray/plane intersection for monocular 3-D localization.

The default extrinsic convention is the OpenCV convention::

    X_camera = R @ X_world + T

With that convention the camera centre in world coordinates is
``C_world = -R.T @ T`` and a camera-frame ray direction is rotated with
``R.T``.  If the calibration stores a camera-to-world pose instead, pass
``extrinsic_convention="camera_to_world"``; then ``T`` is the camera centre
and ``R`` rotates camera vectors into world coordinates.

The implementation contains no per-pixel Python loop.  All pixels and all
matrix products are evaluated as batched PyTorch operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple, Union

import torch


TensorLike = Union[
    torch.Tensor,
    Sequence[float],
    Sequence[Sequence[float]],
    Sequence[Sequence[Sequence[float]]],
]
ExtrinsicConvention = Literal["world_to_camera", "camera_to_world"]


class RayPlaneIntersectionError(ValueError):
    """Raised when strict mode finds one or more rays without a valid hit."""


@dataclass(frozen=True)
class RayPlaneIntersectionResult:
    """Output of :meth:`RayPlaneIntersectionLocator.locate`.

    Every tensor has a leading batch dimension ``N``.  ``points_world`` is
    ``NaN`` for an invalid ray, so callers can safely select it with
    ``valid_mask`` without accidentally using a point behind the camera.

    ``status`` uses the following integer codes:

    * ``0``: valid intersection;
    * ``1``: ray parallel to the plane (including a ray contained in it);
    * ``2``: intersection is behind the camera (negative ray parameter);
    * ``3``: invalid input or a degenerate plane normal.
    """

    points_world: torch.Tensor
    valid_mask: torch.Tensor
    parallel_mask: torch.Tensor
    behind_camera_mask: torch.Tensor
    invalid_input_mask: torch.Tensor
    degenerate_plane_mask: torch.Tensor
    ray_parameter: torch.Tensor
    ray_origins_world: torch.Tensor
    ray_directions_world: torch.Tensor
    status: torch.Tensor


class RayPlaneIntersectionLocator:
    """Locate image pixels by intersecting their viewing rays with a plane.

    Parameters
    ----------
    plane:
        Plane coefficients ``(a, b, c, d)`` for
        ``a*X + b*Y + c*Z + d = 0``.  The default ``(0, 0, 1, 0)`` is the
        world floor ``Z = 0``.
    extrinsic_convention:
        ``"world_to_camera"`` means ``X_c = R @ X_w + T`` (OpenCV default).
        ``"camera_to_world"`` means ``X_w = R @ X_c + T``.
    eps:
        Numerical tolerance for a parallel ray.  Plane coefficients are
        normalized internally, so
        this tolerance is independent of the scale of ``(a, b, c, d)``.
    dtype, device:
        Working PyTorch dtype/device.  Use ``torch.float64`` for calibration
        or high-precision offline evaluation, and ``device="cuda"`` for GPU
        batches.  All inputs are moved to this common dtype/device.
    """

    STATUS_VALID = 0
    STATUS_PARALLEL = 1
    STATUS_BEHIND_CAMERA = 2
    STATUS_INVALID = 3

    def __init__(
        self,
        plane: TensorLike = (0.0, 0.0, 1.0, 0.0),
        *,
        extrinsic_convention: ExtrinsicConvention = "world_to_camera",
        eps: float = 1e-8,
        dtype: torch.dtype = torch.float64,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        if extrinsic_convention not in ("world_to_camera", "camera_to_world"):
            raise ValueError(
                "extrinsic_convention must be 'world_to_camera' or 'camera_to_world'"
            )
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("dtype must be a floating point torch.dtype")

        self.extrinsic_convention = extrinsic_convention
        self.eps = float(eps)
        self.dtype = dtype
        self.device = torch.device(device) if device is not None else None

        plane_tensor = torch.as_tensor(plane, dtype=dtype, device=self.device)
        if plane_tensor.ndim != 1 or plane_tensor.shape[0] != 4:
            raise ValueError("plane must have shape (4,), containing (a, b, c, d)")
        self._default_plane = plane_tensor.clone()

    @staticmethod
    def _batch_matrices(
        value: TensorLike,
        name: str,
        batch_size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Canonicalize one ``3x3`` matrix to shape ``(N, 3, 3)``."""

        matrix = torch.as_tensor(value, dtype=dtype, device=device)
        if matrix.ndim == 2 and tuple(matrix.shape) == (3, 3):
            matrix = matrix.unsqueeze(0)
        elif matrix.ndim != 3 or tuple(matrix.shape[-2:]) != (3, 3):
            raise ValueError(f"{name} must have shape (3, 3) or (N, 3, 3)")

        if matrix.shape[0] == 1:
            matrix = matrix.expand(batch_size, -1, -1)
        elif matrix.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension is {matrix.shape[0]}, expected 1 or {batch_size}"
            )
        return matrix

    @staticmethod
    def _batch_vectors(
        value: TensorLike,
        name: str,
        batch_size: int,
        size: int,
        *,
        dtype: torch.dtype,
        device: torch.device,
        allow_column: bool = True,
    ) -> torch.Tensor:
        """Canonicalize vectors/column vectors to shape ``(N, size)``."""

        vector = torch.as_tensor(value, dtype=dtype, device=device)
        if vector.ndim == 1 and vector.shape[0] == size:
            vector = vector.unsqueeze(0)
        elif (
            allow_column
            and vector.ndim == 2
            and tuple(vector.shape) == (size, 1)
        ):
            vector = vector[:, 0].unsqueeze(0)
        elif vector.ndim == 2 and vector.shape[-1] == size:
            pass
        elif (
            allow_column
            and vector.ndim == 3
            and vector.shape[-2:] == (size, 1)
        ):
            vector = vector[..., 0]
        else:
            expected = f"({size},), (N, {size})"
            if allow_column:
                expected += f", ({size}, 1), or (N, {size}, 1)"
            raise ValueError(f"{name} must have shape {expected}")

        if vector.shape[0] == 1:
            vector = vector.expand(batch_size, -1)
        elif vector.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension is {vector.shape[0]}, expected 1 or {batch_size}"
            )
        return vector

    def _world_rays(
        self,
        pixels: TensorLike,
        K: TensorLike,
        R: TensorLike,
        T: TensorLike,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(origins, directions, finite_input_mask)`` for a pixel batch."""

        pixels_tensor = torch.as_tensor(pixels, dtype=self.dtype, device=self.device)
        if pixels_tensor.ndim == 1 and pixels_tensor.shape[0] == 2:
            pixels_tensor = pixels_tensor.unsqueeze(0)
        elif pixels_tensor.ndim != 2 or pixels_tensor.shape[-1] != 2:
            raise ValueError("pixels must have shape (2,) or (N, 2), containing (u, v)")

        batch_size = pixels_tensor.shape[0]
        K_batch = self._batch_matrices(
            K, "K", batch_size, dtype=self.dtype, device=pixels_tensor.device
        )
        R_batch = self._batch_matrices(
            R, "R", batch_size, dtype=self.dtype, device=pixels_tensor.device
        )
        T_batch = self._batch_vectors(
            T,
            "T",
            batch_size,
            3,
            dtype=self.dtype,
            device=pixels_tensor.device,
        )

        # Homogeneous image coordinate p = (u, v, 1)^T, shape (N, 3, 1).
        ones = torch.ones((batch_size, 1), dtype=self.dtype, device=pixels_tensor.device)
        pixel_h = torch.cat((pixels_tensor, ones), dim=1)

        # K @ d_camera = p, hence d_camera = K^{-1} @ p.  ``solve`` is
        # numerically safer than explicitly forming K^{-1} and is batched.
        try:
            ray_camera = torch.linalg.solve(
                K_batch, pixel_h.unsqueeze(-1)
            ).squeeze(-1)
        except RuntimeError as exc:
            raise ValueError("K must be nonsingular for pixel-to-ray conversion") from exc

        if self.extrinsic_convention == "world_to_camera":
            # X_c = R @ X_w + T, so d_w = R^T @ d_c and
            # C_w = -R^T @ T.  bmm performs N independent matrix products.
            R_transpose = R_batch.transpose(1, 2)
            ray_world_raw = torch.bmm(
                R_transpose, ray_camera.unsqueeze(-1)
            ).squeeze(-1)
            origins_world = -torch.bmm(
                R_transpose, T_batch.unsqueeze(-1)
            ).squeeze(-1)
        else:
            # X_w = R @ X_c + T, so d_w = R @ d_c and C_w = T.
            ray_world_raw = torch.bmm(
                R_batch, ray_camera.unsqueeze(-1)
            ).squeeze(-1)
            origins_world = T_batch

        # Normalize the world direction.  The intersection point is invariant
        # to the scale of a direction, while normalization makes ray_parameter
        # equal to a physical distance when R is a true rotation.
        direction_norm = torch.linalg.vector_norm(ray_world_raw, dim=1, keepdim=True)
        ray_directions_world = ray_world_raw / direction_norm.clamp_min(
            torch.finfo(self.dtype).tiny
        )

        finite_input = (
            torch.isfinite(pixels_tensor).all(dim=1)
            & torch.isfinite(K_batch).all(dim=(1, 2))
            & torch.isfinite(R_batch).all(dim=(1, 2))
            & torch.isfinite(T_batch).all(dim=1)
            & torch.isfinite(ray_camera).all(dim=1)
            & torch.isfinite(origins_world).all(dim=1)
            & torch.isfinite(ray_directions_world).all(dim=1)
            & torch.isfinite(direction_norm[:, 0])
            & (direction_norm[:, 0] > self.eps)
        )
        return origins_world, ray_directions_world, finite_input

    def locate(
        self,
        pixels: TensorLike,
        K: TensorLike,
        R: TensorLike,
        T: TensorLike,
        plane: Optional[TensorLike] = None,
        *,
        raise_on_invalid: bool = False,
    ) -> RayPlaneIntersectionResult:
        """Intersect one or many image pixels with the configured plane.

        Parameters accept either one item or a batch.  ``K``, ``R``, ``T``
        and ``plane`` may be shared by the whole batch or supplied with a
        leading dimension ``N``.  No Python loop is used over that dimension.

        By default, parallel/behind-camera rays return ``valid_mask=False``.
        Set ``raise_on_invalid=True`` when an invalid ray should raise
        :class:`RayPlaneIntersectionError` instead.
        """

        # This first call also canonicalizes pixels and establishes N/device.
        origins, directions, finite_input = self._world_rays(pixels, K, R, T)
        batch_size = origins.shape[0]

        plane_value: TensorLike = self._default_plane if plane is None else plane
        plane_batch = self._batch_vectors(
            plane_value,
            "plane",
            batch_size,
            4,
            dtype=self.dtype,
            device=origins.device,
            # Accept both (a,b,c,d) and a conventional column vector
            # [[a], [b], [c], [d]] (also in batched form).
            allow_column=True,
        )

        normal = plane_batch[:, :3]
        plane_d = plane_batch[:, 3]
        normal_norm = torch.linalg.vector_norm(normal, dim=1)
        plane_ok = (
            torch.isfinite(plane_batch).all(dim=1)
            & torch.isfinite(normal_norm)
            & (normal_norm > self.eps)
        )

        # Normalize (a,b,c,d) by ||(a,b,c)||.  This keeps both the plane
        # equation and the parallel threshold invariant to coefficient scale.
        normal_unit = normal / normal_norm.clamp_min(torch.finfo(self.dtype).tiny).unsqueeze(1)
        d_unit = plane_d / normal_norm.clamp_min(torch.finfo(self.dtype).tiny)

        # For X(t) = C + t*d, n.X(t) + d_plane = 0 gives
        # t = -(n.C + d_plane) / (n.d).  Both dot products are batched.
        denominator = torch.sum(normal_unit * directions, dim=1)
        numerator = -(
            torch.sum(normal_unit * origins, dim=1) + d_unit
        )
        denominator_finite = torch.isfinite(denominator)
        intersectable = (
            finite_input
            & plane_ok
            & denominator_finite
            & (denominator.abs() > self.eps)
        )

        # Avoid NaN/Inf from dividing by zero even for rays that will be
        # marked invalid.  Their result is masked with NaN below.
        safe_denominator = torch.where(
            intersectable, denominator, torch.ones_like(denominator)
        )
        ray_parameter = numerator / safe_denominator
        parameter_finite = torch.isfinite(ray_parameter)

        parallel_mask = (
            finite_input
            & plane_ok
            & denominator_finite
            & (denominator.abs() <= self.eps)
        )
        # A ray is a half-line, so t < 0 is strictly behind its origin.
        # (t == 0 is the camera centre and is a valid mathematical hit.)
        behind_camera_mask = intersectable & parameter_finite & (ray_parameter < 0.0)
        valid_mask = intersectable & parameter_finite & (~behind_camera_mask)

        points_candidate = origins + ray_parameter.unsqueeze(1) * directions
        nan_points = torch.full_like(points_candidate, torch.nan)
        points_world = torch.where(
            valid_mask.unsqueeze(1), points_candidate, nan_points
        )

        invalid_input_mask = ~finite_input
        degenerate_plane_mask = ~plane_ok
        status = torch.full(
            (batch_size,),
            self.STATUS_INVALID,
            dtype=torch.int64,
            device=origins.device,
        )
        status = torch.where(parallel_mask, torch.full_like(status, self.STATUS_PARALLEL), status)
        status = torch.where(
            behind_camera_mask,
            torch.full_like(status, self.STATUS_BEHIND_CAMERA),
            status,
        )
        status = torch.where(valid_mask, torch.full_like(status, self.STATUS_VALID), status)

        if raise_on_invalid and (~valid_mask).any():
            raise RayPlaneIntersectionError(
                "Ray-plane intersection failed for "
                f"{int((~valid_mask).sum().item())}/{batch_size} rays "
                f"(parallel={int(parallel_mask.sum().item())}, "
                f"behind_camera={int(behind_camera_mask.sum().item())}, "
                f"invalid_input={int(invalid_input_mask.sum().item())}, "
                f"degenerate_plane={int(degenerate_plane_mask.sum().item())})"
            )

        return RayPlaneIntersectionResult(
            points_world=points_world,
            valid_mask=valid_mask,
            parallel_mask=parallel_mask,
            behind_camera_mask=behind_camera_mask,
            invalid_input_mask=invalid_input_mask,
            degenerate_plane_mask=degenerate_plane_mask,
            ray_parameter=ray_parameter,
            ray_origins_world=origins,
            ray_directions_world=directions,
            status=status,
        )

    __call__ = locate


__all__ = [
    "RayPlaneIntersectionError",
    "RayPlaneIntersectionLocator",
    "RayPlaneIntersectionResult",
]
