"""PyTorch components for monocular depth based fire localization.

The module contains three independent building blocks:

* :class:`DepthBasedLocator` converts a 2-D fire-base pixel to a 3-D camera
  coordinate using a Depth Anything V2 model and camera intrinsics.
* :class:`DifferentiableRayCastLayer` intersects a pixel ray with a plane while
  keeping the complete operation in the autograd graph.
* :class:`VerticalityPriorLoss` adds a soft vertical-plume prior to a 3-D loss.

The default Hugging Face Depth Anything V2 checkpoints predict *relative*
depth.  A metric checkpoint or a calibration transform must therefore be
provided when metric Z coordinates are required.  ``depth_scale`` and
``depth_offset`` implement the common affine calibration case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping, Optional, Protocol, Sequence, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:  # NumPy is only needed for accepting an ndarray image.
    import numpy as np
except ImportError:  # pragma: no cover - PyTorch-only installations can still use tensors/PIL.
    np = None  # type: ignore[assignment]


# Matrix arguments are accepted as nested Python sequences as well as tensors;
# ``Any`` in the second arm keeps the alias accurate for both vectors and
# matrices while preserving useful annotations at every API boundary.
TensorLike = Union[Tensor, Sequence[Any]]
RGBInput = Any
OutOfBoundsMode = Literal["raise", "clamp", "return_none"]
InvalidDepthMode = Literal["raise", "nearest_valid", "return_none"]
ParallelMode = Literal["raise", "clamp", "nan"]
Reduction = Literal["none", "mean", "sum"]


class ImageProcessor(Protocol):
    """Small protocol covering the Hugging Face image processor API."""

    def __call__(self, *, images: Any, return_tensors: str) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class LocationResult:
    """Optional detailed result returned by :meth:`DepthBasedLocator.locate_with_details`."""

    point_3d: Tensor
    depth: Tensor
    pixel: Tensor
    depth_map: Tensor


class DepthBasedLocator(nn.Module):
    """Lift a 2-D pixel to a 3-D camera coordinate with Depth Anything V2.

    Parameters
    ----------
    model:
        A pre-trained Depth Anything V2 PyTorch module.  The model is put in
        evaluation mode, but its parameters are not frozen permanently, so the
        caller can still fine-tune it if desired.
    K_matrix:
        Camera intrinsic matrix with shape ``(3, 3)``.  The inverse is stored
        as a buffer and moved together with this module.
    processor:
        Optional Hugging Face ``AutoImageProcessor`` (or a compatible callable).
        If omitted, a simple ``[0, 1]`` RGB tensor conversion is used; a real
        Depth Anything checkpoint should normally be used with its processor.
    preprocess:
        Optional replacement for the simple conversion when ``processor`` is
        omitted.  It receives a PIL RGB image and returns ``(3,H,W)`` or
        ``(1,3,H,W)``.
    depth_scale, depth_offset:
        Affine calibration applied to the raw depth map,
        ``Z = raw_depth * depth_scale + depth_offset``.  Use calibration from
        known camera distances, or use a metric-depth checkpoint.
    depth_transform:
        Optional callable applied after the affine calibration.  It is useful
        for a learned/nonlinear metric calibration.
    out_of_bounds:
        Behavior for a pixel outside the image: raise an informative error,
        clamp it to the closest image border, or return ``None``.
    invalid_depth:
        Behavior when the sampled depth is NaN/Inf (and, by default, when it is
        non-positive): raise, search the nearest finite pixel, or return None.
    """

    def __init__(
        self,
        model: nn.Module,
        K_matrix: TensorLike,
        processor: Optional[ImageProcessor] = None,
        preprocess: Optional[Callable[[Any], Tensor]] = None,
        *,
        depth_scale: float = 1.0,
        depth_offset: float = 0.0,
        depth_transform: Optional[Callable[[Tensor], Tensor]] = None,
        out_of_bounds: OutOfBoundsMode = "raise",
        invalid_depth: InvalidDepthMode = "raise",
        max_valid_search_radius: int = 5,
        require_positive_depth: bool = True,
        use_amp: bool = True,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        if out_of_bounds not in ("raise", "clamp", "return_none"):
            raise ValueError(f"Unknown out_of_bounds mode: {out_of_bounds}")
        if invalid_depth not in ("raise", "nearest_valid", "return_none"):
            raise ValueError(f"Unknown invalid_depth mode: {invalid_depth}")
        if max_valid_search_radius < 0:
            raise ValueError("max_valid_search_radius must be non-negative")
        if depth_transform is not None and not callable(depth_transform):
            raise TypeError("depth_transform must be callable")

        self.model = model
        self.processor = processor
        self.preprocess = preprocess
        self.depth_scale = float(depth_scale)
        self.depth_offset = float(depth_offset)
        self.depth_transform = depth_transform
        self.out_of_bounds = out_of_bounds
        self.invalid_depth = invalid_depth
        self.max_valid_search_radius = int(max_valid_search_radius)
        self.require_positive_depth = bool(require_positive_depth)
        self.use_amp = bool(use_amp)

        inferred_device = self._module_device(model)
        self.inference_device = torch.device(device) if device is not None else inferred_device
        self.model.to(self.inference_device)
        self.model.eval()

        K = torch.as_tensor(K_matrix, device=self.inference_device)
        if not K.is_floating_point():
            K = K.float()
        if K.shape != (3, 3):
            raise ValueError(f"K_matrix must have shape (3, 3), got {tuple(K.shape)}")
        if not torch.isfinite(K).all():
            raise ValueError("K_matrix contains NaN or Inf")
        # linalg.inv is differentiable, although K is normally a fixed buffer.
        self.register_buffer("K_inv", torch.linalg.inv(K), persistent=True)

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            try:
                return next(module.buffers()).device
            except StopIteration:
                return torch.device("cpu")

    @classmethod
    def from_huggingface(
        cls,
        K_matrix: TensorLike,
        model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
        *,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> "DepthBasedLocator":
        """Construct a locator from a Hugging Face Depth Anything V2 checkpoint.

        ``transformers`` is imported lazily so the rest of this module remains
        usable in deployments that inject an already-loaded model.
        """

        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as exc:  # pragma: no cover - depends on deployment environment.
            raise RuntimeError(
                "from_huggingface requires `transformers`; install it or inject model and processor directly."
            ) from exc
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModelForDepthEstimation.from_pretrained(model_name)
        return cls(
            model=model,
            K_matrix=K_matrix,
            processor=processor,
            device=device,
            **kwargs,
        )

    @staticmethod
    def _to_pil_rgb(image: RGBInput) -> Any:
        """Convert PIL/NumPy/Tensor input to a PIL RGB image."""

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - PIL is a normal HF dependency.
            raise RuntimeError("PIL is required to convert RGB input") from exc

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, Tensor):
            x = image.detach().cpu()
            if x.ndim != 3:
                raise ValueError("RGB tensor must have shape (3,H,W) or (H,W,3)")
            # Prefer the unambiguous HWC representation when the last axis is
            # RGB; otherwise accept CHW.  (H=3 and CHW with W=3 are inherently
            # ambiguous, so callers should use a non-square/explicit layout.)
            if x.shape[-1] != 3 and x.shape[0] in (1, 3):
                x = x.permute(1, 2, 0)
            if x.shape[-1] != 3:
                raise ValueError("RGB tensor must have exactly three channels")
            if x.is_floating_point():
                # Accept either [0,1] or [0,255] floating point images.
                max_value = float(x.nan_to_num(nan=0.0).max()) if x.numel() else 0.0
                if max_value <= 1.5:
                    x = x * 255.0
                x = x.clamp(0.0, 255.0)
            else:
                x = x.clamp(0, 255)
            return Image.fromarray(x.to(torch.uint8).numpy(), mode="RGB")

        if np is not None and isinstance(image, np.ndarray):
            x_np = image
            if x_np.ndim != 3:
                raise ValueError("RGB ndarray must have shape (H,W,3) or (3,H,W)")
            if x_np.shape[0] in (1, 3) and x_np.shape[-1] != 3:
                x_np = np.transpose(x_np, (1, 2, 0))
            if x_np.shape[-1] != 3:
                raise ValueError("RGB ndarray must have exactly three channels")
            if np.issubdtype(x_np.dtype, np.floating):
                x_np = x_np * 255.0 if float(np.nanmax(x_np)) <= 1.5 else x_np
            x_np = np.nan_to_num(x_np, nan=0.0).clip(0.0, 255.0).astype("uint8")
            return Image.fromarray(x_np, mode="RGB")

        raise TypeError("rgb must be a PIL image, a NumPy ndarray, or a 3-D torch.Tensor")

    def _default_preprocess(self, image: Any) -> Tensor:
        if np is None:  # pragma: no cover
            raise RuntimeError("NumPy is required for the default image conversion")
        array = np.asarray(image, dtype="float32") / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _prepare_inputs(self, image: Any) -> Mapping[str, Any]:
        if self.processor is not None:
            encoded = self.processor(images=image, return_tensors="pt")
            if not isinstance(encoded, Mapping):
                raise TypeError("processor must return a mapping such as BatchFeature")
            return {
                key: value.to(self.inference_device)
                if isinstance(value, Tensor)
                else value
                for key, value in encoded.items()
            }

        tensor = self.preprocess(image) if self.preprocess is not None else self._default_preprocess(image)
        if not isinstance(tensor, Tensor):
            raise TypeError("preprocess must return a torch.Tensor")
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[1] != 3:
            raise ValueError("preprocess must return (3,H,W) or (1,3,H,W)")
        return {"pixel_values": tensor.to(self.inference_device)}

    @staticmethod
    def _extract_depth_tensor(outputs: Any) -> Tensor:
        """Extract ``predicted_depth`` from common HF/custom output formats."""

        if isinstance(outputs, Tensor):
            return outputs
        if isinstance(outputs, Mapping):
            for key in ("predicted_depth", "depth", "depth_map", "logits"):
                if key in outputs and isinstance(outputs[key], Tensor):
                    return outputs[key]
            for value in outputs.values():
                if isinstance(value, Tensor) and value.ndim >= 2:
                    return value
        if hasattr(outputs, "predicted_depth"):
            candidate = getattr(outputs, "predicted_depth")
            if isinstance(candidate, Tensor):
                return candidate
        if isinstance(outputs, (tuple, list)):
            for value in outputs:
                try:
                    return DepthBasedLocator._extract_depth_tensor(value)
                except (TypeError, ValueError):
                    continue
        raise TypeError("Could not find a depth tensor in model output")

    @staticmethod
    def _as_single_depth_map(depth: Tensor) -> Tensor:
        if depth.ndim == 4:
            if depth.shape[0] < 1:
                raise ValueError("Depth output has an empty batch")
            depth = depth[0]
        if depth.ndim == 3:
            if depth.shape[0] == 1:
                depth = depth[0]
            elif depth.shape[-1] == 1:
                depth = depth[..., 0]
            else:
                raise ValueError(f"Expected one depth channel, got shape {tuple(depth.shape)}")
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2-D depth map, got shape {tuple(depth.shape)}")
        return depth

    def _postprocess_depth(self, outputs: Any, image_hw: tuple[int, int]) -> Optional[Tensor]:
        postprocess = getattr(self.processor, "post_process_depth_estimation", None)
        if postprocess is None:
            return None
        try:
            processed = postprocess(outputs, target_sizes=[image_hw])
            if isinstance(processed, (tuple, list)) and processed:
                first = processed[0]
                if isinstance(first, Mapping):
                    for key in ("predicted_depth", "depth"):
                        if key in first and isinstance(first[key], Tensor):
                            return self._as_single_depth_map(first[key])
        except (AttributeError, KeyError, IndexError, TypeError, ValueError):
            # Some custom processors expose the method but use a different
            # output type; the direct tensor extraction below is still valid.
            return None
        return None

    def estimate_depth(self, rgb: RGBInput) -> Tensor:
        """Run Depth Anything V2 and return a calibrated ``(H,W)`` depth map."""

        image = self._to_pil_rgb(rgb)
        width, height = image.size
        inputs = self._prepare_inputs(image)
        autocast_enabled = self.use_amp and self.inference_device.type == "cuda"
        with torch.inference_mode():
            if autocast_enabled:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)

        depth = self._postprocess_depth(outputs, (height, width))
        if depth is None:
            depth = self._as_single_depth_map(self._extract_depth_tensor(outputs))
            depth = depth.float()
            if tuple(depth.shape) != (height, width):
                depth = F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
        else:
            depth = depth.float()
            if tuple(depth.shape) != (height, width):
                depth = F.interpolate(
                    depth.unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

        depth = depth * self.depth_scale + self.depth_offset
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
        if not isinstance(depth, Tensor) or depth.ndim != 2:
            raise TypeError("depth_transform must return a 2-D torch.Tensor")
        if depth.shape[0] == 0 or depth.shape[1] == 0:
            raise ValueError("Depth model returned an empty depth map")
        return depth

    @staticmethod
    def _sample_bilinear(depth_map: Tensor, pixel: Tensor) -> Tensor:
        height, width = depth_map.shape
        if width > 1:
            x_norm = 2.0 * pixel[0] / float(width - 1) - 1.0
        else:
            x_norm = torch.zeros((), device=depth_map.device, dtype=depth_map.dtype)
        if height > 1:
            y_norm = 2.0 * pixel[1] / float(height - 1) - 1.0
        else:
            y_norm = torch.zeros((), device=depth_map.device, dtype=depth_map.dtype)
        grid = torch.stack((x_norm, y_norm)).reshape(1, 1, 1, 2)
        return F.grid_sample(
            depth_map.reshape(1, 1, height, width),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).reshape(())

    def _nearest_valid_depth(self, depth_map: Tensor, pixel: Tensor) -> Optional[Tensor]:
        height, width = depth_map.shape
        center_x = int(torch.round(pixel[0]).item())
        center_y = int(torch.round(pixel[1]).item())
        best: Optional[tuple[float, Tensor]] = None
        for radius in range(self.max_valid_search_radius + 1):
            x0, x1 = max(0, center_x - radius), min(width, center_x + radius + 1)
            y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            patch = depth_map[y0:y1, x0:x1]
            valid = torch.isfinite(patch)
            if self.require_positive_depth:
                valid = valid & (patch > 0)
            indices = torch.nonzero(valid, as_tuple=False)
            if indices.numel() == 0:
                continue
            distances = (indices[:, 1].to(pixel.dtype) + x0 - pixel[0]).square()
            distances = distances + (indices[:, 0].to(pixel.dtype) + y0 - pixel[1]).square()
            min_idx = int(torch.argmin(distances).item())
            value = patch[indices[min_idx, 0], indices[min_idx, 1]]
            best = (float(distances[min_idx].item()), value)
            break
        return None if best is None else best[1]

    def _resolve_depth(self, depth_map: Tensor, pixel: Tensor) -> Optional[Tensor]:
        sampled = self._sample_bilinear(depth_map, pixel)
        is_valid = bool(torch.isfinite(sampled).item())
        if self.require_positive_depth:
            is_valid = is_valid and bool((sampled > 0).item())
        if is_valid:
            return sampled

        if self.invalid_depth == "return_none":
            return None
        if self.invalid_depth == "nearest_valid":
            fallback = self._nearest_valid_depth(depth_map, pixel)
            if fallback is None:
                raise ValueError(
                    "Depth is invalid and no finite positive fallback was found "
                    f"within radius {self.max_valid_search_radius}"
                )
            return fallback
        raise ValueError(f"Depth at pixel ({float(pixel[0]):.3f}, {float(pixel[1]):.3f}) is NaN/Inf/non-positive")

    def locate(self, rgb: RGBInput, point_2d: TensorLike) -> Optional[Tensor]:
        """Return ``(X,Y,Z)`` in the camera frame for a fire-base pixel.

        Pixel coordinates use the image convention ``u=x`` (column) and
        ``v=y`` (row).  Fractional pixels are bilinearly sampled.  A ``None``
        result is ``None`` when a ``"return_none"`` mode is selected.  With
        ``invalid_depth="nearest_valid"``, an error is raised if the complete
        search window contains no valid depth sample.
        """

        result = self.locate_with_details(rgb, point_2d)
        return None if result is None else result.point_3d

    def locate_with_details(self, rgb: RGBInput, point_2d: TensorLike) -> Optional[LocationResult]:
        depth_map = self.estimate_depth(rgb)
        if not isinstance(point_2d, Tensor):
            pixel = torch.as_tensor(point_2d, device=depth_map.device, dtype=depth_map.dtype)
        else:
            pixel = point_2d.to(device=depth_map.device, dtype=depth_map.dtype)
        if pixel.shape != (2,):
            raise ValueError(f"point_2d must have shape (2,), got {tuple(pixel.shape)}")
        if not torch.isfinite(pixel).all():
            raise ValueError("point_2d contains NaN or Inf")

        height, width = depth_map.shape
        inside = bool(
            ((pixel[0] >= 0) & (pixel[0] <= width - 1) & (pixel[1] >= 0) & (pixel[1] <= height - 1)).item()
        )
        if not inside:
            if self.out_of_bounds == "return_none":
                return None
            if self.out_of_bounds == "raise":
                raise ValueError(
                    f"point_2d ({float(pixel[0]):.3f}, {float(pixel[1]):.3f}) is outside image width={width}, height={height}"
                )
            pixel = torch.stack(
                (
                    pixel[0].clamp(0, max(width - 1, 0)),
                    pixel[1].clamp(0, max(height - 1, 0)),
                )
            )

        depth = self._resolve_depth(depth_map, pixel)
        if depth is None:
            return None
        K_inv = self.K_inv.to(device=depth_map.device, dtype=depth_map.dtype)
        pixel_h = torch.cat((pixel, torch.ones(1, device=depth_map.device, dtype=depth_map.dtype)))
        ray = K_inv @ pixel_h
        if not torch.isfinite(ray).all() or abs(float(ray[2].item())) < 1e-8:
            raise ValueError("K^-1 [u,v,1] has an invalid/zero third component")
        ray = ray / ray[2]
        point_3d = ray * depth
        if not torch.isfinite(point_3d).all():
            raise ValueError("Computed 3-D point contains NaN or Inf")
        return LocationResult(point_3d=point_3d, depth=depth, pixel=pixel, depth_map=depth_map)


class DifferentiableRayCastLayer(nn.Module):
    """Differentiable pixel-ray/plane intersection.

    The extrinsic convention is ``X_camera = R @ X_world + T``.  Therefore the
    returned point is in world coordinates, the camera center is
    ``C_world = -R.T @ T``, and the ray direction is
    ``R.T @ K^{-1} [u,v,1]``.  Supplying identity ``R`` and zero ``T`` gives a
    camera-frame plane intersection.
    """

    def __init__(self, *, eps: float = 1e-8, parallel_mode: ParallelMode = "raise") -> None:
        super().__init__()
        if eps <= 0:
            raise ValueError("eps must be positive")
        if parallel_mode not in ("raise", "clamp", "nan"):
            raise ValueError(f"Unknown parallel_mode: {parallel_mode}")
        self.eps = float(eps)
        self.parallel_mode = parallel_mode

    @staticmethod
    def _as_tensor(value: TensorLike, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(device=device, dtype=dtype)
        return torch.as_tensor(value, device=device, dtype=dtype)

    @staticmethod
    def _broadcast_shapes(
        point: Tensor, K: Tensor, R: Tensor, T: Tensor, plane: Tensor
    ) -> tuple[int, ...]:
        try:
            return torch.broadcast_shapes(
                point.shape[:-1], K.shape[:-2], R.shape[:-2], T.shape[:-1], plane.shape[:-1]
            )
        except RuntimeError as exc:
            raise ValueError(
                "Leading dimensions of point_2d, K_matrix, R_matrix, T_matrix and plane_eq are not broadcastable"
            ) from exc

    def forward(
        self,
        point_2d: TensorLike,
        K_matrix: TensorLike,
        R_matrix: TensorLike,
        T_matrix: TensorLike,
        plane_eq: TensorLike,
    ) -> Tensor:
        """Intersect one or many rays with ``aX+bY+cZ+d=0``.

        All operations are tensor operations.  In particular, avoid wrapping an
        existing tensor in ``torch.tensor(...)`` at call sites: this layer keeps
        ``point_2d`` connected to its upstream ROI Refiner graph.
        """

        point_raw = point_2d if isinstance(point_2d, Tensor) else torch.as_tensor(point_2d)
        dtype = point_raw.dtype if point_raw.is_floating_point() else torch.get_default_dtype()
        device = point_raw.device
        point = point_raw.to(device=device, dtype=dtype)
        K = self._as_tensor(K_matrix, device=device, dtype=dtype)
        R = self._as_tensor(R_matrix, device=device, dtype=dtype)
        T = self._as_tensor(T_matrix, device=device, dtype=dtype)
        plane = self._as_tensor(plane_eq, device=device, dtype=dtype)
        # Accept both the common vector form (...,3) and a column-vector
        # extrinsic translation (...,3,1).
        if T.ndim >= 2 and T.shape[-2:] == (3, 1):
            T = T.squeeze(-1)
        if point.ndim >= 2 and point.shape[-2:] == (2, 1):
            point = point.squeeze(-1)
        if point.ndim < 1 or point.shape[-1] != 2:
            raise ValueError(f"point_2d must end in dimension 2, got {tuple(point.shape)}")
        if K.ndim < 2 or K.shape[-2:] != (3, 3):
            raise ValueError(f"K_matrix must end in (3,3), got {tuple(K.shape)}")
        if R.ndim < 2 or R.shape[-2:] != (3, 3):
            raise ValueError(f"R_matrix must end in (3,3), got {tuple(R.shape)}")
        if T.ndim < 1 or T.shape[-1] != 3:
            raise ValueError(f"T_matrix must end in dimension 3, got {tuple(T.shape)}")
        if plane.ndim < 1 or plane.shape[-1] != 4:
            raise ValueError(f"plane_eq must end in dimension 4, got {tuple(plane.shape)}")

        batch_shape = self._broadcast_shapes(point, K, R, T, plane)
        point = point.expand(*batch_shape, 2)
        K = K.expand(*batch_shape, 3, 3)
        R = R.expand(*batch_shape, 3, 3)
        T = T.expand(*batch_shape, 3)
        plane = plane.expand(*batch_shape, 4)

        ones = torch.ones(*batch_shape, 1, device=device, dtype=dtype)
        pixel_h = torch.cat((point, ones), dim=-1).unsqueeze(-1)
        # solve(K, p) is stable and differentiable with respect to both K and p.
        ray_camera = torch.linalg.solve(K, pixel_h).squeeze(-1)
        R_transpose = R.transpose(-1, -2)
        ray_world = (R_transpose @ ray_camera.unsqueeze(-1)).squeeze(-1)
        camera_center = -(R_transpose @ T.unsqueeze(-1)).squeeze(-1)

        normal = plane[..., :3]
        offset = plane[..., 3]
        denominator = (normal * ray_world).sum(dim=-1)
        parallel = denominator.abs() < self.eps
        if self.parallel_mode == "raise" and bool(parallel.any().detach().cpu()):
            raise ValueError("Ray is parallel to the plane (|n dot direction| < eps)")
        if self.parallel_mode == "nan":
            safe_denominator = torch.where(parallel, torch.full_like(denominator, float("nan")), denominator)
        else:
            sign = torch.where(denominator < 0, -torch.ones_like(denominator), torch.ones_like(denominator))
            safe_denominator = torch.where(parallel, sign * self.eps, denominator)

        numerator = (normal * camera_center).sum(dim=-1) + offset
        distance = -numerator / safe_denominator
        return camera_center + distance.unsqueeze(-1) * ray_world


class VerticalityPriorLoss(nn.Module):
    """Penalize horizontal displacement between fire base and fire top.

    ``wind_strength`` is a non-negative, normalized indicator (zero means no
    wind).  The effective penalty is
    ``adaptive_weight / (1 + wind_relaxation * wind_strength)``; strong wind
    consequently relaxes the prior instead of forcing a physically implausible
    perfectly vertical plume.
    """

    def __init__(
        self,
        *,
        adaptive_weight: float = 1.0,
        ratio_weight: float = 1.0,
        cosine_weight: float = 0.1,
        upward_weight: float = 0.1,
        wind_relaxation: float = 1.0,
        eps: float = 1e-6,
        reduction: Reduction = "mean",
    ) -> None:
        super().__init__()
        if adaptive_weight < 0 or ratio_weight < 0 or cosine_weight < 0 or upward_weight < 0:
            raise ValueError("Loss weights must be non-negative")
        if wind_relaxation < 0 or eps <= 0:
            raise ValueError("wind_relaxation must be non-negative and eps must be positive")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"Unknown reduction: {reduction}")
        self.adaptive_weight = float(adaptive_weight)
        self.ratio_weight = float(ratio_weight)
        self.cosine_weight = float(cosine_weight)
        self.upward_weight = float(upward_weight)
        self.wind_relaxation = float(wind_relaxation)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(
        self,
        base_xyz: Tensor,
        top_xyz: Tensor,
        wind_strength: Optional[TensorLike] = None,
    ) -> Tensor:
        if not isinstance(base_xyz, Tensor) or not isinstance(top_xyz, Tensor):
            raise TypeError("base_xyz and top_xyz must be torch.Tensor objects")
        if base_xyz.shape != top_xyz.shape or base_xyz.shape[-1] != 3:
            raise ValueError("base_xyz and top_xyz must have equal shape (...,3)")
        if not base_xyz.is_floating_point() or not top_xyz.is_floating_point():
            raise TypeError("3-D coordinates must use a floating point dtype")

        delta = top_xyz - base_xyz
        horizontal = torch.linalg.vector_norm(delta[..., :2], dim=-1)
        vertical_abs = delta[..., 2].abs()
        # Ratio directly expresses “horizontal displacement relative to height”.
        ratio_penalty = (horizontal / (vertical_abs + self.eps)).square()

        delta_norm = torch.linalg.vector_norm(delta, dim=-1)
        cos_to_up = delta[..., 2] / (delta_norm + self.eps)
        cosine_penalty = 1.0 - cos_to_up
        downward_penalty = F.relu(-delta[..., 2]).square()
        raw = (
            self.ratio_weight * ratio_penalty
            + self.cosine_weight * cosine_penalty
            + self.upward_weight * downward_penalty
        )

        if wind_strength is None:
            effective_weight = torch.as_tensor(self.adaptive_weight, device=raw.device, dtype=raw.dtype)
        else:
            wind = wind_strength if isinstance(wind_strength, Tensor) else torch.as_tensor(wind_strength)
            wind = wind.to(device=raw.device, dtype=raw.dtype).clamp_min(0.0)
            effective_weight = self.adaptive_weight / (1.0 + self.wind_relaxation * wind)
        value = raw * effective_weight
        if self.reduction == "none":
            return value
        if self.reduction == "sum":
            return value.sum()
        return value.mean()


__all__ = [
    "DepthBasedLocator",
    "DifferentiableRayCastLayer",
    "VerticalityPriorLoss",
    "LocationResult",
]

