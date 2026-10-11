"""Online, guarded refinement of a calibrated floor homography.

The floor mapping is represented in the same direction as
``homography_floor.FloorHomography``:

    [u, v, 1]^T ~ H_floor_to_image [X, Y, 1]^T

At run time, measured image anchors and their known metric floor coordinates
can refine this map.  A small RANSAC loop protects the update from a bad
anchor, while reprojection and update-size gates prevent a plausible-looking
but geometrically unsafe calibration from replacing the last good one.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from homography_floor import FloorHomography, estimate_homography


ArrayLike2D = Sequence[Sequence[float]] | np.ndarray


@dataclass(frozen=True)
class HomographyUpdateResult:
    """Audit record returned for every online update attempt."""

    accepted: bool
    status: str
    frame_id: Optional[str]
    inlier_count: int
    correspondence_count: int
    inlier_ratio: float
    mean_reprojection_px: Optional[float]
    median_reprojection_px: Optional[float]
    p95_reprojection_px: Optional[float]
    confidence: float
    version: int
    rollback_available: bool
    reason: Optional[str] = None


def _points(value: ArrayLike2D, name: str) -> np.ndarray:
    """Convert a point list to finite ``(N,2)`` float64 coordinates."""

    try:
        points = np.asarray(value, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (N,2)") from exc
    if len(points) == 0 or not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain at least one finite point")
    return points


def _project(H: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    """Apply a floor-to-image homography and return ``NaN`` at a horizon."""

    homogeneous = np.column_stack((points_xy, np.ones(len(points_xy)))) @ H.T
    denominator = homogeneous[:, 2]
    result = np.full((len(points_xy), 2), np.nan, dtype=np.float64)
    good = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    result[good] = homogeneous[good, :2] / denominator[good, None]
    return result


def _residuals(H: np.ndarray, image_points: np.ndarray, floor_points: np.ndarray) -> np.ndarray:
    """Pixel reprojection residual for each ``floor -> image`` correspondence."""

    projected = _project(H, floor_points)
    residual = np.full(len(image_points), np.inf, dtype=np.float64)
    finite = np.isfinite(projected).all(axis=1)
    residual[finite] = np.linalg.norm(projected[finite] - image_points[finite], axis=1)
    return residual


def _stats(residuals: np.ndarray) -> tuple[Optional[float], Optional[float], Optional[float]]:
    finite = np.asarray(residuals, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return None, None, None
    return (
        float(np.mean(finite)),
        float(np.median(finite)),
        float(np.percentile(finite, 95.0)),
    )


class OnlineHomographyRefiner:
    """RANSAC and quality-gated online floor-homography refinement.

    Parameters are deliberately conservative by default.  The update is
    accepted only when enough anchors agree, their pixel residual is small,
    and the candidate does not move the old calibration by more than
    ``max_update_px`` at the observed floor anchors.
    """

    def __init__(
        self,
        initial: FloorHomography,
        ransac_threshold_px: float = 4.0,
        min_inliers: int = 4,
        min_inlier_ratio: float = 0.6,
        max_update_px: float = 30.0,
        ransac_iterations: int = 200,
        seed: int = 42,
        max_history: int = 32,
    ) -> None:
        if not isinstance(initial, FloorHomography):
            raise TypeError("initial must be a FloorHomography")
        if ransac_threshold_px <= 0.0:
            raise ValueError("ransac_threshold_px must be positive")
        if min_inliers < 4:
            raise ValueError("min_inliers must be at least four")
        if not 0.0 < min_inlier_ratio <= 1.0:
            raise ValueError("min_inlier_ratio must be in (0,1]")
        if max_update_px <= 0.0:
            raise ValueError("max_update_px must be positive")
        if ransac_iterations < 1:
            raise ValueError("ransac_iterations must be positive")
        if max_history < 1:
            raise ValueError("max_history must be positive")

        self.ransac_threshold_px = float(ransac_threshold_px)
        self.min_inliers = int(min_inliers)
        self.min_inlier_ratio = float(min_inlier_ratio)
        self.max_update_px = float(max_update_px)
        self.ransac_iterations = int(ransac_iterations)
        self.seed = int(seed)
        self.max_history = int(max_history)
        self._current = self._copy_homography(initial)
        self._history: list[FloorHomography] = []
        self._version = 0

    @staticmethod
    def _copy_homography(value: FloorHomography, *, source: Optional[str] = None) -> FloorHomography:
        return FloorHomography(
            np.array(value.H_floor_to_image, dtype=np.float64, copy=True),
            np.array(value.H_image_to_floor, dtype=np.float64, copy=True),
            source=value.source if source is None else source,
        )

    @property
    def current(self) -> FloorHomography:
        """Return a defensive copy of the currently active homography."""

        return self._copy_homography(self._current)

    @property
    def version(self) -> int:
        return self._version

    @property
    def rollback_available(self) -> bool:
        return bool(self._history)

    def project_pixel_to_floor(self, pixels: ArrayLike2D) -> np.ndarray:
        """Map ideal/undistorted pixels through the active homography."""

        return self._current.pixel_to_floor_xy(pixels)

    def project_floor_to_pixel(self, floor_points_xy: ArrayLike2D) -> np.ndarray:
        """Project metric floor anchors through the active homography."""

        return self._current.floor_xy_to_pixel(floor_points_xy)

    def _result(
        self,
        *,
        accepted: bool,
        status: str,
        frame_id: Optional[str],
        inlier_count: int,
        correspondence_count: int,
        inlier_ratio: float,
        residuals: Optional[np.ndarray],
        confidence: float,
        reason: Optional[str] = None,
    ) -> HomographyUpdateResult:
        mean, median, p95 = (None, None, None) if residuals is None else _stats(residuals)
        return HomographyUpdateResult(
            accepted=accepted,
            status=status,
            frame_id=None if frame_id is None else str(frame_id),
            inlier_count=int(inlier_count),
            correspondence_count=int(correspondence_count),
            inlier_ratio=float(inlier_ratio),
            mean_reprojection_px=mean,
            median_reprojection_px=median,
            p95_reprojection_px=p95,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            version=self._version,
            rollback_available=self.rollback_available,
            reason=reason,
        )

    def update(
        self,
        image_points_px: ArrayLike2D,
        floor_points_xy: ArrayLike2D,
        frame_id: Optional[str] = None,
    ) -> HomographyUpdateResult:
        """Attempt one guarded RANSAC update from matched anchor points.

        ``image_points_px[i]`` corresponds to ``floor_points_xy[i]``.  The
        active homography is changed only after all gates pass; rejected
        updates leave it bit-for-bit unchanged.
        """

        try:
            image = _points(image_points_px, "image_points_px")
            floor = _points(floor_points_xy, "floor_points_xy")
        except ValueError as exc:
            return self._result(
                accepted=False,
                status="rejected_invalid_input",
                frame_id=frame_id,
                inlier_count=0,
                correspondence_count=0,
                inlier_ratio=0.0,
                residuals=None,
                confidence=0.0,
                reason=str(exc),
            )
        n = len(image)
        if n != len(floor):
            return self._result(
                accepted=False,
                status="rejected_mismatched_points",
                frame_id=frame_id,
                inlier_count=0,
                correspondence_count=max(n, len(floor)),
                inlier_ratio=0.0,
                residuals=None,
                confidence=0.0,
                reason="image_points_px and floor_points_xy must have equal length",
            )
        if n < 4:
            return self._result(
                accepted=False,
                status="rejected_too_few_correspondences",
                frame_id=frame_id,
                inlier_count=0,
                correspondence_count=n,
                inlier_ratio=0.0,
                residuals=None,
                confidence=0.0,
                reason="at least four correspondences are required",
            )

        rng = np.random.default_rng(self.seed + self._version + n)
        best_H: Optional[np.ndarray] = None
        best_inliers: Optional[np.ndarray] = None
        best_score: tuple[int, float] = (-1, np.inf)
        # Including the all-point model makes the update deterministic when
        # the stream contains no outliers and also helps with exactly four
        # anchors (where every RANSAC sample is identical).
        samples: list[np.ndarray] = [np.arange(n, dtype=int)]
        if n > 4:
            samples.extend(rng.choice(n, size=4, replace=False) for _ in range(self.ransac_iterations))
        for sample in samples:
            try:
                candidate = estimate_homography(image[sample], floor[sample])
            except (ValueError, np.linalg.LinAlgError):
                continue
            residuals = _residuals(candidate, image, floor)
            inliers = np.isfinite(residuals) & (residuals <= self.ransac_threshold_px)
            count = int(inliers.sum())
            median = float(np.median(residuals[inliers])) if count else np.inf
            score = (count, -median)
            if score > (best_score[0], -best_score[1]):
                best_H, best_inliers, best_score = candidate, inliers, (count, median)

        if best_H is None or best_inliers is None:
            return self._result(
                accepted=False,
                status="rejected_ransac_failed",
                frame_id=frame_id,
                inlier_count=0,
                correspondence_count=n,
                inlier_ratio=0.0,
                residuals=None,
                confidence=0.0,
                reason="no non-degenerate homography model was found",
            )

        inlier_count = int(best_inliers.sum())
        inlier_ratio = inlier_count / float(n)
        if inlier_count < self.min_inliers or inlier_ratio < self.min_inlier_ratio:
            residuals = _residuals(best_H, image, floor)
            return self._result(
                accepted=False,
                status="rejected_insufficient_inliers",
                frame_id=frame_id,
                inlier_count=inlier_count,
                correspondence_count=n,
                inlier_ratio=inlier_ratio,
                residuals=residuals[best_inliers],
                confidence=0.0,
                reason=(
                    f"need >= {self.min_inliers} inliers and ratio >= "
                    f"{self.min_inlier_ratio:.3f}"
                ),
            )

        # Refit using every RANSAC inlier to reduce sample noise.
        try:
            refined_H = estimate_homography(image[best_inliers], floor[best_inliers])
        except (ValueError, np.linalg.LinAlgError) as exc:
            return self._result(
                accepted=False,
                status="rejected_refit_failed",
                frame_id=frame_id,
                inlier_count=inlier_count,
                correspondence_count=n,
                inlier_ratio=inlier_ratio,
                residuals=None,
                confidence=0.0,
                reason=str(exc),
            )
        refined_residuals = _residuals(refined_H, image, floor)
        refined_inliers = np.isfinite(refined_residuals) & (refined_residuals <= self.ransac_threshold_px)
        if int(refined_inliers.sum()) < self.min_inliers:
            return self._result(
                accepted=False,
                status="rejected_refit_quality",
                frame_id=frame_id,
                inlier_count=int(refined_inliers.sum()),
                correspondence_count=n,
                inlier_ratio=float(refined_inliers.mean()),
                residuals=refined_residuals[refined_inliers],
                confidence=0.0,
                reason="refitted model lost the minimum inlier count",
            )

        current_projection = _project(self._current.H_floor_to_image, floor)
        candidate_projection = _project(refined_H, floor)
        # The update-size gate is evaluated on RANSAC inliers.  An anchor
        # rejected as an outlier must not veto a geometrically consistent
        # update by producing an arbitrarily large extrapolation error.
        comparable = (
            refined_inliers
            & np.isfinite(current_projection).all(axis=1)
            & np.isfinite(candidate_projection).all(axis=1)
        )
        if np.any(comparable):
            update_shift = np.linalg.norm(
                candidate_projection[comparable] - current_projection[comparable], axis=1
            )
            max_shift = float(np.max(update_shift))
            if not np.isfinite(max_shift) or max_shift > self.max_update_px:
                return self._result(
                    accepted=False,
                    status="rejected_update_too_large",
                    frame_id=frame_id,
                    inlier_count=int(refined_inliers.sum()),
                    correspondence_count=n,
                    inlier_ratio=float(refined_inliers.mean()),
                    residuals=refined_residuals[refined_inliers],
                    confidence=0.0,
                    reason=f"maximum anchor shift {max_shift:.3f}px exceeds {self.max_update_px:.3f}px",
                )

        accepted_residuals = refined_residuals[refined_inliers]
        _, median_error, _ = _stats(accepted_residuals)
        confidence = float(refined_inliers.mean()) * float(
            np.exp(-0.5 * (0.0 if median_error is None else median_error / self.ransac_threshold_px))
        )
        source = "online" if frame_id is None else f"online:{frame_id}"
        candidate = FloorHomography(refined_H, np.linalg.inv(refined_H), source=source)
        self._history.append(self._copy_homography(self._current))
        if len(self._history) > self.max_history:
            del self._history[0]
        self._current = candidate
        self._version += 1
        return self._result(
            accepted=True,
            status="accepted",
            frame_id=frame_id,
            inlier_count=int(refined_inliers.sum()),
            correspondence_count=n,
            inlier_ratio=float(refined_inliers.mean()),
            residuals=accepted_residuals,
            confidence=confidence,
        )

    def update_from_floor_homography(
        self,
        candidate: FloorHomography,
        image_points_px: ArrayLike2D,
        floor_points_xy: ArrayLike2D,
        frame_id: Optional[str] = None,
    ) -> HomographyUpdateResult:
        """Validate a pre-estimated candidate through the same guarded path.

        This helper is useful when a detector or a tracker already solved a
        homography but the application still wants the inlier/update audit.
        It temporarily evaluates the candidate as observations, then commits
        only when the regular quality gates accept it.
        """

        if not isinstance(candidate, FloorHomography):
            raise TypeError("candidate must be a FloorHomography")
        image = _points(image_points_px, "image_points_px")
        floor = _points(floor_points_xy, "floor_points_xy")
        if len(image) != len(floor):
            return self._result(
                accepted=False,
                status="rejected_mismatched_points",
                frame_id=frame_id,
                inlier_count=0,
                correspondence_count=max(len(image), len(floor)),
                inlier_ratio=0.0,
                residuals=None,
                confidence=0.0,
                reason="image_points_px and floor_points_xy must have equal length",
            )
        residuals = _residuals(candidate.H_floor_to_image, image, floor)
        inliers = np.isfinite(residuals) & (residuals <= self.ransac_threshold_px)
        if int(inliers.sum()) < self.min_inliers or float(inliers.mean()) < self.min_inlier_ratio:
            return self._result(
                accepted=False,
                status="rejected_candidate_quality",
                frame_id=frame_id,
                inlier_count=int(inliers.sum()),
                correspondence_count=len(image),
                inlier_ratio=float(inliers.mean()) if len(inliers) else 0.0,
                residuals=residuals[inliers],
                confidence=0.0,
                reason="candidate failed inlier or reprojection gate",
            )
        old = self._current
        old_projection = _project(old.H_floor_to_image, floor[inliers])
        new_projection = _project(candidate.H_floor_to_image, floor[inliers])
        shift = np.linalg.norm(new_projection - old_projection, axis=1)
        if not np.isfinite(shift).all() or float(np.max(shift)) > self.max_update_px:
            return self._result(
                accepted=False,
                status="rejected_update_too_large",
                frame_id=frame_id,
                inlier_count=int(inliers.sum()),
                correspondence_count=len(image),
                inlier_ratio=float(inliers.mean()),
                residuals=residuals[inliers],
                confidence=0.0,
                reason=f"maximum anchor shift exceeds {self.max_update_px:.3f}px",
            )
        self._history.append(self._copy_homography(old))
        if len(self._history) > self.max_history:
            del self._history[0]
        self._current = self._copy_homography(candidate)
        self._version += 1
        _, median_error, _ = _stats(residuals[inliers])
        confidence = float(inliers.mean()) * float(
            np.exp(-0.5 * (0.0 if median_error is None else median_error / self.ransac_threshold_px))
        )
        return self._result(
            accepted=True,
            status="accepted_candidate",
            frame_id=frame_id,
            inlier_count=int(inliers.sum()),
            correspondence_count=len(image),
            inlier_ratio=float(inliers.mean()),
            residuals=residuals[inliers],
            confidence=confidence,
        )

    def update_from_anchor_observations(
        self,
        observations: Mapping[str, Sequence[float]],
        anchor_world_xy: Mapping[str, Sequence[float]],
        frame_id: Optional[str] = None,
    ) -> HomographyUpdateResult:
        """Build correspondences from named metadata anchors."""

        common = [name for name in anchor_world_xy if name in observations]
        if len(common) < 4:
            return self._result(
                accepted=False,
                status="rejected_too_few_named_anchors",
                frame_id=frame_id,
                inlier_count=0,
                correspondence_count=len(common),
                inlier_ratio=0.0,
                residuals=None,
                confidence=0.0,
                reason="at least four shared named anchors are required",
            )
        return self.update(
            [observations[name] for name in common],
            [anchor_world_xy[name] for name in common],
            frame_id=frame_id,
        )

    def rollback(self) -> bool:
        """Restore the previous accepted calibration, if one exists."""

        if not self._history:
            return False
        self._current = self._history.pop()
        self._version += 1
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialise active calibration and online-update configuration."""

        return {
            "version": self._version,
            "rollback_available": self.rollback_available,
            "source": self._current.source,
            "H_floor_to_image": self._current.H_floor_to_image.tolist(),
            "H_image_to_floor": self._current.H_image_to_floor.tolist(),
            "config": {
                "ransac_threshold_px": self.ransac_threshold_px,
                "min_inliers": self.min_inliers,
                "min_inlier_ratio": self.min_inlier_ratio,
                "max_update_px": self.max_update_px,
                "ransac_iterations": self.ransac_iterations,
                "seed": self.seed,
                "max_history": self.max_history,
            },
        }

    def save_json(self, path: Path | str) -> Path:
        """Save the active calibration atomically enough for a frame loop."""

        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return output


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _homography_from_json(data: Mapping[str, Any]) -> FloorHomography:
    # Accept both the direct ``FloorHomography.to_dict()`` shape and the
    # ``online_homography`` CLI envelope.
    if isinstance(data.get("homography"), Mapping):
        data = data["homography"]
    H = data.get("H_floor_to_image")
    H_inv = data.get("H_image_to_floor")
    if H is None:
        raise ValueError("Initial homography JSON needs H_floor_to_image")
    H_array = np.asarray(H, dtype=np.float64).reshape(3, 3)
    if H_inv is None:
        H_inv_array = np.linalg.inv(H_array)
    else:
        H_inv_array = np.asarray(H_inv, dtype=np.float64).reshape(3, 3)
    return FloorHomography(H_array, H_inv_array, source=str(data.get("source", "json")))


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSON stream of anchor updates")
    parser.add_argument("--initial", type=Path, default=None, help="Existing FloorHomography JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ransac-threshold-px", type=float, default=4.0)
    parser.add_argument("--min-inliers", type=int, default=4)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.6)
    parser.add_argument("--max-update-px", type=float, default=30.0)
    parser.add_argument("--ransac-iterations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = _load_json(args.input.expanduser().resolve())
    updates = data.get("updates", data.get("frames"))
    if not isinstance(updates, list) or not updates:
        raise ValueError("Input JSON needs a non-empty 'updates' list")
    initial_data: Optional[Mapping[str, Any]] = None
    if args.initial is not None:
        initial_data = _load_json(args.initial.expanduser().resolve())
    elif isinstance(data.get("initial"), dict):
        initial_data = data["initial"]

    if initial_data is None:
        first = updates[0]
        if not isinstance(first, dict):
            raise ValueError("Each update must be a JSON object")
        image = first.get("image_points_px", first.get("image_points"))
        floor = first.get("floor_points_xy", first.get("floor_points"))
        if image is None or floor is None:
            raise ValueError("The first update must contain image and floor points when --initial is omitted")
        initial = FloorHomography.from_correspondences(image, floor)
    else:
        initial = _homography_from_json(initial_data)

    refiner = OnlineHomographyRefiner(
        initial,
        ransac_threshold_px=args.ransac_threshold_px,
        min_inliers=args.min_inliers,
        min_inlier_ratio=args.min_inlier_ratio,
        max_update_px=args.max_update_px,
        ransac_iterations=args.ransac_iterations,
        seed=args.seed,
    )
    results: list[dict[str, Any]] = []
    for item in updates:
        if not isinstance(item, dict):
            raise ValueError("Each update must be a JSON object")
        frame_id = item.get("frame_id")
        if "observations" in item and "anchor_world_xy" in item:
            result = refiner.update_from_anchor_observations(
                item["observations"], item["anchor_world_xy"], frame_id=frame_id
            )
        else:
            image = item.get("image_points_px", item.get("image_points"))
            floor = item.get("floor_points_xy", item.get("floor_points"))
            if image is None or floor is None:
                raise ValueError("Update needs image_points_px/floor_points_xy or named anchor maps")
            result = refiner.update(image, floor, frame_id=frame_id)
        results.append(asdict(result))

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"homography": refiner.to_dict(), "updates": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved={output} version={refiner.version} accepted={sum(r['accepted'] for r in results)}")


if __name__ == "__main__":
    _main()


__all__ = ["HomographyUpdateResult", "OnlineHomographyRefiner"]
