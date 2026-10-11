"""Focused regression tests for the Section 1.2 research artifact."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmark_section_1_2 import audit_real_summary
from depth_localization import DepthBasedLocator, DifferentiableRayCastLayer, VerticalityPriorLoss
from fire_detector import FireDetector
from narrow_localizer import ROIRefinerInference
from temporal_outlier_guard import TemporalCoarseOutlierGuard


class ConstantDepth(nn.Module):
    def forward(self, pixel_values: torch.Tensor, **_: object) -> dict[str, torch.Tensor]:
        return {"predicted_depth": torch.full_like(pixel_values[:, :1], 2.0)}


class ResearchSectionTests(unittest.TestCase):
    def test_audit_recomputes_all_real_crops_inside(self) -> None:
        path = Path(__file__).resolve().parents[1] / "output" / "audit_roi_metric_discrepancy_20261009" / "summary.json"
        if not path.is_file():
            self.skipTest(f"cached audit missing: {path}")
        result = audit_real_summary(path)
        self.assertEqual(result["records"], 36)
        self.assertEqual(result["recomputed"]["gt_inside_crop_count"], 36)
        self.assertEqual(result["recomputed"]["gt_outside_crop_count"], 0)

    def test_depth_locator_backprojects_and_handles_oob(self) -> None:
        K = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
        locator = DepthBasedLocator(ConstantDepth(), K, use_amp=False, out_of_bounds="return_none")
        image = torch.zeros(3, 3, 4)
        point = locator.locate(image, torch.tensor([2.0, 1.0]))
        self.assertTrue(torch.allclose(point, torch.tensor([1.0, 0.0, 2.0])))
        self.assertIsNone(locator.locate(image, torch.tensor([-1.0, 1.0])))

    def test_differentiable_ray_gradient(self) -> None:
        pixel = torch.tensor([0.25, -0.5], dtype=torch.float64, requires_grad=True)
        layer = DifferentiableRayCastLayer()
        point = layer(pixel, torch.eye(3, dtype=torch.float64), torch.eye(3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), torch.tensor([0.0, 0.0, 1.0, -3.0], dtype=torch.float64))
        point.square().sum().backward()
        self.assertIsNotNone(pixel.grad)
        self.assertTrue(torch.isfinite(pixel.grad).all())
        self.assertGreater(float(pixel.grad.abs().sum()), 0.0)

    def test_verticality_wind_relaxes_penalty(self) -> None:
        loss = VerticalityPriorLoss(cosine_weight=0.0, upward_weight=0.0)
        base = torch.zeros(1, 3)
        top = torch.tensor([[1.0, 0.0, 2.0]])
        self.assertLess(float(loss(base, top, wind_strength=torch.tensor([10.0]))), float(loss(base, top)))

    def test_tta_gate_separates_fire_detection_from_localisation(self) -> None:
        result = FireDetector._aggregate_tta(
            [(0.99, (30.0, 30.0)), (0.995, (40.0, 30.0)), (0.992, (30.0, 30.0))],
            size=(224, 224),
            threshold=0.5,
            reliability_scale_px=8.0,
            max_stability_px=2.0,
            stability_reference_size=224.0,
            latency_ms=0.0,
        )
        self.assertTrue(result.detected)
        self.assertFalse(result.localization_accepted)
        self.assertIsNone(result.pixel)
        self.assertIsNotNone(result.raw_pixel)
        self.assertEqual(result.localization_status, "unstable")

    def test_dynamic_roi_side_is_monotonic_and_capped(self) -> None:
        checkpoint = Path(__file__).resolve().parents[1] / "output" / "roi_domain_experiments_cpu_regularized" / "mixed" / "best_roi.pth"
        if not checkpoint.is_file():
            self.skipTest(f"ROI checkpoint missing: {checkpoint}")
        refiner = ROIRefinerInference(checkpoint, device="cpu")
        sides = [refiner.dynamic_side_px((224, 224), value, expansion_factor=8.0,
                                          stability_threshold_px=6.0)
                 for value in (0.0, 6.0, 8.0, 12.0)]
        self.assertEqual(sides[0], sides[1])
        self.assertLessEqual(sides[1], sides[2])
        self.assertLessEqual(sides[2], sides[3])
        self.assertLessEqual(sides[-1], 224.0)

    def test_temporal_guard_replaces_isolated_jump(self) -> None:
        image = torch.zeros(8, 8, 3, dtype=torch.uint8).numpy()
        guard = TemporalCoarseOutlierGuard(
            max_jump_px=20.0, max_frame_mean_abs_diff=25.0, min_stability_px=6.0
        )
        self.assertFalse(guard.update(image, (100.0, 100.0), stability_px=1.0).replaced)
        result = guard.update(image, (150.0, 100.0), stability_px=7.0)
        self.assertTrue(result.replaced)
        self.assertEqual(result.reason, "isolated_jump_replaced")
        self.assertEqual(result.point, (100.0, 100.0))


if __name__ == "__main__":
    unittest.main()

