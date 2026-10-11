"""Small, dependency-light tests for depth_localization.py.

Run with::

    python -m unittest -v test_depth_localization.py
"""

from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from depth_localization import (
    DepthBasedLocator,
    DifferentiableRayCastLayer,
    VerticalityPriorLoss,
)


class FakeDepthModel(nn.Module):
    """Returns a deterministic positive depth map with the input resolution."""

    def forward(self, pixel_values: Tensor, **_: object) -> dict[str, Tensor]:
        # A constant raw map keeps this test independent of a downloaded model.
        return {"predicted_depth": torch.full_like(pixel_values[:, :1], 2.0)}


class NaNDepthModel(nn.Module):
    def forward(self, pixel_values: Tensor, **_: object) -> dict[str, Tensor]:
        depth = torch.full_like(pixel_values[:, :1], 2.0)
        depth[..., 1, 1] = float("nan")
        return {"predicted_depth": depth}


class DepthLocatorTests(unittest.TestCase):
    def test_depth_locator_backprojects_pixel(self) -> None:
        # fx=fy=2, principal point=(1,1), pixel=(2,1), Z=2 => (X,Y,Z)=(1,0,2).
        K = torch.tensor([[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]])
        locator = DepthBasedLocator(FakeDepthModel(), K, use_amp=False)
        rgb = torch.zeros(3, 3, 4)
        point = locator.locate(rgb, torch.tensor([2.0, 1.0]))
        self.assertIsNotNone(point)
        assert point is not None
        self.assertTrue(torch.allclose(point, torch.tensor([1.0, 0.0, 2.0]), atol=1e-5))

    def test_out_of_bounds_and_nan_modes(self) -> None:
        K = torch.eye(3)
        image = torch.zeros(3, 3, 4)
        returning = DepthBasedLocator(
            FakeDepthModel(), K, out_of_bounds="return_none", use_amp=False
        )
        self.assertIsNone(returning.locate(image, torch.tensor([-1.0, 1.0])))

        nearest = DepthBasedLocator(
            NaNDepthModel(), K, invalid_depth="nearest_valid", max_valid_search_radius=1, use_amp=False
        )
        point = nearest.locate(image, torch.tensor([1.0, 1.0]))
        self.assertIsNotNone(point)
        assert point is not None
        self.assertTrue(torch.isfinite(point).all())


class RayAndLossTests(unittest.TestCase):
    def test_ray_cast_gradient_reaches_point_2d(self) -> None:
        point_2d = torch.tensor([0.25, -0.5], dtype=torch.float64, requires_grad=True)
        K = torch.eye(3, dtype=torch.float64)
        R = torch.eye(3, dtype=torch.float64)
        T = torch.zeros(3, dtype=torch.float64)
        # z=3 plane: 0X + 0Y + 1Z - 3 = 0.
        plane = torch.tensor([0.0, 0.0, 1.0, -3.0], dtype=torch.float64)
        point_3d = DifferentiableRayCastLayer()(point_2d, K, R, T, plane)
        loss = point_3d.square().sum()
        loss.backward()
        self.assertIsNotNone(point_2d.grad)
        assert point_2d.grad is not None
        self.assertTrue(torch.isfinite(point_2d.grad).all())
        self.assertGreater(float(point_2d.grad.abs().sum()), 0.0)
        self.assertTrue(torch.allclose(point_3d, torch.tensor([0.75, -1.5, 3.0], dtype=torch.float64)))

    def test_verticality_prior_penalizes_horizontal_offset(self) -> None:
        loss_fn = VerticalityPriorLoss(
            adaptive_weight=1.0, cosine_weight=0.0, upward_weight=0.0, reduction="mean"
        )
        base = torch.tensor([[0.0, 0.0, 0.0]])
        vertical_top = torch.tensor([[0.0, 0.0, 2.0]])
        tilted_top = torch.tensor([[1.0, 0.0, 2.0]])
        self.assertLess(float(loss_fn(base, vertical_top)), float(loss_fn(base, tilted_top)))
        relaxed = loss_fn(base, tilted_top, wind_strength=torch.tensor([10.0]))
        self.assertLess(float(relaxed), float(loss_fn(base, tilted_top)))


if __name__ == "__main__":
    unittest.main()

