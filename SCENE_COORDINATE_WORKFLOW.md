# Low-cost post-ROI 3D workflow

This experiment evaluates alternatives to Homography/IPM and mesh Ray Casting
before purchasing hardware. It keeps the existing detector, ROI checkpoints,
datasets and previous benchmark outputs unchanged.

## Branches

### 1. Scene-coordinate MLP

The small `SceneCoordinateMLP` receives the ROI/coarse pixel together with
camera intrinsics, pose and the corresponding world ray. It directly predicts
`fire_xyz_world` in metres. It does not intersect a mesh at inference time.

This is inexpensive at inference, but it still requires metric XYZ labels and
camera metadata during training. It learns the scene distribution, so it can
fail on a new room, a new camera pose or a surface not represented in the
training scenes.

### 2. KNN/IDW scene-coordinate interpolation

`benchmark_low_cost_3d.py` standardises the same calibration-aware features as
the MLP and stores only the training samples. At test time it averages the
nearest training scene coordinates using inverse-distance weights. The value
of `k` is selected on validation scenes only.

Advantages: no extra training framework, CPU-friendly and easy to inspect.
Disadvantages: it is local interpolation, not a general 3D solver; sparse
coverage and scene changes cause large errors. It must remain scene-separated
from the test split to avoid memorisation.

### 3. Multi-view triangulation

`triangulation_3d.py` solves a linear DLT problem from two or more observations
of the same fire event and calibrated camera poses. It uses no room mesh and
does not assume that the fire is on the floor. The benchmark uses all visible
frames in one synthetic sequence as an offline upper-feasibility experiment.

For a live CCTV stream, replace the full sequence by a causal sliding window
and track the fire event between frames. The method needs genuine parallax:
camera motion that is too small makes the triangulation ill-conditioned, while
large pixel noise creates large depth error. Reprojection error, positive depth
and camera baseline must therefore be logged and used as a rejection gate.

## Reproducible commands

```powershell
cd D:\LAB\SAM_Experiment
.\.venv\Scripts\Activate.ps1

python benchmark_low_cost_3d.py `
  --dataset working\synthetic_fire_3d_v3 `
  --checkpoint output\scene_coordinate_regression_rerun_20261008\best_scene_coordinate_mlp.pth `
  --pixel-key p_fire_noisy_pixel `
  --triangulation-pixel-key p_fire_noisy_pixel `
  --camera-source estimated `
  --output-dir output\low_cost_3d_benchmark_20261008_v2

python visualize_3d_results.py `
  --summary output\low_cost_3d_benchmark_20261008_v2\summary.json `
  --dataset working\synthetic_fire_3d_v3 `
  --output-dir output\low_cost_3d_benchmark_20261008_v2\visualization `
  --write-ply
```

## How to interpret the variants

| Variant | What it tests | How to use it |
|---|---|---|
| noisy pixel + estimated pose | cheapest realistic synthetic stress test | primary feasibility result |
| noisy pixel + true pose | isolates pixel noise | identifies whether ROI or pose dominates |
| clean pixel + true pose | geometry upper bound | diagnostic only; never report as deployed performance |

The benchmark excludes frames where the noisy detector point is missing. It
does not silently substitute the clean label. The clean oracle is run as a
separate, explicitly named ablation so it cannot be confused with a real
detector result.

## Decision rule before buying hardware

1. Keep Ray Casting as the metric reference when a mesh and calibration are
   available.
2. Keep Scene-MLP only if its speed is more important than its current error,
   and retrain it with more scene diversity rather than copying frames.
3. Do not buy hardware for triangulation until the noisy/true-pose variant has
   a low reprojection error and acceptable 3D error. If it is poor even with
   true poses, the problem is mainly pixel noise/parallax and extra hardware
   will not fix it automatically.
4. If the noisy/true-pose result is good but estimated-pose is poor, invest in
   pose/calibration quality rather than a larger detector.
5. If both fail while the oracle is good, improve ROI point quality and use a
   temporal track before changing the 3D geometry.

All current measurements are synthetic/asset-backed metric geometry checks.
They do not establish accuracy for a real CCTV room.
