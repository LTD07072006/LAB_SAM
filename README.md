# LAB_SAM — Fire 2D-to-3D Localisation

This repository contains the Week 6 experiment for fire localisation:

```text
fire detector → coarse 2D point → ROI refinement → camera calibration
→ ray casting against a room surface/mesh → robust 3D aggregation
→ uncertainty and temporal tracking
```

The detector and ROI refiner are separate modules. `best.pth` is the upstream
fire detector checkpoint and `week6_roi_result/best_roi.pth` is the point
refinement checkpoint.

## Main files

- `fire_detector.py`: detector adapter and checkpoint loader.
- `detector_2d.py`: higher-resolution FPN detector used before ROI; it keeps
  the old `best.pth` untouched and returns the same `confidence + pixel`
  contract.
- `train_detector_2d.py`: command-line entry point for training that detector.
- `compare_detector_2d.py`: detector-only old/new comparison with pixel
  metrics and a contact sheet.
- `narrow_localizer.py`: ROIRefiner and inference adapter.
- `main_localization.py`: end-to-end image/sequence pipeline.
- `benchmark_ab.py`: coarse-versus-refined A/B benchmark.
- `compare_v3_roi.py`: same-image visual and pixel-level comparison of
  baseline/coarse, ROI-refined and v3 FPN points.
- `run_week6.py`: one-file coordinator for compare, demo, benchmark and all.
- `v3_detector.py`: v3 MobileNetV4 + FPN inference adapter.
- `camera_calibration.py`: intrinsics, extrinsics and undistortion.
- `locator.py`: ray casting, `GridMap` and `TriangleMesh`.
- `mesh_loader.py`: JSON triangle-mesh loader.
- `train_week6.py`: spatial fire detector training utilities.
- `sam-week6-roi-train-complete.ipynb`: Kaggle ROIRefiner training workflow.
- `synthetic_fire_3d.py`: dependency-light metric synthetic fire/room dataset
  generator with exact 2D-3D camera correspondences and explicit noise.
- `benchmark_five_workflows.py`: same-data benchmark for Ray Casting, IPM,
  GPR residual correction, optional monocular depth and uncertainty-aware
  fusion, including EMA/EKF sequence metrics.
- `benchmark_homography.py`: calibrated floor homography/IPM, DLT/RANSAC and
  ray-casting comparison on metric synthetic scenes.
- `benchmark_multiplane_homography.py`: floor and multi-plane homography/IPM
  benchmark with plane-bank selection, visibility diagnostics, floor fallback,
  EMA smoothing and PNG/HTML/PLY 3D exports.
- `pixel_world_regressor.py`: calibrated homography mapper and residual GPR/IDW
  mapper; the residual model does not replace camera geometry.
- `monocular_depth_adapter.py` and `depth_to_world.py`: optional depth-map or
  Transformers adapter with explicit metric-scale checks.
- `uncertainty_fusion.py`: Ray-protected fusion of 3D candidates.
- `scene_coordinate_regression.py` and `train_scene_coordinate.py`: small
  calibration-aware MLP that predicts metric XYZ directly from a post-ROI
  pixel and camera metadata. It is a low-cost ablation, not a replacement for
  measured geometry.
- `triangulation_3d.py`: mesh-free multi-view DLT triangulation from the same
  fire point observed in several frames with camera parallax.
- `benchmark_low_cost_3d.py`: strict comparison of KNN/IDW, Scene-MLP,
  triangulation, Ray Casting and IPM on one scene-separated test split.
- `build_real_flame_manifest.py`: leakage-safe manifest builder for iPhone,
  webcam or CCTV annotations with optional measured XYZ labels.

See [SYNTHETIC_FIRE_3D.md](SYNTHETIC_FIRE_3D.md) for the synthetic dataset
contract and commands. The current generated artifact is
`working/synthetic_fire_3d_v3/`: 240 scene-level sequences, 960 frames at
640×640, with 672/144/144 train/val/test. It contains 524 visible-fire 2D
labels, 436 observable no-fire frames, and 288 physically-present but occluded
events kept separately through `fire_event=1, fire_visible=0, has_fire=0`.

The synthetic branch is for geometry validation, ROI stress testing and
controlled pretraining; it is not a substitute for real CCTV domain evaluation.

## Five post-ROI 3D branches

The recommended post-ROI experiment keeps calibrated Ray Casting as the main
method and treats IPM, GPR and monocular depth as controlled baselines or
auxiliary estimates:

```text
ROI pixel -> Ray Casting + mesh ------------------------------> XYZ (primary)
         -> Homography/IPM -> floor Z=0 ----------------------> baseline
         -> Ray estimate + GPR/IDW residual ------------------> corrected XYZ
         -> optional metric depth map ------------------------> auxiliary XYZ
         -> uncertainty-aware fusion (Ray gets priority) -----> XYZ
                                                        -> EMA / EKF
```

Run the five-way benchmark on the synthetic metric test split:

```powershell
python benchmark_five_workflows.py `
  --dataset working\synthetic_fire_3d_v3 `
  --split test --max-records 24 --selection even `
  --no-roi --depth-backend none `
  --output-dir output\workflow_geometry_smoke_20261007 --device cpu
```

After the geometry smoke test, include the mixed ROI checkpoint:

```powershell
python benchmark_five_workflows.py `
  --dataset working\synthetic_fire_3d_v3 `
  --roi-checkpoint output\roi_domain_experiments_cpu_regularized\mixed\best_roi.pth `
  --split test --max-records 24 --selection diverse `
  --output-dir output\workflow_roi_smoke_20261007 --device cpu
```

The command writes branch metrics (pixel MAE/PCK, ray hit rate, 3D MAE,
median/P95, XYZ errors, threshold rates, latency and fallback), sequence
jitter/jump statistics, a contact sheet, and a GPR model artifact. GPR is fit
on the train split only. If scikit-learn is not installed, the summary records
the dependency-free IDW residual fallback. Depth is `unavailable` until a
depth-map folder or an optional model with a declared metric scale is supplied;
relative depth is not silently treated as metres.

Export the resulting 3D scene as PNG/HTML/PLY:

```powershell
python visualize_3d_results.py `
  --summary output\workflow_roi_smoke_20261007\summary.json `
  --dataset working\synthetic_fire_3d_v3 `
  --output-dir output\workflow_roi_smoke_20261007\visualization `
  --max-records 24 --write-ply
```

The visualizer includes Ray, IPM, GPR, depth, fusion, EMA and EKF points. A
synthetic or asset-backed result is a geometry validation result, not a claim
about real CCTV metre accuracy.

### Low-cost branches that do not require new hardware

Before buying a camera, markers or a room scanner, the project now includes
two additional post-ROI directions in
[SCENE_COORDINATE_WORKFLOW.md](SCENE_COORDINATE_WORKFLOW.md):

```text
ROI pixel + camera metadata -> KNN/IDW local interpolation -> XYZ
ROI points across frames + camera motion -> multi-view triangulation -> XYZ
```

Run the full CPU benchmark without modifying the existing dataset or previous
outputs:

```powershell
python benchmark_low_cost_3d.py `
  --dataset working\synthetic_fire_3d_v3 `
  --checkpoint output\scene_coordinate_regression_rerun_20261008\best_scene_coordinate_mlp.pth `
  --pixel-key p_fire_noisy_pixel `
  --triangulation-pixel-key p_fire_noisy_pixel `
  --camera-source estimated `
  --output-dir output\low_cost_3d_benchmark_20261008_v2
```

The benchmark enforces a strict detector-observation policy: a missing noisy
pixel is counted as a miss and is never replaced by `p_fire_pixel`. It reports
the KNN/IDW validation-selected `k`, Scene-MLP, Ray and IPM metrics, plus
three explicitly labelled triangulation cases: noisy pixel + estimated pose,
noisy pixel + true pose, and clean pixel + true pose oracle. The last case is
an upper bound for debugging, not a deployable result.

Export all available branches as 3D PNG/HTML/PLY:

```powershell
python visualize_3d_results.py `
  --summary output\low_cost_3d_benchmark_20261008_v2\summary.json `
  --dataset working\synthetic_fire_3d_v3 `
  --output-dir output\low_cost_3d_benchmark_20261008_v2\visualization `
  --write-ply
```

On the current synthetic test, this branch is for cost/feasibility analysis:
Ray Casting remains the strongest metric reference; Scene-MLP is fast but less
accurate; KNN/IDW is weaker with the current sparse scene coverage; and
triangulation is useful only when observations have sufficient parallax and
low pixel/pose noise. None of these synthetic numbers is a real CCTV claim.

### Homography/IPM beyond floor ray casting

For the detailed protocol and current synthetic/ReplicaCAD measurements, see
[HOMOGRAPHY_DIRECTIONS.md](HOMOGRAPHY_DIRECTIONS.md). The multi-plane benchmark
supports a fast floor IPM baseline, a plane bank for tables/cabinets/columns,
a visibility-assisted selector and a safe floor fallback. `surface_oracle` is
an upper-bound ablation because it reads the ground-truth surface label; it is
not an end-to-end deployment result. Every run writes a `summary.json`, plane
inventory, contact sheet, static 3D PNGs, interactive HTML and PLY annotations
to a new output directory.

Example:

```powershell
python benchmark_multiplane_homography.py `
  --dataset working\synthetic_fire_3d_v3 `
  --prediction-summary output\workflow_roi_cpu_regularized_test_full_20261007\summary.json `
  --split test --selection even `
  --output-dir output\multiplane_homography_20261007\synthetic_v3_homography_rerun
```

For an iPhone/webcam session, annotate an LED/display/marker or a safely
controlled source with `p_fire_pixel` and measured `fire_xyz_world`, then build
a scene-separated manifest:

```powershell
python build_real_flame_manifest.py `
  --dataset-root working\iphone_room_session `
  --annotations annotations.json `
  --output working\iphone_room_session\manifest.jsonl `
  --require-xyz --source-type led
```

The manifest builder rejects missing images, duplicate frames and scene leakage
between splits. It does not control or ignite a flame. Measured camera
calibration and mesh must use the same room coordinate frame as `fire_xyz_world`.

## Current benchmark status

Các artifact benchmark được sinh dưới `output/` và không được coi là dữ liệu
đầu vào. Những kết quả dùng calibration/mesh mô phỏng 224×224 chỉ là kiểm thử
provisional, không phải calibration camera thật hay phòng đo thật. Sai số 3D
vật lý chỉ hợp lệ khi có camera đã đo, mesh theo mét và ground truth 3D.

## Run the provisional pipeline

### One-file Week 6 runner

Use the project virtual environment so `pip` and the launched scripts use the
same Python installation:

```powershell
cd D:\LAB\SAM_Experiment
.\.venv\Scripts\Activate.ps1
python -m pip install numpy pillow torch torchvision timm opencv-python
python run_week6.py --mode compare --max-images 10 --device cuda
```

The available modes are:

```powershell
python run_week6.py --mode compare   --max-images 10 --device cuda
python run_week6.py --mode demo      --max-images 10 --device cuda --no-uncertainty
python run_week6.py --mode benchmark --max-images 10 --device cuda
python run_week6.py --mode all       --max-images 10 --device cuda
```

The runner automatically prefers `.venv\Scripts\python.exe`, verifies
`numpy`, Pillow, PyTorch, torchvision and timm, and writes results under
`output\week6_run\`. Override paths with `--labels`, `--dataset`,
`--baseline`, `--roi`, `--v3`, `--calibration` or `--mesh` when necessary.

`--max-images` applies to all three modes. `benchmark_ab.py` uses the same
labels/dataset/checkpoints selected by the runner; it no longer silently falls
back to a hard-coded test directory. Use `--no-uncertainty` for a fast smoke
test and omit it for the uncertainty report.

`compare` evaluates the same labelled fire images in pixel coordinates and
creates annotated images plus `comparison_contact_sheet.png`. Its table is
not a metre-level 3D benchmark. The `demo` and `benchmark` paths can perform
ray casting, but a valid physical result still requires measured calibration,
a metric room mesh and 3D ground truth.

The v3 checkpoint must match the FPN implementation in `v3_detector.py`.
If the checkpoint was saved by a different training file, restore that exact
model definition or retrain/export a compatible checkpoint; do not interpret
the fallback baseline detector as a v3 result.

### Improve the 2D detector before ROI

The detector stage can be retrained independently of ROI and 3D geometry:

```powershell
python train_detector_2d.py `
  --labels "fire-model-data\dataset_labels (1).json" `
  --dataset-root datasets\fire-detection-from-cctv `
  --init-checkpoint fire-model-data\week6_spatial\best_spatial.pth `
  --output-dir fire-model-data\detector_2d_v2 `
  --image-size 640 `
  --batch-size 8 `
  --epochs 35 `
  --device cuda
```

The new model uses a letterbox transform, a reduction-4/8/16 FPN, a 160×160
heatmap for a 640×640 input, positive-only point loss and a separate fire
classification head. The checkpoint is saved as
`fire-model-data\detector_2d_v2\best_detector_2d.pth`; it does not overwrite
`fire-model-data\best.pth` or the previous spatial checkpoint. If pretrained
timm weights cannot be downloaded, add `--no-pretrained`.

#### Train the current detector locally or on Kaggle
The active checkout keeps the detector model, training entry point and complete
ROI notebook. Run the local 2D comparison first, then create a coarse manifest
and train the ROI refiner with the commands above. The old standalone Kaggle
launcher and independent Home Fire YOLO branch are no longer part of the active
workflow; existing datasets and checkpoints are unchanged.
### Direct commands

```powershell
python main_localization.py `
  --samples datasets\fire-detection-from-cctv\data\data\img_data\test\fire `
  --model fire-model-data\best.pth `
  --roi-checkpoint week6_roi_result\best_roi.pth `
  --calibration archives\calibration_test_224.json `
  --mesh archives\floor_mesh_test_224.json `
  --output working\localization_mesh_test.json
```

For an A/B benchmark:

```powershell
python benchmark_ab.py `
  --calibration archives\calibration_test_224.json `
  --mesh archives\floor_mesh_test_224.json `
  --output working\benchmark_ab_test_fire_mesh.json
```

## Important data note

The included public fire datasets contain 2D images, labels and videos. They
do not provide the physical room dimensions, camera calibration or 3D fire
coordinates needed for a valid metric 3D benchmark. Replace the provisional
calibration/mesh with measured files before reporting metre-level accuracy.

## Security note

Credentials are not stored in this repository. The archived SAM notebook had
an old hard-coded Hugging Face token in its saved cell/output; it was replaced
with a redacted placeholder before committing. Use environment or notebook
secrets when authenticating.

