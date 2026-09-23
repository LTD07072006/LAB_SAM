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
- `narrow_localizer.py`: ROIRefiner and inference adapter.
- `main_localization.py`: end-to-end image/sequence pipeline.
- `benchmark_ab.py`: coarse-versus-refined A/B benchmark.
- `camera_calibration.py`: intrinsics, extrinsics and undistortion.
- `locator.py`: ray casting, `GridMap` and `TriangleMesh`.
- `mesh_loader.py`: JSON triangle-mesh loader.
- `train_week6.py`: spatial fire detector training utilities.
- `sam-week6-roi-train-complete.ipynb`: Kaggle ROIRefiner training workflow.

## Current benchmark status

`working/benchmark_ab_test_fire_mesh.json` records a provisional run on 36
test/fire images. The calibration and floor mesh used there are synthetic
geometry matched to 224×224 images; they are not a physical camera
calibration or a measured room. A physical 3D error requires measured camera
pose, a metric room mesh and 3D ground truth.

## Run the provisional pipeline

```powershell
python main_localization.py `
  --samples fire-detection-from-cctv\data\data\img_data\test\fire `
  --model fire-model-data\best.pth `
  --roi-checkpoint week6_roi_result\best_roi.pth `
  --calibration calibration_test_224.json `
  --mesh floor_mesh_test_224.json `
  --output working\localization_mesh_test.json
```

For an A/B benchmark:

```powershell
python benchmark_ab.py `
  --calibration calibration_test_224.json `
  --mesh floor_mesh_test_224.json `
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
