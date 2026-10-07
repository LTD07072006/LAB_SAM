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
- `kaggle_train_detector_2d.py`: Kaggle T4 launcher; discovers code, labels,
  dataset and optional initialization checkpoint under `/kaggle/input`.
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
- `train_home_fire_detector.py`: independent Ultralytics YOLO bbox training
  branch for the 1.9 GB Home Fire dataset.
- `prepare_home_fire_dataset.py`: validates the YOLO layout, counts labels and
  writes class/sample previews without changing the source dataset.
- `home_fire_detector_adapter.py`: converts a YOLO bbox into a 2D
  bottom-contact point and bottom-band candidates for ray casting.
- `home_fire_3d_pipeline.py`: detector-independent bridge from the YOLO
  contact hypotheses to undistortion, multi-ray mesh/GridMap localisation,
  uncertainty and optional tracking.
- `run_home_fire_localization.py`: standalone YOLO-to-3D entry point. It keeps
  the coarse point as a fallback and guards optional ROI refinement with a
  shift limit, heatmap confidence and weighted blending.
- `export_yolo_coarse_manifest.py`: exports weak YOLO contact hypotheses; it
  never modifies the `p_fire` labels.
- `compare_yolo_branch.py`: YOLO bbox/IoU/latency benchmark and visual preview.
- `audit_fire_labels.py`: dependency-free audit of the `1=fire`, `0=no fire`
  classification contract and the original-to-one-class YOLO mapping.
- `synthetic_fire_3d.py`: dependency-light metric synthetic fire/room dataset
  generator with exact 2D-3D camera correspondences and explicit noise.
- `evaluate_synthetic_fire_3d.py`: clean-versus-noisy ray-casting evaluation
  in metres for the generated dataset.

See [SYNTHETIC_FIRE_3D.md](SYNTHETIC_FIRE_3D.md) for the synthetic dataset
contract and commands. The current generated artifact is
`working/synthetic_fire_3d_v3/`: 240 scene-level sequences, 960 frames at
640×640, with 672/144/144 train/val/test. It contains 524 visible-fire 2D
labels, 436 observable no-fire frames, and 288 physically-present but occluded
events kept separately through `fire_event=1, fire_visible=0, has_fire=0`.

The branch includes `benchmark_synthetic_noise.py`, which evaluates 0/1/3/5/10
pixel point noise and calibration scales 0/1/2 on the same fixed test scenes,
and `blender_render_synthetic_fire.py`, an optional Blender adapter that keeps
the metric camera/mesh/XYZ manifest while adding RGB/depth/mask renders. This
synthetic branch is for geometry validation, ROI stress testing and controlled
pretraining; it is not a substitute for real CCTV domain evaluation.

## Current benchmark status

`working/benchmark_ab_test_fire_mesh.json` records a provisional run on 36
test/fire images. The calibration and floor mesh used there are synthetic
geometry matched to 224×224 images; they are not a physical camera
calibration or a measured room. A physical 3D error requires measured camera
pose, a metric room mesh and 3D ground truth.

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

#### Train on a Kaggle Tesla T4
The local `.venv` is CPU-only, so use Kaggle for the full 640×640 run. Your
large Kaggle Dataset may contain only the archive
`fire-detection-from-cctv/data/data.zip`; this is sufficient for the image
input. The notebook/launcher extracts `data/img_data` into the writable
`/kaggle/working` directory and does not require an already extracted
`fire-detection-from-cctv` folder. Code, labels and weights may be in the
same Dataset or in separate attached Datasets.

```text
detector_2d.py
train_week6.py
train_detector_2d.py
kaggle_train_detector_2d.py
dataset_labels (1).json
fire-detection-from-cctv/data/data.zip  # accepted image input
week6_spatial/best_spatial.pth        # optional; used for backbone transfer
```

If all required files are included in the same large Dataset, attaching that
one Dataset is enough. The required files are `detector_2d.py`,
`train_week6.py`, `train_detector_2d.py`, `dataset_labels (1).json` and the
`data.zip` archive. If the archive is in one Dataset and code/labels are in
another, attach both.

In the Kaggle notebook, select `Settings → Accelerator → GPU T4`, then run:

```python
!nvidia-smi
!pip install -q timm
```

Import and run `sam-detector2d-train-640-kaggle-datazip.ipynb`, or run the
training launcher from the code Dataset. Replace `<code-dataset>` with the
slug shown in the right-hand Input panel:

```python
!python /kaggle/input/<code-dataset>/kaggle_train_detector_2d.py \
    --input-root /kaggle/input \
    --output-dir /kaggle/working/detector_2d_v2 \
    --device 0 \
    --image-size 640 \
    --batch-size 4 \
    --grad-accumulation 2 \
    --epochs 35 \
    --freeze-epochs 3 \
    --workers 2
```

Here `--device 0` is converted to PyTorch `cuda:0`, AMP is enabled
automatically, and the effective batch size is `4 × 2 = 8`. If the T4 runs
out of memory, use `--batch-size 2 --grad-accumulation 4`; keep
`--image-size 640`. The launcher automatically searches recursively for
`data.zip`. If there is more than one ZIP with that name, pass the exact ZIP
path explicitly:

```python
!python /kaggle/input/<code-dataset>/kaggle_train_detector_2d.py \
    --code-root /kaggle/input/<code-dataset>/SAM_Experiment \
    --labels "/kaggle/input/<labels-dataset>/fire-model-data/dataset_labels (1).json" \
    --data-zip /kaggle/input/<large-dataset>/fire-detection-from-cctv/data/data.zip \
    --init-checkpoint /kaggle/input/<weights-dataset>/fire-model-data/week6_spatial/best_spatial.pth \
    --output-dir /kaggle/working/detector_2d_v2 \
    --device 0 --image-size 640 --batch-size 4 --grad-accumulation 2 \
    --epochs 35 --workers 2
```

The important paths are read-only under `/kaggle/input`; only the extracted
images, checkpoints,
`history.json` and `test_metrics.json` are written to `/kaggle/working`. At
the end, download the whole `/kaggle/working/detector_2d_v2` directory. Do not
use a Windows path such as `D:\LAB\...` inside Kaggle: Kaggle cannot access
the local disk. When you only have the ZIP, do not pass `--dataset-root`; use
`--data-zip` or let the launcher discover it.

Compare the old and new detector before training/retraining ROI:

```powershell
python compare_detector_2d.py `
  --new fire-model-data\detector_2d_v2\best_detector_2d.pth `
  --split test --max-images 50 `
  --output-dir output\detector_2d_comparison `
  --device cuda
```

Only after the new detector improves detection rate and pixel metrics should it
be used to create a new coarse manifest and retrain ROI:

```powershell
python build_coarse_manifest.py `
  --model fire-model-data\detector_2d_v2\best_detector_2d.pth `
  --output fire-model-data\coarse_manifest_detector_2d_v2.json
python train_roi_localizer.py `
  --init-checkpoint fire-model-data\best.pth `
  --coarse-manifest fire-model-data\coarse_manifest_detector_2d_v2.json
```

The `datasets/D-Fire.zip` dataset is not mixed directly into this point-regression
training command because it contains YOLO bounding boxes rather than the
`p_fire` contact-point labels used here. It can be used later for detector
pretraining or a separate bbox branch.

### Independent Home Fire YOLO branch

The 1.9 GB dataset is kept separate from the point-regression detector:

```text
Home Fire YOLO bbox -> selected fire bbox -> bottom-center/bottom-band pixels
                    -> undistortion -> multi-ray GridMap/mesh intersection
                    -> robust 3D estimate + uncertainty -> tracking
```

For this project the class convention is fixed as follows:

| Stage | `0` | `1` |
|---|---|---|
| Original Home Fire annotation | no fire | fire |
| One-class YOLO checkpoint | fire (the only model class) | not used |

An image without fire must have an empty YOLO label file; do not create a
fake class-`0` bounding box for it. During preparation, all original class-`1`
boxes are kept and remapped to model class `0`; original class-`0` boxes are
dropped. The source dataset is not modified. Then install Ultralytics in the
project virtual environment and train without touching existing checkpoints:

```powershell
cd D:\LAB\SAM_Experiment
.\.venv\Scripts\Activate.ps1
python -m pip install ultralytics
python prepare_home_fire_dataset.py `
  --zip datasets\D-Fire.zip `
  --source-fire-class 1 `
  --manifest-out working\home_fire_manifest_fire1.jsonl `
  --weak-out working\home_fire_weak_points_fire1.jsonl `
  --summary-out working\home_fire_summary_fire1.json `
  --preview-out working\home_fire_preview_fire1.jpg

python train_home_fire_detector.py `
  --dataset-root datasets\D-Fire `
  --model yolo11n.pt `
  --epochs 50 --imgsz 640 --batch 16 --device 0 `
  --single-fire-class 1 `
  --single-class-name fire `
  --project working\home_fire_yolo --name bbox640
```

The resulting checkpoint has exactly one class: model class `0 = fire`.
Benchmark against the original labels with the two ids stated explicitly:
`--source-fire-class 1` for the source annotation and
`--model-fire-class 0` for predictions.

```powershell
python compare_yolo_branch.py `
  --dataset-root datasets\D-Fire `
  --checkpoint working\home_fire_yolo\bbox640\weights\best.pt `
  --source-fire-class 1 --model-fire-class 0 `
  --split test --max-images 100 --device 0 `
  --output-dir output\home_fire_yolo

python export_yolo_coarse_manifest.py `
  --dataset-root datasets\D-Fire `
  --checkpoint working\home_fire_yolo\bbox640\weights\best.pt `
  --model-fire-class 0 --split test --max-images 0 --device 0 `
  --output working\home_fire_yolo_test_coarse.jsonl
```

The preview uses green for YOLO ground truth, red for the prediction, blue
for bottom-center, and orange for bottom-band candidates. These are weak
contact hypotheses: a bbox lower edge is not guaranteed to be the physical
floor contact of a flame. Before training ROI with them, compare against
manually checked contact points and reject truncated/small boxes or samples
with high 3D ray spread.

For the full independent branch, use a measured calibration JSON and a metric
room mesh. The command below does not use the synthetic fallback camera or
the legacy `main_localization.py` workflow:

```powershell
python run_home_fire_localization.py `
  --checkpoint working\home_fire_yolo\bbox640\weights\best.pt `
  --fire-class 0 `
  --samples datasets\D-Fire\test\images `
  --calibration measured_camera.json `
  --mesh measured_room_mesh.json `
  --output output\home_fire_yolo_3d\results.json `
  --max-images 20 --selection diverse --imgsz 640 --device 0
```

To compare the same YOLO detector against the optional Week-6 ROI refiner,
add `--roi-checkpoint week6_roi_result\best_roi.pth`. The default safety
parameters are deliberately conservative:

```text
max_refine_shift_px = 80
min_refine_confidence = 0.15
refine_blend = 0.75
```

An ROI output beyond the shift threshold or below the confidence threshold is
discarded and the YOLO bottom-center is retained. The accepted point is a
blend of coarse and refined points, so a bad ROI crop cannot silently replace
the detector geometry. These pixel guards reduce catastrophic failures; they
do not replace a real calibration, mesh and 3D ground-truth evaluation.

For a calibrated scene, the adapter can feed the existing geometry stage:

```python
from camera_calibration import CameraCalibration
from home_fire_detector_adapter import HomeFireYOLO
from localization import localize_pixels
from mesh_loader import load_triangle_mesh

detector = HomeFireYOLO("working/home_fire_yolo/bbox640/weights/best.pt", fire_class=0)
detection = detector.detect(image)
pixels = detection.contact_pixels(columns=5) if detection.detected else []
calibration = CameraCalibration.from_json("measured_camera.json")
result = localize_pixels(calibration.geometry(), load_triangle_mesh("room_mesh.json"), pixels)
```

This is not a physical metre-level evaluation until the camera calibration,
metric room mesh and 3D fire ground truth use the same coordinate frame.

### Direct commands

```powershell
python main_localization.py `
  --samples datasets\fire-detection-from-cctv\data\data\img_data\test\fire `
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
