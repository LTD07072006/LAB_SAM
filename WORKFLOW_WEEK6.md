# Week 6: 2D detector to 3D fire localisation

The upstream detector remains responsible for fire/no-fire classification and
coarse 2D evidence. The new contribution is the narrow-scene localisation
stage:

```text
detector -> detector_adapter -> ROIRefiner -> calibration/undistortion
         -> bottom contact pixels -> multi-ray GridMap intersection
         -> MAD outlier filter + covariance -> optional 3D tracker
```

## Stage 0: improve the 2D detector before ROI

The detector stage is independently trainable. Do not replace the upstream
checkpoint before measuring the new detector on the same held-out test split:

```text
image -> MobileNetV4 reduction-4/8/16 FPN -> high-resolution heatmap
      -> soft-argmax coarse point + fire confidence -> ROIRefiner
```

Run it with:

```powershell
python train_detector_2d.py `
  --labels "fire-model-data\dataset_labels (1).json" `
  --dataset-root fire-detection-from-cctv `
  --output-dir fire-model-data\detector_2d_v2 `
  --image-size 640 --batch-size 8 --epochs 35 --device cuda
```

The trainer keeps official test images out of training, updates the point when
horizontal flipping is applied, and maps the prediction back from letterboxed
coordinates to the original image. It writes `best_detector_2d.pth`,
`last_detector_2d.pth`, `history.json` and `test_metrics.json` under the output
directory. Existing detector, ROI and v3 checkpoints are not overwritten.

Compare before passing the model downstream:

```powershell
python compare_detector_2d.py `
  --new fire-model-data\detector_2d_v2\best_detector_2d.pth `
  --split test --max-images 50 --device cuda
```

If the new branch is better, generate coarse points with that checkpoint and
retrain the ROI refiner against the new coarse-point distribution:

```powershell
python build_coarse_manifest.py `
  --model fire-model-data\detector_2d_v2\best_detector_2d.pth `
  --output fire-model-data\coarse_manifest_detector_2d_v2.json
python train_roi_localizer.py `
  --init-checkpoint fire-model-data\best.pth `
  --coarse-manifest fire-model-data\coarse_manifest_detector_2d_v2.json
```

The 1.9 GB home-fire set is not silently merged here: its YOLO bbox labels do
not have the same point-label semantics. Use it separately for bbox
pretraining or an adapter experiment.

## Train the ROI refiner

For the complete Kaggle procedure, use
`sam-week6-roi-train-complete.ipynb`. It contains the 16 checked steps for
runtime/input validation, leakage checks, coarse-manifest generation, ROI
training, metrics, checkpoint smoke test, and artifact packaging.

Kaggle may mount attached datasets under
`/kaggle/input/datasets/<owner>/<dataset>`. The complete notebook searches
recursively and matches images against `dataset_labels (1).json`; it does not
assume that a dataset is directly under `/kaggle/input`.

The labels file currently expects images under
`.../data/data/img_data/{train,test}/{fire,smoke,default}/`. The external
`cctv-smoke-and-fire-emergency-detection-dataset` contains an `images/` folder
with `fire_detected_*.png` and `smoke_detected_*.png`; that dataset has no
matching `p_fire` labels and must not be selected as the ROI regression set
unless separate point/bbox/mask annotations are added. The local
`fire-detection-from-cctv/data/data/img_data` tree, or a ZIP containing the
same relative `train/.../img_*.jpg` paths, is the correct input for this
training stage.

For Kaggle, the ready-to-upload archive
`fire-localization-img-data.zip` contains `img_data/train` and
`img_data/test` with `fire`, `smoke`, and `default` folders. Attach it in
addition to the code/model datasets if the complete
`fire-detection-from-cctv` tree is not uploaded.

The first command makes training coarse points from the existing checkpoint.
It does not modify `best.pth`.

```powershell
python build_coarse_manifest.py --model fire-model-data/best.pth
python train_roi_localizer.py --init-checkpoint fire-model-data/best.pth `
  --coarse-manifest fire-model-data/coarse_manifest.json
```

If the upstream detector is not available, omit `--coarse-manifest`; the
trainer uses controlled synthetic point errors as a fallback experiment.

## Run the integrated pipeline

For independent still images, tracking is disabled. For ordered video frames,
add `--sequence` so temporal smoothing and 3D gating are enabled.

```powershell
python main_localization.py `
  --model fire-model-data/best.pth `
  --roi-checkpoint week6_roi_result/best_roi.pth `
  --calibration camera_calibration.example.json
```

The example calibration file is only a schema/example. It must be replaced by
calibration measured for the actual camera and GridMap coordinate frame.

## Evaluation

```powershell
python evaluation_3d.py --mode oracle --images fire-samples
python evaluation_3d.py --mode noisy --pixel-sigma 5
python evaluation_3d.py --mode detector
```

The repository currently has only manual 2D points for a small set of images.
Absolute 3D accuracy is reported only when a physical `--labels-3d` JSON is
provided; otherwise use the oracle/noisy modes for sensitivity analysis and
the pixel/refiner metrics for model comparison.
