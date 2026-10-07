# Paper-inspired 2D-to-3D experiment

```text
2D detector/noisy query -> mixed ROI refiner -> calibration/undistortion
-> multi-ray room-mesh intersection -> robust 3D aggregation -> tracking
```

The supplied papers motivate calibration-first reconstruction, measured
2D-3D marker pairs with solvePnP, noisy-query refinement, and trajectory
stability. This project uses calibrated ray casting instead of stereo disparity
because the current system has one CCTV view and a known room mesh.

## Synthetic prototype

```powershell
.\.venv\Scripts\Activate.ps1
python paper_workflow_3d.py `
  --dataset working\synthetic_fire_3d_v3 `
  --roi-checkpoint output\roi_domain_experiments\mixed\best_roi.pth `
  --output-dir output\paper_workflow_synthetic `
  --split test --max-records 24 --selection even --device cpu
```

The output contains `summary.json`, `sequence_metrics.json` and a contact
sheet: green GT, blue coarse, orange raw ROI, red weighted blend. Use
`--calibration-source estimated` to stress-test calibration noise. Synthetic
metric results are not physical-room accuracy.

## Reliability ablation

The evaluator reports coarse, raw ROI, safe ROI and blend. Safe ROI falls back
to coarse when the shift exceeds `--max-shift-px` or heatmap confidence is
below `--min-heatmap-confidence`; otherwise the blend weight depends on both
confidence and refined/coarse displacement.

The summary separates the mesh-ray hit rate from the IPM projection success
rate. IPM 3D error is evaluated on `fire_surface=floor` by default, because a
table/cabinet/column contact is not on the Z=0 plane. Use
`--ipm-eval-surface all` only as an explicitly labelled diagnostic.

### Homography/IPM and temporal filters

The `homography_floor.py` module adds a planar baseline:

```text
undistorted pixel -> H_image_to_floor -> (X, Y, Z=0)
```

Run it beside mesh ray casting with `--enable-ipm`. IPM is valid for contacts
on the measured floor plane; it should be reported separately from contacts on
tables, cabinets or other surfaces. Use `--temporal-filter ema` for the
existing robust smoother or `--temporal-filter ekf` for the constant-velocity
6-state filter in `ekf_tracker.py`.

For a quick synthetic ablation without neural dependencies:

```powershell
python paper_workflow_3d.py `
  --dataset working\synthetic_fire_3d_v3 `
  --no-roi --enable-ipm --coarse-source manifest `
  --temporal-filter ekf --output-dir output\paper_workflow_ipm_ekf_full `
  --split test --max-records 0
```

This validates the geometry and filtering code. It does not turn a public 2D
fire dataset into a real-room 3D dataset.

## Safe physical-room experiment

Use a coloured marker, LED or small substitute; do not create a real fire.
Measure checkerboard intrinsics, room-marker 3D points, a mesh in the same
coordinate frame, and marker `fire_xyz_world` labels on a held-out CCTV test
split. Start from the three `.example.json` templates. After intrinsic
calibration, compute pose:

```powershell
python calibrate_room_pose_pnp.py `
  --correspondences room_marker_correspondences.json `
  --output measured_camera.json
```

Then run:

```powershell
python paper_workflow_3d.py `
  --dataset measured_manifest_root `
  --calibration-source external --calibration measured_camera.json `
  --mesh measured_room_mesh.json --labels-3d measured_ground_truth_3d.json `
  --coarse-source detector --detector-checkpoint fire-model-data\best.pth `
  --roi-checkpoint output\roi_domain_experiments\mixed\best_roi.pth `
  --output-dir output\paper_workflow_real --split test --max-records 0 --device cuda
```

External calibration/mesh marked `synthetic`, `provisional`, `template` or
`example` is rejected. Report separately pixel MAE/PCK, mesh ray hit rate,
IPM projection success rate, 3D MAE/median/P95,
absolute X/Y/Z errors, thresholds 0.10/0.25/0.50/1.00 m, latency, fallback,
uncertainty and per-video jitter/jumps. Never mix synthetic and held-out CCTV
accuracy in one table.

## If the existing 2D dataset is given metric 3D coordinates

`dataset_labels (1).json` already contains normalized `p_fire` values. It does
not contain metric 3D locations, so it cannot calibrate a camera by itself.
Add a separate `room_anchor_xyz.json` whose records identify the same physical
point in each image:

```json
{
  "metadata": {
    "status": "measured_real_room",
    "camera_id": "cctv_01",
    "units": "metres"
  },
  "records": [
    {
      "image": "img_103.jpg",
      "anchor_id": "floor_A",
      "xyz": [1.0, 0.0, 0.0],
      "camera_id": "cctv_01"
    }
  ]
}
```

Build the 2D-3D file from the existing labels:

```powershell
python build_pnp_correspondences.py `
  --labels "fire-model-data\dataset_labels (1).json" `
  --dataset-root datasets\fire-detection-from-cctv `
  --anchors room_anchor_xyz.json `
  --camera-id cctv_01 `
  --output working\pnp_cctv_01.json
```

Then solve the pose after supplying the intrinsic matrix from checkerboard
calibration:

```powershell
python calibrate_room_pose_pnp.py `
  --correspondences working\pnp_cctv_01.json `
  --status measured_real_room `
  --output measured_camera.json
```

The output reports reprojection error. Use at least 6-10 well-spread points;
four coplanar floor points are mathematically possible but less stable and can
have pose ambiguity. If the CCTV is fixed, pairs from multiple images may be
pooled under one `camera_id`. If the camera moves, solve a separate pose per
camera/frame group.

For a synthetic test, use the same command with anchors generated from the
synthetic manifest and pass `--status synthetic_validation`. That validates the
PnP implementation only; it is not evidence of real-room calibration.

## Export 3D figures and an interactive model

After `paper_workflow_3d.py` has created `summary.json`, export the same scene
as static figures and a rotatable HTML model:

```powershell
python visualize_3d_results.py `
  --summary output\paper_workflow_synthetic\summary.json `
  --dataset working\synthetic_fire_3d_v3 `
  --output-dir output\paper_workflow_synthetic\visualization `
  --max-records 24 --write-ply
```

The output contains:

- `scene_3d_overview.png`: perspective room view;
- `scene_3d_top_view.png`: top view;
- `scene_3d_side_view.png`: side view;
- `scene_3d_interactive.html`: rotate/zoom/pan Plotly model;
- `scene_3d_annotations.ply`: optional coloured mesh and points for a 3D
  viewer such as CloudCompare or MeshLab;
- `visualization_manifest.json`: paths, warnings and source metadata.

The colours are fixed across all views: green ground truth, blue coarse,
orange ROI, red weighted blend, purple camera. A synthetic result is labelled
as metric synthetic geometry; it must not be presented as measured-room
accuracy. For a measured-room summary, pass
`--calibration-source external --calibration measured_camera.json` and use a
measured mesh.
