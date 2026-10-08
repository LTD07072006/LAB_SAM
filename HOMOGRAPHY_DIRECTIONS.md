# Homography/IPM directions for post-ROI fire localisation

This document records the homography branches added after the ROI point and
their evaluation on the metric synthetic datasets already present in this
project. The source datasets and existing output folders are not modified by
the benchmark scripts.

## What homography can and cannot solve

A homography maps an image point to one known planar surface. It is therefore
an appropriate alternative to full ray casting when the fire contact point is
known to lie on the floor, a table top, a cabinet top, or another calibrated
plane. It does not reconstruct an arbitrary 3D point without a plane choice.

The recommended input remains:

```text
coarse detector point or ROI blend point
    -> camera undistortion
    -> floor IPM / multi-plane homography
    -> validity and outlier checks
    -> EMA or EKF on a video sequence
    -> XYZ + PNG/HTML/PLY
```

The mesh is used by the visibility-aware selector only to identify which
horizontal plane is visible. The returned XYZ is still obtained by inverting
the selected plane's homography. This keeps the comparison separate from the
full ray-casting branch.

## Implemented branches

### `benchmark_homography.py`

This is the floor-plane benchmark. It compares:

- analytic homography from camera calibration;
- DLT from four floor anchors;
- DLT from all floor anchors;
- RANSAC homography with synthetic anchor noise and outliers;
- scene-static homography;
- full-mesh ray casting and floor-only ray casting as geometry references.

It is the right script for studying calibration-anchor noise and whether a
real room can be represented by one floor homography.

### `benchmark_multiplane_homography.py`

This script adds the multi-plane branches:

- `floor_ipm`: always map to the continuous floor plane;
- `surface_oracle`: use the manifest `fire_surface` label; upper bound only;
- `plane_bank_nearest`: try horizontal plane patches and choose the nearest
  valid candidate;
- `plane_bank_visible`: cast one visibility ray, identify the first horizontal
  mesh face, then invert that face's homography;
- `plane_bank_visible_floor_fallback`: use the visible plane when it can be
  identified, otherwise use floor IPM;
- the same branches with `_ema` for sequence smoothing and jump gating.

The extractor merges tessellated floor tiles into one continuous floor patch.
This is important for ReplicaCAD-style assets, where duplicated tile vertices
can otherwise make one physical floor look like hundreds of unrelated planes.

## Datasets used

### Procedural metric synthetic v3

```text
working/synthetic_fire_3d_v3
```

The test split has 92 frames with fire contacts on the floor, central cabinet,
left table and right column. It includes metric `fire_xyz_world`, calibrated
camera parameters, noisy detector pixels and the clean pixel oracle.

### ReplicaCAD asset-backed subset

```text
working/asset_backed_replicacad_fire_3d_fast
```

This subset has 12 test frames and all fire contacts are on the floor. It is a
downloaded room asset rendered with a metric camera; it is not real CCTV and
must not be described as a real-world accuracy result.

## Reproduce the current tests

Run from the project virtual environment:

```powershell
cd D:\LAB\SAM_Experiment
.\.venv\Scripts\Activate.ps1
```

Synthetic v3 with the current ROI prediction summary:

```powershell
python benchmark_multiplane_homography.py `
  --dataset working\synthetic_fire_3d_v3 `
  --prediction-summary output\workflow_roi_cpu_regularized_test_full_20261007\summary.json `
  --split test --selection even `
  --output-dir output\multiplane_homography_20261007\synthetic_v3_homography_rerun
```

ReplicaCAD asset-backed floor test:

```powershell
python benchmark_multiplane_homography.py `
  --dataset working\asset_backed_replicacad_fire_3d_fast `
  --mesh working\asset_backed_replicacad_fire_3d_fast\room_mesh.json `
  --split test --selection even `
  --output-dir output\multiplane_homography_20261007\replicacad_homography_rerun
```

For a fast smoke test, add `--max-records 24 --max-visual-records 24`.
The script refuses to overwrite a non-empty output directory; choose a new
directory for every rerun.

## Results obtained

The final full benchmark is saved under
`output/homography_workflows_full_20261008/`. It contains both the floor-only
ray/IPM protocol and the multi-plane protocol, with the same split and input
predictions for each branch. MAE, median and P95 are Euclidean 3D error in
metres. `valid/total` is the fraction with a finite estimate and ground truth.
The machine-readable aggregate is
`output/homography_workflows_full_20261008/combined_summary.json`; the readable
report is `output/homography_workflows_full_20261008/combined_report.md`.

### Synthetic v3, ROI blend, 92-frame test split

| Branch | Valid/total | MAE | Median | P95 | Under 0.25 m | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| Floor IPM | 88/92 (95.7%) | 5.113 m | 0.159 m | 11.824 m | see `summary.json` | about 0.9 ms |
| Plane bank nearest | see `summary.json` | see `summary.json` | see `summary.json` | see `summary.json` | see `summary.json` | see `summary.json` |
| Visible plane | 79/92 (85.9%) | 0.140 m | 0.112 m | 0.370 m | see `summary.json` | about 1.7 ms |
| Visible plane + EMA | 79/92 (85.9%) | 0.109 m | 0.090 m | 0.278 m | see `summary.json` | about 1.7 ms |
| Surface oracle | see `summary.json` | see `summary.json` | see `summary.json` | see `summary.json` | see `summary.json` | see `summary.json` |

The floor-IPM error is intentionally large because this test contains
non-floor contacts. The visible-plane result is strong on this controlled
synthetic scene, but it is not yet a claim that a detector can infer semantic
surface identity in an unconstrained CCTV image.

### ReplicaCAD asset-backed subset, coarse point, 12-frame test split

| Branch | Valid/total | MAE | Median | P95 | Under 0.25 m | Mean latency |
|---|---:|---:|---:|---:|---:|---:|
| Floor IPM | 12/12 (100%) | 0.111 m | 0.110 m | 0.204 m | 100% | 0.9 ms |
| Floor IPM + EMA | 12/12 (100%) | 0.052 m | 0.043 m | 0.105 m | 100% | 0.9 ms |
| Plane bank nearest | 12/12 (100%) | 0.238 m | 0.118 m | 0.680 m | 75% | about 350 ms |
| Visible plane | 0/12 | not available | not available | not available | not available | about 4.3 s |
| Visible plane + floor fallback | 12/12 (100%) | 0.111 m | 0.110 m | 0.204 m | 100% | see `summary.json` |

The ReplicaCAD mesh contains many non-horizontal or semantically unlabelled
asset faces. The visible-plane failure is therefore useful diagnostic
information: it shows that a downloaded mesh needs semantic surface labels or
an explicit floor prior before visibility selection can be trusted. Since all
ReplicaCAD fire contacts in this subset are floor contacts, floor IPM is the
correct primary method for this particular subset.

## How to interpret the results

1. Use floor IPM as the fast baseline when the application contract says the
   fire contact is on the floor.
2. Use multi-plane homography when the scene contains several known planar
   support surfaces. A surface classifier, segmentation mask, or object/plane
   prior is needed to select the plane without ground-truth metadata.
3. Keep `surface_oracle` only as an ablation upper bound. It reads the true
   surface label and is not an end-to-end deployment result.
4. Use `plane_bank_visible` as a geometry diagnostic. It can be useful when a
   calibrated mesh is clean, but it is still a visibility-assisted selector,
   not a replacement for scene understanding.
5. Use the fallback branch only when a valid floor-contact prior is available.
   On a mixed floor/table/cabinet scene, blindly falling back to the floor can
   create metre-level errors.
6. Apply EMA/EKF only after checking ray/plane validity and rejecting large
   jumps. Smoothing cannot repair a wrong plane choice.

## Next development steps

The best next implementation order is:

1. Add a real-scene floor calibration interface: four or more measured floor
   anchors, DLT/RANSAC, reprojection error and a saved `H_floor`.
2. Add plane groups to the room manifest: semantic name, normal, height,
   polygon boundary and calibration confidence. Do not expose individual
   ReplicaCAD mesh triangles to the selector.
3. Add a plane-confidence head or rule from detector/ROI output. A bbox alone
   is not enough for table-versus-floor disambiguation; a mask or contact-band
   feature is preferable.
4. Add a hybrid policy: floor IPM first, multi-plane homography if a plane is
   confidently selected, and full ray casting only when the plane is unknown or
   non-planar.
5. Compare all branches on a scene-separated test protocol and report pixel
   MAE/PCK, plane-selection rate, 3D MAE/median/P95, XYZ MAE, threshold rates,
   latency and temporal jitter.

The existing `visualize_3d_results.py` and the multi-plane benchmark export
PNG, interactive HTML and PLY. These visualisations show the room mesh,
ground-truth points and homography estimates; they do not turn synthetic
coordinates into real CCTV measurements.
