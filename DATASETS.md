# Dataset candidates for LAB_SAM

These datasets are candidates for later experiments. Check each dataset's
license and citation requirements before downloading or redistributing data.

## Fire/smoke detection and localisation

1. **D-Fire — fire and smoke object detection**  
   https://github.com/gaiasd/DFireDataset  
   Useful for detector fine-tuning and testing generalisation beyond the
   current CCTV frames. It includes object-level annotations rather than only
   image-level labels.

2. **MIVIA Fire Detection Dataset**  
   https://mivia.unisa.it/datasets/video-analysis-datasets/fire-detection-dataset/  
   Useful for video-based fire detection and temporal false-positive tests.

3. **FASDD — Fire and Smoke Detection Dataset**  
   https://github.com/IRVLab/FASDD  
   Useful for fire/smoke detection under varied backgrounds and scales. Verify
   the repository's current download and license instructions.

4. **Fire-Smoke Dataset (DeepQuestAI)**  
   https://github.com/DeepQuestAI/Fire-Smoke-Dataset  
   Useful as an additional image-level/object-level source for hard-negative
   and smoke-vs-fire experiments.

5. **FLAME — fire image/video data**  
   https://ieeexplore.ieee.org/document/9660576  
   Useful mainly for fire segmentation/detection and temporal behaviour; it is
   not a replacement for an indoor metric 3D dataset.

## Indoor geometry, depth and camera-pose support

6. **ScanNet**  
   https://www.scan-net.org/  
   Useful for indoor RGB-D geometry, room surfaces and mesh/ray-casting
   experiments. Access and redistribution are subject to its terms.

7. **SUN RGB-D**  
   https://rgbd.cs.princeton.edu/  
   Useful for indoor depth, camera calibration conventions and 3D scene
   reasoning. It does not contain fire labels.

8. **TUM RGB-D Dataset**  
   https://cvg.cit.tum.de/data/datasets/rgbd-dataset  
   Useful for validating camera pose, depth and temporal tracking components.

9. **Matterport3D**  
   https://niessner.github.io/Matterport/  
   Useful for indoor mesh/scene geometry experiments. Review the access and
   license restrictions before use.

10. **Replica Dataset**  
    https://github.com/facebookresearch/Replica-Dataset  
    Useful for controlled indoor mesh and camera-pose tests when a synthetic
    room is acceptable.

## Recommended use in this project

```text
D-Fire / MIVIA / FASDD / Fire-Smoke
    → detector robustness and hard negatives

ScanNet / SUN RGB-D / TUM RGB-D / Replica
    → calibration, mesh, ray casting and tracking infrastructure

Measured room + checkerboard + ArUco/AprilTag + fire-marker positions
    → final physical 3D evaluation
```

The public fire datasets should not be mixed directly into the final 3D test
set: they generally lack synchronized metric camera pose, room dimensions and
3D fire ground truth.
