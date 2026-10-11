# Calibration targets for the physical Ray-Casting workflow

This folder defines the printable targets for the measured-room experiment.
It does not modify the existing datasets, checkpoints or benchmark outputs.

## Recommended ArUco settings on tagsgen.top

Use the **ARUCO TAG** tab with these values:

| Setting | Value | Reason |
|---|---:|---|
| Dictionary | `DICT_6X6` (verify preview says `DICT_6X6_1000`) | Matches the current tagsgen.top UI |
| Tag IDs | `0` through `7` | Eight independently identifiable room anchors |
| Tag size | `120 mm` black square | Easier detection from an iPhone at a few metres |
| White margin | `20 mm` minimum | Prevents the tag from touching nearby graphics |
| Export | `SVG` or `PDF` | Keeps the physical size accurate when printed |
| Scale in the print dialog | `100%` / actual size | Never use “fit to page” |

The **tag size is the side of the black ArUco square**, not the whole sheet.
Print one tag per A4 page, attach it to a rigid flat board, and measure the
printed black square with a ruler. If the measured value is not 120 mm, record
the measured value and use that value in the room-marker layout.

The current page exposes the dictionary family as `DICT_6X6 (50, 100, 250,
1000)` and its preview resolves to `DICT_6X6_1000`. Keep that exact dictionary
when detecting the exported tags. Do not generate the tags on the website and
then configure OpenCV as `DICT_6X6_250`: the family name alone is not enough;
the dictionary variant must match. If a future version of the website exposes
a direct 250-entry selector, the whole session may use `DICT_6X6_250`, but do
not mix the two variants.

## Room placement

The eight tags are **not** meant to be arranged as a flat calibration board.
Place them at measured, fixed locations around the camera view:

- four spread across the floor/wall region;
- two on a side wall or vertical board;
- two at a different height when possible.

The goal is a well-spread 3D set of corners for `solvePnP`, not a decorative
pattern. Use the marker corner coordinates in the room coordinate frame. The
origin and axes must be documented, for example:

```text
origin = rear-left floor corner
X = right, Y = into the room, Z = upward
units = metres
```

Keep the tags fixed after the camera pose has been solved. If one tag is
occluded, use at least six visible tags. A four-point coplanar solution is
mathematically possible but less stable than six to ten well-spread points.

## Checkerboard for camera intrinsics

The tag generator does not create a checkerboard. Use the project generator
with the checked-in configuration:

```powershell
cd D:\LAB\SAM_Experiment
.\.venv\Scripts\python.exe calibration_targets\generate_calibration_targets.py `
  --output-dir calibration_targets\generated
```

The default checkerboard has **9 x 6 inner corners**, therefore 10 x 7
alternating squares, with 25 mm square size. Its physical board is 250 x 175
mm. The 9 x 6 count is the number to pass to OpenCV's checkerboard detector;
do not pass 10 x 7 as the inner-corner count.

For a very wide room or a distant camera, print the same board on A3 with
35 mm squares. Keep the inner-corner count unchanged and record the actual
square size. The board must be flat; tape it to foam board or thick card.

Capture 15-25 checkerboard images at different positions and tilts across the
whole image. Lock focus, exposure and zoom before capturing them. These images
are used only for intrinsics/distortion, not for fire detection training.

## What is and is not integrated in the current code

The existing project already consumes:

```text
camera intrinsics + distortion -> undistortion
room marker 2D/3D correspondences -> solvePnP -> measured_camera.json
measured camera + measured mesh + fire_xyz_world -> Ray Casting benchmark
```

The existing [calibrate_room_pose_pnp.py](../calibrate_room_pose_pnp.py) does
not automatically detect ArUco tags; it consumes correspondence points. The
printed ArUco IDs therefore identify the measured room points and make manual
or future automatic corner extraction unambiguous. The generated files in this
folder do not claim physical calibration until the board dimensions, marker
locations and camera parameters have been measured.

## Files

- `aruco_tagsgen_config.json`: exact settings for tagsgen.top.
- `checkerboard_config.json`: checkerboard geometry and capture protocol.
- `generate_calibration_targets.py`: generates SVG printables for both target
  types; it does not touch existing project data.

