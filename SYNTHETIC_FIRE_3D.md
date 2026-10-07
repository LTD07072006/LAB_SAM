# Synthetic fire 3D dataset branch

Nhánh này tạo dữ liệu tổng hợp có tọa độ mét để kiểm thử workflow:

```text
fire XYZ + camera K/R/C + room mesh
    -> ảnh RGB procedural
    -> p_fire 2D sạch + bbox
    -> nhiễu detector/calibration có kiểm soát
    -> ray casting
    -> sai số 3D theo mét
```

Đây là dataset geometry/robustness, không phải dữ liệu CCTV thật. Kết quả
synthetic không được báo cáo như độ chính xác trong phòng thật.

## 1. Sinh bộ chính

Bộ v3 dùng 240 scene độc lập, mỗi scene 4 frame (960 ảnh), split theo scene
để các frame gần nhau không rơi vào train và test. Mesh mặc định có sàn và
vật cản cuboid; `fire_event` và `has_fire` được tách riêng.

```powershell
cd D:\LAB\SAM_Experiment
.\.venv\Scripts\Activate.ps1
python synthetic_fire_3d.py `
  --output-dir working\synthetic_fire_3d_v3 `
  --scenes 240 `
  --frames-per-scene 4 `
  --image-size 640 640 `
  --mesh-profile obstacles `
  --fire-surface mixed `
  --seed 17 `
  --preview-count 12
```

Nếu chỉ cần smoke test:

```powershell
python synthetic_fire_3d.py `
  --output-dir working\synthetic_fire_3d_v3_smoke `
  --scenes 6 --frames-per-scene 2 `
  --image-size 320 320 --mesh-profile room_obstacles `
  --fire-surface mixed --seed 17 --preview-count 6
```

Các profile mesh:

- `floor`: tương thích bộ v1/v2, chỉ có sàn;
- `obstacles`: sàn + tủ/bàn/cột, phù hợp benchmark ray miss/occlusion;
- `room`: sàn + tường sau + hai tường bên + trần;
- `room_obstacles`: kết hợp phòng kín và vật cản.

## 2. Ý nghĩa nhãn

| Field | Ý nghĩa |
|---|---|
| `has_fire` | nhãn quan sát 2D: `1` có fire nhìn thấy, `0` không có điểm 2D hợp lệ |
| `fire_event` | có đám cháy vật lý trong scene, kể cả khi bị che |
| `fire_visible` | điểm tiếp xúc nhìn thấy từ camera |
| `p_fire_pixel` | điểm chân lửa sạch, dùng làm ground truth 2D |
| `p_fire_noisy_pixel` | đầu ra detector/ROI giả lập, không phải ground truth |
| `fire_xyz_world` | điểm 3D theo mét trong hệ tọa độ phòng |
| `camera.K` | ma trận nội tại thật |
| `camera.R_world_to_camera` | quay world → camera |
| `camera.camera_position` | tâm camera thật |
| `camera_estimated` | calibration/pose có nhiễu đưa vào localizer |
| `bbox_xyxy`, `bbox_yolo` | bao lửa chiếu từ hình học 3D |
| `noise` | thông số nhiễu chính xác của record |

Fire bị che được giữ lại với `fire_event=1`, `fire_visible=0`, `has_fire=0`
và không gán `p_fire_pixel`. Không được dùng các record này như positive
point-label cho ROI; chúng phù hợp để kiểm tra classification/visibility.

## 3. Đánh giá ray casting

```powershell
python evaluate_synthetic_fire_3d.py `
  --dataset working\synthetic_fire_3d_v3 `
  --split test
```

Evaluator báo cáo bốn nhánh:

1. clean point + camera thật: sanity check, phải gần `0 m`;
2. noisy point + camera thật: ảnh hưởng detector/ROI;
3. clean point + camera ước lượng: ảnh hưởng calibration/pose;
4. noisy point + camera ước lượng: điều kiện gần pipeline thực tế.

Ngoài MAE/median/P95, kết quả có ray-hit rate, ngưỡng `0.05/0.10/0.25/0.50/1 m`
và sai số từng trục. Với mesh có vật cản, một ray có thể chạm nhầm bề mặt;
đó là failure cần giữ lại, không nên lọc để làm đẹp số liệu.

## 4. Benchmark nhiều mức nhiễu trên cùng scene

Script này không sinh lại ảnh. Tất cả profile dùng cùng manifest và mesh;
random direction cũng được cố định theo từng record.

```powershell
python benchmark_synthetic_noise.py `
  --dataset working\synthetic_fire_3d_v3 `
  --split test `
  --point-noise-px 0 1 3 5 10 `
  --calibration-scale 0 1 2 `
  --output working\synthetic_fire_3d_v3\noise_benchmark_test.json
```

`calibration-scale=1` tương ứng với noise mặc định của generator; scale 0 là
oracle calibration; scale 2 là stress test. Khi đưa vào báo cáo cần ghi rõ số
scene/fire-visible và không suy diễn kết quả synthetic thành kết quả CCTV.

## 5. Renderer Blender tùy chọn

Generator procedural đủ để kiểm thử hình học. Khi cần ảnh gần thực tế hơn,
chạy adapter bằng Blender; adapter vẫn giữ nguyên camera/XYZ/mesh từ manifest
và thêm RGB, depth, mask:

```powershell
blender -b --python blender_render_synthetic_fire.py -- `
  --dataset working\synthetic_fire_3d_v3 `
  --split test --max-images 24 `
  --output-dir working\synthetic_fire_3d_v3\blender_test
```

Nếu chưa cài Blender, bỏ qua bước này; không ảnh hưởng generator/evaluator.
Đây là renderer có kiểm soát, chưa phải mô phỏng vật lý đầy đủ của ngọn lửa.

## 6. Dùng cho train và benchmark

- Pretrain ROIRefiner bằng `p_fire_pixel` sạch và input giả lập từ
  `p_fire_noisy_pixel`;
- giữ train/test theo `scene_id`, không chia ngẫu nhiên từng frame;
- fine-tune hoặc kiểm tra lại trên ảnh/video thật có `p_fire` được kiểm tra tay;
- benchmark A/B coarse → ray và coarse → ROI → ray trên cùng calibration/mesh;
- báo cáo riêng `fire_visible` và các ca occlusion/ray miss.

Dataset này không thay thế ground truth 3D phòng thật. Bước xác nhận cuối vẫn
là đo kích thước phòng, calibration checkerboard/AprilTag, mesh theo cùng hệ
tọa độ và gán `fire_xyz_world` cho một số vị trí kiểm thử an toàn.
