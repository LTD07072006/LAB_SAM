# Archived files

Các file trong thư mục này là phiên bản cũ hoặc tool phụ không nằm trên
workflow Week 6 hiện tại. Chúng được chuyển nguyên trạng để giữ lịch sử và
có thể khôi phục khi cần; không file nào bị xóa.

## Legacy entry points and detector

- `main.py`: entry point cũ, dùng camera giả lập riêng và không có ROI refiner.
- `main_integrated.py`: baseline tích hợp trước khi tách detector adapter và
  pipeline `main_localization.py`.
- `detector.py`: adapter YOLO cũ, không được workflow hiện tại sử dụng;
  detector upstream hiện được chuẩn hóa qua `detector_adapter.py` và
  `fire_detector.py`.

## Legacy benchmark/test scripts

- `benchmark.py`
- `benchmark_test_manifest.py`
- `detector_hidden_localization_test.py`
- `hidden_localization_test.py`

Các script này giữ lại các thí nghiệm benchmark/hidden-ground-truth cũ. Đánh
giá chính hiện dùng `evaluation_3d.py`.

## Legacy data and visualization tools

- `build_test_manifest.py`
- `label_gt.py`
- `simulate_2d.py`
- `visualize_3d.py`

Đây là các tool chuẩn bị dữ liệu, mô phỏng hoặc vẽ kết quả không bắt buộc để
train/chạy pipeline mới.

## Legacy notebooks

- `sam-3.ipynb`: notebook train cũ.
- `sam-week6-spatial-train.ipynb`: notebook spatial detector cũ; workflow mới
  dùng `sam-week6-roi-train-complete.ipynb` vì mục tiêu là định vị 3D sau
  detector.
- `sam-week6-roi-train.ipynb`: bản notebook ROI rút gọn đã được thay thế bởi
  `sam-week6-roi-train-complete.ipynb`, bản có kiểm tra input/leakage, tạo
  coarse manifest, train, đánh giá và smoke test đầy đủ hơn.
- `sam-detector2d-train-640-kaggle.ipynb`: notebook detector 640 cũ; đã được
  thay bằng `sam-detector2d-train-640-kaggle-datazip.ipynb`, bản xử lý dataset
  lớn chỉ chứa `data.zip`.

## Recently archived experimental utilities

- Các pilot/audit Python cũ của nhánh YOLO/ROI đã được loại khỏi checkout;
  các artifact dữ liệu archive vẫn giữ nguyên.

## Archived provisional 224x224 assets

- `calibration_test_224.json`: calibration mô phỏng cũ cho bộ test 224x224;
  chỉ giữ để tái lập các benchmark lịch sử.
- `floor_mesh_test_224.json`: floor mesh mô phỏng cũ tương ứng với calibration
  224x224.
- `ground_truth.json`: tập điểm pixel nhỏ của benchmark cũ; evaluator legacy đã
  được trỏ vào đường dẫn archive.

Các file này được di chuyển nguyên trạng, không bị xóa. Workflow mới dùng
`working/synthetic_fire_3d_v3`, mesh/calibration đo thật hoặc các manifest mới.

## Archived synthetic and benchmark artifacts

Thư mục `generated_benchmarks/` chứa các bản sinh thử nghiệm cũ được chuyển
nguyên trạng để tránh làm đầy thư mục runtime. Đây là artifact có thể tái tạo,
không phải input bắt buộc của workflow hiện tại:

- `generated_benchmarks/working/`: các phiên bản synthetic fire 3D cũ,
  benchmark smoke và asset-backed ReplicaCAD cũ.
- `generated_benchmarks/output/`: các kết quả benchmark/visualization cũ,
  gồm `summary.json`, `comparison_metrics.json`, `sequence_metrics.json`,
  contact sheet PNG, HTML và PLY nếu có.

Bản đang dùng để kiểm tra workflow hiện tại vẫn nằm tại
`working/synthetic_fire_3d_v3/` và `output/workflow_final_smoke_20261007/`.
Việc archive chỉ di chuyển artifact; không xóa dữ liệu và không thay đổi các
checkpoint, ảnh/video hoặc mã nguồn Python.

## File vẫn giữ ở thư mục gốc

Các entry point hiện tại của Week 6 là `run_week6.py` và
`compare_v3_roi.py`; chúng không nằm trong archive. `v3_detector.py` là
adapter cho checkpoint FPN v3 và cũng được giữ ở thư mục gốc.

`sam-experiment-code/` và `sam-experiment-code.zip` vẫn được giữ ở thư mục
gốc vì chúng là gói code dùng để upload lên Kaggle; không được coi là bản
legacy dù có các module trùng với thư mục gốc.

Nhóm runtime còn lại gồm `train_week6.py`, `narrow_localizer.py`,
`train_roi_localizer.py`, `build_coarse_manifest.py`, `fire_detector.py`,
`detector_adapter.py`, `camera_calibration.py`, `localization.py`,
`locator.py`, `tracking_3d.py`, `temporal_filter.py`, `point_filter.py`,
`main_localization.py`, `evaluation_3d.py`, các file cấu hình/ground truth và
notebook ROI mới. Đây là các thành phần đang dùng cho workflow Week 6.
