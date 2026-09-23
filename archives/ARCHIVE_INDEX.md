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
  dùng `sam-week6-roi-train.ipynb` vì mục tiêu là định vị 3D sau detector.

## File vẫn giữ ở thư mục gốc

Nhóm còn lại gồm `train_week6.py`, `narrow_localizer.py`,
`train_roi_localizer.py`, `build_coarse_manifest.py`, `fire_detector.py`,
`detector_adapter.py`, `camera_calibration.py`, `localization.py`,
`locator.py`, `tracking_3d.py`, `temporal_filter.py`, `point_filter.py`,
`main_localization.py`, `evaluation_3d.py`, các file cấu hình/ground truth và
notebook ROI mới. Đây là các thành phần đang dùng cho workflow Week 6.
