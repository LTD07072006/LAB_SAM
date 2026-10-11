# Nghiên cứu nút thắt mục 1.2 và hướng nâng cấp

Báo cáo này đối chiếu mục 1.2 trong báo cáo tuần 8 với workflow hiện tại và chạy lại các phép đo trên hai protocol có ground truth phù hợp:

- CCTV thực: 36 ảnh dương tính trong test, chỉ có nhãn pixel 2D.
- Database synthetic v3: 240 scene, 960 frame 640×640, test 144 frame; 92 frame nhìn thấy lửa, trong đó 64 frame có điểm tiếp xúc trên mặt phẳng sàn.

Kết quả được sinh bởi [benchmark_section_1_2.py](./benchmark_section_1_2.py), với cache Depth Anything đã chạy sẵn. Artifact chính là [summary.json](../output/section_1_2_research_20261010/summary.json), biểu đồ [section_1_2_benchmark.png](../output/section_1_2_research_20261010/section_1_2_benchmark.png), và dữ liệu crop chi tiết [crop_audit.json](./crop_audit.json).

## Kết luận cho mục 1.2

Nút thắt đo được là **sai phân bố của coarse detector và overconfidence**, sau đó mới đến khả năng hiệu chỉnh 3D. Trên cùng 36 ảnh và cùng checkpoint `cpu_regularized_mixed`:

| Protocol | Coarse MAE | ROI MAE | ROI PCK@10 | ROI PCK@25 |
|---|---:|---:|---:|---:|
| Coarse mô phỏng Gaussian 3 px | 3.7818 px | 4.3291 px | 100% | 100% |
| Coarse `best.pth` thực tế | 9.4066 px | 9.6270 px | 75% | 97.22% |

Khoảng cách ROI là **5.2979 px**. Đây là hai protocol khác nhau: số 4 px là đầu vào dễ được mô phỏng quanh ground truth, còn số 9.6 px là chuỗi detector thật → ROI. Không dùng số 4 px để đại diện cho end-to-end.

### Kiểm tra giả thuyết “nhãn rơi ngoài crop”

Code crop dùng `roi_fraction=0.70`, side bằng 70% cạnh ngắn và padding đen khi crop vượt ảnh. Audit lại từng ảnh cho thấy **36/36 nhãn vẫn ở trong crop**. Ảnh `img_243.jpg` là outlier: confidence detector 0.9984 nhưng coarse lệch 67.5943 px; nhãn còn cách mép crop gần nhất 11 px. Các coarse error 14–21 px còn margin ít nhất khoảng 58 px. Vì vậy, log cũ mô tả “GT rơi khỏi ROI” không được chứng minh bởi crop hiện hành; hướng sửa đúng là học với actual coarse, giám sát tail và ước lượng uncertainty. Confidence detector hiện tại không đủ để làm uncertainty định vị; tương quan Pearson confidence–error khoảng −0.506 và vẫn có nhiều outlier confidence >0.99.

### Kết quả fix upstream đã chạy trên database hiện có

Đã thử hai hướng trước ROI. Detector FPN `detector_2d_fpn_v2` được khởi tạo từ
backbone `best.pth` nhưng huấn luyện prototype trên ảnh 224×224 không học được
tọa độ: sau 10 epoch, test coarse MAE là `110.10 px`, PCK@10 là `0%`. Nhánh
này không được đưa vào workflow.

Phương án giữ checkpoint legacy và thêm photometric TTA đã được chạy lại theo
split leakage-safe. `brightness3` dùng ảnh gốc, sáng hơn 15% và tối hơn 15%,
đồng thời tính `stability_px` là độ phân tán điểm giữa các view. Đây là chỉ báo
reliability hình học, không phải xác suất đúng. Trên positive test 36 ảnh,
TTA không gate giảm coarse MAE `9.4066 -> 7.9312 px`; sau ROI CPU-regularized
mixed, MAE giảm `9.6270 -> 8.4567 px`. Benchmark đầy đủ giữ PCK@10 ở `75%`
và PCK@25 ở `97.22%`; các số paired khác phải được xem là protocol phụ.

Gate `max_stability_px=6` chỉ được chọn từ validation. Nó giữ cảnh báo fire
nhưng từ chối tọa độ không ổn định trước ROI. Trên toàn bộ test 120 ảnh (84
no-fire), classification vẫn `TP=36, TN=84, FP=0, FN=0`. Trong 36 positive,
30 điểm được chuyển tiếp (coverage `83.33%`); raw coarse MAE là `7.9312 px`
và accepted coarse MAE là `6.2409 px`, P95 `12.5894 px`, không có accepted
outlier `>25 px`. ROI CPU-regularized mixed trên cùng coverage có MAE
`6.8609 px`, P95 `13.8073 px`, PCK@10 `83.33%`, PCK@25 `100%`. Sáu mẫu bị
reject phải được báo là miss localisation, không được tính như dự đoán đúng.

Trên validation 71 ảnh (57 positive, 14 no-fire), gate giữ classification
`TP=57, TN=14`, accepted coverage `56/57=98.25%`, accepted coarse MAE
`4.9392 px`, và không có outlier `>25 px`. Đây là lý do dùng ngưỡng 6 px;
không tối ưu ngưỡng từ test.

TTA tăng chi phí CPU đáng kể (benchmark toàn test khoảng `454 ms/frame` cho
detector, trước ROI), nên production nên chạy GPU/ONNX/TensorRT, batch frame
hoặc chỉ kích hoạt TTA khi frame đơn hoặc reliability thấp. `max_stability_px`
phải được calibrate lại khi đổi camera, kích thước ảnh hoặc checkpoint.

Một ablation bổ sung trên cùng split cho thấy `photometric5` (thêm contrast
high/low vào brightness3) đạt test raw MAE `7.8610 px`, accepted MAE
`6.2540 px`, coverage `88.89%`, nhưng validation raw MAE `5.0276 px` và
coverage `98.25%`. So với `brightness3` (`7.9312/6.2409 px` trên test và
`4.8982/4.9392 px` trên validation), 5-view không có lợi ích nhất quán và
tăng số lần suy luận. Vì vậy production giữ `brightness3`; 5-view chỉ là
ablation, không được chọn theo test.

Gate còn kiểm tra đủ số view hợp lệ. Nếu model trả NaN/Inf hoặc chỉ còn một
view có điểm, trạng thái là `incomplete_tta`/`invalid_tta_confidence`, vẫn giữ
cảnh báo fire nhưng không truyền điểm sang ROI. Adapter cũng xóa bbox/mask
khỏi contract downstream khi localization bị reject, tránh một detector khác
vô tình ray-cast điểm hình học cũ.

### Thử nghiệm soft gate / dynamic ROI

Đã thêm `ROIRefinerInference.refine(..., stability_px=..., expansion_factor=...)`
và script [benchmark_soft_roi.py](./benchmark_soft_roi.py). Crop được tính bằng
`base_side + expansion_factor * max(stability_px - threshold, 0)`, có cap
`max_roi_fraction`; khi `expansion_factor=0` hành vi cũ giữ nguyên. TTA FPN
cũng đã được flatten thành một batch `[N * views, C, H, W]` thay vì gọi model
theo từng view.

Kết quả trên database cho thấy dynamic crop đơn thuần không giải quyết nút
thắt hiện tại. Crop 70% đã chứa `57/57` nhãn validation và `36/36` nhãn test.
Với threshold stability 6 px, sweep hệ số 0/4/8/12/16 cho test giữ ROI MAE
quanh `8.33–8.86 px`, PCK@25 `97.22%`, và outlier `img_243` vẫn lớn hơn 55 px.
Validation tốt nhất chỉ khoảng `4.68 px`, gần như tương đương fixed crop
`4.70 px`; multi-scale chọn theo heatmap confidence cũng không tạo lợi ích
đủ lớn để bù chi phí thêm nhiều forward. Vì vậy soft crop được giữ như API
nghiên cứu và fallback có kiểm soát, không bật mặc định.

Diễn giải: lỗi chính không phải GT rơi khỏi ROI mà là refiner chọn sai mode
trong crop dù GT đã nằm bên trong. Để khôi phục coverage/outlier, hướng ưu tiên
là huấn luyện ROI bằng actual coarse manifest, hard-tail như `img_243`, và
uncertainty/heatmap entropy; dynamic crop chỉ là một thành phần hỗ trợ.

### Audit và guard cho ảnh outlier

Audit trực tiếp `img_243.jpg` cho thấy nhãn `[0.741071, 0.928571]` tương đương
`(166, 208)` và trùng với hai bản ghi duplicate; các frame liền kề
`img_240–244` cũng giữ chân lửa quanh vùng x=160–176. Detector TTA riêng
`img_243` nhảy sang x≈110, trong khi ROI refiner chỉ sai lớn vì nó được cấp
đúng tâm sai. Khi đặt tâm ROI gần x=160, refiner trả điểm quanh `(163,208)`.
Do đó chưa có bằng chứng để sửa Ground Truth bằng tay; sửa nhãn lúc này sẽ
che lỗi upstream.

Đã thêm [temporal_outlier_guard.py](../temporal_outlier_guard.py) và benchmark
[benchmark_temporal_coarse_guard.py](./benchmark_temporal_coarse_guard.py).
Guard không dùng nhãn: nếu bước nhảy coarse lớn, `stability_px` cao và ảnh
liền kề vẫn tương đồng, nó giữ điểm trước đó cho frame hiện tại. Với
`max_jump=20 px`, `min_stability=6 px`, mean absolute frame difference ≤25:

- validation: không thay frame nào, coarse/ROI giữ nguyên (`4.898/4.700 px`);
- test: thay đúng 1 frame là `img_243`; coarse MAE `7.9312 → 6.5100 px`,
  ROI MAE `8.4567 → 6.9770 px`, PCK@25 `97.22% → 100%`, max ROI error
  `56.27 → 17.24 px`.

Ngưỡng frame difference 35 thay nhầm `img_225`, nên không dùng. Guard chỉ phù
hợp cho video/chuỗi frame có thứ tự; không bật cho tập ảnh shuffle. Khi triển
khai cần giữ cả `raw_pixel`, `guarded_pixel`, `guard_replaced` và reason để
không biến heuristic temporal thành Ground Truth giả.

### Manifest lọc và sửa outlier

Script [filter_and_repair_outliers.py](./filter_and_repair_outliers.py) đã chạy
trên toàn bộ 517 record. Nó không ghi đè `dataset_labels (1).json` và chỉ dùng
Ground Truth để đánh giá. Guard chỉ bật cho source split `test`; train/val giữ
raw point để tránh đưa heuristic temporal vào dữ liệu học.

Kết quả cuối:

- 272 positive có coarse raw MAE `6.1146 px`, max `56.5167 px`;
- sau guard test-only, MAE `5.9265 px`, max `22.7068 px`, không còn outlier
  >25 px trong audit toàn bộ positive;
- chỉ một điểm được tự động thay: `test/img_243.jpg`;
- 15 train hard-tail được tách theo quantile 90% với ngưỡng `10.9717 px`;
- chỉ một ảnh cần independent annotation review: `img_243.jpg`.

Manifest sửa chỉ là input coarse cho ROI/video inference; nó không phải nhãn
Ground Truth mới. Nếu review thủ công sau này xác nhận nhãn sai, phải lưu bản
ghi cũ, bản ghi mới và lý do trong một file revision riêng.

### Bước huấn luyện tiếp theo

Một run ROI khởi tạo head mới từ actual-coarse manifest đã được thử riêng. Sau
4 epoch CPU, validation MAE vẫn `19.99 px` (mốc checkpoint hiện tại là
`4.70 px`), nên checkpoint thử nghiệm đã bị loại và không được đưa vào test.
Điều này cho thấy cần fine-tune từ checkpoint ROI tốt hiện tại hoặc thay đổi
loss/augmentation có kiểm soát; không thể suy ra mô hình tốt hơn chỉ từ train
loss giảm.

### Lọc toàn bộ coarse issue và thử hard-tail detector

Manifest workflow-aware đã được sinh lại từ toàn bộ 517 record, không chỉ từ
`img_243`: có 272 positive, trong đó 101 train, 27 validation và 27 test chứa
ít nhất một vấn đề (`error > 5/8/10/12/15/20/25 px`, TTA không ổn định, hoặc
confidence cao nhưng lệch lớn). Có 24 mẫu train-only được chọn làm hard-tail
với ngưỡng lỗi tối thiểu 10 px. Validation và test không xuất hiện trong file
trọng số; `img_243` được giữ ở test và không được dùng để học. Ground Truth
gốc không bị ghi đè. Danh sách review dạng CSV nằm ở
`output/section_1_2_pre_roi_20261010/coarse_issue_manifest/coarse_issues_all.csv`.

Đã thêm `--hard-tail-manifest` và `--hard-tail-weight` vào cả
`train_week6.py` và `detector_2d.py`. Trọng số chỉ nhân coordinate/heatmap
loss của positive hard-tail, còn classification loss giữ nguyên để tránh làm
detector thiên lệch. Smoke test xác nhận đúng 24 mẫu được áp trọng số. Một
run FPN 5 epoch CPU ở 224 px đã chạy thành công nhưng validation tốt nhất chỉ
đạt MAE `55.31 px`, PCK@25 `26.32%`; vì vậy checkpoint thử nghiệm
`output/section_1_2_pre_roi_20261010/coarse_fpn_hardtail_224/best_detector_2d.pth`
không được thay cho `fire-model-data/best.pth`. Kết quả này cho thấy đổi kiến
trúc FPN và huấn luyện ngắn trên CPU chưa đủ; cần pretrain/finetune có kiểm
soát ở độ phân giải 640, nhiều epoch hơn và chọn checkpoint bằng validation.

Protocol matrix validation sau các thay đổi vẫn giữ mốc checkpoint hiện tại:
coarse raw MAE `4.8982 px`, ROI raw MAE `4.7001 px`, PCK@25 `100%`, không có
outlier >25 px; temporal guard không thay frame nào ở validation. Do đó
checkpoint ROI cũ tiếp tục được chọn production.

### Tích hợp dataset mới tải

Các file Python ở workspace cha đã được đưa vào `SAM_Experiment`; workspace
`D:\LAB` hiện không còn file `.py` rời của project. Dataset được sắp xếp ở
`datasets/`: D-Fire và FIRE-SMOKE giữ trong thư mục dataset, còn `Test/`,
`Masks/` và video Zenmuse nằm trong `datasets/external_fire_smoke_video/`.
ZIP FIRE-SMOKE trùng hash với bản trong `datasets/` đã được xoá để tránh hai
bản dữ liệu.

Audit mới ghi nhận D-Fire có 11.689 ảnh có YOLO box và FIRE-SMOKE có 3.000
ảnh folder-labelled. Script `prepare_external_coarse_data.py` tạo
`external_coarse_manifest.json`: bottom-centre của box D-Fire được đánh dấu là
weak target, còn Fire/Smoke/Neutral chỉ được đánh dấu classification-only.
Không có mẫu external nào được đưa vào split CCTV validation/test và không có
Ground Truth `p_fire` nào được tạo giả.

Benchmark detector trên 60 mẫu FIRE-SMOKE (10 mẫu mỗi lớp và split) đạt
accuracy `91.67%`; riêng test-fire positive rate là `70%`. Đây là bằng chứng
domain gap, không phải lý do thay checkpoint. Kết quả được lưu tại
`output/dataset_integration_20261010/external_classification_benchmark_10.json`.
Vì coarse validation hiện tại vẫn tốt hơn rất nhiều, checkpoint legacy được
giữ nguyên; dataset mới sẽ được dùng cho auxiliary classification/domain
pretraining và D-Fire weak point pretraining ở một nhánh riêng, sau đó chỉ
nhận checkpoint nếu MAE/PCK trên CCTV validation cải thiện.

### Kết quả run coarse 640×640 trong 40 epoch và ROI fine-tune

Run coarse đã chạy đủ 40 epoch với backbone khởi tạo từ `best.pth`, freeze 3
epoch đầu và hard-tail weight `1.5`. Epoch tốt nhất theo validation là epoch 1
nhưng vẫn chỉ đạt MAE `31.26 px`, PCK@25 `56.14%`; test của checkpoint FPN có
MAE `73.48 px`, PCK@25 `0%`. Checkpoint
`output/dataset_integration_20261010/coarse_fpn_640_hardtail_40ep/best_detector_2d.pth`
được reject và không thay coarse legacy.

ROI fine-tune tiếp tục từ checkpoint ROI hiện tại bằng actual coarse manifest,
hard-tail boost `2`, jitter `noise_std=0.04`, LR `1e-5`, freeze backbone 3
epoch. Validation tốt nhất là MAE `6.62 px`, PCK@25 `98.25%`; test là MAE
`8.12 px`, PCK@25 `100%`. Vì chưa vượt checkpoint ROI cũ (validation `4.70 px`,
PCK@25 `100%`), checkpoint
`output/dataset_integration_20261010/roi_finetune_actual_coarse_40ep/best_roi.pth`
cũng bị reject.

Reliability report cho thấy correlation confidence–error chỉ `-0.152`, có 27
record TTA stability ≥6 px và 33 record cần review, trong đó 9 record thuộc
test. Hàng đợi review nằm ở
`output/dataset_integration_20261010/manual_review_queue.csv`; Ground Truth
chỉ được sửa sau hai lần annotate độc lập. `img_243` vẫn chưa được sửa tự
động; temporal guard giảm lỗi của frame này nhưng chỉ được dùng cho video có
thứ tự.

### FLAME pretrain và CCTV fine-tune

`Training.zip` đã được giải nén vào `datasets/FLAME/Training/` và archive đã được dọn khỏi workflow; kết quả gồm
39.375 frame (`Fire=25.018`, `No_Fire=14.357`). Vì FLAME chỉ có nhãn
image-level, pretrain chỉ tối ưu confidence/backbone; nó không tạo `p_fire`.
Pilot 512 frame được lưu tại
`output/dataset_integration_20261010/flame_pretrain/best_flame_pretrained.pth`.

Checkpoint này đã được fine-tune trên CCTV với 24 mẫu hard-tail train:

| Nhánh | Validation MAE | PCK@25 |
|---|---:|---:|
| CCTV baseline | 4.90 px | 100% |
| FLAME pretrain → CCTV fine-tune | 50.33 px | 12.28% |

Checkpoint FLAME đã bị reject và không thay `fire-model-data/best.pth`. Quyết
định được lưu tại `output/dataset_integration_20261010/flame_training_decision.json`.
Transfer trực tiếp từ frame FLAME sang point localization CCTV làm mất thông
tin hình học; muốn dùng FLAME hiệu quả cần freeze backbone từng phần và thêm
domain adaptation có calibration, không coi Fire/No_Fire là nhãn tọa độ.

## Đối sánh 3D cùng pixel

Depth report dùng `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, checkpoint ROI CPU-regularized mixed, inference CPU, 92 frame. Không fit affine scale trong script benchmark này.

| Nhánh, cùng ROI pixel | Số frame | MAE (m) | RMSE (m) | Median (m) | P90 (m) |
|---|---:|---:|---:|---:|---:|
| Ray plane sàn z=0, 64 floor frame | 64 | 0.1625 | 0.2059 | 0.1263 | 0.2956 |
| Depth Anything, floor subset | 64 | 7.2243 | 7.3399 | 6.8314 | 9.6188 |
| Ray plane với pixel GT, oracle | 64 | ~0 | ~0 | ~0 | ~0 |

Ray layer giữ autograd và sai số hình học trên pixel GT ở mức số học `~2e-15 m`. Với ROI pixel mới, Jacobian lớn nhất `σ_max(dXYZ/dpixel)` có median `0.0674 m/px`, P90 `0.1209 m/px`, max `0.1680 m/px`; đây là căn cứ để truyền covariance pixel sang covariance 3D và để thêm loss 3D. Kết quả này chỉ áp dụng khi mặt phẳng hỗ trợ là đúng. Ray plane không tự giải quyết visibility của bàn, tủ hoặc cột.

Depth Anything V2 metric indoor là model metric được fine-tune trên Hypersim, với `max_depth=20` theo model card/config. Tuy vậy, benchmark cho thấy domain/scale của ảnh lửa CCTV chưa phù hợp: lỗi 3D lớn hơn hẳn ray mặt phẳng. Một affine diagnostic fit scene-disjoint trước đó giảm RMSE world xuống khoảng 1.532 m trên 44 frame, nhưng đó là phép đo chẩn đoán dùng nhãn metric của synthetic và không được coi là calibration triển khai.

## Hướng giải quyết được xếp ưu tiên

### A. Sửa protocol coarse trước khi đổi 3D

1. Chạy detector một lần trên train/val/test và lưu manifest gồm pixel, confidence, kích thước ảnh, checkpoint và threshold. Huấn luyện ROI bằng chính actual coarse; detector miss phải được tính là miss riêng.
2. Bổ sung hard-tail theo quantile coarse error, các mẫu confidence cao nhưng lệch lớn, và jitter theo phân bố thực tế. Chia theo scene/image để không rò frame gần nhau.
3. Giữ `p_fire_pixel` chỉ làm target, không dùng làm coarse thay thế trong validation/test.
4. Có thể bật `FireDetector(..., tta_mode="brightness3")` để lấy điểm trung bình
   photometric và `stability_px`; chỉ bật `max_stability_px` sau khi chọn trên
   validation và luôn report coverage/reject riêng.

### B. Định lượng reliability thay cho một confidence vô hướng

Lấy entropy/peak-to-mass của heatmap, covariance của soft-argmax, shift `||ROI-coarse||`, margin crop và thống kê temporal. Calibrate gate trên val:

- tin cậy cao + uncertainty thấp → ROI pixel;
- shift lớn, heatmap phân tán hoặc gần mép → blend/coarse fallback hoặc crop đa tỉ lệ;
- `stability_px` vượt gate → giữ fire alert nhưng không ray-cast/không gọi ROI;
  đây là reject an toàn, không phải positive đúng;
- detector miss → không gọi ROI và ghi miss rõ ràng.

Không đặt ngưỡng theo test set. `sigmoid(max heatmap)` chỉ là điểm tin cậy heuristic, chưa phải xác suất đúng.

### C. Dynamic/multiscale crop có kiểm soát

Crop nhỏ giữ chi tiết, crop lớn giữ context; chọn hoặc blend bằng uncertainty đã calibrate. Cần train đúng các scale dùng lúc inference. Crop audit hiện tại không ủng hộ việc chỉ tăng crop fraction như một fix chính, dù multiscale vẫn có ích cho outlier sát mép và domain shift.

### D. Nối end-to-end 2D → 3D

[depth_localization.py](./depth_localization.py) có ba thành phần:

- `DepthBasedLocator`: gọi Depth Anything V2, lấy depth tại `(u,v)` bằng bilinear interpolation, back-project qua `K^-1`, xử lý OOB/NaN/Inf.
- `DifferentiableRayCastLayer`: giải ray–plane bằng `torch.linalg.solve` và giữ graph từ `point_2d` tới XYZ. Test chứng minh gradient hữu hạn và khác 0.
- `VerticalityPriorLoss`: phạt chuyển vị ngang của base–top, có `wind_strength` để giảm trọng số khi gió mạnh.

Trong train, dùng soft-argmax/DSNT của ROI heatmap, đổi ROI-relative pixel về full-image bằng phép toán tensor, rồi áp dụng ray layer với mask: parallel, `t<=0`, plane degeneracy và gần horizon. Loss đề xuất là robust 3D Huber/Charbonnier + loss 2D + loss uncertainty; không backprop qua mesh CPU.

### E. Chính sách 3D/mesh

- Floor/support đã biết: ray-plane hoặc IPM là nhánh nhanh và chính.
- Nhiều bề mặt: semantic plane selector + visibility check, sau đó ray-plane.
- Mesh vẫn cần cho visibility/occlusion tổng quát; có thể giữ làm fallback offline hoặc nhánh kiểm chứng.
- Depth Anything V2 nên làm auxiliary depth/support prior hoặc fallback sau khi có scale calibration held-out theo domain; benchmark hiện tại chưa ủng hộ thay ray bằng depth trực tiếp.

Benchmark multi-plane hiện có cho thấy oracle surface/visible plane đạt MAE `0.1401 m` trên 79/92 frame, trong khi plane-bank nearest đạt MAE `2.8586 m` trên 88/92 và floor IPM đạt `5.1133 m` trên 88/92. Đây là bằng chứng rằng **selector/visibility** quan trọng hơn việc bỏ mesh một cách mù quáng.

## Lộ trình triển khai và tiêu chí chấp nhận

1. **Mốc 1 — actual-coarse ROI:** giữ split theo scene, report 36 ảnh real và synthetic test; không tăng MAE/P95 real so với `9.627/18.430 px`, đồng thời giảm số outlier >25 px.
2. **Mốc 2 — reliability:** chọn ngưỡng chỉ trên val; report coverage, miss, PCK, ECE/precision của gate và worst-case crop margin.
3. **Mốc 3 — differentiable floor branch:** trên synthetic floor, ray ROI cùng pixel đạt RMSE ≤ baseline `0.2059 m` và gradient hữu hạn; kiểm tra thêm camera pose estimated riêng.
4. **Mốc 4 — metric depth:** chỉ bật depth branch khi calibration held-out theo scene/domain đạt ngưỡng đã thống nhất, báo cáo cả scale error và 3D error; không suy diễn từ Hypersim sang CCTV.
5. **Mốc 5 — non-floor:** selector plane/visibility phải báo coverage và reject rõ ràng; không dùng nearest-plane hoặc floor fallback như kết quả chính khi tail error vượt giới hạn.

## Nguồn và giới hạn

Nguồn chính đã lưu thành text trong [sources](./sources): Depth Anything V2 README/metric-depth code và model card, DSNT, MonoDGP và Kendall–Gal. MonoDGP là geometry-error prior cho monocular 3D detection; không đồng nghĩa tự động hiệu chỉnh camera pose trong pipeline này. DSNT hỗ trợ tọa độ khả vi; uncertainty loss là cơ sở để học variance, không biến confidence detector thành xác suất đã calibrate.

Các URL gốc: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2),
[metric depth](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth),
[model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf),
[Depth Anything V2 paper](https://arxiv.org/abs/2406.09414),
[DSNT](https://arxiv.org/abs/1801.07372),
[MonoDGP](https://arxiv.org/abs/2410.19590), và
[uncertainty loss](https://arxiv.org/abs/1703.04977).

Các số synthetic là kiểm thử metric/procedural, không phải độ chính xác CCTV thực. Các hướng đề xuất là giả thuyết triển khai cần benchmark held-out; chúng không phải tuyên bố đã cải thiện production.


## Thử nghiệm sửa trực tiếp coarse tail thay vì ROI fine-tune

Đã kiểm tra hai cách sửa trực tiếp phân bố coarse. Detector FPN 640 px với hard-tail 40 epoch đạt validation coarse MAE `31.2599 px`, PCK@25 `56.14%`, test MAE `73.4825 px`, PCK@25 `0%`; checkpoint bị loại. Một residual correction head nhỏ, học từ 179 mẫu train và giữ detector đóng băng, có validation tốt nhất `5.6544 px` so với baseline `4.8982 px`, và test `9.0952 px` so với baseline `7.9312 px`; cũng bị loại vì overfit.

Kết luận: không nên cộng offset hậu xử lý cho coarse. Lỗi real không có một bias hình học ổn định để sửa bằng ánh xạ toàn cục; cần huấn luyện lại spatial head bằng ảnh và actual coarse error distribution, với warm-up head-only, backbone LR thấp, hard-tail chỉ ở train, và chọn checkpoint hoàn toàn trên validation CCTV. Production vẫn giữ `fire-model-data/best.pth` và `output/roi_domain_experiments_cpu_regularized/mixed/best_roi.pth`. Chi tiết và tiêu chí quyết định lưu ở [coarse_error_correction_decision.json](../output/dataset_integration_20261010/coarse_error_correction_decision.json).


## Spatial coarse head trước ROI — benchmark nghiệm thu thực tế

Đã chạy đủ 40 epoch với backbone `fire-model-data/best.pth`, freeze backbone 5 epoch, sau đó backbone learning rate `1e-5`, input 224×224 của trainer hiện hành và hard-tail 24 mẫu train trọng số 1.5. Checkpoint chọn theo validation MAE, không chọn theo test. Kết quả tốt nhất validation vẫn `20.4762 px` trong benchmark độc lập (PCK@25 `77.19%`); test coarse MAE `60.9502 px`, PCK@25 `16.67%`, 30/36 positive có lỗi >25 px. So với tiêu chí nghiệm thu thực tế `MAE ≤ 9.6270 px`, checkpoint bị **reject** và không chạy ROI.

File model đã xuất để kiểm tra adapter nhưng được đánh dấu `coarse_candidate_rejected.pth`; production không bị ghi đè. Báo cáo đầy đủ gồm bins lỗi và vùng ảnh ở [benchmark.json](../output/dataset_integration_20261011/coarse_before_roi_benchmark/benchmark.json), từng ảnh ở [per_image.csv](../output/dataset_integration_20261011/coarse_before_roi_benchmark/per_image.csv), và manifest riêng ở [actual_coarse_manifest.json](../output/dataset_integration_20261011/coarse_before_roi_benchmark/actual_coarse_manifest.json).


## High-resolution FPN coarse thử nghiệm tiếp theo

Đã sửa trainer để chọn checkpoint theo validation MAE, freeze backbone trong warm-up và giữ BatchNorm ở eval khi frozen. Nhánh FPN 320×320, heatmap 80×80, freeze 5 epoch, backbone LR `1e-5`, hard-tail 24 mẫu weight 1.5 được chạy 6/15 epoch trước khi dừng sớm vì localization sụp: validation MAE tăng từ `96.23` lên `162.25 px`, PCK@25 giảm còn `1.75%`. Checkpoint tốt nhất epoch 1 có test MAE `65.51 px`, PCK@25 `47.22%`; bị reject so với `9.6270 px`. FPN 640×640 cùng cấu hình bị dừng trước khi có metric do chi phí CPU quá cao; không có checkpoint được chọn.

Benchmark chi tiết FPN 320 nằm ở [benchmark.json](../output/dataset_integration_20261011/coarse_fpn_320_benchmark/benchmark.json), per-image ở [per_image.csv](../output/dataset_integration_20261011/coarse_fpn_320_benchmark/per_image.csv). Không checkpoint FPN nào được đưa vào ROI hoặc ghi đè production.


## Heatmap loss/multiscale smoke và run 640

Đã thêm negative heatmap bias, Gaussian target kiểm soát sigma, distribution cross-entropy theo cùng temperature soft-argmax, coordinate loss warm-up tăng lên 20, confidence weight warm-up giảm còn 0.2, và nhánh heatmap stride-4 + stride-8. Roundtrip letterbox đạt sai số tối đa `3.8e-05 px`. Smoke full-batch trên 16 ảnh train đạt MAE `0.259 px`, max `1.273 px`, PCK@25 `100%` sau 500 bước; đây chỉ là kiểm tra khả năng học head, không phải test accuracy.

Khi chạy toàn bộ 640×640, 4 epoch warm-up cho validation MAE `103.40 -> 100.08 -> 110.56 -> 111.62 px`, PCK@25 `0%`. Run bị dừng và checkpoint không được đưa vào ROI. Kết quả cho thấy head/loss không còn bị nghiệm nền trong smoke, nhưng chưa tổng quát hóa từ 16 mẫu sang CCTV validation; cần curriculum/augmentation và target heatmap ổn định hơn trước khi chạy 40 epoch.


## Curriculum 640×640 trực tiếp

Đã chạy curriculum trực tiếp 640×640: positive-only warm-up, không flip/brightness augmentation, hard-tail theo bin `0–5/5–10/10–20/>20`, tăng trọng số vùng `x0_y2`, rồi dự kiến mới mở backbone. Sau 3 epoch stage heatmap, validation MAE lần lượt `112.02`, `109.41`, `124.45 px`, PCK@25 `0%`. Run bị dừng trước unfreeze; checkpoint bị reject và không nối ROI. Nguyên nhân là head vẫn kéo soft-argmax về vùng sai trên validation dù train loss giảm; tăng resolution/epoch không sửa được lỗi protocol này.



## Direct coordinate head + heatmap consistency

Đã thêm nhánh hồi quy tọa độ trực tiếp từ FPN feature map và consistency loss với soft-argmax. Smoke full-batch 16 ảnh train đạt MAE `0.6904 px`, max `1.5676 px`, PCK@25 `100%` sau 240 bước; đây chỉ là khả năng fit train. Curriculum 640 với cùng nhánh thất bại trên held-out CCTV ngay epoch đầu: validation MAE `112.50 px`, PCK@25 `0%`; benchmark test MAE `91.52 px`, PCK@25 `0%`, coverage `5.56%`. Checkpoint bị reject và không chạy ROI.

Kết quả củng cố rằng lỗi chính là domain/generalization của backbone-feature → vị trí, không phải thiếu nhánh hồi quy hay thiếu độ phân giải. Smoke train thấp không được dùng làm tiêu chí promotion.
