# C0–C8: hồ sơ thực nghiệm residual head PEMS08

Đây là hướng dẫn code/bằng chứng, không phải báo cáo học thuật. Kết quả lịch sử trước lần tiếp tục được giữ trong `README_before_v2.md`, `status_before_v2.json`, `writing_guide_before_v2.md`.

| Bước | Trạng thái |
|---|---|
| C0 | PARTIAL |
| C1 | PARTIAL |
| C2 | PASS |
| C3 | PASS |
| PILOT | PASS |
| C4 | PASS |
| C5 | PASS |
| C6 | PASS |
| C7 | PASS |
| C8 | PASS |

**C1_CONDITIONAL_PASS, vẫn NUMERIC_MISMATCH.** Không tuyên bố bitwise hay tương đương khoa học toàn bộ pipeline. Tiêu chí và giới hạn: `protocol_v2.json`, `run_v2/c1.json`. C0 PARTIAL vì chưa biết đầy đủ nguồn code preprocessing và tập dữ liệu fit pretrained. PASS ở các bước còn lại nghĩa đã thực thi/kiểm tra, không nghĩa mô hình luôn tốt hơn baseline.

## Mô hình và dữ liệu

- Head chọn bằng validation nội vùng: `h32`, width 32; seed demo cố định 0 trước test.
- Feature mỗi node gồm 12 prediction baseline chuẩn hóa, 12 Flow lịch sử ngắn chuẩn hóa, sin/cos giờ và ngày tương đối: 28 chiều. Không có target tương lai, Occupancy hay Speed trong feature head.
- MLP dùng chung giữa node: Linear → ReLU → Linear; lớp cuối khởi tạo 0. Prediction gốc + 538,5 × head, alpha=1; alpha=0 chính xác baseline. Backbone đóng băng toàn bộ, eval, không AMP/TF32.
- Giữ nguyên train/val/test 10700/3567/3566. Không đổi dataset, checkpoint epoch68 hoặc export gốc. Chọn head bằng nhãn validation nội vùng, bỏ 132 lượt nhãn chồng; vẫn báo metric đầy đủ riêng.
- Baseline đã chọn checkpoint bằng validation gốc có nhãn chồng: hạn chế này còn tồn tại. Head học dự báo in-sample của backbone, không phải out-of-fold stacking.
- Hai config h16/h32 seed0; sau khóa config chạy thêm seed1/2. Đây là 3 seed head trên cùng backbone, không phải 3 lần huấn luyện LSTTN.

## Kết quả test đã đo

| Phương pháp | Seed | Horizon phút | MAE | RMSE | MAPE % |
|---|---:|---:|---:|---:|---:|
| residual | 0 | 15 | 13.350615 | 21.614094 | 8.957319 |
| residual | 0 | 30 | 14.145271 | 23.412915 | 9.922427 |
| residual | 0 | 60 | 15.285206 | 25.447540 | 11.325127 |
| residual | 1 | 15 | 13.352362 | 21.613835 | 8.925778 |
| residual | 1 | 30 | 14.158550 | 23.435562 | 9.978943 |
| residual | 1 | 60 | 15.290928 | 25.488335 | 11.321921 |
| residual | 2 | 15 | 13.353457 | 21.613048 | 8.938219 |
| residual | 2 | 30 | 14.167452 | 23.444108 | 10.093532 |
| residual | 2 | 60 | 15.288864 | 25.477531 | 11.396291 |
| baseline_local | 0 | 15 | 13.399619 | 21.644651 | 8.991706 |
| baseline_local | 0 | 30 | 14.193403 | 23.488661 | 9.927707 |
| baseline_local | 0 | 60 | 15.388007 | 25.568519 | 11.378056 |
| baseline_original_export | 0 | 15 | 13.399619 | 21.644651 | 8.991706 |
| baseline_original_export | 0 | 30 | 14.193403 | 23.488661 | 9.927707 |
| baseline_original_export | 0 | 60 | 15.388007 | 25.568519 | 11.378056 |

Nguồn: `run_v2/test_metrics.csv`. Chênh lệch dương/âm theo sensor/giờ/Flow: `run_v2/test_groups.csv`. Quartile trên target test chỉ để mô tả hậu nghiệm, không dùng tuning. Giờ là tương đối từ index; không có kết luận ngày thường/cuối tuần. Không có kiểm định ý nghĩa thống kê.

## Cấu hình và tài nguyên

| Config | Seed | Số tham số | Epoch chạy | MAE val nội vùng | Giây train+val | Peak VRAM GB | Peak RSS GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| h16 | 0 | 668 | 7 | 14.674578 | 15.93 | 0.006 | 2.818 |
| h32 | 0 | 1324 | 7 | 14.668509 | 9.47 | 0.007 | 2.483 |
| h32 | 1 | 1324 | 8 | 14.670924 | 16.56 | 0.007 | 2.484 |
| h32 | 2 | 1324 | 12 | 14.672517 | 25.24 | 0.007 | 2.353 |

Cache nén thực tế: 284.66 MB. Nguồn số đo từng tập: `run_v2/cache/<split>/manifest.json`. Pilot batch2: [0.24710620000000105, 0.25352830000000104, 0.27896639999999984]; không gán thời gian khởi động thành tốc độ ổn định.
Theo dõi GPU: `run_v2/gpu_monitor.csv`; RSS lấy mẫu 10ms có thể bỏ lỡ đỉnh ngắn. Thời gian train từng run gồm validation; inference/cache/head có phạm vi đo khác nhau.

| Phương án | Đánh giá |
|---|---|
| RTX3050 local | Phương án đã dùng và ưu tiên: đủ bộ nhớ trong số đo thật, cache/checkpoint lưu local và resume được; không mất thời gian chuyển môi trường |
| Kaggle | GPU/quota tài khoản và giới hạn phiên UNKNOWN; chưa đo |
| Colab | GPU/quota/gói tài khoản/storage UNKNOWN; chưa đo |
| GPU thuê | Nhà cung cấp, báo giá, lưu trữ, chi phí thực UNKNOWN; không thuê hoặc thanh toán |

## XAI và phạm vi diễn giải

- `run_v2/xai.json`, `xai_attributions.npz`, `xai_summary.csv`: 16 vị trí test đều theo index, 6 nhóm feature, 2 cách che, toàn bộ 170 sensor × 12 horizon.
- Đây là attribution/sensitivity của **residual head**. Nhánh cộng baseline cố định; che prediction trong feature head không xóa baseline ở nhánh cộng. Không gọi là giải thích đầu vào thô của toàn LSTTN.
- Che bằng trung bình train theo sensor hoặc 0 trong không gian feature. Flow normalized=0 là giữa thang MinMax, không phải Flow thô bằng0. Cặp sin/cos=(0,0) có thể ngoài manifold; hai giá trị che là kiểm tra độ nhạy, không là can thiệp nhân quả.
- Không thay trọng số, không train lại architecture. IG thực hiện sau occlusion: xem `run_v2/ig_results.json`, `ig_summary.csv`, `ig_attributions.npz`, `ig_head_heatmap.png`. Chỉ giải thích head, kiểm tra completeness và độ ổn định 64/256 bước. Trạng thái IG: PASS. Giá trị NOT RUN trong xai.json là tại thời điểm hoàn thành occlusion, trước bước IG bổ sung.

## Chạy lại và resume

Từ gốc repo, dùng venv hiện tại; các lệnh tái sử dụng artifact hoàn tất sau xác minh. Không chạy lại `verify` để cố đổi kết luận. Nếu thay config/code feature phải tạo phiên bản run riêng, không sửa protocol của run này.

```powershell
.\venv\Scripts\python.exe -B extension_C0_C8\test_pipeline.py
.\venv\Scripts\python.exe -B extension_C0_C8\run_remaining.py
.\venv\Scripts\python.exe -B extension_C0_C8\pipeline.py demo
```

Cache theo shard64, inference batch2; atomic rename, checksum và fingerprint. Head checkpoint resume mỗi epoch gồm optimizer, best weights và RNG shuffle; chạy lại bỏ qua run hoàn tất sau kiểm tra hash. Demo dùng hai mẫu cache, không chạy lại full test.

## Bằng chứng kiểm thử

- `run_v2/unit_tests_extended.log`: alpha0/zero init, gradient head, mask metric, feature không nhận target, Adam resume, loại nhãn chồng validation.
- `run_v2/pilot.json`: inference/cache roundtrip, direct repeat, only-head gradients và tài nguyên.
- `run_v2/final_integrity.json`: hash cache, index toàn bộ, direct-inference đối chiếu ở ba tập, hash baseline/export.
- `run_v2/provenance.json`, `code_manifest.json`, `status_final.json`: config/hash/môi trường/trạng thái. Code mới chưa commit: không gán Git HEAD thành commit đã chứa thí nghiệm.
- `run_v2/logs/`: log từng bước. `audit_gate.py` là gate lịch sử nghiêm ngặt trước v2, không dùng nó để phủ định quyết định conditional đã đăng ký.

## Hình và hướng dẫn viết

Các hình: `run_v2/test_MAE.png`, `test_RMSE.png`, `test_MAPE_percent.png`, `node_gain_60.png`, `head_learning_curves.png`, `xai_heatmap.png`, `xai_replacement_sensitivity.png` (đều dưới run_v2). Đọc `writing_guide.md` và tự điền `paper_reading.md`.
