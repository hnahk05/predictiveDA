# Hồ sơ C0–C8: dừng đúng điều kiện nghiệm thu

Không phải báo cáo học thuật để nộp. Ngày kiểm tra 20/11/2026.

## Công việc thực sự đã chạy

- Đọc lại protocol/config/log/export và checkpoint evidence; tái sử dụng smoke GPU đã hợp lệ về nhận diện dữ liệu.
- Chạy thêm inference CPU bốn mẫu bằng cùng script, checkpoint, chỉ số và tolerance; không train. Kết quả mới: `../verification_B_C1/smoke_results_cpu_diagnostic.json`.
- Tạo script tổng hợp trạng thái và hash code `audit_gate.py`; kết quả `status.json`.
- Xác minh nguồn paper và chuẩn bị phiếu đọc trống, dàn ý/câu hỏi tự viết.

| Bước | Trạng thái | Bằng chứng và ý nghĩa |
|---|---|---|
| C0 | PARTIAL | `../verification_B_C1/report.md`, `../protocol.md`, `../artifacts/report-inputs/PEMS08/result.json`; xác định baseline thực tế nhưng provenance preprocessing/pretraining chưa đầy đủ; không khẳng định sạch leakage |
| C1 | FAIL | GPU 5/8160, CPU 9/8160 vượt atol=1e-5, rtol=1e-4; shape/finite/target/strict load đều đạt |
| C2 | BLOCKED | Không tạo cache khi C1 chưa đạt |
| C3 | BLOCKED | Không triển khai/huấn luyện head phụ thuộc cache chưa nghiệm thu |
| PILOT | PARTIAL | Có đo inference cũ; chưa có pilot cache/head/XAI |
| C4 | BLOCKED | 0 cấu hình head đã chạy |
| C5 | BLOCKED | 0 seed head đã chạy |
| C6 | BLOCKED | Không có metric residual head |
| C7 | BLOCKED | Chưa chạy occlusion của hệ thống mới; không gán E5 cũ thành kết quả của head mới |
| C8 | PARTIAL | Có gói bằng chứng, lệnh kiểm tra và tài liệu tự học; chưa có demo residual head |

## Chẩn đoán C1

GPU: max abs 0,0001220703125; mean abs 0,00001122524195. CPU: max abs 0,00018310546875; mean abs 0,00002044404619. Chuyển CPU không giải quyết FAIL.

Scaler có độ dốc 1077/2=538,5. Một khoảng FP32 cỡ 2^-23 ở vùng output chuẩn hóa gần -1 sau inverse trở thành 0,00006419420242. Con số này khớp độ lớn một số sai khác đã ghi nhận. Do inverse cộng 1 rồi nhân 538,5, sai khác tương đối trong đơn vị Flow gần 0 có thể lớn. Đây là cơ chế số học phù hợp bằng chứng, chưa cô lập nguồn sai khác trong kernel/batch/library. Không đổi tolerance hoặc clip prediction âm để vượt kiểm tra.

Log baseline ghi TF32 tắt; smoke cũng tắt. Checkpoint SHA, pretrained state, index, target/mask và eval đều được kiểm tra. Để đóng nguyên nhân cần chạy các mẫu thất bại với bản source snapshot và môi trường export gốc, kiểm soát thiết bị/batch/kernel; không chỉ thử nhiều setting rồi chọn cái PASS. Không yêu cầu huấn luyện lại backbone.

## Nguồn gốc và metric

Commit hiện tại xem `status.json`, khác commit huấn luyện `c4c7abb5cbccde7fe87754776ba01fb8ce5fa383` (working tree dirty theo protocol). Không gán HEAD hiện tại thành code huấn luyện. Hash từng file code hiện tại được ghi trong JSON.

Baseline giữ split 10700/3567/3566, MinMax Flow 0/1077, 11 timestamp nhãn chung tại train/val và val/test. Cache mới nếu được mở khóa phải giữ index này và công bố hạn chế; không gọi giao thức này là target-disjoint. Scope dữ liệu dùng pretrain cần kiểm tra riêng trước tuyên bố hoàn toàn không leakage.

| Horizon | MAE | RMSE | MAPE % |
|---|---:|---:|---:|
| 15 phút | 13,399619264 | 21,644650691 | 8,991705985 |
| 30 phút | 14,193402832 | 23,488660842 | 9,927706523 |
| 60 phút | 15,388007026 | 25,568519388 | 11,378056425 |

Đây là **kết quả đã có**, nguồn `../results/extension/error_analysis/overall_metrics_verified.csv`, không phải metric tính mới từ bốn mẫu. MAE gồm zero, RMSE bỏ target zero, MAPE bỏ target dưới 0,01; MAPE báo %. Không tính RMSE bằng trung bình RMSE từng batch.

## Tài nguyên: ưu tiên local để chẩn đoán, chưa duyệt chạy lớn

Số đo đã có từ `../results/extension/inference_cost_v2.json`, không phải pilot C2/C3 mới:

| Batch | Inference trung bình ± std (ms/batch) | Peak allocated VRAM (GB) | Peak process RSS lấy mẫu (GB) |
|---|---:|---:|---:|
| 1 | 118,10 ± 1,90 | 0,764 | 1,732 |
| 2 | 227,13 ± 1,21 | 1,523 | 1,820 |
| 4 | 452,58 ± 1,64 | 3,038 | 1,837 |

5 warmup, 20 batch đo; GB thập phân. Batch 4 reserved khoảng 4,47 GB. RSS đo mỗi 10 ms có thể bỏ lỡ đỉnh ngắn. Độ lệch chuẩn trong một phiên không phải khoảng tin cậy thời gian toàn dự án.

| Phương án | VRAM/tốc độ thực tế | Ổn định, lưu/khôi phục | Chi phí/quota | Lựa chọn |
|---|---|---|---|---|
| RTX 3050 local | 6 GB; batch 1–4 có số đo trên | Lưu file trực tiếp, cần tránh sleep/quá nhiệt; chưa thử cache resume | Không trả thuê; điện UNKNOWN | **Ưu tiên duy nhất hiện tại**, môi trường đã chạy, không cần chuyển dữ liệu |
| Kaggle | GPU/quota tài khoản UNKNOWN | Giới hạn phiên và lưu artifact của tài khoản UNKNOWN | UNKNOWN | Kiểm tra tài khoản, quota, persistent output trước khi xét |
| Colab | GPU/VRAM/quota UNKNOWN | Khả năng ngắt phiên/storage UNKNOWN | Gói tài khoản và giá UNKNOWN | Kiểm tra runtime, Drive và giới hạn thực tế |
| GPU thuê | Chưa chọn nhà cung cấp/GPU | Storage, snapshot và băng thông UNKNOWN | Báo giá hiện hành UNKNOWN; ngân sách người dùng 500.000đ | Chưa thuê hoặc thanh toán |

Ngoại suy **compute-only**, không phải thời gian export: 14.267 mẫu train+val × 0,118098 giây ≈ 28,1 phút ở batch 1. Chưa đo I/O, nén, hash, các sample train thiếu long history, nhiệt độ lâu dài; không dùng con số này để hứa thời hạn. Tốc độ export/head, kích thước cache và thời gian C2–C7: **UNKNOWN / [CHỜ KẾT QUẢ]**. Không đưa khoảng sai số tùy ý khi chưa có pilot end-to-end.

## Chạy lại

Từ gốc repository, không cần cài thêm dependency:

```powershell
.\venv\Scripts\python.exe -B extension_C0_C8\audit_gate.py --output extension_C0_C8\status_rerun.json
.\venv\Scripts\python.exe -B verification_B_C1\smoke_test.py --device cpu --output verification_B_C1\smoke_cpu_rerun.json
```

Tên output phải chưa tồn tại. Gate trả exit code 2 nếu điều kiện chưa đạt; đây là dừng có chủ đích. Gate chỉ tổng hợp evidence, không tự chạy train. JSON CPU hiện tại được chạy với `torch.set_num_threads(4)`; không so thời gian lệnh trên với benchmark GPU.

## Điều kiện cho người tiếp quản

1. Giải quyết C1 bằng đối chứng kiểm soát; giữ mọi kết quả thất bại. Nếu cần xây lại tiêu chuẩn tương đương số học, phải lập luận và ghi giao thức mới minh bạch, không sửa kết luận cũ.
2. Khi C1 đạt, pilot cache vài batch train/val; feature chỉ từ lịch sử trước forecast origin, prediction baseline và lịch biết trước. Target lưu riêng dùng loss; không đưa vào feature. Có manifest index/checkpoint/scaler/code, atomic batch write và xác minh resume.
3. Chỉ sau cache đạt mới làm head nhỏ, alpha=0 identity, backbone eval/frozen và gradient check; khóa tối đa 4 cấu hình trước train, chọn bằng val rồi 3 seed head. Không tự dùng test để quyết định feature/hyperparameter.
4. XAI hệ thống mới còn chưa có kết quả. Khi được mở khóa, phải ghi scope head hay toàn pipeline; occlusion lịch sử gốc phải forward lại nhánh liên quan, không chỉ che cache rồi nói giải thích dữ liệu thô. IG tùy chọn chỉ sau occlusion.
