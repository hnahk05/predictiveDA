# Tái hiện LSTTN cho dự báo lưu lượng giao thông

Mã nguồn và kết quả của báo cáo cuối kỳ *Nghiên cứu và tái hiện mô hình LSTTN trong bài toán dự báo giao thông*.

Sinh viên: Đỗ Mai Hằng - MSV: 24022647 

Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội.

Học phần: Phân tích dữ liệu dự báo - INT3234E.

Giảng viên: TS. Nguyễn Thị Hậu - Trợ giảng: Nguyễn Văn Công

Trên PEMS08, LSTTN dùng 14 ngày lịch sử (4.032 bước) cùng một giờ quan sát gần nhất (12 bước) để dự báo lưu lượng trong giờ tiếp theo tại 170 trạm. Dự án dùng Transformer tiền huấn luyện của tác giả và huấn luyện lại giai đoạn dự báo. [Bài báo LSTTN, Knowledge-Based Systems (2024)](https://doi.org/10.1016/j.knosys.2024.111637) · [Mã nguồn của tác giả](https://github.com/GeoX-Lab/LSTTN)

## Kết quả chính trên PEMS08

Bảng dưới tương ứng với Bảng 4.1 của báo cáo. Ba phương pháp tự chạy dùng cùng 3.566 cửa sổ kiểm thử. Persistence lặp lại quan sát cuối tại mỗi trạm. Historical Average (HA) lấy trung bình từ tập huấn luyện theo trạm, thứ trong tuần và khung 5 phút của thời điểm đích.

| Phương pháp | MAE 15′ | RMSE 15′ | MAPE 15′ | MAE 30′ | RMSE 30′ | MAPE 30′ | MAE 60′ | RMSE 60′ | MAPE 60′ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | 19,53 | 29,80 | 11,97% | 24,19 | 36,66 | 14,93% | 34,59 | 50,47 | 21,68% |
| HA | 23,27 | 40,63 | 14,47% | 23,26 | 40,61 | 14,48% | 23,23 | 40,56 | 14,50% |
| **LSTTN tái hiện** | **13,40** | **21,64** | **8,99%** | **14,19** | **23,49** | **9,93%** | **15,39** | **25,57** | **11,38%** |
| LSTTN do tác giả công bố | 13,17 | 20,78 | 8,63% | 13,71 | 21,89 | 9,09% | 14,54 | 23,47 | 9,77% |

MAE và RMSE có đơn vị xe/trạm/5 phút. MAE dùng toàn bộ nhãn, RMSE bỏ nhãn bằng 0, còn MAPE dùng nhãn từ 0,01 trở lên. Ở mốc 60 phút, MAE tái hiện giảm 55,5% so với Persistence và 33,8% so với HA, đồng thời cao hơn kết quả công bố 0,85 xe. [Metric LSTTN](results/experiment_metrics.csv) · [Metric đối chứng](results/extension/baselines_pems08.csv) · [Số liệu bài gốc](results/paper_reference.csv)

Run chính `PEMS08-repo96-author-pretrained`, attempt `0001`, dùng seed 0, batch 32 và 100 epoch trên RTX A6000. Checkpoint có validation loss thấp nhất ở epoch 68 được chọn để test. Chiều biểu diễn Transformer là 96, kích thước ẩn của hai nhánh xu hướng dài hạn và chu kỳ là 32/32, theo cấu hình tương thích với checkpoint công bố. Bài báo mô tả các giá trị 64 và 4/4. Chỉ có một lần huấn luyện chính và có 11 thời điểm nhãn trùng tại mỗi ranh giới train/validation và validation/test. Các run METR-LA (15 epoch), PEMS04 (2 epoch) và PEMS-BAY (1 epoch) phục vụ kiểm tra triển khai. [Hồ sơ thực nghiệm](results/experiment_manifest.json) · [Kiểm tra ranh giới tập dữ liệu](results/eda/PEMS08/pems08_overlap_summary.csv)

## Các phân tích trong báo cáo

- **Sai số theo giờ đích.** Ở mốc 60 phút, MAE thấp nhất lúc 2 giờ (7,05 xe) và cao nhất lúc 7 giờ (21,21 xe). Giờ được gán theo thời điểm cần dự báo. [Số liệu theo giờ](results/extension/error_analysis/hourly_metrics.csv)
- **Độ nhạy của LSTTN.** Trên cùng 256 cửa sổ, MAE 60 phút là 15,07 xe với đầu vào gốc, tăng lên 17,45 khi thay lịch sử dài bằng hồ sơ theo lịch từ tập huấn luyện và 69,10 khi thay lịch sử ngắn bằng trung bình theo trạm. Các phép thay thế giữ nguyên trọng số lúc suy luận. [Kết quả theo từng phép thử](results/extension/perturbation_pems08.csv)
- **Giải thích phần hiệu chỉnh dự báo (residual head).** Trên 16 cửa sổ, phép che sáu nhóm đặc trưng cho thấy head nhạy với dự báo cơ sở. Nhánh cộng dự báo cơ sở được giữ cố định. Integrated Gradients đạt kiểm tra tính đầy đủ và độ ổn định giữa 64 và 256 bước nội suy tại cả ba mốc. Kết luận XAI chỉ áp dụng cho phần hiệu chỉnh. [Giao thức và kết quả residual head](extension_C0_C8/README.md) · [Kiểm định IG](extension_C0_C8/run_v2/ig_results.json)
- **Phép tích chập đồ thị.** Trong nhánh chu kỳ, phép co chỉ số `bnh,nm->bnh` nhân đặc trưng mỗi nút với tổng trọng số hàng tương ứng. Với tổng hàng bằng 1, đầu ra của riêng phép toán này bằng đầu vào. Ảnh hưởng đến độ chính xác cần được đánh giá bằng một thí nghiệm huấn luyện đối chứng. [Kiểm tra phép toán](results/extension/graph_conv_check.json)
- **Chi phí suy luận.** Trên RTX 3050 Laptop, sau 5 lô làm nóng và 20 lô đo, batch 1 mất trung bình 118,10 ms và dùng tối đa 0,71 GiB bộ nhớ GPU cấp phát. Trong phép đo theo thành phần, Transformer chiếm 105,81 ms và Graph WaveNet chiếm 8,87 ms. [Số đo batch 1, 2 và 4](results/extension/inference_cost_v2.csv)

## Chạy lại và kiểm chứng

Chạy các lệnh từ thư mục gốc. Môi trường huấn luyện dùng Python 3.9.25, PyTorch 1.8.1+cu111, easy-torch 1.3.2 và NumPy 1.23.5. Trên Ubuntu hoặc WSL, chạy lệnh đầu khi tạo môi trường lần đầu.

```bash
bash environment/setup_linux.sh
source .local_checks/linux-env/bin/activate
```

**Tính lại các chỉ số LSTTN** từ prediction và ground truth đã xuất. Chuẩn bị riêng `artifacts/report-inputs/PEMS08/` với `metadata.json` và đủ 112 file `batch_*.npz` của attempt `0001`. Lệnh sau chạy trên CPU và chỉ cần bộ export cùng môi trường Python của dự án.

```bash
python -B -m utils.prediction_export artifacts/report-inputs/PEMS08
```

Các khóa `3`, `6`, `12` tương ứng 15, 30 và 60 phút. Nhân `MAPE_ratio` với 100 để đối chiếu với bảng.

**Chạy đối chứng và phân tích bổ sung.** Chuẩn bị dữ liệu đã tiền xử lý ở `datasets/PEMS08/`, đồ thị ở `datasets/sensor_graph/` và pretrained Transformer ở `pretrained_transformers/PEMS08.pt`. Các phép đo trực tiếp trên mô hình còn cần checkpoint forecast epoch 68 tại đường dẫn ghi trong hồ sơ suy luận. Dataset, checkpoint forecast và export được lưu riêng ngoài Git. [Nguồn dữ liệu của tác giả](https://github.com/GeoX-Lab/LSTTN#datasets) · [Đường dẫn và hash checkpoint](results/extension/inference_cost.json)

```bash
python scripts/extension/run_baselines_pems08.py --batch-size 16
python scripts/extension/analyze_pems08_errors.py --bootstrap-reps 2000 --seed 20261101
python scripts/extension/verify_graph_conv.py
python scripts/extension/measure_inference_cost.py --softmax-compat on
python scripts/extension/run_perturbation_pems08.py --batch-size 2
```

**Kiểm tra inference từ checkpoint.** Smoke test nạp checkpoint epoch 68 và so sánh bốn mẫu với export gốc. Hồ sơ CPU/GPU ghi nhận một số phần tử vượt dung sai. Việc tái tính metric từ export và việc chạy lại forward trả lời hai câu hỏi kiểm chứng riêng. [Hướng dẫn smoke test](verification_B_C1/README.md) · [Chẩn đoán CPU](verification_B_C1/smoke_results_cpu_diagnostic.json) · [Chẩn đoán GPU](verification_B_C1/smoke_results_diagnostic.json) · [Kết luận C1](extension_C0_C8/run_v2/c1.json)

## Tra cứu trong repo

| Nội dung | Vị trí |
|---|---|
| Kiến trúc LSTTN, Transformer, Graph WaveNet | [models/](models/) |
| Huấn luyện và cấu hình | [main.py](main.py), [runners/](runners/), [cấu hình PEMS08](configs/PEMS08/forecasting.py), [cấu hình theo lần chạy](configs/actual/) |
| Khảo sát dữ liệu và hình minh họa | [Notebook PEMS08](notebooks/PEMS08_EDA.ipynb), [hình báo cáo](results/report_figures/) |
| Giao thức, checkpoint và nguồn gốc lần chạy | [protocol.md](protocol.md), [training_provenance_pems08.json](output/training_provenance_pems08.json) |
