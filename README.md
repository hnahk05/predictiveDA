# predictiveDA — Thực nghiệm LSTTN

Dự án triển khai lại **giai đoạn forecast của LSTTN**, sử dụng checkpoint tiền huấn luyện do tác giả công bố. Mục tiêu là kiểm tra kết quả dự báo giao thông trên phần cứng có tài nguyên hạn chế.

Nguồn gốc: [GeoX-Lab/LSTTN](https://github.com/GeoX-Lab/LSTTN). [Paper LSTTN](https://arxiv.org/abs/2403.16495).

## Phạm vi đã thực hiện

| Dataset | Epoch forecast | Global batch | GPU | Vai trò |
|---|---:|---:|---|---|
| PEMS08 | 100 | 32 | RTX A6000 | Thực nghiệm chính |
| METR-LA | 15 | 4 | RTX 3050 Laptop 6 GB | Thăm dò |
| PEMS04 | 2 | 2 | RTX 3050 Laptop 6 GB | Thăm dò |
| PEMS-BAY | 1 | 2 | RTX 3050 Laptop 6 GB | Thăm dò |

Đã thực hiện EDA PEMS08, baseline Persistence/HA, phân tích sai số, kiểm chứng phép toán graph, đo chi phí suy luận và kiểm tra độ nhạy đầu vào/Integrated Gradients và chưa thực hiện lại toàn bộ thực nghiệm.

## Kết quả chính PEMS08

Run `PEMS08-repo96-author-pretrained`, attempt `0001`, chọn best theo validation tại epoch **68**, sau đó reload để test. Cấu hình dùng width **96**, hidden **32/32**, theo repo và checkpoint công bố; khác mức 64/4 nêu trong paper.

| Khoảng dự báo | MAE | RMSE | MAPE (%) |
|---|---:|---:|---:|
| 15 phút | 13,40 | 21,64 | 8,99 |
| 30 phút | 14,19 | 23,49 | 9,93 |
| 60 phút | 15,39 | 25,57 | 11,38 |

MAE/RMSE có đơn vị xe/5 phút/node. Số liệu đầy đủ ở [experiment_metrics.csv](results/experiment_metrics.csv), thông số và run ID ở [experiment_manifest.json](results/experiment_manifest.json). [paper_reference.csv](results/paper_reference.csv) chứa số liệu tham khảo từ tài liệu, không phải baseline tự chạy.

## Cấu trúc chính

| Thư mục | Nội dung |
|---|---|
| `models/`, `runners/`, `utils/` | Mô hình, huấn luyện và xử lý dữ liệu/metric |
| `configs/<dataset>/` | Config gốc của tác giả |
| `config_paper/` | Config chuẩn bị theo paper; chưa dùng cho kết quả cuối |
| `configs/actual/` | Config tái lập thông số thực chạy |
| `scripts/extension/` | Script baseline, kiểm chứng và phân tích bổ sung |
| `notebooks/PEMS08_EDA.ipynb` | EDA, hình dự báo và minh chứng thực thi |
| `results/extension/` | Kết quả các phân tích bổ sung |

## Môi trường và dữ liệu

Môi trường huấn luyện đã kiểm tra: **Python 3.9.25, Torch 1.8.1+cu111, easy-torch 1.3.2, NumPy 1.23.5**. Dùng Ubuntu/Linux hoặc Ubuntu WSL có driver NVIDIA phù hợp nếu cần GPU.

Trên máy mới, mở terminal Ubuntu tại thư mục repo và chạy lệnh sau. Bỏ qua nếu môi trường đã có:

```bash
bash environment/setup_linux.sh
```

Đặt dữ liệu trong `datasets/<dataset>/`, checkpoint pretrain trong `pretrained_transformers/`. Link dữ liệu công bố nằm trong [README repo tác giả](https://github.com/GeoX-Lab/LSTTN#datasets).

Các phân tích PEMS08 cần export đã xác minh trong `artifacts/report-inputs/PEMS08/`. Đo suy luận, kiểm graph thích nghi và IG còn cần checkpoint forecast best epoch 68. Dữ liệu, export và checkpoint thực nghiệm lớn **không được đưa vào Git**; cần chuẩn bị riêng để chạy lại. Notebook có output đã lưu để xem kết quả.

## Chạy phân tích bổ sung

Chạy trong **terminal Ubuntu/WSL**, tại thư mục repo, khi đã có đủ dữ liệu và artifact tương ứng:

```bash
source .local_checks/linux-env/bin/activate

python scripts/extension/run_baselines_pems08.py --batch-size 16
python scripts/extension/analyze_pems08_errors.py --bootstrap-reps 2000 --seed 20261101
python scripts/extension/verify_graph_conv.py
python scripts/extension/measure_inference_cost.py --softmax-compat on
python scripts/extension/run_perturbation_pems08.py --batch-size 2
```

## Trích dẫn

Luo, Q., He, S., Han, X., Wang, Y., & Li, H. (2024). *LSTTN: A Long-Short Term Transformer-based spatiotemporal neural network for traffic flow forecasting*. Knowledge-Based Systems, 293, 111637. [DOI](https://doi.org/10.1016/j.knosys.2024.111637).
