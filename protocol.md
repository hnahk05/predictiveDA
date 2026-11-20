- Dataset: PEMS08

- Baseline: LSTTN tái hiện cuối kì, bổ sung XAI, mở rộng

- Best checkpoint: 68

- MAE_paper (15 mins): 13.17
- RMSE_paper (15 mins): 20.78
- MAPE_paper (15 mins): 8.63
- MAE_paper (30 mins): 13.71
- RMSE_paper (30 mins): 21.89
- MAPE_paper (30 mins): 9.09
- MAE_paper (60 mins): 14.54
- RMSE_paper (60 mins): 23.47
- MAPE_paper (60 mins): 9.77

- MAE_reproduce (15 mins): 13.40
- RMSE_reproduce (15 mins): 21.64
- MAPE_reproduce (15 mins): 8.99
- MAE_reproduce (30 mins): 14.19
- RMSE_reproduce (30 mins): 23.49
- MAPE_reproduce (30 mins): 9.93
- MAE_reproduce (60 mins): 15.39
- RMSE_reproduce (60 mins): 25.57
- MAPE_reproduce (60 mins): 11.38

# Model
- Model name: LSTTN (Long-Short Term Transformer-based Spatio-temporal Neural Network)
- Task: Traffic Flow Forecasting 
- Dataset: PEMS08
- Input: 12 bước ngắn hạn (60 phút) và 4032 bước dài hạn (14 ngày)
- Output: 12 bước dự báo tương lai (60 phút)
- Time interval: 5 phút/bước
- Number of nodes: 170 cảm biến
- Input features: Flow, Occupancy, Speed
- Target: Flow
- Main components:
  - Pretrained Transformer: Trích xuất thông tin dài hạn
  - Graph WaveNet: Học quan hệ không gian và thời gian ngắn hạn
  - Trend/Periodicity: Khai thác xu hướng và tính chu kỳ
  - Fusion: Kết hợp các đặc trưng để dự báo
- Pretrained Transformer: Sử dụng checkpoint do tác giả cung cấp, đóng băng tham số trong giai đoạn forecasting
- Best forecasting checkpoint: Epoch 68
- Baseline: Kết quả tái hiện LSTTN trên PEMS08, sử dụng để so sánh với các thực nghiệm mở rộng và XAI

# XAI
1. Feature Ablation / Occlusion
2. Integrated Gradients (nếu có thể mới thực hiện)

# Experiment Configuration

### Dataset Configuration
- Dataset: PEMS08
- Number of sensors: 170
- Sampling interval: 5 minutes
- Short-term input length: 12
- Long-term input length: 4032
- Prediction horizon: 12
- Target feature: Flow

### Training Configuration
- Batch size: 32
- Maximum epochs: 100
- Optimizer: Adam
- Learning rate: Cần xác nhận từ config
- LR scheduler: MultiStepLR
- Scheduler milestones: [1, 18, 36, 54, 72]
- Scheduler gamma: 0.5
- Random seed: Cần xác nhận
- Loss function: Cần xác nhận từ code

### Model Configuration
- Transformer representation dimension: 96
- Long-term trend hidden dimension: 32
- Periodicity hidden dimension: 32
- Graph WaveNet parameters: Cần xác nhận từ config
- Pretrained Transformer: Frozen

### Hardware
- GPU dùng cho thực nghiệm baseline: NVIDIA RTX A6000 48 GB (có thể)
- GPU máy hiện tại: NVIDIA RTX 3050 Laptop 6 GB

## Cấu hình và nguồn gốc thực nghiệm PEMS08

### 1. Định danh lần chạy

- **Run ID:** `PEMS08-repo96-author-pretrained`
- **Attempt:** `0001`
- **Dataset:** PEMS08
- **Phạm vi:** Huấn luyện giai đoạn dự báo từ checkpoint tiền huấn luyện do tác giả công bố; không tiền huấn luyện lại từ đầu.
- **Config lúc chạy:** `configs/PEMS08/forecasting.py`

### 2. Cấu hình huấn luyện thực tế

| Thông số | Giá trị |
|---|---|
| Số epoch | 100 |
| Epoch của checkpoint được chọn | 68 |
| Tiêu chí chọn checkpoint | Validation loss thấp nhất |
| Batch size toàn bộ | 32 |
| GPU thực tế | 1 × NVIDIA RTX A6000 |
| Transformer width | 96 |
| Hidden nhánh dài hạn / chu kỳ | 32 / 32 |
| Lịch sử ngắn / dài | 12 / 4.032 bước |
| Patch size | 12 |
| Số lớp Transformer encoder | 4 |
| Số attention head | 4 |
| Số bước dự báo | 12 |
| Các horizon đánh giá | 3, 6, 12 bước — tương ứng 15, 30, 60 phút |
| Optimizer | Adam |
| Learning rate ban đầu | 0.005 |
| Weight decay | 0.00001 |
| Adam epsilon | 0.00000001 |
| Scheduler | MultiStepLR |
| Milestones | `[1, 18, 36, 54, 72]` |
| Gamma | 0.5 |
| Seed | 0 |

Config gốc khai báo 2 GPU, nhưng hồ sơ thực thi xác nhận số GPU hiệu dụng là 1. Cấu hình width 96 và hidden 32/32 theo repo/checkpoint công bố, khác width 64 và hidden 4/4 được mô tả trong paper.

### 3. Git commit và bản mã nguồn dùng khi chạy

- **Commit ghi trong hồ sơ bản mã nguồn mang đi chạy:**
  `c4c7abb5cbccde7fe87754776ba01fb8ce5fa383`
- **Trạng thái tại thời điểm đóng gói:** `working_tree_dirty: true`.
- **Commit repo tác giả được dự án ghi nhận:**
  `68185aa8ffd266d4d1a3acffeb6875a144335554`

Do có thay đổi chưa commit tại thời điểm đóng gói, commit `c4c7abb…` không xác định đầy đủ mã nguồn đã chạy. Bản sao mã nguồn và SHA-256 từng tệp được sử dụng để bổ sung khả năng truy vết.

Thư mục bản sao lưu, tính từ thư mục gốc dự án:

```text
.local_checks/restored-pems08/pems08-backup-20261031T195318Z-5220693f/
```

Các tệp bằng chứng trong thư mục này:

```text
environment/source-version.json
backup-manifest.json
configs/PEMS08/forecasting.py
source/
runs/PEMS08-repo96-author-pretrained/0001/completed.json
```

### 4. SHA-256 của config và mã nguồn

Các hash dưới đây được tính trên bản sao lưu đi kèm lần chạy, không phải mặc định trên mã nguồn hiện tại.

| Tệp trong bản sao lưu | SHA-256 |
|---|---|
| `configs/PEMS08/forecasting.py` | `22dbf9f8a9a7839861c9353a298a8499ba5f8820c9e9c2d7d84ea93b0ff3ee08` |
| `source/models/lsttn.py` | `3f3d67b970c885c4399e2093bf59b4097fb03567dde17512cdea1148c6c4b4c1` |
| `source/runners/lsttn.py` | `495760c871dd376f7162d2d3993a6f138de286f4748768586ba4424fa1f47a35` |
| `source/utils/datasets.py` | `3e917d543ca01485b649fcbbc51b76881f1ed7520bd3684162314e4d90550271` |

Đây là SHA-256 của từng tệp, không phải một hash duy nhất đại diện cho toàn bộ repository.

### 5. Checkpoint đầu vào: Transformer tiền huấn luyện

- **Nguồn:** Checkpoint do tác giả công bố.
- **Đường dẫn trong dự án:** `pretrained_transformers/PEMS08.pt`
- **Width:** 96.
- **Nạp trọng số:** Hồ sơ lần chạy ghi nhận strict load thành công.
- **SHA-256:**

```text
4b7efb021ce4b432347f46f4197ab8dbab9f01c7e0a0265f81a764cb1633a73e
```

### 6. Checkpoint đầu ra: LSTTN forecast

- **Tên tệp:** `LSTTN_best_val_loss.pt`
- **Epoch được chọn:** 68.
- **Đường dẫn bên trong thư mục bản sao lưu:**

```text
runs/PEMS08-repo96-author-pretrained/0001/checkpoints/a63106bc1277e979c479021bcb4929ac/LSTTN_best_val_loss.pt
```

- **SHA-256:**

```text
d3b8b3fa9f988c5018266e84f24397c24a60311024c08ad7f7bd531020bab55d
```

### 7. Phạm vi xác minh

SHA-256 của config và hai checkpoint nêu trên đã được tính lại và khớp giá trị trong hồ sơ lưu trữ. Hash của các tệp Python trong bản sao mã nguồn cũng được kiểm tra; danh sách chi tiết nằm tại:

```text
output/training_provenance_pems08.json
```

Việc kiểm tra hash xác nhận tính toàn vẹn của các tệp được đối chiếu, không thay thế cho việc chạy lại huấn luyện hoặc chứng minh kết quả tương đương hoàn toàn với paper.

Để tránh nhầm lẫn, Git HEAD hiện tại tại thời điểm kiểm kê là `15bce516f931f1fa78bb4c1dd2e126d392760320`; đây không phải commit được hồ sơ bản sao lưu ghi nhận cho lần huấn luyện PEMS08.