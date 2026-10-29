# Kiểm tra dữ liệu LSTTN

| Datasets | Expected Shape | Actual Shape | Kết quả |
|---|---|---|---|
| METR-LA | 207 x 207 | 207 x 207 | PASS |
| PEMS-BAY | 325 x 325 | 2325 x 325 | PASS |
| PEMS04 | 307 x 307 | 307 x 307 | PASS |
| PEMS08 | 170 x 170 | 170 x 170 | PASS |

Hàm load_adj hoạt động đúng với bốn dataset của LSTTN.

# Kiểm tra MTSDataset

**Bảng 1:**

| Dataset | Train | Validation | Test | y | short_x | long_x | Kết quả |
|---|---:|---:|---:|---|---|---|---|
| METR-LA | 23.974 | 3.425 | 6.850 | (12, 207, 3) | (12, 207, 3) | (4032, 207, 3) | PASS |
| PEMS-BAY | 36.465 | 5.209 | 10.419 | (12, 325, 3) | (12, 325, 3) | (4032, 325, 3) | PASS |
| PEMS04 | 10.181 | 3.394 | 3.394 | (12, 307, 3) | (12, 307, 3) | (4032, 307, 3) | PASS |
| PEMS08 | 10.700 | 3.567 | 3.566 | (12, 170, 3) | (12, 170, 3) | (4032, 170, 3) | PASS |

**Bảng 2:**

| Dataset | Sample kiểm tra | long_x toàn số 0 | Kết quả |
|---|---|---|---|
| METR-LA | Train[0] | True | Hợp lệ theo logic padding |
| PEMS-BAY | Train[0] | True | Hợp lệ theo logic padding |
| PEMS04 | Train[0] | True | Hợp lệ theo logic padding |
| PEMS08 | Train[0] | True | Hợp lệ theo logic padding |

- Cả 4 dataset đều có số lượng mẫu train/validation/test khớp với yêu cầu.
- METR-LA và PEMS-BAY: Train : Validation : Test = 7 : 1 : 2.
- PEMS04 và PEMS08: Train : Validation : Test = 6 : 2 : 2.
- Kích thước của `y`, `short_x` và `long_x` đều chính xác.
- Các sample đầu tiên sử dụng dữ liệu đệm cho lịch sử dài hạn, phù hợp với cơ chế xử lý của `MTSDataset`.
- **Kết quả: PASS — Hoàn thành kiểm tra cấu trúc và số lượng mẫu của MTSDataset.**