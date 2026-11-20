# Chạy lại kiểm chứng PEMS08

Từ PowerShell tại `D:\Project_Github\predictiveDA`:

```powershell
.\venv\Scripts\python.exe -B verification_B_C1\smoke_test.py --output verification_B_C1\smoke_results_rerun.json
```

Chọn tên JSON mới mỗi lần. Script từ chối ghi đè file đã tồn tại. Không cần cài hoặc nâng cấp môi trường. Có thể thêm `--device cpu` để chạy CPU, nhưng kết quả số học và thời gian có thể khác GPU. Mặc định tự chọn CUDA nếu có.

Script đọc dataset thật qua `MTSDataset`, lấy bốn vị trí test 0, 31, 32, 3565 qua DataLoader batch 1, load pretrained và forecasting checkpoint strict, bật eval/no_grad, chuyển prediction về đơn vị gốc bằng đúng hàm của runner và so sánh với export ghép theo index. Không tạo runner huấn luyện, không backward, không optimizer, không sửa baseline. Batch 1 được chọn ngay từ đầu cho GPU 6 GB; nếu vẫn OOM thì ghi FAIL, không đổi mô hình.

Exit code 0: PASS. Exit code 1: FAIL hoặc BLOCKED; đọc JSON để phân biệt. FAIL vì tolerance không có nghĩa checkpoint không load được. `smoke_results.json` lưu lần chạy đầu; `smoke_results_diagnostic.json` bổ sung tọa độ các giá trị vượt tolerance trong lần chạy tiếp theo, giữ nguyên ngưỡng.

Đọc `report.md` để hiểu giới hạn giao thức. Hash chứng minh nhận diện/tính toàn vẹn file; shape, giá trị hữu hạn và đối chiếu prediction mới là bằng chứng inference. Kiểm chứng bốn mẫu không thay thế đánh giá toàn tập test.
