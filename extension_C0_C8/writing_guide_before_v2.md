# Khung tự viết báo cáo và chuẩn bị bảo vệ

Chỉ là dàn ý/câu hỏi; không có đoạn học thuật để nộp nguyên văn. Residual head chưa được triển khai/huấn luyện do C1 FAIL. Mọi mục kết quả head phải giữ [CHỜ KẾT QUẢ].

## Phương pháp đề xuất

- Phát biểu giả thuyết: loại sai số nào của baseline có thể sửa bằng head nhỏ?
- Viết công thức `y_new = y_baseline + alpha * head(features)`; tự định nghĩa đơn vị, shape từng biến.
- Liệt kê feature và thời điểm chúng thực sự sẵn có; tách feature khỏi target của loss.
- Giải thích backbone frozen, eval, dropout; phân biệt không cập nhật trọng số với chế độ eval.
- Mô tả alpha=0 identity, tham số head và loss: [CHỜ TRIỂN KHAI/KIỂM THỬ].
- Tự trả lời: tại sao không sửa đồ thị trong cùng thí nghiệm? Vì sao không được chọn feature dựa trên test?

## Thiết lập thực nghiệm

- Dataset, kênh thực tế, lịch sử 12/4032, horizon 12; vẽ timeline window.
- Ghi split thực 10700/3567/3566, nhãn chồng, nguồn MinMax; không đổi thành 7:2:1 trên giấy.
- Trình bày baseline checkpoint và provenance code; phân biệt commit hiện tại với code lúc huấn luyện.
- Nêu metric/mask/unit, config head, tiêu chí chọn val, số seed thực chạy.
- Tách số đo lịch sử inference khỏi pilot cache/head chưa có.
- Tự trả lời: thế nào là input overlap hợp lệ? Tại sao label overlap là hạn chế? Pretrained có thấy val/test chưa?

## Kết quả và thảo luận

- Baseline/E1/E2 có bằng chứng cũ: dẫn đúng file, không gọi là lần chạy mới.
- Bảng baseline so head 15/30/60: [CHỜ KẾT QUẢ].
- Cấu hình/seed/loss curve: [CHỜ KẾT QUẢ].
- Mean/std head không đại diện nhiều seed backbone; chưa có kiểm định không tuyên bố cải thiện có ý nghĩa thống kê.
- Tự trả lời: nếu head chỉ cải thiện val mà test giảm, bạn có được chỉnh tiếp theo test không? So sánh có cùng mẫu/mask không?

## XAI

- Chọn scope: head trên cache hay toàn hệ thống trên lịch sử gốc; nêu output (node,horizon).
- Feature Ablation/Occlusion là che input khi giữ trọng số, không phải train lại ablation kiến trúc.
- Lịch sử ngắn có thể chia nhóm 4+4+4 bước; giá trị che và kiểm tra độ nhạy phải định nghĩa trước. Đây là phương án chưa chạy.
- Không có Occupancy/Speed trong tensor baseline đã xử lý; không vẽ attribution cho các kênh đó.
- Tập mẫu đại diện, raw attribution, heatmap/bar chart: [CHỜ KẾT QUẢ].
- IG: chưa thực hiện; không thay thế Feature Ablation bắt buộc.
- Tự trả lời: che Flow trong short nhưng giữ nguyên phần Flow trùng trong long giải thích nhánh hay thông tin toàn hệ thống? Vì sao attribution không chứng minh nhân quả?

## Hạn chế và hướng phát triển

- C1 chưa đạt ngưỡng, không che giấu FAIL bằng một số MAE trung bình nhỏ.
- Split nhãn chồng; provenance preprocessing/pretraining chưa đầy đủ; ngày lịch chưa xác minh.
- Một seed backbone; head chưa có seed nào; chưa có kết quả mô hình cải tiến.
- Hạn tài nguyên và thời gian; số đo local không chứng minh tốc độ cloud.
- Tự trả lời: phát biểu nào là kết quả thực nghiệm, phát biểu nào chỉ là giả thuyết hoặc ước lượng?

## Danh mục bảng/hình và dữ liệu gốc

| Bảng/hình dự định | Bằng chứng tương đối từ repo root | Có thể dùng ngay? |
|---|---|---|
| Timeline split và shape | `verification_B_C1/smoke_results.json`, `utils/datasets.py` | Có; tự vẽ và ghi 11 nhãn chồng |
| So sánh checkpoint/export GPU và CPU | `verification_B_C1/smoke_results_diagnostic.json`, `smoke_results_cpu_diagnostic.json` cùng thư mục | Có; phải ghi FAIL |
| Baseline MAE/RMSE/MAPE | `results/extension/error_analysis/overall_metrics_verified.csv` | Có, kết quả cũ |
| Persistence/HA | `results/extension/baselines_pems08.csv`, `baselines_pems08_protocol.json` cùng thư mục | Có, giữ caveat train-label overlap |
| Benchmark inference | `results/extension/inference_cost_v2.json` | Có, chỉ phạm vi benchmark cũ |
| Pilot cache/head và kích thước disk | [CHỜ KẾT QUẢ] | Chưa |
| 4 config, lựa chọn bằng validation | [CHỜ KẾT QUẢ] | Chưa |
| Head seed 0/1/2, mean/std | [CHỜ KẾT QUẢ] | Chưa |
| Baseline vs residual head trên test | [CHỜ KẾT QUẢ] | Chưa |
| Heatmap/bar chart occlusion hệ thống mới | [CHỜ KẾT QUẢ] | Chưa |
| IG/completeness | [CHỜ KẾT QUẢ, TÙY CHỌN] | Chưa |

## Việc sinh viên tự làm

1. Điền bốn phiếu trong `paper_reading.md`, ghi trang/phương trình, tự kiểm tra reference LSTTN.
2. Đọc Dataset → model → scaler → metric; giải thích một index cụ thể bằng lời và shape.
3. Tự viết theo mẫu trường, trích dẫn đúng nguồn; không viết các bước BLOCKED thành đóng góp đã hoàn thành.
4. Tập demo lệnh gate và giải thích vì sao nó dừng; chuẩn bị trả lời tolerance, leakage, freeze/eval, XAI scope.
5. Sau khi có kết quả mới mới thay [CHỜ KẾT QUẢ]; giữ cả cấu hình không thắng và các trường hợp suy giảm.
