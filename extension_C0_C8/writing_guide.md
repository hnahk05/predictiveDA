# Dàn ý tự viết và bảo vệ — dùng kết quả đã chạy

Không phải đoạn văn để nộp. Tự viết lập luận và trích dẫn; không suy từ PASS rằng mô hình cải thiện mọi metric.

## 1. Phương pháp đề xuất

- Tự giải thích công thức baseline + alpha × correction; phân biệt đơn vị normalized và Flow.
- Vẽ 28 feature → MLP nhỏ → 12 correction mỗi node; nhánh cộng baseline riêng.
- Nêu tại sao các feature biết được tại forecast origin; target chỉ dùng loss.
- Giải thích zero initialization, alpha0, backbone frozen/eval; dẫn code `pipeline.py:Head`, `Context.features`, `train_one`.
- Câu hỏi: head shared giữa sensor có học quan hệ trực tiếp giữa sensor không? Vì sao không thêm graph fix cùng thí nghiệm?

## 2. Thiết lập thực nghiệm

- Dùng `protocol_v2.json` để tự mô tả hai config, Adam/lr/epoch/patience, metric chọn val, seed.
- Dùng `run_v2/provenance.json` và `code_manifest.json` cho hash/config/môi trường.
- Nêu split thật 60/20/20 và nhãn chồng. Giải thích mask validation nội vùng loại 132 lượt nhãn, không đổi split hay metric test.
- Nêu checkpoint backbone có lịch sử chọn val cũ; pretraining data scope vẫn UNKNOWN.
- Câu hỏi: vì sao head 3 seed không phải LSTTN 3 seed? Vì sao train prediction không out-of-fold là hạn chế?

## 3. Bảng và hình kết quả cần dùng

| Bảng/hình | Nguồn file dưới extension_C0_C8/run_v2 | Điều cần tự phân tích |
|---|---|---|
| C1 numerical vs metric | c1.json, c1_numeric_violations.csv, prior_cpu_gpu_violations.csv | Vì sao conditional pass không phải bitwise pass? |
| Hai config và các seed thực chạy | config_seed_results.csv; heads/*/result.json | Config chọn bằng metric nào, có train đủ12epoch không? |
| Mean/std validation seed | validation_summary.json | Std giữa seed head có giới hạn gì? |
| MAE/RMSE/MAPE 15/30/60 | test_metrics.csv; test_MAE.png, test_RMSE.png, test_MAPE_percent.png | Metric/horizon nào tăng, giảm? Không bỏ hàng bất lợi |
| Gain/loss node/giờ/Flow | test_groups.csv; node_gain_60.png | Có node suy giảm không? Quartile hậu nghiệm dùng để làm gì? |
| Đường học | head_learning_curves.png; heads/*/result.json | Điểm chọn checkpoint, khả năng overfit |
| Tài nguyên | pilot.json; cache/*/manifest.json; heads/*/result.json; gpu_monitor.csv | Tách khởi động, cache, train, RAM và VRAM |
| XAI | xai_summary.csv; xai_heatmap.png; xai_replacement_sensitivity.png | Kết luận có đổi khi đổi giá trị che không? |

## 4. XAI

- Nêu scope head, sample selection cố định 16 vị trí, sensor/horizon, các nhóm feature, hai cách che.
- Raw attribution có trong `xai_attributions.npz`; index và cấu hình trong `xai.json`.
- Câu hỏi: vì sao che cached prediction vẫn giữ nhánh cộng baseline? Có được nói đây là đóng góp lịch sử thô vào toàn LSTTN không?
- Câu hỏi: Flow normalized0 có nghĩa gì? Cặp sin/cos0 có hợp lệ không? Vì sao không suy nhân quả?
- IG bổ sung: `ig_results.json`, `ig_summary.csv`, `ig_attributions.npz`, `ig_head_heatmap.png`. Tự kiểm tra trạng thái PASS/PARTIAL, completeness và độ ổn định 64/256 bước; không gộp với occlusion hoặc nói đây là IG của backbone.

## 5. Hạn chế và hướng phát triển

- Tự đánh giá numeric mismatch, provenance chưa đủ, overlap từ baseline, calendar chưa xác minh.
- In-sample residual learning, một backbone seed, ít cấu hình và không có kiểm định ý nghĩa thống kê.
- XAI chỉ head và nhạy với reference; không gọi là ablation huấn luyện lại.
- Mọi thí nghiệm mới chưa chạy ghi [CHỜ KẾT QUẢ]; không dùng test để chỉnh lại cấu hình sau khi xem bảng.

## 6. Việc bạn tự làm

1. Đọc bốn paper và tự điền phiếu trong `paper_reading.md`, kiểm tra trang/phương trình và nguồn trích dẫn.
2. Tự viết theo mẫu trường, Việt hóa bảng/hình/công thức và kiểm tra tất cả số liệu với CSV.
3. Giải thích được một cửa sổ dữ liệu, một batch head, metric masking và việc không truyền gradient vào backbone.
4. Tập demo `pipeline.py demo`; metric demo2mẫu không phải metric toàn test.
5. Tự commit code và bằng chứng cần thiết lên GitHub theo yêu cầu môn; cân nhắc kích thước cache/checkpoint, không commit dữ liệu lớn vô ý. Agent chưa push hoặc tạo commit.
