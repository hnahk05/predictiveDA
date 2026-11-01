# Độ nhạy đầu vào khi suy luận — KHÔNG phải ablation huấn luyện lại

Run PEMS08-repo96-author-pretrained / 0001, best epoch 68; strict-load, eval. Không cập nhật tham số.
256 cửa sổ lấy ngẫu nhiên không hoàn lại bằng seed 0; danh sách và donor xáo trộn nằm trong selected_windows.csv. IG dùng 32 cửa sổ con.
Hồ sơ dài và baseline ngắn chỉ dùng 10.711 timestamp nhãn train duy nhất, gồm 11 timestamp chung validation theo repo; không dùng nhãn test. Nhóm lịch là thứ và khoảng 5 phút từ 2016-07-01 (thứ Sáu).

## Sanity gốc so với export cùng index

- 15 phút: MAE suy luận 13.40400775; export 13.40400778; chênh lệch tuyệt đối 3.78e-08.
- 30 phút: MAE suy luận 14.01034915; export 14.01034922; chênh lệch tuyệt đối 7.05e-08.
- 60 phút: MAE suy luận 15.07355442; export 15.07355451; chênh lệch tuyệt đối 9.17e-08.
Ngưỡng MAE khai báo trước 1e-4; khác GPU/batch FP32 không yêu cầu khớp từng bit. Max chênh lệch prediction 0.000244141.

## Quan sát

- original, 60 phút: MAE 15.0736, RMSE 25.5458; MAE thay đổi +0.00% so với gốc cùng subset.
- long_train_calendar_profile, 60 phút: MAE 17.4497, RMSE 28.1485; MAE thay đổi +15.76% so với gốc cùng subset.
- long_shuffled_windows, 60 phút: MAE 45.2415, RMSE 65.7425; MAE thay đổi +200.14% so với gốc cùng subset.
- short_train_node_mean, 60 phút: MAE 69.0953, RMSE 95.5692; MAE thay đổi +358.39% so với gốc cùng subset.
- short_flow_gaussian_noise, 60 phút: MAE 15.1269, RMSE 25.5660; MAE thay đổi +0.35% so với gốc cùng subset.

## Cách hiểu IG và giới hạn

IG tự cài bằng 20 khoảng nội suy (21 điểm, quy tắc hình thang), batch 1. Đầu ra giải thích là Flow dự báo trung bình trên 170 node tại từng horizon, không phải một node riêng. Baseline là trung bình chuẩn hóa theo node từ train; lịch sử dài giữ nguyên.
CSV lấy trị tuyệt đối attribution trước khi trung bình qua cửa sổ và node. Theo thời gian cộng hai kênh, theo kênh cộng 12 bước. Tỷ trọng |IG| chỉ là cách tổng hợp độ nhạy, không phải tỷ lệ đóng góp nhân quả.
Sai số completeness tuyệt đối trung bình 0.148271, lớn nhất 1.26194 đơn vị Flow dự báo trung bình. Xem ig_completeness.csv; 20 bước chỉ là xấp xỉ, không khẳng định đã hội tụ tích phân.
Nhiễu chỉ cộng vào Flow ngắn: sigma=0,01 miền chuẩn hóa, tương đương 5,385 xe/5 phút; giữ mã thời gian, không clip. Mean ngắn thay cả hai kênh được model sử dụng bằng trung bình train nên không còn lịch thời gian thực.
Mô hình được huấn luyện với đầu vào thật. Thay trung bình, xáo trộn hoặc thay clock có thể tạo đầu vào ngoài phân phối. Long và short có phần lịch sử chồng nhau, nên không phải các nguồn độc lập. Donor xáo trộn có thể ở tương lai so với cửa sổ nhận; không coi đây là cách dự báo hợp lệ.
Độ nhạy KHÔNG bằng đóng góp nhân quả; không phải ablation huấn luyện lại. Một subset, một nhiễu và một baseline IG không cho phép kết luận tổng quát hoặc so sánh nhân quả giữa các nhánh.
