# Ghi chú phân tích sai số PEMS08

Nguồn: run `PEMS08-repo96-author-pretrained`, attempt `0001`, best epoch 68. Không chạy lại mô hình.

## Kiểm chứng và cách tính

Đã kiểm SHA-256 113 file export/metadata; đủ 3.566 cửa sổ, 170 node. Metric tổng khớp bảng chuẩn với atol=1e-8, rtol=0.
MAE giữ nhãn 0; RMSE loại nhãn 0; MAPE loại nhãn <0,01. Tổng sai số được cộng ở float64. Các phân nhóm tái tạo được MAE/RMSE tổng.
Gom theo timestamp mục tiêu = forecast_start + horizon - 1. Lịch tái dựng từ 2016-07-01 00:00, mỗi 5 phút; không đổi múi giờ.

## Quan sát mô tả

- **15 phút:** MAE 13.3996; node MAE cao nhất 10 (32.8696), thấp nhất 109 (2.4257). Pearson với Flow trung bình 0.5043, Spearman 0.5078.
  Theo giờ: MAE lớn nhất 7h (19.1275), nhỏ nhất 2h (6.3196). Thứ Hai–Sáu 13.8608; Thứ Bảy–Chủ nhật 12.4332.
  Theo phân vị: Q1 MAE 6.8541, nhóm cao nhất Q4 MAE 19.4773. Khoảng percentile bootstrap 95%: [12.9857, 13.7873].
- **30 phút:** MAE 14.1934; node MAE cao nhất 10 (41.7291), thấp nhất 62 (2.3949). Pearson với Flow trung bình 0.4711, Spearman 0.4959.
  Theo giờ: MAE lớn nhất 7h (19.9273), nhỏ nhất 2h (6.4483). Thứ Hai–Sáu 14.6593; Thứ Bảy–Chủ nhật 13.2170.
  Theo phân vị: Q1 MAE 7.4298, nhóm cao nhất Q4 MAE 20.6129. Khoảng percentile bootstrap 95%: [13.7562, 14.6186].
- **60 phút:** MAE 15.3880; node MAE cao nhất 78 (56.0627), thấp nhất 109 (2.9851). Pearson với Flow trung bình 0.4444, Spearman 0.4823.
  Theo giờ: MAE lớn nhất 7h (21.2074), nhỏ nhất 2h (7.0486). Thứ Hai–Sáu 15.8485; Thứ Bảy–Chủ nhật 14.4231.
  Theo phân vị: Q1 MAE 8.2709, nhóm cao nhất Q4 MAE 22.2037. Khoảng percentile bootstrap 95%: [14.9455, 15.8577].

## Giới hạn

Các quan sát là mô tả, không chứng minh Flow, khung giờ hoặc loại ngày gây ra sai số. Không có p-value hoặc kết luận về khác biệt có ý nghĩa thống kê.
Phân vị được tính riêng từ target test từng horizon; các quan sát bằng ngưỡng ở cùng nhóm nên số mẫu nhóm có thể không bằng nhau. Đây là phân tích sau đánh giá, không chọn model bằng test.
Bootstrap lấy lại toàn bộ khối ngày, giữ các node và cửa sổ trong ngày đi cùng nhau. Có 13 ngày (9 ngày thứ Hai–Sáu, 4 ngày cuối tuần); ngày đầu/cuối có khối chưa đầy đủ ở một số horizon. MAE được tính bằng tỷ số tổng sai số/tổng số quan sát, không trung bình không trọng số giữa các ngày.
Khoảng 2,5–97,5% từ 2000 lần bootstrap mô tả dao động do lấy mẫu test của **một model cố định**, không phải dao động giữa seed huấn luyện. Seed 20261101 chỉ điều khiển lấy mẫu bootstrap. Phương pháp không mô hình hóa phụ thuộc giữa các ngày; số ngày ít nên khoảng có tính thăm dò.
MAE tuyệt đối có cùng đơn vị xe/5 phút/node; node lưu lượng thấp có thể có MAE thấp mà chưa chứng minh dự báo tương đối tốt hơn. Chưa có nhãn sự kiện để suy diễn sai số do tai nạn hoặc tắc đường.

## Tệp kết quả

`node_metrics.csv` chứa cả 170 node mỗi horizon; `node_extremes.csv` chứa 10 cao/10 thấp; `node_correlations.csv` chứa tương quan. Các bảng còn lại tương ứng giờ, loại ngày, phân vị, từng ngày và bootstrap. PNG đều dpi=200 và có dòng nguồn.
Thời gian đến trước bước ghi protocol/ghi chú: 12.65 giây. Không sửa README, config, graph, split hoặc results/experiment_*.
