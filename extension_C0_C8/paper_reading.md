# Phiếu tự đọc — không điền hộ Related Work

Nguồn được kiểm tra ngày 20/11/2026. Bạn tự điền mỗi mục và ghi số trang/phương trình/bảng làm bằng chứng. Không suy từ abstract rằng một mô hình chắc chắn tốt hơn LSTTN trên giao thức của bạn.

## 1. STEP

**Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting** — Zezhi Shao, Zhao Zhang, Fei Wang, Yongjun Xu; KDD 2022.

Nguồn tác giả: [arXiv 2206.09113](https://arxiv.org/abs/2206.09113); [DOI](https://doi.org/10.1145/3534678.3539396); [code tác giả](https://github.com/GestaltCogTeam/STEP).

| Mục tự điền | Ghi chú của bạn và vị trí bằng chứng |
|---|---|
| Bài toán | |
| Hạn chế phương pháp trước | |
| Ý tưởng chính | |
| Kiến trúc | |
| Dataset và split | |
| Metrics và masking | |
| Điểm mạnh/yếu | |
| Liên hệ với LSTTN | |

Câu hỏi tự kiểm tra:
1. Pretraining học dự đoán tương lai hay tái dựng dữ liệu bị che? Chỉ ra loss.
2. Patch và timestep khác nhau thế nào? Vì sao điều đó ảnh hưởng bộ nhớ?
3. Đặc trưng pretrained đi vào downstream ở đâu? Tự vẽ tensor shapes.
4. So với LSTTN, lịch sử dài được sử dụng giống/khác ở điểm nào, dựa vào code hoặc hình nào?

## 2. STID

**Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting** — Zezhi Shao, Zhao Zhang, Fei Wang, Wei Wei, Yongjun Xu; CIKM 2022.

Nguồn tác giả: [arXiv 2208.05233](https://arxiv.org/abs/2208.05233); [DOI arXiv](https://doi.org/10.48550/arXiv.2208.05233).

| Mục tự điền | Ghi chú của bạn và vị trí bằng chứng |
|---|---|
| Bài toán | |
| Hạn chế phương pháp trước | |
| Ý tưởng chính | |
| Kiến trúc | |
| Dataset và split | |
| Metrics và masking | |
| Điểm mạnh/yếu | |
| Liên hệ với LSTTN | |

Câu hỏi tự kiểm tra:
1. Hai mẫu có cùng chuỗi Flow nhưng khác sensor/giờ được phân biệt bằng cách nào?
2. Những thành phần nào là tham số học, những thành phần nào chỉ là index?
3. Bảng ablation nào chứng minh vai trò spatial và temporal identity?
4. Trước khi so hiệu năng với LSTTN, bạn phải đồng nhất những yếu tố giao thức nào?

## 3. STAEformer

**STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting** — Hangchen Liu, Zheng Dong, Renhe Jiang, Jiewen Deng, Jinliang Deng, Quanjun Chen, Xuan Song; CIKM 2023. Danh sách bảy tác giả được đối chiếu trang đầu PDF.

Nguồn: [arXiv 2308.10425](https://arxiv.org/abs/2308.10425); [PDF](https://arxiv.org/pdf/2308.10425); [DOI](https://doi.org/10.1145/3583780.3615160); [code tác giả](https://github.com/XDZhelheim/STAEformer).

| Mục tự điền | Ghi chú của bạn và vị trí bằng chứng |
|---|---|
| Bài toán | |
| Hạn chế phương pháp trước | |
| Ý tưởng chính | |
| Kiến trúc | |
| Dataset và split | |
| Metrics và masking | |
| Điểm mạnh/yếu | |
| Liên hệ với LSTTN | |

Câu hỏi tự kiểm tra:
1. Adaptive embedding có shape nào và học được điều gì mà lịch cố định không biểu diễn?
2. Attention theo thời gian khác theo không gian ở chiều tensor nào?
3. Bằng chứng nào tách tác dụng embedding khỏi độ lớn mô hình?
4. Tên “SOTA” có cho phép kết luận hơn baseline của bạn khi split khác không?

## 4. Bài thực sự trích dẫn LSTTN

**Pretraining-improved Spatiotemporal graph network for the generalization performance enhancement of traffic forecasting** — Xiangyue Zhang, Chao Li, Ling Ji, Yuyun Kang, Mingming Pan, Zhuo Liu, Qiang Qi; Scientific Reports 15, 27668, xuất bản 29/07/2025.

Nguồn nhà xuất bản: [bài và thông tin tác giả](https://www.nature.com/articles/s41598-025-11375-2); [DOI](https://doi.org/10.1038/s41598-025-11375-2); [PDF và References](https://www.nature.com/articles/s41598-025-11375-2.pdf).

Bằng chứng quan hệ trích dẫn: kết quả lập chỉ mục PDF của chính nhà xuất bản hiển thị mục References với Luo, He, Han, Wang và tên LSTTN. Đây là trích dẫn thực, không suy từ chủ đề tương tự. Truy cập trực tiếp toàn văn qua công cụ gặp chặn đăng nhập/reCAPTCHA; chưa xác định số thứ tự tham khảo hoặc phân tích đoạn sử dụng trích dẫn. Bạn cần mở PDF và ghi lại số mục/trang. Không khẳng định bài này cải thiện LSTTN chỉ từ việc có trích dẫn.

| Mục tự điền | Ghi chú của bạn và vị trí bằng chứng |
|---|---|
| Bài toán | |
| Hạn chế phương pháp trước | |
| Ý tưởng chính | |
| Kiến trúc | |
| Dataset và split | |
| Metrics và masking | |
| Điểm mạnh/yếu | |
| Liên hệ với LSTTN; số reference/trang | |

Câu hỏi tự kiểm tra:
1. LSTTN được trích dẫn để nói về hạn chế, làm baseline hay làm thành phần phương pháp? Chỉ ra đoạn.
2. “Generalization” được tác giả kiểm chứng bằng thí nghiệm cụ thể nào?
3. Có dùng cùng split, horizon và masking với LSTTN của bạn không?
4. Ý tưởng nào có thể kiểm tra với backbone frozen và ngân sách nhỏ, ý tưởng nào cần train lại?
